""" 
    Deep fitness inference, using warm-up training with simple fitness.
    1. Runs full-data simple fitness inference (~1 min)
    2. Warm-up trains network on {genotype, simple_logw}
    3. End-to-end train network on time-series ngs data 
"""
import os, yaml
import argparse
from loguru import logger
import wandb
import torch
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from deepfitness import networks, callbacks, autobatch
from deepfitness.models.deep_latent import DeepFitnessLatentModel
from hackerargs import args
from deepfitness.data import tsngs, annotate, splits, loaders
from deepfitness.genotype import dataflows
from deepfitness.genotype.feature_stores import FeatureStore
from deepfitness.predict import load_network_from_pl_ckpt
from deepfitness import warmup
from deepfitness.data import datasets
from deepfitness.data.loaders import ManualBatchSampler

# disable warning on using 0 dataloader numworkers (main thread only):
# we assume dataset fits in RAM
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def save_info(log_dirmul_precision: float) -> None:
    """ Save optimization info to yaml. """
    out_yaml = args.setdefault(
        'output_info_yaml',
        os.path.join(args.get('output_folder'), 'output_parameters.yaml')
    )
    os.makedirs(os.path.dirname(out_yaml), exist_ok = True)
    with open(out_yaml, 'w') as f:
        yaml.dump({'log_dirmul_precision': log_dirmul_precision}, f)
    logger.info(f'Saved output params to {out_yaml}')
    return


def main():
    torch.set_float32_matmul_precision('medium')

    logger.info(f'Using dataflow: {args.get("ft.dataflow")}')
    dataflow = dataflows.get_dataflow(args.get('ft.dataflow'))

    tsngs_df = tsngs.construct_tsngs_df(
        df = pd.read_csv(args.get('csv')),
        genotype_col = args.get('genotype_col'),
        round_cols = args.setdefault('round_cols', None),
        rounds_before = args.setdefault('rounds_before', None),
        rounds_after = args.setdefault('rounds_after', None),
        schema = dataflow.schema,
        skip_filters = args.setdefault('tsngs.skip_filters', False),
        verbose = True,
    )

    """
        1. Simple fitness inference with latent frequencies
    """
    # use train rounds only
    if args.setdefault('test_split_by_round', False):
        int_list = lambda s: [int(c) for c in s]
        train_round_idxs = int_list(args.get('train_round_idxs'))
        val_round_idxs = int_list(args.setdefault('val_round_idxs', []))
        test_round_idxs = int_list(args.get('test_round_idxs'))
        train_tsngs_df, val_tsngs_df, test_tsngs_df = tsngs.split_tsngsdf_by_rounds(tsngs_df, train_round_idxs, val_round_idxs, test_round_idxs)
        simple_tsngs_df = train_tsngs_df
    else:
        simple_tsngs_df = tsngs_df

    sfit_dict = warmup.infer_simplefitness_latent(simple_tsngs_df)

    simple_logw_col = 'Simple log inferred fitness'
    assert simple_tsngs_df.get_genotypes() == tsngs_df.get_genotypes()
    tsngs_df.update_with_col(simple_logw_col, sfit_dict['log_fitness'])

    """
        Setup for deep neural net training
    """
    # create feature store
    if args.setdefault('use_feature_store', False):
        fstore = FeatureStore(tsngs_df.get_genotypes(), dataflow.featurizer)
        dataflow.add_feature_store(fstore)
    else:
        logger.info(f'Skipping feature store ... if desired, use flag `use_feature_store`')

    # single source of train/val/test split
    datasplits = splits.DataSplits(tsngs_df)

    # record val metrics
    for split_name, _tdf in datasplits.data_splits.items():
        if _tdf is None:
            continue
        logger.debug(split_name)
        logger.debug(f'NLLs: (extrapolative, normalized by n in mask) ...')
        logger.debug(f'\tAll rounds NLL: {_tdf.compute_all_rounds_nll(simple_logw_col)}')
        logger.debug(f'\tLast round NLL: {_tdf.compute_last_round_nll(simple_logw_col)}')

    network_class = networks.get_network(args.get('net.network'))
    network = network_class()

    """
        2. Warmup train deep net, in place, on simple fitness values
    """
    if args.setdefault('warmup.disable', False):
        mse_ckpt = None
    else:
        mse_ckpt = warmup.mse_warmup_train(
            network = network, 
            datasplits = datasplits,
            dataflow = dataflow, 
            simple_logw_col = simple_logw_col,
        )

    """
        3. End-to-end train deep net on time-series data,
        regularizing to simple fitness
    """
    # Retrieve trained log_abundance, subset to training set genotypes.
    # Log abundance has underdetermined scale in data generative model,
    # so further training it here can separate its scale from that of
    # val/test genotype idxs.
    train_tsngs_df = datasplits.get('train', 'train')
    if args.get('test_split_by_genotype'):
        train_gt_idxs = train_tsngs_df.df['original_index']
        init_log_abundance = sfit_dict['log_abundance'][train_gt_idxs]
    else:
        train_gt_idxs = list(range(len(sfit_dict['log_abundance'])))
        init_log_abundance = sfit_dict['log_abundance']

    fitness_model = DeepFitnessLatentModel(
        network, 
        args.setdefault('loss_func', 'dirichlet_multinomial'),
        len(train_gt_idxs),
        regularize_logw_col = simple_logw_col,
        init_log_abundance = torch.tensor(init_log_abundance),
        init_log_dirmul_precision = sfit_dict['log_dirmul_precision'],
    )

    # build callbacks - validation stats & model checkpoint saving
    convergence_detector = EarlyStopping(
        monitor = 'train_loss',
        patience = args.setdefault('e2e.convergence.patience', 30),
        check_on_train_epoch_end = True,
        verbose = True,
    )
    cbacks = [convergence_detector]

    if datasplits.has(rounds = 'train', genotypes = 'val'):
        val_tsngs_df = datasplits.get(rounds = 'train', genotypes = 'val')
        cbacks.append(
            callbacks.ValidationCallback(
                val_tsngs_df, 
                dataflow, 
                simple_logw_col,
                every_n_epochs = args.setdefault('e2e.val_every_n_epochs', 1)
            )
        )
        ckpt_stat = args.setdefault('e2e.model_select_stat', 'val_last_round_nll')
    elif datasplits.has(rounds = 'val', genotypes = 'train'):
        val_tsngs_df = datasplits.get(rounds = 'val', genotypes = 'train')
        if val_tsngs_df is not None:
            cbacks.append(
                callbacks.ValidationCallback(
                    val_tsngs_df, 
                    dataflow, 
                    simple_logw_col,
                    every_n_epochs = args.setdefault('e2e.val_every_n_epochs', 1)
                )
            )
        test_tsngs_df = datasplits.get(rounds = 'test', genotypes = 'train')
        if test_tsngs_df is not None:
            cbacks.append(
                callbacks.ValidationCallback(
                    test_tsngs_df, 
                    dataflow, 
                    simple_logw_col,
                    every_n_epochs = args.setdefault('e2e.val_every_n_epochs', 1),
                    name = 'test',
                )
            )
        ckpt_stat = args.setdefault('e2e.model_select_stat', 'test_last_round_nll')
    else:
        ckpt_stat = args.setdefault('e2e.model_select_stat', 'train_loss')
    # model ckpt selection by val performance or training loss
    e2e_ckpt = callbacks.build_checkpoint_callback(ckpt_stat, 'e2e')
    cbacks.append(e2e_ckpt)

    # autoscale batch size - maximize to fit in memory
    if not args.setdefault('skip_autobatch', False):
        batch_size = autobatch.find_batch_size(
            model = fitness_model, 
            tsngs_df = datasplits.get(rounds = 'train', genotypes = 'train'),
            dataflow = dataflow,
            batch_class_name = 'tsngs_latentwithfitness',
            simple_logw_col = simple_logw_col,
        )
    else:
        batch_size = args.setdefault('e2e.batch_size', 16384)

    # build loader
    e2e_train_dataset = datasets.TorchTSNGSLatentwithFitnessDataset(
        tsngs_df = datasplits.get(rounds = 'train', genotypes = 'train'),
        featurizer = dataflow.featurizer,
        feature_store = dataflow.feature_store,
        target_logw_col = simple_logw_col,
    )
    e2e_train_dataloader = DataLoader(
        dataset = e2e_train_dataset,
        batch_sampler = ManualBatchSampler(
            batch_size = batch_size,
            batch_ids = e2e_train_dataset.get_batch_ids(),
        ),
        collate_fn = dataflow.collater,
        pin_memory = args.setdefault('dataloader_pin_memory', True),
        num_workers = args.setdefault('dataloader_num_workers', 0),
        persistent_workers = args.setdefault('dataloader_persistent_workers', False),
    )

    batches_per_epoch = len(e2e_train_dataloader)
    if args.setdefault('e2e.accumulate_grad', True):
        acc_grad_batches = batches_per_epoch
    else:
        acc_grad_batches = 1
    fitness_model.set_data_properties(len(e2e_train_dataset), batches_per_epoch)

    # build trainer logger; default value is True
    train_logger = True
    if args.setdefault('wandb.use', False) is True:
        train_logger = WandbLogger(
            project = args.setdefault('wandb.project', 'deepfitness'),
            prefix = 'e2e'
        )
        wandb.run.define_metric(ckpt_stat, goal = 'minimize')

    # build trainer
    trainer = pl.Trainer(
        accelerator = args.get('accelerator'), 
        devices = 1,
        max_epochs = args.setdefault('epochs', 1000),
        accumulate_grad_batches = acc_grad_batches,
        callbacks = cbacks,
        logger = train_logger,
    )

    trainer.fit(fitness_model, train_dataloaders = e2e_train_dataloader)

    # if held-out genotypes, get best model by val nll.
    # otherwise, just use e2e model with best train loss
    if datasplits.has(rounds = 'train', genotypes = 'val'):
        best_ckpt = get_best_ckpt([mse_ckpt, e2e_ckpt])
    else:
        best_ckpt = e2e_ckpt
    logger.info(f'Best model path: {best_ckpt.best_model_path}')

    logger.info(f'Loading learned items ...')
    learned = load_learned_items(best_ckpt, network_class, sfit_dict, train_gt_idxs)
    best_network = learned['network']

    # save to yaml
    save_info(learned['log_dirmul_precision'])

    # annotate train/train with log_abundance
    train_tsngs_df.update_with_col('log_abundance', learned['log_abundance'])

    # annotate all splits with predicted log fitness
    annotate.annotate_data_splits(
        datasplits, 
        dataflow, 
        best_network, 
        logw_col_name = 'Deep log inferred fitness',
        w_col_name = 'Deep inferred fitness',
    )
    datasplits.write_data_split_csvs_to_file()

    logger.info('Done.')
    return


"""
    Loading checkpoints and models
"""
def get_best_ckpt(ckpts: list[ModelCheckpoint | None]) -> ModelCheckpoint:
    ckpts = [ckpt for ckpt in ckpts if ckpt is not None]
    scores = [ckpt.best_model_score for ckpt in ckpts]
    return ckpts[scores.index(min(scores))]


def get_checkpoint_model_type(ckpt: ModelCheckpoint) -> str:
    """ Get model type `mse` or `e2e` from checkpoint """
    if 'model-mse' in ckpt.best_model_path:
        return 'mse'
    elif 'model-e2e' in ckpt.best_model_path:
        return 'e2e'
    assert False


def load_learned_items(
    ckpt: ModelCheckpoint,
    network_class: type,
    sfit_dict: dict,
    train_gt_idxs: list[int],
) -> dict:
    """ Infers ckpt as `mse` or `e2e`, and returns dict containing
        network, log abundance, and log dirmul precision.
        
        Returns log abundance only on train_gt_idxs.
    """
    model_type = get_checkpoint_model_type(ckpt)
    if model_type == 'mse':
        return load_mse_items(ckpt, network_class, sfit_dict, train_gt_idxs)
    elif model_type == 'e2e':
        return load_e2e_items(ckpt, network_class, train_gt_idxs)


def load_mse_items(
    mse_ckpt: ModelCheckpoint,
    network_class: type,
    sfit_dict: dict,
    train_gt_idxs: list[int],
) -> dict:
    network = load_network_from_pl_ckpt(
        mse_ckpt.best_model_path, 
        network_class
    )
    return {
        'log_dirmul_precision': sfit_dict['log_dirmul_precision'],
        'log_abundance': sfit_dict['log_abundance'][train_gt_idxs],
        'network': network
    }


def load_e2e_items(
    e2e_ckpt: ModelCheckpoint,
    network_class: type,
    train_gt_idxs: list[int],
) -> dict:
    best_model = DeepFitnessLatentModel.load_from_checkpoint(
        e2e_ckpt.best_model_path,
        network = network_class(),
        loss_func_name = args.get('loss_func'),
        num_train_genotypes = len(train_gt_idxs)
    )
    return {
        'log_dirmul_precision': best_model.get_log_dirmul_precision(),
        'log_abundance': best_model.get_log_abundance(),
        'network': best_model.network
    }


if __name__ == '__main__':
    """
        python -m deepfitness.scripts.train_deep_latent --config deepfitness/options/dryrun_deep_latent.yaml
    """
    # load args from yaml, then update with cli
    parser = argparse.ArgumentParser(description = """\
        Core options: [-h] [--config] [--csv] [--genotype_col]
            [--round_cols OR --rounds_before, --rounds_after] 
            [--project_output_folder] [--ft.dataflow]

        Trains a deep fitness model on csv, using readcount data from round_cols.
        Saves output files to project_output_folder. If using wandb, saves to
        project_output_folder/wandb/<run-id> instead. 

        More command-line options are available beyond the core described above.
        To see them, try running the script. You can refer to the default
        yaml file read, or look at the saved_args.yaml file in the output
        folder.
    """)
    args.parse_args(parser, 'deepfitness/options/default_deep_latent.yaml')
    
    assert 'csv' in args
    assert 'genotype_col' in args
    assert 'project_output_folder' in args
    assert 'ft.dataflow' in args

    args.setdefault('model_type', 'deep_fitness')

    # create output folder
    output_folder = args.get('project_output_folder')
    if args.setdefault('wandb.use', False):
        wandb.init(
            project = args.setdefault('wandb.project', 'deepfitness'),
            config = args,
            name = args.setdefault('wandb.run_name', None),
        )
        output_folder = os.path.join(output_folder, 'wandb', f'{wandb.run.id}')
    args.setdefault('output_folder', output_folder)
    os.makedirs(output_folder, exist_ok = True)

    # create log
    log_file = os.path.join(output_folder, f'log.log')
    logger.add(log_file)
    logger.info(f'Saving logs to {log_file=}')

    main()

    # save args to yaml
    if args.get('wandb.use'):
        wandb.config.update(args)
    saved_args_yaml = os.path.join(output_folder, 'args_train_deep_latent.yaml')
    args.save_to_yaml(saved_args_yaml)

