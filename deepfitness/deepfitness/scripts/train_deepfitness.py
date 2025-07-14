""" 
    Deep fitness inference, using warm-up training with simple fitness.
    1. Runs full-data simple fitness inference (~1 min)
    2. Warm-up trains network on {genotype, simple_logw}
    3. End-to-end train network on time-series ngs data 
"""
import os
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
from deepfitness.models.deep_fitness import DeepFitnessModel
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
        1. Simple fitness inference on full dataset
    """
    # use train rounds only
    if args.setdefault('test_split_by_round', False):
        int_list = lambda s: [int(c) for c in s]
        train_round_idxs = int_list(args.get('train_round_idxs'))
        val_round_idxs = int_list(args.get('val_round_idxs'))
        test_round_idxs = int_list(args.get('test_round_idxs'))
        train_tsngs_df, val_tsngs_df, test_tsngs_df = tsngs.split_tsngsdf_by_rounds(tsngs_df, train_round_idxs, val_round_idxs, test_round_idxs)
        simple_tsngs_df = train_tsngs_df
    else:
        simple_tsngs_df = tsngs_df

    simple_logw = warmup.infer_simplefitness(simple_tsngs_df)

    simple_logw_col = 'Simple log inferred fitness'
    tsngs_df.update_with_col(name = simple_logw_col, values = simple_logw)

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
        logger.debug(split_name)
        logger.debug(f'NLLs: (extrapolative, normalized by n in mask) ...')
        logger.debug(f'\tAll rounds NLL: {_tdf.compute_all_rounds_nll(simple_logw_col)}')
        logger.debug(f'\tLast round NLL: {_tdf.compute_last_round_nll(simple_logw_col)}')

    network_class = networks.get_network(args.get('net.network'))
    network = network_class()

    """
        2. Warmup train deep net, in place, on simple fitness values
    """
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
    loss_func_name = args.setdefault('loss_func', 'dirichlet_multinomial')
    fitness_model = DeepFitnessModel(
        network = network, 
        loss_func_name = loss_func_name,
        regularize_logw_col = simple_logw_col,
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
    elif datasplits.has(rounds = 'test', genotypes = 'train'):
        val_tsngs_df = datasplits.get(rounds = 'test', genotypes = 'train')
        cbacks.append(
            callbacks.ValidationCallback(
                val_tsngs_df, 
                dataflow, 
                simple_logw_col,
                every_n_epochs = args.setdefault('e2e.val_every_n_epochs', 1)
            )
        )
        ckpt_stat = args.setdefault('e2e.model_select_stat', 'train_loss')
    else:
        ckpt_stat = args.setdefault('e2e.model_select_stat', 'train_loss')
    e2e_ckpt = callbacks.build_checkpoint_callback(ckpt_stat, 'e2e')
    cbacks.append(e2e_ckpt)

    # autoscale batch size - maximize to fit in memory
    if not args.setdefault('skip_autobatch', False):
        batch_size = autobatch.find_batch_size(
            model = fitness_model, 
            tsngs_df = datasplits.get(rounds = 'train', genotypes = 'train'),
            dataflow = dataflow,
            batch_class_name = 'tsngs_withfitness',
            simple_logw_col = simple_logw_col,
        )
    else:
        batch_size = args.setdefault('e2e.batch_size', 16384)

    # build loader
    e2e_train_dataset = datasets.TorchTSNGSwithFitnessDataset(
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
        logger = train_logger
    )

    trainer.fit(fitness_model, train_dataloaders = e2e_train_dataloader)

    # get best model, among warm-up MSE training and e2e training
    # if held-out genotypes, get best model by val nll.
    # otherwise, just use e2e model with best train loss
    if datasplits.has(rounds = 'train', genotypes = 'val'):
        best_ckpt = get_best_ckpt([mse_ckpt, e2e_ckpt])
    else:
        best_ckpt = e2e_ckpt
    logger.info(f'Best model path: {best_ckpt}')

    network = load_network_from_pl_ckpt(
        best_ckpt.best_model_path, 
        network_class
    )
    annotate.annotate_data_splits(
        datasplits, 
        dataflow, 
        network, 
        logw_col_name = 'Deep log inferred fitness',
        w_col_name = 'Deep inferred fitness',
    )
    datasplits.write_data_split_csvs_to_file()

    logger.info('Done.')
    return


def get_best_ckpt(ckpts: list[ModelCheckpoint]) -> ModelCheckpoint:
    scores = [ckpt.best_model_score for ckpt in ckpts]
    return ckpts[scores.index(min(scores))]


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.train_deepfitness

        python -m deepfitness.scripts.train_deepfitness --config deepfitness/options/deepfit_fgfr1.yaml
    """
    # load args from yaml, then update with cli
    parser = argparse.ArgumentParser(description = """\
        Core options: [-h] [--config] [--csv] [--genotype_col]
            [--round_cols OR --rounds_before, --rounds_after] 
            [--project_output_folder] [--ft.dataflow]

        Trains a deep fitness model on csv, using readcount data from round_cols. Saves output files to project_output_folder. If using wandb, saves to
        project_output_folder/wandb/<run-id> instead. 

        More command-line options are available beyond the core described above.
        To see them, try running the script. You can refer to the default
        yaml file read, or look at the saved_args.yaml file in the output
        folder.
    """)
    args.parse_args(parser, 'deepfitness/options/default_deepfitness.yaml')
    
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
    saved_args_yaml = os.path.join(output_folder, 'args_traindeepfitness.yaml')
    args.save_to_yaml(saved_args_yaml)

