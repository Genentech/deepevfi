"""
    Functions for warmup training of deep fitness models,
    using simple fitness inferred logw.

    Steps:
    1. Runs full-data simple fitness inference (~1 min)
    2. Warm-up trains network on {genotype, simple_logw}
    After, can do (in separate script):
    3. End-to-end train network on time-series ngs data 
"""
import os, copy
from loguru import logger
import pandas as pd
from numpy.typing import NDArray
import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger

from deepfitness import networks, callbacks
from hackerargs import args
from deepfitness.data import tsngs, loaders, splits
from deepfitness.data.tsngs import TimeSeriesNGSDataFrame
from deepfitness.models.simple_fulldata import SimpleFitnessFullDataModel
from deepfitness.models.simple_latent import SimpleFitnessFullDataLatentModel

from deepfitness.models.warmup_model import WarmupModel
from deepfitness.genotype import dataflows


"""
    Infer fitness
"""
def infer_simplefitness(tsngs_df: TimeSeriesNGSDataFrame) -> NDArray:
    """ Infers simple fitness, using full data for loss (no minibatches).
        Garbage collects model to clear GPU memory.
    """
    logger.debug(f'Device memory allocated: {torch.cuda.memory_allocated()}')
    logger.debug(f'Device memory reserved: {torch.cuda.memory_reserved()}')
    fitness_model = SimpleFitnessFullDataModel(
        tsngs_df = tsngs_df,
        loss_func_name = args.setdefault('simple.loss_func', 'multinomial')
    )
    fitness_model.train()
    log_fitness = copy.deepcopy(fitness_model.get_log_fitness())

    fitness_model.detach_cleanup()
    del fitness_model
    torch.cuda.empty_cache()

    logger.debug(f'Device memory allocated: {torch.cuda.memory_allocated()}')
    logger.debug(f'Device memory reserved: {torch.cuda.memory_reserved()}')
    return log_fitness


def infer_simplefitness_latent(
    tsngs_df: TimeSeriesNGSDataFrame
) -> dict[str, NDArray]:
    """ Infers simple fitness in latent frequency model.
        Garbage collects model to clear GPU memory when done.

        Returns dict with items
            log_fitness
            log_abundance
            log_dirmul_precision
    """
    logger.debug(f'Device memory allocated: {torch.cuda.memory_allocated()}')
    logger.debug(f'Device memory reserved: {torch.cuda.memory_reserved()}')

    model = SimpleFitnessFullDataLatentModel(tsngs_df, 'dirichlet_multinomial')
    model.train()

    d = {
        'log_fitness': copy.deepcopy(model.get_log_fitness()),
        'log_abundance': copy.deepcopy(model.get_log_abundance()),
        'log_dirmul_precision': copy.deepcopy(model.get_log_dirmul_precision()),
    }

    model.detach_cleanup()
    del model
    torch.cuda.empty_cache()

    logger.debug(f'Device memory allocated: {torch.cuda.memory_allocated()}')
    logger.debug(f'Device memory reserved: {torch.cuda.memory_reserved()}')
    return d


"""
    MSE warmup training
"""
def mse_warmup_train(
    network: torch.nn.Module,
    datasplits: splits.DataSplits,
    dataflow: dataflows.DataFlow,
    simple_logw_col: str,
) -> ModelCheckpoint:
    """ Warm-up trains network on simple logw, MSE loss.
        Early stopping on val genotype set. 
        Returns a WarmupModel checkpoint.
    """
    logger.info(f'Entering MSE warmup ...')
    logger.debug(f'Device memory allocated: {torch.cuda.memory_allocated()}')
    logger.debug(f'Device memory reserved: {torch.cuda.memory_reserved()}')

    warmup_model = WarmupModel(network)

    train_dataloader = loaders.warmup_dataloader(
        tsngs_df = datasplits.get(rounds = 'train', genotypes = 'train'),
        dataflow = dataflow,
        target_logw_col = simple_logw_col
    )

    """
        Build cbacks: list of callbacks for MSE warmup training.
        If validation genotypes exist, use validation callback, and
        select best checkpoint by validation NLL.
        Otherwise, select best checkpoint by training loss.
    """
    convergence_detector = EarlyStopping(
        monitor = 'train_loss',
        patience = args.setdefault('warmup.convergence.patience', 25),
        check_on_train_epoch_end = True,
        verbose = True,
    )
    cbacks = [convergence_detector]

    ckpt_stat = args.setdefault('warmup.model_select_stat', 'train_loss')
    if datasplits.has(rounds = 'train', genotypes = 'val'):
        val_tsngs_df = datasplits.get(rounds = 'train', genotypes = 'val')
        cbacks.append(
            callbacks.ValidationCallback(
                val_tsngs_df, 
                dataflow, 
                simple_logw_col,
                every_n_epochs = args.setdefault('warmup.val_every_n_epochs', 1)
            )
        )
        ckpt_stat = args.setdefault('warmup.model_select_stat', 'val_last_round_nll')
    if datasplits.has(rounds = 'test', genotypes = 'train'):
        val_tsngs_df = datasplits.get(rounds = 'test', genotypes = 'train')
        cbacks.append(
            callbacks.ValidationCallback(
                val_tsngs_df, 
                dataflow, 
                simple_logw_col,
                every_n_epochs = args.setdefault('warmup.val_every_n_epochs', 1),
                name = 'test',
            )
        )
    # model ckpt selection by val performance or training loss
    mse_ckpt = callbacks.build_checkpoint_callback(ckpt_stat, 'mse')
    cbacks.append(mse_ckpt)

    # build trainer logger; default value is True
    train_logger = True
    if args.setdefault('wandb.use', False) is True:
        train_logger = WandbLogger(
            project = args.setdefault('wandb.project', 'deepfitness'),
            prefix = 'warmup'
        )
        wandb.run.define_metric(ckpt_stat, goal = 'minimize')

    # build trainer
    trainer = pl.Trainer(
        accelerator = args.get('accelerator'), 
        devices = 1,
        max_epochs = args.setdefault('warmup.epochs', 1000),
        callbacks = cbacks,
        logger = train_logger
    )
    trainer.fit(warmup_model, train_dataloaders = train_dataloader)

    # Restore network to best version
    network_class = networks.get_network(args.get('net.network'))
    best_model = WarmupModel.load_from_checkpoint(
        checkpoint_path = mse_ckpt.best_model_path, 
        network = network_class()
    )
    network = best_model.network

    logger.debug(f'Device memory allocated: {torch.cuda.memory_allocated()}')
    logger.debug(f'Device memory reserved: {torch.cuda.memory_reserved()}')
    return mse_ckpt


"""
    Tests
"""
def __test_warmup(
    network: torch.nn.Module,
    tsngs_df: TimeSeriesNGSDataFrame,
    dataflow: dataflows.DataFlow
) -> ModelCheckpoint:
    """ TESTING """
    simple_logw = infer_simplefitness(tsngs_df)
    simple_logw_col = 'Simple logw'
    tsngs_df.update_with_col(name = simple_logw_col, values = simple_logw)

    # data split
    datasplits = splits.DataSplits(tsngs_df)

    warmup_model = WarmupModel(network)
    train_dataloader = loaders.warmup_dataloader(
        tsngs_df = datasplits.get(rounds = 'train', genotypes = 'train'), 
        dataflow = dataflow, 
        target_logw_col = simple_logw_col,
    )
    val_tsngs_df = datasplits.get(rounds = 'train', genotypes = 'val')

    select_stat = args.setdefault('model_select_stat', 'val_last_round_nll')
    mse_ckpt = callbacks.build_checkpoint_callback(select_stat, 'mse')
    cbacks = [
        callbacks.ValidationCallback(val_tsngs_df, dataflow, simple_logw_col),
        mse_ckpt,
    ]

    trainer = pl.Trainer(
        accelerator = args.get('accelerator'), 
        devices = 1,
        max_epochs = args.setdefault('warmup.epochs', 1000),
        callbacks = cbacks,
    )
    trainer.fit(
        model = warmup_model, 
        train_dataloaders = train_dataloader
    )

    print(hash(tuple(network.embedder.parameters())))
    network_class = networks.get_network(args.get('net.network'))
    best_model = WarmupModel.load_from_checkpoint(
        checkpoint_path = mse_ckpt.best_model_path, 
        network = network_class()
    )
    network = best_model.network
    print(hash(tuple(network.embedder.parameters())))

    return mse_ckpt


def test():
    torch.set_float32_matmul_precision('high')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    logger.info(f'Using dataflow: {args.get("ft.dataflow")}')
    dataflow = dataflows.get_dataflow(args.get('ft.dataflow'))

    tsngs_df = tsngs.construct_tsngs_df(
        df = pd.read_csv(args.get('csv')),
        genotype_col = args.get('genotype_col'),
        round_cols = args.setdefault('round_cols', None),
        rounds_before = args.setdefault('rounds_before', None),
        rounds_after = args.setdefault('rounds_after', None),
        schema = dataflow.schema,
        skip_filters = False,
    )

    network_class = networks.get_network(args.get('net.network'))
    network = network_class()

    __test_warmup(network, tsngs_df, dataflow)

    # make prediction
    # annotate predicted logw
    from deepfitness.predict import get_pred_logw
    from deepfitness.models.deep_fitness import DeepFitnessModel
    pred_dataloader = loaders.predict_dataloader(tsngs_df, dataflow)
    pred_logw = get_pred_logw(
        fitness_model = DeepFitnessModel(network, 'multinomial'), 
        pred_dataloader = pred_dataloader
    )

    # pred_logw should match tsngs_df order exactly
    tsngs_df.update_with_col('Warmup MSE log inferred fitness', pred_logw)

    # annotate target/pred fq cols
    tsngs_df.update_with_target_pred_cols('Warmup MSE log inferred fitness')

    # save df to file
    tsngs_df.df.to_csv(args.get('output_folder') + '/warmup_test.csv')
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.warmup
    """
    args.parse_args('deepfitness/options/test_warmup.yaml')

    mandatory_args = [
        'csv',
        'genotype_col',
        'output_folder',
        'ft.dataflow',
    ]
    for mandatory_arg in mandatory_args:
        assert mandatory_arg in args

    # create output folder
    output_folder = args.get('output_folder')
    os.makedirs(output_folder, exist_ok=True)

    test()

