"""
    Callbacks for pytorch lightning trainers
"""
import os
from loguru import logger
import numpy as np
from numpy.typing import NDArray
import wandb
from scipy.stats import pearsonr, spearmanr

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from hackerargs import args
from deepfitness.data import loaders
from deepfitness.genotype.dataflows import DataFlow
from deepfitness.data.tsngs import TimeSeriesNGSDataFrame
from deepfitness.utils import tensor_to_np


def build_checkpoint_callback(select_stat: str, name: str) -> ModelCheckpoint:
    """ Builds a checkpoint saving callback.
        Uses filename model-{name}-{wandb.run.id, if exists}-{selectstat=}
    """
    ckpt_folder = args.setdefault(
        'checkpoint_folder',
        args.get('output_folder') + '/checkpoints/'
    )
    os.makedirs(ckpt_folder, exist_ok = True)

    filename_prefix = 'model'
    if name:
        filename_prefix += f'-{name}'
    if args.setdefault('wandb.use', False):
        filename_prefix += f'-{wandb.run.id}'
    return ModelCheckpoint(
        dirpath = ckpt_folder, 
        save_top_k = 1, 
        monitor = select_stat,
        mode = 'min',
        filename = filename_prefix + '-{' + select_stat + ':.2f}'
    )


class ValidationCallback(Callback):
    def __init__(
        self, 
        val_tsngs_df: TimeSeriesNGSDataFrame,
        dataflow: DataFlow,
        simple_logw_col: str | None = None,
        every_n_epochs: int = 1,
        name: str = 'val',
    ):
        """
            Callback provided to a pl.Trainer.
            On validation epoch end, runs model to obtain
            current predicted fitness, then computes summary stats
            such as last-round NLL, all-round NLL, and
            MSE loss to simple inferred fitness.
        """
        self.tsngs_df = val_tsngs_df
        self.pred_dl = loaders.predict_dataloader(self.tsngs_df, dataflow)
        self.simple_logw_col = simple_logw_col
        self.every_n_epochs = every_n_epochs
        self._n_skipped_epochs = 0
        self.name = name

    def custom_get_pred_logw(self, model: pl.LightningModule) -> NDArray:
        """ This uses a second trainer (for prediction) on a model that
            is trained by another trainer. Care must be taken.
            https://github.com/Lightning-AI/lightning/discussions/16258.

            Returns predictions of logw on genotype variants in tsngs_df,
            using the same order.
        """
        preds = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.pred_dl):
                batch = model.transfer_batch_to_device(batch, model._device, 0)
                output = model(batch)
                preds.append(output)
        return tensor_to_np(torch.concat(preds))

    def on_train_epoch_end(
        self, 
        trainer: pl.Trainer, 
        model: pl.LightningModule,
    ):
        """ Validation step after training epoch.
            Use model to infer fitness, then compute summary stats.
        """
        if self.every_n_epochs > 1:
            if self._n_skipped_epochs != self.every_n_epochs - 1:
                self._n_skipped_epochs += 1
                return
            else:
                self._n_skipped_epochs = 0

        model.eval()
        pred_logw = self.custom_get_pred_logw(model)
        model.train()

        logw_col = f'__{self.name}_logw'
        self.tsngs_df.update_with_col(logw_col, pred_logw, overwrite = True)

        if self.simple_logw_col:
            metrics_d = self.__metrics(self.simple_logw_col, logw_col)
            for stat, val in metrics_d.items():
                self.log(f'{self.name}_{stat}', val)
                logger.debug(f'{self.name}_{stat}: {val}')

        last_round_nll = self.tsngs_df.compute_last_round_nll(logw_col)
        all_rounds_nll = self.tsngs_df.compute_all_rounds_nll(logw_col)
        self.log(f'{self.name}_last_round_nll', last_round_nll)
        self.log(f'{self.name}_all_rounds_nll', all_rounds_nll)
        logger.debug(f'{self.name} last-round NLL: {last_round_nll}')
        logger.debug(f'{self.name} all-rounds NLL: {all_rounds_nll}')
        return
    
    def __metrics(self, logw_col1: str, logw_col2: str) -> dict[str, float]:
        logws1 = self.tsngs_df.df[logw_col1]
        logws2 = self.tsngs_df.df[logw_col2]

        return {
            'mse_loss': np.mean((logws1 - logws2)**2),
            'pearsonr': pearsonr(logws1, logws2)[0],
            'spearmanr': spearmanr(logws1, logws2)[0],
        }
