"""
    Automatically find largest batch size that fits in memory
"""
from loguru import logger
import torch
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import copy

from hackerargs import args
from deepfitness.data.tsngs import TimeSeriesNGSDataFrame
from deepfitness.data import loaders
from deepfitness.genotype.dataflows import DataFlow


class BatchSizeModule(pl.LightningDataModule):
    def __init__(
        self,
        tsngs_df: TimeSeriesNGSDataFrame,
        dataflow: DataFlow,
        batch_class_name: str,
        batch_size: int = 512,
        simple_logw_col: str | None = None,
    ):
        super().__init__()
        self.tsngs_df = tsngs_df
        self.dataflow = dataflow
        self.batch_class_name = batch_class_name
        self.batch_size = batch_size
        self.simple_logw_col = simple_logw_col
        
    def prepare_data(self) -> None:
        return
 
    def setup(self, stage: str) -> None:
        return

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.batch_class_name == 'tsngs_withfitness':
            assert self.simple_logw_col is not None
            return loaders.tsngs_withfitness_dataloader(
                tsngs_df = self.tsngs_df,
                dataflow = self.dataflow,
                target_logw_col = self.simple_logw_col,
                batch_size = self.batch_size
            )
        elif self.batch_class_name == 'tsngs_latentwithfitness':
            assert self.simple_logw_col is not None
            return loaders.tsngs_latentwithfitness_dataloader(
                tsngs_df = self.tsngs_df,
                dataflow = self.dataflow,
                target_logw_col = self.simple_logw_col,
                batch_size = self.batch_size
            )
        elif self.batch_class_name == 'tsngs':
            return loaders.tsngs_dataloader(
                tsngs_df = self.tsngs_df,
                dataflow = self.dataflow,
                batch_size = self.batch_size
            )
        assert False, 'Invalid batch class name.'


def find_batch_size(
    model: pl.LightningModule,
    tsngs_df: TimeSeriesNGSDataFrame,
    dataflow: DataFlow,
    batch_class_name: str,
    simple_logw_col: str | None = None,
) -> int:
    """
        Automatically find largest batch size that fits in memory
    """
    logger.info(f'Searching for largest batch size that fits in memory ...')
    model.batch_size = 0
    trainer = pl.Trainer(accelerator = args.get('accelerator'), devices = 1)
    tuner = Tuner(trainer)
    module = BatchSizeModule(
        tsngs_df, 
        dataflow, 
        batch_class_name,
        simple_logw_col = simple_logw_col
    )
    tuner.scale_batch_size(model, datamodule = module, mode = 'power')
    batch_size = copy.copy(model.batch_size)
    del model.batch_size
    logger.info(f'Found {batch_size=}')
    return batch_size


