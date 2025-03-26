import numpy as np
import pandas as pd
from numpy.typing import NDArray

from torch.utils.data import DataLoader, Sampler
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from deepfitness.data.tsngs import TimeSeriesNGSDataFrame
from deepfitness.data.spr import SPRDataFrame
from hackerargs import args
from deepfitness.genotype import dataflows
from deepfitness.data import datasets


"""
    DataLoaders
"""
def tsngs_dataloader(
    tsngs_df: TimeSeriesNGSDataFrame,
    dataflow: dataflows.DataFlow,
    batch_size: int | None = None,
) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
    dataset = datasets.TorchTSNGSDataset(
        tsngs_df = tsngs_df,
        featurizer = dataflow.featurizer,
        feature_store = dataflow.feature_store,
    )
    batch_sampler = ManualBatchSampler(
        batch_size = batch_size if batch_size else args.get('batch_size'),
        batch_ids = dataset.get_batch_ids(),
    )
    dataloader = DataLoader(
        dataset = dataset,
        batch_sampler = batch_sampler,
        collate_fn = dataflow.collater,
        pin_memory = args.setdefault('dataloader_pin_memory', True),
        num_workers = args.setdefault('dataloader_num_workers', 0),
        persistent_workers = args.setdefault('dataloader_persistent_workers', False),
    )
    return dataloader


def tsngs_latent_dataloader(
    tsngs_df: TimeSeriesNGSDataFrame,
    dataflow: dataflows.DataFlow,
    batch_size: int | None = None,
) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
    dataset = datasets.TorchTSNGSLatentDataset(
        tsngs_df = tsngs_df,
        featurizer = dataflow.featurizer,
        feature_store = dataflow.feature_store,
    )
    batch_sampler = ManualBatchSampler(
        batch_size = batch_size if batch_size else args.get('batch_size'),
        batch_ids = dataset.get_batch_ids(),
    )
    dataloader = DataLoader(
        dataset = dataset,
        batch_sampler = batch_sampler,
        collate_fn = dataflow.collater,
        pin_memory = args.setdefault('dataloader_pin_memory', True),
        num_workers = args.setdefault('dataloader_num_workers', 0),
        persistent_workers = args.setdefault('dataloader_persistent_workers', False),
    )
    return dataloader


def tsngs_withfitness_dataloader(
    tsngs_df: TimeSeriesNGSDataFrame,
    dataflow: dataflows.DataFlow,
    target_logw_col: str,
    batch_size: int | None = None,
) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
    dataset = datasets.TorchTSNGSwithFitnessDataset(
        tsngs_df = tsngs_df,
        featurizer = dataflow.featurizer,
        feature_store = dataflow.feature_store,
        target_logw_col = target_logw_col,
    )
    batch_sampler = ManualBatchSampler(
        batch_size = batch_size if batch_size else args.get('batch_size'),
        batch_ids = dataset.get_batch_ids(),
    )
    dataloader = DataLoader(
        dataset = dataset,
        batch_sampler = batch_sampler,
        collate_fn = dataflow.collater,
        pin_memory = args.setdefault('dataloader_pin_memory', True),
        num_workers = args.setdefault('dataloader_num_workers', 0),
        persistent_workers = args.setdefault('dataloader_persistent_workers', False),
    )
    return dataloader


def tsngs_latentwithfitness_dataloader(
    tsngs_df: TimeSeriesNGSDataFrame,
    dataflow: dataflows.DataFlow,
    target_logw_col: str,
    batch_size: int | None = None,
) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
    dataset = datasets.TorchTSNGSLatentwithFitnessDataset(
        tsngs_df = tsngs_df,
        featurizer = dataflow.featurizer,
        feature_store = dataflow.feature_store,
        target_logw_col = target_logw_col,
    )
    batch_sampler = ManualBatchSampler(
        batch_size = batch_size if batch_size else args.get('batch_size'),
        batch_ids = dataset.get_batch_ids(),
    )
    dataloader = DataLoader(
        dataset = dataset,
        batch_sampler = batch_sampler,
        collate_fn = dataflow.collater,
        pin_memory = args.setdefault('dataloader_pin_memory', True),
        num_workers = args.setdefault('dataloader_num_workers', 0),
        persistent_workers = args.setdefault('dataloader_persistent_workers', False),
    )
    return dataloader


def warmup_dataloader(
    tsngs_df: TimeSeriesNGSDataFrame,
    dataflow: dataflows.DataFlow,
    target_logw_col: str,
) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
    warmup_dataset = datasets.TorchWarmupDataset(
        tsngs_df, 
        featurizer = dataflow.featurizer,
        feature_store = dataflow.feature_store,
        target_logw_col = target_logw_col
    )
    dataloader = DataLoader(
        dataset = warmup_dataset,
        batch_size = args.setdefault('warmup.batch_size', 1024),
        shuffle = True,
        drop_last = False,
        collate_fn = dataflow.collater,
        pin_memory = args.setdefault('dataloader_pin_memory', True),
        num_workers = args.setdefault('dataloader_num_workers', 0),
        persistent_workers = args.setdefault('dataloader_persistent_workers', False),
    )
    return dataloader


def predict_dataloader(
    tsngs_df: TimeSeriesNGSDataFrame,
    dataflow: dataflows.DataFlow,
) -> EVAL_DATALOADERS:
    dataset = datasets.TorchGenotypeDataset(
        tsngs_df, 
        dataflow.featurizer,
        feature_store = dataflow.feature_store,
    )
    return DataLoader(
        dataset = dataset,
        batch_size = args.setdefault('predict.batch_size', 512),
        shuffle = False,
        drop_last = False,
        collate_fn = dataflow.collater,
        num_workers = args.setdefault('dataloader_num_workers', 0),
    )


def spr_dataloader(
    spr_df: SPRDataFrame,
    dataflow: dataflows.DataFlow,
    pkd_col: str,
) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
    spr_dataset = datasets.TorchSPRDataset(
        spr_df, 
        featurizer = dataflow.featurizer,
        feature_store = dataflow.feature_store,
        pkd_col = pkd_col
    )
    dataloader = DataLoader(
        dataset = spr_dataset,
        batch_size = args.setdefault('spr.batch_size', 1024),
        shuffle = True,
        drop_last = False,
        collate_fn = dataflow.collater,
        pin_memory = args.setdefault('dataloader_pin_memory', True),
        num_workers = args.setdefault('dataloader_num_workers', 0),
        persistent_workers = args.setdefault('dataloader_persistent_workers', False),
    )
    return dataloader


"""
    Batch sampler
"""
class ManualBatchSampler(Sampler):
    def __init__(self, batch_size: int, batch_ids: list):
        """ Yields batches of indices to access a long-form dataset,
            where any given batch has the same `batch_id`, which for example
            can be a timepoint round identifier.
            Over one epoch, all valid training points are visited exactly once.
            Indices access a map-style long-form torch dataset.
        
            Inputs
            ------
            batch_size: int
                Number of items in a batch.
                Yielded batch size can be less than this value, depending
                on the total count of any batch_id.
            batch_ids: list of items (e.g., int, or tuple)
                A flattened list of batch identifiers for a long-form
                dataset (expected length ~ G x T).
        """
        self.batch_size = batch_size
        self.dataset_size = len(batch_ids)

        cats = pd.Categorical(batch_ids)
        self.code_to_idxs = {
            round_pair_idx: np.where(cats.codes == code)[0]
            for code, round_pair_idx in enumerate(list(cats.categories))
        }

    def __len__(self) -> int:
        """ The number of batches in the full dataset. """
        total_batches = 0
        for _, data_idxs in self.code_to_idxs.items():
            num_batches = len(data_idxs) // self.batch_size + 1
            total_batches += num_batches
        return total_batches

    def __iter__(self):
        """ Yield a batch of indices, with size batch_size, such that
            all indices in the batch have the same batch_id. 
            Indices are used by map-style torch dataset.
            Uniformly samples from batch_ids.
        """
        for _, data_idxs in self.code_to_idxs.items():
            np.random.shuffle(data_idxs)

            num_batches = len(data_idxs) // self.batch_size + 1
            for batch_num in range(num_batches):
                start = self.batch_size * batch_num
                end = self.batch_size * (batch_num + 1)
                yield data_idxs[start:end]

