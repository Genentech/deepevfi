"""
    Note - to avoid memory duplication on multiprocessing workers due to
    copy-on-access (https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662),
    ensure __getitem__ accesses np arrays.
    List of strings are stored as:
    >>> strings_byte = np.array(strings).astype(np.string_)
    and accessed as
    >>> str(strings_byte[0], encoding='utf-8')
"""
import functools
import numpy as np

import torch
from torch.utils.data import Dataset

from deepfitness.data.tsngs import TimeSeriesNGSDataFrame
from deepfitness.genotype.featurizers import GenotypeFeaturizer
from deepfitness.genotype.feature_stores import FeatureStore
from deepfitness.data import datapoints
from deepfitness.data.databatches import DataBatch


class TorchTSNGSDataset(Dataset):
    def __init__(
        self,
        tsngs_df: TimeSeriesNGSDataFrame,
        featurizer: GenotypeFeaturizer,
        feature_store: FeatureStore | None,
    ):
        """ A map-style torch dataset holding:
            - a TimeSeriesNGSDataFrame data object,
            - a GenotypeFeaturizer function
            - optional feature_store: {genotype str: np array}
        """
        self.tsngs_df = tsngs_df
        self.featurizer = featurizer
        self.feature_store = feature_store
        self.long_df = tsngs_df.form_long_train_df()

        self.long_genotypes = np.array(self.long_df['Genotype']).astype(np.string_)
        self.long_genotype_idxs = np.array(self.long_df['Genotype index'])
        self.long_counts = np.array(self.long_df['Count'])
        self.long_next_counts = np.array(self.long_df['Next count'])
        self.flat_round_pair_idx_list = np.array(self.long_df['Round pair index'])
        self.steps_to_next_round = np.array(self.long_df['Steps to next round'])

    def __len__(self) -> int:
        return len(self.long_df)

    def get_genotype_tensor(self, genotype: str) -> torch.Tensor:
        if self.feature_store is not None:
            genotype_tensor = torch.tensor(self.feature_store[genotype])
        else:
            genotype_tensor = self.featurizer.featurize(genotype)
        return genotype_tensor

    @functools.cache
    def __getitem__(self, idx: int) -> datapoints.TSNGSDataPoint:
        """ Get idx-th element in long-form dataframe.
            A single sample includes:
            - Genotype
            - Genotype idx
            - Count
            - Next count
            - Round idx
            - Steps to next round
            Must return torch tensors. (Does not have to be on device). 
        """
        assert idx < len(self.long_df)
        genotype = str(self.long_genotypes[idx], encoding = 'utf-8')
        return datapoints.TSNGSDataPoint(
            genotype_tensor = self.get_genotype_tensor(genotype), 
            genotype_idx = self.long_genotype_idxs[idx],
            count = self.long_counts[idx], 
            next_count = self.long_next_counts[idx],
            round_pair_idx = self.flat_round_pair_idx_list[idx],
            steps_to_next_round = self.steps_to_next_round[idx],
        )

    def get_batch_ids(self) -> list[int]:
        """ Long-form list of batch ids: manually sample batches by batch id """
        return self.flat_round_pair_idx_list
    

class TorchGenotypeDataset(Dataset):
    def __init__(
        self,
        tsngs_df: TimeSeriesNGSDataFrame,
        featurizer: GenotypeFeaturizer,
        feature_store: FeatureStore | None,
    ):
        """ A map-style torch dataset holding:
            - a TimeSeriesNGSDataFrame data object,
            - a GenotypeFeaturizer function
        """
        self.tsngs_df = tsngs_df
        self.featurizer = featurizer
        self.feature_store = feature_store
        self.long_df = tsngs_df.form_long_genotype_df()
        self.long_genotypes = np.array(self.long_df['Genotype']).astype(np.string_)

    def __len__(self) -> int:
        return len(self.long_df)

    def get_genotype_tensor(self, genotype: str) -> torch.Tensor:
        if self.feature_store is not None:
            genotype_tensor = torch.tensor(self.feature_store[genotype])
        else:
            genotype_tensor = self.featurizer.featurize(genotype)
        return genotype_tensor

    @functools.cache
    def __getitem__(self, idx: int) -> datapoints.GenotypeDataPoint:
        assert idx < len(self.long_df)
        genotype = str(self.long_genotypes[idx], encoding = 'utf-8')
        return datapoints.GenotypeDataPoint(
            genotype_tensor = self.get_genotype_tensor(genotype), 
        )


class TorchWarmupDataset(Dataset):
    def __init__(
        self,
        tsngs_df: TimeSeriesNGSDataFrame,
        featurizer: GenotypeFeaturizer,
        feature_store: FeatureStore | None,
        target_logw_col: str,
    ):
        """ A map-style torch dataset holding:
            - a TimeSeriesNGSDataFrame data object,
            - a GenotypeFeaturizer function
        """
        self.tsngs_df = tsngs_df
        self.featurizer = featurizer
        self.feature_store = feature_store
        self.long_df = tsngs_df.form_long_warmup_df(target_logw_col)
        self.long_genotypes = np.array(self.long_df['Genotype']).astype(np.string_)
        self.target_logw = np.array(self.long_df['Target logw'])

    def __len__(self) -> int:
        return len(self.long_df)

    def get_genotype_tensor(self, genotype: str) -> torch.Tensor:
        if self.feature_store is not None:
            genotype_tensor = torch.tensor(self.feature_store[genotype])
        else:
            genotype_tensor = self.featurizer.featurize(genotype)
        return genotype_tensor

    @functools.cache
    def __getitem__(self, idx: int) -> datapoints.WarmupDataPoint:
        assert idx < len(self.long_df)
        genotype = str(self.long_genotypes[idx], encoding = 'utf-8')
        return datapoints.WarmupDataPoint(
            genotype_tensor = self.get_genotype_tensor(genotype), 
            target_logw = self.target_logw[idx],
        )


class TorchTSNGSwithFitnessDataset(Dataset):
    def __init__(
        self,
        tsngs_df: TimeSeriesNGSDataFrame,
        featurizer: GenotypeFeaturizer,
        feature_store: FeatureStore | None,
        target_logw_col: str,
    ):
        """ A map-style torch dataset for training on time-series data
            (with time series batch sampler) with reference fitness values.
        """
        self.tsngs_df = tsngs_df
        self.featurizer = featurizer
        self.feature_store = feature_store
        self.long_df = tsngs_df.form_long_train_df(
            target_logw_col = target_logw_col
        )
        self.long_genotypes = np.array(self.long_df['Genotype']).astype(np.string_)
        self.long_genotype_idxs = np.array(self.long_df['Genotype index'])
        self.target_logw = np.array(self.long_df[target_logw_col])
        self.long_counts = np.array(self.long_df['Count'])
        self.long_next_counts = np.array(self.long_df['Next count'])
        self.flat_round_pair_idx_list = np.array(self.long_df['Round pair index'])
        self.steps_to_next_round = np.array(self.long_df['Steps to next round'])

    def __len__(self) -> int:
        return len(self.long_df)

    def get_genotype_tensor(self, genotype: str) -> torch.Tensor:
        if self.feature_store is not None:
            genotype_tensor = torch.tensor(self.feature_store[genotype])
        else:
            genotype_tensor = self.featurizer.featurize(genotype)
        return genotype_tensor

    @functools.cache
    def __getitem__(self, idx: int) -> datapoints.TSNGSwithFitnessDataPoint:
        assert idx < len(self.long_df)
        genotype = str(self.long_genotypes[idx], encoding = 'utf-8')
        return datapoints.TSNGSwithFitnessDataPoint(
            genotype_tensor = self.get_genotype_tensor(genotype), 
            genotype_idx = self.long_genotype_idxs[idx],
            count = self.long_counts[idx], 
            next_count = self.long_next_counts[idx],
            round_pair_idx = self.flat_round_pair_idx_list[idx],
            steps_to_next_round = self.steps_to_next_round[idx],
            target_logw = self.target_logw[idx],
        )

    def get_batch_ids(self) -> list[int]:
        """ Long-form list of batch ids: manually sample batches by batch id """
        return self.flat_round_pair_idx_list


class TorchTSNGSLatentDataset(Dataset):
    def __init__(
        self,
        tsngs_df: TimeSeriesNGSDataFrame,
        featurizer: GenotypeFeaturizer,
        feature_store: FeatureStore | None,
    ):
        """ A map-style torch dataset for training on time-series data
            (with time series batch sampler).
        """
        self.tsngs_df = tsngs_df
        self.featurizer = featurizer
        self.feature_store = feature_store
        self.long_df = tsngs_df.form_long_latent_train_df()

        self.long_genotypes = np.array(self.long_df['Genotype']).astype(np.string_)
        self.long_genotype_idxs = np.array(self.long_df['Genotype index'])
        self.long_counts = np.array(self.long_df['Count'])
        self.long_time = np.array(self.long_df['Time'])
        self.long_first_time = np.array(self.long_df['First time'])
        self.long_track_idx = np.array(self.long_df['Track index'])

    def __len__(self) -> int:
        return len(self.long_df)

    def get_genotype_tensor(self, genotype: str) -> torch.Tensor:
        if self.feature_store is not None:
            genotype_tensor = torch.tensor(self.feature_store[genotype])
        else:
            genotype_tensor = self.featurizer.featurize(genotype)
        return genotype_tensor

    @functools.cache
    def __getitem__(self, idx: int) -> datapoints.TSNGSLatentDataPoint:
        assert idx < len(self.long_df)
        genotype = str(self.long_genotypes[idx], encoding = 'utf-8')
        return datapoints.TSNGSLatentDataPoint(
            genotype_tensor = self.get_genotype_tensor(genotype), 
            genotype_idx = self.long_genotype_idxs[idx],
            count = self.long_counts[idx], 
            time = self.long_time[idx],
            first_time = self.long_first_time[idx],
            track_idx = self.long_track_idx[idx],
        )

    def get_batch_ids(self) -> list[tuple[int, int]]:
        """ Long-form list of batch ids: manually sample batches by batch id """
        return list(zip(self.long_track_idx, self.long_time))


class TorchTSNGSLatentwithFitnessDataset(Dataset):
    def __init__(
        self,
        tsngs_df: TimeSeriesNGSDataFrame,
        featurizer: GenotypeFeaturizer,
        feature_store: FeatureStore | None,
        target_logw_col: str,
    ):
        """ A map-style torch dataset for training on time-series data
            (with time series batch sampler) with reference fitness values.
        """
        self.tsngs_df = tsngs_df
        self.featurizer = featurizer
        self.feature_store = feature_store
        self.long_df = tsngs_df.form_long_latent_train_df(target_logw_col)

        self.long_genotypes = np.array(self.long_df['Genotype']).astype(np.string_)
        self.long_genotype_idxs = np.array(self.long_df['Genotype index'])
        self.long_counts = np.array(self.long_df['Count'])
        self.long_time = np.array(self.long_df['Time'])
        self.long_first_time = np.array(self.long_df['First time'])
        self.long_track_idx = np.array(self.long_df['Track index'])
        self.target_logw = np.array(self.long_df[target_logw_col])

    def __len__(self) -> int:
        return len(self.long_df)

    def get_genotype_tensor(self, genotype: str) -> torch.Tensor:
        if self.feature_store is not None:
            genotype_tensor = torch.tensor(self.feature_store[genotype])
        else:
            genotype_tensor = self.featurizer.featurize(genotype)
        return genotype_tensor

    @functools.cache
    def __getitem__(self, idx: int) -> datapoints.TSNGSLatentwithFitnessDataPoint:
        assert idx < len(self.long_df)
        genotype = str(self.long_genotypes[idx], encoding = 'utf-8')
        return datapoints.TSNGSLatentwithFitnessDataPoint(
            genotype_tensor = self.get_genotype_tensor(genotype), 
            genotype_idx = self.long_genotype_idxs[idx],
            count = self.long_counts[idx], 
            time = self.long_time[idx],
            first_time = self.long_first_time[idx],
            track_idx = self.long_track_idx[idx],
            target_logw = self.target_logw[idx],
        )

    def get_batch_ids(self) -> list[tuple[int, int]]:
        """ Long-form list of batch ids: manually sample batches by batch id """
        return list(zip(self.long_track_idx, self.long_time))