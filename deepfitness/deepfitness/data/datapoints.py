"""
    DataPoints contain a single data sample.
    They are returned by torch dataset __getitem__.
"""
import numpy as np
from typing import Union, Any

import torch


class DataPoint:
    def __init__(self):
        pass


class GenotypeDataPoint(DataPoint):
    def __init__(
        self,
        genotype_tensor: torch.Tensor, 
    ):
        self.store = {
            'genotype_tensor': genotype_tensor,
        }

    def __getitem__(self, key: str) -> Any: 
        return self.store[key]


class WarmupDataPoint(DataPoint):
    def __init__(
        self,
        genotype_tensor: torch.Tensor,
        target_logw: torch.Tensor,
    ):
        self.store = {
            'genotype_tensor': genotype_tensor,
            'target_logw': target_logw,
        }

    def __getitem__(self, key: str) -> Any: 
        return self.store[key]


class TSNGSDataPoint(DataPoint):
    """ Properties
        ----------
        genotype_tensor
            torch.Tensor representation of a genotype, for deep fitness
        genotype_idx: int
            Index of genotype in df, used for simple fitness
        count: torch.Tensor, int
            Read count at current round
        next_count: torch.Tensor, int
            Read count at next round
        round_pair_idx: int
            Index of current round
        steps_to_next_round: float
            Number of steps/generations to simulate to next round.
            
        All valid datapoints must have count > 0.
    """
    def __init__(
        self, 
        genotype_tensor: torch.Tensor, 
        genotype_idx: int,
        count: int, 
        next_count: int,
        round_pair_idx: int,
        steps_to_next_round: float,
    ):
        self.store = {
            'genotype_tensor': genotype_tensor,
            'genotype_idx': genotype_idx,
            'count': count,
            'next_count': next_count,
            'round_pair_idx': round_pair_idx,
            'steps_to_next_round': steps_to_next_round,
        }

    def __getitem__(self, key: str) -> Any: 
        return self.store[key]


class TSNGSwithFitnessDataPoint(DataPoint):
    def __init__(
        self, 
        genotype_tensor: torch.Tensor, 
        genotype_idx: int,
        count: int, 
        next_count: int,
        round_pair_idx: int,
        steps_to_next_round: float,
        target_logw: float,
    ):
        self.store = {
            'genotype_tensor': genotype_tensor,
            'genotype_idx': genotype_idx,
            'count': count,
            'next_count': next_count,
            'round_pair_idx': round_pair_idx,
            'steps_to_next_round': steps_to_next_round,
            'target_logw': target_logw,
        }

    def __getitem__(self, key: str) -> Any: 
        return self.store[key]


class TSNGSLatentDataPoint(DataPoint):
    def __init__(
        self, 
        genotype_tensor: torch.Tensor, 
        genotype_idx: int,
        count: int, 
        time: float,
        first_time: float,
        track_idx: int,
    ):
        self.store = {
            'genotype_tensor': genotype_tensor,
            'genotype_idx': genotype_idx,
            'count': count,
            'time': time,
            'first_time': first_time,
            'track_idx': track_idx,
        }

    def __getitem__(self, key: str) -> Any: 
        return self.store[key]


class TSNGSLatentwithFitnessDataPoint(DataPoint):
    def __init__(
        self, 
        genotype_tensor: torch.Tensor, 
        genotype_idx: int,
        count: int, 
        time: float,
        first_time: float,
        track_idx: int,
        target_logw: float,
    ):
        self.store = {
            'genotype_tensor': genotype_tensor,
            'genotype_idx': genotype_idx,
            'count': count,
            'time': time,
            'first_time': first_time,
            'track_idx': track_idx,
            'target_logw': target_logw,
        }

    def __getitem__(self, key: str) -> Any: 
        return self.store[key]