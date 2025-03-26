"""
    DataBatch contains a batch of samples.
    It is output of collater, which uses a list of TSNGSDatpoints.
"""
import numpy as np
import torch

from deepfitness.data import datapoints as p


"""
    Batching functions
"""
def stack(key: str, points: list[p.DataPoint]) -> torch.Tensor:
    """ List of (x, ...) -> (B, x, ...) """
    return torch.stack([pt[key] for pt in points])

def tensor(key: str, points: list[p.DataPoint]) -> torch.Tensor:
    """ List of (1) -> (B, 1) """
    return torch.tensor(np.array([pt[key] for pt in points]))

def singleton(key: str, points: list[p.DataPoint]) -> torch.Tensor:
    """ List of (1) -> (1) """
    return torch.tensor(points[0][key])


"""
    Batch classes
"""
class DataBatch:
    def __init__(self):
        self.store = dict()
    
    def __getitem__(self, key: str) -> torch.Tensor:
        return self.store[key] 
       
    def to_device(self, device: torch.device):
        """ Sends all held data (tensors) to device. """
        for key, val in self.store.items():
            self.store[key] = val.to(device)
        return self

    def pin_memory(self):
        for key in self.store:
            if isinstance(self.store[key], torch.Tensor):
                self.store[key] = self.store[key].pin_memory()
        return self


class GenotypeDataBatch(DataBatch):
    def __init__(
        self,
        points: list[p.GenotypeDataPoint],
        **kwargs,
    ):
        """
            Reads from kwargs first (so that, e.g., pad_variable_len_seqs
            can submit padded genotype tensors), and then takes values from
            points for keys not provided manually in kwargs.
        """
        default_key_to_action = {
            'genotype_tensor': stack,
        }
        self.store = dict(kwargs)
        for key, action in default_key_to_action.items():
            if key not in self.store:
                self.store[key] = action(key, points)
        self.batch_size = len(points)
        
    def __len__(self) -> int:
        return self.batch_size


class WarmupBatch(DataBatch):
    def __init__(
        self,
        points: list[p.WarmupDataPoint],
        **kwargs,
    ):
        """
            Reads from kwargs first (so that, e.g., pad_variable_len_seqs
            can submit padded genotype tensors), and then takes values from
            points for keys not provided manually in kwargs.
        """
        default_key_to_action = {
            'genotype_tensor': stack,
            'target_logw': tensor,
        }
        self.store = dict(kwargs)
        for key, action in default_key_to_action.items():
            if key not in self.store:
                self.store[key] = action(key, points)
        self.batch_size = len(points)
        
    def __len__(self) -> int:
        return self.batch_size


class TSNGSDataBatch(DataBatch):
    def __init__(
        self,
        points: list[p.TSNGSDataPoint],
        **kwargs,
    ):
        """ Contains a batch of genotype tensors, counts, next_counts, and
            optional other info (such as padding masks).

            Properties
            ----------
            genotype_tensor: [batch_size, ...]
                torch.Tensor representation of a genotype, for deep fitness
            genotype_idx: torch.Tensor, list[int], [batch_size]
                Indices of genotypes in df, for simple fitness
            count: torch.Tensor, int, [batch_size, 1]
                Read counts at current round
            next_count: torch.Tensor, int, [batch_size, 1]
                Read counts at next round
            round_pair_idx: int
                Index of current round pair (before/after)
            steps_to_next_round: int
                Number of steps/generations to simulate to next round.
                
            Provided to DeepFitnessModel (a pl.LightningModule), which
            sends it to device, and uses it in training_step().
            All data stored here should be torch tensors.

            Reads from kwargs first (so that, e.g., pad_variable_len_seqs
            can submit padded genotype tensors), and then takes values from
            points for keys not provided manually in kwargs.
        """
        default_key_to_action = {
            'genotype_tensor': stack,
            'genotype_idx': tensor,
            'count': tensor,
            'next_count': tensor,
            'round_pair_idx': singleton,
            'steps_to_next_round': singleton,
        }
        self.store = dict(kwargs)
        for key, action in default_key_to_action.items():
            if key not in self.store:
                self.store[key] = action(key, points)
        self.batch_size = len(points)

    def __len__(self) -> int:
        return self.batch_size
    

class TSNGSwithFitnessDataBatch(DataBatch):
    def __init__(
        self,
        points: list[p.TSNGSwithFitnessDataPoint],
        **kwargs,
    ):
        default_key_to_action = {
            'genotype_tensor': stack,
            'genotype_idx': tensor,
            'count': tensor,
            'next_count': tensor,
            'round_pair_idx': singleton,
            'steps_to_next_round': singleton,
            'target_logw': tensor,
        }
        self.store = dict(kwargs)
        for key, action in default_key_to_action.items():
            if key not in self.store:
                self.store[key] = action(key, points)
        self.batch_size = len(points)

    def __len__(self) -> int:
        return self.batch_size


class TSNGSLatentDataBatch(DataBatch):
    def __init__(
        self,
        points: list[p.TSNGSLatentDataPoint],
        **kwargs,
    ):
        default_key_to_action = {
            'genotype_tensor': stack,
            'genotype_idx': tensor,
            'count': tensor,
            'time': singleton,
            'first_time': tensor,
            'track_idx': singleton,
        }
        self.store = dict(kwargs)
        for key, action in default_key_to_action.items():
            if key not in self.store:
                self.store[key] = action(key, points)
        self.batch_size = len(points)

    def __len__(self) -> int:
        return self.batch_size


class TSNGSLatentwithFitnessDataBatch(DataBatch):
    def __init__(
        self,
        points: list[p.TSNGSLatentwithFitnessDataPoint],
        **kwargs,
    ):
        default_key_to_action = {
            'genotype_tensor': stack,
            'genotype_idx': tensor,
            'count': tensor,
            'time': singleton,
            'first_time': tensor,
            'track_idx': singleton,
            'target_logw': tensor,
        }
        self.store = dict(kwargs)
        for key, action in default_key_to_action.items():
            if key not in self.store:
                self.store[key] = action(key, points)
        self.batch_size = len(points)

    def __len__(self) -> int:
        return self.batch_size