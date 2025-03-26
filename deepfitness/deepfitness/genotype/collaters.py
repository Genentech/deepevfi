"""
    Collaters are functions used by torch DataLoader to convert
    a list of DataPoints into a DataBatch object, which wraps
    a dict with tensor values.

    Input: List of B TSNGSDataPoint objects, with fields
        'genotype_tensor', 'count', 'next_count'.
    Output: TSNGSDataBatch, with the same fields, and optional extra fields.
        'genotype_tensor': B x (singleton input shape)
        'count': float tensor, B x 1 
        'next_count': float tensor, B x 1,
        etc. 

    Input samples are obtained by dataloader, using batch_sampler to
    obtain a list of indices (ints) that index into a tensor 
    dataset.

    Collater function signature can only take samples as input, as 
    collate function is passed to torch DataLoader to call.
    To use additional information in collation (e.g., padding index),
    store and access them through args.
"""
import torch
import numpy as np

from deepfitness.data import datapoints as p
from deepfitness.data import databatches as b
from hackerargs import args
from deepfitness.genotype import featurizers


def points_to_batch_type(datapoints: list[p.DataPoint]) -> b.DataBatch:
    type_to_batch = {
        p.GenotypeDataPoint: b.GenotypeDataBatch,
        p.WarmupDataPoint: b.WarmupBatch,
        p.TSNGSDataPoint: b.TSNGSDataBatch,
        p.TSNGSwithFitnessDataPoint: b.TSNGSwithFitnessDataBatch,
        p.TSNGSLatentDataPoint: b.TSNGSLatentDataBatch,
        p.TSNGSLatentwithFitnessDataPoint: b.TSNGSLatentwithFitnessDataBatch,
    }
    return type_to_batch[type(datapoints[0])]


def check_valid(samples: list[p.TSNGSDataPoint]) -> None:
    unique = lambda l: len(set(l)) == 1
    if type(samples[0]) == p.TSNGSDataPoint:
        assert unique([sample['round_pair_idx'] for sample in samples])
        assert unique([sample['steps_to_next_round'] for sample in samples])
    return


def default_collater(samples: list[p.DataPoint]) -> b.DataBatch:
    check_valid(samples)
    batch_type = points_to_batch_type(samples)
    batch = batch_type(samples)
    return batch


def pad_variable_len_seqs(samples: list[p.DataPoint]) -> b.DataBatch:
    """ Pads variable length genotype tensor sequences.
        Expects tensors to be int indices, so -1 padding value is used.
        For a list of B genotype tensors with max length L, 
        returns padded tensor and padding mask, both with shape B x L.
        Adds field 'padding_mask'.
        For an alphabet of length A, index A is used as padding index.
    """
    check_valid(samples)
    alphabet_size = len(featurizers.alphabets[args.get('ft.alphabet')])
    pad_value = alphabet_size
    batch_gts = torch.nn.utils.rnn.pad_sequence(
        sequences = [s['genotype_tensor'] for s in samples], 
        batch_first = True, 
        padding_value = pad_value
    )

    batch_type = points_to_batch_type(samples)
    batch = batch_type(
        samples,
        genotype_tensor = batch_gts,
        padding_mask = (batch_gts == pad_value),
    )
    return batch


# getter
def get_collater(name: str):
    name_to_collater = {
        'default_collater': default_collater,
        'pad_variable_len_seqs': pad_variable_len_seqs,
    }
    return name_to_collater[name]