"""
    Converters are stateless functions that convert one data type
    to a different data type.
"""
import torch


def str2idxtensor(x: str, char_to_idx = dict) -> torch.Tensor:
    """ Convert string of length N, with alphabet of length A, into
        a tensor of size N of int indices.
    """
    return torch.tensor([char_to_idx[c] for c in x])