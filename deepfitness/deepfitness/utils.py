import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from numpy.typing import NDArray


def tensor_to_np(
    tensor: torch.Tensor, 
    reduce_singleton: bool = True
) -> NDArray:
    """ Convert torch.tensor to np.array. This is a blocking operation. """
    accept_types = [np.ndarray, np.float64, float]
    if type(tensor) in accept_types:
        return tensor
    x = tensor.to('cpu').detach().numpy()
    if reduce_singleton:
        if x.shape == (1,) or x.shape == torch.Size([]):
            return float(x)
    return x


def freeze_module(module: torch.nn.Module):
    module.eval()
    for param in module.parameters(recurse=True):
        param.requires_grad = False


def fill_masked_tensor(
    masked_values: torch.Tensor, 
    mask: torch.Tensor,
) -> NDArray:
    """ Fills mask-shaped array with masked_values.
        masked_values: shape M x 1
        mask: shape N x 1, where N > M
        Output: N x 1
    """
    n = len(mask)
    tensor = torch.zeros([n], dtype=masked_values.dtype)
    idxs = torch.nonzero(mask).flatten()
    tensor[idxs] = masked_values
    return tensor_to_np(tensor)

