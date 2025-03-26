"""
    Simulate evolutionary dynamics
"""
from dataclasses import dataclass
import torch


@dataclass
class SimulationResult:
    mask_r0_counts: torch.Tensor
    mask: torch.Tensor
    mask_log_p1_pred: torch.Tensor


def simulate_mask_log_fqs(
    log_W: torch.Tensor, 
    inp_counts: torch.Tensor, 
    steps: float = 1.0,
    mask: torch.Tensor | None = None,
) -> SimulationResult:
    """ Outputs log frequencies from simulating simple evolutionary dynamics
        on inp_counts using log_W.

        inp_counts shape: G x 1
            Same shape as log_W.
        steps: float, number of generations to simulate.

        output shape: Can be less than G x 1, based on inp_counts mask (> 0).
        Acts only on non-zero entries of inp_counts.
        If there are N of these where N < G, the output vector has shape N x 1.
    """
    if mask is None:
        mask = torch.where(inp_counts > 0, True, False)
    log_W = torch.squeeze(log_W)[mask]
    inp = inp_counts[mask]

    log_sim = torch.log(inp) + steps * log_W
    log_sim = log_sim - torch.logsumexp(log_sim, dim = 0)

    return SimulationResult(
        mask_r0_counts = inp_counts[mask],
        mask = mask,
        mask_log_p1_pred = log_sim,
    )


# batching
def sim_batched(
    log_W: torch.Tensor, 
    inp_counts: torch.Tensor, 
    steps: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """ log_W: B x G
        inp_counts: G x 1
        mask: G x 1
        
        output shape: B x N where N is the number of non-zero entries
            in inp_counts. N < G.
    """
    if mask is None:
        mask = torch.where(inp_counts > 0, True, False)

    log_W = log_W[:, mask]
    # B x N

    inp = inp_counts[mask]

    log_sim = torch.log(inp) + steps * log_W
    # B x N

    normalizer = torch.logsumexp(log_sim, dim = 1)
    normalizer = normalizer[:, None]
    # B x N

    log_sim = log_sim - normalizer
    return log_sim

