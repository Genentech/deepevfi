"""
    Loss functions
    - Acts on two consecutive time points
    - Only updates w based on non-zero elements in r0_counts:
      genotypes absent in r0 but present in r1 cannot be predicted
      with our fitness equation, as we do not explicitly model mutation rates,
      so these rows provide no useful learning signal.
    - Supports variable num. generations between time points
"""
import torch
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial
from pyro.distributions import DirichletMultinomial
from hackerargs import args


"""
    Losses
"""
def loss_KL(
    mask_logp_pred: torch.Tensor,
    mask_counts: torch.Tensor, 
) -> torch.Tensor:
    round_1_fqs = mask_counts / torch.sum(mask_counts)
    loss = F.kl_div(mask_logp_pred, round_1_fqs, reduction = "sum")
    return loss


def loss_multinomial_nll(
    mask_logp_pred: torch.Tensor,
    mask_counts: torch.Tensor, 
) -> torch.Tensor:
    pop_1 = int(mask_counts.sum().detach().cpu())
    if 'tasks.multinomial.validate_args' in args:
        likelihood_dist = Multinomial(
            total_count = pop_1, 
            logits = mask_logp_pred,
            validate_args = args.get('tasks.multinomial.validate_args')
        )
    else:
        likelihood_dist = Multinomial(
            total_count = pop_1, 
            logits = mask_logp_pred,
        )
    nll = -1 * likelihood_dist.log_prob(mask_counts)
    return nll


def loss_dirmul_nll(
    mask_logp_pred: torch.Tensor,
    mask_counts: torch.Tensor, 
    log_precision: torch.Tensor | None = None,
) -> torch.Tensor:
    concentration = torch.exp(mask_logp_pred)

    if log_precision is None:
        concentration = concentration * mask_counts.sum()
        # concentration = concentration * mask_counts.sum()**(3/4)
        # concentration = concentration * torch.sqrt(mask_counts.sum())
    else:
        concentration = concentration * torch.exp(log_precision)

    # force validity (entries > 0)
    # can trigger on variants with extremely low log pred fq
    concentration = torch.max(concentration, torch.tensor(1e-30))

    dir_mul = DirichletMultinomial(
        concentration = concentration,
        total_count = mask_counts.sum(),
    )
    nll = -1 * dir_mul.log_prob(mask_counts)
    return nll


"""
    Getter
"""
def get_loss_function(name: str) -> callable:
    loss_to_func = {
        'kl': loss_KL, 
        'multinomial': loss_multinomial_nll,
        'dirichlet_multinomial': loss_dirmul_nll,
    }
    if name not in loss_to_func:
        print(f'ERROR: Invalid loss function name')
    return loss_to_func[name]