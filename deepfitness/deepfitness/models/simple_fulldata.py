from loguru import logger
import numpy as np
from numpy.typing import NDArray
import functools

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deepfitness import tasks, simulate
from hackerargs import args
from deepfitness.utils import tensor_to_np
from deepfitness.data.tsngs import TimeSeriesNGSDataFrame

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.warning(f'Using {device=}')


class SimpleFitnessFullDataModel:
    def __init__(
        self, 
        tsngs_df: TimeSeriesNGSDataFrame,
        loss_func_name: str,
        init_log_fitness: torch.Tensor | None = None,
        gradient_stop_idxs: list[int] | None = None,
    ):
        """
            Simple fitness model, fit to full dataset, which offers
            significantly more stable training than batching over rounds and
            genotypes.
            This is not compatible with the pytorch lightning batch dataloader
            framework.

            This class handles parameters, loading data to device,
            lbfgs optimizer, and training loop.

            Optional parameters
            -------------------
            init_log_fitness: Specifies values to initialize log fitness
            gradient_mask_idxs: List of indices to stop gradient on.
                Used to calculate profile likelihood.
        """
        self.tsngs_df = tsngs_df
        self.num_genotypes = tsngs_df.get_num_genotypes()
        self.loss_func_name = loss_func_name
        self.loss_func = tasks.get_loss_function(loss_func_name)
        self.init_log_fitness = init_log_fitness
        self.gradient_stop_idxs = gradient_stop_idxs

        # init parameter
        if init_log_fitness is None:
            init_val = torch.randn(
                self.num_genotypes, 
                dtype = torch.float32,
                device = device,
            )
        else:
            init_val = init_log_fitness.clone()
            init_val = init_val.to(dtype = torch.float32)
            init_val = init_val.to(device)
            assert len(init_val) == self.num_genotypes
        self.log_fitness = torch.nn.Parameter(init_val)
        clamp = lambda grad: torch.clamp(grad, -10.0, 10.0)
        self.log_fitness.register_hook(clamp)

        self.log_dirmul_precision = torch.nn.Parameter(
            torch.tensor([5.0], dtype = torch.float32, device = device)
        )

        if gradient_stop_idxs is not None:
            self.apply_stop_grad_hook(gradient_stop_idxs)
            self.profiled_idxs = [i for i in range(len(self.init_log_fitness))
                                  if i not in set(self.gradient_stop_idxs)]

        # init optimizer
        self.optimizer = torch.optim.Adam(
            [self.log_fitness, self.log_dirmul_precision],
            lr = args.setdefault('simple.lr', 1e-1),
            weight_decay = 0,
        )
        self.scheduler = ReduceLROnPlateau(
            optimizer = self.optimizer, 
            patience = args.setdefault('simple.reducelronplateau.patience', 20),
            factor = args.setdefault('simple.reducelronplateau.factor', 0.3),
            verbose = True,
        )
        self.num_epochs = args.setdefault('simple.epochs', 10000)

        # init data on device
        round_cols = self.tsngs_df.get_round_cols()
        self.rounds_before = self.tsngs_df.get_before_round_cols()
        self.rounds_after = self.tsngs_df.get_after_round_cols()
        self.rounds_before_idxs = [
            round_cols.index(r) for r in self.rounds_before
        ]
        self.rounds_after_idxs = [
            round_cols.index(r) for r in self.rounds_after
        ]

        df = self.tsngs_df.df
        self.torch_data = torch.tensor(
            np.array(df[round_cols].astype(float).T),
            device = device
        )
        self.masks = torch.where(self.torch_data > 0, True, False)
        # T x G

        self.total_n_train_points = self.masks[:-1].sum()

    """
        Setup
    """
    def apply_stop_grad_hook(self, gradient_stop_idxs: list[int]) -> None:
        """ Register hook to self.log_fitness, stopping gradient at idxs.
            Used to compute profile likelihood.
        """
        stop_idx_set = set(gradient_stop_idxs)
        def hook(grad):
            grad = grad.clone()
            for i in range(grad.size(0)):
                if i in stop_idx_set:
                    grad[i] = 0
            return grad
        self.log_fitness.register_hook(hook)
        return

    """
        Access
    """
    def get_log_fitness(self) -> NDArray:
        return tensor_to_np(self.log_fitness)

    def get_train_loss(self) -> float:
        """ Compute train loss at current parameter values. """
        loss = self.compute_fulldata_train_loss(self.log_fitness)
        return float(tensor_to_np(loss))

    @functools.cache
    def get_counts(self, round_idx: int) -> torch.Tensor:
        return self.torch_data[round_idx]

    @functools.cache
    def get_mask(self, round_idx: int) -> torch.Tensor:
        """ Boolean mask where counts[round_idx] > 0 """
        return self.masks[round_idx]

    """
        Research - not for use in production
    """
    def __research__get_obs_fisher_info_matrix(
        self, 
        extra_idx: int
    ) -> torch.Tensor:
        """ Evaluates fisher info matrix at centered log_fitness.
            Denoting current log fitness shape as G, we remove
            extra_idx and return a (G-1) x (G-1) matrix due to
            identifiability properties.
        """
        centered_log_fitness = self.log_fitness - torch.mean(self.log_fitness)
        fim = torch.autograd.functional.hessian(
            lambda log_fit: self.__research__identifiable_nll(
                log_fit, 
                miss_idx = extra_idx
            ), 
            torch.concat((
                centered_log_fitness[:extra_idx],
                centered_log_fitness[extra_idx + 1:],
            )),
        )
        return fim

    def __research__identifiable_nll(
        self, 
        log_fitness_missing: torch.Tensor,
        miss_idx: int,
    ) -> torch.Tensor:
        """ Compute NLL in an identifiable function of G-1 dim log fitness
            values. Fill in missing dim such that full log fitness
            vector sums to 0.
            Used to compute fisher information matrix.
        """
        assert len(log_fitness_missing) == len(self.log_fitness) - 1
        missing_val = -1 * torch.sum(log_fitness_missing).unsqueeze(0)
        full_log_fit = torch.concat((
            log_fitness_missing[:miss_idx], 
            missing_val,
            log_fitness_missing[miss_idx:]
        ))
        return self.compute_fulldata_train_loss(full_log_fit)

    """
        Loss
    """
    def compute_fulldata_train_loss(
        self, 
        log_fitness: torch.Tensor
    ) -> torch.Tensor:
        """ Compute loss over entire training dataset, 
            over all rounds and genotypes
        """
        steps_per_round = self.tsngs_df.get_steps_per_round()

        total_loss = torch.tensor([0.0], requires_grad = True, device = device)
        for i, (r0_idx, r1_idx) in enumerate(zip(
            self.rounds_before_idxs, self.rounds_after_idxs
        )):
            curr_counts = self.get_counts(r0_idx)
            next_counts = self.get_counts(r1_idx)
            mask = self.get_mask(r0_idx)

            sim_result = simulate.simulate_mask_log_fqs(
                log_W = log_fitness, 
                inp_counts = curr_counts, 
                steps = steps_per_round[i],
                mask = mask,
            )

            if self.loss_func_name == 'dirichlet_multinomial':
                loss = self.loss_func(
                    mask_logp_pred = sim_result.mask_log_p1_pred, 
                    mask_counts = next_counts[mask],
                    log_precision = self.log_dirmul_precision
                )
            else:
                loss = self.loss_func(
                    mask_logp_pred = sim_result.mask_log_p1_pred, 
                    mask_counts = next_counts[mask]
                )
            total_loss = total_loss + loss

        total_loss = total_loss / self.total_n_train_points
        return total_loss

    def batched_fulldata_loglik(
        self, 
        batched_logfit: torch.Tensor,
        subset_genotype_idxs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Input batched_logfit: B x G. Outputs B x 1 loss vector.
            Used to evaluate likelihood at many fitness vectors.

            If subset_genotype_idxs provided, evaluate NLL only on those
            genotypes. If |subset_genotype_idxs| = S, then batched_logfit
            should be B x S. 
        """
        if subset_genotype_idxs is not None:
            assert subset_genotype_idxs.shape[0] == batched_logfit.shape[1]
            subset_genotype_idxs = subset_genotype_idxs.to(device)

        steps_per_round = self.tsngs_df.get_steps_per_round()
        batched_logfit = batched_logfit.to(device)

        batch_size = batched_logfit.shape[0]

        total_loglik = torch.zeros((batch_size), requires_grad = True, device = device)
        for i, (r0_idx, r1_idx) in enumerate(zip(
            self.rounds_before_idxs, self.rounds_after_idxs
        )):
            curr_counts = self.get_counts(r0_idx)
            next_counts = self.get_counts(r1_idx)
            mask = self.get_mask(r0_idx)
            # G x 1

            if subset_genotype_idxs is not None:
                curr_counts = curr_counts[subset_genotype_idxs]
                next_counts = next_counts[subset_genotype_idxs]
                mask = mask[subset_genotype_idxs]
                # S x 1

                if all(mask == False):
                    continue

            mask_log_p1_pred = simulate.sim_batched(
                log_W = batched_logfit, 
                inp_counts = curr_counts,
                steps = steps_per_round[i],
                mask = mask
            )
            mask_r1_counts = next_counts[mask]

            if sum(mask_r1_counts) == 0:
                continue

            # m = Multinomial(
            #     int(sum(mask_r1_counts)), 
            #     logits = mask_log_p1_pred
            # )
            # loglik = m.log_prob(mask_r1_counts)
            nll = tasks.loss_dirmul_nll(
                mask_log_p1_pred,
                curr_counts[mask],
                next_counts[mask],
                log_precision = torch.tensor([9.1955], device = device),
            )
            loglik = nll * -1
            # (B)
            
            total_loglik = total_loglik + loglik

        return total_loglik

    def batched_fulldata_dirmul_loglik_with_other(
        self, 
        batched_logfit: torch.Tensor,
        subset_genotype_idxs: torch.Tensor,
        log_dirmul_precision: torch.Tensor,
    ) -> torch.Tensor:
        """ Groups data into subset genotype idxs and everything else.
        """
        assert subset_genotype_idxs.shape[0] + 1 == batched_logfit.shape[1]
        subset_genotype_idxs = subset_genotype_idxs.to(device)

        all_other_idxs = torch.ones(
            self.torch_data.shape[-1], 
            device = device, 
            dtype = torch.bool
        )
        for i in subset_genotype_idxs:
            all_other_idxs[i] = False

        batched_logfit = batched_logfit.to(device)
        log_dirmul_precision = log_dirmul_precision.to(device)

        steps_per_round = self.tsngs_df.get_steps_per_round()
        batch_size = batched_logfit.shape[0]

        total_loglik = torch.zeros((batch_size), requires_grad = True, device = device)
        for i, (r0_idx, r1_idx) in enumerate(zip(
            self.rounds_before_idxs, self.rounds_after_idxs
        )):
            curr_counts = self.get_counts(r0_idx)
            next_counts = self.get_counts(r1_idx)
            mask = self.get_mask(r0_idx)
            # G x 1

            # build others
            other_idxs = all_other_idxs & mask
            other_curr_ct = curr_counts[other_idxs].sum().unsqueeze(0)
            other_next_ct = next_counts[other_idxs].sum().unsqueeze(0)
            other_mask_val = torch.tensor([bool(len(other_idxs) > 0)], device = device)

            curr_counts = curr_counts[subset_genotype_idxs]
            next_counts = next_counts[subset_genotype_idxs]
            mask = mask[subset_genotype_idxs]
            # S x 1

            # append other
            curr_counts = torch.concatenate([curr_counts, other_curr_ct])
            next_counts = torch.concatenate([next_counts, other_next_ct])
            mask = torch.concatenate([mask, other_mask_val])

            if all(mask == False):
                continue

            mask_log_p1_pred = simulate.sim_batched(
                log_W = batched_logfit, 
                inp_counts = curr_counts,
                steps = steps_per_round[i],
                mask = mask
            )
            mask_r1_counts = next_counts[mask]

            if sum(mask_r1_counts) == 0:
                continue

            nll = tasks.loss_dirmul_nll(
                mask_log_p1_pred,
                next_counts[mask],
                log_precision = log_dirmul_precision,
            )
            loglik = nll * -1
            # (B)
            
            total_loglik = total_loglik + loglik
        return total_loglik

    """
        Training
    """
    def converged(self, losses: list[float]) -> bool:
        distance = 50
        return len(losses) > distance and losses[-distance] == losses[-1]

    def training_step(self):
        self.optimizer.zero_grad()
        loss = self.compute_fulldata_train_loss(self.log_fitness)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss)
        return loss

    def train(self) -> None:
        """ Optimizes simple fitness parameters. """
        loss = tensor_to_np(self.compute_fulldata_train_loss(self.log_fitness))
        losses = [loss]
        logger.info(f'Starting full-data simple fitness optimization ...')
        logger.info(f'Epoch 0: {loss=}')
        for epoch_idx in range(int(self.num_epochs)):
            # update step
            loss = self.training_step()

            loss = tensor_to_np(loss)
            losses.append(loss)
            logger.info(f'Epoch {epoch_idx + 1}: {loss=}')

            if self.converged(losses):
                logger.info('Detected convergence -- stopping')
                break
        return
    
    """
        Cleanup
    """
    def detach_cleanup(self):
        """ Detach all tensors within. Use to clear GPU memory. 
            >>> model.detach_cleanup()
            >>> del model
            >>> torch.cuda.empty_cache()
        """
        self.log_fitness.detach()
        self.log_dirmul_precision.detach()
        self.get_counts.cache_clear()
        self.get_mask.cache_clear()
        del self.masks
        del self.torch_data
        del self.total_n_train_points
        del self.log_fitness
        del self.log_dirmul_precision
        return