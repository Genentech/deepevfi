"""
    EVFI: full data, latent frequency parameterization
    Given GxT count data, infers Gx1 fitness vector and 
    Gx1 abundance vector.
    Uses empirical data to determine presence/absence time ranges.
"""

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import functools

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deepfitness import tasks, roundtracks
from hackerargs import args
from deepfitness.utils import tensor_to_np
from deepfitness.data.tsngs import TimeSeriesNGSDataFrame

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.warning(f'Using {device=}')


class SimpleFitnessFullDataLatentModel:
    def __init__(
        self, 
        tsngs_df: TimeSeriesNGSDataFrame,
        loss_func_name: str,
        init_log_fitness: torch.Tensor | None = None,
        init_log_abundance: torch.Tensor | None = None,
        init_log_dirmul_precision: float = 5.0,
    ):
        """
            EVFI: full data, latent frequency parameterization
            Given GxT count data, infers Gx1 fitness vector and 
            Gx1 abundance vector.
            Uses empirical data to determine presence/absence time ranges.

            This class handles parameters, loading data to device,
            lbfgs optimizer, and training loop.

            Optional
            -------------------
            init_log_fitness: Specifies values to initialize log fitness
        """
        self.tsngs_df = tsngs_df
        self.num_genotypes = tsngs_df.get_num_genotypes()
        self.loss_func_name = loss_func_name
        self.loss_func = tasks.get_loss_function(loss_func_name)
        self.init_log_fitness = init_log_fitness

        # init data on device
        round_cols = self.tsngs_df.get_round_cols()
        self.num_rounds = len(round_cols)
        self.rounds_before = self.tsngs_df.get_before_round_cols()
        self.rounds_after = self.tsngs_df.get_after_round_cols()
        self.rounds_before_idxs = [
            round_cols.index(r) for r in self.rounds_before
        ]
        self.rounds_after_idxs = [
            round_cols.index(r) for r in self.rounds_after
        ]
        self.steps_per_round = tsngs_df.get_steps_per_round()

        df = self.tsngs_df.df
        self.torch_data = torch.tensor(
            np.array(df[round_cols].astype(float).T),
            device = device
        )
        self.masks = torch.where(self.torch_data > 0, True, False)
        # T x G

        count_data = np.array(df[round_cols])
        self.first_presence = torch.tensor(
            roundtracks.get_first_presence(count_data), 
            device = device
        )
        self.len_presence = torch.tensor(
            roundtracks.get_len_presence(count_data),
            device = device
        )
        # G x 1, int

        self.cumulative_time = torch.tensor(
            np.cumsum([0.0] + self.steps_per_round),
            device = device
        )
        self.first_times = self.get_first_times(self.first_presence)
        # G x 1, float

        self.total_n_train_points = self.masks[:-1].sum()

        # init parameters
        self.log_fitness = torch.nn.Parameter(
            self.initialize_log_fitness(init_log_fitness)
        )
        self.log_dirmul_precision = torch.nn.Parameter(
            torch.tensor([init_log_dirmul_precision], 
            dtype = torch.float32, 
            device = device)
        )
        self.log_abundance = torch.nn.Parameter(
            self.initialize_log_abundance(init_log_abundance)
        )

        # hooks
        clamp = lambda grad: torch.clamp(grad, -10.0, 10.0)
        self.log_fitness.register_hook(clamp)

        # init optimizer    
        self.optimizer = torch.optim.Adam(
            [self.log_fitness, self.log_dirmul_precision, self.log_abundance],
            lr = args.setdefault('simple.lr', 1e-3),
            weight_decay = 0,
        )
        self.scheduler = ReduceLROnPlateau(
            optimizer = self.optimizer, 
            patience = args.setdefault('simple.reducelronplateau.patience', 20),
            factor = args.setdefault('simple.reducelronplateau.factor', 0.3),
            verbose = True,
        )
        self.num_epochs = args.setdefault('simple.epochs', 1000)

    """
        Init
    """
    def initialize_log_fitness(
        self, 
        init_log_fitness: torch.Tensor | None
    ) -> torch.Tensor:
        if init_log_fitness is None:
            init_val = torch.randn(
                self.num_genotypes, 
                dtype = torch.float32,
                device = device,
            )
        else:
            assert len(init_log_fitness) == self.num_genotypes
            init_val = init_log_fitness.clone()
            init_val = init_val.to(dtype = torch.float32)
            init_val = init_val.to(device)
        return init_val

    def initialize_log_abundance(
        self, 
        init_log_abundance: torch.Tensor | None
    ) -> torch.Tensor:
        if init_log_abundance is None:
            init_val = self.smart_init_log_abundance()
        else:
            assert len(init_log_abundance) == self.num_genotypes
            init_val = init_log_abundance.clone()
            init_val = init_val.to(dtype = torch.float32)
            init_val = init_val.to(device)
        return init_val

    def smart_init_log_abundance(self) -> torch.Tensor:
        """ Heuristic strategy for initializing log abundance, using count
            data: set to log of count in first observed round.
            returns tensor float32 on device
        """
        gt_idxs = torch.tensor(range(self.num_genotypes), device = device)
        first_counts = torch.stack(
            [self.torch_data[fp][gt_idx]
            for fp, gt_idx in zip(self.first_presence.to(torch.int), gt_idxs)]
        )
        return torch.log(first_counts)

    """
        Time
    """
    def get_first_times(self, first_presence: torch.Tensor) -> torch.Tensor:
        """ Compute the first time when each variant enters population.
            first_presence: G x 1 tensor, of round indices of first presence
            Returns G x 1 float time tensor.
        """
        return torch.stack([
            self.cumulative_time[idx]
            for idx in first_presence.to(torch.int)
        ])

    """
        Access
    """
    def get_log_fitness(self) -> NDArray:
        return tensor_to_np(self.log_fitness)

    def get_log_abundance(self) -> NDArray:
        return tensor_to_np(self.log_abundance)

    def get_log_dirmul_precision(self) -> float:
        return tensor_to_np(self.log_dirmul_precision)

    def get_nll(self) -> float:
        """ Compute NLL at current parameter values. """
        return float(tensor_to_np(self.nll(self.log_fitness, self.log_abundance)))

    @functools.cache
    def get_counts(self, round_idx: int) -> torch.Tensor:
        return self.torch_data[round_idx]

    @functools.cache
    def get_mask(self, round_idx: int) -> torch.Tensor:
        """ Boolean mask where counts[round_idx] > 0 """
        return self.masks[round_idx]

    """
        Loss, prediction
    """
    def predict_unmasked_log_frequencies(
        self, 
        log_fitness: torch.Tensor,
        log_abundance: torch.Tensor,
        time: float,
        first_times: torch.Tensor,        
    ) -> torch.Tensor:
        """ Predict genotype variant frequencies at t = time
            log_fitness: G
            log_abundance: G
            first_times: G
        """
        assert len(log_fitness.shape) == 1, 'Batching not supported'
        num_genotypes = log_fitness.shape

        t = torch.ones(num_genotypes, device = device) * time
        time_since = t - first_times
        assert all(time_since > 0), 'Careful'

        pred_la = log_abundance + time_since * log_fitness
        pred_log_fq = pred_la - torch.logsumexp(pred_la, dim = 0)
        return pred_log_fq

    def predict_masked_log_frequencies(
        self, 
        log_fitness: torch.Tensor,
        log_abundance: torch.Tensor,
        round_idx: int,
        first_presence: torch.Tensor,
        len_presence: torch.Tensor,
        first_times: torch.Tensor,
    ) -> torch.Tensor:
        """ Predict genotype variant frequencies at round_idx, using
            - log_fitness
            - log_abundance
            - first_presence
            - len_presence
            - steps_per_round
            Only predicts for variants that are "present" at round_idx:
            denote this num M.
            Returns an (M)-shape log frequency vector.
            If log_fitness/log_abundance has shape (B, G), returns (B, M).
        """
        # If shape (G), make (1, G)
        if len(log_fitness.shape) == 1:
            log_fitness = log_fitness.unsqueeze(0)
            log_abundance = log_abundance.unsqueeze(0)
        # shape (B, G)

        gt_mask = self.get_mask_at_round(
            round_idx,
            first_presence,
            len_presence
        )
        num_genotypes = first_presence.shape

        cum_time = self.cumulative_time[round_idx]
        t = torch.ones(num_genotypes, device = device) * cum_time
        time_since = t - first_times
        time_since = time_since.unsqueeze(0)

        masked_la = log_abundance[:, gt_mask]
        pred_la = masked_la + time_since[:, gt_mask] * log_fitness[:, gt_mask]
        pred_log_fq = pred_la - torch.logsumexp(pred_la, dim = -1, keepdim = True)

        if pred_log_fq.shape[0] == 1:
            return pred_log_fq[0]
        return pred_log_fq

    def get_mask_at_round(
        self, 
        round_idx: int,
        first_presence: torch.Tensor,
        len_presence: torch.Tensor,
    ) -> torch.Tensor:
        """ Mask = True for gt if gt is 'present' at round_idx """
        num_genotypes = first_presence.shape
        t = torch.ones(num_genotypes, device = device) * round_idx
        gt_mask = torch.all(torch.stack((
            (first_presence <= t),
            (t < first_presence + len_presence)
        )), axis = 0)
        assert torch.any(gt_mask), 'Mask is all false'
        return gt_mask

    def nll(
        self, 
        log_fitness: torch.Tensor,
        log_abundance: torch.Tensor
    ) -> torch.Tensor:
        """ Compute loss over entire training dataset, 
            over all rounds and genotypes
        """
        total_loss = torch.tensor([0.0], requires_grad = True, device = device)
        for i, round_idx in enumerate(range(self.num_rounds)):
            counts = self.get_counts(round_idx)
            mask_log_p1_pred = self.predict_masked_log_frequencies(
                log_fitness, 
                log_abundance,
                round_idx,
                self.first_presence,
                self.len_presence,
                self.first_times,
            )
            gt_mask = self.get_mask_at_round(
                round_idx,
                self.first_presence,
                self.len_presence,
            )
            mask_counts = counts[gt_mask]

            if self.loss_func_name == 'dirichlet_multinomial':
                nll = self.loss_func(
                    mask_logp_pred = mask_log_p1_pred, 
                    mask_counts = mask_counts, 
                    log_precision = self.log_dirmul_precision
                )
            else:
                nll = self.loss_func(
                    mask_logp_pred = mask_log_p1_pred, 
                    mask_counts = mask_counts, 
                )
            total_loss = total_loss + nll

        total_loss = total_loss / self.total_n_train_points
        return total_loss

    """
        Grouping
    """
    def batched_loglik_with_other(
        self,
        b_log_fitness: torch.Tensor,
        b_log_abundance: torch.Tensor,
        subset_genotype_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """ Groups genotypes into S items (len of subset_genotype_idxs) + 1
            for all others, then batch evaluates NLL.
            All input parameters have shape:
            - b_log_fitness: B x S'
            - b_log_abundance: B x S'
            
            Computes
            - first_presence
            - len_presence
            for 'others'. 
        """
        b_log_fitness = b_log_fitness.to(device)
        b_log_abundance = b_log_abundance.to(device)
        batch_size = b_log_fitness.shape[0]

        g_data = self.group_data(self.torch_data, subset_genotype_idxs)
        (num_timepoints, num_genotypes) = g_data.shape
        first_presence = torch.tensor(
            roundtracks.get_first_presence(tensor_to_np(g_data).T),
            device = device
        )
        len_presence = torch.tensor(
            roundtracks.get_len_presence(tensor_to_np(g_data).T),
            device = device
        )
        first_times = self.get_first_times(first_presence)

        total_loglik = torch.zeros((batch_size), requires_grad = True, device = device)
        for i, round_idx in enumerate(range(self.num_rounds)):
            counts = g_data[round_idx]

            mask_log_p1_pred = self.predict_masked_log_frequencies(
                b_log_fitness, 
                b_log_abundance,
                round_idx,
                first_presence,
                len_presence,
                first_times
            )
            gt_mask = self.get_mask_at_round(
                round_idx,
                first_presence,
                len_presence,
            )
            mask_counts = counts[gt_mask]

            nll = tasks.loss_dirmul_nll(
                mask_logp_pred = mask_log_p1_pred, 
                mask_counts = mask_counts, 
                log_precision = self.log_dirmul_precision
            )
            loglik = nll * -1
            # (B)

            total_loglik = total_loglik + loglik
        return total_loglik

    def group_data(
        self,
        torch_data: torch.Tensor,
        subset_genotype_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """ torch_data: T x G
            Groups data into S items (len of subset_genotype_idxs) + 1
            for all others (summed).
            Output: T x (S+1)
        """
        subset_data = torch_data[:, subset_genotype_idxs]
        # T x S

        other_idxs = torch.ones(
            torch_data.shape[-1], 
            device = device, 
            dtype = torch.bool
        )
        for i in subset_genotype_idxs:
            other_idxs[i] = False

        other_data = torch_data[:, other_idxs].sum(axis = 1).unsqueeze(-1)
        # T x 1

        return torch.concatenate([subset_data, other_data], axis = 1)

    def grouped_obs_fisher_info_matrix(
        self, 
        parameters: torch.Tensor,
        subset_genotype_idxs: torch.Tensor,
    ):
        """ Evaluate fisher info matrix at given log_fitness, log_abundance
            for three-grouped data.
            Parameters should be for {query, ref, others}, and be:
            - query_log_fitness
            - others_log_fitness
            - query_log_abundance
            - others_log_abundance
            We take ref_log_fitness and ref_log_abundance as zero.
        """
        fim = -1 * torch.autograd.functional.hessian(
            lambda _params: self.grouped_identifiable_loglik(
                _params, 
                subset_genotype_idxs
            ), 
            parameters,
        )
        return fim

    def grouped_identifiable_loglik(
        self, 
        params: torch.Tensor, 
        subset_genotype_idxs: torch.Tensor
    ) -> torch.Tensor:
        """ Parameters should be for {query, ref, others}, and be:
            - query_log_fitness
            - others_log_fitness
            - query_log_abundance
            - others_log_abundance
            We take ref_log_fitness and ref_log_abundance as zero.
        """
        b_log_fitness = torch.stack([params[0], torch.tensor(0.0), params[1]]).unsqueeze(0)
        b_log_abundance = torch.stack([params[2], torch.tensor(0.0), params[3]]).unsqueeze(0)
        loglik = self.batched_loglik_with_other(
            b_log_fitness,
            b_log_abundance,
            subset_genotype_idxs
        )
        return loglik

    """
        Training
    """
    def converged(self, losses: list[float]) -> bool:
        distance = 50
        return len(losses) > distance and losses[-distance] == losses[-1]

    def training_step(self) -> torch.Tensor:
        self.optimizer.zero_grad()
        loss = self.nll(self.log_fitness, self.log_abundance)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss)
        return loss

    def train(self) -> None:
        """ Optimizes simple fitness parameters. """
        loss = tensor_to_np(self.nll(self.log_fitness, self.log_abundance))
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
        self.log_abundance.detach()
        self.log_dirmul_precision.detach()
        self.get_counts.cache_clear()
        self.get_mask.cache_clear()
        del self.masks
        del self.torch_data
        del self.cumulative_time
        del self.total_n_train_points
        del self.log_fitness
        del self.log_abundance
        del self.log_dirmul_precision
        return