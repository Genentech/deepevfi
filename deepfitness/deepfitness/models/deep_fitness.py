from loguru import logger
from numpy.typing import NDArray

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from deepfitness import utils, tasks, simulate
from deepfitness.data.databatches import DataBatch, TSNGSwithFitnessDataBatch
from hackerargs import args


class DeepFitnessModel(pl.LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        loss_func_name: str,
        regularize_logw_col: str | None = None,
    ):
        """ Trains a network.

            Options
            -------
            regularize_logw_col
                If given, add regularizer loss to match output logw
                to target logw (e.g., simple fitness results).
        """
        super().__init__()
        self.network = network
        self.loss_func_name = loss_func_name
        self.loss_func = tasks.get_loss_function(loss_func_name)
        self.log_dirmul_precision = torch.nn.Parameter(torch.tensor([5.0]))

        self.regularize_logw_col = regularize_logw_col
        if self.regularize_logw_col:
            weight = args.setdefault('regularize_logw_weight', 1)
            self.regularize_logw_weight = weight

        self._epoch = 0
        self._train_losses = []
        self._train_batch_sizes = []

        self.accum_v2 = args.setdefault('accumulate_v2', False)
        if self.accum_v2:
            self.automatic_optimization = False

    """
        Setup
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = args.setdefault('lr', 1e-3), 
            weight_decay = args.setdefault('weight_decay', 1e-4)
        )
        scheduler = ReduceLROnPlateau(
            optimizer = optimizer, 
            patience = args.setdefault('reducelronplateau.patience', 10),
            factor = args.setdefault('reducelronplateau.factor', 0.3),
            verbose = True,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss_accumv2' if self.accum_v2 else 'train_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    def set_data_properties(self, n_datapoints: int, n_batches_per_epoch: int):
        self.n_datapoints = n_datapoints
        self.n_batches_per_epoch = n_batches_per_epoch
        return

    """
        Training loop and inference
    """
    def transfer_batch_to_device(
        self, 
        batch: DataBatch, 
        device: torch.device, 
        dataloader_idx: int
    ) -> DataBatch:
        """ Sends batch to device. Called automatically by
            pytorch lightning before training_step.
            For control flow, see:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks
        """
        if callable(getattr(batch, 'to_device', None)):
            batch_on_device = batch.to_device(device)
        else:
            batch_on_device = super().transfer_batch_to_device(
                batch, 
                device, 
                dataloader_idx
            )
        return batch_on_device

    def forward(self, batch: DataBatch) -> torch.Tensor:
        return self.network.forward(batch)

    def training_step(self, batch: DataBatch, batch_idx: int) -> torch.Tensor:
        """ batch is dict (output of collater) with values = tensors on device.
            Its mandatory fields are 'genotype_tensor', 'count', 'next_count'.
            It may contain additional fields too.

            Option: self.accum_v2 (bool)
                If True, accumulates gradients as sum of (loss / total n. train points)
                over each batch in epoch. Steps backwards once at end of epoch.
                Requires self.n_datapoints and self.n_batches_per_epoch.
                If 1 batch is used per round, this is identical to loss in
                simple evfi.

                Otherwise, returns loss / batch_size.
                If accumulate_grad_batches is set in pytorch lightning trainer,
                then grads are summed over batches.
        """
        pred_logw = self.network.forward(batch)
        sim_result = simulate.simulate_mask_log_fqs(
            log_W = pred_logw, 
            inp_counts = batch['count'], 
            steps = batch['steps_to_next_round'],
        )
        mask = sim_result.mask

        if self.loss_func_name == 'dirichlet_multinomial':
            loss_timeseries = self.loss_func(
                mask_logp_pred = sim_result.mask_log_p1_pred, 
                mask_counts = batch['next_count'][mask],
                log_precision = self.log_dirmul_precision,
            )
        else:
            loss_timeseries = self.loss_func(
                mask_logp_pred = sim_result.mask_log_p1_pred, 
                mask_counts = batch['next_count'][mask]
            )
        batch_size = len(sim_result.mask_r0_counts)
        loss = loss_timeseries
        self.__mylog('train_ts_loss', loss_timeseries / batch_size, batch_size)

        if self.regularize_logw_col and self.regularize_logw_weight != 0:
            loss_reg_logw = self.regularize_logw(pred_logw, batch)
            loss += loss_reg_logw
            self.__mylog('train_reglogw_loss', loss_reg_logw / batch_size, batch_size)

        self.__mylog('train_loss', loss / batch_size, batch_size)
        self._train_losses.append(loss)
        self._train_batch_sizes.append(batch_size)

        if self.accum_v2:
            self.__mylog('train_ts_loss_accumv2', loss_timeseries / self.n_datapoints, 1)
            self.__mylog('train_loss_accumv2', loss / self.n_datapoints, 1)

            opt = self.optimizers()
            self.manual_backward(loss / self.n_datapoints)
            if (batch_idx + 1) % self.n_batches_per_epoch == 0:
                opt.step()
                opt.zero_grad()
            return
        else:
            return loss / batch_size

    """
        Regularizers
    """
    def regularize_logw(
        self, 
        pred_logw: torch.Tensor, 
        batch: TSNGSwithFitnessDataBatch
    ) -> torch.Tensor:
        """ Regularization loss: MSE loss on predicted logw to target logw
            (e.g., from simple fitness inference).
        """
        loss_func = torch.nn.MSELoss(reduction = 'sum')
        target_logw = batch['target_logw']
        pred = torch.squeeze(pred_logw)
        return loss_func(pred, target_logw) * self.regularize_logw_weight

    """
        Logging
    """
    def __mylog(self, name: str, val: float, batch_size: int) -> None:
        self.log(
            name, 
            val, 
            prog_bar = True, 
            on_step = True, 
            on_epoch = True, 
            batch_size = batch_size
        )

    def on_train_epoch_end(self) -> None:
        """ losses is list of tensors """
        to_np = lambda x: utils.tensor_to_np(x, reduce_singleton = False)
        np_losses = to_np(torch.stack(self._train_losses))
        np_batch_sizes = self._train_batch_sizes
        # account for variable batch sizes
        epoch_loss = sum(np_losses) / sum(np_batch_sizes)

        logger.debug(f'Epoch {self._epoch} training loss: {epoch_loss}')
        # logger.debug(f'{np_losses}, {np_batch_sizes}')
        self._train_losses.clear()
        self._train_batch_sizes.clear()
        self._epoch += 1
        return

