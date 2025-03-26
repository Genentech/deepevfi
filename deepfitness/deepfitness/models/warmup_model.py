from loguru import logger

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

from deepfitness.data.databatches import WarmupBatch
from hackerargs import args
from deepfitness import utils


class WarmupModel(pl.LightningModule):
    def __init__(self, network: torch.nn.Module):
        """ Warm-up trains network, MSE loss on simple inferred fitness.
            Val/test dataloaders should be TSNGS with fitness, to evaluate
            held-out NLL.
        """
        super().__init__()
        self.network = network

        self.loss_func = torch.nn.MSELoss(reduction = 'sum')

        self._epoch = 0
        self._train_losses = []
        self._train_batch_sizes = []

    """
        Setup
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = args.setdefault('warmup.lr', 1e-3), 
            weight_decay = args.setdefault('warmup.weight_decay', 0.0)
        )
        scheduler = ReduceLROnPlateau(
            optimizer = optimizer, 
            patience = args.setdefault('warmup.reducelronplateau.patience', 10),
            factor = args.setdefault('warmup.reducelronplateau.factor', 0.3),
            verbose = True,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    """
        Training loop and inference
    """
    def transfer_batch_to_device(
        self, 
        batch: WarmupBatch, 
        device: torch.device, 
        dataloader_idx: int
    ) -> WarmupBatch:
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

    def forward(self, batch: WarmupBatch) -> torch.Tensor:
        return torch.flatten(self.network.forward(batch))

    def training_step(self, batch: WarmupBatch, batch_idx: int):
        """ batch is dict (output of collater) with values = tensors on device.
            Its mandatory fields are 'genotype_tensor', 'target_logw'
            It may contain additional fields too.
        """
        pred_logw = self.forward(batch)
        loss = self.loss_func(pred_logw, batch['target_logw'])
        batch_size = len(batch)

        self.log(
            "train_loss", 
            loss / batch_size, 
            prog_bar = True, 
            on_epoch = True,
            batch_size = batch_size,
        )
        self._train_losses.append(loss)
        self._train_batch_sizes.append(batch_size)
        return loss / batch_size

    """
        Logging
    """
    def on_train_epoch_end(self) -> None:
        """ losses is list of tensors """
        to_np = lambda x: utils.tensor_to_np(x)
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