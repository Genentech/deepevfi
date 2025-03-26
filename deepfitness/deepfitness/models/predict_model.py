import torch

import pytorch_lightning as pl

from deepfitness.data.databatches import DataBatch


class PredictModel(pl.LightningModule):
    def __init__(self, network: torch.nn.Module):
        """ Thin wrapper around torch.nn.Module, to run prediction using
            pytorch lightning.
        """
        super().__init__()
        self.network = network
    
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
        return torch.flatten(self.network.forward(batch))