from loguru import logger

import torch
import pytorch_lightning as pl

from deepfitness import tasks, simulate
from deepfitness.data.databatches import TSNGSDataBatch
from hackerargs import args
from deepfitness import stats
from deepfitness import utils


class SimpleFitnessModel(pl.LightningModule):
    def __init__(self, num_genotypes: int, loss_func_name: str):
        """ Contains functions training_step, configure_optimizers,
            validation_step.
            Holds simple fitness parameters, and trains them, using
            pytorch lightning batch dataloaders.
            Batching means updating parameters on each pair of rounds,
            which might cause instability. 
        """
        super().__init__()
        self.automatic_optimization = False
        
        self.num_genotypes = num_genotypes
        self.loss_func_name = loss_func_name
        self.loss_func = tasks.get_loss_function(loss_func_name)

        init_val = torch.randn(
            self.num_genotypes, 
            dtype = torch.float32
        )
        self.log_fitness = torch.nn.Parameter(init_val)
        clamp = lambda grad: torch.clamp(grad, -10.0, 10.0)
        self.log_fitness.register_hook(clamp)

        self._epoch = 0
        self._train_losses = []
        self._train_batch_sizes = []

        self._detail_val_metrics = False
        if args.get('stages') in ['train_only', 'train_and_eval']:
            if args.setdefault('test_split_by_round', False) is True:
                test_rounds = args.get('test_round_idxs')
                if len(test_rounds) == 1:
                    self._detail_val_metrics = True
                    self._val_r0_counts = []
                    self._val_r1_counts = []
                    self._val_logp1_preds = []
                    logger.info(f'Logging detailed validation metrics')

    """
        Setup
    """
    def configure_optimizers(self):
        return torch.optim.LBFGS(
            [self.log_fitness],
            lr = args.setdefault('lr', 1e-3), 
            tolerance_grad = 1e-8,
            max_iter = 10,
            line_search_fn = 'strong_wolfe',
        )

    """
        Training loop and inference
    """
    def transfer_batch_to_device(
        self, 
        batch: TSNGSDataBatch, 
        device: torch.device, 
        dataloader_idx: int
    ) -> TSNGSDataBatch:
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

    def forward(self, batch: TSNGSDataBatch) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(self, batch):
        genotype_idxs = batch['genotype_idx']
        pred_logw = self.log_fitness[genotype_idxs]

        sim_result = simulate.simulate_mask_log_fqs(
            log_W = pred_logw, 
            inp_counts = batch['count'], 
            steps = batch['steps_to_next_round'],
        )
        mask = sim_result.mask
        loss = self.loss_func(
            mask_logp_pred = sim_result.mask_log_p1_pred, 
            mask_counts = batch['next_count'][mask]
        )
        batch_size = len(sim_result.mask_r0_counts)
        return loss, batch_size

    def training_step(self, batch: TSNGSDataBatch, batch_idx: int):
        """ batch is dict (output of collater) with values = tensors on device.
            Its mandatory fields are 'genotype_tensor', 'count', 'next_count'.
            It may contain additional fields too.
        """
        opt = self.optimizers()

        def closure():
            opt.zero_grad()
            loss, batch_size = self.compute_loss(batch)
            normalized_loss = loss / batch_size
            normalized_loss.backward()
            return normalized_loss

        opt.step(closure = closure)

        # log, after stepping
        loss, batch_size = self.compute_loss(batch)
        self.log(
            "train_loss", 
            loss, 
            prog_bar = True, 
            on_step = True, 
            on_epoch = True,
            batch_size = batch_size,
        )
        self._train_losses.append(loss)
        self._train_batch_sizes.append(batch_size)
        return loss / batch_size

    def validation_step(self, batch: TSNGSDataBatch, batch_idx: int):
        genotype_idxs = batch['genotype_idx']
        pred_logw = self.log_fitness[genotype_idxs]
        sim_result = simulate.simulate_mask_log_fqs(
            log_W = pred_logw, 
            inp_counts = batch['count'], 
            steps = batch['steps_to_next_round'],
        )
        mask = sim_result.mask
        loss = self.loss_func(
            mask_logp_pred = sim_result.mask_log_p1_pred, 
            mask_counts = batch['next_count'][mask]
        )
        batch_size = len(sim_result.mask_r0_counts)

        self.log(
            "val_loss", 
            loss, 
            prog_bar = True, 
            on_step = True, 
            on_epoch = True,
            batch_size = batch_size,
        )

        if self._detail_val_metrics:
            """ Save predictions over entire validation set,
                to compute detailed stats on val epoch end.
            """            
            self._val_r0_counts.append(sim_result.mask_r0_counts)
            self._val_r1_counts.append(batch['next_count'][mask])
            self._val_logp1_preds.append(sim_result.mask_log_p1_pred)

        return loss / batch_size

    """
        Logging
    """
    def on_train_epoch_end(self) -> None:
        """ losses is list of tensors """
        to_np = lambda x: utils.tensor_to_np(x, reduce_singleton = False)
        np_losses = to_np(torch.stack(self._train_losses))
        np_batch_sizes = self._train_batch_sizes
        # account for variable batch sizes
        epoch_loss = sum(np_losses) / sum(np_batch_sizes)

        logger.warning(f'Epoch {self._epoch} training loss: {epoch_loss}')
        logger.warning(f'{np_losses}, {np_batch_sizes}')
        self._train_losses.clear()
        self._train_batch_sizes.clear()
        self._epoch += 1
        return

    def on_validation_epoch_end(self) -> None:
        if self._detail_val_metrics:
            stats_dict = stats.compute_metrics(
                mask_r0_counts = torch.concat(self._val_r0_counts),
                mask_r1_counts = torch.concat(self._val_r1_counts),
                mask_log_p1_pred = torch.concat(self._val_logp1_preds),
            )
            logger.info('Validation metrics')
            for key, val in stats_dict.items():
                logger.info(f'  {key}: {val}')

            self._val_r0_counts.clear()
            self._val_r1_counts.clear()
            self._val_logp1_preds.clear()
        return    
