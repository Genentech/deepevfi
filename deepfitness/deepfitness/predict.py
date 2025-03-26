"""
    Code responsible for obtaining deep fitness predictions
"""
from typing import Union
from numpy.typing import NDArray

from deepfitness import utils
from hackerargs import args
from deepfitness.models.predict_model import PredictModel

import torch

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS


"""
    Checkpoint loading
"""
def plmodule_ckpt_to_torch_nn_module_state_dict(ckpt: dict) -> dict:
    """ Converts a pytorch lightning module checkpoint dict, from either
        DeepFitnessModel or WarmupModel
        to a state dict that can be loaded into a torch.nn.module.
        The pl module classes receive `network` in init and store it in self.
    """
    state_dict = ckpt['state_dict']
    # expect `network.` to be prefix in state_dict keys
    # update keys like network.embedder.weight -> embedder.weight
    network_keys = [k for k in list(state_dict) if 'network.' in k]
    assert len(network_keys) > 0, 'Error: Missing expected prefix `network.` for keys in state_dict.'
    new_state_dict = {key.replace('network.', ''): state_dict.pop(key)
                      for key in network_keys}
    return new_state_dict


def load_network_from_pl_ckpt(
    ckpt_path: str, 
    network_class: torch.nn.Module
) -> torch.nn.Module:
    """ Loads network params from a pl module checkpoint: expected to be
        either DeepFitnessModel or WarmupModel; these classes receive
        `network` in init and store it in self.
    """
    ckpt = torch.load(ckpt_path)
    state_dict = plmodule_ckpt_to_torch_nn_module_state_dict(ckpt)
    net = network_class()
    net.load_state_dict(state_dict)
    net.eval()
    return net


"""
    Getting logw predictions
"""
def get_pred_logw_with_network(
    net: torch.nn.Module,
    pred_dataloader: EVAL_DATALOADERS,
) -> NDArray:
    fitness_model = PredictModel(net)
    fitness_model.eval()
    with torch.no_grad():
        pred = get_pred_logw(fitness_model, pred_dataloader)
    return pred


def get_pred_logw(
    fitness_model: pl.LightningModule,
    pred_dataloader: EVAL_DATALOADERS,
    ckpt_path: None | str = None,
) -> NDArray:
    """ Uses model (or ckpt_path) to predict logw. """
    trainer = pl.Trainer(
        accelerator = args.get('accelerator'), 
        devices = 1,
        use_distributed_sampler = False,
    )
    if ckpt_path is not None:
        pred_logw = trainer.predict(
            fitness_model,
            dataloaders = pred_dataloader,
            ckpt_path = ckpt_path,
        )
    else:
        pred_logw = trainer.predict(
            fitness_model,
            dataloaders = pred_dataloader,
        )

    # pred_logw: (num. batches)-len list of tensors, shape (batch_size, 1)
    pred_logw = torch.flatten(torch.cat(pred_logw))
    # (dataset_size)
    pred_logw = utils.tensor_to_np(pred_logw)
    return pred_logw