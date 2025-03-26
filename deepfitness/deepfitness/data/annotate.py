from typing import Union
import numpy as np

import torch
import pytorch_lightning as pl

from deepfitness.data.splits import DataSplits
from deepfitness.data import loaders
from deepfitness.genotype.dataflows import DataFlow
from deepfitness.data.tsngs import TimeSeriesNGSDataFrame
from deepfitness.predict import get_pred_logw_with_network


def annotate_data_splits(
    data_splits: DataSplits,
    dataflow: DataFlow,
    network: torch.nn.Module,
    logw_col_name: str = 'Log inferred fitness',
    w_col_name: str = 'Inferred fitness',
) -> None:
    """
        Annotate all TimeSeriesNGSDataFrame in data_splits with
        inferred fitness, target/pred cols. Modifies tsngs_df in place.
    """
    for name, tsngs_df in data_splits.data_splits.items():
        if tsngs_df is not None:
            print(f'Annotating {name} ...')
            annotate_tsngsdf(
                tsngs_df, 
                dataflow, 
                network,
                logw_col_name = logw_col_name,
                w_col_name = w_col_name,
            )
    return


def annotate_tsngsdf(
    tsngs_df: TimeSeriesNGSDataFrame,
    dataflow: DataFlow,
    network: torch.nn.Module,
    logw_col_name: str = 'Log inferred fitness',
    w_col_name: str = 'Inferred fitness',
):
    """ Annotate TimeSeriesNGSDataFrame with inferred fitness,
        target/pred cols. Modifies tsngs_df in place.
    """
    pred_dataloader = loaders.predict_dataloader(tsngs_df, dataflow)

    # annotate predicted logw
    pred_logw = get_pred_logw_with_network(network, pred_dataloader)
    # pred_logw should match tsngs_df order exactly
    tsngs_df.update_with_col(logw_col_name, pred_logw)
    tsngs_df.update_with_col(w_col_name, np.exp(pred_logw))

    # annotate target/pred fq cols
    tsngs_df.update_with_target_pred_cols(logw_col_name)
    return

