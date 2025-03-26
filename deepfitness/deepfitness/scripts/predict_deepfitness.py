""" 
    For development, run model in evaluation mode.
    Suggested workflow:
    - Provide --train_yaml arg, so that model hyperparameters can be
      read
    - 
"""
import os
from loguru import logger
import torch
import pandas as pd
import numpy as np

from deepfitness import networks
from deepfitness.predict import load_network_from_pl_ckpt
from deepfitness.predict import get_pred_logw_with_network
from hackerargs import args
from deepfitness.data import predict
from deepfitness.data import loaders
from deepfitness.genotype import dataflows


def main():
    torch.set_float32_matmul_precision('high')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    logger.info(f'Using dataflow: {args.get("ft.dataflow")}')
    dataflow = dataflows.get_dataflow(args.get('ft.dataflow'))

    tsngs_df = predict.tsngsdf_from_genotype_csv(
        df = pd.read_csv(args.get('csv')),
        genotype_col = args.get('genotype_col'),
        schema = dataflow.schema,
    )

    predict_dataloader = loaders.predict_dataloader(tsngs_df, dataflow)

    network_class = networks.get_network(args.get('net.network'))
    network = load_network_from_pl_ckpt(args.get('checkpoint'), network_class)

    pred_logw = get_pred_logw_with_network(network, predict_dataloader)
    tsngs_df.update_with_col('Log inferred fitness', pred_logw)
    tsngs_df.update_with_col('Inferred fitness', np.exp(pred_logw))

    for fake_col in args.get('__fake_columns'):
        tsngs_df.remove_col(fake_col)

    fn = os.path.basename(args.get('csv')).replace('.csv', '')
    tsngs_df.save_df_to_file(
        out_fn = args.get('output_folder') + f'/{fn}_fitnessinfer.csv'
    )

    logger.info('Done.')
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.predict_deepfitness
    """
    # load args from yaml, then update with cli
    args.parse_args('deepfitness/options/default_predict.yaml')
    
    # update pred args from train_yaml file, if exists
    if 'train_yaml' in args:
        args.update_with_train_yaml()

    mandatory_args = [
        'csv',
        'genotype_col',
        'output_folder',
        'checkpoint',
    ]
    for mandatory_arg in mandatory_args:
        assert mandatory_arg in args

    # create output folder
    output_folder = args.get('output_folder')
    os.makedirs(output_folder, exist_ok=True)

    # create log
    log_file = os.path.join(output_folder, f'log.log')
    logger.add(log_file)
    logger.info(f'Saving logs to {log_file=}')

    main()

    # save args to yaml
    saved_args_yaml = os.path.join(output_folder, 'args_predict.yaml')
    args.save_to_yaml(saved_args_yaml)

