""" 
    Trains a Simple Fitness model: one parameter per genotype.
    Optimization updates model once per entire dataset batch, which
    significantly improves training stability vs. minibatching over
    rounds and genotypes.
    Optimization uses L-BFGS which optimizes faster than gradient descent.
"""
import os
import argparse
from loguru import logger
import torch
import pandas as pd, numpy as np

from deepfitness.models.simple_fulldata import SimpleFitnessFullDataModel
from hackerargs import args
from deepfitness.genotype import schemas
from deepfitness.data import tsngs


def main():
    tsngs_df = tsngs.construct_tsngs_df(
        df = pd.read_csv(args.get('csv')),
        genotype_col = args.get('genotype_col'),
        round_cols = args.setdefault('round_cols', None),
        rounds_before = args.setdefault('rounds_before', None),
        rounds_after = args.setdefault('rounds_after', None),
        schema = schemas.AnyStringSchema(),
        skip_filters = args.setdefault('tsngs.skip_filters', False),
        verbose = True,
    )

    torch.manual_seed(int(args.setdefault('random_seed', 0)))

    loss_name = args.setdefault('simple.loss_func', 'multinomial')
    fitness_model = SimpleFitnessFullDataModel(
        tsngs_df = tsngs_df,
        loss_func_name = loss_name,
    )
    fitness_model.train()
    
    pred_logw = fitness_model.get_log_fitness()
    logw_col_name = 'Simple log inferred fitness'
    tsngs_df.update_with_col(logw_col_name, pred_logw)
    tsngs_df.update_with_target_pred_cols(logw_col_name)
    tsngs_df.update_with_col('Simple inferred fitness', np.exp(pred_logw))

    out_csv = args.setdefault(
        'output_csv', 
        args.get('output_folder') + '/simplefitness.csv'
    )
    tsngs_df.save_df_to_file(out_csv)

    logger.info('Done.')
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.train_simplefitness

        python -m deepfitness.scripts.train_simplefitness --csv example/TEAD_subset500.csv --genotype_col HELMnolinker --round_cols [0,1,2,3,4,5,6] --output_folder example/simplefitness_output/
    """
    # load args from yaml, then update with cli
    parser = argparse.ArgumentParser(description = """
        Core options: [-h] [--config] [--csv] [--genotype_col]
            [--round_cols OR --rounds_before, --rounds_after] [--output_folder] [--output_csv]

        Requires either --round_cols, or --rounds_before and --rounds_after;
        but not both.
        Runs simple fitness inference on csv using readcount data from rounds.
        Saves output files to output_folder.

        More command-line options are available beyond the core described above.
        To see them, try running the script. You can refer to the default
        yaml file read, or look at the saved_args.yaml file in the output
        folder.
    """)
    args.parse_args(parser, 'deepfitness/options/default_simple.yaml')

    assert 'csv' in args
    assert 'genotype_col' in args
    assert 'output_folder' in args

    args.setdefault('model_type', 'simple_fitness')

    # create output folder
    output_folder = args.get('output_folder')
    os.makedirs(output_folder, exist_ok = True)

    # create log
    log_file = os.path.join(output_folder, f'log.log')
    logger.add(log_file)
    logger.info(f'Saving logs to {log_file=}')

    main()

    # save args to yaml
    saved_args_yaml = os.path.join(output_folder, 'saved_args.yaml')
    args.save_to_yaml(saved_args_yaml)

