"""
    Estimates p(fitness of query > fitness of ref)
    query: all genotype variants of interest
    ref: top fq genotype in last round, or any other reference of interest.

    Groups data into query, reference, and all others (summed),
    then evaluates likelihood of log fitness values for query, ref, other
    over 2d grid.

    Compute cost is largely independent on num. genotypes, but does depend on
    num. timepoints. On TEAD3 toy datase with 7 timepoints, gets ~9 it/s.
"""
from loguru import logger
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray

import torch

from hackerargs import args
from deepfitness.genotype import schemas
from deepfitness.data import tsngs
from deepfitness.models.simple_fulldata import SimpleFitnessFullDataModel
from deepfitness import utils


def __research__prob_improve_without_other(
    model: SimpleFitnessFullDataModel,
    query_idx: int,
    ref_idx: int,
) -> float:
    """ Estimates p(fitness of query > fitness of ref).
        
        Uses just data from query, reference to compute p(c1, c2 | w1, w2)
        for counts "c" on a grid, over values of w1/w2. 
        Ignores all other data. 
        Not for production - research use only.
    """
    subset_idxs = torch.tensor([query_idx, ref_idx])
    budget = 10000

    log_fitness_ratios = np.linspace(-5, 5, num = budget)
    batched_logfit = np.stack((log_fitness_ratios, np.zeros(budget)), axis = -1)
    with torch.no_grad():
        lls = model.batched_fulldata_loglik(
            torch.tensor(batched_logfit), 
            subset_genotype_idxs = subset_idxs
        )

    ps = torch.exp(lls - torch.logsumexp(lls, 0))
    positive_idxs = [i for i, rat in enumerate(log_fitness_ratios) if rat > 0]
    return float(utils.tensor_to_np(sum(ps[positive_idxs])))


def prob_improve(
    model: SimpleFitnessFullDataModel,
    query_idx: int,
    ref_idx: int,
    log_dirmul_precision: torch.Tensor
) -> float:
    """ Estimates p(fitness of query > fitness of ref).
        Groups data into query, reference, and all others (summed),
        then evaluates likelihood of log fitness values for query, ref, other
        over 2d grid.
    """
    if query_idx == ref_idx:
        return 0.5
    subset_idxs = torch.tensor([query_idx, ref_idx])
    budget = 100

    lot = []
    for rat1 in np.linspace(-5, 5, num = budget):
        for rat2 in np.linspace(-5, 5, num = budget):
            lot.append([rat1, 0, rat2])
    batched_logfit = np.array(lot)

    with torch.no_grad():
        lls = model.batched_fulldata_dirmul_loglik_with_other(
            torch.tensor(batched_logfit), 
            subset_genotype_idxs = subset_idxs,
            log_dirmul_precision = log_dirmul_precision
        )

    ps = torch.exp(lls - torch.logsumexp(lls, 0))
    positive_idxs = [i for i, tpl in enumerate(lot) if tpl[0] > 0]
    return float(utils.tensor_to_np(sum(ps[positive_idxs])))


def main():
    df = pd.read_csv(args.get('csv'))
    fitness_col = args.get('fitness_col')

    tsngs_df = tsngs.construct_tsngs_df(
        df = pd.read_csv(args.get('csv')),
        genotype_col = args.get('genotype_col'),
        round_cols = args.setdefault('round_cols', None),
        rounds_before = args.setdefault('rounds_before', None),
        rounds_after = args.setdefault('rounds_after', None),
        schema = schemas.AnyStringSchema(),
    )
    log_fitness = np.log(tsngs_df.df[fitness_col])

    # get reference genotype
    if 'reference_genotype' in args and args['reference_genotype'] is not None:
        ref_gt = args.get('reference_genotype')
        logger.info(f'Using reference_genotype: {ref_gt}')
        ref_idx = tsngs_df.get_genotypes().index(ref_gt)
    else:
        if args.get('round_cols') is not None:
            last_round_col = str(args.get('round_cols')[-1])
        else:
            last_round_col = str(args.get('rounds_after')[-1])
        logger.info(f'Using max fq genotype in round {last_round_col } as reference')
        ref_idx = df[df[last_round_col] == max(df[last_round_col])].index[0]

    model = SimpleFitnessFullDataModel(
        tsngs_df, 
        'dirichlet_multinomial', 
        init_log_fitness = torch.tensor(log_fitness),
    )

    # obtain log_dirmul_precision
    logger.info(f'\n Inferring dirichlet-multinomial log precision ...')
    model.train()
    log_dirmul_precision = model.log_dirmul_precision.detach()
    logger.info(f'Found {log_dirmul_precision=}')

    # subset queries
    ref_log_fitness = log_fitness[ref_idx]
    logger.info(f'Found {ref_log_fitness=}')
    query_idxs = [i for i, lf in enumerate(log_fitness) if lf >= ref_log_fitness]
    logger.info(f'Found {len(query_idxs)} candidate variants')
    df['Is reference variant'] = False
    df.loc[ref_idx, 'Is reference variant'] = True

    # get probability of having higher fitness
    pbs = [prob_improve(model, q_idx, ref_idx, log_dirmul_precision)
           for q_idx in tqdm(query_idxs)]

    prob_improve_col = 'Probability fitness improves on fitness of reference'
    df[prob_improve_col] = np.nan
    df[prob_improve_col].iloc[query_idxs] = pbs

    df = df.sort_values(by = prob_improve_col, ascending = False)

    logger.info(f'Saving to {args.get("output_csv")}')
    path = Path(args.get('output_csv'))
    path.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(args.get('output_csv'))
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.{script_name}

        python -m deepfitness.scripts.compute_prob_improvement --csv example/simplefitness_output/test_ci.csv --fitness_col "Simple inferred fitness" --genotype_col HELMnolinker --round_cols 0,1,2,3,4,5,6 --output_csv example/simplefitness_output/test_ci_pimprove_new.csv
    """
    parser = argparse.ArgumentParser(
        description = """
            Estimates p(fitness of query > fitness of ref)
            query: all genotype variants
            ref: top fq genotype in last round, or any other reference

            Groups data into query, reference, and all others (summed),
            then evaluates likelihood of log fitness values for query, ref, other
            over 2d grid.

            Compute cost is largely independent on num. genotypes, but does depend on
            num. timepoints. On TEAD3 toy datase with 7 timepoints, gets ~9 it/s.
        """
    )
    parser.add_argument('--csv', required = True)
    parser.add_argument('--fitness_col', required = True,
        default = 'Simple inferred fitness',
        help = 'Name of column containing fitness values'
    )
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes'
    )
    parser.add_argument('--reference_genotype', default = None,
        help = 'Optional: Reference genotype. If not provided, uses max fq genotype in last round'
    )
    parser.add_argument('--round_cols', 
        help = 'Name of columns for time-series rounds'
    )
    parser.add_argument('--rounds_before', 
        help = 'Columns for time-series rounds; before[i] -> after[i]'
    )
    parser.add_argument('--rounds_after', 
        help = 'Columns for time-series rounds; before[i] -> after[i]'
    )
    parser.add_argument('--output_csv', required = True)
    args.parse_args(parser)

    main()
