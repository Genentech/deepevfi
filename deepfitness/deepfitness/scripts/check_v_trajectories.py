"""
    Probability of v-trajectory (frequency decreases then increases),
    indicating violation of assumed dynamics.
"""
from loguru import logger
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import itertools
import os
import pathlib

from scipy.stats import binom, beta
import scipy.integrate as integrate

import swifter
import multiprocessing as mp

from hackerargs import args

"""
    Violation of dynamics assumptions -- p(x,y,z)
"""
def moments(c, n):
    mean = c / n
    std = np.sqrt((mean) * (1 - mean) / n)
    return mean, std


def fast_accept_vtraj(
    c1, c2, c3, n1, n2, n3,
    z_score_threshold: float = 10.0,
) -> float | None:
    """ Quickly use Gaussian approximation to binomial proportion
        to say p(v-traj) = 1 when counts are very high, and posterior
        densities are very narrow. 

        p(gaussian 1 > gaussian 2) = p(gaussian1 - gaussian2 > 0);
        where g1-g2 is Gaussian with mean m1-m2, and std = sqrt(s1^2 + s2^2).
    """
    m1, s1 = moments(c1, n1)
    m2, s2 = moments(c2, n2)
    m3, s3 = moments(c3, n3)

    # prob of g1 > g2; g1 - g2 > 0:
    g1m2_std = np.sqrt(s1**2 + s2**2)
    z_score_of_mean_above_0_g1m2 = (m1 - m2) / g1m2_std

    g3m2_std = np.sqrt(s3**2 + s2**2)
    z_score_of_mean_above_0_g3m2 = (m3 - m2) / g3m2_std

    if z_score_of_mean_above_0_g1m2 > z_score_threshold:
        if z_score_of_mean_above_0_g3m2 > z_score_threshold:
            return 1.0
    
    return None


def prob_vtraj(c1: int, c2: int, c3: int, n1: int, n2: int, n3: int) -> float:
    """ Compute probability of a v-trajectory: p decreses then increases;
        p(p_1 > p_2 and p_2 < p_3), given counts c and total counts n,
        where c_i ~ Binomial(p_i, n_i).

        When counts are very high, adaptive quadrature can fail to
        find the area with high density (driven primarily by logpdf of c2,n2),
        and severely underestimate the true integral value.
        To address this, we evaluate the integral in the region:
            posterior mean +/- 200 * std, around c2/n2,
        which can be narrower than [0, 1].
    """
    def lik_vtraj(p_2: float) -> float:
        """ Computes likelihood of v-trajectory:
            p(p_2) * (1 - BetaCDF(p_1 = p_2)) * (1 - BetaCDF(p_3 = p_2))
        """
        p_p2 = beta.logpdf(p_2, c2 + 1, n2 - c2 + 1)
        sf1 = beta.sf(p_2, c1 + 1, n1 - c1 + 1)
        sf3 = beta.sf(p_2, c3 + 1, n3 - c3 + 1)
        return np.exp(p_p2) * sf1 * sf3

    result = fast_accept_vtraj(c1, c2, c3, n1, n2, n3)
    if result is not None:
        return result

    m2, s2 = moments(c2, n2)
    z_dist = 200
    lower = max(0, m2 - z_dist * s2)
    upper = min(1, m2 + z_dist * s2)
    p, err = integrate.quad(lik_vtraj, lower, upper)

    # print(c1, c2, c3, n1, n2, n3)
    # print(c1/n1, c2/n2, c3/n3)

    return np.clip(p, 0, 1)


def candidate_vtraj_triples(
    counts: list[int], 
    totals: list[int],
) -> list[tuple[int, int, int]]:
    """ Given N counts and totals, returns a set of triplet indices
        that are candidate v-trajectories: frequency decreases then increases.
    """
    fqs = [ct / tot for ct, tot in zip(counts, totals)]
    idxs = list(range(len(fqs)))
    triplets = itertools.combinations(idxs, 3)

    def is_cand(i, j, k) -> bool:
        return fqs[i] > fqs[j] and fqs[j] < fqs[k]
    return [trio for trio in triplets if is_cand(*trio)]


def candidate_rows(
    df: pd.DataFrame,
    round_cols: list[str],
):
    count_df = df[round_cols]
    total_counts = count_df.sum(axis = 'rows')
    totals = list(total_counts)

    cand_triples = lambda crow: candidate_vtraj_triples(list(crow), totals)

    cand_idxs = [idx for idx, row in tqdm(
                    count_df.iterrows(), 
                    total = len(count_df)
                 )
                 if len(cand_triples(row)) > 0]
    return cand_idxs


def parallel_process_row(counts: list[int], totals: list[int]):
    """ Single worker function for parallelization.
        Assumes that counts/totals provided has at least one candidate
        triple.
    """
    cand_trios = candidate_vtraj_triples(counts, totals)
    get_prob_vtraj = lambda trio: prob_vtraj(*[counts[i] for i in trio],
                                             *[totals[i] for i in trio])
    return max(get_prob_vtraj(trio) for trio in cand_trios)


def check_dynamics_violations_parallel(
    df: pd.DataFrame,
    round_cols: list[str],
    violations_col: str,
):
    count_df = df[round_cols]
    total_counts = count_df.sum(axis = 'rows')
    totals = list(total_counts)

    logger.info(f'Finding candidate rows ...')
    cand_rows = candidate_rows(df, round_cols)
    logger.info(f'Found {len(cand_rows)} candidate rows.')

    dfs = df.iloc[cand_rows]
    count_dfs = dfs[round_cols]

    # num_processes = mp.cpu_count()
    num_processes = 96
    logger.info(f'Computing in parallel with starmap with {num_processes=}...')

    inputs = [[list(row), list(total_counts)]
              for idx, row in count_dfs.iterrows()]
    with mp.Pool(num_processes) as pool:
        violations = pool.starmap(
            parallel_process_row,
            tqdm(inputs, total = len(inputs))
        )

    df.loc[cand_rows, violations_col] = violations
    return df


def check_dynamics_violations_serial(
    df: pd.DataFrame,
    round_cols: list[str],
    violations_col: str,
) -> pd.DataFrame:
    """ Annotate count df with new columns
    """
    count_df = df[round_cols]
    total_counts = count_df.sum(axis = 'rows')
    totals = list(total_counts)

    dd = defaultdict(list)
    for idx, row in tqdm(count_df.iterrows(), total = len(count_df)):
        counts = list(row)

        cand_trios = candidate_vtraj_triples(counts, totals)
        get_prob_vtraj = lambda trio: prob_vtraj(*[counts[i] for i in trio],
                                                 *[totals[i] for i in trio])

        if len(cand_trios) > 0:
            p_violation = max(get_prob_vtraj(trio) for trio in cand_trios)
        else:
            p_violation = np.nan
        dd[violations_col].append(p_violation)
    df[violations_col] = dd[violations_col]
    return df


"""
    Main
"""
def main():
    csv = args.get('csv')
    genotype_col = args.get('genotype_col')
    round_cols = [str(r) for r in args.get('round_cols')]

    logger.info(f"Loading data {csv=} with {genotype_col=}...")
    df = pd.read_csv(csv)

    df[round_cols] = df[round_cols].fillna(0)
    count_df = df[round_cols]
    total_counts = count_df.sum(axis = 'rows')
    for round_col in round_cols:
        df[f'{round_col} fq'] = df[round_col] / df[round_col].sum()

    # testing
    violations_col = 'Prob. dynamics violation'
    annot_df = check_dynamics_violations_parallel(df, round_cols, violations_col)

    # 
    threshold = 0.95
    
    vdf = df[df[violations_col] >= threshold]
    counts_violated = vdf[round_cols].sum(axis = 'rows')

    print('Read counts violating dynamics, by round:')
    print(counts_violated / total_counts)

    if max(counts_violated / total_counts) > 0.1:
        logger.error(
            """❌ More than 10\% of reads in a round violate dynamics
            """
        )

    output_csv = args.setdefault('output_csv', None)
    if output_csv:
        output_path = os.path.split(output_csv)[0]
        pathlib.Path(output_path).mkdir(parents = True, exist_ok = True)

        annot_df.to_csv(output_csv)
        logger.info(f'Wrote to {output_csv=}.')

    logger.info('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Sanity check a count table, checking conditions
            assumed by the mathematical model underlying fitness inference.
            Probability of v-trajectory (frequency decreases then increases),
            indicating violation of assumed dynamics.
        """
    )
    parser.add_argument('--csv', required = True)
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes'
    )
    parser.add_argument('--round_cols', required = True, 
        help = 'Name of columns for time-series rounds, interpreted sequentially'
    )
    parser.add_argument('--output_csv')
    args.parse_args(parser)
    main()
