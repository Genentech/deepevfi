from loguru import logger
import argparse
import numpy as np
import pandas as pd

from hackerargs import args


"""
    Checks
"""
def check_dynamics(fq_df: pd.DataFrame):
    logger.info(f'Checking dynamics ...')
    dfs = fq_df[fq_df.max(axis = 'columns') >= 0.001].copy()
    
    dfs['Violated dynamics'] = [bool(__count_violations(list(row)))
                                for idx, row in dfs.iterrows()]
    n_violated = sum(dfs['Violated dynamics'])
    logger.warning(f'Found {n_violated} among {len(dfs)} high max frequency trajectories')

    logger.info(f'Sum frequency by round that violate dynamics assumption')
    logger.info(dfs[dfs['Violated dynamics']].sum(axis = 'rows'))

    if n_violated / len(dfs) > 0.1:
        logger.error(
            """❌ Many high-frequency trajectories increase and decrease more
                than expected by fitness dynamics.
                Fundamental modeling assumptions may be violated, and
                modeling may give garbage results.
            """
        )
    return


def __count_violations(fqs: list[float]) -> int:
    """ Count violations in a list of frequencies.
        If frequencies follow fitness dynamics:
        - Frequency should never decrease, then increase
        - Frequency can increase, then decrease, up to one time. 
    """
    fq_threshold = 0.001
    rel_change_threshold = 0.01

    updown = lambda x1, x2: 'up' if x1 < x2 else 'down'
    uds = ['none'] + [updown(fqs[i], fqs[i + 1]) for i in range(len(fqs) - 1)]
    num_up_down = 0
    num_down_up = 0
    for i in range(1, len(uds)):
        prior, curr = uds[i - 1], uds[i]
        prior_fq, curr_fq = fqs[i - 1], fqs[i]

        if curr_fq > fq_threshold:
            if prior_fq > 0:
                rel_change = (curr_fq - prior_fq) / prior_fq
            else:
                rel_change = np.inf

            if rel_change >= rel_change_threshold:
                if prior == 'down' and curr == 'up':
                    num_down_up += 1
                if prior == 'up' and curr == 'down':
                    num_up_down += 1

    num_violations = num_down_up + max(num_up_down - 1, 0)
    return num_violations


def check_high_fq_genotypes_presence(fq_df: pd.DataFrame):
    """ Check that genotype frequencies > threshold at any round
        are present in the prior round.
    """
    logger.info(f'Checking that high frequency genotypes are present in prior round ...')
    threshold = 0.01
    skip0_df = fq_df[fq_df.columns[1:]]
    skiplast_df = fq_df[fq_df.columns[:-1]]
    masked_df = skiplast_df.where(skip0_df >= threshold)
    prior_fqs = masked_df.to_numpy().flatten()
    prior_fqs = prior_fqs[~np.isnan(prior_fqs)]
    if any(bool(fq == 0) for fq in prior_fqs):
        logger.error(
            """❌ Some genotypes with frequency above 0.01 are not present
                in prior round. 
                Time series data may be too jumpy and not smooth enough for
                good fitness modeling results.
                If possible, obtain additional time point data in between
                your current rounds (increase measurement density).
                Or, your data may have spliced different datasets together. 
            """
        )
    return


"""
    Main
"""
def main():
    csv = args.get('csv')
    genotype_col = args.get('genotype_col')
    rounds = [str(r) for r in args.get('round_cols')]

    logger.info(f"Loading data {csv=} with {genotype_col=}...")
    df = pd.read_csv(csv).set_index(genotype_col)
    df = df[rounds]

    logger.info(f'Filling na with zero ...')
    df = df.fillna(0)
    fq_df = df / df.sum(axis = 'rows')

    # Number of rounds
    logger.info(f'Number of time points = {len(rounds)}...')
    if len(rounds) < 2:
        logger.error('❌ Fitness inference requires at least 2 time points.')
    if len(rounds) == 2:
        logger.warning(
            f"""⚠️ Simple fitness inference with two time points is
                equivalent to using later_round/prior_round
                enrichment.
            """
        )

    # Entries are integer read counts with sufficiently high read depth
    logger.info(f'Checking total read count ...')
    total_reads = df.sum(axis = 'rows')
    logger.info(total_reads)
    if any(tot < 1000 for tot in total_reads):
        logger.warning(f'⚠️ Some time points have <1000 reads.')
    if any(tot <= 1 for tot in total_reads):
        logger.error(
            f"""❌ Some time points have 1 or fewer reads.
                Did you provide frequencies instead of counts?
                Our loss functions expect read counts.
            """
        )

    # Number of unique genotypes
    num_unique_gts = len(set(df.index))
    logger.info(f'Found {num_unique_gts} unique genotypes')
    if num_unique_gts < 100:
        logger.warning(f'⚠️ Num. unique genotypes < 100.')

    # Check max frequency by timepoint
    logger.info(f'Checking max frequency genotype by round ...')
    max_fqs = fq_df.max(axis = 'rows')
    logger.info(max_fqs)
    if all(max_fqs) < 0.01:
        logger.error(
            f"""⚠️❌ No round has >0.01 max genotype frequency. 
                Population may be too diverse with too weak selection 
                pressure for meaningful directed evolution, unless 
                total read counts are *very* high.
            """
        )

    # Check high frequency genotypes are present in prior rounds
    check_high_fq_genotypes_presence(fq_df)

    # Check dynamics
    check_dynamics(fq_df)

    logger.info('Done.')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """
            Sanity check a count table, checking conditions
            assumed by the mathematical model underlying fitness inference
        """
    )
    parser.add_argument('--csv', required = True)
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes'
    )
    parser.add_argument('--round_cols', required = True, 
        help = 'Name of columns for time-series rounds, interpreted sequentially'
    )
    args.parse_args(parser)
    main()
