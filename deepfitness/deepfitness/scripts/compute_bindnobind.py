"""
    Impute a bind/no-bind label to a fitness-inferred CSV.

    For a round R, the bind/no-bind procedure finds an inferred
    fitness threshold using genotypes that maintain similar
    frequency from round R-1 to round R.
"""
from loguru import logger
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from hackerargs import args


def find_threshold(
    df: pd.DataFrame, 
    round_cols: list[str], 
    round_idx: int,
    fitness_col: str,
) -> float | None:
    """ Finds an inferred fitness threshold for a round.
        First, finds genotypes that maintain similar frequency from
        round R-1 to R, where frequency ignores new sequences in round R
        introduced by mutation.
        Then, takes the median fitness of these genotypes.
        Using median means fitness col can be log fitness or regular fitness.

        Intuitive interpretation: genotypes with fitness = threshold
        are able to maintain similar frequency in round R, e.g., they
        survived and did not deplete strongly during round R. 
        In particular, all found genotypes that actually maintain similar
        frequency have at least 2 rounds of information that support their
        activity, upon washing with the selection stringency at round R. 
        Using a fitness threshold enables calling binders for genotypes
        that occur in other rounds as well. 
    """
    assert 0 < round_idx < len(round_cols), f'Cannot use {round_idx=}'
    r0, r1 = round_cols[round_idx - 1], round_cols[round_idx]

    mask = np.where(df[r0] > 0, True, False)

    prev_counts = df[r0][mask]
    prev_fqs = prev_counts / np.sum(prev_counts)
    curr_counts = df[r1][mask]
    curr_fqs = curr_counts / np.sum(curr_counts)

    min_reads = 10
    has_min_reads = (prev_counts >= min_reads)
    logger.info(f'{has_min_reads.sum()} passed min reads')

    tol = 0.2
    enrich_ratio = (curr_fqs - prev_fqs) / prev_fqs
    has_similar_fq = (1 <= enrich_ratio) & (enrich_ratio <= 1 + tol)
    logger.info(f'{has_similar_fq.sum()} passed similar fq matching')

    crit = has_min_reads & has_similar_fq
    logger.info(f'{crit.sum()} passed both')
    if crit.sum() == 0:
        return None

    idxs = crit[crit].index
    fs = df[fitness_col][idxs]

    threshold = np.median(fs)
    return threshold


# main
def main():
    """ Adds additional columns to input dataframe:
        a bind/no-bind imputed label for each round, from the
        second round to the last round.
    """
    df = pd.read_csv(args.get('csv'))
    round_cols = [str(r) for r in args.get('round_cols')]
    fitness_col = args.get('fitness_col')

    for round_idx in range(1, len(round_cols)):
        round_col = round_cols[round_idx]
        threshold = find_threshold(df, round_cols, round_idx, fitness_col)
        logger.info(f'{round_col=}: fitness threshold = {threshold}')
        logger.info(f'Among {len(df)} gts, {sum(df[fitness_col] >= threshold)} pass')

        if threshold is not None:
            col = f'Bind/no-bind, {round_col}: fitness threshold {threshold}'
            df[col] = (df[fitness_col] >= threshold)

    out_file = args.get('output_csv')
    logger.info(f'Saving to {out_file}')
    path = Path(out_file)
    path.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(out_file)
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.compute_bindnobind
    """
    parser = argparse.ArgumentParser(
        description = """
            Compute bind/no-bind labels for a count table, given inferred
            fitness values. Computes one bind/no-bind label per round, which
            represents ability to survive binding challenge given the selection
            pressure and competing drugs in that round.
        """
    )
    parser.add_argument('--csv', required = True)
    parser.add_argument('--fitness_col', required = True,
        help = 'Name of column containing inferred fitness values'
    )
    parser.add_argument('--round_cols', required = True, 
        help = 'Name of columns for time-series rounds'
    )
    parser.add_argument('--output_csv', required = True)
    args.parse_args(parser)

    main()