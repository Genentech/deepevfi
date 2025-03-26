"""
    Compute evidence scores on a CSV, given a fitness column
"""
from loguru import logger
import argparse
import pandas as pd
from pathlib import Path

from hackerargs import args
from deepfitness.genotype import schemas
from deepfitness.data import tsngs
from deepfitness import stats


def main():
    tsngs_df = tsngs.construct_tsngs_df(
        df = pd.read_csv(args.get('csv')),
        genotype_col = args.get('genotype_col'),
        round_cols = args.get('round_cols'),
        schema = schemas.AnyStringSchema(),
    )

    tpd = tsngs_df.update_with_target_pred_cols(args.get('log_fitness_col'))

    round_cols = [str(r) for r in args.get('round_cols')]
    df = stats.calc_evidence_stats(
        df = tsngs_df.df,
        round_cols = round_cols,
        pred_cols = tpd['Pred cols'],
        obs_cols = tpd['Target cols'],
    )

    logger.info(f'Saving to {args.get("output_csv")}')
    path = Path(args.get('output_csv'))
    path.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(args.get('output_csv'))
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.compute_evidence_scores
    """
    parser = argparse.ArgumentParser(
        description = """
            Computes evidence scores on a time-series CSV with a fitness column.
            Evidence score considers a mix of how many rounds have >=10 reads
            (or some threshold), and the fit between predicted and observed 
            readcounts.

            Parallelized using dask when possible.
        """
    )
    parser.add_argument('--csv', required = True)
    parser.add_argument('--log_fitness_col', required = True,
        help = 'Name of column containing log fitness values'
    )
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes'
    )
    parser.add_argument('--round_cols', required = True, 
        help = 'Name of columns for time-series rounds'
    )
    parser.add_argument('--output_csv', required = True)
    args.parse_args(parser)

    main()
