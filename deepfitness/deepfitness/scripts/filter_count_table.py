"""
    Applies standard filters to csv: readcount, min fq, min max read
"""
from loguru import logger
import argparse
import pandas as pd

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
        skip_filters = True,
    )

    logger.warning(f'Before filtering ...')
    logger.info(tsngs_df.describe())

    tsngs_df.filter_max_frequency(verbose = True)
    tsngs_df.filter_max_readcount(verbose = True)
    if args.setdefault('filt_consec', True):
        tsngs_df.filter_consecutive(verbose = True)
    tsngs_df.filter_duplicate_genotypes(verbose = True)

    logger.warning(f'After filtering ...')
    logger.info(tsngs_df.describe())

    tsngs_df.save_df_to_file(args.get('output_csv'))
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.filter_count_table

        python -m deepfitness.scripts.filter_count_table --csv /home/shenm19/prj/deepfitness/s3data/ab.phage.mgp130.aug2023/mgp130_mfilt.csv --genotype_col fv_heavy --round_cols [R0,R1,R2,R3,R4] --output_csv /home/shenm19/prj/deepfitness/s3data/ab.phage.mgp130.aug2023/test.csv
    """
    parser = argparse.ArgumentParser(
        description = """
            Applies standard data processing filters to count table.
        """
    )
    parser.add_argument('--csv', required = True)
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes'
    )
    parser.add_argument('--round_cols', 
        help = 'Columns for time-series rounds, interpreted sequentially'
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
