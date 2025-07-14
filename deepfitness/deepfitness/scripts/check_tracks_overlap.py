"""
    Applies standard filters to csv: readcount, min fq, min max read
"""
from loguru import logger
import argparse
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
import sys

from hackerargs import args
from deepfitness.genotype import schemas
from deepfitness.data import tsngs
from deepfitness import roundtracks


def main():
    tsngs_df = tsngs.construct_tsngs_df(
        df = pd.read_csv(args.get('csv')),
        genotype_col = args.get('genotype_col'),
        rounds_before = args.get('rounds_before'),
        rounds_after = args.get('rounds_after'),
        schema = schemas.AnyStringSchema(),
        skip_filters = True,
    )
    df = tsngs_df.df

    gt_col = args.get('genotype_col')
    rounds_before = args.get('rounds_before')
    rounds_after = args.get('rounds_after')

    roundtracks.check_genotype_overlap_across_parallel_tracks(
        df, 
        gt_col, 
        rounds_before, 
        rounds_after, 
        verbose = True
    )
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.check_tracks_overlap

        python -m deepfitness.scripts.check_tracks_overlap --csv example/TEAD_subset500.csv --genotype_col HELMnolinker --rounds_before 0,1,2,4,5 --rounds_after 1,2,3,5,6
    """
    parser = argparse.ArgumentParser(
        description = """
            Checks if parallel round tracks have overlapping genotypes.
        """
    )
    parser.add_argument('--csv', required = True)
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes'
    )
    parser.add_argument('--rounds_before', 
        help = 'Columns for time-series rounds; before[i] -> after[i]'
    )
    parser.add_argument('--rounds_after', 
        help = 'Columns for time-series rounds; before[i] -> after[i]'
    )
    args.parse_args(parser)

    main()
