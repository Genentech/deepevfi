"""
    Check genotype schema on CSV
"""
from tqdm import tqdm
from loguru import logger
import argparse
import pandas as pd
import random

from hackerargs import args
from deepfitness.genotype import dataflows
from deepfitness.data import tsngs
from deepfitness.data.tsngs import TimeSeriesNGSDataFrame


def main():
    dataflow_name = args.get('ft.dataflow')

    logger.info(f'Using dataflow: {dataflow_name}')
    dataflow = dataflows.get_dataflow(dataflow_name)

    schema = dataflow.schema()
    logger.info(f'Checking {schema=}')

    tsngs_df = tsngs.construct_tsngs_df(
        df = pd.read_csv(args.get('csv')),
        genotype_col = args.get('genotype_col'),
        round_cols = args.get('round_cols'),
        schema = schema,
    )

    # spot check
    spot_check = bool('random_spot_check_num' in args)

    genotypes = tsngs_df.get_genotypes()
    if spot_check:
        random.shuffle(genotypes)
        genotypes = genotypes[:args.get('random_spot_check_num')]

    invalid = []
    logger.info(f'Checking schema on {len(genotypes)} genotypes ...')
    for genotype in tqdm(genotypes):
        if not schema.is_valid(genotype):
            invalid.append(genotype)
            logger.warning(f'Invalid under schema: {genotype=}')

    logger.info(f'Found {len(invalid)} invalid out of {len(genotypes)}')
    frac_valid = 1 - (len(invalid) / len(genotypes))
    logger.info(f'Percent valid: {frac_valid:.2%}')

    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.check_genotype_schema

        python -m deepfitness.scripts.check_genotype_schema --csv /home/shenm19/prj/deepfitness/s3data/ab.phage.mgp130.aug2023/mgp130_mfilt.csv --genotype_col fv_heavy --round_cols R0,R1,R2,R3,R4 --ft.dataflow string_to_tensor
    """
    parser = argparse.ArgumentParser(
        description = """
            Checks that all genotypes in csv conform to schema.
        """
    )
    parser.add_argument('--csv', required = True)
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes'
    )
    parser.add_argument('--round_cols', required = True, 
        help = 'Name of columns for time-series rounds'
    )
    parser.add_argument('--ft.dataflow', required = True, 
        help = 'Dataflow containing a schema to check.'
    )
    parser.add_argument('--random_spot_check_num', type = int, 
        help = 'Number of genotypes to randomly spot check. Otherwise, all genotypes are checked.'
    )
    args.parse_args(parser)

    main()
