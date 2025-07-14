"""
    Precompute a feature store {genotype: NDArray}
"""
from loguru import logger
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

from hackerargs import args
from deepfitness.genotype import dataflows, feature_stores


def main():
    csv = args.get('csv')
    genotype_col = args.get('genotype_col')
    dataflow_name = args.get('dataflow')

    df = pd.read_csv(csv)
    genotypes = list(df[genotype_col])

    dataflow = dataflows.get_dataflow(dataflow_name)
    
    fstore = feature_stores.FeatureStore(genotypes, dataflow.featurizer)
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.precompute_feature_store
    """
    parser = argparse.ArgumentParser(
        description = """
            Precompute a feature store
        """
    )
    parser.add_argument('--csv', required = True)
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes'
    )
    parser.add_argument('--dataflow', required = True, 
        help = 'Name of dataflow'
    )
    args.parse_args(parser)

    main()
