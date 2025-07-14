"""
    Given two CSVs A and B with inferred fitness values,
    scales B inferred fitness to A's scale, by finding shared overlapping
    genotypes between A and B.
    Fails if no shared overlapping genotypes exist.
    Returns CSVs A and B merged together.

    This procedure is only valid if A and B are against the same target,
    interpreting fitness as "ability to survive to the next generation,
    competing against other genotypes, in a particular selective environment".
    For example, A and B could be two different macrocycle codon libraries,
    selected against the same target.
"""
import sys
from loguru import logger
import argparse, os
import pandas as pd, numpy as np

from hackerargs import args
from deepfitness import stats


def main():
    ref_df = pd.read_csv(args.get('reference_fitness_csv'))
    other_df = pd.read_csv(args.get('other_fitness_csv'))

    genotype_col = args.get('genotype_col')
    if stats.num_shared_genotypes(ref_df, other_df, genotype_col) == 0:
        logger.error(f'Found 0 shared genotypes -- cannot proceed.')
        sys.exit(1)

    merge_df = stats.merge_fitness_dfs(
        ref_df, 
        other_df, 
        genotype_col,
        args.get('fitness_col')
    )

    out_fn = args.get('output_csv')
    os.makedirs(os.path.dirname(out_fn), exist_ok = True)
    merge_df.to_csv(out_fn, index = False)
    logger.info(f'Saved results to {out_fn}')
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.merge_two_fitness_csvs
    """
    parser = argparse.ArgumentParser(
        description = """
            Merge two fitness CSVs A and B, by rescaling B's inferred fitness
            to A's scale, using shared overlapping genotypes.
            Fails if no shared overlapping genotypes exist.

            Fitness values in A and B are merged using evidence statistics,
            so both CSVs should have `compute_evidence_scores` computed on them
            already.

            This procedure is only valid if A and B are against the same target,
            interpreting fitness as "ability to survive to the next generation,
            competing against other genotypes, in a particular selective environment".
            For example, A and B could be two different macrocycle codon libraries,
            selected against the same target.
        """
    )
    parser.add_argument('--reference_fitness_csv', required = True)
    parser.add_argument('--other_fitness_csv', required = True)
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes, in both CSVs.'
    )
    parser.add_argument('--fitness_col', 
        default = 'Simple inferred fitness',
        help = 'Fitness col in both CSVs'
    )
    parser.add_argument('--output_csv', required = True)
    args.parse_args(parser)

    main()
