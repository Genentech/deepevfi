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
from tqdm import tqdm
import scipy

from hackerargs import args
from deepfitness import stats


def main():
    csv_list = args.get('fitness_csvs')
    genotype_col = args.get('genotype_col')
    fitness_col = args.get('fitness_col')
    n_csvs = len(csv_list)

    logger.info(f'Reading {n_csvs} CSVs ...')
    fn_to_df = {fn: pd.read_csv(fn) for fn in tqdm(csv_list)}

    # precompute genotype sets
    logger.info(f'Computing genotype sets ...')
    gt_sets = {fn: set(fn_to_df[fn][genotype_col]) for fn in tqdm(csv_list)}

    # create matrix of num shared genotypes
    shared_mat = np.zeros((n_csvs, n_csvs))
    logger.info(f'Computing matrix of shared genotypes between pairs ...')
    timer = tqdm(total = int(n_csvs * (n_csvs - 1)/2))
    for i in range(n_csvs):
        for j in range(i + 1, n_csvs):
            ns = len(gt_sets[csv_list[i]] & gt_sets[csv_list[j]])
            shared_mat[i][j] = ns
            shared_mat[j][i] = ns
            timer.update()
    timer.close()

    # compute num. connected subgraphs
    adj_mat = np.where(shared_mat > 0, 1, 0)
    n_comps, labels = scipy.sparse.csgraph.connected_components(
        adj_mat, 
        directed = False
    )
    if n_comps == n_csvs:
        logger.info(f'Found zero shared genotypes -- stopping')
        sys.exit()
    logger.info(f'Found {n_comps} distinct groups')

    # by each group, merge
    group_ids = set(labels)
    for group_id in group_ids:
        idxs = [i for i, v in enumerate(labels) if v == group_id]

        # merge
        logger.info(f'Merging group {group_id} ...')
        logger.info(f'Starting with {csv_list[idxs[0]]} ... ')
        merge_df = fn_to_df[csv_list[idxs[0]]]
        i = idxs[0]
        seen_idxs = set([i])
        get_neighbors = lambda i: [j for j in idxs if shared_mat[i][j]]
        neighbors = get_neighbors(i)
        while neighbors:
            jdx = neighbors[0]
            neighbors = neighbors[1:]
            seen_idxs.add(jdx)

            logger.info(f'Merging {csv_list[jdx]} in ...')
            merge_df = stats.merge_fitness_dfs(
                merge_df,
                fn_to_df[csv_list[jdx]],
                genotype_col,
                fitness_col
            )
            
            # add neighbors
            for ndx in get_neighbors(jdx):
                if ndx not in neighbors and ndx not in seen_idxs:
                    neighbors.append(ndx)

        assert len(seen_idxs) == len(idxs)            

        # save
        output_folder = args.get('output_folder')
        out_fn = output_folder + f'merged_group_{group_id}.csv'
        os.makedirs(os.path.dirname(out_fn), exist_ok = True)
        merge_df.to_csv(out_fn, index = False)
        logger.info(f'Saved results to {out_fn}')

        with open(output_folder + f'group_{group_id}_input_files.txt', 'w') as f:
            f.write('\n'.join([csv_list[i] for i in idxs]))
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.merge_many_fitness_csvs

        python -m deepfitness.scripts.merge_many_fitness_csvs --fitness_csvs [/home/shenm19/prj/deepfitness/_datasets/dcp.prelim/simplefitness/hIL4R-L10/simple_jointrounds_evidence.csv,/home/shenm19/prj/deepfitness/_datasets/dcp.prelim/simplefitness/hIL4R-L10/simple_control_evidence.csv] --genotype_col "Peptide" --output_folder /home/shenm19/prj/deepfitness/_datasets/dcp.prelim/simplefitness/hIL4R-L10/test_merge/ 
    """
    parser = argparse.ArgumentParser(
        description = """
            Merge fitness CSVs into the smallest number of groups, based on
            how genotypes are shared among CSVs.
            Outputs 1 merged CSV per group.            

            Fitness values are merged using evidence statistics,
            so all CSVs should have `compute_evidence_scores` computed on them
            already.

            This procedure is only valid if all CSVs are against the same target,
            interpreting fitness as "ability to survive to the next generation,
            competing against other genotypes, in a particular selective environment".
        """
    )
    parser.add_argument('--fitness_csvs', required = True,
        help = 'List of fitness csv files (comma delimited by default)')
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes, in all CSVs.'
    )
    parser.add_argument('--fitness_col', 
        default = 'Simple inferred fitness',
        help = 'Fitness col in all CSVs'
    )
    parser.add_argument('--output_folder', required = True)
    args.parse_args(parser)

    main()
