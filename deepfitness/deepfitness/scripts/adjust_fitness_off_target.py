"""
    Adjust fitness inferred using [target + instrument] rounds, using
    fitness inferred from [instrument only] / "off-target/control" rounds.
    This is typical for peptide directed evolution (Cunningham/Schroeder Labs).
"""
from loguru import logger
import argparse, os
import pandas as pd, numpy as np

from hackerargs import args
from deepfitness import stats


def get_scaling_ratio_with_readcount_threshold(
    merge_df: pd.DataFrame,
    joint_fitness_col: str,
    off_fitness_col: str,
    joint_round: str,
    off_round: str
) -> float:
    """ Use readcount-thresholded df to estimate scaling ratio
        using less noisy data.
    """
    threshold = int(args.get('read_count_threshold'))
    joint_crit = (merge_df[joint_round] >= threshold)
    off_crit = (merge_df[off_round] >= threshold)
    filt_df = merge_df[joint_crit & off_crit]
    print(f'Using readcount threshold {threshold}, filtered from {len(merge_df)} to {len(filt_df)} rows ...')

    joint_over_off_scale = stats.upper_bound_offtarget_scale(
        joint_fitness = np.array(filt_df[joint_fitness_col]),
        off_fitness = np.array(filt_df[off_fitness_col]),
    )
    return joint_over_off_scale


def main():
    joint_df = pd.read_csv(args.get('joint_fitness_csv'))
    off_df = pd.read_csv(args.get('off_fitness_csv'))

    # drop columns
    drop_cols = [
        'Simple log inferred fitness',
        'Deep log inferred fitness',
    ]
    joint_df = joint_df.drop(columns = drop_cols, errors = 'ignore')

    # rename fitness column
    genotype_col = args.get('genotype_col')
    fitness_col = args.get('fitness_col')
    joint_fitness_col = f'{fitness_col} - joint'
    on_fitness_col = f'{fitness_col} - on-target'
    off_fitness_col = f'{fitness_col} - off-target'
    joint_df = joint_df.rename(columns = {fitness_col: joint_fitness_col})
    off_df = off_df.rename(columns = {fitness_col: off_fitness_col})

    # drop to just genotype, rounds, fitness
    last_joint_round = args.get('last_joint_round')
    last_off_round = args.get('last_off_round')
    last_joint_round_renm = last_joint_round + '_joint'
    last_off_round_renm = last_off_round + '_off'
    joint_df = joint_df[[genotype_col, last_joint_round, joint_fitness_col]]
    off_df = off_df[[genotype_col, last_off_round, off_fitness_col]]

    joint_df = joint_df.rename(columns = {last_joint_round: last_joint_round_renm})
    off_df = off_df.rename(columns = {last_off_round: last_off_round_renm})

    # merge
    merge_df = joint_df.merge(off_df, on = genotype_col, how = 'outer')

    joint_over_off_scale = get_scaling_ratio_with_readcount_threshold(
        merge_df,
        joint_fitness_col,
        off_fitness_col,
        last_joint_round_renm,
        last_off_round_renm,
    )
    _, on_fitness, off_fitness = stats.estimate_ontarget_fitness_given_ratio(
        joint_fitness = np.array(merge_df[joint_fitness_col]),
        off_fitness = np.array(merge_df[off_fitness_col]),
        joint_over_off_scale = joint_over_off_scale
    )
    merge_df[on_fitness_col] = on_fitness
    merge_df[off_fitness_col] = off_fitness

    out_fn = args.get('output_csv')
    os.makedirs(os.path.dirname(out_fn), exist_ok = True)
    merge_df.to_csv(out_fn, index = False)
    logger.info(f'Saved results to {out_fn}')
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.adjust_fitness_off_target
    """
    parser = argparse.ArgumentParser(
        description = """
            Adjusts joint fitness [target + instrument] with off-target fitness
            [instrument only] to estimate on-target fitness [target only],
            using a mathematical upper bound on the unknown proportionality
            constants.
            Outputs CSV with joint/on/off fitness all on same scale.
        """
    )
    parser.add_argument('--joint_fitness_csv', required = True)
    parser.add_argument('--off_fitness_csv', required = True)
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes, in both CSVs.'
    )
    parser.add_argument('--fitness_col', 
        default = 'Simple inferred fitness',
        help = 'Fitness col in both CSVs'
    )
    parser.add_argument('--last_joint_round', required = True,
        help = """
            Round used for readcount thresholding to estimate scaling ratio using
            less noisy data.
            If available, we suggest using the latest round after joint selection,
            with same before round as the last_off_round. E.g., 3->4 and 3->4C.
        """
    )
    parser.add_argument('--last_off_round', required = True,
        help = """
            Round used for readcount thresholding to estimate scaling ratio using
            less noisy data.
            If available, we suggest using the latest round after joint selection,
            with same before round as the last_off_round. E.g., 3->4 and 3->4C.
        """
    )
    parser.add_argument('--read_count_threshold', default = 10,
        help = 'Read count threshold to estimate scaling ratio using less noisy data.'
    )
    parser.add_argument('--output_csv', required = True)
    args.parse_args(parser)

    main()
