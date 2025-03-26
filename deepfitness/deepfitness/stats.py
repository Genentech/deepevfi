""" 
    Statistics
"""
from loguru import logger
import pandas as pd, numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import copy

from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize_scalar

import torch
from torch.distributions.multinomial import Multinomial


from deepfitness import simulate, tasks
from hackerargs import args
from deepfitness.utils import tensor_to_np

import swifter
swifter.set_defaults(allow_dask_on_strings = True)


"""
    Rescale fitness & merge fitness CSVs
"""
def num_shared_genotypes(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    genotype_col: str
) -> int:
    return len(set(df1[genotype_col]) & set(df2[genotype_col]))


def merge_fitness_dfs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    genotype_col: str,
    fitness_col: str
) -> pd.DataFrame:
    """ Merges df2 into df1, returning merged df.
        Rescales df2 fitness to df1 scale, then combines fitness values,
        weighted by log(total readcount in campaign,
        only including rounds with >= 10 reads).
        Also merges evidence scores.
    """
    assert num_shared_genotypes(df1, df2, genotype_col) > 0
    assert not any(np.isnan(df1[fitness_col]))
    assert not any(np.isnan(df2[fitness_col]))

    error_col = 'Error (log2 sq)'
    assert error_col in df1.columns and error_col in df2.columns, \
        """ Input CSVs must have `compute_evidence_scores` run on them first """

    scaled_df2 = rescale_fitness(df1, df2, genotype_col, fitness_col)
    
    # merge df
    merge_df = df1.merge(
        scaled_df2, 
        on = genotype_col, 
        how = 'outer',
        suffixes = ('_ref', '_other')
    )

    # merge fitness using weights
    num_reads_threshold = args.setdefault('evidence_reads_threshold', 10)
    tr_thresh = f'Total reads on rounds with >= {num_reads_threshold} reads'

    merge_df[fitness_col] = merge_nan_values_weighted(
        merge_df[f'{fitness_col}_ref'],
        merge_df[f'{fitness_col}_other'],
        w1 = np.log(merge_df[f'{tr_thresh}_ref']),
        w2 = np.log(merge_df[f'{tr_thresh}_other']),
    )

    # merge evidence score
    merge_df = merge_evidence_scores(merge_df, '_ref', '_other')

    # drop columns
    merge_df = merge_df.drop(
        columns = [c for c in merge_df.columns if '_ref' in c or '_other' in c]
    )
    return merge_df


def rescale_fitness(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    genotype_col: str,
    fitness_col: str,
) -> pd.DataFrame | None:
    """ 
        Returns a new dataframe, copying df2, with fitness_col scaled to df1.
        Rescales using robust mean of fitness ratios among shared genotypes.
        If failure due to lack of shared genotypes, returns None.
    """
    num_shared = num_shared_genotypes(df1, df2, genotype_col)
    if num_shared == 0:
        return None
    elif num_shared < 5:
        logger.warning(f'Found {num_shared=}')
    else:
        logger.info(f'Found {num_shared=}')

    im_df = df1.merge(
        df2, 
        on = genotype_col, 
        how = 'inner',
        suffixes = ('_ref', '_other')
    )
    ratios = im_df[f'{fitness_col}_ref'] / im_df[f'{fitness_col}_other']

    # robust mean
    pct_to_drop = 0.025
    n_to_drop = int(len(ratios) * pct_to_drop) 
    ratios = sorted(ratios)[n_to_drop : len(ratios) - n_to_drop]

    scale_factor = np.mean(ratios)
    assert not np.isnan(scale_factor)

    # rescale
    scaled_df2 = df2.copy()
    scaled_df2[fitness_col] *= scale_factor
    return scaled_df2


def merge_nan_values_weighted(
    f1: list[float],
    f2: list[float],
    w1: list[float],
    w2: list[float],
) -> list[float]:
    """ Merge values. If either is NaN, use non-NaN value.
        w1 and w2 are normalized to sum to 1.
    """
    def merge_single(fit1, fit2, we1, we2) -> float:
        if np.isnan(fit1):
            return fit2
        elif np.isnan(fit2):
            return fit1
        return (we1 * fit1 + we2 * fit2) / (we1 + we2)
    
    return [merge_single(v1, v2, x1, x2) for v1, v2, x1, x2 in zip(f1, f2, w1, w2)]


"""
    Adjust joint fitness and off-target fitness
"""
def estimate_ontarget_fitness(
    joint_fitness: NDArray, 
    off_fitness: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """ For joint_fitness on [target + instrument] and off_fitness from
        [instrument only], infer on-target fitness for [target only].
        Joint_fitness has unknown scaling constant c1, and off_fitness
        has unknown scaling constant c2.
        Uses a mathematical upper bound on c1/c2 to convert off_fitness onto
        the scale of joint_fitness, to estimate on_target fitness (on the
        scale of joint_fitness).
        Assumes that joint_fitness = on_fitness + off_fitness.

        Returns
        -------
        joint_fitness
        on_fitness (estimated), on scale of joint_fitness
        off_fitness: rescaled to joint_fitness scale
    """
    return estimate_ontarget_fitness_given_ratio(
        joint_fitness,
        off_fitness, 
        upper_bound_offtarget_scale(joint_fitness, off_fitness)
    )


def estimate_ontarget_fitness_given_ratio(
    joint_fitness: NDArray, 
    off_fitness: NDArray,
    joint_over_off_scale: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """ Assumes that joint_fitness = on_fitness + off_fitness.
        Returns
        -------
        joint_fitness
        on_fitness (estimated), on scale of joint_fitness
        off_fitness: rescaled to joint_fitness scale
    """
    rescaled_off = joint_over_off_scale * off_fitness
    on_fitness = joint_fitness - rescaled_off
    return joint_fitness, on_fitness, rescaled_off


def upper_bound_offtarget_scale(
    joint_fitness: NDArray,
    off_fitness: NDArray,
) -> float:
    """ Computes upper bound on scale ratio c1/c2 for joint_fitness
        (with unknown scale c1) and off_fitness (unknown scale c2).
        joint_fitness and off_fitness should be aligned in same genotype order.

        Input
        -----
        joint_fitness: G x 1, floats
        off_fitness: G x 1, floats
    """
    ratios = joint_fitness / off_fitness
    if all(np.isnan(ratios)):
        logger.error(f'Failed to adjust joint fitness with off-target fitness; no overlap between genotypes found.')
        exit(1)
    return np.nanmin(ratios)


"""
    Time-series statistics
"""
def compute_metrics(
    mask_r0_counts: torch.Tensor,
    mask_r1_counts: torch.Tensor,
    mask_log_p1_pred: torch.Tensor,
) -> dict[str, float | int]:
    """ Compute evaluation metrics.
        Input tensors should be masked, meaning that r0_count > 0 at every
        entry, and all tensors have the same length.

        Enrichment stats are computed on double-masked data, requiring
        r0_count > 0 *and* r1_count > 0.
    """
    assert torch.count_nonzero(mask_r0_counts) == len(mask_r0_counts)
    assert len(mask_r0_counts) == len(mask_r1_counts) == len(mask_log_p1_pred)

    get_fq = lambda vec: tensor_to_np(vec / vec.sum())
    r0_fqs = get_fq(mask_r0_counts)
    r1_fqs = get_fq(mask_r1_counts)
    pred_fqs = tensor_to_np(torch.exp(mask_log_p1_pred))

    obs_enrich = r1_fqs / r0_fqs
    pred_enrich = pred_fqs / r0_fqs

    noise_ws = noise_weights(np.array(mask_r0_counts), np.array(mask_r1_counts))
    threshold10 = (mask_r0_counts > 10)

    # Compute log enrichments on double masked data: R0 > 0 and R1 > 0
    doublemask = (r1_fqs > 0)
    log_obs_enrich = np.log(r1_fqs[doublemask] / r0_fqs[doublemask])
    log_pred_enrich = np.log(pred_fqs[doublemask] / r0_fqs[doublemask])

    nll = tensor_to_np(tasks.loss_multinomial_nll(mask_log_p1_pred, mask_r1_counts))
    stats = {
        'Multinomial NLL': nll,
        'Multinomial NLL, divided by num gts': nll / len(mask_r1_counts),
        'Noise-weighted enrichment pearsonr': weighted_pearsonr(
            pred_enrich, obs_enrich, noise_ws
        ),
        'Enrichment pearsonr': pearsonr(pred_enrich, obs_enrich)[0],
        'Enrichment spearmanr': spearmanr(pred_enrich, obs_enrich)[0],
        'Enrichment pearsonr, min 10 input reads': pearsonr(
            pred_enrich[threshold10], 
            obs_enrich[threshold10])[0],
        'Enrichment spearmanr, min 10 input reads': spearmanr(
            pred_enrich[threshold10], 
            obs_enrich[threshold10]
        )[0],
        'Log enrichment pearsonr': pearsonr(log_pred_enrich, log_obs_enrich)[0],
        'Log enrichment spearmanr': spearmanr(
            log_pred_enrich, 
            log_obs_enrich
        )[0],
        'Frequency pearsonr': pearsonr(pred_fqs, r1_fqs)[0],
        'Frequency spearmanr': spearmanr(pred_fqs, r1_fqs)[0],
        'N, single mask': len(mask_r1_counts),
        'N, double mask (for log enrichment)': int(sum(doublemask)),
    }
    return stats


def noise_weights(r0_counts: NDArray, r1_counts: NDArray) -> float:
    inv_w = (1 / (r0_counts + 0.5)) + (1 / (r1_counts + 0.5))
    return 1 / inv_w


def weighted_pearsonr(x: NDArray, y: NDArray, w: NDArray) -> float:
    def m(x, w):
        return np.average(x, weights = w)

    def cov(x, y, w):
        return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


"""
    Evidence heuristic score, using pred/target cols in a df.
"""
def calc_evidence_stats(
    df: pd.DataFrame,
    round_cols: list[str],
    pred_cols: list[str],
    obs_cols: list[str],
) -> pd.DataFrame:
    """ Compute evidence for each inferred fitness value, using
        error in fit, number of timepoints with useful number of reads.

        Evidence score should be comparable across experiments,
        so they cannot be functions of multiple genotypes
        (e.g., no normalization allowed).

        Updates df with new columns:
        - Evidence (weighted)
        - Error (log2 sq)
        - Num. timepoints >= {num_reads_threshold} reads
        - Error, max over rounds (log2 sq)
        - Total reads on rounds with >= {num_reads_threshold} reads
    """
    logger.info(f'Computing evidence stats for inferred fitness values ...')
    num_reads_threshold = args.setdefault('evidence_reads_threshold', 10)

    df = df.reset_index(drop = True)

    error_col = 'Error (log2 sq)'
    logger.info(f'Annotating {error_col} ...')
    df[error_col] = df.swifter.apply(
        lambda row: __calc_log2_sq_error_row(row, pred_cols, obs_cols),
        axis = 1,
    )

    num_useful_timepoints_col = f'Num. timepoints >= {num_reads_threshold} reads'
    logger.info(f'Annotating {num_useful_timepoints_col} ...')
    df[num_useful_timepoints_col] = df.swifter.apply(
        lambda row: sum(row[round_cols] >= num_reads_threshold),
        axis = 1,
    )

    # annotate other stats
    max_error_col = 'Error, max over rounds (log2 sq)'
    logger.info(f'Annotating {max_error_col} ...')
    df[max_error_col] = df.swifter.apply(
        lambda row: __calc_log2_sq_error_row(row, pred_cols, obs_cols,
                                             take_max = True),
        axis = 1,
    )

    tr_thresh = f'Total reads on rounds with >= {num_reads_threshold} reads'
    logger.info(f'Annotating {tr_thresh} ...')
    df[tr_thresh] = df.swifter.apply(
        lambda row: sum(row[r] for r in round_cols
                        if row[r] >= num_reads_threshold),
        axis = 1,
    )

    num_train_rounds = f'Num training rounds'
    logger.info(f'Annotating {num_train_rounds} ...')
    df[num_train_rounds] = df.swifter.apply(
        lambda row: sum(row[obs_cols] > 0),
        axis = 1,
    )

    # compute weighted evidence
    evidence_col = 'Evidence (weighted)'
    logger.info(f'Annotating {evidence_col} ...')
    df[evidence_col] = -1 * df[error_col] + \
        2 * np.log2(df[num_useful_timepoints_col])
    return df


def __calc_log2_sq_error_row(
    row: pd.Series, 
    pred_cols: list[str], 
    obs_cols: list[str],
    take_max: bool = False,
) -> float:
    preds = np.array(row[pred_cols], dtype = float)
    targets = np.array(row[obs_cols], dtype = float)

    errors = []
    for pred, target in zip(preds, targets):
        if target > 0:
            if pred > 0:
                error = (np.log2(target) - np.log2(pred))**2
            else:
                error = 100
            errors.append(error)
    if take_max:
        return max(errors)
    else:
        return np.mean(errors)


def merge_evidence_scores(
    mdf: pd.DataFrame, 
    ref_suffix: str,
    other_suffix: str,
) -> pd.DataFrame:
    """ Merges weighted evidence score. Builds on the observation that
        when fitness CSVs can be merged, fitness inference could have been
        run on the merged input CSVs as well. We merge evidence scores to be
        consistent with the alternate approach as if we had
        run fitness inference on the merged input CSV instead.
    """
    rs, os = ref_suffix, other_suffix
    num_reads_threshold = args.setdefault('evidence_reads_threshold', 10)

    # columns to be updated
    error_col = 'Error (log2 sq)'
    num_useful_timepoints_col = f'Num. timepoints >= {num_reads_threshold} reads'
    max_error_col = 'Error, max over rounds (log2 sq)'
    tr_thresh = f'Total reads on rounds with >= {num_reads_threshold} reads'
    num_train_rounds = f'Num training rounds'
    evidence_col = 'Evidence (weighted)'

    # Update all columns
    _sum = lambda col: mdf[f'{col}{rs}'].fillna(0) + mdf[f'{col}{os}'].fillna(0)
    _max = lambda col: np.maximum(
        mdf[f'{col}{rs}'].fillna(0), 
        mdf[f'{col}{os}'].fillna(0)
    )

    mdf[num_train_rounds] = _sum(num_train_rounds)
    mdf[tr_thresh] = _sum(tr_thresh)
    mdf[num_useful_timepoints_col] = _sum(num_useful_timepoints_col)
    mdf[max_error_col] = _max(max_error_col)

    mdf[error_col] = merge_nan_values_weighted(
        mdf[f'{error_col}{rs}'],
        mdf[f'{error_col}{os}'],
        w1 = mdf[f'{num_train_rounds}{rs}'],
        w2 = mdf[f'{num_train_rounds}{os}'],
    )
    mdf[evidence_col] = -1 * mdf[error_col] + \
        2 * np.log2(mdf[num_useful_timepoints_col])
    return mdf

