"""
    Code related to tracks of rounds, & presence of variants in rounds
    in a track.

    A track is a sequential series of rounds that follow each other.
    Parallel tracks can be specified using before / after, which are lists
    of rounds, where before[i] -> after[i].
"""
from loguru import logger
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm
import sys
import scipy


"""
    Tracks
"""
def check_all_parallel(before: list[str], after: list[str]) -> bool:
    """ Check that before/after specifies parallel tracks; i.e., 
        all rounds connect to exactly one or zero other rounds. 
    """
    return len(set(before)) == len(before)


def find_parallel_tracks(
    before: list[str], 
    after: list[str]
) -> list[list[str]]:
    """ Returns a list of parallel tracks, which are lists of rounds. """
    assert check_all_parallel(before, after), 'Tracks are not parallel'
    has_next = lambda round: round in before
    get_next = lambda round: after[before.index(round)]

    seen = set()
    tracks = []
    for start_idx in range(len(before)):
        curr = before[start_idx]
        if curr in seen:
            continue

        # build parallel track
        track = [curr]
        while has_next(curr):
            seen.add(curr)
            curr = get_next(curr)
            track.append(curr)
        tracks.append(track)
    return tracks


def check_genotype_overlap_across_parallel_tracks(
    df: pd.DataFrame,
    genotype_col: str,
    rounds_before: list[str] | None,
    rounds_after: list[str] | None,
    verbose: bool = False,
) -> int:
    """ Check that tracks are parallel, and that genotypes in tracks
        connect all tracks (so fitness inference can be run on all tracks
        jointly).

        Returns num. connected components.
    """
    if rounds_before is None or rounds_after is None:
        return
    
    # check tracks are parallel
    assert check_all_parallel(rounds_before, rounds_after), 'Tracks are not parallel'

    # find tracks
    tracks = find_parallel_tracks(rounds_before, rounds_after)
    if len(tracks) == 1:
        return

    if verbose:
        logger.info(f'Found {len(tracks)} parallel tracks:')
        for track in tracks:
            logger.info(track)

    def get_gts_in_track(track: list[str]) -> set[str]:
        present_mask = (df[track] > 0).any(axis = 1)
        return set(df[genotype_col][present_mask])

    # create matrix of num shared genotypes on tracks
    n_tracks = len(tracks)
    n_pairs = n_tracks * (n_tracks + 1) / 2
    timer = tqdm(total = n_pairs)
    shared_mat = np.zeros((n_tracks, n_tracks))
    if verbose:
        logger.info(f'Computing matrix of shared genotypes between pairs ...')
    for idx in range(len(tracks)):
        gts_idx = get_gts_in_track(tracks[idx])
        for jdx in range(idx + 1, len(tracks)):
            gts_jdx = get_gts_in_track(tracks[jdx])

            has_overlap = len(gts_idx & gts_jdx) > 0
            shared_mat[idx][jdx] = int(has_overlap)
            shared_mat[jdx][idx] = int(has_overlap)
            if verbose:
                logger.info(f'Tracks {tracks[idx]},{tracks[jdx]}: {has_overlap=}')
            timer.update()
    timer.close()

    if verbose:
        logger.info('Searching for num. connected components in adj matrix ...')
    adj_mat = np.where(shared_mat > 0, 1, 0)
    n_comps, labels = scipy.sparse.csgraph.connected_components(
        adj_mat, 
        directed = False
    )
    if verbose:
        logger.info(f'Found {n_comps} distinct groups')
    return n_comps


"""
    Presence: 
    Given a G x T count matrix for a single track,
    compute time round indices when each variant is "present",
    for latent frequency model.
    
    first presence: Earliest round with non-zero count
    last presence: Earliest round with zero count AND all rounds after = 0
"""
def get_first_presence(count_mat: NDArray) -> NDArray:
    """ Earliest round with non-zero count.
        count matrix: G x T, on a single track
        Computes a G-len vector of round indices in [0, T-1].
        Variants never observed are assigned T.
    """
    mask = count_mat > 0
    (G, T) = mask.shape
    ds = []
    for round_idx in range(T):
        d = np.ones(G) * T
        d[mask[:, round_idx]] = round_idx
        ds.append(d)
    return np.min(np.stack(ds), axis = 0).astype(int)

def get_last_presence(count_mat: NDArray) -> NDArray:
    """ Earliest round with zero count AND all after = 0
        count matrix: G x T, on a single track
        Computes a G-len vector of round indices in [0, T-1].
        If variant is in final population, it is assigned T.
    """
    is_zero = np.array(count_mat == 0)
    (G, T) = count_mat.shape
    last_presence = np.ones(G) * T
    for round_idx in range(T - 1, 0, -1):
        all_zero_idx_to_end = np.all(is_zero[:, round_idx:], axis = 1)
        last_presence[all_zero_idx_to_end] = round_idx
    return last_presence

def get_len_presence(counts_mat: NDArray) -> NDArray:
    return get_last_presence(counts_mat) - get_first_presence(counts_mat) + 1
