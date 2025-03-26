from loguru import logger
import pandas as pd, numpy as np
import random, functools, os, copy
from collections import Counter
from typing import Tuple
from numpy.typing import NDArray

import torch

from deepfitness import simulate, tasks, roundtracks
from hackerargs import args
from deepfitness.genotype.schemas import GenotypeStrSchema, AnyStringSchema
from deepfitness.utils import tensor_to_np, fill_masked_tensor


class TimeSeriesNGSDataFrame:
    def __init__(
        self,
        df: pd.DataFrame, 
        genotype_col: str,
        rounds_before: list[str],
        rounds_after: list[str],
        schema: GenotypeStrSchema = AnyStringSchema,
        steps_per_round: list[float] | None = None,
        skip_filters: bool = True,
        reset_index_save: bool = False,
        verbose: bool = False,
    ):
        """ A lightweight wrapper around a pd.DataFrame describing a 
            time series NGS dataset.

            Input
            -----
            df: pd.DataFrame
                A DataFrame of read counts of genotypes over time points.
            genotype_col: str
                Name of column containing genotypes.
            rounds_before / rounds_after: list[str]
                Columns to define count matrix. For each index i,
                rounds_before[i] -> rounds_after[i].
            schema: GenotypeStrSchema
                The schema that genotype_col strings are expected to adhere to.
            -------------------
            steps_per_round: list[float] or None
                The i-th element is the number of generations from
                round before[i] -> after[i].
                Can be used to handle varying selection stringency.
                If None, attempts to read from args. Defaults to 1 for each round.
            skip_filters: bool
                If True, do not filter genotypes lacking consecutive non-zero
                reads (fitness cannot be inferred for these). 
            reset_index_save: bool
                Used for train/val/test splitting on genotypes. If True,
                resets indices, and saves to `original_index`.
            verbose: bool
                Logs at higher verbosity level.
        """
        self.df = df
        self.genotype_col = genotype_col
        self.rounds_before = [str(r) for r in rounds_before]
        self.rounds_after = [str(r) for r in rounds_after]
        self.schema = schema
        self.verbose = verbose
        self.log('Setting up TimeSeriesNGSDataFrame ...')

        assert len(self.rounds_before) == len(self.rounds_after), \
            f'{self.rounds_before=}, {self.rounds_after=}'
        # build round_cols from unique rounds in before/after
        self.round_cols = copy.copy(self.rounds_before)
        for r in self.rounds_after:
            if r not in self.round_cols:
                self.round_cols.append(r)
        self.non_round_cols = [c for c in df.columns if c not in self.round_cols]
        self.log(f'Input dataframe has size {len(self.df)}')
        self.log(f'Using {genotype_col=}')
        self.log(f'Using {self.round_cols=}, {self.rounds_before=}, {self.rounds_after=}')

        # setup
        df.loc[:, self.round_cols] = df.loc[:, self.round_cols].fillna(0)
        self.df = df

        if reset_index_save:
            self.df = df.rename_axis('original_index').reset_index()

        if not skip_filters:
            self.filter_max_frequency()
            self.filter_max_readcount()
            self.filter_consecutive()
            self.filter_duplicate_genotypes()
            self.df = self.df.reset_index(drop = True)
            self.spotcheck_schema()

        if steps_per_round is not None:
            self.steps_per_round = steps_per_round
        elif steps_per_round is None:
            if 'steps_per_round' not in args:
                self.steps_per_round = [1.0] * len(self.rounds_after)
            else:
                var_times = [float(t) for t in args.get('steps_per_round')]
                assert len(var_times) == len(self.rounds_after)
                self.steps_per_round = var_times
        self.log(f'{self.steps_per_round=}')

        self.log_num_valid_training_samples_per_round()
        self.log('... Finished setting up TimeSeriesNGSDataset.')

    def log(self, message: str) -> None:
        """ Log based on verbosity level. """
        if self.verbose:
            logger.info(message)
        return

    """
        Filters
    """
    def filter_duplicate_genotypes(self, verbose: bool = False) -> None:
        """ Overwrites self.df """
        if verbose:
            logger.info(f'Filtering duplicate genotypes ...')
        orig_len = len(self.df)
        self.df = self.df[~self.df[self.genotype_col].duplicated()]
        if len(self.df) != orig_len:
            logger.warning(f'WARNING - filtered out duplicate genotypes! We recommend pre-filtering them out.')
        return

    def filter_max_frequency(self, verbose: bool = False) -> None:
        """ Overwrites self.df.
            Filters rows in df whose max frequency is below a threshold, such
            as the PCR error rate threshold.
        """
        threshold = args.setdefault('filter_max_frequency_threshold', 1e-6)
        if verbose:
            logger.info(
                f'Filtering rows with max frequency over time < {threshold} ...'
            )
        
        # annotate df with frequencies, before filtering
        fq_cols = []
        for round_col in self.round_cols:
            name = f'{round_col} frequency, before filtering'
            self.df[name] = self.df[round_col] / self.df[round_col].sum()
            fq_cols.append(name)

        self.df = self.df[self.df[fq_cols].max(axis='columns') >= threshold]
        if verbose:
            logger.info(f'Reduced to {len(self.df)} rows.')
        return

    def filter_max_readcount(self, verbose: bool = False) -> None:
        """ Overwrites self.df
            Filters rows in df whose max read count is below a threshold.
        """
        threshold = args.setdefault('filter_max_readcount_threshold', 10)
        if verbose:
            logger.info(
                f'Filtering rows with max readcount over time < {threshold} ...'
            )
        crit = (self.df[self.round_cols].max(axis='columns') >= threshold)
        self.df = self.df[crit]
        if verbose:
            logger.info(f'Reduced to {len(self.df)} rows.')
        return

    def filter_consecutive(self, verbose: bool = False) -> None:
        """ Overwrites self.df.
            Filters rows in df that lack two consecutive timepoints with
            non-zero read count.
        """
        if verbose:
            logger.info(
                f'Filtering genotypes lacking two consecutive' + 
                'timepoints with non-zero read count...'
            )
        has_consec = np.zeros(len(self.df), dtype = bool)
        for r0, r1 in zip(self.rounds_before, self.rounds_after):
            has_consec |= (self.df[r0] > 0) & (self.df[r1] > 0)
        orig_len = len(self.df)
        self.df = self.df[has_consec]
        if len(self.df) != orig_len:
            logger.warning(f'WARNING - filtered out genotypes lacking consecutive rounds with non-zero reads; fitness cannot be inferred for them.  We recommend pre-filtering them out.')
            logger.warning(f'New dataframe size: {len(self.df)}')
        return

    """
        Checks
    """
    def spotcheck_schema(self, num: int = 1000) -> None:
        """ Random spot check genotype str schema on df. """
        gts = self.get_genotypes()
        random.shuffle(gts)
        gts = gts[:num]

        self.log(f'Spot checking schema on {num} random genotypes ...')
        for gt in gts:
            if not self.schema.is_valid(gt):
                logger.error(f'{gt} failed schema check')
                exit(1)
        return

    """
        Properties
    """
    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    def get_num_genotypes(self) -> int:
        return len(self.df)

    def get_num_rounds(self) -> int:
        return len(self.round_cols)

    def get_round_cols(self) -> list[str]:
        return self.round_cols

    def get_before_round_cols(self) -> list[str]:
        return self.rounds_before

    def get_after_round_cols(self) -> list[str]:
        return self.rounds_after
 
    def get_non_round_cols(self) -> list[str]:
        """ Get columns that are not rounds. """
        return self.non_round_cols

    def get_num_reads(self) -> list[int]:
        """ Returns list of number of reads per round. """
        return list(self.df[self.round_cols].sum(axis = 'rows'))

    def get_genotypes(self) -> list[str]:
        return list(self.df[self.genotype_col])

    def get_genotype_col(self) -> str:
        return self.genotype_col

    def get_steps_per_round(self) -> list[float]:
        return self.steps_per_round

    @functools.cache
    def get_num_train_pairs_per_round(self) -> list[int]:
        """ Returns list of valid num. training points in each before/after
            pair: count in before round must be >0.            
        """
        pd_mask = self.df[self.rounds_before] > 0
        num_nonzero_per_round = list(pd_mask.sum(axis = 'rows'))
        return num_nonzero_per_round

    def describe(self) -> pd.DataFrame:
        """ Get summary stats """
        summary = pd.DataFrame(columns = self.round_cols)
        for col in self.round_cols:
            values = self.df[col]
            summary.loc['Total reads', col] = values.sum()
            summary.loc['Unique genotypes', col] = sum(values > 0)
            summary.loc['Max frequency', col] = max(values / values.sum())
        return summary

    """
        Update
    """
    def update_with_col(
        self, 
        name: str, 
        values: NDArray,
        overwrite: bool = False
    ) -> None:
        if not overwrite:
            assert name not in self.df.columns
        assert len(values) == len(self.df)
        self.df.loc[:, name] = values
        return

    def update_with_target_pred_cols(
        self, 
        logw_col: str
    ) -> dict[str, list[str]]:
        """ Adds columns: Target / Pred masked {round_col} ({logw_col}),
            by predicting using prior round as input = `extrapolated`.
            Not a valid prediction under latent frequency model.
            Returns pred/target column names.
        """
        df = self.df
        assert logw_col in df.columns, \
            'Require df to be annotated with logw first'

        log_w = torch.tensor(np.array(df[logw_col]))

        steps_per_round = self.get_steps_per_round()
        target_cols = []
        pred_cols = []
        for i, (before_col, after_col) in enumerate(
            zip(self.rounds_before, self.rounds_after)
        ):
            r0 = torch.tensor(np.array(df[before_col]))
            r1 = torch.tensor(np.array(df[after_col]))
            r1_round_col = after_col
            steps = steps_per_round[i]

            sim_result = simulate.simulate_mask_log_fqs(
                log_W = log_w,
                inp_counts = r0,
                steps = steps
            )
            mask = sim_result.mask
            mask_log_p1_pred = sim_result.mask_log_p1_pred
            mask_p1_pred = torch.exp(mask_log_p1_pred)

            mask_r1 = r1[mask]
            mask_r1_fqs = mask_r1 / mask_r1.sum()

            target_col_nm = f'Target masked {r1_round_col} ({logw_col})'
            pred_col_nm = f'Pred masked {r1_round_col} ({logw_col}), extrapolated'

            df.loc[:, target_col_nm] = fill_masked_tensor(mask_r1_fqs, mask)
            df.loc[:, pred_col_nm] = fill_masked_tensor(mask_p1_pred, mask)

            target_cols.append(target_col_nm)
            pred_cols.append(pred_col_nm)
        return {'Target cols': target_cols, 'Pred cols': pred_cols}

    def remove_col(self, col: str) -> None:
        del self.df[col]
        return

    """
        I/O
    """
    def save_df_to_file(self, out_fn: str) -> None:
        os.makedirs(os.path.dirname(out_fn), exist_ok = True)
        self.df.to_csv(out_fn, index = False)
        logger.info(f'Saved results to {out_fn}')
        return

    """
        Data views on dataframe: Long format, for deep fitness
        Train//prediction.
    """
    def form_long_train_df(
        self,
        target_logw_col: str | None = None,
    ) -> pd.DataFrame:
        """ Convert tsngs_df to long format, to be indexed by
            ManualBatchSampler, for pytorch lightning training for
            deep fitness.
            Only keeps valid training pairs with count > 0.
            Keeps same order.

            An individual training sample / a row, is:
            - Genotype
            - Round pair index
            - Count
            - Next count
            - Steps to next round
            which is a valid training example only if [Count] is non-zero.
            These info columns are packaged into a TSNGSDatapoint.

            If tsngs_df shape is (G)x(T), then long_df length is (Gx(T-1)).

            If target_logw_col is given, then also include that column.
        """
        long_df = pd.DataFrame()
        for idx, (r0, r1) in enumerate(
            zip(self.rounds_before, self.rounds_after)
        ):
            mask = (self.df[r0] > 0)
            subset_cols = [self.genotype_col, r0, r1]
            if target_logw_col:
                assert target_logw_col in self.df.columns
                subset_cols.append(target_logw_col)
            df_slice = self.df[mask][subset_cols].copy()

            df_slice['Genotype index'] = df_slice.index
            df_slice['Round pair index'] = idx
            num_steps = self.steps_per_round[idx]
            df_slice['Steps to next round'] = num_steps

            col_renaming = {
                self.genotype_col: 'Genotype',
                r0: 'Count', 
                r1: 'Next count'
            }
            df_slice = df_slice.rename(col_renaming, axis='columns')

            long_df = pd.concat([long_df, df_slice])

        self.log(f'Total dataset size: {len(long_df)}')
        return long_df

    def log_num_valid_training_samples_per_round(self):
        long_train_df = self.form_long_train_df()
        flat_round_pair_idx_list = list(long_train_df['Round pair index'])
        counts = Counter(flat_round_pair_idx_list)

        self.log('Num. valid samples per round:')
        for round_pair_idx, count in sorted(counts.items()):
            self.log(f'{round_pair_idx}: {count}')
        return

    def form_long_genotype_df(self) -> pd.DataFrame:
        """ Convert tsngs_df to long format, with just genotypes.
            Used to predict inferred fitness for genotypes.

            If tsngs_df shape is (G)x(T), then long_df length is Gx1.
        """
        long_df = pd.DataFrame()
        long_df['Genotype'] = self.df[self.genotype_col]
        return long_df

    def form_long_warmup_df(self, target_logw_col: str) -> pd.DataFrame:
        """ Convert tsngs_df to long format, with genotypes and target_logw col.
            Used for warmup training.

            If tsngs_df shape is (G)x(T), then long_df length is Gx2.
        """
        long_df = pd.DataFrame()
        long_df['Genotype'] = self.df[self.genotype_col]
        long_df['Target logw'] = self.df[target_logw_col]
        return long_df

    """
        Latent model: long training view, tracks, presence, etc.
    """
    @functools.cache
    def form_long_latent_train_df(
        self,
        target_logw_col: str | None = None,
    ) -> pd.DataFrame:
        """ Convert tsngs_df to long format for latent model training,
            to be indexed by ManualBatchSampler,
            for pytorch lightning training for deep latent model.
            Only keeps valid training points that are "present".
            Keeps same order.

            An individual training sample / a row, is:
            - genotype
            - time
            - first time
            - count
            - track_idx
            These info columns are packaged into a TSNGSLatentwithFitnessDataPoint.

            If tsngs_df shape is (G)x(T), then long_df length is (GxT) x 1.
            If target_logw_col is given, then also include that column.

            Requires rounds_before/rounds_after to be parallel tracks.
        """
        assert roundtracks.check_all_parallel(self.rounds_before, self.rounds_after), \
            'Latent model does not support non-parallel tracks'
        long_df = pd.DataFrame()

        tracks = self.get_parallel_tracks()
        for track_idx, track in enumerate(tracks):
            count_mat = np.array(self.df[track])

            firsts = roundtracks.get_first_presence(count_mat)
            lens = roundtracks.get_len_presence(count_mat)

            # all tracks are parallel means rounds_before is unique, enabling
            # us to query into steps_per_round
            get_before_idx = lambda r: self.rounds_before.index(r)
            get_steps = lambda r: self.steps_per_round[get_before_idx(r)]
            steps = [get_steps(r) for r in track[:-1]]
            cum_steps = np.cumsum([0] + steps)
            first_times = np.array([cum_steps[first_r] for first_r in firsts])

            for round_idx, round_col in enumerate(track):
                time = cum_steps[round_idx]
                gt_selector = (firsts <= round_idx) & (round_idx < firsts + lens)

                subset_cols = [self.genotype_col, round_col]
                if target_logw_col:
                    assert target_logw_col in self.df.columns
                    subset_cols.append(target_logw_col)

                df_slice = self.df[gt_selector][subset_cols].copy()
                df_slice['Genotype index'] = df_slice.index
                df_slice['Time'] = time
                df_slice['First time'] = first_times[gt_selector]
                df_slice['Track index'] = track_idx

                col_renaming = {
                    self.genotype_col: 'Genotype',
                    round_col: 'Count', 
                }
                df_slice = df_slice.rename(col_renaming, axis='columns')

                long_df = pd.concat([long_df, df_slice])

        self.log(f'Total dataset size: {len(long_df)}')
        return long_df

    def get_parallel_tracks(self) -> list[list[str]]:
        """ Get parallel tracks on rounds_before/rounds_after """
        return roundtracks.find_parallel_tracks(
            self.rounds_before, 
            self.rounds_after
        )

    """
        Statistics
    """
    def compute_last_round_nll(self, logw_col: str) -> float:
        """ Multinomial nll divided by num. observations in last round.
            Extrapolative: predicts using prior observed counts.
        """
        log_w = torch.tensor(np.array(self.df[logw_col]))
        steps_per_round = self.get_steps_per_round()
        r0 = torch.tensor(np.array(self.df[self.rounds_before[-1]]))
        r1 = torch.tensor(np.array(self.df[self.rounds_after[-1]]))
        steps = steps_per_round[-1]

        sim_result = simulate.simulate_mask_log_fqs(log_w, r0, steps)
        mask = sim_result.mask
        nll = tasks.loss_multinomial_nll(sim_result.mask_log_p1_pred, r1[mask])
        return nll / sum(mask)

    def compute_all_rounds_nll(self, logw_col: str) -> float:
        """ Multinomial nll divided by num. observations in all rounds.
            Extrapolative: predicts using prior observed counts.
        """
        df = self.df
        assert logw_col in df.columns, \
            'Require df to be annotated with logw first'

        log_w = torch.tensor(np.array(df[logw_col]))

        steps_per_round = self.get_steps_per_round()
        nlls = []
        n_per_round = []
        for i, (before, after) in enumerate(
            zip(self.rounds_before, self.rounds_after)
        ):
            r0 = torch.tensor(np.array(df[before]))
            r1 = torch.tensor(np.array(df[after]))
            steps = steps_per_round[i]

            sim_result = simulate.simulate_mask_log_fqs(
                log_W = log_w,
                inp_counts = r0,
                steps = steps
            )
            mask = sim_result.mask
            nll = tasks.loss_multinomial_nll(sim_result.mask_log_p1_pred, r1[mask])

            nlls.append(nll)
            n_per_round.append(sum(mask))

        total_nll = sum(nlls) / sum(n_per_round)
        return total_nll


"""
    Constructor
"""
def construct_tsngs_df(
    df: pd.DataFrame, 
    genotype_col: str,
    round_cols: list[str] | None = None,
    rounds_before: list[str] | None = None,
    rounds_after: list[str] | None = None,
    schema: GenotypeStrSchema = AnyStringSchema,
    steps_per_round: list[float] | None = None,
    **kwargs,
) -> TimeSeriesNGSDataFrame:
    """ Construct TimeSeriesNGSDataFrame, converting round_cols (if given)
        to rounds_before / rounds_after, interpreting rounds sequentially.
    """
    before_after = bool(rounds_before) and bool(rounds_after)
    assert before_after != bool(round_cols), \
        'Require either round_cols or [rounds_before/after], but not both'

    if round_cols:
        rounds_before = round_cols[:-1]
        rounds_after = round_cols[1:]

    return TimeSeriesNGSDataFrame(
        df = df,
        genotype_col = genotype_col,
        rounds_before = rounds_before,
        rounds_after = rounds_after,
        schema = schema,
        steps_per_round = steps_per_round,
        **kwargs
    )


def copy_tsngs_df_settings(
    df: pd.DataFrame,
    tsngs_df: TimeSeriesNGSDataFrame,
    **kwargs
) -> TimeSeriesNGSDataFrame:
    """ Construct new tsngs_df using df, and settings from tsngs_df. """
    return TimeSeriesNGSDataFrame(
        df = df,
        genotype_col = tsngs_df.genotype_col,
        rounds_before = tsngs_df.rounds_before,
        rounds_after = tsngs_df.rounds_after,
        schema = tsngs_df.schema,
        steps_per_round = tsngs_df.steps_per_round,
        **kwargs
    )


"""
    Train/test splitting
"""
def subset_tsngsdf_by_genotype_idxs(
    tsngs_df: TimeSeriesNGSDataFrame, 
    gt_idxs: list[int]
) -> TimeSeriesNGSDataFrame:
    """ Subsets a tsngs_df by genotype idxs, using iloc.
        Reset df index, saving original to `original_index`.
    """
    assert len(gt_idxs) > 0
    return TimeSeriesNGSDataFrame(
        df = tsngs_df.df.iloc[gt_idxs].copy(), 
        genotype_col = tsngs_df.genotype_col,
        rounds_before = tsngs_df.rounds_before,
        rounds_after = tsngs_df.rounds_after,
        schema = tsngs_df.schema,
        steps_per_round = tsngs_df.steps_per_round,
        skip_filters = True,
        reset_index_save = True,
    )


def split_tsngsdf_by_genotypes(
    tsngs_df: TimeSeriesNGSDataFrame, 
    train_idxs: list[int], 
    val_idxs: list[int],
    test_idxs: list[int]
) -> Tuple[TimeSeriesNGSDataFrame, TimeSeriesNGSDataFrame]:
    """ Split TimeSeriesNGSDataFrame by genotype into train/val/test. """
    train_dataset = subset_tsngsdf_by_genotype_idxs(tsngs_df, train_idxs)
    val_dataset = subset_tsngsdf_by_genotype_idxs(tsngs_df, val_idxs)

    if len(test_idxs) > 0:
        test_dataset = subset_tsngsdf_by_genotype_idxs(tsngs_df, test_idxs)
    else:
        test_dataset = None
    return train_dataset, val_dataset, test_dataset


def split_tsngsdf_by_rounds(
    tsngs_df: TimeSeriesNGSDataFrame | None, 
    train_round_idxs: list[int], 
    val_round_idxs: list[int],
    test_round_idxs: list[int]
) -> Tuple[
    TimeSeriesNGSDataFrame | None, 
    TimeSeriesNGSDataFrame | None, 
    TimeSeriesNGSDataFrame | None
]:
    """ Split TimeSeriesNGSDataFrame by rounds.
        Indices index into rounds_before and rounds_after.
    """
    if tsngs_df is None:
        return None, None, None
    assert sorted(train_round_idxs) == train_round_idxs
    assert sorted(val_round_idxs) == val_round_idxs
    assert sorted(test_round_idxs) == test_round_idxs

    idx_in = lambda _list, idxs: [_list[i] for i in idxs]

    train_dataset = TimeSeriesNGSDataFrame(
        df = tsngs_df.df.copy(), 
        genotype_col = tsngs_df.genotype_col,
        rounds_before = idx_in(tsngs_df.rounds_before, train_round_idxs),
        rounds_after = idx_in(tsngs_df.rounds_after, train_round_idxs),
        schema = tsngs_df.schema,
        steps_per_round = idx_in(tsngs_df.steps_per_round, train_round_idxs),
        skip_filters = True,
    )
    if len(val_round_idxs) > 0:
        val_dataset = TimeSeriesNGSDataFrame(
            df = tsngs_df.df.copy(), 
            genotype_col = tsngs_df.genotype_col,
            rounds_before = idx_in(tsngs_df.rounds_before, val_round_idxs),
            rounds_after = idx_in(tsngs_df.rounds_after, val_round_idxs),
            schema = tsngs_df.schema,
            steps_per_round = idx_in(tsngs_df.steps_per_round, val_round_idxs),
            skip_filters = True,
        )
    else:
        val_dataset = None
    test_dataset = TimeSeriesNGSDataFrame(
        df = tsngs_df.df.copy(), 
        genotype_col = tsngs_df.genotype_col,
        rounds_before = idx_in(tsngs_df.rounds_before, test_round_idxs),
        rounds_after = idx_in(tsngs_df.rounds_after, test_round_idxs),
        schema = tsngs_df.schema,
        steps_per_round = idx_in(tsngs_df.steps_per_round, test_round_idxs),
        skip_filters = True,
    )
    return train_dataset, val_dataset, test_dataset
