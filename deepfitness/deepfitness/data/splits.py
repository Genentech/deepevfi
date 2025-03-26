import numpy as np, pandas as pd
from typing import Union
from loguru import logger

from hackerargs import args
from deepfitness.data.tsngs import TimeSeriesNGSDataFrame
from deepfitness.data.tsngs import split_tsngsdf_by_rounds
from deepfitness.data.tsngs import split_tsngsdf_by_genotypes
from deepfitness.genotype import dataflows


class DataSplits:
    def __init__(
        self,
        tsngs_df: TimeSeriesNGSDataFrame
    ):
        """ Builds and stores train/val/test TimeSeriesNGSDataFrames, 
            self.data_splits = dict{str: TimeSeriesNGSDataFrame}
        """
        self.full_tsngs_df = tsngs_df

        stages = args.get('stages')
        assert stages in ['train_only', 'train_and_eval']
        self.data_splits = self.prepare_train_test_split(stages)

    def get(
        self, 
        rounds: str, 
        genotypes: str
    ) -> TimeSeriesNGSDataFrame:
        """ Retrieve tsngs_df for given genotype/rounds split. """
        assert rounds in ['train', 'val', 'test']
        assert genotypes in ['train', 'val', 'test']
        key = f'{rounds}_rounds_{genotypes}_gts'
        assert key in self.data_splits, 'Attempted to get split that does not exist'
        return self.data_splits[key]

    def has(self, rounds: str, genotypes: str) -> bool:
        """ Check if genotype/rounds split exists. """
        assert rounds in ['train', 'val', 'test']
        assert genotypes in ['train', 'val', 'test']
        key = f'{rounds}_rounds_{genotypes}_gts'
        return key in self.data_splits

    """
        Train test split
    """
    def prepare_train_test_split(
        self, 
        stages: str
    ) -> dict[str, TimeSeriesNGSDataFrame]:
        """ Split on genotypes or rounds, yielding 1, 2, 3 or 6 slices.
            rounds: train / val / test
            genotypes: train / val / test
            - train_rounds_train_genotypes
            - train_rounds_val_genotypes
            - train_rounds_test_genotypes
            - val_rounds_train_genotypes
            - val_rounds_val_genotypes
            - val_rounds_test_genotypes
            - test_rounds_train_genotypes
            - test_rounds_val_genotypes
            - test_rounds_test_genotypes

            Returns dict where each term above is a key, and value is
            TorchTSNGSDataset.
            
            Options
            - stages: 'train_only', 'train_and_eval', 'predict'
                - test_split_by_round: bool
                    - train_round_idxs: list[int]
                    - val_round_idxs: list[int]
                    - test_round_idxs: list[int]
                - test_split_by_genotype: bool
                    - train_genotype_split_percent: float
                    - val_genotype_split_percent: float
                    - test_genotype_split_percent: float

            Test genotype % could be 0, but train/val % must be > 0.
                    
            Number of data slices
            ---------------------
            1: Only training.
                Only train_rounds_train_genotypes is built, which
                is the full dataset.
            2: Split by round
                If we split on rounds, then we build:
                    - train_rounds_train_genotypes
                    - test_rounds_train_genotypes
            3: Split by genotypes
                If we split on genotypes, then we build:
                    - train_rounds_train_genotypes
                    - train_rounds_val_genotypes
                    - train_rounds_test_genotypes
            6: All splits
                This occurs if we split by both round and genotype. 
        """
        assert stages in ['train_only', 'train_and_eval']
        split_by_round = args.setdefault('test_split_by_round', False)
        split_by_genotype = args.setdefault('test_split_by_genotype', False)
        if stages == 'train_only':
            data_splits = {
                'train_rounds_train_gts': self.full_tsngs_df
            }
            return data_splits
        elif stages == 'train_and_eval':
            # set up variables
            if split_by_round is True:
                train_round_idxs = args.get('train_round_idxs')
                val_round_idxs = args.get('val_round_idxs')
                test_round_idxs = args.get('test_round_idxs')
                train_round_idxs = [int(i) for i in train_round_idxs]
                val_round_idxs = [int(i) for i in val_round_idxs]
                test_round_idxs = [int(i) for i in test_round_idxs]
            if split_by_genotype:
                gt_split_seed = args.get('random_seed')
                gt_train_pct = args.get('train_genotype_split_percent')
                gt_val_pct = args.get('val_genotype_split_percent')
                gt_test_pct = args.get('test_genotype_split_percent')
                for pct in [gt_train_pct, gt_val_pct]:
                    assert pct > 0
                assert gt_train_pct + gt_val_pct + gt_test_pct == 1.0

                # form train/val/test idxs
                idxs = list(range(self.full_tsngs_df.get_num_genotypes()))
                rng = np.random.default_rng(int(gt_split_seed))
                rng.shuffle(idxs)
                cut_test_idx = int(len(idxs) * gt_test_pct)
                cut_valtest_idx = cut_test_idx + int(len(idxs) * gt_val_pct)
                test_gt_idxs = idxs[:cut_test_idx]
                val_gt_idxs = idxs[cut_test_idx:cut_valtest_idx]
                train_gt_idxs = idxs[cut_valtest_idx:]

            # perform splitting by case
            if split_by_round is True and split_by_genotype is False:
                train_data, val_data, test_data = split_tsngsdf_by_rounds(
                    self.full_tsngs_df, 
                    train_round_idxs = train_round_idxs,
                    val_round_idxs = val_round_idxs,
                    test_round_idxs = test_round_idxs,
                )
                data_splits = {
                    'train_rounds_train_gts': train_data,
                    'val_rounds_train_gts': val_data,
                    'test_rounds_train_gts': test_data,
                }
            elif split_by_round is False and split_by_genotype is True:
                train_data, val_data, test_data = split_tsngsdf_by_genotypes(
                    self.full_tsngs_df, 
                    train_gt_idxs, 
                    val_gt_idxs, 
                    test_gt_idxs,
                )
                data_splits = {
                    'train_rounds_train_gts': train_data,
                    'train_rounds_val_gts': val_data,
                    'train_rounds_test_gts': test_data,
                }
            elif split_by_round is True and split_by_genotype is True:
                train_gts, val_gts, test_gts = split_tsngsdf_by_genotypes(
                    self.full_tsngs_df, 
                    train_gt_idxs, 
                    val_gt_idxs, 
                    test_gt_idxs,
                )
                tr_r_tr_gts, va_r_tr_gts, te_r_tr_gts = split_tsngsdf_by_rounds(
                    train_gts,
                    train_round_idxs = train_round_idxs,
                    val_round_idxs = val_round_idxs,
                    test_round_idxs = test_round_idxs,
                )
                tr_r_val_gts, va_r_val_gts, te_r_val_gts = split_tsngsdf_by_rounds(
                    val_gts,
                    train_round_idxs = train_round_idxs,
                    val_round_idxs = val_round_idxs,
                    test_round_idxs = test_round_idxs,
                )
                tr_r_te_gts, va_r_te_gts, te_r_te_gts = split_tsngsdf_by_rounds(
                    test_gts,
                    train_round_idxs = train_round_idxs,
                    val_round_idxs = val_round_idxs,
                    test_round_idxs = test_round_idxs,
                )
                data_splits = {
                    'train_rounds_train_gts': tr_r_tr_gts,
                    'train_rounds_val_gts': tr_r_val_gts,
                    'train_rounds_test_gts': tr_r_te_gts,
                    'val_rounds_train_gts': va_r_tr_gts,
                    'val_rounds_val_gts': va_r_val_gts,
                    'val_rounds_test_gts': va_r_te_gts,
                    'test_rounds_train_gts': te_r_tr_gts,
                    'test_rounds_val_gts': te_r_val_gts,
                    'test_rounds_test_gts': te_r_te_gts,
                }
        else:
            raise ValueError
        return data_splits

    """
        I/O
    """
    def write_data_split_csvs_to_file(self) -> None:
        """ Write each data split tsngs_df to file, in args `output_folder`.
            CSV name format: {train/test}_rounds_{train/val/test}_gts.csv
        """
        for name, tsngs_df in self.data_splits.items():
            if tsngs_df is None:
                continue
            out_fn = args.get('output_folder') + f'/{name}.csv'
            tsngs_df.save_df_to_file(out_fn)
            logger.info(f'Wrote {out_fn}')
        return
