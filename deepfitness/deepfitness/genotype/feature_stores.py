"""
    Precompute feature map {genotype: np array}, write/read from disk
"""
import pickle
import multiprocessing as mp
import os
import numpy as np
import hashlib
from loguru import logger
from tqdm import tqdm
from numpy.typing import NDArray
from pathlib import Path

from hackerargs import args
from deepfitness.genotype import featurizers


class FeatureStore:
    def __init__(
        self,
        genotypes: list[str],
        featurizer: featurizers.GenotypeFeaturizer
    ):
        """ Handles a feature store: {genotype: np array}.
            Loads from disk, based on tsngs_df genotype set and featurizer options.
            Otherwise, computes from scratch, then saves to file.
        """
        self.genotypes = genotypes
        self.featurizer = featurizer

        self.name = self.get_name()
        print(self.name)
        self.file_name = self.get_file_name()

        self.store = self.load_from_file()
        if self.store is None:
            logger.info('No feature store found on disk, computing ...')
            self.store = self.compute_feature_store()
            logger.info('Done. Saving to file ...')
            self.save_to_file()
            logger.info('Done.')
        else:
            logger.info('Feature store successfully loaded from file.')

    def __getitem__(self, genotype: str) -> NDArray:
        return self.store[genotype]

    def compute_feature_store(self) -> dict[str, NDArray]:
        """ Compute feature store {genotype: NDArray} """
        logger.info(f'Computing feature store: {self.name} ...')
        parallelize = args.setdefault('ft.parallelize_feature_store', True)

        if not parallelize:
            store = {
                gt: np.array(self.featurizer.featurize(gt))
                for gt in tqdm(self.genotypes)
            }
        else:
            inputs = [(self, gt) for gt in self.genotypes] 
            with mp.Pool(num_processes := mp.cpu_count()) as pool:
                fts = pool.starmap(
                    parallelize_featurize,
                    tqdm(inputs, total = len(inputs))
                )
            store = {gt: ft for gt, ft in zip(self.genotypes, fts)}

        return store

    def get_name(self) -> str:
        """ Unique name for feature store, based on tsngs_df and featurizer. 
            Name is based on genotype set (order invariant),
            and featurizer options.
        """
        sorted_gts = sorted(self.genotypes)
        gt_hash = hashlib.md5(str(sorted_gts).encode('utf-8')).hexdigest()
        featurizer_name = str(self.featurizer)
        return str(gt_hash) + '_' + featurizer_name
    
    def get_file_name(self) -> str:
        """ Get pickle file name with path """ 
        ft_store_dir = args.setdefault(
            'ft.store_dir', 
            f'{os.getcwd()}/.featurestore/'
        )
        return f'{ft_store_dir}/{self.name}.pkl'

    """
        IO
    """
    def file_exists(self) -> bool:
        return os.path.isfile(self.file_name)

    def load_from_file(self) -> dict[str, NDArray] | None:
        """ Load {genotype: NDArray} from file. """
        if not self.file_exists():
            return None
        logger.info(f'Loading feature store from {self.file_name}')
        with open(self.file_name, 'rb') as f:
            d = pickle.load(f)
        return d
    
    def save_to_file(self) -> None:
        """ Save {genotype: NDArray} store to file. """
        path = Path(self.file_name)
        path.parent.mkdir(parents = True, exist_ok = True)
        with open(self.file_name, 'wb') as f:
            pickle.dump(self.store, f)
        logger.info(f'Saved feature store to {self.file_name}')
        return


def parallelize_featurize(fstore: FeatureStore, gt):
    return np.array(fstore.featurizer.featurize(gt))