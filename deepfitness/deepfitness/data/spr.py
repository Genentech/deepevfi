from loguru import logger
import pandas as pd, numpy as np
import random, functools, os, copy
from collections import Counter
from typing import Tuple
from numpy.typing import NDArray

import torch

from hackerargs import args
from deepfitness.genotype.schemas import GenotypeStrSchema, AnyStringSchema
from deepfitness.utils import tensor_to_np, fill_masked_tensor


class SPRDataFrame:
    def __init__(
        self,
        df: pd.DataFrame,
        genotype_col: str,
        pkd_col: str,
        schema: GenotypeStrSchema = AnyStringSchema,
    ):
        """ A lightweight wrapper around a pd.DataFrame describing a SPR dataset.

            Input
            -----
            df: pd.DataFrame
                A DataFrame of read counts of genotypes over time points.
            genotype_col: str
                Name of column containing genotypes.
            pkd_col: str
                pKD (-1 * log10(KD molar))
            schema: GenotypeStrSchema
                The schema that genotype_col strings are expected to adhere to.
        """
        self.df = df
        self.genotype_col = genotype_col
        self.pkd_col = pkd_col
        self.schema = schema

    def form_spr_df(self) -> pd.DataFrame:
        """ Convert to spr_df with genotypes and pkd col. """
        spr_df = self.df[[self.genotype_col, self.pkd_col]].copy()
        return spr_df