"""
    Handles predictions on CSV of genotypes.

    Reads in CSV of genotypes, converts it to TimeSeriesNGSDataFrame. 
"""
from typing import Union
import pandas as pd

from .tsngs import TimeSeriesNGSDataFrame
from deepfitness.genotype import schemas
from hackerargs import args


def tsngsdf_from_genotype_csv(
    df: pd.DataFrame,
    genotype_col: str,
    schema: schemas.GenotypeStrSchema,
) -> TimeSeriesNGSDataFrame:
    """ For prediction: df contains genotype strings matching schema.
        If round_col is None, add fake rounds to instantiate
        TimeSeriesNGSDataFrame.

        Value of 0 is used for fake count to prevent training.
    """
    fake_cols = ['__fake_round', '__fake_next_round']
    for fake_col in fake_cols:
        assert fake_col not in df.columns
        df[fake_col] = 0
    args.setdefault('__fake_columns', fake_cols)

    tsngs_df = TimeSeriesNGSDataFrame(
        df = df,
        genotype_col = genotype_col,
        rounds_before = ['__fake_round'],
        rounds_after = ['__fake_next_round'],
        schema = schema,
        skip_filters = True,
    )
    return tsngs_df