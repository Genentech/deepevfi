# data

This folder handles time-series ngs ("tsngs") datasets. The primary view
is "tsngs", used for training on time-series ngs data. Other views include
"genotype", which views only all genotypes and discards time info; this is
used for predicting fitness for genotypes. Another view is "warmup" which
only includes genotype and target_logw, and is used to warm-up train a 
deep fitness neural net on inferred simple fitness to improve training
stability.

tsngs.TimeSeriesNGSDataFrame is a lightweight wrapper around a pandas 
dataframe for a CSV that describes the time-series NGS data. It performs
a small amount of data cleaning, formatting, and filtering, and provides
different views (e.g., time slices, genotype slices) for constructing
train/test splits.

tsngs_torch handles dataset objects for torch use, by dataloaders.
- TSNGSDatapoint: A class object describing a single data point
- TSNGSDataBatch: A class object for a batch of data points
- TorchTSNGSDataset: Handles the entire dataset for torch loading. The main
representation of the dataset here is "long-form", where each row contains
a single training point at a signle round.

datamodule.TSNGSDataModule owns a TimeSeriesNGSDataFrame and a dataflow,
which describes a genotype schema, featurizer, and collater. 
It constructs train/test splits, and builds torch dataloaders. 

