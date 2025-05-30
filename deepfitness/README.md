# deepfitness

DeepFitness is a package for statistical inference and training machine learning models on time series DNA sequencing data ("count tables") from wet-lab directed evolution campaigns. 
We use a model of evolutionary dynamics to infer fitness from data.
Inferred fitness values can be used to rank genotypes to propose superior designs over conventional proposal methods, such as selecting by highest frequency in the final round, or using round-over-round enrichment (inferred fitness is the principled generalization of round-over-round enrichment to 3 or more rounds). 
Fitness oracles, in the form of deep neural networks, can be trained and used for downstream design tasks.
This repository includes python code run from the command line to perform fitness inference, train neural networks, perform data sanity checking, and more. A python plotting library is also provided, intended to be used in interactive jupyter notebooks, or served as images on a web app.
This package is intended to support manual use by a computational scientist, serve as the foundation for interactive web apps for wet-lab biologists, and provide trained deep neural net fitness oracle models.

# Setup
Git clone this repository. If you use a personal access token, use https; e.g.,

Install the conda environment using mamba.

```Shell
conda install -c conda-forge mamba
mamba env create -f env.yml
```

Using env.yml may not work, especially if you require installing a specific version of cuda. A more manual installation would look like this:

```Shell
conda create -n deepfitness python=3.10
conda activate deepfitness
conda config --add channels conda-forge
conda install mamba
mamba install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia
mamba install wandb pytorch-lightning pandas scikit-learn pyro-ppl colorcet seaborn rdkit ipykernel boto3 typing_extensions loguru swifter
```

Finally, to enable imports to work inside this repository, run:

```Shell
pip install -e .
```

# Input data and workflow
Input data are expected to be formatted as a "count table", which is a CSV
with a genotype column and round columns. Each row should be a unique genotype.
Each entry should be the number of sequencing reads for a unique genotype
in a particular round. Fitness inference requires at least 2 timepoint rounds.
This package does not contain any code for processing DNA sequencing files
into count tables.

# Running inference

To run EVFI (without deep learning), use:

```Shell
python -m deepfitness.scripts.train_simplefitness \
    --csv example/TEAD_subset500.csv --genotype_col HELMnolinker \
    --round_cols [0,1,2,3,4,5,6] --output_folder example/output_evfi/
```

To run DeepEVFI, use:

```Shell
python -m deepfitness.scripts.train_deep_latent --config <local_repo_path>/run-benchmarks/gt/filtzero_without_lastround/config_files/final_deep_latent_tead_1fc_p2tl_filtzero.yaml
```

We provide four scripts: simplefitness, simple_latent, deepfitness, and deep_latent.
- Simple vs. deep: Whether or not deep learning is used.
- Latent vs. not latent: Latent indicates that initial variant frequencies are also inferred alongside fitness. Not latent means that only fitness is inferred.

In our experiments, we found that simplefitness, and deep_latent performed the best. These represent the methods reported in our manuscript, and the methods that we recommend for applied use. We provide the simple_latent and deepfitness scripts as research artifacts.

# Scripts

Python scripts for command-line use are provided in `deepfitness/scripts/`. 
As scripts import code from around the package, run them using `python -m`.

All scripts can be run with `-h` flag for a description and main configurable options, specified via CLI `--key val` format. More complex scripts can also be configured using yaml files: `--options_yaml your_args.yaml`.

**Working with count tables; i.e., before fitness inference:**
- `check_count_table_sanity.py`: Check data for violations of assumptions in mathematical model of fitness inference
- `check_genotype_schema.py`: Check genotypes match provided schema
- `filter_count_table.py`: Apply standard data filters, removing duplicates, genotypes with very low read count / frequency, and genotypes lacking consecutive rounds with non-zero reads (fitness cannot be inferred for these).
- `filter_count_table_stream.py`: Applies some of the standard data filters to a CSV too large to load into memory. 
- `check_tracks_overlap.py`: For data with multiple parallel round tracks but the same selection conditions, check that parallel tracks have overlapping genotypes - if so, then fitness inference can be run on all tracks simultaneously.

**Running fitness inference and training models:**
- `train_simplefitness.py`: Infer a fitness value for each genotype, solely using readcount information and ignoring genotype information. Simple fitness inference is simple and fast: a run takes about 1-2 minutes.
- `train_deepfitness.py`: Perform fitness inference while also training a neural network to predict fitness from genotype. Deep fitness inference is best run
with a wandb sweep to perform hyperparameter optimization. Individual runs can take 5-50 minutes.

**After running fitness inference:**
- `clean_wandb_sweep.py`: Housekeeping script, for extracting the best wandb run and checkpoint, table of run statistics, and cleaning up local wandb files
- `predict_deepfitness.py`: Load a deep neural net checkpoint and run it on new data
- `compute_bindnobind.py`: Binarize inferred fitness values into bind/no-bind labels
- `compute_evidence_scores.py`: Compute heuristic evidence scores, based on model fit and read counts, for inferred fitness values
- `compute_uncertainty_profile_likelihood.py`: Computes a 95% confidence interval for inferred fitness values using a profile likelihood approach.
- `adjust_fitness_off_target.py`: Adjust joint [target + instrument] fitness values with control/off-target [instrument only] fitness values, assuming that joint fitness = on-target fitness + off_target fitness. Estimates a mathematically-derived upper bound on the unknown proportionality constants using readcount-thresholded data for robustness to noise.
- `merge_two_fitness_csvs.py`: Merges two fitness CSVs, using shared overlapping genotypes to robustly rescale one fitness scale to the other. This operation is only valid when both campaigns are against the same target, but the campaigns can have different starting libraries or codon tables. 
- `merge_many_fitness_csvs.py`: The many version of the above. Merges a list of fitness CSVs into the smallest possible number of groups by computing pairwise structure of shared genotypes. Outputs a merged CSV for each group. 

