# Deep Evolutionary Fitness Inference for Variant Nomination from Directed Evolution

This repository accompanies the manuscript on EVFI/DeepEVFI for directed evolution campaigns. It bundles:

- public and newly released multi-round NGS datasets,
- our deepfitness implementation (EVFI and DeepEVFI),
- third-party baselines (ACIDES, Enrich2),
- benchmarking scripts, notebooks, and result CSVs used in the paper.

## Quick Start

1. **Clone** this repository.
2. **Create environments** for each method (recommended via `conda` + `mamba`).
   - `deepfitness/env.yml`
   - `ACIDES/env.yml`
   - `Enrich2/env.yml`
3. **Install deepfitness** for local development:
   ```bash
   conda env create -f deepfitness/env.yml
   conda activate deepfitness
   pip install hackerargs
   pip install -e deepfitness
   ```
4. **Run EVFI** on a sample subset:
   ```bash
   python -m deepfitness.scripts.train_simplefitness \
       --csv deepfitness/example/TEAD_subset500.csv \
       --genotype_col HELMnolinker \
       --round_cols [0,1,2,3,4,5,6] \
       --output_folder deepfitness/example/output_evfi
   ```
5. **Run DeepEVFI** using provided configs (requires GPU-ready env):
   ```bash
   python -m deepfitness.scripts.train_deep_latent \
       --config run-benchmarks/gt/filtzero_without_lastround/config_files/final_deep_latent_tead_1fc_p2tl_filtzero.yaml \
       --project_output_folder /path/to/output
   ```

## Repository Structure

- `deepfitness/` – EVFI & DeepEVFI library, CLI scripts, example data, and env spec.
- `ACIDES/`, `Enrich2/` – vendor code and wrappers to reproduce baselines.
- `datasets/` – filtered count tables and preprocessing notebooks.
- `data-exp/` – SPR KD measurement data, used in figures in the paper.
- `results-data/` – CSV exports of some of the manuscript figures/metrics.
- `run-benchmarks/`, `run-alltime/` – command templates for running benchmarks and fitness inference as used in the paper.
- `notebooks/` – figure notebooks and rendered outputs.
- `utils/` – helper functions shared across scripts.

## Data Sources

- `datasets/__raw/` contains pre-filtered count tables; see `datasets/README.md` for filtering steps.
- `datasets/filter.ipynb` documents the additional filtering for running benchmarking, generating `datasets/filtzero_without_lastround` and `datasets/filtzero_without_2ndtolastround`.
- `data-exp/` holds SPR measurements: `tead3_spr_v3.csv`, `exp_merged_efh.csv`.

Ensure you respect any data usage agreements before redistribution.

## Reproducing Manuscript Results

1. **Prepare environments** for each method as above.
2. **Generate commands**:
   - `run-benchmarks/gt/run.sh` and `run-alltime/gt/run.sh` list example invocations.
   - Update absolute paths (`/evfi-manuscript-public/...`) to match your workspace.
3. **Run methods**:
   - DeepEVFI/EVFI: `python -m deepfitness.scripts.train_deep_latent` or `train_simplefitness`.
   - ACIDES baseline: `run_acides.py` after installing `ACIDES/env.yml`.
   - Enrich2 baseline: `run_enrich2.py` within `Enrich2` env.
4. **Collect outputs** into `results-data/` layout to compare against provided CSVs.

The `notebooks/` folder reads from these output directories to regenerate manuscript plots.

## deepfitness Overview

Core entry points are under `deepfitness/deepfitness/scripts/`:

- **Preprocessing**: `filter_count_table.py`, `check_count_table_sanity.py`, `check_genotype_schema.py`.
- **Inference**: `train_simplefitness.py`, `train_simple_latent.py`, `train_deepfitness.py`, `train_deep_latent.py`.
- **Post-processing**: `predict_deepfitness.py`, `compute_evidence_scores.py`, `compute_uncertainty_profile_likelihood.py`, `merge_*_fitness_csvs.py`.

Configuration can be supplied via CLI flags or YAML files (see `deepfitness/deepfitness/options/`).

## Baseline Methods

- `ACIDES/` packages the ACIDES codebase with our runner script. Follow `ACIDES/README.md` for setup.
- `Enrich2/` includes the Enrich2 release plus a driver script for batch experiments.

## Results and Figures

- `results-data/benchmark-filtzero-without-lastround/` contains published CSV metrics.
- `notebooks/*.ipynb` regenerate the SPR comparison figures; PDFs/PNGs are exported alongside.

## Support

If you use this code or datasets, please cite our manuscript.
