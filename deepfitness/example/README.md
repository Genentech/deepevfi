# example

This folder contains a toy dataset, `TEAD_subset500.csv` with 500 rows. It conforms to the correct count table format expected as input.
- genotype_col: 'HELMnolinker'
- round_cols: 0,1,2,3,4,5,6
Each entry is an integer read count, and all genotypes are unique.

## Before running fitness inference
First, let's visualize the data and run sanity checks on it.

- The jupyter notebook `plot_count_table.ipynb` shows an example of exploratory data analysis and visualization on `TEAD_subset500.csv` using our plotting library.

Let's now run a sanity check on it. Run this script in your command line - it finishes in less than 1 second on this dataset, and prints out some flags and warnings.

```Shell
python -m deepfitness.scripts.check_count_table_sanity \
  --csv "example/TEAD_subset500.csv" --genotype_col "HELMnolinker" \
  --round_cols 0,1,2,3,4,5,6
```


## Running simple fitness inference
To run simple non-deep fitness inference on this dataset, use:

```Shell
python -m deepfitness.scripts.train_simplefitness \
    --csv example/TEAD_subset500.csv --genotype_col HELMnolinker \
    --round_cols 0,1,2,3,4,5,6 --output_folder example/simplefitness_output/
```

With an A100 GPU, this took about 10 seconds, and saved an output CSV at
`example/simplefitness_output/simplefitness.csv`. 

Inspect the output file to see that new columns have been added to the CSV.

```Shell
head -n 3 example/simplefitness_output/simplefitness.csv
```

## After running fitness inference

The jupyter notebook `example/simplefitness_output/analyze_simplefitness.ipynb` uses our plotting library to visualize and analyze the results of our fitness inference, including model fit, and plotting trajectories.
