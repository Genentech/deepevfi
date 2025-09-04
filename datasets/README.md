`__raw` contains processed files, which are further filtered by `filter.ipynb` to produce filtered versions in `filtzero_without_lastround` and `filtzero_without_2ndtolastround`.

Compared to truly raw genotype count tables, the files in `__raw` are:
- All files are filtered to remove rows lacking any consecutive timepoints with non-zero reads.
These rows are filtered because fitness inference via empirical gradient matching cannot infer fitness for them.
- fgfr1.csv is from Path E, which uses least stringent selective pressure as last round.
- fgfr1_AHOent0.25.csv processes the genotypes in fgfr1.csv with AHO alignment, then keeping amino acid positions with at least 0.25 entropy, thus removing amino acid positions that do not vary meaningfully over variants.
- tead3_filt1consec_present2ndtolast.csv: After the consecutive row filter, rows with exactly 1 total readcount are then removed, and then rows with zero readcount in the 2nd-to-last timepoint are also removed, because these variants have invalid enrichment from the 2nd-to-last timepoint to the final timepoint, and cannot be used in the test set. Finally, rows with invalid SMILES are removed.
