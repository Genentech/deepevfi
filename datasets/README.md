Compared to `raw`, Data-{A-G}.csv are filtered to remove rows lacking any consecutive timepoints with non-zero reads. These rows are filtered because fitness inference via empirical gradient matching cannot infer fitness for them.

Compared to `raw`, tead3_filt_validsmiles.csv is filtered to keep rows that satisfy all of these criteria:
- Row has at least one pair of consecutive rounds with non-zero reads
- Row has valid SMILES

Compared to `raw`, tead3_mfilt_validsmiles.csv is filtered to keep rows that satisfy all of these criteria:
- Row has at least one pair of consecutive rounds with non-zero reads
- Row has max readcount 10 or greater
- Row has max frequency 1e-6 or greater
- Row has valid SMILES


fgfr1.csv: From Path E, which uses least stringent selective pressure as last round.