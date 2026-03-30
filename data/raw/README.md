# Required raw inputs

Place these files here before running `python run_full_research_pipeline.py`:

1. `FINALdataset.csv`
   - Main stock-level input used by `step1_prepare_data.py`.

2. `rf_factor.csv`
   - Monthly risk-free rate series used by `step3_combine_trees.py`.
   - Expected to be in percent units (e.g., `0.12` means 0.12% monthly).

If either file is missing, the pipeline exits early with a clear message.
