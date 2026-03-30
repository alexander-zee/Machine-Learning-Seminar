# AP pruning outputs (generated)

Part 2 writes one subfolder per design (e.g. `Ward_clusters_10`, `LME_OP_Investment`) containing:

- `results_full_l0_*_l2_*.csv`, `results_cv_*_l0_*_l2_*.csv` — LASSO path Sharpe grids
- `lambda_grid_meta.json` — λ₀ / λ₂ vectors for the run
- After Part 3 `pick_best_lambda`: `train_SR_*.csv`, `valid_SR_*.csv`, `test_SR_*.csv`, `Selected_Ports_*.csv`, `Selected_Ports_Weights_*.csv`

**Do not commit** these CSVs; the folder is tracked only so the path exists after clone.
