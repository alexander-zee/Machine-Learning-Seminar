# Machine Learning Seminar — AP-Trees with Time-Varying Kernel Weighting

## Overview

This project extends the Asset Pricing Trees (AP-Trees) framework of Bryzgalova, Pelger & Zhu (2025) —
*Forest Through the Trees: Building Cross-Sections of Stock Returns* — by introducing two key innovations:

1. **Kernel-based time-varying weighting** within the AP-tree pruning step, allowing the SDF to adapt
   to shifting economic conditions and recent observations rather than treating all history equally.
2. **Random Projection (RP) trees** as an alternative to the standard median-split AP-tree partitioning,
   enabling more flexible high-dimensional splits based on random linear combinations of characteristics.

---

## Authors

- Alexander Zee
- Maarten Schelhaas
- Aiden de Haan
- Michael Reijngoud

---

## Repository Structure

```
root/
│
├── 0_code/                         Original R code from Bryzgalova et al. (2025).
│                                   Used for reproduction and cross-validation with
│                                   our Python implementation.
│
├── data/
│   ├── raw/                        Raw input data (not included in repo).
│   │   ├── FINALdataset.csv        CRSP/Compustat panel data (characteristics + returns)
│   │   └── Factor datasets
│   │
│   ├── prepared/
│   │   └── panel.parquet           Cleaned, rank-transformed panel (output of step 1)
│   │
│   └── results/
│       ├── tree_portfolios/        AP-tree candidate portfolios, one subfolder per triplet
│       │                           e.g. LME_BEME_OP/level_all_excess_combined_filtered.csv
│       ├── rp_tree_portfolios/     RP-tree candidate portfolios, same layout
│       ├── mice_rp_tree_portfolios/ MICE-imputed RP-tree portfolios (n_features sweep)
│       ├── grid_search/
│       │   ├── tree/               AP-tree LASSO/CV results, organised by kernel
│       │   │   ├── uniform/
│       │   │   ├── gaussian/
│       │   │   ├── exponential/
│       │   │   └── gaussian_tms/
│       │   ├── rp_tree/            RP-tree LASSO/CV results (uniform)
│       │   └── mice_rp_tree/       MICE RP-tree LASSO/CV results
│       └── diagnostics/            Weight diagnostics, bandwidth analysis, outlier checks
│
├── part_1_portfolio_creation/
│   └── tree_portfolio_creation/    Core pipeline for building candidate portfolios
│
├── part_2_AP_pruning/
│   ├── kernels/                    Kernel weight classes
│   └── *.py                        LASSO/LARS pruning routines and entry points
│
├── part_3_metrics_collection/      Sharpe ratio picking, factor regressions,
│                                   transaction costs, table exports
│
├── part_4_plots/                   Visualisation scripts for all paper figures
│
├── tests/                          Unit tests to verify Python ↔ R equivalence
│
│   — Root-level entry points —
│
├── build_tree_portfolios.py        ⬅ Run FIRST: steps 1–4 for all 36 AP-tree triplets
├── features_rp.py                  Full pipeline for MICE RP-tree (all n_features)
├── run_all_rp_cross_sections.py    Full RP-tree pipeline for all 36 triplets
├── standard_uniform_all.py         AP-tree uniform baseline — all 36 triplets
├── standard_gaussian_all.py        AP-tree Gaussian kernel (svar) — all 36 triplets
├── standard_gaussian_tms_all.py    AP-tree Gaussian kernel (TMS) — all 36 triplets
├── standard_exponential_all.py     AP-tree exponential kernel — all 36 triplets
├── standard_gaussian_rp_all.py     RP-tree Gaussian kernel — all 36 triplets
├── standard_gaussian_tms_rp_all.py RP-tree Gaussian kernel (TMS) — all 36 triplets
├── standard_exponential_rp_all.py  RP-tree exponential kernel — all 36 triplets
│
├── requirements.txt
└── README.md
```

---

## Pipeline — Step by Step

The pipeline is divided into four sequential stages. Each stage reads from the previous stage's output.

### Stage 1 — Portfolio Creation (`part_1_portfolio_creation/`)

Located in `part_1_portfolio_creation/tree_portfolio_creation/`.

| Script | Purpose |
|---|---|
| `step1_prepare_data.py` | Reads raw CRSP/Compustat CSV, rank-transforms characteristics cross-sectionally, and writes `data/prepared/panel.parquet`. Also builds the state-variable CSV (SVAR, DEF, TMS). |
| `step1b_impute_data.py` | Applies MICE imputation to the panel (used for the RP tree extension). |
| `step2_tree_portfolios.py` | Standard AP-tree portfolio construction: for each of the `3^depth` characteristic orderings, recursively splits stocks at the monthly cross-sectional median and computes value-weighted returns at all tree nodes. |
| `step2_RP_tree_portfolios.py` | RP-tree variant: instead of single-variable median splits, uses random linear projections of characteristics, building 81 trees per triplet. |
| `step2_mice_rp_portfolios.py` | Same as RP-trees but operates on the MICE-imputed panel. |
| `step2_cluster_portfolios.py` | Experimental: replaces median splits with Ward agglomerative clustering over all characteristics simultaneously. |
| `step3_combine_trees.py` | Stacks all tree-orderings, deduplicates portfolios with identical return histories, and subtracts the risk-free rate. Outputs `level_all_excess_combined_filtered.csv`. |
| `step3_combine_RP_trees.py` | Same combination step for RP-trees. |
| `step3_combine_mice_rp.py` | Same combination step for MICE RP-trees. |
| `step4_filter_portfolios.py` | Removes single-sort portfolios that are collinear with standard decile sorts (AP-trees only). |

### Stage 2 — AP Pruning (`part_2_AP_pruning/`)

The pruning stage selects a sparse set of portfolios from the candidates using LASSO/LARS with cross-validation, following the BPZ framework.

**Kernels** (`part_2_AP_pruning/kernels/`):

| Kernel | Description |
|---|---|
| `uniform.py` | Equal weights on all observations in the estimation window (baseline). Included for interface consistency — its weights are not actually used in the LASSO step. |
| `gaussian.py` | Weights observations by their distance from the current month in the macroeconomic state space (SVAR, DEF, TMS or a univariate subset). |
| `exponential.py` | Weights observations by a smooth time-based exponential decay: recent months receive more weight, older months less. No state-variable conditioning. |
| `base.py` | Abstract base class defining the `weights()` interface for all kernels. |

**Entry Points:**

| Script | Purpose |
|---|---|
| `AP_Pruning.py` | Main pruning entry point for standard AP-trees. Runs the CV grid search over `(lambda0, lambda2)` and optionally over kernel bandwidths. |
| `RP_Pruning.py` | Same as above, adapted for RP-tree portfolio files. |
| `Mice_RP_Pruning.py` | Same as above, adapted for MICE RP-tree portfolio files. |
| `lasso.py` / `lasso_core.py` | LARS solver and core LASSO logic with depth-based adjustment weights. |
| `lasso_uniform.py` | CV helper for the uniform (static) kernel — faster since μ/Σ do not need recomputing each month. |
| `lasso_kernel_validation.py` | CV helper for kernel-weighted runs. |
| `lasso_kernel_full_fit.py` | Full out-of-sample fit using the winning hyperparameters, producing the test-period SDF. |
| `lasso_valid_par_full.py` | Routing for AP Pruning, and creates folds if selected. |


### Stage 3 — Metrics Collection (`part_3_metrics_collection/`)

After pruning, this stage selects the best hyperparameters and computes output metrics.

| Script | Purpose |
|---|---|
| `pick_best_lambdas.py` | Reads all CV result CSVs, selects the `(lambda0, lambda2)` pair (and bandwidth for kernel runs) that maximises the validation Sharpe ratio, and extracts the corresponding portfolio weights. |
| `mice_pick_best_lambdas.py` | Same, with adapted file-naming conventions for the MICE RP-tree runs. |
| `uniform_full_fit.py` | Produces the same full-fit output format for the uniform baseline as the kernel scripts, enabling a consistent comparison. |
| `sr_test_ledoit_wolf.py` | Ledoit-Wolf corrected Sharpe ratio significance test. |
| `transaction_costs.py` | Computes net-of-transaction-cost returns by penalising turnover in underlying stocks. |
| `tc_batch_runner.py` | Batch runner for transaction cost computations across all cross-sections and kernels. |
| `ff5.py` / `ff5_batch_regression.py` | Constructs the SDF as a weighted portfolio combination, regresses it against Fama-French 5 factors, and reports alpha and p-values. |
| `mice_ff5.py` / `mice_ff5_batch_regression.py` | Same as above for MICE RP-tree portfolios. |
| `factor_regression.py` | General factor regression utility. |
| `aggregate_rp_tc_summaries.py` | Aggregates transaction-cost summaries across RP and AP tree runs. |
| `create_sr_table_all.py` | Exports the Sharpe ratio summary table. |
| `export_table51_*.py` | Export formatted comparison tables (Table 5.1 variants) for uniform vs Gaussian/TMS kernels. |
| `rp_oos_ff5_multikernel_table.py` | Multi-kernel OOS FF5 alpha comparison table for RP trees. |
| `plot_rp_weights_ff5_alpha.py` | Plots RP-tree portfolio weights alongside FF5 alphas. |

### Stage 4 — Plots (`part_4_plots/`)

| Script | Purpose |
|---|---|
| `alpha_and_SR_plot.py` | Plots out-of-sample Sharpe ratios and FF5 alphas across kernels and triplets. |
| `bandwidth_diagnostics.py` | Diagnostic plots for kernel bandwidth selection. |
| `outlier_diagnostics.py` | Flags and visualises extreme kernel weight observations (e.g. the Oct 2008 spike). |
| `plot_state_variables.py` | Time-series plots of the macroeconomic state variables (SVAR, DEF, TMS). |
| `plot_tc_scatter.py` | Scatter plots of gross vs net-of-transaction-cost Sharpe ratios. |
| `visualize_kernel_weights.py` | Visualises the kernel weight distribution for a set of target months, showing effective sample size. |

---

## Root-Level Full-Run Scripts

These scripts are the primary entry points for running the complete pipeline across all 36 LME-anchored triplets
(or the full n_features sweep for the MICE RP extension). Each script handles the grid search, hyperparameter
selection, and full out-of-sample fit in one invocation, with multiprocessing and a progress-tracking CSV so
runs can be resumed after interruption.

| Script | Trees | Kernel | State variable |
|---|---|---|---|
| `build_tree_portfolios.py` | AP-tree | — | — |
| `standard_uniform_all.py` | AP-tree | Uniform (baseline) | — |
| `standard_gaussian_all.py` | AP-tree | Gaussian | `svar` (stock variance) |
| `standard_gaussian_tms_all.py` | AP-tree | Gaussian | `TMS` (term spread) |
| `standard_exponential_all.py` | AP-tree | Exponential (time-decay) | — (time-based) |
| `standard_gaussian_rp_all.py` | RP-tree | Gaussian | `svar` |
| `standard_gaussian_tms_rp_all.py` | RP-tree | Gaussian | `TMS` |
| `standard_exponential_rp_all.py` | RP-tree | Exponential | — |
| `run_all_rp_cross_sections.py` | RP-tree | Uniform | — |
| `features_rp.py` | MICE RP-tree | Uniform / Gaussian / Exponential | configurable |

**`features_rp.py`** is the master pipeline for the MICE-imputed RP-tree extension. It sweeps over
`n_features_per_split ∈ {1, …, 10}`, running data preparation, tree construction, pruning, and metric
collection. The active kernel is controlled by commenting/uncommenting the relevant `run_pipeline(...)` call
at the bottom of the file. The uniform run is active by default; Gaussian (TMS) and exponential runs are
commented out.


---

## Replicating Results

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the raw data files in `data/raw/`:
   - `FINALdataset.csv` — CRSP/Compustat panel
   - `F-F_Research_Data_5_Factors_2x3.csv`
   - `F-F_Research_Data_Factors.csv`
   - `rf_factor.csv`
   - `state_variables.csv` ( generate via `step1_prepare_data.py` -> this one is directly under `data/`)


### Step 0 — Build AP-Tree Portfolios (required once)
 
```bash
# First run — prepares data and builds all 36 triplets
python build_tree_portfolios.py
 
# Subsequent runs — skip if panel.parquet and any completed triplets already exist
python build_tree_portfolios.py --skip-step1 --skip-existing
```

### Running the Baseline (Uniform AP-Trees)

```bash
python standard_uniform_all.py
```

This will:
- Run the 3-fold CV grid search over all 36 triplets
- Select best `(lambda0, lambda2)` per triplet
- Produce full out-of-sample fits in `data/results/grid_search/tree/uniform/`

### Running the Kernel Extensions (AP-Trees)

```bash
# Gaussian kernel (svar)
python standard_gaussian_all.py

# Gaussian kernel (TMS)
python standard_gaussian_tms_all.py

# Exponential kernel
python standard_exponential_all.py
```

### Running the RP-Tree Extensions

```bash
# Build RP portfolios and run uniform pruning for all 36 triplets
python run_all_rp_cross_sections.py --skip-existing-part1 --pick-best

# Kernel-weighted RP runs
python standard_gaussian_rp_all.py
python standard_exponential_rp_all.py
```

### Running the MICE RP-Tree Extension

```bash
python features_rp.py
```

To switch kernels, open `features_rp.py` and toggle the `run_pipeline(...)` call at the bottom.

### Cross-Validation with Original R Code

The original Bryzgalova et al. R code is preserved in `0_code/`. Tests in `tests/` compare
key numerical outputs between the Python implementation and R reference values to validate correctness.
However some of these might no longer work correctly due to changed functions,
but they did verify the correctness of the reproduction into python.

