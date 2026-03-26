# Machine Learning Seminar – AP-Trees Extensions

## Overview

This project extends the Asset Pricing Trees (AP-Trees) framework of Bryzgalova et al. (2025) by improving two key components:

1. **Portfolio construction** via clustering methods
2. **Stochastic Discount Factor (SDF) estimation** via state-dependent (kernel-weighted) methods

The objective is to evaluate whether these modifications improve out-of-sample SDF spanning and Sharpe ratios.

---

## Project Structure

```
project/
│
├── main.py orchestrates the full pipeline
├── README.md this file
├── requirements.txt package dependencies
├── .gitignore git ignore rules
├── LICENSE project license
│
├── 1_Portfolio_Creation/
│ │
│ ├── Tree_Portfolio_Creation/
│ │ ├── step1_prepare_data.py ORIGINAL reads raw CRSP/Compustat,
│ │ │ converts characteristics to
│ │ │ cross-sectional quantile ranks,
│ │ │ writes one CSV per year
│ │ │
│ │ ├── step2_tree_portfolios.py ORIGINAL for each of the 3^depth feature
│ │ │ orderings, recursively splits
│ │ │ stocks by median of one
│ │ │ characteristic at each node,
│ │ │ computes value-weighted returns
│ │ │ for all intermediate nodes
│ │ │
│ │ ├── step2_cluster_portfolios.py OURS replaces median splits with
│ │ │ Ward agglomerative clustering
│ │ │ over all characteristics
│ │ │ simultaneously. Produces same
│ │ │ output format as step2_tree
│ │ │
│ │ ├── step3_combine_trees.py ORIGINAL column-binds all tree orderings,
│ │ │ deduplicates portfolios with
│ │ │ identical return histories,
│ │ │ subtracts risk-free rate
│ │ │
│ │ ├── step4_filter_portfolios.py ORIGINAL removes pure single-sort
│ │ │ portfolios (collinear with
│ │ │ standard decile sorts)
│ │ │
│ │ └── tree_portfolio_helper.py ORIGINAL recursive split logic and
│ │ value-weighted return computation
│ │ for one year of data
│ │
│ ├── Traditional_Portfolios/ (ORIGINAL)
│ │ ├── decile_portfolios.py single-sorted decile portfolios per characteristic
│ │ ├── double_sort_portfolios.py 4x4 double-sorted portfolios for all characteristic pairs
│ │ ├── triplesort32_portfolios.py 2x4x4 triple‑sorted portfolios (32 total)
│ │ └── triplesort64_portfolios.py 4x4x4 triple‑sorted portfolios (64 total)
│ │
│ └── (other benchmarks) (optional)
│
├── 2_AP_Pruning/
│ │
│ ├── ap_pruning.py ORIGINAL orchestrates the full pruning
│ │ (refactored) pipeline: applies depth-based
│ │ adjustment weights, runs CV,
│ │ calls LARS, saves results.
│ │ Refactored vs R to accept
│ │ estimate_moments as argument
│ │
│ ├── lars_solver.py ORIGINAL thin wrapper around sklearn's
│ │ lars_path with ridge augmentation
│ │ (replicates R's lars package)
│ │
│ ├── moments_static.py ORIGINAL equal-weighted sample mean and
│ │ (extracted) covariance — the paper's baseline.
│ │ Logic extracted from R's
│ │ lasso_valid_par_full.R
│ │
│ ├── moments_rolling.py ORIGINAL rolling 20-year window moments —
│ │ (extracted) the paper's time-varying benchmark.
│ │ Made explicit here for clean
│ │ comparison
│ │
│ └── moments_kernel.py OURS kernel-weighted mean and covariance
│ conditioned on current market state
│ (VIX, realized variance, term spread).
│ Implements equations (1)-(2) from
│ our proposal / Kim & Oh (2025)
│
├── 3_Metrics_Collection/
│ │
│ ├── pick_best_lambda.py ORIGINAL reads all CV result files, finds
│ │ (lambda0, lambda2) maximizing
│ │ validation Sharpe, extracts
│ │ selected portfolio weights
│ │
│ ├── factor_regression.py ORIGINAL constructs SDF as weighted
│ │ portfolio combination, runs
│ │ time-series OLS against FF3,
│ │ FF5, XSF, FF11 factor sets
│ │
│ └── sharpe.py ORIGINAL computes Sharpe ratio curve
│ vs number of portfolios (SR-N)
│
├── 4_Plots/ (currently empty; will contain visualizations)
│
├── data/ our own processed data (outputs)
├── paper_data/ original data from the paper (inputs) and results

```

---

## Methodology

### 1. Baseline (AP-Trees)

* Portfolio returns constructed via characteristic-based splits
* SDF estimated using:

  * Static (full sample)
  * Rolling window (20-year)

### 2. Time-Varying SDF

* Kernel-weighted estimation based on economic state variables
* State vector includes:

  * Stock variance (SVAR)
  * Default spread (DEF)
  * Term spread (TMS)
* Conditional moments:

  * Mean vector μ
  * Covariance matrix Σ

### 3. Clustering Extension

* Replace median splits with hierarchical clustering
* Ward linkage used to construct portfolios
* Clusters formed based on joint characteristic similarity

### 4. Evaluation

* Out-of-sample Sharpe ratio
* Pricing errors
* Comparison across:

  * Static
  * Rolling
  * Kernel-based
  * Clustering + Kernel

---

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Run the main pipeline:

```
python main.py
```

---

## Workflow

* Each module is developed independently:

  * `data/` → data preparation
  * `sdf/` → SDF estimation
  * `clustering/` → portfolio construction
  * `evaluation/` → results

* Work in separate branches (do not commit directly to main)

---

## Notes

* Ensure no look-ahead bias (use only past information)
* Align all data at monthly frequency
* Apply regularization for numerical stability
* Baselines must be implemented before extensions

---

## Authors

* Alexander Zee
* Maarten Schelhaas
* Aiden de Haan
* Michael Reijngoud

---

## References

* Bryzgalova, Pelger, Zhu (2025) – *Forest Through the Trees*
* Kim & Oh (2025) – Local estimation with state variables
* Welch & Goyal (2008) – Macroeconomic predictors
