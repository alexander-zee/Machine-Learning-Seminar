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
├── data/
│   ├── raw/              # original datasets (never modified)
│   └── processed/        # cleaned and aligned data
│
├── src/
│   ├── data/             # data loading & preprocessing
│   ├── sdf/              # SDF estimation (static, rolling, kernel)
│   ├── clustering/       # clustering-based portfolio construction
│   └── evaluation/       # performance metrics & comparison
│
├── notebooks/            # exploratory analysis
├── results/              # output (figures, tables)
│
├── main.py               # entry point
├── requirements.txt
└── README.md
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
