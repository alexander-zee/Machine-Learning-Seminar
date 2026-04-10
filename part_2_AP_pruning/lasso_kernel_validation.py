"""
lasso_kernel_validation.py — Kernel CV fold estimation (Phase 1).

Per-month kernel-weighted estimation for CV folds, storing only Sharpe
ratios.  For each validation month t*: kernel-weighted moments → SVD →
LARS.  Accumulates monthly SDF returns, computes SR at the end.

Output: one compact CSV per (lambda0, lambda2, h_idx), named:
    results_cv_{fold}_l0_{i+1}_l2_{j+1}_h_{h_idx}.csv
    Columns: valid_SR, portsN

The full / test run lives entirely in lasso_kernel_full_fit.py.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from .lasso_core import one_month_lars


def kernel_cv_helper(ports_train, ports_valid, lambda0, lambda2,
                     main_dir, sub_dir, adj_w, cv_name, kmin, kmax,
                     kernel, h_idx, state_train, state_valid):
    """
    Validation-only per-month kernel-weighted estimation.

    Runs the validation loop for one CV fold and writes one CSV per
    (lambda0, lambda2) combo containing the out-of-sample validation SR
    at every k in [kmin, kmax].

    Parameters
    ----------
    ports_train  : DataFrame (T_train, N)
    ports_valid  : DataFrame (T_valid, N)
    lambda0      : list of l0 candidates
    lambda2      : list of l2 candidates
    main_dir     : str  — root output directory
    sub_dir      : str  — triplet subfolder
    adj_w        : (N,) depth-based pre-weights
    cv_name      : str  — e.g. 'cv_3'
    kmin, kmax   : int
    kernel       : kernel instance (bandwidth already set)
    h_idx        : int  — 1-indexed, used in CSV filename
    state_train  : array-like (T_train,)
    state_valid  : array-like (T_valid,)
    """
    _state_train    = np.asarray(state_train)
    _state_valid    = np.asarray(state_valid)
    ports_train_arr = ports_train.values
    ports_valid_arr = ports_valid.values

    # Accumulators: (l0_idx, l2_idx) -> k -> list of monthly SDF returns
    valid_returns = defaultdict(lambda: defaultdict(list))

    for t in range(len(ports_valid)):
        if (t + 1) % 20 == 0 or t == 0:
            print(f"    validation month {t+1}/{len(ports_valid)}", flush=True)

        s       = float(_state_valid[t])
        monthly = one_month_lars(
            ports_train_arr, _state_train, s, kernel,
            lambda0, lambda2, adj_w, kmin, kmax)

        for (i, j, k), sdf_w in monthly.items():
            valid_returns[(i, j)][k].append(float(ports_valid_arr[t] @ sdf_w))

    n_combos    = len(valid_returns)
    n_k_example = len(next(iter(valid_returns.values()))) if valid_returns else 0
    print(f"    validation done — {n_combos} (l0,l2) combos, ~{n_k_example} k values each",
          flush=True)

    # ---- Compute Sharpe ratios and save compact CSVs ----
    print(f"    computing Sharpe ratios and writing CSVs...", flush=True)
    for i in range(len(lambda0)):
        for j in range(len(lambda2)):
            key   = (i, j)
            all_k = sorted(valid_returns[key].keys())
            if not all_k:
                continue

            rows = []
            for k in all_k:
                v   = np.array(valid_returns[key][k])
                std = v.std(ddof=1)
                rows.append({
                    'portsN':   k,
                    'valid_SR': float(v.mean() / std) if std > 0 else np.nan,
                })

            pd.DataFrame(rows)[['valid_SR', 'portsN']].to_csv(
                os.path.join(main_dir, sub_dir,
                             f'results_{cv_name}_l0_{i+1}_l2_{j+1}_h_{h_idx}.csv'),
                index=False)

    print(f"    {cv_name} h_{h_idx} done — saved {len(lambda0)*len(lambda2)} CSVs",
          flush=True)