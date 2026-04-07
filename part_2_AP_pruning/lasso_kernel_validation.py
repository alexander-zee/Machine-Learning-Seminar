"""
lasso_kernel_validation.py — Kernel CV fold estimation (Phase 1).

Per-month kernel-weighted estimation, storing only Sharpe ratios.
For each evaluation month t*: kernel-weighted moments → SVD → LARS.
Accumulates monthly SDF returns, computes SR at the end.

Output: one compact CSV per (lambda0, lambda2), named with h_idx.
    CV fold:  (portsN, valid_SR)
    Full fit: (portsN, test_SR)    ← only used by kernel_full_fit

No betas are stored — they vary per month and the mean is meaningless.
Betas are recovered in Phase 2 (kernel_full_fit / kernel_reconstruct).

Functions
---------
kernel_cv_helper : per-month kernel → LARS, stores SR-only CSVs
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from .lasso_core import one_month_lars


def kernel_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
                     main_dir, sub_dir, adj_w, cv_name, kmin, kmax,
                     kernel, h_idx, state_train, state_valid, state_test):
    """
    Per-month kernel-weighted estimation, storing only Sharpe ratios.

    During CV folds (ports_valid is not None): runs validation loop only.
    During full fit (ports_valid is None): runs test loop only.
    """
    _state_train    = np.asarray(state_train)
    ports_train_arr = ports_train.values

    # Accumulators: (l0_idx, l2_idx) -> k -> list of monthly SDF returns
    valid_returns = defaultdict(lambda: defaultdict(list))
    test_returns  = defaultdict(lambda: defaultdict(list))

    # ---- Validation loop ----
    if ports_valid is not None:
        _state_valid    = np.asarray(state_valid)
        ports_valid_arr = ports_valid.values

        for t in range(len(ports_valid)):
            if (t + 1) % 20 == 0 or t == 0:
                print(f"    validation month {t+1}/{len(ports_valid)}", flush=True)

            s       = float(_state_valid[t])
            monthly = one_month_lars(
                ports_train_arr, _state_train, s, kernel,
                lambda0, lambda2, adj_w, kmin, kmax)

            for (i, j, k), sdf_w in monthly.items():
                valid_returns[(i, j)][k].append(float(ports_valid_arr[t] @ sdf_w))

        n_combos = len(valid_returns)
        n_k_example = len(next(iter(valid_returns.values()))) if valid_returns else 0
        print(f"    validation done — {n_combos} (l0,l2) combos, ~{n_k_example} k values each",
              flush=True)

    # ---- Test loop ----
    # Only run during full fit (ports_valid is None).
    if ports_valid is None and state_test is not None:
        _state_test    = np.asarray(state_test)
        ports_test_arr = ports_test.values

        for t in range(len(ports_test)):
            if (t + 1) % 20 == 0 or t == 0:
                print(f"    test month {t+1}/{len(ports_test)}", flush=True)

            s       = float(_state_test[t])
            monthly = one_month_lars(
                ports_train_arr, _state_train, s, kernel,
                lambda0, lambda2, adj_w, kmin, kmax)

            for (i, j, k), sdf_w in monthly.items():
                test_returns[(i, j)][k].append(float(ports_test_arr[t] @ sdf_w))

        n_combos = len(test_returns)
        n_k_example = len(next(iter(test_returns.values()))) if test_returns else 0
        print(f"    test done — {n_combos} (l0,l2) combos, ~{n_k_example} k values each",
              flush=True)

    # ---- Compute Sharpe ratios and save compact CSVs ----
    print(f"    computing Sharpe ratios and writing CSVs...", flush=True)
    for i in range(len(lambda0)):
        for j in range(len(lambda2)):
            key = (i, j)

            if ports_valid is not None:
                all_k = sorted(valid_returns[key].keys())
            else:
                all_k = sorted(test_returns[key].keys())

            if not all_k:
                continue

            rows = []
            for k in all_k:
                row = {'portsN': k}

                if ports_valid is not None:
                    v   = np.array(valid_returns[key][k])
                    std = v.std(ddof=1)
                    row['valid_SR'] = float(v.mean() / std) if std > 0 else np.nan
                else:
                    v   = np.array(test_returns[key][k])
                    std = v.std(ddof=1)
                    row['test_SR'] = float(v.mean() / std) if std > 0 else np.nan

                rows.append(row)

            if not rows:
                continue

            results = pd.DataFrame(rows)

            if ports_valid is not None:
                col_order = ['valid_SR', 'portsN']
            else:
                col_order = ['test_SR', 'portsN']
            results = results[[c for c in col_order if c in results.columns]]

            results.to_csv(
                os.path.join(main_dir, sub_dir,
                             f'results_{cv_name}_l0_{i+1}_l2_{j+1}_h_{h_idx}.csv'),
                index=False)

    print(f"    {cv_name} h_{h_idx} done — saved {len(lambda0)*len(lambda2)} CSVs",
          flush=True)