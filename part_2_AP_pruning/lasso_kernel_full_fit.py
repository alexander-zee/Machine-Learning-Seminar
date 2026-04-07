"""
lasso_kernel_full_fit.py — Kernel test evaluation and reconstruction (Phase 2).

Called after pick_best_lambda_kernel identifies the winning (l0*, l2*, h*).
Runs the per-month kernel loop for just that single combination.

Functions
---------
kernel_full_fit    : test evaluation → saves results_full CSV with test_SR
kernel_reconstruct : full reconstruction → saves per-month betas and SDF returns
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from .lasso import lasso
from .lasso_kernel_validation import kernel_cv_helper


def kernel_full_fit(ports, lambda0_star, lambda2_star, main_dir, sub_dir,
                    adj_w, kernel, h_idx, state,
                    n_train_valid=360, kmin=5, kmax=50):
    """
    Run the test evaluation for one specific (lambda0*, lambda2*, h*) combo.

    Trains on all n_train_valid months, evaluates on test months.
    Saves results_full_l0_1_l2_1_h_{h_idx}.csv with test_SR per k.

    Parameters
    ----------
    ports          : DataFrame (T_total, N) of adj_ports
    lambda0_star   : scalar — winning lambda0
    lambda2_star   : scalar — winning lambda2
    main_dir       : str — output directory
    sub_dir        : str — triplet subfolder
    adj_w          : (N,) depth-based pre-weights
    kernel         : kernel instance with winning bandwidth h*
    h_idx          : int — 1-indexed bandwidth index (for CSV filename)
    state          : pd.Series (T_total,) state variable
    n_train_valid  : int
    kmin, kmax     : int
    """
    os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)

    ports_train = ports.iloc[:n_train_valid]
    ports_test  = ports.iloc[n_train_valid:]
    state_train = state.iloc[:n_train_valid]
    state_test  = state.iloc[n_train_valid:]

    lambda0_list = [lambda0_star]
    lambda2_list = [lambda2_star]

    print(f"  kernel_full_fit: running test evaluation ({len(ports_test)} months)...",
          flush=True)

    kernel_cv_helper(
        ports_train, None, ports_test, lambda0_list, lambda2_list,
        main_dir, sub_dir, adj_w, 'full', kmin, kmax,
        kernel=kernel, h_idx=h_idx,
        state_train=state_train, state_valid=None,
        state_test=state_test)


def kernel_reconstruct(ports, adj_w, lambda0_star, lambda2_star,
                       kernel, state, n_train_valid=360,
                       kmin=5, kmax=50, output_dir=None):
    """
    Full reconstruction: rerun the winning combo and save per-month detail.

    This gives you the T x N matrix of weights for any k you want.

    Parameters
    ----------
    ports          : DataFrame (T_total, N) of adj_ports (pre-scaled returns)
    adj_w          : (N,) depth-based pre-weights
    lambda0_star   : scalar — winning lambda0
    lambda2_star   : scalar — winning lambda2
    kernel         : kernel instance with winning bandwidth h*
    state          : pd.Series (T_total,) state variable
    n_train_valid  : int — train+valid months
    kmin, kmax     : int — portfolio count bounds
    output_dir     : Path — where to save the reconstruction CSVs

    Output
    ------
    Saves to output_dir:
        reconstruction_monthly_sdf_returns.csv
            Columns: month_idx, k, sdf_return

        reconstruction_monthly_betas.csv
            Columns: month_idx, k, port_col_1, ..., port_col_N
            The depth-adjusted betas (b = beta_raw * adj_w, normalized)

    Returns
    -------
    dict with keys:
        'sdf_returns'   : dict k -> list of monthly SDF returns
        'monthly_betas' : dict k -> list of (N,) beta vectors per test month
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    ports_train_arr = ports.iloc[:n_train_valid].values
    ports_test      = ports.iloc[n_train_valid:]
    ports_test_arr  = ports_test.values
    port_cols       = ports.columns.tolist()

    state_train = np.asarray(state.iloc[:n_train_valid])
    state_test  = np.asarray(state.iloc[n_train_valid:])

    T_test = len(ports_test)

    # Accumulators
    sdf_returns   = defaultdict(list)
    monthly_betas = defaultdict(list)
    sdf_rows  = []
    beta_rows = []

    for t in range(T_test):
        if (t + 1) % 20 == 0 or t == 0:
            print(f"    reconstruct month {t+1}/{T_test}", flush=True)

        s = float(state_test[t])

        # Kernel weights and weighted mean
        w_kern  = kernel.weights(state_train, s)
        mu      = (w_kern[:, None] * ports_train_arr).sum(axis=0)
        mu_bar  = mu.mean()

        # SVD of weighted residuals
        resid    = ports_train_arr - mu[None, :]
        sqrt_w   = np.sqrt(w_kern)
        weighted = sqrt_w[:, None] * resid

        U, s_raw, Vt = np.linalg.svd(weighted, full_matrices=False)

        bessel = 1.0 - np.sum(w_kern ** 2)
        s_corr = s_raw / np.sqrt(bessel) if bessel > 0 else s_raw

        mask = s_corr > 1e-10
        s_corr, Vt = s_corr[mask], Vt[mask]
        gamma = len(s_corr)

        if gamma == 0:
            continue

        V           = Vt.T
        sigma_tilde = V @ np.diag(s_corr) @ V.T
        inv_sqrt    = V @ np.diag(1.0 / s_corr) @ V.T

        mu_robust = mu + lambda0_star * mu_bar
        mu_tilde  = inv_sqrt @ mu_robust

        beta_subset, K_subset = lasso(
            sigma_tilde, mu_tilde, lambda2_star, 100, kmin, kmax)

        if beta_subset.shape[0] == 0:
            continue

        seen_k = {}
        for r in range(beta_subset.shape[0]):
            k = int(K_subset[r])
            b = beta_subset[r] * adj_w
            b = b / np.abs(b.sum())
            w = b / adj_w
            seen_k[k] = (w, b)

        for k, (w, b) in seen_k.items():
            sdf_ret = float(ports_test_arr[t] @ w)
            sdf_returns[k].append(sdf_ret)
            monthly_betas[k].append(b)

            sdf_rows.append({'month_idx': t, 'k': k, 'sdf_return': sdf_ret})
            beta_row = {'month_idx': t, 'k': k}
            for col, val in zip(port_cols, b):
                beta_row[col] = val
            beta_rows.append(beta_row)

    if output_dir is not None:
        pd.DataFrame(sdf_rows).to_csv(
            os.path.join(output_dir, 'reconstruction_monthly_sdf_returns.csv'),
            index=False)
        pd.DataFrame(beta_rows).to_csv(
            os.path.join(output_dir, 'reconstruction_monthly_betas.csv'),
            index=False)
        print(f"    Reconstruction saved to {output_dir}", flush=True)

    return {'sdf_returns': dict(sdf_returns),
            'monthly_betas': dict(monthly_betas)}