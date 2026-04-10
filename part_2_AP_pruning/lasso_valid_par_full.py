"""
lasso_valid_par_full.py — CV folds + full fit for AP-Pruning grid search.

Changes vs original:
    - lasso_valid_full accepts kernel_cls, bandwidths, state
    - loops over bandwidths internally, exactly like lambda0/lambda2
    - lasso_cv_helper routes to _static_cv_helper (UniformKernel, original
      code completely unchanged) or _kernel_cv_helper (per-month loop)
    - CSV filename now includes h_idx:
          results_{cv_name}_l0_{i}_l2_{j}_h_{h_idx}.csv
    - compute_moments() added as shared utility for kernel-weighted mu/sigma
    - lasso.py: completely unchanged

Functions
---------
lasso_valid_full    : loops over all (lambda0, lambda2, h), saves all CSVs
lasso_cv_helper     : routes static vs kernel path for one (kernel, h_idx)
_static_cv_helper   : original one-shot mu/sigma -> LARS (unchanged logic)
_run_one_lambda0    : inner LARS loop for static path (unchanged logic)
_kernel_cv_helper   : per-month kernel-weighted mu/sigma -> LARS
_one_month_lars     : moments + eigendecomp + LARS for one prediction month
compute_moments     : kernel-weighted mu and sigma
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .lasso import lasso
from .kernels.base import BaseKernel
from .kernels.uniform import UniformKernel


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def lasso_valid_full(ports, lambda0, lambda2, main_dir, sub_dir, adj_w,
                     n_train_valid=360, cvN=3, runFullCV=False,
                     kmin=5, kmax=50, RunParallel=False, ParallelN=10,
                     kernel_cls=None, bandwidths=None, state=None):
    """
    Orchestrate CV folds and full fit across all (lambda0, lambda2, h).

    Loops over bandwidths exactly like lambda0 and lambda2 — all three
    are vectors and all combinations are evaluated and saved to CSVs.

    Parameters
    ----------
    kernel_cls  : kernel class. Defaults to UniformKernel.
    bandwidths  : list of h values. UniformKernel: [None], runs once.
                  GaussianKernel: list of floats from bandwidth_grid().
    state       : (T,) pd.Series of monthly state variable values.
                  None for UniformKernel.
    """
    if kernel_cls is None:
        kernel_cls = UniformKernel
    if bandwidths is None:
        bandwidths = [None]

    os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)

    ports_test = ports.iloc[n_train_valid:]
    state_test = None if state is None else state.iloc[n_train_valid:]

    n_valid    = n_train_valid // cvN
    fold_range = range(1, cvN + 1) if runFullCV else range(cvN, cvN + 1)

    # Loop over bandwidths — symmetric with lambda0/lambda2
    for h_idx, h in enumerate(bandwidths, start=1):

        # Instantiate one kernel per bandwidth value
        if issubclass(kernel_cls, UniformKernel):
            kernel = UniformKernel()
        else:
            kernel = kernel_cls(h=h)

        for fold in fold_range:
            val_start, val_end = (fold - 1) * n_valid, fold * n_valid
            ports_valid = ports.iloc[val_start:val_end]
            ports_train = pd.concat([ports.iloc[:val_start],
                                      ports.iloc[val_end:n_train_valid]])

            state_train = (None if state is None else
                           pd.concat([state.iloc[:val_start],
                                       state.iloc[val_end:n_train_valid]]))
            state_valid = None if state is None else state.iloc[val_start:val_end]

            lasso_cv_helper(
                ports_train, ports_valid, ports_test, lambda0, lambda2,
                main_dir, sub_dir, adj_w, f'cv_{fold}', kmin, kmax,
                RunParallel, ParallelN,
                kernel=kernel, h_idx=h_idx,
                state_train=state_train, state_valid=state_valid,
                state_test=state_test)

        # Full fit: train on all n_train_valid months, evaluate on test
        state_train_full = None if state is None else state.iloc[:n_train_valid]

        lasso_cv_helper(
            ports.iloc[:n_train_valid], None, ports_test, lambda0, lambda2,
            main_dir, sub_dir, adj_w, 'full', kmin, kmax,
            RunParallel, ParallelN,
            kernel=kernel, h_idx=h_idx,
            state_train=state_train_full, state_valid=None,
            state_test=state_test)


# ---------------------------------------------------------------------------
# Router: static (uniform) vs kernel (per-month)
# ---------------------------------------------------------------------------

def lasso_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
                    main_dir, sub_dir, adj_w, cv_name, kmin=5, kmax=50,
                    RunParallel=False, ParallelN=10,
                    kernel=None, h_idx=1,
                    state_train=None, state_valid=None, state_test=None):
    """
    Route to static path for UniformKernel or per-month path for all others.
    """
    if kernel is None:
        kernel = UniformKernel()

    if isinstance(kernel, UniformKernel):
        _static_cv_helper(
            ports_train, ports_valid, ports_test, lambda0, lambda2,
            main_dir, sub_dir, adj_w, cv_name, kmin, kmax,
            RunParallel, ParallelN, h_idx=h_idx)
    else:
        _kernel_cv_helper(
            ports_train, ports_valid, ports_test, lambda0, lambda2,
            main_dir, sub_dir, adj_w, cv_name, kmin, kmax,
            kernel=kernel, h_idx=h_idx,
            state_train=state_train, state_valid=state_valid,
            state_test=state_test)


# ---------------------------------------------------------------------------
# Path 1: static (UniformKernel) — original code, logic completely unchanged
# Only difference vs original: h_idx appears in the CSV filename
# ---------------------------------------------------------------------------

def _static_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
                       main_dir, sub_dir, adj_w, cv_name, kmin=5, kmax=50,
                       RunParallel=False, ParallelN=10, h_idx=1):
    """
    Original one-shot estimation.
    mu/sigma computed once from training data, eigendecomposed once,
    LARS run once per (lambda0, lambda2). Behavior unchanged from original.
    """
    mu     = ports_train.values.mean(axis=0)
    sigma  = np.cov(ports_train.values, rowvar=False)
    mu_bar = mu.mean()

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    gamma = min(min(ports_train.shape), int(np.sum(eigenvalues > 1e-10)))
    D, V  = eigenvalues[:gamma], eigenvectors[:, :gamma]

    sigma_tilde = V @ np.diag(np.sqrt(D)) @ V.T
    mu_robust   = mu.reshape(-1, 1) + np.array(lambda0).reshape(1, -1) * mu_bar
    mu_tilde    = V @ np.diag(1.0 / np.sqrt(D)) @ V.T @ mu_robust

    args = (lambda0, lambda2, sigma_tilde, mu_tilde, ports_train, ports_valid,
            ports_test, adj_w, kmin, kmax, main_dir, sub_dir, cv_name, h_idx)

    if RunParallel:
        Parallel(n_jobs=ParallelN)(
            delayed(_run_one_lambda0)(i, *args) for i in range(len(lambda0)))
    else:
        for i in range(len(lambda0)):
            _run_one_lambda0(i, *args)


def _run_one_lambda0(i, lambda0, lambda2, sigma_tilde, mu_tilde,
                     ports_train, ports_valid, ports_test,
                     adj_w, kmin, kmax, main_dir, sub_dir, cv_name, h_idx):
    """
    Run LARS for all lambda2 values at one fixed lambda0 index.
    Logic unchanged from original. Only change: h_idx in CSV filename.
    """
    for j, l2 in enumerate(lambda2):
        beta_subset, K_subset = lasso(sigma_tilde, mu_tilde[:, i], l2, 100, kmin, kmax)
        n_res = beta_subset.shape[0]
        if n_res == 0:
            continue

        train_SR = np.zeros(n_res)
        test_SR  = np.zeros(n_res)
        valid_SR = np.zeros(n_res)
        betas    = np.zeros((n_res, ports_train.shape[1]))

        for r in range(n_res):
            b = beta_subset[r] * adj_w
            b = b / np.abs(b.sum())
            w = b / adj_w

            sdf_train = ports_train.values @ w
            sdf_test  = ports_test.values  @ w
            train_SR[r] = sdf_train.mean() / sdf_train.std(ddof=1)
            test_SR[r]  = sdf_test.mean()  / sdf_test.std(ddof=1)
            if ports_valid is not None:
                sdf_valid   = ports_valid.values @ w
                valid_SR[r] = sdf_valid.mean() / sdf_valid.std(ddof=1)
            betas[r] = b

        if ports_valid is not None:
            meta = pd.DataFrame({'train_SR': train_SR, 'valid_SR': valid_SR,
                                  'test_SR': test_SR, 'portsN': K_subset})
        else:
            meta = pd.DataFrame({'train_SR': train_SR, 'test_SR': test_SR,
                                  'portsN': K_subset})

        results = pd.concat(
            [meta, pd.DataFrame(betas, columns=ports_train.columns)], axis=1)
        results.to_csv(
            os.path.join(main_dir, sub_dir,
                         f'results_{cv_name}_l0_{i+1}_l2_{j+1}_h_{h_idx}.csv'),
            index=False)


# ---------------------------------------------------------------------------
# Path 2: kernel — per-month estimation loop
# ---------------------------------------------------------------------------

def _kernel_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
                       main_dir, sub_dir, adj_w, cv_name, kmin, kmax,
                       kernel, h_idx, state_train, state_valid, state_test):
    """
    Per-month kernel-weighted estimation (Section 4.2/4.3 of the paper).

    For each evaluation month t*:
        s = state[t*]  (already lagged: S_{t*-1})
        kernel weights over T_train months  ->  (T_train,)
        mu_h(s), sigma_h(s) via compute_moments
        eigendecompose sigma_h(s)
        LARS for all (lambda0, lambda2)  ->  omega(t*)
        SDF return = ports_eval[t*] @ omega(t*)

    Accumulates monthly SDF returns per (lambda0, lambda2, k).
    Sharpe ratio computed from the full time series of returns.
    One CSV saved per (lambda0, lambda2), named with h_idx.
    Betas in CSV are mean betas across all evaluation months.
    """
    _state_train    = np.asarray(state_train)
    ports_train_arr = ports_train.values
    port_cols       = ports_train.columns

    # (l0_idx, l2_idx) -> k -> list of monthly SDF returns
    valid_returns = defaultdict(lambda: defaultdict(list))
    test_returns  = defaultdict(lambda: defaultdict(list))
    betas_acc     = defaultdict(lambda: defaultdict(list))

    # Validation loop
    if ports_valid is not None:
        _state_valid    = np.asarray(state_valid)
        ports_valid_arr = ports_valid.values

        for t in range(len(ports_valid)):
            s       = float(_state_valid[t])
            monthly = _one_month_lars(
                ports_train_arr, _state_train, s, kernel,
                lambda0, lambda2, adj_w, kmin, kmax)

            for (i, j, k), (sdf_w, b) in monthly.items():
                valid_returns[(i, j)][k].append(float(ports_valid_arr[t] @ sdf_w))
                betas_acc[(i, j)][k].append(b)

    # Test loop
    if state_test is not None:
        _state_test    = np.asarray(state_test)
        ports_test_arr = ports_test.values

        for t in range(len(ports_test)):
            s       = float(_state_test[t])
            monthly = _one_month_lars(
                ports_train_arr, _state_train, s, kernel,
                lambda0, lambda2, adj_w, kmin, kmax)

            for (i, j, k), (sdf_w, _) in monthly.items():
                test_returns[(i, j)][k].append(float(ports_test_arr[t] @ sdf_w))

    # Save one CSV per (lambda0, lambda2)
    for i in range(len(lambda0)):
        for j in range(len(lambda2)):
            key = (i, j)

            if ports_valid is not None:
                common_k = sorted(
                    set(valid_returns[key].keys()) & set(test_returns[key].keys()))
            else:
                common_k = sorted(test_returns[key].keys())

            if not common_k:
                continue

            rows = []
            for k in common_k:
                row = {'portsN': k, 'train_SR': np.nan}

                if ports_valid is not None and valid_returns[key][k]:
                    v   = np.array(valid_returns[key][k])
                    std = v.std(ddof=1)
                    row['valid_SR'] = float(v.mean() / std) if std > 0 else np.nan

                if test_returns[key][k]:
                    v   = np.array(test_returns[key][k])
                    std = v.std(ddof=1)
                    row['test_SR'] = float(v.mean() / std) if std > 0 else np.nan

                mean_b = (np.mean(betas_acc[key][k], axis=0)
                          if betas_acc[key][k]
                          else np.zeros(ports_train_arr.shape[1]))
                for col, val in zip(port_cols, mean_b):
                    row[col] = val

                rows.append(row)

            if not rows:
                continue

            results   = pd.DataFrame(rows)
            meta_cols = (['train_SR', 'valid_SR', 'test_SR', 'portsN']
                         if ports_valid is not None
                         else ['train_SR', 'test_SR', 'portsN'])
            beta_cols = [c for c in results.columns if c not in set(meta_cols)]
            results   = results[[c for c in meta_cols if c in results.columns] + beta_cols]

            results.to_csv(
                os.path.join(main_dir, sub_dir,
                             f'results_{cv_name}_l0_{i+1}_l2_{j+1}_h_{h_idx}.csv'),
                index=False)


def _one_month_lars(ports_train_arr, state_train_arr, s_current,
                    kernel, lambda0, lambda2, adj_w, kmin, kmax):
    """
    For one prediction month with current state s_current:
        compute kernel-weighted mu and sigma over all training months
        eigendecompose sigma
        run LARS for all (lambda0, lambda2) combinations

    Returns
    -------
    dict : (l0_idx, l2_idx, k) -> (sdf_weights, betas)
        sdf_weights : (N,) weights — dot with test return -> one SDF return
        betas       : (N,) accumulated to compute mean betas across months
    """
    mu, sigma = compute_moments(
        ports_train_arr, kernel, state_train_arr, s_current)
    mu_bar = mu.mean()

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    gamma = min(ports_train_arr.shape[0], ports_train_arr.shape[1],
                int(np.sum(eigenvalues > 1e-10)))

    if gamma == 0:
        return {}

    D, V        = eigenvalues[:gamma], eigenvectors[:, :gamma]
    sigma_tilde = V @ np.diag(np.sqrt(D)) @ V.T
    inv_sqrt_D  = V @ np.diag(1.0 / np.sqrt(D)) @ V.T

    results = {}

    for i, l0 in enumerate(lambda0):
        mu_robust = mu + l0 * mu_bar        # (N,)
        mu_tilde  = inv_sqrt_D @ mu_robust  # (N,)

        for j, l2 in enumerate(lambda2):
            beta_subset, K_subset = lasso(
                sigma_tilde, mu_tilde, l2, 100, kmin, kmax)

            if beta_subset.shape[0] == 0:
                continue

            for r in range(beta_subset.shape[0]):
                k = int(K_subset[r])
                b = beta_subset[r] * adj_w
                b = b / np.abs(b.sum())
                w = b / adj_w
                results[(i, j, k)] = (w, b)

    return results


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def compute_moments(returns: np.ndarray,
                    kernel: BaseKernel,
                    state_train: np.ndarray,
                    state_current: float):
    """
    Compute kernel-weighted mean and covariance of portfolio returns.

    With UniformKernel: equivalent to np.mean(axis=0) and np.cov.
    With GaussianKernel: months whose state is close to state_current
                         receive higher weight.

    Parameters
    ----------
    returns       : (T_train, N) excess returns
    kernel        : any BaseKernel instance
    state_train   : (T_train,) state variable values for training months
    state_current : float — S_{t*-1}, the current state value

    Returns
    -------
    mu    : (N,)   kernel-weighted mean return
    sigma : (N, N) kernel-weighted covariance matrix
    """
    w     = kernel.weights(state_train, state_current)  # (T_train,)
    mu    = (w[:, None] * returns).sum(axis=0)           # (N,)
    resid = returns - mu[None, :]                         # (T_train, N)
    sigma = (w[:, None] * resid).T @ resid                # (N, N)
    return mu, sigma
