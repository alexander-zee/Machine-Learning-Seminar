<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
"""
lasso_valid_par_full.py — CV folds + full fit for AP-Pruning grid search.

lasso_valid_full  : orchestrates CV folds and full fit, saves all result CSVs
lasso_cv_helper   : computes mu/sigma, eigendecomposition, runs LARS for all
                    (lambda0, lambda2) combinations and saves one CSV per pair
_run_one_lambda0  : inner loop over lambda2 for one fixed lambda0
"""

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .lasso import lasso


def lasso_valid_full(ports, lambda0, lambda2, main_dir, sub_dir, adj_w,
                     n_train_valid=360, cvN=3, runFullCV=False,
                     kmin=5, kmax=50, RunParallel=False, ParallelN=10):
    """
    Orchestrate CV folds and full fit for the complete (lambda0, lambda2) grid.
    Input  : ports (T x N) depth-weighted excess returns, lambda grids, adj_w (N,).
    Output : saves results_cv_{fold}_l0_{i}_l2_{j}.csv and results_full_l0_{i}_l2_{j}.csv
             for every grid point — one CSV per (lambda0, lambda2) combination.
    """
    os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)

    ports_test = ports.iloc[n_train_valid:]
    n_valid = n_train_valid // cvN
    fold_range = range(1, cvN + 1) if runFullCV else range(cvN, cvN + 1)

    for i in fold_range:
        val_start, val_end = (i - 1) * n_valid, i * n_valid
        ports_valid = ports.iloc[val_start:val_end]
        ports_train = pd.concat([ports.iloc[:val_start], ports.iloc[val_end:n_train_valid]])
        lasso_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
                        main_dir, sub_dir, adj_w, f'cv_{i}', kmin, kmax, RunParallel, ParallelN)

    # Full fit on entire train+valid period (ports_valid=None means no valid_SR saved)
    lasso_cv_helper(ports.iloc[:n_train_valid], None, ports_test, lambda0, lambda2,
                    main_dir, sub_dir, adj_w, 'full', kmin, kmax, RunParallel, ParallelN)


def lasso_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
                    main_dir, sub_dir, adj_w, cv_name, kmin=5, kmax=50,
                    RunParallel=False, ParallelN=10):
    """
    Estimate mu and sigma on training data, build the regression reformulation,
    then run LARS for every (lambda0, lambda2) combination.
    Input  : ports_train (T_train x N), ports_valid (T_valid x N) or None,
             ports_test (T_test x N), lambda grids, adj_w (N,), cv_name string.
    Output : saves results_{cv_name}_l0_{i}_l2_{j}.csv for each grid point.
             CV files: train_SR, valid_SR, test_SR, portsN, betas.
             Full files: train_SR, test_SR, portsN, betas.
    """
    mu = ports_train.values.mean(axis=0)
    sigma = np.cov(ports_train.values, rowvar=False)
    mu_bar = mu.mean()

    # Eigendecompose sigma, keep only gamma meaningful directions
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    gamma = min(min(ports_train.shape), int(np.sum(eigenvalues > 1e-10)))
    D, V = eigenvalues[:gamma], eigenvectors[:, :gamma]

    # Regression reformulation: sigma_tilde (design matrix) and mu_tilde (response)
    # mu_robust[:,i] = mu + lambda0[i] * mu_bar  — one column per lambda0 value
    sigma_tilde = V @ np.diag(np.sqrt(D)) @ V.T
    mu_robust = mu.reshape(-1, 1) + np.array(lambda0).reshape(1, -1) * mu_bar
    mu_tilde = V @ np.diag(1.0 / np.sqrt(D)) @ V.T @ mu_robust

    args = (lambda0, lambda2, sigma_tilde, mu_tilde, ports_train, ports_valid,
            ports_test, adj_w, kmin, kmax, main_dir, sub_dir, cv_name)

    if RunParallel:
        Parallel(n_jobs=ParallelN)(delayed(_run_one_lambda0)(i, *args) for i in range(len(lambda0)))
    else:
        for i in range(len(lambda0)):
            _run_one_lambda0(i, *args)


def _run_one_lambda0(i, lambda0, lambda2, sigma_tilde, mu_tilde,
                     ports_train, ports_valid, ports_test,
                     adj_w, kmin, kmax, main_dir, sub_dir, cv_name):
    """
    Run LARS for all lambda2 values at one fixed lambda0 index i.
    Input  : i (lambda0 index), sigma_tilde/mu_tilde from eigendecomposition,
             train/valid/test DataFrames, adj_w (N,), lambda2 list.
    Output : saves one CSV per lambda2 value containing SR metrics and betas
             for all k in [kmin, kmax], with 1-based filename indices.
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
            # Rescale from depth-weighted space back to original portfolio space
            b = beta_subset[r] * adj_w
            b = b / np.abs(b.sum())
            w = b / adj_w  # weights to apply to original unscaled returns

            sdf_train = ports_train.values @ w
            sdf_test  = ports_test.values @ w
            train_SR[r] = sdf_train.mean() / sdf_train.std(ddof=1)
            test_SR[r]  = sdf_test.mean()  / sdf_test.std(ddof=1)
            if ports_valid is not None:
                sdf_valid = ports_valid.values @ w
                valid_SR[r] = sdf_valid.mean() / sdf_valid.std(ddof=1)
            betas[r] = b

        if ports_valid is not None:
            meta = pd.DataFrame({'train_SR': train_SR, 'valid_SR': valid_SR, 'test_SR': test_SR, 'portsN': K_subset})
        else:
            meta = pd.DataFrame({'train_SR': train_SR, 'test_SR': test_SR, 'portsN': K_subset})

        results = pd.concat([meta, pd.DataFrame(betas, columns=ports_train.columns)], axis=1)
        results.to_csv(os.path.join(main_dir, sub_dir, f'results_{cv_name}_l0_{i+1}_l2_{j+1}.csv'), index=False)
=======
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
>>>>>>> f2ac003 (inital kernel setup)
=======
"""
lasso_valid_par_full.py — CV folds + full fit for AP-Pruning grid search.

Two paths:
    Path 1 (UniformKernel): original static code, completely unchanged.
        - mu/sigma computed once, LARS once per (lambda0, lambda2)
        - One CSV per (l0, l2, h) with rows per k, full beta columns
        - Filename: results_{cv_name}_l0_{i}_l2_{j}_h_{h_idx}.csv

    Path 2 (non-uniform kernel): per-month estimation loop.
        Phase 1 — grid search: compact CSVs with only Sharpe ratios.
            - For each evaluation month t*: kernel-weighted moments → LARS
            - Accumulate monthly SDF returns per (l0, l2, k)
            - Save one CSV per (l0, l2) with rows per k:
                  (portsN, valid_SR, test_SR)   for cv folds
                  (portsN, train_SR, test_SR)    for full fit
            - No betas stored — they differ each month, mean is meaningless
            - Filename: results_{cv_name}_l0_{i}_l2_{j}_h_{h_idx}.csv

        Phase 2 — reconstruction: once best (l0*, l2*, h*) are known,
            kernel_reconstruct() reruns only that combo and saves:
            - Per-month betas: one CSV per (l0, l2) with T rows per k
            - Or the monthly SDF return time series
            - Called separately after pick_best_lambda identifies the winner

Functions
---------
lasso_valid_full       : loops over all (lambda0, lambda2, h), saves CSVs
lasso_cv_helper        : routes static vs kernel path for one (kernel, h_idx)
_static_cv_helper      : original one-shot mu/sigma -> LARS (unchanged logic)
_run_one_lambda0       : inner LARS loop for static path (unchanged logic)
_kernel_cv_helper      : per-month kernel -> LARS, stores SR-only CSVs
_one_month_lars        : moments + eigendecomp + LARS for one prediction month
compute_moments        : kernel-weighted mu and sigma
kernel_reconstruct     : Phase 2 — rerun winning combo, save per-month betas
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

        print(f"  bandwidth {h_idx}/{len(bandwidths)}: {kernel}", flush=True)

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
    print(i)

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
# Path 2: kernel — per-month estimation loop (Phase 1: SR-only CSVs)
# ---------------------------------------------------------------------------

def _kernel_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
                       main_dir, sub_dir, adj_w, cv_name, kmin, kmax,
                       kernel, h_idx, state_train, state_valid, state_test):
    """
    Phase 1: Per-month kernel-weighted estimation, storing only Sharpe ratios.

    For each evaluation month t*:
        s = state[t*]  (already lagged: S_{t*-1})
        kernel weights over T_train months  ->  (T_train,)
        mu_h(s), sigma_h(s) via weighted moments
        eigendecompose sigma_h(s)
        LARS for all (lambda0, lambda2)  ->  omega(t*) for each k
        SDF return = ports_eval[t*] @ omega(t*)

    Accumulates monthly SDF returns per (lambda0_idx, lambda2_idx, k).
    Sharpe ratio computed from the full time-series of returns.

    Output: one CSV per (lambda0, lambda2), named with h_idx.
        Columns: portsN, [train_SR], valid_SR (if cv fold), test_SR
        Rows: one per k in [kmin, kmax] that appeared in the LARS path

    No betas are stored — they vary per month and the mean is meaningless.
    Betas are recovered in Phase 2 (kernel_reconstruct) for the winning combo.
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
            monthly = _one_month_lars(
                ports_train_arr, _state_train, s, kernel,
                lambda0, lambda2, adj_w, kmin, kmax)

            for (i, j, k), sdf_w in monthly.items():
                valid_returns[(i, j)][k].append(float(ports_valid_arr[t] @ sdf_w))

    # ---- Test loop ----
    # Only run during full fit (ports_valid is None).
    # During CV folds we only need valid_SR for selection — skipping the
    # test loop saves 276 expensive per-month LARS calls per fold.
    if ports_valid is None and state_test is not None:
        _state_test    = np.asarray(state_test)
        ports_test_arr = ports_test.values

        for t in range(len(ports_test)):
            if (t + 1) % 20 == 0 or t == 0:
                print(f"    test month {t+1}/{len(ports_test)}", flush=True)

            s       = float(_state_test[t])
            monthly = _one_month_lars(
                ports_train_arr, _state_train, s, kernel,
                lambda0, lambda2, adj_w, kmin, kmax)

            for (i, j, k), sdf_w in monthly.items():
                test_returns[(i, j)][k].append(float(ports_test_arr[t] @ sdf_w))

    # ---- Save one compact CSV per (lambda0, lambda2) ----
    for i in range(len(lambda0)):
        for j in range(len(lambda2)):
            key = (i, j)

            if ports_valid is not None:
                # CV fold: only validation SRs
                all_k = sorted(valid_returns[key].keys())
            else:
                # Full fit: only test SRs
                all_k = sorted(test_returns[key].keys())

            if not all_k:
                continue

            rows = []
            for k in all_k:
                row = {'portsN': k}

                if ports_valid is not None:
                    # CV fold: only valid_SR
                    v   = np.array(valid_returns[key][k])
                    std = v.std(ddof=1)
                    row['valid_SR'] = float(v.mean() / std) if std > 0 else np.nan
                else:
                    # Full fit: only test_SR
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


def _one_month_lars(ports_train_arr, state_train_arr, s_current,
                    kernel, lambda0, lambda2, adj_w, kmin, kmax):
    """
    For one prediction month with current state s_current:
        compute kernel weights
        kernel-weighted mu
        thin SVD of weighted residual matrix (faster than eigh on NxN cov)
        apply Bessel correction to singular values
        run LARS for all (lambda0, lambda2) combinations

    Returns
    -------
    dict : (l0_idx, l2_idx, k) -> sdf_weights (N,)
        sdf_weights : dot with that month's return vector -> one SDF return
        Only the last LARS solution per k is kept (same as original).
    """
    T, N = ports_train_arr.shape

    # Kernel weights and weighted mean
    w      = kernel.weights(state_train_arr, s_current)  # (T,) sums to 1
    mu     = (w[:, None] * ports_train_arr).sum(axis=0)  # (N,)
    mu_bar = mu.mean()

    # Weighted residuals for SVD
    resid    = ports_train_arr - mu[None, :]              # (T, N)
    sqrt_w   = np.sqrt(w)                                  # (T,)
    weighted = sqrt_w[:, None] * resid                     # (T, N)

    # Thin SVD: cost O(T²N) vs O(N³) for eigh — much faster when T < N
    U, s_raw, Vt = np.linalg.svd(weighted, full_matrices=False)

    # Bessel correction: raw cov eigenvalues = s_raw²
    # Corrected eigenvalues = s_raw² / (1 - Σw²)
    # So corrected singular values = s_raw / sqrt(1 - Σw²)
    bessel = 1.0 - np.sum(w ** 2)
    if bessel > 0:
        s = s_raw / np.sqrt(bessel)
    else:
        s = s_raw

    mask = s > 1e-10
    s, Vt = s[mask], Vt[mask]
    gamma = len(s)

    if gamma == 0:
        return {}

    V = Vt.T                                               # (N, gamma)
    sigma_tilde = V @ np.diag(s) @ V.T                    # (N, N) — sqrt of cov
    inv_sqrt    = V @ np.diag(1.0 / s) @ V.T              # (N, N) — inv sqrt of cov

    results = {}

    for i, l0 in enumerate(lambda0):
        mu_robust = mu + l0 * mu_bar        # (N,)
        mu_tilde  = inv_sqrt @ mu_robust     # (N,)

        for j, l2 in enumerate(lambda2):
            beta_subset, K_subset = lasso(
                sigma_tilde, mu_tilde, l2, 100, kmin, kmax)

            if beta_subset.shape[0] == 0:
                continue

            for r in range(beta_subset.shape[0]):
                k = int(K_subset[r])
                b = beta_subset[r] * adj_w
                b = b / np.abs(b.sum())
                w_sdf = b / adj_w
                results[(i, j, k)] = w_sdf  # only sdf_weights, no betas

    return results


# ---------------------------------------------------------------------------
# Phase 2: kernel reconstruction — rerun winning combo, save per-month detail
# ---------------------------------------------------------------------------

def kernel_reconstruct(ports, adj_w, lambda0_star, lambda2_star,
                       kernel, state, n_train_valid=360,
                       kmin=5, kmax=50, output_dir=None):
    """
    Phase 2: After best (lambda0*, lambda2*, h*) are selected, rerun the
    kernel loop for that single combination and save full per-month detail.

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
            One row per (month, k) — the actual SDF return for that month

        reconstruction_monthly_betas.csv
            Columns: month_idx, k, port_col_1, port_col_2, ..., port_col_N
            One row per (month, k) — the actual beta vector for that month
            These are the depth-adjusted betas (b = beta_raw * adj_w, normalized)

    Returns
    -------
    dict with keys:
        'sdf_returns' : dict k -> np.array of monthly SDF returns (T_test,)
        'monthly_betas' : dict k -> list of (N,) beta vectors, one per test month
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
    N      = ports_train_arr.shape[1]

    # Wrap scalars in lists for _one_month_lars interface
    lambda0_list = [lambda0_star]
    lambda2_list = [lambda2_star]

    # Accumulators
    sdf_returns   = defaultdict(list)      # k -> list of floats
    monthly_betas = defaultdict(list)      # k -> list of (N,) arrays

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

        # SVD of weighted residuals (faster than eigh on NxN cov)
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

        # Process each k from the LARS path
        seen_k = {}
        for r in range(beta_subset.shape[0]):
            k = int(K_subset[r])
            b = beta_subset[r] * adj_w
            b = b / np.abs(b.sum())
            w = b / adj_w
            seen_k[k] = (w, b)   # last solution per k wins (same as Phase 1)

        for k, (w, b) in seen_k.items():
            sdf_ret = float(ports_test_arr[t] @ w)
            sdf_returns[k].append(sdf_ret)
            monthly_betas[k].append(b)

            sdf_rows.append({'month_idx': t, 'k': k, 'sdf_return': sdf_ret})
            beta_row = {'month_idx': t, 'k': k}
            for col, val in zip(port_cols, b):
                beta_row[col] = val
            beta_rows.append(beta_row)

    # Save to disk
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


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def compute_moments(returns: np.ndarray,
                    kernel: BaseKernel,
                    state_train: np.ndarray,
                    state_current: float):
    """
    Compute kernel-weighted mean and covariance of portfolio returns.

    With UniformKernel: exactly reproduces np.mean(axis=0) and np.cov
                        (Bessel-corrected, dividing by T-1).
    With GaussianKernel: months whose state is close to state_current
                         receive higher weight. Covariance uses the
                         effective-sample-size correction 1/(1 - Σw²).

    Parameters
    ----------
    returns       : (T_train, N) excess returns
    kernel        : any BaseKernel instance
    state_train   : (T_train,) state variable values for training months
    state_current : float — S_{t*-1}, the current state value

    Returns
    -------
    mu    : (N,)   kernel-weighted mean return
    sigma : (N, N) kernel-weighted covariance matrix (Bessel-corrected)
    """
    w     = kernel.weights(state_train, state_current)  # (T_train,) sums to 1
    mu    = (w[:, None] * returns).sum(axis=0)           # (N,)
    resid = returns - mu[None, :]                         # (T_train, N)

    # Raw weighted covariance: Σ w_t (R_t - μ)(R_t - μ)ᵀ
    sigma_raw = (w[:, None] * resid).T @ resid            # (N, N)

    # Bessel-like correction: divide by (1 - Σw²) instead of 1.
    # For uniform weights w_t = 1/T: Σw² = T·(1/T)² = 1/T,
    # so correction = 1/(1 - 1/T) = T/(T-1), giving sigma_raw * T/(T-1)
    # which matches np.cov (ddof=1).
    # For non-uniform kernel weights: this uses the effective sample size
    # n_eff = 1/Σw², giving correction n_eff/(n_eff - 1).
    bessel = 1.0 - np.sum(w ** 2)
    if bessel > 0:
        sigma = sigma_raw / bessel
    else:
        sigma = sigma_raw  # fallback: single point, no correction possible

    return mu, sigma
>>>>>>> 9458664 (store sharpe ratio for each h,lambda0,2,k combo correct)
=======
"""
lasso_valid_par_full.py — Grid search orchestration and routing.

Thin entry point that loops over bandwidths and CV folds, routing each
call to the appropriate implementation:
    - UniformKernel  → lasso_uniform.static_cv_helper (original code)
    - Non-uniform    → lasso_kernel_validation.kernel_cv_helper (per-month)

For non-uniform kernels, the full fit is NOT run during the grid search.
Call kernel_full_fit() separately after selecting the winning hyperparameters.

Functions
---------
lasso_valid_full : orchestrate CV folds across all (lambda0, lambda2, h)
lasso_cv_helper  : route one call to uniform or kernel path
"""

import os

import pandas as pd

from .kernels.uniform import UniformKernel
from .lasso_uniform import static_cv_helper
from .lasso_kernel_validation import kernel_cv_helper


def lasso_valid_full(ports, lambda0, lambda2, main_dir, sub_dir, adj_w,
                     n_train_valid=360, cvN=3, runFullCV=False,
                     kmin=5, kmax=50, RunParallel=False, ParallelN=10,
                     kernel_cls=None, bandwidths=None, state=None):
    """
    Orchestrate CV folds (and full fit for uniform) across all (lambda0, lambda2, h).

    For UniformKernel: runs both CV folds and full fit (cheap — one-shot).
    For non-uniform kernels: runs only CV folds. The full fit is deferred
        to kernel_full_fit() after the winning (l0*, l2*, h*) are selected.

    Parameters
    ----------
    kernel_cls  : kernel class. Defaults to UniformKernel.
    bandwidths  : list of h values. UniformKernel: [None].
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

    is_uniform = issubclass(kernel_cls, UniformKernel)

    for h_idx, h in enumerate(bandwidths, start=1):

        if is_uniform:
            kernel = UniformKernel()
        else:
            kernel = kernel_cls(h=h)

        print(f"  bandwidth {h_idx}/{len(bandwidths)}: {kernel}", flush=True)

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

        # Full fit: only for uniform (cheap). Kernel defers to kernel_full_fit.
        if is_uniform:
            state_train_full = None if state is None else state.iloc[:n_train_valid]
            lasso_cv_helper(
                ports.iloc[:n_train_valid], None, ports_test, lambda0, lambda2,
                main_dir, sub_dir, adj_w, 'full', kmin, kmax,
                RunParallel, ParallelN,
                kernel=kernel, h_idx=h_idx,
                state_train=state_train_full, state_valid=None,
                state_test=state_test)

    if not is_uniform:
        print(f"  Grid search done. Run kernel_full_fit() for the winning combo.",
              flush=True)


def lasso_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
                    main_dir, sub_dir, adj_w, cv_name, kmin=5, kmax=50,
                    RunParallel=False, ParallelN=10,
                    kernel=None, h_idx=1,
                    state_train=None, state_valid=None, state_test=None):
    """
    Route one call to the uniform or kernel path.
    """
    if kernel is None:
        kernel = UniformKernel()

    if isinstance(kernel, UniformKernel):
        static_cv_helper(
            ports_train, ports_valid, ports_test, lambda0, lambda2,
            main_dir, sub_dir, adj_w, cv_name, kmin, kmax,
            RunParallel, ParallelN, h_idx=h_idx)
    else:
        kernel_cv_helper(
            ports_train, ports_valid, ports_test, lambda0, lambda2,
            main_dir, sub_dir, adj_w, cv_name, kmin, kmax,
            kernel=kernel, h_idx=h_idx,
            state_train=state_train, state_valid=state_valid,
            state_test=state_test)
>>>>>>> b8e3a07 (refactor because file got big)
