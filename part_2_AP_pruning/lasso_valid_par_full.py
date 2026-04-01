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
from lasso import lasso


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