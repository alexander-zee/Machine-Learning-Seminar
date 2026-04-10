"""
lasso_uniform.py — Static (UniformKernel) estimation path.

Original code from Bryzgalova et al. (2025), logic completely unchanged.
mu/sigma computed once, eigendecomposed once, LARS run once per (lambda0, lambda2).
One CSV per (l0, l2) with rows per k, full beta columns.

Only cosmetic change: h_idx appears in the CSV filename for consistency
with the kernel path.

Functions
---------
static_cv_helper  : one-shot estimation for one fold or full fit
run_one_lambda0   : inner LARS loop for one fixed lambda0 index
"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .lasso import lasso


def static_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
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
            delayed(run_one_lambda0)(i, *args) for i in range(len(lambda0)))
    else:
        for i in range(len(lambda0)):
            run_one_lambda0(i, *args)


def run_one_lambda0(i, lambda0, lambda2, sigma_tilde, mu_tilde,
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