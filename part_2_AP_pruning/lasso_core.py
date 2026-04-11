"""
lasso_core.py — Shared computational building blocks for AP-Pruning.

Used by both the kernel validation loop and the kernel full fit / reconstruction.

Functions
---------
one_month_lars   : kernel weights → SVD → LARS for one prediction month
compute_moments  : kernel-weighted mean and covariance (Bessel-corrected)
"""

import numpy as np

from .lasso import lasso
from .kernels.base import BaseKernel


def one_month_lars(ports_train_arr, state_train_arr, s_current,
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

    # Weighted residuals
    resid    = ports_train_arr - mu[None, :]              # (T, N)
    sqrt_w   = np.sqrt(w)                                  # (T,)
    weighted = sqrt_w[:, None] * resid                     # (T, N)

    # Thin SVD — fast when T < N. Falls back to eigh on the weighted
    # covariance if SVD fails to converge (e.g. near-degenerate kernel weights).
    bessel = 1.0 - np.sum(w ** 2)

    try:
        U, s_raw, Vt = np.linalg.svd(weighted, full_matrices=False)

        # Bessel correction on singular values
        s = s_raw / np.sqrt(bessel) if bessel > 0 else s_raw

        mask = s > 1e-10
        s, Vt = s[mask], Vt[mask]
        gamma = len(s)
        if gamma == 0:
            return {}

        V           = Vt.T
        sigma_tilde = V @ np.diag(s) @ V.T
        inv_sqrt    = V @ np.diag(1.0 / s) @ V.T

    except np.linalg.LinAlgError:
        # SVD did not converge — fall back to eigh on the weighted covariance.
        # Results are equivalent; eigh is more stable for near-degenerate matrices.
        sigma_raw = (w[:, None] * resid).T @ resid          # (N, N)
        sigma     = sigma_raw / bessel if bessel > 0 else sigma_raw

        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        idx          = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        gamma = int(np.sum(eigenvalues > 1e-10))
        if gamma == 0:
            return {}

        D, V        = eigenvalues[:gamma], eigenvectors[:, :gamma]
        sigma_tilde = V @ np.diag(np.sqrt(D)) @ V.T
        inv_sqrt    = V @ np.diag(1.0 / np.sqrt(D)) @ V.T

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
                results[(i, j, k)] = w_sdf

    return results


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