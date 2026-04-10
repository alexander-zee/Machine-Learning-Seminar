"""
Kernel-weighted conditional mean and covariance; ridge mean–variance weights.
"""

from __future__ import annotations

import numpy as np

from .kernel import combined_analogue_kernel_weights, effective_sample_size


def weighted_mean_cov(
    R: np.ndarray,
    w: np.ndarray,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    R : (n, p) past returns, w : (n,) nonnegative, sum to 1.
    Returns mu (p,), Sigma (p, p) PSD + ridge*I.
    """
    _, p = R.shape
    w = np.nan_to_num(w, nan=0.0)
    sw = w.sum()
    if sw <= 0:
        return np.full(p, np.nan), np.full((p, p), np.nan)
    w = w / sw
    mu = w @ R
    X = R - mu
    Sigma = (X.T * w) @ X
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma += ridge * np.eye(p)
    return mu, Sigma


def mean_variance_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    ridge: float = 1e-4,
) -> np.ndarray:
    """
    w ∝ Sigma^{-1} mu (tangency), then L1-normalized to sum(|w|)=1.
    """
    p = len(mu)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(Sigma)):
        return np.full(p, np.nan)
    S = Sigma + ridge * np.eye(p)
    try:
        w = np.linalg.solve(S, mu)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(S, mu, rcond=None)
    s = np.sum(np.abs(w))
    if s <= 1e-16:
        return np.full(p, np.nan)
    return w / s


def rolling_kernel_mv_weights(
    R: np.ndarray,
    states: np.ndarray,
    bandwidth: float,
    min_train: int,
    ridge_sigma: float,
    store_kernels: bool = False,
    use_time_decay: bool = False,
    time_window_m: int = 120,
    time_decay_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray] | None]:
    """
    For each t, analogue kernel weights on 0..t-1 (Gaussian × optional exponential time),
    then mu_t, w_mv_t.

    Returns
    -------
    W_mv : (T, p)
    MU_cond : (T, p)
    ess : (T,) effective number of kernel analogues
    kernels : list of length T (or None) — w_t on past, variable lengths
    """
    T, p = R.shape
    W = np.full((T, p), np.nan)
    MU = np.full((T, p), np.nan)
    ess = np.full(T, np.nan)
    kernels: list[np.ndarray] | None = [] if store_kernels else None
    for t in range(T):
        wk = combined_analogue_kernel_weights(
            states,
            t,
            bandwidth,
            min_train,
            use_time_decay=use_time_decay,
            time_window_m=time_window_m,
            time_decay_lambda=time_decay_lambda,
        )
        if not np.all(np.isfinite(wk)):
            if kernels is not None:
                kernels.append(np.array([]))
            continue
        R_past = R[:t]
        if R_past.shape[0] != len(wk):
            if kernels is not None:
                kernels.append(np.array([]))
            continue
        mu_t, Sig_t = weighted_mean_cov(R_past, wk, ridge=ridge_sigma * 0.1)
        w_mv = mean_variance_weights(mu_t, Sig_t, ridge=ridge_sigma)
        W[t] = w_mv
        MU[t] = mu_t
        ess[t] = effective_sample_size(wk)
        if kernels is not None:
            kernels.append(np.copy(wk))
    return W, MU, ess, kernels
