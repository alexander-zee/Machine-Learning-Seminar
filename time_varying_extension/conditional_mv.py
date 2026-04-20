"""
Conditional mean–variance helpers for the time-varying extension bundle.

This module used to be missing from some working trees, which broke imports like::

    from .conditional_mv import mean_variance_weights, rolling_kernel_mv_weights

The implementations here are intentionally **small and robust** (ridge-regularized
inverse, nonnegative projection) and are written to satisfy the unit tests in
``tests/test_research_figures.py`` while keeping ``research_figures.py`` usable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def mean_variance_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    Classic (long-only projected) mean–variance tangency weights:

        w ∝ Σ^{-1} μ, then clip negatives and L1-normalize.

    Parameters
    ----------
    mu:
        Shape (p,) expected excess returns.
    Sigma:
        Shape (p, p) covariance matrix.
    ridge:
        Diagonal jitter added to Σ for numerical stability.
    """
    mu_v = np.asarray(mu, dtype=float).reshape(-1)
    S = np.asarray(Sigma, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("Sigma must be a square matrix")
    if mu_v.size != S.shape[0]:
        raise ValueError("mu and Sigma have incompatible shapes")

    p = int(mu_v.size)
    if p == 0:
        return mu_v

    S2 = S + float(ridge) * np.eye(p, dtype=float)
    # Solve S w_raw = mu (symmetric PSD-ish after ridge)
    try:
        w_raw = np.linalg.solve(S2, mu_v)
    except np.linalg.LinAlgError:
        w_raw = np.linalg.lstsq(S2, mu_v, rcond=None)[0]

    # Long-only projection: drop shorts, renormalize to sum to 1.
    w = np.maximum(w_raw, 0.0)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        # Fallback: equal weight if everything collapsed.
        return np.ones(p, dtype=float) / float(p)
    w /= s
    return w


def rolling_kernel_mv_weights(
    R: np.ndarray,
    states: np.ndarray,
    *,
    bandwidth: float,
    min_train: int,
    ridge_sigma: float,
    store_kernels: bool = False,
    use_time_decay: bool = False,
    time_window_m: int | None = None,
    time_decay_lambda: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[np.ndarray] | None]:
    """
    Rolling, **lag-1 tradable** mean–variance weights using a Gaussian kernel on states.

    For each month t, weights w_t are computed using only information in months < t:

    - analogue weights over past months τ < t based on ||s_{t-1} - s_τ|| in state space
    - optional exponential time decay within a finite window (``use_time_decay``)

    Returns
    -------
    W : np.ndarray, shape (T, p)
        Weight matrix (rows sum to ~1; early rows are NaN until ``min_train``).
    mu_path : None
        Reserved for richer diagnostics (unused by current callers).
    ess : np.ndarray, shape (T,)
        A simple effective-sample-size proxy: 1 / sum_k k^2.
    kernels : list[np.ndarray] | None
        If ``store_kernels``, a length-T list where entry t is length t (or empty).
    """
    R = np.asarray(R, dtype=float)
    states = np.asarray(states, dtype=float)
    if R.ndim != 2:
        raise ValueError("R must be 2D (T, p)")
    if states.ndim != 2:
        raise ValueError("states must be 2D (T, d_state)")
    T, p = R.shape
    if len(states) != T:
        raise ValueError("R and states must have same length T")

    min_train = int(max(2, min_train))
    bw = float(bandwidth)
    if not np.isfinite(bw) or bw <= 0:
        raise ValueError("bandwidth must be a positive finite float")

    W = np.full((T, p), np.nan, dtype=float)
    ess = np.full(T, np.nan, dtype=float)
    kernels: list[np.ndarray] | None = [] if store_kernels else None

    # Precompute squared bandwidth once (Gaussian analogue weights).
    denom = 2.0 * (bw**2)

    for t in range(T):
        if store_kernels is True and kernels is not None:
            kernels.append(np.array([], dtype=float))

        if t < min_train:
            continue

        # Information set for month t is through t-1 (lag-1 tradability).
        t_end = t  # exclusive end for past data
        if t_end < 2:
            continue

        R_past = R[:t_end, :]
        S_past = states[:t_end, :]
        s_ref = states[t - 1, :]  # conditioning state (known at start of month t)

        # Distances in state space (Gaussian kernel).
        diff = S_past - s_ref.reshape(1, -1)
        d2 = np.sum(diff * diff, axis=1)
        k = np.exp(-d2 / denom)

        if use_time_decay:
            m = int(time_window_m) if time_window_m is not None else int(min_train)
            lam = float(time_decay_lambda) if time_decay_lambda is not None else 0.9
            if m < 1 or not (0.0 < lam < 1.0):
                raise ValueError("use_time_decay requires time_window_m>=1 and 0<lambda<1")

            ages = (t_end - 1) - np.arange(t_end)  # 0 for most recent past month
            mask = ages < m
            decay = np.zeros_like(k, dtype=float)
            decay[mask] = np.power(lam, ages[mask].astype(float))
            k = k * decay

        # Normalize kernel weights over the past.
        sk = float(np.sum(k))
        if not np.isfinite(sk) or sk <= 0:
            k_norm = np.ones(t_end, dtype=float) / float(t_end)
        else:
            k_norm = k / sk

        # Weighted mean/cov of past returns under k_norm.
        wcol = k_norm.reshape(-1, 1)
        mu_w = np.sum(wcol * R_past, axis=0)
        Xc = R_past - mu_w.reshape(1, -1)
        Sigma_w = (wcol * Xc).T @ Xc
        # Bessel-like stability: if nearly rank-1, ridge helps a lot.
        Sigma_w = 0.5 * (Sigma_w + Sigma_w.T)

        w_t = mean_variance_weights(mu_w, Sigma_w, ridge=float(ridge_sigma))
        W[t, :] = w_t

        # ESS proxy (>=1): inverse Herfindahl of normalized kernel masses.
        ess_t = 1.0 / float(np.sum(k_norm * k_norm)) if np.isfinite(np.sum(k_norm * k_norm)) else float("nan")
        ess[t] = ess_t

        if store_kernels is True and kernels is not None:
            kernels[-1] = k_norm.astype(float)

    return W, None, ess, kernels
