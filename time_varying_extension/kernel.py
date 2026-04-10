"""
Kernel weights over past time indices (historical analogues).

- Gaussian (RBF) on continuous state similarity.
- Optional exponential decay in calendar age j = t − τ (one-sided time kernel).

Dual-style diagnostics: effective number of analogues, weight turnover.
"""

from __future__ import annotations

import numpy as np


def gaussian_kernel_unnormalized(
    states: np.ndarray,
    t: int,
    bandwidth: float,
    min_train: int = 36,
) -> np.ndarray:
    """
    Unnormalized Gaussian weights K_G(||s_t - s_τ||) for τ = 0..t-1.
    Log-sum stabilization for numerical stability.
    """
    w = np.full(t, np.nan)
    if t < min_train or t < 1 or bandwidth <= 0:
        return w
    diff = states[:t] - states[t]
    dist2 = np.sum(diff * diff, axis=1)
    logits = -0.5 * dist2 / (bandwidth ** 2)
    logits -= np.max(logits)
    k = np.exp(logits)
    return k


def gaussian_kernel_weights(
    states: np.ndarray,
    t: int,
    bandwidth: float,
    min_train: int = 36,
) -> np.ndarray:
    """
    Normalized weights w_0..w_{t-1} proportional to K(s_t, s_tau).

    Uses only tau < t (no look-ahead). Rows of ``states`` align with time.

    Parameters
    ----------
    states : (T, d) array
    t : current index in [0, T-1]
    bandwidth : scalar h > 0 (Euclidean distance scale)
    min_train : require t >= min_train else returns NaN vector
    """
    k = gaussian_kernel_unnormalized(states, t, bandwidth, min_train)
    if not np.all(np.isfinite(k)):
        return k
    s = k.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full(t, np.nan)
    return k / s


def exponential_time_weights_unnormalized(
    t: int,
    time_window_m: int,
    time_decay_lambda: float,
) -> np.ndarray:
    """
    Unnormalized one-sided time kernel: proportional to λ^j 1{1 <= j < m},
    with j = t - τ the age in months between past index τ and current t.

    Matches the usual geometric decay story: smaller j (more recent) gets larger weight.
    Requires ``time_window_m >= 2`` so that some j in [1, m-1] exist.
    """
    w = np.zeros(t)
    if t < 1 or time_window_m < 2:
        return w
    if not (0.0 < time_decay_lambda < 1.0):
        return w
    j = t - np.arange(t, dtype=int)
    mask = (j >= 1) & (j < time_window_m)
    w[mask] = time_decay_lambda ** j[mask]
    return w


def combined_analogue_kernel_weights(
    states: np.ndarray,
    t: int,
    bandwidth: float,
    min_train: int = 36,
    use_time_decay: bool = False,
    time_window_m: int = 120,
    time_decay_lambda: float = 0.95,
) -> np.ndarray:
    """
    Normalized weights w_τ ∝ K_G(state) · K_E(time) when ``use_time_decay``;
    otherwise same as ``gaussian_kernel_weights``.

    K_E uses unnormalized λ^j on the support 1 <= j < m; global normalization
    follows from the product with K_G.
    """
    k_g = gaussian_kernel_unnormalized(states, t, bandwidth, min_train)
    if not np.all(np.isfinite(k_g)):
        return k_g
    if not use_time_decay:
        s = k_g.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full(t, np.nan)
        return k_g / s
    k_e = exponential_time_weights_unnormalized(t, time_window_m, time_decay_lambda)
    prod = k_g * k_e
    s = prod.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full(t, np.nan)
    return prod / s


def suggest_bandwidth_median_dist(states: np.ndarray, max_t: int, sample_step: int = 3) -> float:
    """
    Simple rule: median pairwise distance among a subsample of past states up to max_t.
    """
    idx = np.arange(0, min(max_t, len(states)), sample_step)
    if len(idx) < 5:
        return 1.0
    S = states[idx]
    # pairwise distances on a smaller subsample if needed
    m = min(len(idx), 80)
    S = S[:m]
    d2 = []
    for i in range(m):
        diff = S - S[i]
        d2.append(np.sqrt(np.sum(diff * diff, axis=1)))
    d2 = np.concatenate(d2)
    d2 = d2[d2 > 1e-12]
    if len(d2) == 0:
        return 1.0
    med = float(np.median(d2))
    return max(med, 1e-8)


def effective_sample_size(weights: np.ndarray) -> float:
    """1 / sum w_i^2 when weights sum to 1 (effective number of analogues)."""
    w = weights[np.isfinite(weights)]
    if w.size == 0 or w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    return float(1.0 / np.dot(w, w))


def total_variation_distance(w1: np.ndarray, w2: np.ndarray) -> float:
    """0.5 * sum |w1 - w2| for nonnegative vectors (pad with zeros if lengths differ)."""
    n = max(len(w1), len(w2))
    a = np.zeros(n)
    b = np.zeros(n)
    a[: len(w1)] = np.nan_to_num(w1, nan=0.0)
    b[: len(w2)] = np.nan_to_num(w2, nan=0.0)
    return float(0.5 * np.sum(np.abs(a - b)))
