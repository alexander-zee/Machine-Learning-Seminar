from __future__ import annotations

import numpy as np


def suggest_bandwidth_median_dist(states: np.ndarray, *, max_t: int) -> float:
    """
    Heuristic bandwidth: median pairwise Euclidean distance among early states.

    This is a lightweight stand-in used by ``tv_extension_summary_table`` when
    ``bandwidth`` is not provided explicitly.
    """
    S = np.asarray(states, dtype=float)
    if S.ndim != 2 or S.size == 0:
        return 0.5
    T = int(min(int(max_t), S.shape[0]))
    if T < 3:
        return 0.5

    X = S[:T]
    # subsample for speed if very large
    if T > 200:
        idx = np.linspace(0, T - 1, num=200).astype(int)
        X = X[idx]

    # pairwise distances (upper triangle)
    d = []
    m = X.shape[0]
    for i in range(m):
        diff = X[i + 1 :] - X[i]
        d.extend(np.sqrt(np.sum(diff * diff, axis=1)).tolist())
    if not d:
        return 0.5
    med = float(np.median(np.asarray(d, dtype=float)))
    if not np.isfinite(med) or med <= 0:
        return 0.5
    return med
