"""
Supervisor-style summaries: risk-premium drivers, split locations, top combinations.

Complements dual-interpretation diagnostics (kernel concentration, turnover).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kernel import total_variation_distance
from .tree_column_parse import parse_portfolio_column


def time_varying_risk_premium_scores(
    states: np.ndarray,
    mu_cond: np.ndarray,
    w_mv: np.ndarray,
    state_names: tuple[str, ...] = ("LME (vw)", "feat1 (vw)", "feat2 (vw)"),
) -> pd.DataFrame:
    """
    Scalar conditional premium proxy: rp_t = w_mv_t' mu_cond_t (when finite).

    Scores which state coordinate aligns most with *time variation* in rp_t
    via absolute correlation (simple, robust).
    """
    T = states.shape[0]
    rp = np.array(
        [
            float(np.dot(w_mv[t], mu_cond[t]))
            if np.all(np.isfinite(w_mv[t])) and np.all(np.isfinite(mu_cond[t]))
            else np.nan
            for t in range(T)
        ]
    )
    rows = []
    rp_c = rp - np.nanmean(rp)
    for j, name in enumerate(state_names):
        s = states[:, j]
        sc = s - np.nanmean(s)
        num = np.nansum(rp_c * sc)
        den = np.sqrt(np.nansum(rp_c**2) * np.nansum(sc**2))
        corr = float(num / den) if den > 1e-16 else float("nan")
        rows.append(
            {
                "state_variable": name,
                "abs_corr_with_rp": abs(corr),
                "corr_with_rp": corr,
                "std_state": float(np.nanstd(s)),
                "std_rp_contribution_proxy": abs(corr) * float(np.nanstd(rp)),
            }
        )
    df = pd.DataFrame(rows).sort_values("abs_corr_with_rp", ascending=False)
    return df


def weight_mass_by_split_depth(
    portfolio_columns: list[str],
    mean_abs_weights: np.ndarray,
) -> pd.DataFrame:
    """
    Where the tree puts mass: average |w| grouped by parsed depth.
    """
    depths = []
    for c, aw in zip(portfolio_columns, mean_abs_weights):
        d = parse_portfolio_column(c).depth
        if d < 0:
            continue
        depths.append((d, aw))
    df = pd.DataFrame(depths, columns=["depth", "mean_abs_w"])
    g = df.groupby("depth", as_index=False)["mean_abs_w"].sum().sort_values("depth")
    g.rename(columns={"mean_abs_w": "sum_mean_abs_w"}, inplace=True)
    return g


def top_weight_co_movements(w_mv: np.ndarray, columns: list[str], top_n: int = 10) -> pd.DataFrame:
    """
    Top ``top_n`` unordered pairs (i,j) by time-mean |w_i(t) w_j(t)|.
    """
    T, p = w_mv.shape
    scores = []
    for i in range(p):
        for j in range(i + 1, p):
            prod = w_mv[:, i] * w_mv[:, j]
            m = np.nanmean(np.abs(prod))
            scores.append((m, i, j))
    scores.sort(reverse=True, key=lambda x: x[0])
    out = []
    for rank, (m, i, j) in enumerate(scores[:top_n], start=1):
        out.append(
            {
                "rank": rank,
                "portfolio_i": columns[i],
                "portfolio_j": columns[j],
                "mean_abs_w_i_w_j": m,
            }
        )
    return pd.DataFrame(out)


def top_kernel_turnover_months(
    kernel_weight_history: list[np.ndarray],
    dates_yyyymm: list[int] | np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    ``kernel_weight_history[t]`` = weights on 0..t-1 (length t).
    TV distance between consecutive weight vectors (padded).
    """
    rows = []
    prev = None
    for t, w in enumerate(kernel_weight_history):
        if prev is not None and prev.size > 0 and w.size > 0:
            tv = total_variation_distance(prev, w)
            rows.append({"t_index": t, "yyyymm": dates_yyyymm[t], "kernel_tv_from_prev": tv})
        prev = w
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.nlargest(top_n, "kernel_tv_from_prev")


def dual_style_kernel_metrics(
    kernel_weight_history: list[np.ndarray],
) -> pd.DataFrame:
    """
    Time series of effective analogue count 1/sum w^2 (paper-style concentration).
    """
    from .kernel import effective_sample_size

    rows = []
    for t, w in enumerate(kernel_weight_history):
        if w.size == 0:
            rows.append({"t_index": t, "eff_analogues": float("nan")})
        else:
            rows.append({"t_index": t, "eff_analogues": effective_sample_size(w)})
    return pd.DataFrame(rows)
