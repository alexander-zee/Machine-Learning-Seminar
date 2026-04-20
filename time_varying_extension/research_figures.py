"""
OOS performance and research figures: static tangency vs TV (Gaussian; Gaussian × time decay).

Baseline static: sample mean–variance tangency on R[0:n_train_valid), held fixed on the test window.
TV strategies: lag-1 weights w_{t-1}' r_t (weights use only past returns in kernel construction).

OOS Sharpe is **monthly** (mean / std of monthly excess returns on the OOS mask), aligned with the
AP grid convention; ``write_research_figure_bundle`` also writes ``tv_oos_strategy_factor_table.csv``
(CAPM/FF5 on those same OOS months) for comparison with ``ap_pruned_summary_table`` output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .conditional_mv import mean_variance_weights, rolling_kernel_mv_weights


def static_tangency_weights(
    R: np.ndarray,
    n_fit: int,
    ridge_sigma: float,
) -> np.ndarray:
    """Full-sample (equal-weight) mean and cov on R[0:n_fit), L1-normalized tangency."""
    n_fit = max(2, min(int(n_fit), len(R)))
    sub = R[:n_fit].astype(float)
    mu = np.nanmean(sub, axis=0)
    Sigma = np.cov(sub, rowvar=False, ddof=1)
    if not np.all(np.isfinite(Sigma)):
        Sigma = np.nan_to_num(Sigma, nan=0.0)
    return mean_variance_weights(mu, Sigma, ridge=ridge_sigma)


def realized_returns_lag_weights(
    R: np.ndarray,
    W: np.ndarray,
    lag: int = 1,
) -> np.ndarray:
    """
    r_t = w_{t-lag}^T r_t for month t (predictive / tradable timing).

    lag=1: weight known from information through t-1 earns return in t.
    """
    T, p = R.shape
    out = np.full(T, np.nan)
    for t in range(lag, T):
        w = W[t - lag]
        if np.all(np.isfinite(w)) and np.all(np.isfinite(R[t])):
            out[t] = float(np.dot(w, R[t]))
    return out


def static_hold_returns(R: np.ndarray, w: np.ndarray, t_start: int) -> np.ndarray:
    """r_t = w^T r_t for t >= t_start; NaN before."""
    T = len(R)
    out = np.full(T, np.nan)
    for t in range(t_start, T):
        if np.all(np.isfinite(w)) and np.all(np.isfinite(R[t])):
            out[t] = float(np.dot(w, R[t]))
    return out


def monthly_sharpe_ratio(returns: np.ndarray, periods_per_year: float = 12.0) -> float:
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    if sd < 1e-16:
        return float("nan")
    return float(np.sqrt(periods_per_year) * np.mean(x) / sd)


def monthly_sharpe_simple_masked(returns: np.ndarray, mask: np.ndarray) -> float:
    """Monthly Sharpe = mean / std of monthly excess returns (same convention as AP grid test_SR)."""
    x = np.asarray(returns, dtype=float)[mask]
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd < 1e-16:
        return float("nan")
    return float(np.mean(x) / sd)


def oos_return_stats(returns: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Descriptive stats on masked OOS months (sanity-check implied Sharpe)."""
    x = np.asarray(returns, dtype=float)[mask]
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {
            "n_months": int(x.size),
            "mean_monthly": None,
            "std_monthly": None,
            "sharpe_monthly": None,
            "sharpe_annualized_sqrt12": None,
        }
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    sm = float(mu / sd) if sd > 1e-16 else float("nan")
    return {
        "n_months": int(x.size),
        "mean_monthly": mu,
        "std_monthly": sd,
        "sharpe_monthly": sm,
        "sharpe_annualized_sqrt12": float(np.sqrt(12.0) * sm) if np.isfinite(sm) else None,
    }


def oos_mask(T: int, n_train_valid: int, min_train_months: int, lag: int = 1) -> np.ndarray:
    """Boolean mask for months used in OOS evaluation."""
    t0 = max(int(n_train_valid), int(min_train_months) + int(lag))
    m = np.zeros(T, dtype=bool)
    m[t0:T] = True
    return m


def kernel_weight_matrix(kernels: list[np.ndarray], T: int) -> np.ndarray:
    """
    H[t, tau] = normalized analogue weight on past month tau when current index is t;
    NaN if tau >= t or missing.
    """
    H = np.full((T, T), np.nan, dtype=float)
    for t in range(T):
        if t >= len(kernels):
            continue
        w = kernels[t]
        if w is None or len(w) == 0:
            continue
        if len(w) != t:
            continue
        H[t, :t] = w
    return H


def pick_representation_months(
    states: np.ndarray,
    test_mask: np.ndarray,
    n: int = 3,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Pick up to ``n`` distinct month indices in the test window: low / mid / high LME (state column 0).
    """
    idx = np.where(test_mask)[0]
    if len(idx) == 0:
        return []
    s0 = states[idx, 0]
    order = np.argsort(s0)
    if len(order) >= 3:
        raw = [int(idx[order[0]]), int(idx[order[len(order) // 2]]), int(idx[order[-1]])]
    elif len(order) == 2:
        raw = [int(idx[order[0]]), int(idx[order[1]])]
    else:
        raw = [int(idx[order[0]])]
    seen: set[int] = set()
    picks: list[int] = []
    for j in raw:
        if j not in seen:
            seen.add(j)
            picks.append(j)
        if len(picks) >= n:
            break
    if len(picks) < n and rng is None:
        rng = np.random.default_rng(0)
    while len(picks) < n and rng is not None:
        c = int(rng.choice(idx))
        if c not in seen:
            seen.add(c)
            picks.append(c)
    return picks[:n]


def compute_research_bundle(
    R: np.ndarray,
    states: np.ndarray,
    bandwidth: float,
    min_train_months: int,
    ridge_sigma: float,
    n_train_valid: int,
    time_window_m: int,
    time_decay_lambda: float,
) -> dict[str, Any]:
    """
    Run static tangency + two TV kernels (state-Gaussian only; + time decay).
    Returns weights, kernels, realized returns, and OOS Sharpe ratios.
    """
    T, p = R.shape
    w_static = static_tangency_weights(R, n_train_valid, ridge_sigma)
    r_static = static_hold_returns(R, w_static, n_train_valid)

    W_g, _, ess_g, ker_g = rolling_kernel_mv_weights(
        R,
        states,
        bandwidth=bandwidth,
        min_train=min_train_months,
        ridge_sigma=ridge_sigma,
        store_kernels=True,
        use_time_decay=False,
    )
    W_gt, _, ess_gt, ker_gt = rolling_kernel_mv_weights(
        R,
        states,
        bandwidth=bandwidth,
        min_train=min_train_months,
        ridge_sigma=ridge_sigma,
        store_kernels=True,
        use_time_decay=True,
        time_window_m=time_window_m,
        time_decay_lambda=time_decay_lambda,
    )

    r_tv_g = realized_returns_lag_weights(R, W_g, lag=1)
    r_tv_gt = realized_returns_lag_weights(R, W_gt, lag=1)

    mask = oos_mask(T, n_train_valid, min_train_months, lag=1)
    sr_static_m = monthly_sharpe_simple_masked(r_static, mask)
    sr_g_m = monthly_sharpe_simple_masked(r_tv_g, mask)
    sr_gt_m = monthly_sharpe_simple_masked(r_tv_gt, mask)
    sr_static_a = float(np.sqrt(12.0) * sr_static_m) if np.isfinite(sr_static_m) else float("nan")
    sr_g_a = float(np.sqrt(12.0) * sr_g_m) if np.isfinite(sr_g_m) else float("nan")
    sr_gt_a = float(np.sqrt(12.0) * sr_gt_m) if np.isfinite(sr_gt_m) else float("nan")

    H_g = kernel_weight_matrix(ker_g or [], T)
    H_gt = kernel_weight_matrix(ker_gt or [], T)

    return {
        "w_static": w_static,
        "W_gaussian": W_g,
        "W_gaussian_time": W_gt,
        "r_static": r_static,
        "r_tv_gaussian": r_tv_g,
        "r_tv_gaussian_time": r_tv_gt,
        "ess_gaussian": ess_g,
        "ess_gaussian_time": ess_gt,
        "kernels_gaussian": ker_g,
        "kernels_gaussian_time": ker_gt,
        "H_gaussian": H_g,
        "H_gaussian_time": H_gt,
        "oos_mask": mask,
        "sharpe_oos_monthly_static": sr_static_m,
        "sharpe_oos_monthly_tv_gaussian": sr_g_m,
        "sharpe_oos_monthly_tv_gaussian_time": sr_gt_m,
        "sharpe_oos_annualized_sqrt12_static": sr_static_a,
        "sharpe_oos_annualized_sqrt12_tv_gaussian": sr_g_a,
        "sharpe_oos_annualized_sqrt12_tv_gaussian_time": sr_gt_a,
        "n_train_valid": n_train_valid,
        "n_oos_months": int(np.sum(mask)),
    }


def write_research_figure_bundle(
    R: np.ndarray,
    states: np.ndarray,
    dates_yyyymm: np.ndarray,
    cols: list[str],
    bandwidth: float,
    min_train_months: int,
    ridge_sigma: float,
    n_train_valid: int,
    time_window_m: int,
    time_decay_lambda: float,
    output_dir: Path,
    feat1: str,
    feat2: str,
) -> dict[str, Any]:
    """Compute bundle, write JSON/CSV, and save all research figures."""
    from . import plots as tv_plots

    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bundle = compute_research_bundle(
        R,
        states,
        bandwidth=bandwidth,
        min_train_months=min_train_months,
        ridge_sigma=ridge_sigma,
        n_train_valid=n_train_valid,
        time_window_m=time_window_m,
        time_decay_lambda=time_decay_lambda,
    )

    T = len(dates_yyyymm)
    mask = bundle["oos_mask"]
    summary = {
        "n_train_valid": n_train_valid,
        "n_oos_months": int(bundle["n_oos_months"]),
        "sharpe_oos_monthly": {
            "static_tangency_train_valid": bundle["sharpe_oos_monthly_static"],
            "tv_kernel_gaussian_state": bundle["sharpe_oos_monthly_tv_gaussian"],
            "tv_kernel_gaussian_x_time_decay": bundle["sharpe_oos_monthly_tv_gaussian_time"],
        },
        "sharpe_oos_annualized_sqrt12": {
            "static_tangency_train_valid": bundle["sharpe_oos_annualized_sqrt12_static"],
            "tv_kernel_gaussian_state": bundle["sharpe_oos_annualized_sqrt12_tv_gaussian"],
            "tv_kernel_gaussian_x_time_decay": bundle["sharpe_oos_annualized_sqrt12_tv_gaussian_time"],
        },
        "oos_diagnostics": {
            "static": oos_return_stats(bundle["r_static"], mask),
            "tv_gaussian": oos_return_stats(bundle["r_tv_gaussian"], mask),
            "tv_gaussian_time": oos_return_stats(bundle["r_tv_gaussian_time"], mask),
        },
        "notes": (
            "Primary Sharpe is monthly: mean/std of monthly excess returns on OOS months (same scale as "
            "AP grid train_SR / test_SR). Annualized analogue: sqrt(12) * monthly SR — see "
            "sharpe_oos_annualized_sqrt12 and oos_diagnostics.sharpe_annualized_sqrt12.\n\n"
            "INTERPRETATION: Large test Sharpe is common here NOT because of a formula bug, but because "
            "(1) the k portfolios are Selected_Ports from AP/LASSO tuned on validation SR, "
            "(2) tangency on 10 anomaly-style portfolios can yield very high in-sample-like test ratios, "
            "(3) no transaction costs. For a thesis, report Ledoit–Wolf or bootstrap CI on SR, "
            "equal-weight baseline on the same k assets, subsamples, and/or net-of-cost returns."
        ),
    }
    (output_dir / "tv_oos_sharpe_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    df_ret = pd.DataFrame(
        {
            "yyyymm": dates_yyyymm[:T],
            "r_static_oos": bundle["r_static"][:T],
            "r_tv_gaussian": bundle["r_tv_gaussian"][:T],
            "r_tv_gaussian_time": bundle["r_tv_gaussian_time"][:T],
            "oos_eval_mask": mask[:T],
        }
    )
    df_ret.to_csv(output_dir / "tv_oos_strategy_returns.csv", index=False)

    try:
        from part_3_metrics_collection.tv_extension_summary_table import (
            build_tv_oos_factor_table_long,
        )

        df_fac = build_tv_oos_factor_table_long(
            bundle,
            dates_yyyymm[:T],
            feat1,
            feat2,
            int(R.shape[1]),
            float(bandwidth),
            int(min_train_months),
            float(ridge_sigma),
            int(n_train_valid),
            int(time_window_m),
            float(time_decay_lambda),
        )
        df_fac.to_csv(output_dir / "tv_oos_strategy_factor_table.csv", index=False)
    except Exception as exc:
        import warnings

        warnings.warn(
            f"tv_oos_strategy_factor_table.csv not written ({exc!r}). "
            "Install statsmodels and ensure Fama–French data can be downloaded.",
            UserWarning,
            stacklevel=1,
        )

    tv_plots.plot_oos_sharpe_comparison(
        [
            "Static tangency\n(train+valid)",
            "TV: Gaussian\n(state)",
            "TV: Gaussian ×\ntime decay",
        ],
        [
            bundle["sharpe_oos_monthly_static"],
            bundle["sharpe_oos_monthly_tv_gaussian"],
            bundle["sharpe_oos_monthly_tv_gaussian_time"],
        ],
        fig_dir / "fig_oos_sharpe_comparison.png",
        title=f"OOS monthly Sharpe (μ/σ) — LME × {feat1} × {feat2}",
    )

    if np.any(mask):
        t_start = int(np.where(mask)[0][0])
        tv_plots.plot_kernel_analogue_heatmap(
            bundle["H_gaussian"],
            dates_yyyymm,
            fig_dir / "fig_kernel_heatmap_gaussian_state.png",
            t_start_row=t_start,
            title="Analogue weights — Gaussian state kernel (normalized w_τ | t)",
        )
        tv_plots.plot_kernel_analogue_heatmap(
            bundle["H_gaussian_time"],
            dates_yyyymm,
            fig_dir / "fig_kernel_heatmap_gaussian_x_time.png",
            t_start_row=t_start,
            title="Analogue weights — Gaussian state × exponential time decay",
        )

        months_pick = pick_representation_months(states, mask, n=3)
        if months_pick:
            tv_plots.plot_kernel_weight_slices(
                bundle["kernels_gaussian"] or [],
                bundle["kernels_gaussian_time"] or [],
                dates_yyyymm,
                months_pick,
                fig_dir / "fig_kernel_weight_slices_rep_months.png",
            )

        tv_plots.plot_mv_weights_over_time(
            dates_yyyymm,
            bundle["W_gaussian_time"],
            cols,
            fig_dir / "fig_mv_weights_test_top10.png",
            title="TV weights (Gaussian × time) — top 10 | test window",
            max_legend_items=10,
            index_range=(t_start, T),
        )

        tv_plots.plot_effective_analogues(
            dates_yyyymm,
            bundle["ess_gaussian_time"],
            fig_dir / "fig_effective_analogues_gaussian_time.png",
            title="Effective analogues 1/Σw² — Gaussian × time (concentration)",
        )

        tv_plots.plot_l1_static_vs_tv(
            dates_yyyymm,
            bundle["W_gaussian_time"],
            bundle["w_static"],
            t_start,
            fig_dir / "fig_l1_distance_static_vs_tv.png",
            title=r"$\|\omega_t^{TV} - \omega^{static}\|_1$ — TV (Gaussian×time) vs static tangency",
        )

    return bundle
