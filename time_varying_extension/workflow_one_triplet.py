"""
End-to-end time-varying kernel MV workflow for a single (LME, feat1, feat2) triplet.

Reads existing pipeline CSVs only; does not call AP_Pruning or pick_best_lambda.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .conditional_mv import rolling_kernel_mv_weights
from .diagnostics import (
    dual_style_kernel_metrics,
    time_varying_risk_premium_scores,
    top_kernel_turnover_months,
    top_weight_co_movements,
    weight_mass_by_split_depth,
)
from .kernel import suggest_bandwidth_median_dist
from .plots import (
    plot_depth_mass,
    plot_effective_analogues,
    plot_mv_weights_over_time,
    plot_state_and_premium,
)
from .state_panel import align_states_to_returns, monthly_vw_state


def _load_portfolio_subset(
    portfolio_csv: Path,
    selected_ports_csv: Path | None,
    portfolio_columns_subset: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    if selected_ports_csv is not None and selected_ports_csv.is_file():
        sub = pd.read_csv(selected_ports_csv, nrows=0)
        use_cols = [c.strip().strip('"') for c in sub.columns]
        full = pd.read_csv(portfolio_csv, usecols=use_cols)
        return full, use_cols
    if portfolio_columns_subset is not None:
        use_cols = list(portfolio_columns_subset)
        full = pd.read_csv(portfolio_csv, usecols=use_cols)
        return full, use_cols
    raise ValueError(
        "Provide selected_ports_csv (e.g. Selected_Ports_10.csv) or "
        "portfolio_columns_subset — MV on full AP matrix is too high-dimensional."
    )


def _write_interpretation_txt(
    path: Path,
    feat1: str,
    feat2: str,
    risk_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    mean_turnover_weights: float,
    mean_eff_analogues: float,
    kernel_notes: list[str] | None = None,
) -> None:
    if len(risk_df):
        r2 = risk_df.dropna(subset=["abs_corr_with_rp"]).sort_values(
            "abs_corr_with_rp", ascending=False
        )
        top_state = str(r2.iloc[0]["state_variable"]) if len(r2) else "n/a (no finite correlations)"
    else:
        top_state = "n/a"
    lines = [
        "Time-varying kernel extension — interpretation notes",
        "==================================================",
        "",
        "Dual interpretation (Goulet Coulombe et al., 2024):",
        "- Primal view: predictor contributions (your baseline AP-trees + LASSO).",
        "- Dual view: each period's objective is expressed via weights on *historical",
        "  analogues* (here: past months). Kernel weights w_tau are those analogue weights.",
        "- Effective number of analogues = 1/sum w^2 (concentration).",
        "- Turnover of kernel weights (TV distance) measures how fast the analogue mix shifts.",
        "",
    ]
    if kernel_notes:
        lines.extend(kernel_notes)
        lines.append("")
    lines.extend(
        [
            f"Triplet: LME x {feat1} x {feat2}",
            "",
            "Which state variable tracks the conditional premium proxy w'mu most?",
            f"  (largest |correlation|): {top_state}",
            "",
            "Where are the splits (heavier tree depth load)?",
        ]
    )
    if len(depth_df) == 0:
        lines.append("  (empty: column names are not tree-style, e.g. need tree.node format)")
    for _, r in depth_df.iterrows():
        lines.append(f"  depth {int(r['depth'])}: sum(mean|w|) = {r['sum_mean_abs_w']:.6f}")
    lines.extend(
        [
            "",
            f"Mean eff. analogues (kernel): {mean_eff_analogues:.2f}",
            f"Mean L1 turnover of MV weights (sum abs(w_t - w_prev)): {mean_turnover_weights:.6f}",
            "",
            "See tv_top_pairs.csv for the 10 portfolio pairs whose weights co-move most.",
            "See tv_kernel_turnover_top.csv for months when the analogue mix changed most.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_time_varying_one_triplet(
    feat1: str,
    feat2: str,
    portfolio_csv: Path | str,
    panel_parquet: Path | str,
    output_dir: Path | str,
    selected_ports_csv: Path | str | None = None,
    portfolio_columns_subset: list[str] | None = None,
    y_min: int = 1964,
    y_max: int = 2016,
    bandwidth: float | None = None,
    min_train_months: int = 60,
    ridge_sigma: float = 1e-3,
    use_time_decay: bool = False,
    time_window_m: int = 120,
    time_decay_lambda: float = 0.95,
) -> dict[str, Any]:
    """
    Run kernel-weighted conditional MV on one triplet's selected portfolios.

    Parameters
    ----------
    feat1, feat2
        Names after LME (e.g. OP, Investment).
    portfolio_csv
        Filtered combined excess returns (same file AP-Pruning uses).
    panel_parquet
        Prepared panel with LME, feat1, feat2, size, yy, mm.
    output_dir
        Writes figures, CSVs, and interpretation_tv.txt here.
    selected_ports_csv
        e.g. Selected_Ports_10.csv — defines the portfolio subset (recommended).
    portfolio_columns_subset
        Alternative explicit column names matching portfolio_csv.
    bandwidth
        Gaussian kernel bandwidth; if None, uses median-distance heuristic.
    min_train_months
        Earliest month index at which kernel MV is computed.
    ridge_sigma
        Regularization on Sigma for tangency weights.
    use_time_decay
        If True, multiply Gaussian state weights by geometric time decay λ^j with
        j = t − τ (months) and 1 ≤ j < ``time_window_m``.
    time_window_m
        Maximum age j (months) for nonzero time kernel (must be ≥ 2).
    time_decay_lambda
        Decay base λ ∈ (0, 1); higher = slower decay in calendar time.
    """
    portfolio_csv = Path(portfolio_csv)
    panel_parquet = Path(panel_parquet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if selected_ports_csv is not None:
        selected_ports_csv = Path(selected_ports_csv)

    ports_df, cols = _load_portfolio_subset(
        portfolio_csv, selected_ports_csv, portfolio_columns_subset
    )
    R = ports_df.to_numpy(dtype=float)
    T, p = R.shape

    state_df = monthly_vw_state(panel_parquet, feat1, feat2, y_min=y_min, y_max=y_max)
    states = align_states_to_returns(state_df, T)
    dates = state_df["yyyymm"].to_numpy(dtype=int)

    if bandwidth is None:
        bandwidth = suggest_bandwidth_median_dist(states, max_t=min(T, 400))

    kernel_cfg = {
        "bandwidth": bandwidth,
        "use_time_decay": use_time_decay,
        "time_window_m": time_window_m,
        "time_decay_lambda": time_decay_lambda,
        "min_train_months": min_train_months,
    }
    (output_dir / "tv_kernel_params.json").write_text(
        json.dumps(kernel_cfg, indent=2), encoding="utf-8"
    )

    W, MU, ess, kernels = rolling_kernel_mv_weights(
        R,
        states,
        bandwidth=bandwidth,
        min_train=min_train_months,
        ridge_sigma=ridge_sigma,
        store_kernels=True,
        use_time_decay=use_time_decay,
        time_window_m=time_window_m,
        time_decay_lambda=time_decay_lambda,
    )

    mean_abs_w = np.nanmean(np.abs(W), axis=0)
    depth_df = weight_mass_by_split_depth(cols, mean_abs_w)
    risk_df = time_varying_risk_premium_scores(
        states,
        MU,
        W,
        state_names=("LME (vw)", f"{feat1} (vw)", f"{feat2} (vw)"),
    )
    pairs_df = top_weight_co_movements(W, cols, top_n=10)
    kern_df = dual_style_kernel_metrics(kernels or [])
    if kernels:
        turn_df = top_kernel_turnover_months(kernels, dates, top_n=10)
    else:
        turn_df = pd.DataFrame()

    rp = np.array(
        [
            float(np.dot(W[t], MU[t]))
            if np.all(np.isfinite(W[t])) and np.all(np.isfinite(MU[t]))
            else np.nan
            for t in range(T)
        ]
    )

    dw = np.diff(W, axis=0)
    mean_w_turn = float(np.nanmean(np.sum(np.abs(dw), axis=1))) if T > 1 else float("nan")
    mean_eff = float(np.nanmean(ess))

    risk_df.to_csv(output_dir / "tv_risk_premium_by_state.csv", index=False)
    depth_df.to_csv(output_dir / "tv_depth_mass.csv", index=False)
    pairs_df.to_csv(output_dir / "tv_top_pairs.csv", index=False)
    kern_df.to_csv(output_dir / "tv_effective_analogues.csv", index=False)
    if not turn_df.empty:
        turn_df.to_csv(output_dir / "tv_kernel_turnover_top.csv", index=False)

    out_w = pd.DataFrame(W, columns=cols)
    out_w.insert(0, "yyyymm", dates)
    out_w.to_csv(output_dir / "tv_mv_weights.csv", index=False)

    knotes: list[str] | None = None
    if use_time_decay:
        knotes = [
            "Kernel (combined): w_tau ∝ K_G(||s_t - s_tau||) * lambda^j 1{1 <= j < m},",
            f"  j = t - tau (months), lambda={time_decay_lambda}, m={time_window_m}.",
        ]

    _write_interpretation_txt(
        output_dir / "interpretation_tv.txt",
        feat1,
        feat2,
        risk_df,
        depth_df,
        mean_w_turn,
        mean_eff,
        kernel_notes=knotes,
    )

    wtitle = f"Time-varying MV weights — LME x {feat1} x {feat2}"
    if use_time_decay:
        wtitle += f" (Gaussian + time decay λ={time_decay_lambda}, m={time_window_m})"
    plot_mv_weights_over_time(
        dates,
        W,
        cols,
        output_dir / "figures" / "tv_weights_over_time.png",
        title=wtitle,
    )
    plot_state_and_premium(
        dates,
        states,
        rp,
        output_dir / "figures" / "tv_states_and_premium_proxy.png",
        feat1_name=feat1,
        feat2_name=feat2,
    )
    if len(depth_df):
        plot_depth_mass(depth_df, output_dir / "figures" / "tv_depth_mass.png")
    plot_effective_analogues(dates, ess, output_dir / "figures" / "tv_effective_analogues.png")

    return {
        "bandwidth": bandwidth,
        "use_time_decay": use_time_decay,
        "time_window_m": time_window_m,
        "time_decay_lambda": time_decay_lambda,
        "columns": cols,
        "risk_premium_by_state": risk_df,
        "depth_mass": depth_df,
        "top_pairs": pairs_df,
        "weights": W,
        "conditional_mean": MU,
        "effective_analogues": ess,
        "output_dir": output_dir,
    }
