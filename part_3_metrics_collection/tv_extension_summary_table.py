"""
TV extension — OOS summary table (long format) comparable to ``ap_pruned_summary_table``.

Each triplet yields **three** rows (static tangency, TV Gaussian state, TV Gaussian × time),
with **monthly** OOS Sharpe (mean/std), CAPM and FF5 regressions on the same OOS months
as ``compute_research_bundle`` / ``tv_oos_sharpe_summary.json``.

Use ``merge_ap_tv_long_tables`` to stack AP (one row per triplet) with TV rows for side-by-side comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from part_3_metrics_collection.ff5 import run_capm_regression, run_ff5_regression_detailed
from part_3_metrics_collection.pick_best_lambdas import pruning_results_base
from time_varying_extension.kernel import suggest_bandwidth_median_dist
from time_varying_extension.research_figures import compute_research_bundle, oos_return_stats
from time_varying_extension.state_panel import align_states_to_returns, monthly_vw_state
from time_varying_extension.workflow_one_triplet import _load_portfolio_subset

# (strategy_id, bundle key for return series)
_TV_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("static_tangency_train_valid", "r_static"),
    ("tv_kernel_gaussian_state", "r_tv_gaussian"),
    ("tv_kernel_gaussian_x_time_decay", "r_tv_gaussian_time"),
)


def _safe_capm_ff5(
    rr: np.ndarray, dd: np.ndarray
) -> tuple[
    tuple[Any, ...] | None,
    dict[str, Any] | None,
]:
    rr = np.asarray(rr, dtype=float)
    dd = np.asarray(dd, dtype=int)
    m = np.isfinite(rr)
    rr, dd = rr[m], dd[m]
    if rr.size < 24:
        return None, None
    try:
        capm = run_capm_regression(rr, dd)
        ff5 = run_ff5_regression_detailed(rr, dd)
    except Exception:
        return None, None
    return capm, ff5


def build_tv_oos_factor_table_long(
    bundle: dict[str, Any],
    dates_yyyymm: np.ndarray,
    feat1: str,
    feat2: str,
    n_primitive_ports: int,
    bandwidth: float,
    min_train_months: int,
    ridge_sigma: float,
    n_train_valid: int,
    time_window_m: int,
    time_decay_lambda: float,
) -> pd.DataFrame:
    """
    One row per TV research strategy; OOS definition matches ``oos_mask`` (lag-1 TV returns).
    """
    mask = bundle["oos_mask"]
    dates_yyyymm = np.asarray(dates_yyyymm, dtype=int)
    rows: list[dict[str, Any]] = []

    for strategy_id, rkey in _TV_STRATEGIES:
        r = bundle[rkey]
        st = oos_return_stats(r, mask)
        rr = np.asarray(r, dtype=float)[mask]
        dd = dates_yyyymm[mask]
        fin = np.isfinite(rr)
        rr, dd = rr[fin], dd[fin]

        capm, ff5 = _safe_capm_ff5(rr, dd)

        row: dict[str, Any] = {
            "family": "TV_extension",
            "strategy": strategy_id,
            "feat1": feat1,
            "feat2": feat2,
            "subdir": f"LME_{feat1}_{feat2}",
            "n_primitive_ports": int(n_primitive_ports),
            "oos_n_months": st["n_months"],
            "oos_SR_monthly": st["sharpe_monthly"],
            "oos_SR_annualized_sqrt12": st["sharpe_annualized_sqrt12"],
            "oos_mean_monthly": st["mean_monthly"],
            "oos_std_monthly": st["std_monthly"],
            "bandwidth": float(bandwidth),
            "min_train_months": int(min_train_months),
            "ridge_sigma": float(ridge_sigma),
            "n_train_valid": int(n_train_valid),
            "time_window_m": int(time_window_m),
            "time_decay_lambda": float(time_decay_lambda),
            "lambda0": np.nan,
            "lambda2": np.nan,
        }

        if capm is not None and capm[0] is not None:
            a, b, pa, pb, nobs = capm
            row["capm_alpha_monthly"] = a
            row["capm_beta_mkt"] = b
            row["capm_p_alpha"] = pa
            row["capm_p_beta"] = pb
            row["capm_nobs"] = nobs
        else:
            row["capm_alpha_monthly"] = np.nan
            row["capm_beta_mkt"] = np.nan
            row["capm_p_alpha"] = np.nan
            row["capm_p_beta"] = np.nan
            row["capm_nobs"] = np.nan

        if ff5 is not None:
            row["ff5_alpha_monthly"] = ff5["alpha"]
            row["ff5_p_alpha"] = ff5["p_alpha"]
            row["ff5_r2"] = ff5["r2"]
            row["ff5_beta_Mkt-RF"] = ff5["beta_Mkt-RF"]
            row["ff5_nobs"] = ff5["nobs"]
        else:
            row["ff5_alpha_monthly"] = np.nan
            row["ff5_p_alpha"] = np.nan
            row["ff5_r2"] = np.nan
            row["ff5_beta_Mkt-RF"] = np.nan
            row["ff5_nobs"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def tv_one_triplet_table(
    feat1: str,
    feat2: str,
    portfolio_csv: Path,
    selected_ports_csv: Path,
    panel_parquet: Path,
    y_min: int = 1964,
    y_max: int = 2016,
    bandwidth: float | None = None,
    min_train_months: int = 60,
    ridge_sigma: float = 1e-3,
    n_train_valid: int = 360,
    time_window_m: int = 120,
    time_decay_lambda: float = 0.95,
) -> pd.DataFrame:
    """Load data like ``run_time_varying_one_triplet``, build research bundle, return factor table."""
    from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
        canonical_feat_pair,
    )

    feat1, feat2 = canonical_feat_pair(feat1, feat2)

    ports_df, cols = _load_portfolio_subset(
        Path(portfolio_csv), Path(selected_ports_csv), None
    )
    R = ports_df.to_numpy(dtype=float)
    T, p = R.shape

    state_df = monthly_vw_state(panel_parquet, feat1, feat2, y_min=y_min, y_max=y_max)
    T_state = int(len(state_df))
    if T_state != T:
        # Typical real pipeline: ``panel.parquet`` is filtered to [y_min, y_max] while the
        # combined portfolio CSV can include extra early/late months. Align by taking the
        # trailing overlap (both series are sorted chronologically).
        if T_state < T:
            R = R[-T_state:, :]
            T = T_state
        else:
            R = np.pad(
                R,
                ((T_state - T, 0), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )
            T = T_state

    states = align_states_to_returns(state_df, T)
    dates = state_df["yyyymm"].to_numpy(dtype=int)

    bw = bandwidth
    if bw is None:
        bw = float(suggest_bandwidth_median_dist(states, max_t=min(T, 400)))

    bundle = compute_research_bundle(
        R,
        states,
        bandwidth=bw,
        min_train_months=min_train_months,
        ridge_sigma=ridge_sigma,
        n_train_valid=n_train_valid,
        time_window_m=time_window_m,
        time_decay_lambda=time_decay_lambda,
    )

    return build_tv_oos_factor_table_long(
        bundle,
        dates,
        feat1,
        feat2,
        p,
        bw,
        min_train_months,
        ridge_sigma,
        n_train_valid,
        time_window_m,
        time_decay_lambda,
    )


def ap_pruned_csv_to_long(ap_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize AP summary CSV rows to the same long-schema as TV (one row per AP SDF)."""
    rows: list[dict[str, Any]] = []
    for _, r in ap_df.iterrows():
        k = int(r["k_managed_ports"])
        rows.append(
            {
                "family": "AP_pruned",
                "strategy": f"sdf_validation_tuned_k{k}",
                "feat1": r["feat1"],
                "feat2": r["feat2"],
                "subdir": r["subdir"],
                "n_primitive_ports": k,
                "oos_n_months": np.nan,
                "oos_SR_monthly": r.get("test_SR_monthly", np.nan),
                "oos_SR_annualized_sqrt12": (
                    float(np.sqrt(12.0) * r["test_SR_monthly"])
                    if pd.notna(r.get("test_SR_monthly"))
                    else np.nan
                ),
                "oos_mean_monthly": r.get("test_mean_monthly", np.nan),
                "oos_std_monthly": r.get("test_std_monthly", np.nan),
                "bandwidth": np.nan,
                "min_train_months": np.nan,
                "ridge_sigma": np.nan,
                "n_train_valid": np.nan,
                "time_window_m": np.nan,
                "time_decay_lambda": np.nan,
                "lambda0": r.get("lambda0", np.nan),
                "lambda2": r.get("lambda2", np.nan),
                "capm_alpha_monthly": r.get("capm_alpha_monthly", np.nan),
                "capm_beta_mkt": r.get("capm_beta_mkt", np.nan),
                "capm_p_alpha": r.get("capm_p_alpha", np.nan),
                "capm_p_beta": r.get("capm_p_beta", np.nan),
                "capm_nobs": r.get("capm_nobs", np.nan),
                "ff5_alpha_monthly": r.get("ff5_alpha_monthly", np.nan),
                "ff5_p_alpha": r.get("ff5_p_alpha", np.nan),
                "ff5_r2": r.get("ff5_r2", np.nan),
                "ff5_beta_Mkt-RF": r.get("ff5_beta_Mkt-RF", np.nan),
                "ff5_nobs": r.get("ff5_nobs", np.nan),
            }
        )
    return pd.DataFrame(rows)


def merge_ap_tv_long_tables(ap_df: pd.DataFrame | None, tv_df: pd.DataFrame) -> pd.DataFrame:
    """Stack AP long rows (optional) under/above TV rows; same columns (union fill NaN)."""
    parts: list[pd.DataFrame] = []
    if ap_df is not None and len(ap_df):
        parts.append(ap_pruned_csv_to_long(ap_df))
    parts.append(tv_df)
    out = pd.concat(parts, ignore_index=True)
    return out


def build_tv_summary_all_triplets(
    grid_dir: Path,
    ports_dir: Path,
    port_name: str,
    panel_parquet: Path,
    selected_k: int,
    y_min: int = 1964,
    y_max: int = 2016,
    min_train_months: int = 60,
    ridge_sigma: float = 1e-3,
    n_train_valid: int = 360,
    time_window_m: int = 120,
    time_decay_lambda: float = 0.95,
) -> pd.DataFrame:
    """Run TV table for every cross-section with ``Selected_Ports_{k}.csv`` and portfolio CSV."""
    from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
        all_triplet_pairs,
        canonical_feat_pair,
    )

    all_rows: list[pd.DataFrame] = []
    for f1, f2 in all_triplet_pairs():
        cf1, cf2 = canonical_feat_pair(f1, f2)
        sub = f"LME_{cf1}_{cf2}"
        sel = pruning_results_base(grid_dir, cf1, cf2) / f"Selected_Ports_{selected_k}.csv"
        pcsv = ports_dir / sub / port_name
        if not sel.is_file() or not pcsv.is_file():
            continue
        print(f"TV summary: {sub} (k={selected_k})")
        try:
            df = tv_one_triplet_table(
                cf1,
                cf2,
                pcsv,
                sel,
                panel_parquet,
                y_min=y_min,
                y_max=y_max,
                min_train_months=min_train_months,
                ridge_sigma=ridge_sigma,
                n_train_valid=n_train_valid,
                time_window_m=time_window_m,
                time_decay_lambda=time_decay_lambda,
            )
            all_rows.append(df)
        except Exception as e:
            print(f"  skip {sub}: {e}")

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="TV extension OOS table (monthly SR + CAPM/FF5), optionally merge AP CSV."
    )
    p.add_argument(
        "--mode",
        choices=("one", "all"),
        default="one",
        help="one triplet or all triplets with Selected_Ports_k",
    )
    p.add_argument("--feat1", default="OP")
    p.add_argument("--feat2", default="Investment")
    p.add_argument(
        "--portfolio-csv",
        type=Path,
        default=None,
        help="Filtered combined excess returns (one triplet folder); required for mode=one",
    )
    p.add_argument(
        "--selected-ports",
        type=Path,
        default=None,
        help="Selected_Ports_k.csv; default grid_dir/LME_feat_feat/Selected_Ports_{k}.csv for mode=one",
    )
    p.add_argument("--grid-dir", type=Path, default=Path("data/results/grid_search/tree"))
    p.add_argument("--ports-dir", type=Path, default=Path("data/results/tree_portfolios"))
    p.add_argument(
        "--port-name",
        default="level_all_excess_combined_filtered.csv",
    )
    p.add_argument("--panel-parquet", type=Path, default=Path("data/prepared/panel.parquet"))
    p.add_argument("--k", type=int, default=10, help="Selected_Ports index (match AP pruning k)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--merge-ap",
        type=Path,
        default=None,
        help="Optional ap_pruned_summary_k*.csv to stack with TV rows",
    )
    args = p.parse_args()

    if args.mode == "all":
        tv_df = build_tv_summary_all_triplets(
            args.grid_dir,
            args.ports_dir,
            args.port_name,
            args.panel_parquet,
            selected_k=args.k,
        )
    else:
        from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
            canonical_feat_pair,
        )

        f1, f2 = canonical_feat_pair(args.feat1, args.feat2)
        sub = f"LME_{f1}_{f2}"
        port_csv = args.portfolio_csv or (args.ports_dir / sub / args.port_name)
        sel = args.selected_ports or (
            pruning_results_base(args.grid_dir, f1, f2) / f"Selected_Ports_{args.k}.csv"
        )
        tv_df = tv_one_triplet_table(
            f1,
            f2,
            port_csv,
            sel,
            args.panel_parquet,
        )

    ap_part = None
    if args.merge_ap is not None and args.merge_ap.is_file():
        ap_part = pd.read_csv(args.merge_ap)

    out_df = merge_ap_tv_long_tables(ap_part, tv_df) if ap_part is not None else tv_df

    out = args.out
    if out is None:
        out = Path("data/results") / f"tv_extension_summary_k{args.k}.csv"
        if args.mode == "all":
            out = Path("data/results") / f"tv_extension_summary_all_k{args.k}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} rows to {out}")


if __name__ == "__main__":
    main()
