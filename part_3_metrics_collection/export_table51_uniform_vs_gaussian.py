#!/usr/bin/env python3
"""
Build a CSV like thesis Table 5.1: Uniform vs Gaussian kernel, OOS monthly Sharpe + FF5 alpha [t].

**Uniform Sharpe (default ``--uniform-sr summary``):** ``test_SR_monthly`` from
``ap_pruned_summary_k{k}.csv`` — the same number as ``pick_best_lambda`` / the AP grid
``results_full_*`` row for the validation-chosen (λ0, λ2). That matches your main summary table.

**``--uniform-sr naive-master``:** recompute mean(test) / std(test, ddof=1) from
``Selected_Ports_k @ weights`` on the full test window (can differ from the grid if the grid
was built with a different SR definition or stale CSVs).

**Gaussian:** naive mean/std on ``excess_return`` in ``full_fit_detail_k{k}.csv``, or
``test_SR`` from ``full_fit_summary_k{k}.csv`` if detail is missing.

Characteristic labels: Size=LME, Val=BEME, Mom=r12_2, Prof=OP, etc.

From repo root::

    python part_3_metrics_collection/export_table51_uniform_vs_gaussian.py
    python part_3_metrics_collection/export_table51_uniform_vs_gaussian.py --k 5 --out path/to/out.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
    all_triplet_pairs,
    canonical_feat_pair,
)
from part_3_metrics_collection.ff5 import load_master_test_returns, run_ff5_regression_detailed

# Internal name -> short label as in many asset-pricing tables (cf. Table 5.1 screenshot)
FEAT_TO_PAPER: dict[str, str] = {
    "BEME": "Val",
    "r12_2": "Mom",
    "OP": "Prof",
    "Investment": "Inv",
    "ST_Rev": "SRev",
    "LT_Rev": "LRev",
    "AC": "Acc",
    "LTurnover": "Turn",
    "IdioVol": "IVol",
}


def _naive_monthly_sharpe(r: np.ndarray) -> float:
    x = np.asarray(r, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    s = float(np.std(x, ddof=1))
    if s <= 0:
        return float("nan")
    return float(np.mean(x) / s)


def _ff5_alpha_t(test_r: np.ndarray, test_dates: np.ndarray) -> tuple[float, float]:
    det = run_ff5_regression_detailed(test_r, test_dates)
    if det is None:
        return float("nan"), float("nan")
    return float(det["alpha"]), float(det["t_alpha"])


def _uniform_metrics(
    feat1: str,
    feat2: str,
    *,
    grid_dir: Path,
    ports_dir: Path,
    port_name: str,
    k: int,
    n_train_valid: int,
    uniform_sr_mode: str,
    summary_by_subdir: dict[str, float] | None,
) -> tuple[float, float, float]:
    subdir = "_".join(["LME", feat1, feat2])
    try:
        test_r, test_dates = load_master_test_returns(
            feat1,
            feat2,
            k,
            grid_dir,
            ports_dir,
            port_name,
            n_train_valid=n_train_valid,
        )
    except (FileNotFoundError, ValueError, OSError):
        return float("nan"), float("nan"), float("nan")

    a, t = _ff5_alpha_t(test_r, test_dates)

    if uniform_sr_mode == "summary" and summary_by_subdir is not None:
        sr = summary_by_subdir.get(subdir, float("nan"))
        if not np.isfinite(sr):
            sr = _naive_monthly_sharpe(test_r)
    else:
        sr = _naive_monthly_sharpe(test_r)

    return sr, a, t


def _maybe_scale_kernel_detail_returns(r: np.ndarray) -> np.ndarray:
    """
    ``kernel_full_fit`` sometimes writes ``excess_return`` on a ×100 scale relative to
    master CSV returns; FF5 factors are in decimal/month. Match uniform-table α scale.
    """
    x = np.asarray(r, dtype=float)
    m = float(np.nanmean(np.abs(x))) if np.any(np.isfinite(x)) else float("nan")
    if np.isfinite(m) and m > 0.05:
        return x / 100.0
    return x


def _gaussian_metrics(
    feat1: str,
    feat2: str,
    *,
    grid_dir: Path,
    k: int,
) -> tuple[float, float, float]:
    subdir = "_".join(["LME", feat1, feat2])
    gbase = grid_dir / "gaussian" / subdir / "full_fit"
    detail_path = gbase / f"full_fit_detail_k{k}.csv"
    summary_path = gbase / f"full_fit_summary_k{k}.csv"

    if detail_path.is_file():
        df = pd.read_csv(detail_path)
        r = pd.to_numeric(df["excess_return"], errors="coerce").to_numpy(dtype=float)
        sr = _naive_monthly_sharpe(r)
        if "Date" in df.columns:
            d = pd.to_numeric(df["Date"], errors="coerce").to_numpy(dtype=np.int64)
            mask = np.isfinite(r) & np.isfinite(d.astype(float))
            r_ff = _maybe_scale_kernel_detail_returns(r)
            a, t = _ff5_alpha_t(r_ff[mask], d[mask])
        else:
            a, t = float("nan"), float("nan")
        return sr, a, t

    if summary_path.is_file():
        sm = pd.read_csv(summary_path)
        if not sm.empty and "test_SR" in sm.columns:
            return float(sm.iloc[0]["test_SR"]), float("nan"), float("nan")

    return float("nan"), float("nan"), float("nan")


def _exponential_metrics(
    feat1: str,
    feat2: str,
    *,
    grid_dir: Path,
    k: int,
) -> tuple[float, float, float]:
    """Same layout as Gaussian but under ``grid_dir/exponential/LME_*/full_fit/``."""
    subdir = "_".join(["LME", feat1, feat2])
    gbase = grid_dir / "exponential" / subdir / "full_fit"
    detail_path = gbase / f"full_fit_detail_k{k}.csv"
    summary_path = gbase / f"full_fit_summary_k{k}.csv"

    if detail_path.is_file():
        df = pd.read_csv(detail_path)
        r = pd.to_numeric(df["excess_return"], errors="coerce").to_numpy(dtype=float)
        sr = _naive_monthly_sharpe(r)
        if "Date" in df.columns:
            d = pd.to_numeric(df["Date"], errors="coerce").to_numpy(dtype=np.int64)
            mask = np.isfinite(r) & np.isfinite(d.astype(float))
            r_ff = _maybe_scale_kernel_detail_returns(r)
            a, t = _ff5_alpha_t(r_ff[mask], d[mask])
        else:
            a, t = float("nan"), float("nan")
        return sr, a, t

    if summary_path.is_file():
        sm = pd.read_csv(summary_path)
        if not sm.empty and "test_SR" in sm.columns:
            return float(sm.iloc[0]["test_SR"]), float("nan"), float("nan")

    return float("nan"), float("nan"), float("nan")


def _fmt_alpha_t(alpha: float, t: float) -> str:
    if not (np.isfinite(alpha) and np.isfinite(t)):
        return ""
    # Monthly alphas are often O(1e-3); keep 4 decimals so values are not rounded to 0.00.
    return f"{alpha:.4f} [{t:.2f}]"


def _load_summary_sr_map(grid_dir: Path, k: int) -> dict[str, float] | None:
    path = grid_dir / f"ap_pruned_summary_k{k}.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "subdir" not in df.columns or "test_SR_monthly" not in df.columns:
        return None
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        key = str(row["subdir"])
        v = pd.to_numeric(row["test_SR_monthly"], errors="coerce")
        if pd.isna(v):
            continue
        out[key] = float(v)
    return out if out else None


def build_table(
    *,
    grid_dir: Path,
    ports_dir: Path,
    port_name: str,
    k: int,
    n_train_valid: int,
    uniform_sr_mode: str,
    pairs: Iterable[tuple[str, str]] | None = None,
    show_progress: bool = False,
    include_exponential: bool = False,
) -> pd.DataFrame:
    summary_map = (
        _load_summary_sr_map(grid_dir, k) if uniform_sr_mode == "summary" else None
    )
    feat_pairs = list(all_triplet_pairs() if pairs is None else pairs)
    rows: list[dict] = []
    loop = feat_pairs
    if show_progress:
        from part_3_metrics_collection.ff5 import load_ff5_research_panel

        print("Loading Fama–French 5 factors (one download per run, then cached)…", flush=True)
        load_ff5_research_panel()
        loop = tqdm(
            feat_pairs,
            desc="Uniform vs Gaussian table",
            unit="triplet",
            file=sys.stderr,
        )
    for f1, f2 in loop:
        c1, c2 = canonical_feat_pair(f1, f2)
        u_sr, u_a, u_t = _uniform_metrics(
            c1, c2,
            grid_dir=grid_dir,
            ports_dir=ports_dir,
            port_name=port_name,
            k=k,
            n_train_valid=n_train_valid,
            uniform_sr_mode=uniform_sr_mode,
            summary_by_subdir=summary_map,
        )
        g_sr, g_a, g_t = _gaussian_metrics(c1, c2, grid_dir=grid_dir, k=k)
        row = {
            "Char1": "Size",
            "Char2": FEAT_TO_PAPER[c1],
            "Char3": FEAT_TO_PAPER[c2],
            "Uniform_SR": u_sr,
            "Uniform_FF5_alpha": u_a,
            "Uniform_FF5_t": u_t,
            "Uniform_FF5_alpha_bracket_t": _fmt_alpha_t(u_a, u_t),
            "Gaussian_SR": g_sr,
            "Gaussian_FF5_alpha": g_a,
            "Gaussian_FF5_t": g_t,
            "Gaussian_FF5_alpha_bracket_t": _fmt_alpha_t(g_a, g_t),
            "_sort_sr": u_sr,
        }
        if include_exponential:
            e_sr, e_a, e_t = _exponential_metrics(c1, c2, grid_dir=grid_dir, k=k)
            row["Exponential_SR"] = e_sr
            row["Exponential_FF5_alpha"] = e_a
            row["Exponential_FF5_t"] = e_t
            row["Exponential_FF5_alpha_bracket_t"] = _fmt_alpha_t(e_a, e_t)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("_sort_sr", ascending=True, na_position="last").reset_index(drop=True)
    df.insert(0, "Id", np.arange(1, len(df) + 1))
    return df.drop(columns=["_sort_sr"])


def main() -> None:
    os.chdir(REPO)
    p = argparse.ArgumentParser(description="Export Table 5.1-style CSV (uniform vs Gaussian).")
    p.add_argument("--grid-dir", type=Path, default=Path("data/results/grid_search/tree"))
    p.add_argument("--ports-dir", type=Path, default=Path("data/results/tree_portfolios"))
    p.add_argument("--port-name", type=str, default="level_all_excess_combined_filtered.csv")
    p.add_argument("--k", type=int, default=10, help="Managed portfolio count k")
    p.add_argument("--n-train-valid", type=int, default=360)
    p.add_argument(
        "--uniform-sr",
        choices=("summary", "naive-master"),
        default="summary",
        help="summary: test_SR_monthly from ap_pruned_summary_k{k} (matches pick_best grid). "
        "naive-master: mean/std on master test returns only.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: grid-dir/table51_uniform_vs_gaussian_k{k}.csv)",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm bar and preload FF factors once (much faster than N separate downloads).",
    )
    args = p.parse_args()
    out = args.out or (args.grid_dir / f"table51_uniform_vs_gaussian_k{args.k}.csv")

    df = build_table(
        grid_dir=args.grid_dir,
        ports_dir=args.ports_dir,
        port_name=args.port_name,
        k=args.k,
        n_train_valid=args.n_train_valid,
        uniform_sr_mode=args.uniform_sr,
        show_progress=args.progress,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(out, index=False)
    except PermissionError:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        alt = out.with_name(f"{out.stem}_{stamp}{out.suffix}")
        df.to_csv(alt, index=False)
        print(
            f"Permission denied writing {out} (file open in Excel?). "
            f"Wrote instead: {alt}",
            file=sys.stderr,
        )
        out = alt
    smode = args.uniform_sr
    if smode == "summary":
        sp = args.grid_dir / f"ap_pruned_summary_k{args.k}.csv"
        print(
            f"Uniform SR: from {sp.name}"
            if sp.is_file()
            else f"Uniform SR: summary file missing ({sp}), used naive-master fallback per row"
        )
    else:
        print("Uniform SR: naive mean/std on master test returns")
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
