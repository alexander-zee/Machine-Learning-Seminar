#!/usr/bin/env python3
"""
Scatter: out-of-sample monthly Sharpe **gain** (kernel minus uniform) vs winning
bandwidth multiplier ``h* / sigma_S`` on a log axis — same idea as thesis Figure 8.

Reads, per triplet ``LME_feat1_feat2`` under ``<grid_base>/<kernel>/``:

* ``SR_grid_{k}.csv`` or ``SR_grid_k{k}.csv`` + ``grid_manifest.json`` → validation winner
  ``(h_idx, …)`` and ``h*`` from ``bandwidths[h_idx-1]``.
* ``sigma_S`` = std of the state over the first ``n_train_valid`` months (``svar`` for
  ``gaussian``, ``TMS`` for ``gaussian-tms``).
* OOS monthly Sharpe on the **test** window from ``full_fit/full_fit_detail_k{k}.csv``
  (column ``excess_return``), with the same ×100 scaling heuristic as
  ``export_table51_uniform_vs_gaussian._maybe_scale_kernel_detail_returns``.
* Uniform baseline Sharpe: ``Selected_Ports_{k}`` @ weights on the master panel
  (``load_master_test_returns``), same test window.

Examples::

    cd <repo>
    # AP trees (defaults below)
    python part_4_plots/plot_bandwidth_sr_gain_scatter.py \\
        -o Figures/Results/fig_sr_gain_vs_bandwidth_ap.png

    # RP trees
    python part_4_plots/plot_bandwidth_sr_gain_scatter.py --rp \\
        -o Figures/Results/fig_sr_gain_vs_bandwidth_rp.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
    all_triplet_pairs,
    triplet_subdir_name,
)
from part_3_metrics_collection.ff5 import load_master_test_returns


N_TRAIN_VALID_DEFAULT = 360

# Default tick multipliers for the thesis-style bandwidth axis (matches
# ``GaussianKernel.bandwidth_grid(..., n=5)`` relative to σ_S).
DEFAULT_SIGMA_TICK_MULTS = np.array([0.05, 0.16, 0.50, 1.58, 5.00], dtype=float)


def _naive_monthly_sharpe(r: np.ndarray) -> float:
    x = np.asarray(r, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    s = float(np.std(x, ddof=1))
    if s <= 0:
        return float("nan")
    return float(np.mean(x) / s)


def _maybe_scale_kernel_detail_returns(r: np.ndarray) -> np.ndarray:
    x = np.asarray(r, dtype=float)
    m = float(np.nanmean(np.abs(x))) if np.any(np.isfinite(x)) else float("nan")
    if np.isfinite(m) and m > 0.05:
        return x / 100.0
    return x


def _sr_grid_path(triplet_kernel_dir: Path, k: int) -> Path | None:
    p1 = triplet_kernel_dir / f"SR_grid_{k}.csv"
    if p1.is_file():
        return p1
    p2 = triplet_kernel_dir / f"SR_grid_k{k}.csv"
    if p2.is_file():
        return p2
    return None


def winner_h_star(triplet_kernel_dir: Path, k: int) -> float | None:
    """Validation-max ``h`` from manifest + SR grid (1-based idx columns)."""
    man_p = triplet_kernel_dir / "grid_manifest.json"
    sr_p = _sr_grid_path(triplet_kernel_dir, k)
    if not man_p.is_file() or sr_p is None:
        return None
    with open(man_p, encoding="utf-8") as f:
        m = json.load(f)
    df = pd.read_csv(sr_p)
    if df.empty or "valid_SR" not in df.columns:
        return None
    sr = pd.to_numeric(df["valid_SR"], errors="coerce").to_numpy(dtype=float)
    if not np.any(np.isfinite(sr)):
        return None
    bi = int(np.nanargmax(sr))
    row = df.iloc[bi]
    for col in ("l0_idx", "l2_idx", "h_idx"):
        if col not in row.index:
            return None
    ih = int(row["h_idx"]) - 1
    bws = m.get("bandwidths", [])
    if ih < 0 or ih >= len(bws):
        return None
    h_raw = bws[ih]
    if h_raw == "uniform" or h_raw is None:
        return None
    try:
        return float(h_raw)
    except (TypeError, ValueError):
        return None


def kernel_test_sr(
    grid_base: Path, kernel_name: str, subdir: str, k: int
) -> float:
    detail = grid_base / kernel_name / subdir / "full_fit" / f"full_fit_detail_k{k}.csv"
    if not detail.is_file():
        return float("nan")
    df = pd.read_csv(detail)
    if "excess_return" not in df.columns:
        return float("nan")
    r = pd.to_numeric(df["excess_return"], errors="coerce").to_numpy(dtype=float)
    r = _maybe_scale_kernel_detail_returns(r)
    return _naive_monthly_sharpe(r)


def paper_label(feat1: str, feat2: str) -> str:
    m = {
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
    return f"Size×{m.get(feat1, feat1)}×{m.get(feat2, feat2)}"


def main() -> None:
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument(
        "--rp",
        action="store_true",
        help="Shorthand: RP grid + rp_tree_portfolios + level_all_excess_combined.csv.",
    )
    pa.add_argument(
        "--grid-base",
        type=Path,
        default=None,
        help="Grid root (contains uniform/, gaussian/, …). Default: tree or RP with --rp.",
    )
    pa.add_argument(
        "--ports-dir",
        type=Path,
        default=None,
        help="Directory with LME_* / master return CSV.",
    )
    pa.add_argument(
        "--port-file",
        type=str,
        default=None,
        help="Master panel filename under each LME_* folder.",
    )
    pa.add_argument("--k", type=int, default=10)
    pa.add_argument("--n-train-valid", type=int, default=N_TRAIN_VALID_DEFAULT)
    pa.add_argument("--state-csv", type=Path, default=Path("data/state_variables.csv"))
    pa.add_argument(
        "--kernels",
        nargs="+",
        default=["gaussian", "gaussian-tms"],
        help="Kernel folder names under grid_base (e.g. gaussian gaussian-tms).",
    )
    pa.add_argument(
        "--state-col",
        nargs="+",
        default=None,
        help="One state column per kernel (same order as --kernels). "
        "Default: svar for gaussian, TMS for gaussian-tms, else svar.",
    )
    pa.add_argument("-o", "--out", type=Path, required=True)
    pa.add_argument("--dpi", type=int, default=150)
    pa.add_argument("--annotate-below", type=float, default=-0.05)
    pa.add_argument(
        "--xlim-mult",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help=(
            "X-axis limits in units of h*/sigma_S (log scale). "
            f"Default: {float(DEFAULT_SIGMA_TICK_MULTS[0]):.2f} {float(DEFAULT_SIGMA_TICK_MULTS[-1]):.2f}"
        ),
    )
    pa.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Y-axis limits for SR gain. Default: auto from data with a small pad.",
    )
    pa.add_argument(
        "--sigma-tick-mults",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Log-axis tick positions in units of h*/sigma_S. "
            f"Default: {' '.join(f'{m:.2f}' for m in DEFAULT_SIGMA_TICK_MULTS)}"
        ),
    )
    pa.add_argument(
        "--legend-loc",
        type=str,
        default="lower center",
        help="Matplotlib legend loc=... (default lower center, thesis-style).",
    )
    pa.add_argument(
        "--legend-ncol",
        type=int,
        default=3,
        help="Legend columns (default 3 so baseline + 4 series fits one row).",
    )
    pa.add_argument(
        "--legend-bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("X0", "Y0", "W", "H"),
        help="Optional bbox_to_anchor for the legend (fractions of axes).",
    )
    pa.add_argument(
        "--no-vlines",
        action="store_true",
        help="Disable faint vertical gridlines at the sigma tick multipliers.",
    )
    args = pa.parse_args()

    if args.rp:
        grid_base = Path("data/results/grid_search/rp_tree")
        ports_dir = Path("data/results/rp_tree_portfolios")
        port_file = "level_all_excess_combined.csv"
    else:
        grid_base = Path("data/results/grid_search/tree")
        ports_dir = Path("data/results/tree_portfolios")
        port_file = "level_all_excess_combined_filtered.csv"

    if args.grid_base is not None:
        grid_base = args.grid_base
    if args.ports_dir is not None:
        ports_dir = args.ports_dir
    if args.port_file is not None:
        port_file = args.port_file

    if args.state_col is None:
        cols = []
        for kn in args.kernels:
            if kn in ("gaussian",):
                cols.append("svar")
            elif kn in ("gaussian-tms", "gaussian_tms"):
                cols.append("TMS")
            else:
                cols.append("svar")
    else:
        cols = list(args.state_col)
        if len(cols) != len(args.kernels):
            raise SystemExit("--state-col must have same length as --kernels")

    state_df = pd.read_csv(args.state_csv)
    if "MthCalDt" in state_df.columns:
        state_df = state_df.set_index("MthCalDt")

    rows: list[dict] = []
    for feat1, feat2 in all_triplet_pairs():
        sub = triplet_subdir_name(feat1, feat2)
        try:
            test_r, _ = load_master_test_returns(
                feat1,
                feat2,
                args.k,
                grid_base,
                ports_dir,
                port_file,
                n_train_valid=args.n_train_valid,
            )
        except (FileNotFoundError, ValueError, OSError):
            continue
        sr_u = _naive_monthly_sharpe(test_r)
        if not np.isfinite(sr_u):
            continue

        for kernel_name, st_col in zip(args.kernels, cols):
            tdir = grid_base / kernel_name / sub
            h_star = winner_h_star(tdir, args.k)
            if h_star is None or not np.isfinite(h_star):
                continue
            st = pd.to_numeric(state_df[st_col], errors="coerce").iloc[: args.n_train_valid]
            sig = float(st.std(ddof=1))
            if not np.isfinite(sig) or sig <= 0:
                continue
            mult = h_star / sig
            sr_k = kernel_test_sr(grid_base, kernel_name, sub, args.k)
            if not np.isfinite(sr_k):
                continue
            rows.append(
                {
                    "subdir": sub,
                    "kernel": kernel_name,
                    "h_star": h_star,
                    "mult": mult,
                    "sr_uniform": sr_u,
                    "sr_kernel": sr_k,
                    "gain": sr_k - sr_u,
                    "label": paper_label(feat1, feat2),
                }
            )

    if not rows:
        raise SystemExit("No points to plot — check grid paths, SR_grid, manifest, full_fit.")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 10,
            "figure.titlesize": 11,
        }
    )

    # Slightly taller canvas so title + outside legend don't feel cramped.
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    markers = {"gaussian": "o", "gaussian-tms": "^", "gaussian_tms": "^"}

    df = pd.DataFrame(rows)
    legend_handles: list = []
    legend_labels: list[str] = []

    def _scatter_group(kn: str, positive: bool, color: str, leg: str) -> None:
        if kn not in args.kernels:
            return
        g = df[df["kernel"] == kn]
        sub = g[g["gain"] > 0] if positive else g[g["gain"] <= 0]
        if sub.empty:
            return
        mk = markers.get(kn, "o")
        ax.scatter(
            sub["mult"],
            sub["gain"],
            s=38,
            marker=mk,
            c=color,
            alpha=0.88,
            edgecolors="black",
            linewidths=0.35,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=mk,
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.35,
            )
        )
        legend_labels.append(leg)

    _scatter_group("gaussian", True, "#1f77b4", "Gaussian (SVAR), outperform uniform")
    _scatter_group("gaussian", False, "#ff7f0e", "Gaussian (SVAR), underperform uniform")
    _scatter_group("gaussian-tms", True, "#2ca02c", "Gaussian (TMS), outperform uniform")
    _scatter_group("gaussian-tms", False, "#9467bd", "Gaussian (TMS), underperform uniform")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, zorder=2)

    sigma_ticks = (
        np.asarray(args.sigma_tick_mults, dtype=float)
        if args.sigma_tick_mults is not None
        else DEFAULT_SIGMA_TICK_MULTS.copy()
    )
    sigma_ticks = sigma_ticks[np.isfinite(sigma_ticks) & (sigma_ticks > 0)]
    sigma_ticks = np.unique(np.sort(sigma_ticks))

    if not args.no_vlines:
        for m in sigma_ticks:
            ax.axvline(float(m), color="#cccccc", lw=0.8, ls=":", zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"Winning bandwidth multiplier $h^*/\sigma_S$ (log scale)")
    ax.set_ylabel(r"Out-of-sample SR gain vs Uniform ($k=%d$)" % args.k)
    ax.grid(True, linestyle=":", alpha=0.55, which="both")
    n_cs = int(df["subdir"].nunique())
    ax.set_title(
        f"(A)  Winning bandwidth vs out-of-sample SR gain — {n_cs} cross-sections\n"
        f"Gaussian (SVAR) and Gaussian (TMS), $k={args.k}$. "
        "Each marker is one cross-section; bandwidth from validation max SR.",
        loc="left",
        fontsize=10,
    )

    x0, x1 = (
        float(args.xlim_mult[0]),
        float(args.xlim_mult[1]),
    ) if args.xlim_mult is not None else (float(sigma_ticks[0]), float(sigma_ticks[-1]))
    # Add a little breathing room on log axes so edge clusters aren't clipped.
    ax.set_xlim(x0 * 0.92, x1 * 1.06)
    ax.set_xticks(sigma_ticks)
    # NOTE: matplotlib mathtext treats "_" as subscript unless braced.
    ax.set_xticklabels([rf"${m:.2f}\,\sigma_{{S}}$" for m in sigma_ticks])

    if args.ylim is None:
        y = pd.to_numeric(df["gain"], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            pad = 0.03 * (float(np.nanmax(y)) - float(np.nanmin(y)) + 1e-9)
            ax.set_ylim(float(np.nanmin(y)) - pad, float(np.nanmax(y)) + pad)
    else:
        ax.set_ylim(float(args.ylim[0]), float(args.ylim[1]))

    handles: list = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.9, label="No gain vs uniform"),
        *legend_handles,
    ]
    labels: list[str] = ["No gain vs uniform", *legend_labels]

    legend_loc = str(args.legend_loc).replace("_", " ")
    legend_kw = dict(
        handles=handles,
        labels=labels,
        loc=legend_loc,
        ncol=int(args.legend_ncol),
        fontsize=8,
        framealpha=0.92,
        borderaxespad=0.8,
        # Thesis-style: legend sits below the axes to avoid covering points.
        bbox_to_anchor=(0.5, -0.22) if args.legend_bbox is None else tuple(args.legend_bbox),
    )
    ax.legend(**legend_kw)

    thr = float(args.annotate_below)
    for _, r in df.iterrows():
        if r["gain"] >= thr:
            continue
        ax.annotate(
            r["label"],
            (r["mult"], r["gain"]),
            fontsize=6,
            color="#c0392b",
            alpha=0.9,
            xytext=(3, 3),
            textcoords="offset points",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.25)
    print(f"Wrote {args.out.resolve()}  ({len(df)} markers)")


if __name__ == "__main__":
    main()
