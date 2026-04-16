#!/usr/bin/env python3
"""
Pruned-basis figure: weights (bars) + FF5 gross α (red) and optional net α after TC (blue).

Supports **AP (median) trees** and **RP trees** (``--tree ap|rp|both``). For ``both``,
one PDF/PNG with two subplots (left AP, right RP) for the same triplet and ``k``.

- **AP:** ``decode_column`` + ``assign_nodes_month`` (same as ``transaction_costs`` on main).
  Default returns file: ``level_all_excess_combined_filtered.csv`` (pruning input).
- **RP:** ``decode_rp_column`` + ``projection_matrices.npz`` from step2 RP.

From repo root::

    # RP only (defaults: grid_search/rp_tree, rp_tree_portfolios)
    python part_3_metrics_collection/plot_rp_weights_ff5_alpha.py \\
        --tree rp --feat1 BEME --feat2 OP --k 10 \\
        --out Figures/figures/pruned_ff5_LME_BEME_OP_k10_rp.pdf

    # AP only (defaults: grid_search/tree, tree_portfolios, *_filtered.csv)
    python part_3_metrics_collection/plot_rp_weights_ff5_alpha.py \\
        --tree ap --feat1 BEME --feat2 OP --k 10 \\
        --out Figures/figures/pruned_ff5_LME_BEME_OP_k10_ap.pdf

    # Both panels side-by-side
    python part_3_metrics_collection/plot_rp_weights_ff5_alpha.py \\
        --tree both --feat1 BEME --feat2 OP --k 10 \\
        --out Figures/figures/pruned_ff5_LME_BEME_OP_k10_ap_rp.pdf

    # Gross only
    python part_3_metrics_collection/plot_rp_weights_ff5_alpha.py ... --no-net
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from part_3_metrics_collection.ff5 import (
    Y_MAX,
    Y_MIN,
    generate_dates,
    load_ff5_research_panel,
    run_ff5_regression_detailed,
)
from part_3_metrics_collection.transaction_costs import (
    _build_month_index,
    _load_panel,
    _run_tc_loop,
    decode_column,
    decode_rp_column,
    load_rp_projection_dict,
)

TreeKind = Literal["ap", "rp", "both"]


@dataclass
class PanelResult:
    """One subplot’s worth of data (AP or RP)."""

    kind: Literal["ap", "rp"]
    cols: list[str]
    weights: np.ndarray
    alphas: list[float]
    ses: list[float]
    alphas_net: list[float]
    ses_net: list[float]
    has_net: bool
    footnote: str


def _plot_ff5_alpha_ci(
    ax,
    xpos: np.ndarray,
    alpha: list[float],
    se: list[float],
    z: float,
    color: str,
    err_linestyle: str,
    cap_w: float,
    zorder: int,
) -> None:
    a = np.asarray(alpha, dtype=float)
    s = np.asarray(se, dtype=float)
    for i, xi in enumerate(xpos):
        if not (np.isfinite(a[i]) and np.isfinite(s[i])):
            continue
        lo = float(a[i] - z * s[i])
        hi = float(a[i] + z * s[i])
        ax.plot(
            [xi, xi],
            [lo, hi],
            color=color,
            linestyle=err_linestyle,
            linewidth=1.15,
            solid_capstyle="butt",
            zorder=zorder,
        )
        ax.plot(
            [xi - cap_w, xi + cap_w],
            [hi, hi],
            color=color,
            linestyle=err_linestyle,
            linewidth=1.15,
            solid_capstyle="butt",
            zorder=zorder,
        )
        ax.plot(
            [xi - cap_w, xi + cap_w],
            [lo, lo],
            color=color,
            linestyle=err_linestyle,
            linewidth=1.15,
            solid_capstyle="butt",
            zorder=zorder,
        )
    ax.scatter(
        xpos,
        a,
        color=color,
        s=36,
        edgecolors=color,
        linewidths=0.8,
        zorder=zorder + 1,
    )


def _load_weights_vector(path: Path, k: int) -> np.ndarray:
    w = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=float)
    if len(w) == k + 1 and np.isclose(w[0], 0.0):
        w = w[1:]
    if len(w) != k:
        raise ValueError(f"Expected {k} weights in {path}, got {len(w)}")
    return w


def _default_paths(kind: Literal["ap", "rp"]) -> tuple[Path, Path, str]:
    if kind == "ap":
        return (
            Path("data/results/grid_search/tree"),
            Path("data/results/tree_portfolios"),
            "level_all_excess_combined_filtered.csv",
        )
    return (
        Path("data/results/grid_search/rp_tree"),
        Path("data/results/rp_tree_portfolios"),
        "level_all_excess_combined.csv",
    )


def _resolved_paths(
    args: argparse.Namespace,
    kind: Literal["ap", "rp"],
) -> tuple[Path, Path, str]:
    """Merge CLI overrides with defaults. For ``--tree both``, generic ``--grid-dir`` is ignored."""
    d_g, d_p, d_n = _default_paths(kind)
    single = args.tree != "both"
    if kind == "ap":
        g = args.ap_grid_dir or (args.grid_dir if single else None) or d_g
        po = args.ap_ports_dir or (args.ports_dir if single else None) or d_p
        pn = args.ap_port_name or (args.port_name if single else None) or d_n
    else:
        g = args.rp_grid_dir or (args.grid_dir if single else None) or d_g
        po = args.rp_ports_dir or (args.ports_dir if single else None) or d_p
        pn = args.rp_port_name or (args.port_name if single else None) or d_n
    return g, po, pn


def compute_panel(
    kind: Literal["ap", "rp"],
    feat1: str,
    feat2: str,
    k: int,
    grid_dir: Path,
    ports_dir: Path,
    port_name: str,
    proj_npz: Path | None,
    n_train_valid: int,
    returns_scale: float,
    panel_parquet: Path,
    no_net: bool,
) -> PanelResult:
    subdir = "_".join(["LME", feat1, feat2])
    g, po, pn = grid_dir, ports_dir, port_name

    base = g / subdir
    sel_path = base / f"Selected_Ports_{k}.csv"
    w_path = base / f"Selected_Ports_Weights_{k}.csv"
    port_path = po / subdir / pn

    if not sel_path.is_file():
        raise FileNotFoundError(f"Missing {sel_path}")
    if not w_path.is_file():
        raise FileNotFoundError(f"Missing {w_path}")
    if not port_path.is_file():
        raise FileNotFoundError(f"Missing {port_path}")

    cols = list(pd.read_csv(sel_path, nrows=0).columns)
    if len(cols) != k:
        raise ValueError(f"Selected_Ports has {len(cols)} columns, k={k}")

    weights = _load_weights_vector(w_path, k)
    ports = pd.read_csv(port_path)
    test = ports.iloc[n_train_valid:][cols]

    all_dates = generate_dates(Y_MIN, Y_MAX)
    test_dates = all_dates[n_train_valid:]
    if len(test_dates) != len(test):
        raise ValueError(
            f"Date/return length mismatch: dates {len(test_dates)} vs rows {len(test)}"
        )

    feats_order = ["LME", feat1, feat2]
    col_info: dict = {}

    all_months = _build_month_index()
    test_months = all_months.iloc[n_train_valid : n_train_valid + len(test)].reset_index(
        drop=True
    )
    test_yy = test_months["yy"].to_numpy(dtype=int).tolist()
    test_mm = test_months["mm"].to_numpy(dtype=int).tolist()

    panel = None
    if not no_net and panel_parquet.is_file():
        panel = _load_panel(panel_parquet, feats_order, test_months)

    if panel is not None:
        if kind == "rp":
            proj_npz_path = (
                proj_npz if proj_npz is not None else (po / subdir / "projection_matrices.npz")
            )
            try:
                proj_by = load_rp_projection_dict(proj_npz_path)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"{e}\n"
                    "Net alphas for RP need projection_matrices.npz from step2 RP "
                    "(or pass --proj-npz / --no-net)."
                ) from e
            col_info = {c: decode_rp_column(c, feats_order, proj_by) for c in cols}
        else:
            col_info = {c: decode_column(c, feats_order) for c in cols}

    alphas: list[float] = []
    ses: list[float] = []
    alphas_net: list[float] = []
    ses_net: list[float] = []

    T = len(test)
    for j, c in enumerate(cols):
        r = pd.to_numeric(test[c], errors="coerce").to_numpy(dtype=float) / float(returns_scale)
        mask = np.isfinite(r)
        det = (
            run_ff5_regression_detailed(r[mask], test_dates[mask])
            if int(mask.sum()) > 12
            else None
        )
        if det is None:
            alphas.append(float("nan"))
            ses.append(float("nan"))
        else:
            alphas.append(det["alpha"])
            ses.append(det["se_alpha"])

        if panel is None or no_net:
            alphas_net.append(float("nan"))
            ses_net.append(float("nan"))
            continue

        sdf_w = np.zeros((T, len(cols)), dtype=float)
        sdf_w[:, j] = 1.0
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                tc = _run_tc_loop(r, sdf_w, cols, col_info, panel, test_yy, test_mm)
        except Exception as e:
            print(f"Warning: TC failed for column {c} ({kind}): {e}", flush=True)
            alphas_net.append(float("nan"))
            ses_net.append(float("nan"))
            continue
        net_r = r - tc
        mask_n = np.isfinite(net_r)
        det_n = (
            run_ff5_regression_detailed(net_r[mask_n], test_dates[mask_n])
            if int(mask_n.sum()) > 12
            else None
        )
        if det_n is None:
            alphas_net.append(float("nan"))
            ses_net.append(float("nan"))
        else:
            alphas_net.append(det_n["alpha"])
            ses_net.append(det_n["se_alpha"])

    has_net = panel is not None and not no_net and any(np.isfinite(alphas_net))

    if kind == "rp":
        fn = (
            r"Net $\alpha$: Bemelmans-style TC; RP stock sets from projection_matrices.npz "
            r"(same as step2 RP)."
        )
    else:
        fn = (
            r"Net $\alpha$: Bemelmans-style TC; AP stock sets from median-tree splits "
            r"(decode_column / assign_nodes_month)."
        )
    if not has_net:
        fn = r"Gross $\alpha$ only (no panel or --no-net)."

    return PanelResult(
        kind=kind,
        cols=cols,
        weights=weights,
        alphas=alphas,
        ses=ses,
        alphas_net=alphas_net,
        ses_net=ses_net,
        has_net=has_net,
        footnote=fn,
    )


def _draw_axes(
    ax1: plt.Axes,
    ax2: plt.Axes,
    res: PanelResult,
    subdir: str,
    k: int,
    z: float,
    xlabel: str,
) -> None:
    x = np.arange(k)
    dx = 0.14
    cap_w = 0.055
    x_g = x - dx
    x_n = x + dx

    ax2.bar(
        x,
        res.weights,
        width=0.55,
        color="#d9d9d9",
        edgecolor="#888888",
        linewidth=0.6,
        zorder=1,
    )
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Weight", fontsize=10)
    ax2.tick_params(axis="y", direction="in", labelsize=9)

    _plot_ff5_alpha_ci(ax1, x_g, res.alphas, res.ses, z, "#c0392b", "-", cap_w, zorder=4)
    if res.has_net:
        _plot_ff5_alpha_ci(ax1, x_n, res.alphas_net, res.ses_net, z, "#2980b9", "--", cap_w, zorder=5)

    parts: list[float] = []
    for i in range(len(res.alphas)):
        if np.isfinite(res.alphas[i]) and np.isfinite(res.ses[i]):
            parts.extend([res.alphas[i] - z * res.ses[i], res.alphas[i] + z * res.ses[i]])
    if res.has_net:
        for i in range(len(res.alphas_net)):
            if np.isfinite(res.alphas_net[i]) and np.isfinite(res.ses_net[i]):
                parts.extend(
                    [
                        res.alphas_net[i] - z * res.ses_net[i],
                        res.alphas_net[i] + z * res.ses_net[i],
                    ]
                )
    if parts:
        lo_b, hi_b = float(np.nanmin(parts)), float(np.nanmax(parts))
        pad = 0.12 * (hi_b - lo_b + 1e-9)
        ax1.set_ylim(lo_b - pad, hi_b + pad)

    ax1.axhline(0.0, color="black", linewidth=0.9, zorder=2)
    ax1.set_ylabel(r"FF5 $\alpha$ (monthly)", fontsize=10)
    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.tick_params(axis="y", direction="in", labelsize=9)

    labels = [str(c) for c in res.cols]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)

    label = "AP (median trees)" if res.kind == "ap" else "RP"
    ax1.set_title(f"{label} — {subdir} — k={k}", fontsize=10, pad=8)

    leg = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#c0392b",
            markerfacecolor="#c0392b",
            linestyle="None",
            markersize=6,
            label="Gross FF5 α",
        ),
    ]
    if res.has_net:
        leg.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="#2980b9",
                markerfacecolor="#2980b9",
                linestyle="None",
                markersize=6,
                label="Net FF5 α (after TC)",
            )
        )
    leg.append(
        Patch(facecolor="#d9d9d9", edgecolor="#888888", linewidth=0.6, label="Weight")
    )
    ax1.legend(handles=leg, loc="upper left", frameon=True, fontsize=8)


def main() -> None:
    os.chdir(REPO)
    p = argparse.ArgumentParser(
        description="Pruned basis: weights + FF5 gross/net α (AP median trees, RP, or both)."
    )
    p.add_argument(
        "--tree",
        choices=("ap", "rp", "both"),
        default="rp",
        help="Which tree type to plot: ap, rp, or both (two subplots).",
    )
    p.add_argument("--feat1", default="BEME")
    p.add_argument("--feat2", default="OP")
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--grid-dir",
        type=Path,
        default=None,
        help="When --tree is ap or rp only: override grid_search root. Ignored for --tree both (use --ap-grid-dir / --rp-grid-dir).",
    )
    p.add_argument(
        "--ports-dir",
        type=Path,
        default=None,
        help="When --tree is ap or rp only: override portfolios root. Ignored for --tree both.",
    )
    p.add_argument(
        "--port-name",
        type=str,
        default=None,
        help="When --tree is ap or rp only: override combined returns CSV filename.",
    )
    p.add_argument("--ap-grid-dir", type=Path, default=None, help="AP: grid_search root (also for --tree both).")
    p.add_argument("--ap-ports-dir", type=Path, default=None, help="AP: tree_portfolios root.")
    p.add_argument(
        "--ap-port-name",
        type=str,
        default=None,
        help="AP: combined returns file (default level_all_excess_combined_filtered.csv).",
    )
    p.add_argument("--rp-grid-dir", type=Path, default=None, help="RP: grid_search root (also for --tree both).")
    p.add_argument("--rp-ports-dir", type=Path, default=None, help="RP: rp_tree_portfolios root.")
    p.add_argument(
        "--rp-port-name",
        type=str,
        default=None,
        help="RP: combined returns file (default level_all_excess_combined.csv).",
    )
    p.add_argument(
        "--proj-npz",
        type=Path,
        default=None,
        help="RP only: projection_matrices.npz (default: <ports-dir>/triplet/projection_matrices.npz).",
    )
    p.add_argument("--n-train-valid", type=int, default=360)
    p.add_argument(
        "--returns-scale",
        type=float,
        default=100.0,
        help="Divide raw portfolio returns by this before FF5 (100 if CSV is in percent points).",
    )
    p.add_argument("--z", type=float, default=1.96)
    p.add_argument("--out", type=Path, default=Path("Figures/figures/pruned_weights_ff5_alpha.pdf"))
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--no-net", action="store_true")
    p.add_argument("--panel-parquet", type=Path, default=Path("data/prepared/panel.parquet"))
    args = p.parse_args()

    tree: TreeKind = args.tree  # type: ignore[assignment]
    subdir = "_".join(["LME", args.feat1, args.feat2])
    z = float(args.z)

    load_ff5_research_panel()

    if not args.panel_parquet.is_file() and not args.no_net:
        print(f"Warning: panel missing {args.panel_parquet} — net alphas will be omitted.", flush=True)

    panels: list[PanelResult] = []
    if tree in ("ap", "both"):
        ag, apo, apn = _resolved_paths(args, "ap")
        panels.append(
            compute_panel(
                "ap",
                args.feat1,
                args.feat2,
                args.k,
                ag,
                apo,
                apn,
                None,
                args.n_train_valid,
                args.returns_scale,
                args.panel_parquet,
                args.no_net,
            )
        )
    if tree in ("rp", "both"):
        rg, rpo, rpn = _resolved_paths(args, "rp")
        panels.append(
            compute_panel(
                "rp",
                args.feat1,
                args.feat2,
                args.k,
                rg,
                rpo,
                rpn,
                args.proj_npz,
                args.n_train_valid,
                args.returns_scale,
                args.panel_parquet,
                args.no_net,
            )
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "figure.dpi": args.dpi,
        }
    )

    if len(panels) == 1:
        fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
        ax2 = ax1.twinx()
        pr = panels[0]
        xl = "Pruned basis (column)" if pr.kind == "ap" else "RP portfolio (pruned basis)"
        _draw_axes(ax1, ax2, pr, subdir, args.k, z, xl)
        bottom = 0.24
        note = (
            r"Gross $\alpha$: FF5 on test excess returns per pruned basis. Bars: static master weights. "
            + pr.footnote
        )
        if args.no_net or not pr.has_net:
            note = (
                r"Gross $\alpha$: FF5 on test excess returns per pruned basis. Bars: static master weights."
            )
    else:
        fig, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14.0, 4.9))
        ax2a = ax1a.twinx()
        ax2b = ax1b.twinx()
        _draw_axes(ax1a, ax2a, panels[0], subdir, args.k, z, "AP pruned basis (column)")
        _draw_axes(ax1b, ax2b, panels[1], subdir, args.k, z, "RP pruned basis (column)")
        fig.suptitle(f"Pruned portfolios — {subdir} — k={args.k}", fontsize=11, y=1.02)
        bottom = 0.30
        note = (
            r"Gross $\alpha$: FF5 on test excess returns; bars: static master weights. "
            "Left panel: "
            + panels[0].footnote
            + " Right panel: "
            + panels[1].footnote
        )
        if args.no_net or not (panels[0].has_net or panels[1].has_net):
            note = r"Gross $\alpha$: FF5 on test excess returns; bars: static master weights."

    fig.tight_layout()
    fig.subplots_adjust(bottom=bottom)
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=7.5, wrap=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = args.out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
