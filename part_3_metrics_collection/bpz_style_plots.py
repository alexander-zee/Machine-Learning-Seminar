"""
BPZ-style *layout* for figures (line charts, heatmaps) — seminar content only.

We compare AP-Trees vs the clustering (Ward) extension using the **same monthly Sharpe
ratio** definition as the rest of the Python pipeline (see ``lasso_valid_par_full.py``).
There is **no separate gross/net SR** in this codebase unless you add it yourself.

Run from repo root:
  python -m part_3_metrics_collection.bpz_style_plots --fig3 docs/bpz_style_figure3_template.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_OUT = REPO_ROOT / "data" / "results" / "figures_seminar"


def plot_figure3_cross_section_sr(
    csv_path: Path,
    out_path: Path | None = None,
    title: str | None = None,
) -> Path:
    """
    CSV columns (required):
      cross_section_id, AP_Trees_SR, Clustering_SR

    Optional extra numeric columns are ignored. Missing values are skipped per series.
    """
    df = pd.read_csv(csv_path)
    required = ("cross_section_id", "AP_Trees_SR", "Clustering_SR")
    for c in required:
        if c not in df.columns:
            raise ValueError(
                f"CSV must include columns {required}; missing {c!r} in {csv_path}"
            )

    x = np.arange(len(df))
    labels = df["cross_section_id"].astype(str).tolist()

    series = [
        ("AP-Trees", "AP_Trees_SR", {"color": "#1f77b4", "linestyle": "-", "marker": "o"}),
        (
            "Clustering extension",
            "Clustering_SR",
            {"color": "#d62728", "linestyle": "-", "marker": "s"},
        ),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    for name, col, style in series:
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().sum() == 0:
            continue
        ax.plot(
            x,
            y,
            label=name,
            linewidth=1.2,
            markersize=4,
            **style,
        )

    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Cross-sections")
    ax.set_ylabel("Monthly Sharpe ratio (SR)")
    ax.set_title(
        title
        or "Seminar (BPZ-style layout): monthly SR — AP-Trees vs clustering extension"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    out = out_path or FIG_OUT / "figure3_cross_section_SR_bpz_style.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_figure4_heatmap_pair(
    sr_ap: np.ndarray,
    sr_clustering: np.ndarray,
    cross_section_labels: list[int] | np.ndarray,
    n_assets: list[int] | np.ndarray = (5, 10, 20, 40),
    out_path: Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """
    Two-panel heatmap (same *metric*: monthly SR), analogous to comparing two models
    in the paper's Figure 4 *style* — not gross vs net.

    Each array: shape (n_cross_sections, len(n_assets)).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True)
    mats = [
        (sr_ap, "Monthly SR — AP-Trees"),
        (sr_clustering, "Monthly SR — clustering extension"),
    ]
    all_m = np.vstack([m.ravel() for m, _ in mats])
    if vmin is None:
        vmin = float(np.nanmin(all_m))
    if vmax is None:
        vmax = float(np.nanmax(all_m))
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax

    ims = []
    for ax, (Z, ttl) in zip(axes, mats):
        im = ax.imshow(
            np.asarray(Z),
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ims.append(im)
        ax.set_title(ttl, fontsize=10)
        ax.set_yticks(range(len(cross_section_labels)))
        ax.set_yticklabels([str(x) for x in cross_section_labels], fontsize=6)
        ax.set_xticks(range(len(n_assets)))
        ax.set_xticklabels([str(i) for i in n_assets])
        ax.set_xlabel("Number of managed portfolios")

    axes[0].set_ylabel("Cross-section")
    fig.suptitle(
        "Seminar (BPZ-style layout): SR heatmaps — AP-Trees vs clustering",
        fontsize=11,
        y=1.02,
    )
    fig.colorbar(ims[-1], ax=axes, orientation="vertical", fraction=0.03, pad=0.04, label="SR")
    fig.tight_layout()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    out = out_path or FIG_OUT / "figure4_heatmap_pair_bpz_style.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fig3",
        type=Path,
        help="CSV path for line chart (see docs/bpz_style_figure3_template.csv)",
    )
    p.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = p.parse_args()
    if args.fig3:
        out = plot_figure3_cross_section_sr(args.fig3, out_path=args.out)
        print(f"Wrote: {out}")
    else:
        p.print_help()


if __name__ == "__main__":
    _main()
