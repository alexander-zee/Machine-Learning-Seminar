"""
diagnose_outlier.py
===================
Diagnoses numerical instability in a kernel full-fit result for one
cross-section. Produces two plots:

  1. Monthly excess returns — outliers by deviation from mean highlighted
  2. SR sensitivity — kernel vs uniform, each removing their own worst months
     by |return - mean| first. Shows whether the kernel's SR collapse is
     driven by a few concentrated outlier months vs the uniform baseline.

Note: the n_eff plot is already produced by visualize_kernel_weights.py.

All inputs are read from existing pipeline outputs — no reruns needed.

Usage
-----
    python diagnose_outlier.py

Swap the CONFIG block to analyse a different cross-section / kernel.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← swap these to analyse a different cross-section / kernel
# ─────────────────────────────────────────────────────────────────────────────
FEAT1         = "Investment"
FEAT2         = "LTurnover"
KERNEL_NAME   = "gaussian-tms"   # subfolder name under GRID_SEARCH_PATH
STATE_COL     = "svar"       # shown in plot titles only
K             = 10
N_TRAIN_VALID = 360
Y_MIN, Y_MAX  = 1964, 2016

GRID_SEARCH_PATH = Path("data/results/grid_search/tree")
OUTPUT_DIR       = Path("data/results/diagnostics/outlier_diagnosis")

MAX_REMOVE = 20   # max months to remove in sensitivity plot

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def subdir() -> str:
    return f"LME_{FEAT1}_{FEAT2}"

def full_fit_dir(kernel: str) -> Path:
    return GRID_SEARCH_PATH / kernel / subdir() / "full_fit"

def generate_dates(y_min, y_max):
    dates = []
    for y in range(y_min, y_max + 1):
        for m in range(1, 13):
            dates.append(pd.Timestamp(year=y, month=m, day=1))
    return dates

def sr_sensitivity(returns: np.ndarray, max_remove: int):
    """
    Compute SR after iteratively removing months with largest |return - mean|.
    Returns list of SR values for n=0,1,...,max_remove months removed.
    """
    mean_ret   = returns.mean()
    abs_dev    = np.abs(returns - mean_ret)
    sorted_idx = np.argsort(abs_dev)[::-1]
    sr_values  = []
    for n in range(max_remove + 1):
        keep = np.ones(len(returns), dtype=bool)
        if n > 0:
            keep[sorted_idx[:n]] = False
        r   = returns[keep]
        std = r.std(ddof=1)
        sr_values.append(float(r.mean() / std) if std > 0 else np.nan)
    return sr_values

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading data for {subdir()} — kernel={KERNEL_NAME}", flush=True)

# Kernel
k_summary   = pd.read_csv(full_fit_dir(KERNEL_NAME) / f"full_fit_summary_k{K}.csv").iloc[0]
reported_sr = float(k_summary["test_SR"])
k_returns   = pd.read_csv(full_fit_dir(KERNEL_NAME) / f"full_fit_detail_k{K}.csv")["excess_return"].values
print(f"  kernel SR = {reported_sr:.4f}", flush=True)

# Uniform
u_summary   = pd.read_csv(full_fit_dir("uniform") / f"full_fit_summary_k{K}.csv").iloc[0]
uniform_sr  = float(u_summary["test_SR"])
u_returns   = pd.read_csv(full_fit_dir("uniform") / f"full_fit_detail_k{K}.csv")["excess_return"].values
print(f"  uniform SR = {uniform_sr:.4f}", flush=True)

all_dates  = generate_dates(Y_MIN, Y_MAX)
test_dates = all_dates[N_TRAIN_VALID : N_TRAIN_VALID + len(k_returns)]

# Kernel outlier info for plot 1
k_mean      = k_returns.mean() # type: ignore #
k_abs_dev   = np.abs(k_returns - k_mean)
k_sorted    = np.argsort(k_abs_dev)[::-1]
outlier_idx = set(k_sorted[:MAX_REMOVE])

print(f"  kernel mean={k_mean:.4f}  std={k_returns.std(ddof=1):.4f}  " # type: ignore #
      f"max|dev|={k_abs_dev.max():.4f} at "
      f"{test_dates[k_abs_dev.argmax()].strftime('%Y-%m')}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    ":",
    "xtick.labelsize":   8,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "figure.dpi":        150,
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
title_prefix = f"LME x {FEAT1} x {FEAT2}  [{KERNEL_NAME}, state={STATE_COL}]"

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Monthly excess returns, coloured by deviation from mean
# ─────────────────────────────────────────────────────────────────────────────
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(11, 5))

bar_colors = ["#E07B54" if i in outlier_idx else
              ("#2980B9" if r >= 0 else "#AEC6CF")
              for i, r in enumerate(k_returns)]

ax.bar(test_dates, k_returns, color=bar_colors, alpha=0.8, width=25) # type: ignore #
ax.axhline(k_mean, color="#333333", lw=1.2, ls="--",
           label=f"Mean = {k_mean:.3f}")

legend_elements = [
    Patch(facecolor="#E07B54", alpha=0.8,
          label=f"Top {MAX_REMOVE} outliers by |return \u2212 mean|"),
    Patch(facecolor="#2980B9", alpha=0.8, label="Normal months (positive)"),
    Patch(facecolor="#AEC6CF", alpha=0.8, label="Normal months (negative)"),
    plt.Line2D([0], [0], color="#333333", lw=1.2, ls="--", # type: ignore #
               label=f"Mean = {k_mean:.3f}"),
]
ax.legend(handles=legend_elements, fontsize=8,
          framealpha=0.9, edgecolor="#cccccc", loc="upper left")

ax.set_ylabel("Monthly excess return", labelpad=6)
ax.set_xlabel("Test month", labelpad=5)
ax.set_title(
    f"(1)  Monthly excess returns \u2014 outliers by deviation from mean\n{title_prefix}",
    loc="left", fontsize=10, fontweight="bold", pad=8)
ax.xaxis.set_major_locator(mdates.YearLocator(3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.tight_layout()
out1 = OUTPUT_DIR / f"1_returns_{subdir()}_{KERNEL_NAME}.png"
fig.savefig(out1, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"  Saved {out1}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — SR sensitivity: kernel vs uniform, each removing own worst months
# ─────────────────────────────────────────────────────────────────────────────
n_removed  = list(range(0, MAX_REMOVE + 1))
k_sr_vals  = sr_sensitivity(k_returns, MAX_REMOVE) # type: ignore #
u_sr_vals  = sr_sensitivity(u_returns, MAX_REMOVE) # type: ignore #

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(n_removed, k_sr_vals, color="#2980B9", lw=2, marker="o",
        ms=5, markerfacecolor="white", markeredgecolor="#2980B9",
        markeredgewidth=1.5,
        label=f"{KERNEL_NAME.capitalize()} kernel (own outliers removed)")

ax.plot(n_removed, u_sr_vals, color="#E07B54", lw=2, marker="s",
        ms=5, markerfacecolor="white", markeredgecolor="#E07B54",
        markeredgewidth=1.5, ls="--",
        label="Uniform (own outliers removed)")

# Full-sample reference lines (at n=0)
ax.axhline(reported_sr, color="#2980B9", lw=0.8, ls=":",
           alpha=0.6, label=f"Kernel full-sample SR = {reported_sr:.2f}")
ax.axhline(uniform_sr, color="#E07B54", lw=0.8, ls=":",
           alpha=0.6, label=f"Uniform full-sample SR = {uniform_sr:.2f}")

# Annotate starting values
ax.annotate(f"{k_sr_vals[0]:.2f}", xy=(0, k_sr_vals[0]),
            xytext=(0.4, k_sr_vals[0] + 0.01),
            fontsize=8, color="#2980B9")
ax.annotate(f"{u_sr_vals[0]:.2f}", xy=(0, u_sr_vals[0]),
            xytext=(0.4, u_sr_vals[0] - 0.03),
            fontsize=8, color="#E07B54")

ax.set_xlabel("Number of months removed (largest |return \u2212 mean| first)",
              labelpad=6)
ax.set_ylabel("Out-of-sample Sharpe Ratio", labelpad=6)
ax.set_title(
    f"(2)  SR sensitivity \u2014 kernel vs uniform (each removing own outliers)\n{title_prefix}",
    loc="left", fontsize=10, fontweight="bold", pad=8)
ax.set_xticks(n_removed)
ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#cccccc")

fig.tight_layout()
out2 = OUTPUT_DIR / f"2_sr_sensitivity_{subdir()}_{KERNEL_NAME}.png"
fig.savefig(out2, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"  Saved {out2}", flush=True)

print("\nDone. Both plots saved to", OUTPUT_DIR, flush=True)