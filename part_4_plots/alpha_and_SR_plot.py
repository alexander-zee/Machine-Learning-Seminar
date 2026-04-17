"""
plot_sr_alpha.py
================
Produces two separate square PNG figures:
  fig_sr_panel.png     — Panel (a): out-of-sample monthly Sharpe ratios
  fig_alpha_panel.png  — Panel (b): FF5-alpha t-statistics

Cross-sections sorted by Uniform SR ascending (Uniform line is monotone).

Usage
-----
    python plot_sr_alpha.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("data/results/diagnostics")
OUTPUT_DIR  = Path("data/results/diagnostics")
K           = 10

KERNELS = {
    "uniform":      dict(label="Uniform",          color="#E07B54", marker="o", ls="-",  lw=2.2, ms=5,   zorder=5),
    "gaussian":     dict(label="Gaussian (SVAR)",  color="#5B8DB8", marker="s", ls="--", lw=1.5, ms=5,   zorder=4),
    "exponential":  dict(label="Exponential",      color="#6AB187", marker="^", ls="--", lw=1.5, ms=5,   zorder=4),
    "gaussian-tms": dict(label="Gaussian (TMS)",   color="#9B6B9E", marker="D", ls=":",  lw=1.5, ms=4.5, zorder=4),
}

CHAR_LABELS = {
    "LME": "Size", "BEME": "Val", "r12_2": "Mom", "OP": "Prof",
    "Investment": "Inv", "ST_Rev": "SRev", "LT_Rev": "LRev",
    "AC": "Acc", "LTurnover": "Turn", "IdioVol": "IVol",
}

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_kernel(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"ff5_results_{name}_k{K}.csv"
    if not path.exists():
        print(f"  [WARN] {path} not found — skipping {name}")
        return None
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["cross_section"] = df["char1"] + "_" + df["char2"] + "_" + df["char3"]
    return df

dfs = {k: load_kernel(k) for k in KERNELS}
dfs = {k: v for k, v in dfs.items() if v is not None}

# Sort order from Uniform
base    = dfs["uniform"].sort_values("sr", ascending=True).reset_index(drop=True)
base["rank"] = np.arange(1, len(base) + 1)
order   = base["cross_section"].tolist()
x_ticks = base["rank"].astype(str).tolist()
x       = np.arange(len(order))

def align(df: "pd.DataFrame | None", col: str) -> np.ndarray:
    if df is None:
        return np.full(len(order), np.nan)
    lookup = df.set_index("cross_section")[col].to_dict()
    return np.array([lookup.get(cs, np.nan) for cs in order])

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
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "figure.dpi":        150,
})

def draw_series(ax, col):
    for name, style in KERNELS.items():
        if name not in dfs:
            continue
        vals = align(dfs[name], col)
        ax.plot(x, vals,
                color=style["color"], marker=style["marker"],
                ls=style["ls"], lw=style["lw"], ms=style["ms"],
                zorder=style["zorder"], label=style["label"],
                markerfacecolor="white" if name != "uniform" else style["color"],
                markeredgecolor=style["color"], markeredgewidth=1.2)

def set_xticks(ax):
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks, rotation=0)
    ax.set_xlabel("Cross-sections (sorted by Uniform SR)", labelpad=6)

# ─────────────────────────────────────────────────────────────────────────────
# Figure (a) — Sharpe Ratios
# ─────────────────────────────────────────────────────────────────────────────
fig_a, ax_a = plt.subplots(figsize=(9, 8))
fig_a.subplots_adjust(right=0.78)   # leave room for right-side legend

draw_series(ax_a, "sr")
set_xticks(ax_a)

ax_a.set_ylabel("Monthly Sharpe Ratio (SR)", labelpad=6)
ax_a.set_title("(a)  Out-of-sample Sharpe ratios", loc="left",
               fontsize=10, fontweight="bold", pad=8)

# Legend outside to the right
ax_a.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
    framealpha=0.9,
    edgecolor="#cccccc",
    ncol=1,
)

# End-of-series SR value annotations, staggered
offsets_y = {"uniform": 0, "gaussian": 8, "exponential": 16, "gaussian-tms": -8}
for name, style in KERNELS.items():
    if name not in dfs:
        continue
    vals = align(dfs[name], "sr")
    last = vals[~np.isnan(vals)][-1]
    ax_a.annotate(
        f"{last:.2f}",
        xy=(x[-1], last),
        xytext=(4, offsets_y.get(name, 0)), textcoords="offset points",
        color=style["color"], fontsize=7.5, va="center",
        fontweight="bold" if name == "uniform" else "normal",
    )

fig_a.text(
    0.42, -0.01,
    "Note: 36 cross-sections (Size + 2 characteristics), sorted by Uniform SR ascending. "
    "$k=10$ basis assets.",
    ha="center", fontsize=7.2, color="#555555", style="italic",
)

fig_a.savefig(OUTPUT_DIR / "fig_sr_panel.png", bbox_inches="tight", dpi=200)
print("Saved fig_sr_panel.png")
plt.close(fig_a)

# ─────────────────────────────────────────────────────────────────────────────
# Figure (b) — FF5 Alpha t-statistics
# ─────────────────────────────────────────────────────────────────────────────
fig_b, ax_b = plt.subplots(figsize=(9, 8))
fig_b.subplots_adjust(right=0.78)

draw_series(ax_b, "alpha_ff5_tstat")
set_xticks(ax_b)

# Significance reference lines
for val, lbl, y_off in [(2.576, "1%", 0.25), (1.960, "5%", -0.65)]:
    ax_b.axhline(val, color="#888888", lw=0.8, ls="--", zorder=1)
    ax_b.text(x[-1] + 0.4, val + y_off, lbl, fontsize=7, color="#555555")
ax_b.axhline(0, color="#333333", lw=0.8, ls="-", zorder=1)

ax_b.set_ylabel(r"$t$-statistic ($\alpha_{\mathrm{FF5}}$)", labelpad=6)
ax_b.set_title(r"(b)  $t$-statistics of FF5 alpha", loc="left",
               fontsize=10, fontweight="bold", pad=8)

ax_b.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
    framealpha=0.9,
    edgecolor="#cccccc",
    ncol=1,
)

fig_b.text(
    0.42, -0.01,
    "Note: 36 cross-sections (Size + 2 characteristics), sorted by Uniform SR ascending. "
    "$k=10$ basis assets. Dashed lines at $t=1.96$ (5\\%) and $t=2.58$ (1\\%).",
    ha="center", fontsize=7.2, color="#555555", style="italic",
)

fig_b.savefig(OUTPUT_DIR / "fig_alpha_panel.png", bbox_inches="tight", dpi=200)
print("Saved fig_alpha_panel.png")
plt.close(fig_b)