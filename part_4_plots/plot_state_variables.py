"""
plot_state_variables.py
=======================
Plots the three state variables (SVAR, TMS, DEF) used in the kernel estimation,
with NBER recession shading, so we can see how well each tracks the business cycle.

Usage: python plot_state_variables.py
Reads:  data/state_variables.csv   (MthCalDt, svar, TMS, DEF)
Writes: data/results/diagnostics/fig_state_variables.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
STATE_CSV  = Path("data/state_variables.csv")
OUTPUT_DIR = Path("data/results/diagnostics")

# NBER recession dates (peak → trough, inclusive), 1964–2016
NBER_RECESSIONS = [
    ("1969-12", "1970-11"),
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
]

STATE_VARS = [
    dict(col="svar", label="SVAR (Realized Variance)",
         color="#C0392B", ylabel="Monthly realized variance",
         note="Episodic spikes at crises; near-zero otherwise"),
    dict(col="TMS",  label="TMS (Term Spread, 10Y − 3M)",
         color="#2980B9", ylabel="Spread (pp)",
         note="Monetary cycle; Volcker inversion 1979–82"),
    dict(col="DEF",  label="DEF (Default Spread, BAA − AAA)",
         color="#27AE60", ylabel="Spread (pp)",
         note="Credit cycle; spikes in recessions"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(STATE_CSV, parse_dates=["MthCalDt"])
df = df.sort_values("MthCalDt").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.2,
    "grid.linestyle":    ":",
    "xtick.labelsize":   8,
    "ytick.labelsize":   9,
    "axes.labelsize":    9,
    "figure.dpi":        150,
})

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                         gridspec_kw={"hspace": 0.45})

def shade_recessions(ax):
    for start, end in NBER_RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="#AAAAAA", alpha=0.25, zorder=0)

for ax, sv in zip(axes, STATE_VARS):
    if sv["col"] not in df.columns:
        ax.set_visible(False)
        continue

    shade_recessions(ax)

    ax.plot(df["MthCalDt"], df[sv["col"]],
            color=sv["color"], lw=1.2, zorder=2)

    # Rolling mean to show the trend
    roll = df[sv["col"]].rolling(12, center=True).mean()
    ax.plot(df["MthCalDt"], roll,
            color=sv["color"], lw=2.2, alpha=0.5, ls="--", zorder=3,
            label="12m rolling mean")

    ax.set_ylabel(sv["ylabel"], labelpad=4)
    ax.set_title(sv["label"], loc="left", fontsize=10,
                 fontweight="bold", pad=6)
    ax.text(0.99, 0.97, sv["note"], transform=ax.transAxes,
            fontsize=7.5, color="#555555", ha="right", va="top",
            style="italic")

    ax.legend(fontsize=7.5, loc="upper right",
              framealpha=0.8, edgecolor="#cccccc")

# x-axis: year ticks every 4 years
axes[-1].set_xlabel("Date", labelpad=5)
import matplotlib.dates as mdates
axes[-1].xaxis.set_major_locator(mdates.YearLocator(4))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=0, ha="center")

# Recession legend patch
rec_patch = mpatches.Patch(color="#AAAAAA", alpha=0.5, label="NBER recession")
fig.legend(handles=[rec_patch], loc="lower center", ncol=1,
           fontsize=8, framealpha=0.8, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle("State Variables used in Kernel Estimation  (1964–2016)",
             fontsize=11, fontweight="bold", y=1.01)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUTPUT_DIR / "fig_state_variables.png"
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"Saved {out}")
plt.close(fig)