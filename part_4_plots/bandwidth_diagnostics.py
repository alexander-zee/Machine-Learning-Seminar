"""
plot_bandwidth_diagnostics.py
==============================
Two bandwidth diagnostic plots for Gaussian kernel(s):

Plot A — Scatter: for each cross-section, plot the winning bandwidth multiplier
  (h/sigma_s) against the out-of-sample test SR gain vs uniform baseline.
  Optionally plot both Gaussian (SVAR) and Gaussian (TMS) on the same axes.

Plot B — Single cross-section: for one cross-section, plot validation SR at
  k=10 vs bandwidth multiplier (h/sigma_s), with one line per lambda0.

Usage
-----
    python -m part_4_plots.plot_bandwidth_diagnostics

CONFIG block below:
  - Set KERNELS_A to ["gaussian"] for SVAR only, or
    ["gaussian", "gaussian-tms"] to show both on Plot A.
  - Swap FEAT1_B / FEAT2_B / KERNEL_B for Plot B.
"""

from __future__ import annotations
from pathlib import Path
from itertools import combinations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Plot A — which kernels to include. Use one or both:
#   ["gaussian"]                   — SVAR only
#   ["gaussian", "gaussian-tms"]   — both side by side
KERNELS_A = ["gaussian", "gaussian-tms"]

# Plot B — single cross-section bandwidth curve
KERNEL_B = "gaussian-tms"
FEAT1_B  = "Investment"
FEAT2_B  = "LTurnover"

K_TARGET = 10
CV_NAME  = "cv_3"

CHARACTERISTICS = [
    'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]

CHAR_LABELS = {
    'BEME': 'Val', 'r12_2': 'Mom', 'OP': 'Prof', 'Investment': 'Inv',
    'ST_Rev': 'SRev', 'LT_Rev': 'LRev', 'AC': 'Acc',
    'LTurnover': 'Turn', 'IdioVol': 'IVol',
}

# Visual style per kernel for Plot A
KERNEL_STYLES = {
    "gaussian": {
        "label_pos": "Gaussian (SVAR) — outperforms",
        "label_neg": "Gaussian (SVAR) — underperforms",
        "color_pos": "#2980B9",
        "color_neg": "#E07B54",
        "marker":    "o",
        "offset":    (6, 0),       # annotation offset (x, y) in points
    },
    "gaussian-tms": {
        "label_pos": "Gaussian (TMS) — outperforms",
        "label_neg": "Gaussian (TMS) — underperforms",
        "color_pos": "#27AE60",
        "color_neg": "#9B6B9E",
        "marker":    "^",
        "offset":    (-80, 0),
    },
}

GRID_SEARCH_PATH = Path("data/results/grid_search/tree")
OUTPUT_DIR       = Path("data/results/diagnostics/bandwidth_diagnostics")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_manifest(kernel: str, subdir: str) -> dict:
    path = GRID_SEARCH_PATH / kernel / subdir / "grid_manifest.json"
    with open(path) as f:
        return json.load(f)

def load_kernel_summary(kernel: str, subdir: str) -> dict | None:
    path = (GRID_SEARCH_PATH / kernel / subdir / "full_fit"
            / f"full_fit_summary_k{K_TARGET}.csv")
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0]
    return {"test_SR": float(row["test_SR"]), "h": float(row["h"])}

def load_uniform_sr(subdir: str) -> float | None:
    path = (GRID_SEARCH_PATH / "uniform" / subdir / "full_fit"
            / f"full_fit_summary_k{K_TARGET}.csv")
    if not path.exists():
        return None
    return float(pd.read_csv(path).iloc[0]["test_SR"])

def get_valid_sr(kernel: str, subdir: str, i_l0: int,
                 i_l2: int, i_h: int) -> float | None:
    path = (GRID_SEARCH_PATH / kernel / subdir /
            f"results_{CV_NAME}_l0_{i_l0}_l2_{i_l2}_h_{i_h}.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["valid_SR", "portsN"])
    row = df[df["portsN"] == K_TARGET]
    if row.empty:
        return None
    return float(row["valid_SR"].iloc[0])

def get_sigma_s(kernel: str, subdirs: list[str]) -> tuple[list, list, float]:
    """Return bandwidths, multipliers, sigma_s from the first available manifest."""
    for sd in subdirs:
        try:
            m = load_manifest(kernel, sd)
            bw = m["bandwidths"]
            sig = bw[0] / 0.05
            return bw, [h / sig for h in bw], sig
        except FileNotFoundError:
            continue
    raise RuntimeError(f"No grid_manifest.json found for kernel={kernel}")

def load_records(kernel: str, subdirs: list, pairs: list,
                 sigma_s: float) -> pd.DataFrame:
    """Load test SR, winning h, gain vs uniform for all cross-sections."""
    records = []
    for sd, (f1, f2) in zip(subdirs, pairs):
        k_res = load_kernel_summary(kernel, sd)
        u_sr  = load_uniform_sr(sd)
        if k_res is None or u_sr is None:
            continue
        h_win = k_res["h"]
        records.append({
            "subdir":     sd,
            "feat1":      f1,
            "feat2":      f2,
            "label":      f"Size×{CHAR_LABELS.get(f1,f1)}×{CHAR_LABELS.get(f2,f2)}",
            "test_SR":    k_res["test_SR"],
            "uniform_SR": u_sr,
            "gain":       k_res["test_SR"] - u_sr,
            "h_win":      h_win,
            "multiplier": h_win / sigma_s,
        })
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
pairs   = list(combinations(CHARACTERISTICS, 2))
subdirs = [f"LME_{f1}_{f2}" for f1, f2 in pairs]

# Use the first kernel in KERNELS_A as the reference for bandwidth grid
ref_kernel                    = KERNELS_A[0]
bandwidths, multipliers, sig_s = get_sigma_s(ref_kernel, subdirs)
n_h                           = len(bandwidths)

print(f"Reference kernel: {ref_kernel}")
print(f"Bandwidths:   {[f'{h:.5f}' for h in bandwidths]}")
print(f"Multipliers:  {[f'{m:.2f}' for m in multipliers]}")

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    ":",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "figure.dpi":        150,
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot A — Scatter: winning multiplier vs test SR gain vs uniform
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

# Vertical grid lines at candidate bandwidths
for m in multipliers:
    ax.axvline(m, color="#cccccc", lw=0.8, ls=":", zorder=1)
ax.axhline(0, color="#333333", lw=1.2, ls="--", label="No gain vs uniform",
           zorder=2)

for kernel in KERNELS_A:
    style = KERNEL_STYLES[kernel]
    _, _, ks = get_sigma_s(kernel, subdirs)
    df = load_records(kernel, subdirs, pairs, ks)
    print(f"  {kernel}: {len(df)} cross-sections loaded", flush=True)

    pos = df[df["gain"] >= 0]
    neg = df[df["gain"] <  0]

    # Slight horizontal jitter when two kernels overlap on the same multiplier
    jitter = 0.02 if len(KERNELS_A) > 1 and kernel == KERNELS_A[1] else 0.0

    ax.scatter(pos["multiplier"] * (1 + jitter), pos["gain"],  # type: ignore
               color=style["color_pos"], marker=style["marker"],
               s=55, alpha=0.8, zorder=4, label=style["label_pos"])
    ax.scatter(neg["multiplier"] * (1 + jitter), neg["gain"],  # type: ignore
               color=style["color_neg"], marker=style["marker"],
               s=55, alpha=0.8, zorder=4, label=style["label_neg"])

    # Annotate underperformers
    ox, oy = style["offset"]
    for _, row in df[df["gain"] < -0.05].iterrows():
        ax.annotate(
            row["label"],
            xy=(row["multiplier"] * (1 + jitter), row["gain"]),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=6.5, color=style["color_neg"], va="center",
        )

ax.set_xscale("log")
ax.set_xticks(multipliers)
ax.set_xticklabels([f"{m:.2f}$\\sigma_s$" for m in multipliers])
ax.set_xlabel("Winning bandwidth multiplier  $h^* / \\sigma_s$  (log scale)",
              labelpad=6)
ax.set_ylabel(f"Out-of-sample SR gain vs Uniform  ($k={K_TARGET}$)", labelpad=6)

kernel_label = " and ".join(
    "Gaussian (SVAR)" if k == "gaussian" else "Gaussian (TMS)"
    for k in KERNELS_A
)
ax.set_title(
    f"(A)  Winning bandwidth vs out-of-sample SR gain — all 36 cross-sections\n"
    f"{kernel_label}, $k={K_TARGET}$. "
    "Each dot is one cross-section; bandwidth selected on validation.",
    loc="left", fontsize=10, fontweight="bold", pad=8)
ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#cccccc", ncol=2)

fig.tight_layout()
suffix = "_".join(KERNELS_A)
outA   = OUTPUT_DIR / f"A_bandwidth_gain_scatter_{suffix}.png"
fig.savefig(outA, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"  Saved {outA}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot B — Validation SR vs bandwidth for one cross-section
# ─────────────────────────────────────────────────────────────────────────────
subdir_b   = f"LME_{FEAT1_B}_{FEAT2_B}"
manifest_b = load_manifest(KERNEL_B, subdir_b)
lambda0s   = manifest_b["lambda0"]
n_l0       = len(lambda0s)
n_l2       = len(manifest_b["lambda2"])
bw_b, mult_b, sig_b = get_sigma_s(KERNEL_B, subdirs)

records_b = []
for i_l0 in range(1, n_l0 + 1):
    for h_idx in range(1, len(bw_b) + 1):
        vals = []
        for i_l2 in range(1, n_l2 + 1):
            v = get_valid_sr(KERNEL_B, subdir_b, i_l0, i_l2, h_idx)
            if v is not None and np.isfinite(v):
                vals.append(v)
        records_b.append({
            "l0_idx":     i_l0,
            "lambda0":    lambda0s[i_l0 - 1],
            "h_idx":      h_idx,
            "multiplier": mult_b[h_idx - 1],
            "valid_SR":   max(vals) if vals else np.nan,
        })

df_b       = pd.DataFrame(records_b)
u_sr_b     = load_uniform_sr(subdir_b)
k_res_b    = load_kernel_summary(KERNEL_B, subdir_b)
h_win_b    = k_res_b["h"] if k_res_b else None
mult_win_b = (h_win_b / sig_b) if h_win_b else None

fig, ax = plt.subplots(figsize=(9, 6))

colors_l0 = ["#2980B9", "#E07B54", "#27AE60"]
markers    = ["o", "s", "^"]

for i_l0, (color, marker) in enumerate(zip(colors_l0, markers), 1):
    sub = df_b[df_b["l0_idx"] == i_l0].sort_values("multiplier")
    l0v = lambda0s[i_l0 - 1]
    ax.plot(sub["multiplier"], sub["valid_SR"],
            color=color, marker=marker, lw=1.8, ms=6,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.5,
            label=f"$\\lambda_0 = {l0v}$")

if u_sr_b is not None:
    ax.axhline(u_sr_b, color="#888888", lw=1.2, ls="--",
               label=f"Uniform test SR = {u_sr_b:.2f}")

if mult_win_b is not None:
    ax.axvline(mult_win_b, color="#9B6B9E", lw=1.2, ls=":",
               label=f"Selected $h^*$ = {mult_win_b:.2f}$\\sigma_s$")

kernel_b_label = "Gaussian (SVAR)" if KERNEL_B == "gaussian" else "Gaussian (TMS)"
ax.set_xscale("log")
ax.set_xticks(mult_b)
ax.set_xticklabels([f"{m:.2f}" for m in mult_b])
ax.set_xlabel("Bandwidth multiplier  $h / \\sigma_s$  (log scale)", labelpad=6)
ax.set_ylabel(f"Validation SR at $k={K_TARGET}$  (best $\\lambda_2$)", labelpad=6)
ax.set_title(
    f"(B)  Validation SR vs bandwidth — "
    f"LME $\\times$ {CHAR_LABELS.get(FEAT1_B, FEAT1_B)} $\\times$ "
    f"{CHAR_LABELS.get(FEAT2_B, FEAT2_B)}\n"
    f"{kernel_b_label} — one line per $\\lambda_0$, "
    f"best $\\lambda_2$ per point. Purple line = selected $h^*$.",
    loc="left", fontsize=10, fontweight="bold", pad=8)
ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")

fig.tight_layout()
outB = OUTPUT_DIR / f"B_bandwidth_single_{subdir_b}_{KERNEL_B}.png"
fig.savefig(outB, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"  Saved {outB}", flush=True)

print("\nDone. Both plots saved to", OUTPUT_DIR, flush=True)