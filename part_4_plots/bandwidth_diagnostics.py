"""
plot_bandwidth_diagnostics.py
==============================
Two bandwidth diagnostic plots for the Gaussian (SVAR) kernel:

Plot A — Scatter: for each cross-section, plot the winning bandwidth multiplier
  (h/sigma_s) against the out-of-sample test SR gain vs uniform baseline.
  One dot per cross-section. Answers: do cross-sections that selected a
  narrower h tend to perform worse out of sample?

Plot B — Single cross-section: for one cross-section, plot validation SR at
  k=10 vs bandwidth multiplier (h/sigma_s), with one line per lambda0.
  Shows the non-monotone relationship between bandwidth and performance.

Usage
-----
    python -m part_4_plots.plot_bandwidth_diagnostics

CONFIG block below — swap FEAT1_B/FEAT2_B for Plot B.
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
KERNEL_NAME = "gaussian"
K_TARGET    = 10
CV_NAME     = "cv_3"

# Plot B cross-section
FEAT1_B = "BEME"
FEAT2_B = "LT_Rev"

CHARACTERISTICS = [
    'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]

CHAR_LABELS = {
    'BEME': 'Val', 'r12_2': 'Mom', 'OP': 'Prof', 'Investment': 'Inv',
    'ST_Rev': 'SRev', 'LT_Rev': 'LRev', 'AC': 'Acc',
    'LTurnover': 'Turn', 'IdioVol': 'IVol',
}

GRID_SEARCH_PATH = Path("data/results/grid_search/tree")
OUTPUT_DIR       = Path("data/results/diagnostics/bandwidth_diagnostics")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_manifest(subdir: str) -> dict:
    path = GRID_SEARCH_PATH / KERNEL_NAME / subdir / "grid_manifest.json"
    with open(path) as f:
        return json.load(f)

def load_kernel_summary(subdir: str) -> dict | None:
    """Load test SR and winning h from full_fit_summary."""
    path = (GRID_SEARCH_PATH / KERNEL_NAME / subdir / "full_fit"
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

def get_valid_sr(subdir: str, i_l0: int, i_l2: int, i_h: int) -> float | None:
    path = (GRID_SEARCH_PATH / KERNEL_NAME / subdir /
            f"results_{CV_NAME}_l0_{i_l0}_l2_{i_l2}_h_{i_h}.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["valid_SR", "portsN"])
    row = df[df["portsN"] == K_TARGET]
    if row.empty:
        return None
    return float(row["valid_SR"].iloc[0])

# ─────────────────────────────────────────────────────────────────────────────
# Load manifest to get sigma_s and multipliers
# ─────────────────────────────────────────────────────────────────────────────
pairs   = list(combinations(CHARACTERISTICS, 2))
subdirs = [f"LME_{f1}_{f2}" for f1, f2 in pairs]

manifest = None
for sd in subdirs:
    try:
        manifest = load_manifest(sd)
        break
    except FileNotFoundError:
        continue

if manifest is None:
    raise RuntimeError("No grid_manifest.json found.")

bandwidths  = manifest["bandwidths"]
n_h         = len(bandwidths)
sigma_s     = bandwidths[0] / 0.05   # h_min = 0.05 * sigma_s
multipliers = [h / sigma_s for h in bandwidths]

print(f"Bandwidths:   {[f'{h:.5f}' for h in bandwidths]}")
print(f"Multipliers:  {[f'{m:.2f}' for m in multipliers]}")

# ─────────────────────────────────────────────────────────────────────────────
# Load test SR, winning h, uniform SR for all 36 cross-sections  (Plot A)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading test SRs for {len(subdirs)} cross-sections...", flush=True)

records_a = []
for sd, (f1, f2) in zip(subdirs, pairs):
    k_res = load_kernel_summary(sd)
    u_sr  = load_uniform_sr(sd)
    if k_res is None or u_sr is None:
        print(f"  [SKIP] {sd} — missing summary", flush=True)
        continue
    h_win  = k_res["h"]
    mult_w = h_win / sigma_s
    gain   = k_res["test_SR"] - u_sr
    records_a.append({
        "subdir":      sd,
        "feat1":       f1,
        "feat2":       f2,
        "label":       f"Size×{CHAR_LABELS.get(f1,f1)}×{CHAR_LABELS.get(f2,f2)}",
        "test_SR":     k_res["test_SR"],
        "uniform_SR":  u_sr,
        "gain":        gain,
        "h_win":       h_win,
        "multiplier":  mult_w,
    })

df_a = pd.DataFrame(records_a)
print(f"  Loaded {len(df_a)} cross-sections", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load validation SR vs h for Plot B
# ─────────────────────────────────────────────────────────────────────────────
subdir_b   = f"LME_{FEAT1_B}_{FEAT2_B}"
manifest_b = load_manifest(subdir_b)
lambda0s   = manifest_b["lambda0"]
n_l0       = len(lambda0s)
n_l2       = len(manifest_b["lambda2"])

records_b = []
for i_l0 in range(1, n_l0 + 1):
    for h_idx in range(1, n_h + 1):
        vals = []
        for i_l2 in range(1, n_l2 + 1):
            v = get_valid_sr(subdir_b, i_l0, i_l2, h_idx)
            if v is not None and np.isfinite(v):
                vals.append(v)
        records_b.append({
            "l0_idx":     i_l0,
            "lambda0":    lambda0s[i_l0 - 1],
            "h_idx":      h_idx,
            "multiplier": multipliers[h_idx - 1],
            "valid_SR":   max(vals) if vals else np.nan,
        })

df_b       = pd.DataFrame(records_b)
u_sr_b     = load_uniform_sr(subdir_b)
k_res_b    = load_kernel_summary(subdir_b)
h_win_b    = k_res_b["h"] if k_res_b else None
mult_win_b = (h_win_b / sigma_s) if h_win_b else None

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
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "figure.dpi":        150,
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot A — Scatter: winning multiplier vs test SR gain vs uniform
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

pos_mask = df_a["gain"] >= 0
neg_mask = df_a["gain"] <  0

ax.scatter(df_a.loc[pos_mask, "multiplier"], df_a.loc[pos_mask, "gain"],  # type: ignore
           color="#2980B9", s=50, alpha=0.8, zorder=4,
           label="Kernel outperforms uniform")
ax.scatter(df_a.loc[neg_mask, "multiplier"], df_a.loc[neg_mask, "gain"],  # type: ignore
           color="#E07B54", s=50, alpha=0.8, zorder=4,
           label="Kernel underperforms uniform")

# Label cross-sections with gain below -0.05
for _, row in df_a[df_a["gain"] < -0.05].iterrows():
    ax.annotate(
        row["label"],
        xy=(row["multiplier"], row["gain"]),
        xytext=(6, 0), textcoords="offset points",
        fontsize=7, color="#E07B54", va="center",
    )

ax.axhline(0, color="#333333", lw=1.2, ls="--", label="No gain vs uniform")

# Mark the 5 candidate multipliers as vertical guides
for m in multipliers:
    ax.axvline(m, color="#cccccc", lw=0.8, ls=":", zorder=1)

ax.set_xscale("log")
ax.set_xticks(multipliers)
ax.set_xticklabels([f"{m:.2f}$\\sigma_s$" for m in multipliers])
ax.set_xlabel("Winning bandwidth multiplier  $h^* / \\sigma_s$  (log scale)",
              labelpad=6)
ax.set_ylabel(f"Out-of-sample SR gain vs Uniform  ($k={K_TARGET}$)", labelpad=6)
ax.set_title(
    "(A)  Winning bandwidth vs out-of-sample SR gain — all 36 cross-sections\n"
    f"Gaussian (SVAR) kernel, $k={K_TARGET}$.  "
    "Each dot is one cross-section; bandwidth selected on validation.",
    loc="left", fontsize=10, fontweight="bold", pad=8)
ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")

fig.tight_layout()
outA = OUTPUT_DIR / f"A_bandwidth_gain_scatter_{KERNEL_NAME}.png"
fig.savefig(outA, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"  Saved {outA}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot B — Validation SR vs bandwidth for one cross-section
# ─────────────────────────────────────────────────────────────────────────────
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

# Mark winning bandwidth
if mult_win_b is not None:
    ax.axvline(mult_win_b, color="#9B6B9E", lw=1.2, ls=":",
               label=f"Selected $h^*$ = {mult_win_b:.2f}$\\sigma_s$")

ax.set_xscale("log")
ax.set_xticks(multipliers)
ax.set_xticklabels([f"{m:.2f}" for m in multipliers])
ax.set_xlabel("Bandwidth multiplier  $h / \\sigma_s$  (log scale)", labelpad=6)
ax.set_ylabel(f"Validation SR at $k={K_TARGET}$  (best $\\lambda_2$)", labelpad=6)
ax.set_title(
    f"(B)  Validation SR vs bandwidth — "
    f"LME $\\times$ {CHAR_LABELS.get(FEAT1_B, FEAT1_B)} $\\times$ "
    f"{CHAR_LABELS.get(FEAT2_B, FEAT2_B)}\n"
    f"Gaussian (SVAR) kernel — one line per $\\lambda_0$, "
    f"best $\\lambda_2$ per point. Purple line = selected $h^*$.",
    loc="left", fontsize=10, fontweight="bold", pad=8)

ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")

fig.tight_layout()
outB = OUTPUT_DIR / f"B_bandwidth_single_{subdir_b}_{KERNEL_NAME}.png"
fig.savefig(outB, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"  Saved {outB}", flush=True)

print("\nDone. Both plots saved to", OUTPUT_DIR, flush=True)