"""
visualize_kernel_weights.py — Standalone diagnostic for kernel weights.

For a chosen cross-section and kernel, computes the (T_test x T_train) weight
matrix and produces three plots:
  1. Heatmap       — all 276 test months × all 360 training months
  2. Line profiles — weight vectors for a handful of selected test months
  3. Effective-N   — how many 'equivalent' training months the kernel uses per
                     test month  (n_eff = 1 / sum(w_t^2))

Works with any kernel from part_2_AP_pruning.kernels.  Switch KERNEL in the
CONFIG block; the uniform kernel is a useful sanity check (flat heatmap).

Usage
-----
     python -m part_4_plots.visualize_kernel_weights

No pipeline re-run needed — reads h from full_fit_summary_k{K}.csv which is
written by kernel_full_fit at the end of the main run.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Kernel imports — use the same classes as the main pipeline
from part_2_AP_pruning.kernels.gaussian     import GaussianKernel
from part_2_AP_pruning.kernels.uniform      import UniformKernel
from part_2_AP_pruning.kernels.exponential  import ExponentialKernel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FEAT1         = "Investment"
FEAT2         = "LTurnover"
K             = 10
N_TRAIN_VALID = 360

STATE_CSV        = Path("data/state_variables.csv")
GRID_SEARCH_PATH = Path("data/results/grid_search/tree")
OUTPUT_DIR       = Path("data/results/diagnostics/kernel_weights")

Y_MIN, Y_MAX = 1964, 2016

STATE_COL_MAP = {
    "gaussian":     "svar",
    "gaussian-tms": "TMS",
    "exponential":  "svar",
}

# ── Kernel selection ─────────────────────────────────────────────────────────
# Choose one.  For kernels with bandwidth, h is read automatically from the
# full_fit summary CSV written by the pipeline.  For UniformKernel, h=None.

KERNEL_NAME = "gaussian-tms"    # must match the subfolder name in GRID_SEARCH_PATH
                             # "gaussian" | "uniform" | "exponential"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_kernel(kernel_name: str, feat1: str, feat2: str, k: int):
    if kernel_name == "uniform":
        return UniformKernel()

    subdir  = f"LME_{feat1}_{feat2}"
    summary = (
        GRID_SEARCH_PATH / kernel_name / subdir / "full_fit"
        / f"full_fit_summary_k{k}.csv"
    )
    if not summary.exists():
        raise FileNotFoundError(
            f"Summary not found:\n  {summary}\n"
            "Run the main pipeline first (kernel_full_fit must have completed)."
        )
    row = pd.read_csv(summary).iloc[0]
    h   = float(row["h"])
    print(f"  Loaded best h = {h:.6f}  ({summary})", flush=True)

    if kernel_name in ("gaussian", "gaussian-tms"):
        return GaussianKernel(h=h)
    if kernel_name == "exponential":
        return ExponentialKernel(lam=h, m=N_TRAIN_VALID)

    raise ValueError(f"Unknown kernel_name: '{kernel_name}'")


def _generate_month_labels(y_min: int, y_max: int) -> list[str]:
    labels = []
    for y in range(y_min, y_max + 1):
        for m in range(1, 13):
            labels.append(f"{y}-{m:02d}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Core: build the (T_test × T_train) weight matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_weight_matrix(state: pd.Series, kernel, n_train_valid: int) -> np.ndarray:
    """
    For every test month t*, compute kernel weights over all training months.

    Returns
    -------
    W : (T_test, T_train) ndarray — each row sums to 1
    """
    state_arr   = state.values
    state_train = state_arr[:n_train_valid]
    state_test  = state_arr[n_train_valid:]
    T_test      = len(state_test)

    W = np.zeros((T_test, n_train_valid))
    for t, s_cur in enumerate(state_test):
        W[t] = kernel.weights(state_train, float(s_cur))

    return W


# ─────────────────────────────────────────────────────────────────────────────
# Save weight matrix as CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_weight_csv(W, state, all_labels, n_train_valid, output_dir,
                    feat1, feat2, kernel_name):
    """
    Save the (T_test × T_train) weight matrix as a self-contained CSV.

    Structure
    ---------
    Rows   : one per test month (276 rows)
    Columns:
        date        — test month label  (e.g. '1990-01')
        s_current   — state value of the test month (the kernel 'query')
        <T_train columns> labelled  'YYYY-MM (s=X.XXXXXX)'
                    — each cell is the kernel weight assigned to that
                      training month given the test month's state
    """
    state_arr   = state.values
    state_train = state_arr[:n_train_valid]
    state_test  = state_arr[n_train_valid:]

    train_labels = all_labels[:n_train_valid]
    test_labels  = all_labels[n_train_valid : n_train_valid + len(state_test)]

    # Column headers for the 360 training months including their state value
    train_cols = [
        f"{lbl} (s={s:.6f})"
        for lbl, s in zip(train_labels, state_train)
    ]

    df = pd.DataFrame(W, columns=train_cols)
    df.insert(0, "date",      test_labels)
    df.insert(1, "s_current", state_test)

    out = output_dir / f"weights_LME_{feat1}_{feat2}_{kernel_name}.csv"
    df.to_csv(out, index=False)
    print(f"  Weight CSV saved      → {out}", flush=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Heatmap  (T_test × T_train)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(W, train_labels, test_labels, output_dir, feat1, feat2, kernel_name):
    T_test, T_train = W.shape

    x_step  = max(1, T_train // 12)
    y_step  = max(1, T_test  // 12)
    x_ticks = list(range(0, T_train, x_step))
    y_ticks = list(range(0, T_test,  y_step))

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(W, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([train_labels[i] for i in x_ticks],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([test_labels[i] for i in y_ticks], fontsize=7)

    ax.set_xlabel("Training month (kernel context)", fontsize=11)
    ax.set_ylabel("Test month (prediction target)", fontsize=11)
    ax.set_title(
        f"{kernel_name.capitalize()} kernel weights — LME_{feat1}_{feat2}\n"
        f"Each row: how the {T_train} training months are weighted "
        f"for that test month's moment estimation",
        fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Kernel weight", fontsize=10)

    fig.tight_layout()
    out = output_dir / f"heatmap_LME_{feat1}_{feat2}_{kernel_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Heatmap saved         → {out}", flush=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Weight profiles for selected test months
# ─────────────────────────────────────────────────────────────────────────────

def plot_weight_profiles(W, train_labels, test_labels, output_dir,
                         feat1, feat2, kernel_name, n_lines=6):
    T_test, T_train = W.shape

    # Pick evenly-spaced test months + the most concentrated one
    selected = list(np.linspace(0, T_test - 1, n_lines, dtype=int))
    most_concentrated = int(W.max(axis=1).argmax())
    if most_concentrated not in selected:
        selected[-1] = most_concentrated

    x_step  = max(1, T_train // 10)
    x_ticks = list(range(0, T_train, x_step))

    fig, ax = plt.subplots(figsize=(13, 5))
    cmap = plt.colormaps["plasma"]
    colors  = [cmap(i / max(len(selected) - 1, 1)) for i in range(len(selected))]

    for color, t_idx in zip(colors, selected):
        label = test_labels[t_idx] if t_idx < len(test_labels) else f"test[{t_idx}]"
        ax.plot(W[t_idx], lw=1.2, alpha=0.85, color=color, label=label)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([train_labels[i] for i in x_ticks],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Training month (kernel context)", fontsize=11)
    ax.set_ylabel("Kernel weight", fontsize=11)
    ax.set_title(
        f"Kernel weight profiles — LME_{feat1}_{feat2}  [{kernel_name}]\n"
        f"Each line: weight vector for one test month over all {T_train} training months",
        fontsize=11,
    )
    ax.legend(title="Test month", fontsize=8, title_fontsize=9,
              loc="upper right", framealpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    fig.tight_layout()
    out = output_dir / f"profiles_LME_{feat1}_{feat2}_{kernel_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Weight profiles saved → {out}", flush=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Effective sample size over test months
# ─────────────────────────────────────────────────────────────────────────────

def plot_effective_n(W, test_labels, output_dir, feat1, feat2, kernel_name):
    T_test, T_train = W.shape
    n_eff = 1.0 / np.sum(W ** 2, axis=1)   # (T_test,)

    x_step  = max(1, T_test // 12)
    x_ticks = list(range(0, T_test, x_step))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(n_eff, lw=1.4, color="steelblue", label="$n_{eff}$")
    ax.axhline(T_train, ls="--", color="grey", lw=1,
               label=f"Uniform baseline (n={T_train})")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([test_labels[i] for i in x_ticks],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Effective sample size  $n_{eff} = 1/\\Sigma w_t^2$", fontsize=11)
    ax.set_xlabel("Test month", fontsize=11)
    ax.set_title(
        f"Effective sample size of kernel — LME_{feat1}_{feat2}  [{kernel_name}]\n"
        f"Lower = kernel more concentrated; flat at {T_train} = uniform",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = output_dir / f"effective_n_LME_{feat1}_{feat2}_{kernel_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Effective-N saved     → {out}", flush=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(
    feat1: str       = FEAT1,
    feat2: str       = FEAT2,
    kernel_name: str = KERNEL_NAME,
    k: int           = K,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Kernel weight diagnostic — LME_{feat1}_{feat2}  kernel={kernel_name}  k={k}",
          flush=True)

    # 1. Instantiate kernel (reads best h from summary CSV if needed)
    kernel = _load_kernel(kernel_name, feat1, feat2, k)
    print(f"  Kernel: {kernel!r}", flush=True)

    # 2. Load state variable
    state_df = pd.read_csv(STATE_CSV, index_col="MthCalDt", parse_dates=True)
    state = state_df[STATE_COL_MAP.get(kernel_name, "svar")]
    print(f"  State variable loaded: {len(state)} months", flush=True)

    # State diagnostics
    state_train = state.iloc[:N_TRAIN_VALID]
    sigma_s = state_train.std()
    mean_s  = state_train.mean()
    print(f"  Training window svar — mean={mean_s:.6f}  std(sigma_s)={sigma_s:.6f}",
          flush=True)
    h = getattr(kernel, 'h', None)
    if h is not None:
        print(f"  Bandwidth h={h:.6f}  →  h/sigma_s={h/sigma_s:.3f}x  "
              f"(multiplier selected on validation set)", flush=True)
    else:
        print("  No bandwidth (UniformKernel)", flush=True)

    # 3. Month labels for axis ticks
    all_labels   = _generate_month_labels(Y_MIN, Y_MAX)
    train_labels = all_labels[:N_TRAIN_VALID]
    test_labels  = all_labels[N_TRAIN_VALID:]

    # 4. Build weight matrix  (T_test × T_train)
    print("  Computing weight matrix...", flush=True)
    W = compute_weight_matrix(state, kernel, N_TRAIN_VALID)
    print(f"  W shape: {W.shape}  (T_test × T_train)", flush=True)
    print(f"  Row sums — min={W.sum(axis=1).min():.6f}  max={W.sum(axis=1).max():.6f}",
          flush=True)

    # 5. Save CSV
    save_weight_csv(W, state, all_labels, N_TRAIN_VALID, OUTPUT_DIR, feat1, feat2, kernel_name)

    # 6. Plots
    plot_heatmap(W, train_labels, test_labels, OUTPUT_DIR, feat1, feat2, kernel_name)
    plot_weight_profiles(W, train_labels, test_labels, OUTPUT_DIR, feat1, feat2, kernel_name)
    plot_effective_n(W, test_labels, OUTPUT_DIR, feat1, feat2, kernel_name)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()