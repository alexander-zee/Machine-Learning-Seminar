"""
plot_tc_scatter.py
------------------
Scatter plot: gross SR gain over uniform (x) vs net SR gain over uniform (y)
for all kernels x all cross-sections.

One point per (kernel, cross-section). Color = kernel. Shape = kernel.
Quadrant lines at zero. Symmetric axes. Summary table annotation.

Usage
-----
    python -m part_4_plots.plot_tc_scatter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

GRID_SEARCH_PATH = Path('data/results/grid_search/tree')
OUTPUT_PATH      = Path('data/results/figures/tc_scatter.pdf')
PORT_N           = 10

KERNELS = ['gaussian', 'exponential', 'gaussian-tms']

KERNEL_STYLE = {
    'gaussian':     {'color': '#2166ac', 'marker': 'o', 'label': 'Gaussian'},
    'exponential':  {'color': '#d6604d', 'marker': 's', 'label': 'Exponential'},
    'gaussian-tms': {'color': '#4dac26', 'marker': '^', 'label': 'Gaussian-TMS'},
}

# ── Load data ──────────────────────────────────────────────────────────────────

def load_summary(kernel: str) -> pd.DataFrame:
    path = GRID_SEARCH_PATH / kernel / f'tc_summary_all_k{PORT_N}.csv'
    if not path.exists():
        print(f"  Warning: missing summary for {kernel}: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df['status'] == 'done'].copy()
    df['kernel'] = kernel
    return df


def load_uniform() -> pd.DataFrame:
    path = GRID_SEARCH_PATH / 'uniform' / f'tc_summary_all_k{PORT_N}.csv'
    if not path.exists():
        raise FileNotFoundError(f"Uniform summary not found: {path}")
    df = pd.read_csv(path)
    df = df[df['status'] == 'done'].copy()
    return df[['cross_section', 'gross_SR', 'net_SR']].rename(
        columns={'gross_SR': 'gross_SR_uniform', 'net_SR': 'net_SR_uniform'}
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    uniform = load_uniform()

    all_rows = []
    for kernel in KERNELS:
        df = load_summary(kernel)
        if df.empty:
            continue
        df = df.merge(uniform, on='cross_section', how='inner')
        df['gross_gain'] = df['gross_SR'] - df['gross_SR_uniform']
        df['net_gain']   = df['net_SR']   - df['net_SR_uniform']
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No data loaded — check that tc_summary_all_k10.csv files exist.")

    data = pd.concat(all_rows, ignore_index=True)

    # ── Symmetric axis limit ───────────────────────────────────────────────────
    abs_max = max(
        data['gross_gain'].abs().max(),
        data['net_gain'].abs().max(),
    )
    lim = np.ceil(abs_max * 10) / 10 + 0.05

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))

    # Quadrant shading
    ax.axvspan(-lim, 0, ymin=0.5, ymax=1.0, color='#f7f7f7', zorder=0)
    ax.axvspan(0,  lim, ymin=0.5, ymax=1.0, color='#e8f4e8', zorder=0)
    ax.axvspan(-lim, 0, ymin=0.0, ymax=0.5, color='#fde8e8', zorder=0)
    ax.axvspan(0,  lim, ymin=0.0, ymax=0.5, color='#fef3e8', zorder=0)

    # Zero lines
    ax.axhline(0, color='#888888', linewidth=0.8, zorder=1)
    ax.axvline(0, color='#888888', linewidth=0.8, zorder=1)

    # Scatter per kernel
    for kernel in KERNELS:
        sub   = data[data['kernel'] == kernel]
        style = KERNEL_STYLE[kernel]
        ax.scatter(
            sub['gross_gain'], sub['net_gain'],
            color=style['color'], marker=style['marker'],
            s=60, alpha=0.8, linewidths=0.4, edgecolors='white',
            label=style['label'], zorder=3,
        )

    # Quadrant labels
    label_kw = dict(fontsize=8, color='#555555', style='italic')
    ax.text( lim * 0.97,  lim * 0.97, 'Better gross\n& net',     ha='right', va='top',    **label_kw)
    ax.text(-lim * 0.97,  lim * 0.97, 'Worse gross\nbetter net', ha='left',  va='top',    **label_kw)
    ax.text( lim * 0.97, -lim * 0.97, 'Better gross\nworse net', ha='right', va='bottom', **label_kw)
    ax.text(-lim * 0.97, -lim * 0.97, 'Worse gross\n& net',      ha='left',  va='bottom', **label_kw)

    # Axes
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Gross SR gain over uniform', fontsize=11)
    ax.set_ylabel('Net SR gain over uniform',   fontsize=11)
    ax.set_title(
        f'Transaction cost robustness: kernel vs uniform (k={PORT_N})',
        fontsize=12, pad=12,
    )

    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # ── Summary table annotation ───────────────────────────────────────────────
    lines = ['Cross-sections beating uniform']
    lines.append(f"{'Kernel':<16} {'Gross':>6} {'Net':>6}")
    lines.append('-' * 32)
    for kernel in KERNELS:
        sub     = data[data['kernel'] == kernel]
        n_gross = (sub['gross_gain'] > 0).sum()
        n_net   = (sub['net_gain']   > 0).sum()
        n_total = len(sub)
        label   = KERNEL_STYLE[kernel]['label']
        lines.append(f"{label:<16} {n_gross:>3}/{n_total:<3}  {n_net:>3}/{n_total:<3}")

    ax.text(
        0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes,
        fontsize=7.5, family='monospace',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  alpha=0.85, edgecolor='#cccccc'),
        zorder=5,
    )

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    fig.savefig(OUTPUT_PATH.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"Saved -> {OUTPUT_PATH}")
    print(f"Saved -> {OUTPUT_PATH.with_suffix('.png')}")
    plt.show()


if __name__ == '__main__':
    main()