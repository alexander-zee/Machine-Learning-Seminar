"""
tc_batch.py
-----------
Run transaction cost diagnostics for all kernels × all 36 cross-sections.

For each kernel the function finds the appropriate detail file, calls the
correct entry point (compute_net_sharpe or compute_net_sharpe_uniform), and
saves per-cross-section results next to the existing full_fit files.

At the end a combined summary CSV is written per kernel:
    data/results/grid_search/tree/{kernel}/tc_summary_all_k{k}.csv

Usage
-----
    python -m part_3_metrics_collection.tc_batch_runner
"""

from itertools import combinations
from pathlib import Path
import traceback

import pandas as pd

from part_3_metrics_collection.transaction_costs import (
    compute_net_sharpe,
    compute_net_sharpe_uniform,
)

# ── Config ─────────────────────────────────────────────────────────────────────

CHARACTERISTICS = [
    'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]

PAIRS            = list(combinations(CHARACTERISTICS, 2))
PORT_N           = 10
GRID_SEARCH_PATH = Path('data/results/grid_search/tree')
PANEL_PATH       = Path('data/prepared/panel.parquet')
N_TRAIN_VALID    = 360

# Kernels that use full_fit_detail (per-month weights)
KERNEL_KERNELS = ['gaussian', 'exponential', 'gaussian-tms']

# ── Helpers ────────────────────────────────────────────────────────────────────

def subdir(feat1: str, feat2: str) -> str:
    return f'LME_{feat1}_{feat2}'


# ── Per-kernel runners ─────────────────────────────────────────────────────────

def run_kernel(kernel_name: str) -> pd.DataFrame:
    """
    Run TC diagnostics for one kernel across all 36 cross-sections.
    Returns a summary DataFrame with one row per cross-section.
    """
    print(f"\n{'═'*60}")
    print(f"  Kernel: {kernel_name}")
    print(f"{'═'*60}")

    rows = []

    for feat1, feat2 in PAIRS:
        cs = subdir(feat1, feat2)
        detail_path = (
            GRID_SEARCH_PATH / kernel_name / cs / 'full_fit'
            / f'full_fit_detail_k{PORT_N}.csv'
        )

        if not detail_path.exists():
            print(f"  [{cs}] SKIP — detail file not found: {detail_path}")
            rows.append({
                'feat1': feat1, 'feat2': feat2, 'cross_section': cs,
                'gross_SR': None, 'net_SR': None,
                'SR_loss': None, 'mean_TC': None, 'status': 'missing',
            })
            continue

        print(f"\n  [{cs}]")
        try:
            result = compute_net_sharpe(
                detail_path   = detail_path,
                panel_path    = PANEL_PATH,
                features      = ['LME', feat1, feat2],
                n_train_valid = N_TRAIN_VALID,
                label         = kernel_name,
            )
            rows.append({
                'feat1':        feat1,
                'feat2':        feat2,
                'cross_section': cs,
                'gross_SR':     result['gross_SR'],
                'net_SR':       result['net_SR'],
                'SR_loss':      result['gross_SR'] - result['net_SR'],
                'mean_TC':      float(result['tc_series'].mean()),
                'status':       'done',
            })
        except ValueError as e:
            if 'months were skipped' in str(e):
                print(f"  [{cs}] SKIP — incomplete detail file ({e})")
                rows.append({
                    'feat1': feat1, 'feat2': feat2, 'cross_section': cs,
                    'gross_SR': None, 'net_SR': None,
                    'SR_loss': None, 'mean_TC': None, 'status': 'skipped_incomplete',
                })
            else:
                print(f"  [{cs}] ERROR: {e}")
                traceback.print_exc()
                rows.append({
                    'feat1': feat1, 'feat2': feat2, 'cross_section': cs,
                    'gross_SR': None, 'net_SR': None,
                    'SR_loss': None, 'mean_TC': None, 'status': f'error: {e}',
                })
        except Exception as e:
            print(f"  [{cs}] ERROR: {e}")
            traceback.print_exc()
            rows.append({
                'feat1': feat1, 'feat2': feat2, 'cross_section': cs,
                'gross_SR': None, 'net_SR': None,
                'SR_loss': None, 'mean_TC': None, 'status': f'error: {e}',
            })

    summary = pd.DataFrame(rows)
    out_path = GRID_SEARCH_PATH / kernel_name / f'tc_summary_all_k{PORT_N}.csv'
    summary.to_csv(out_path, index=False)
    print(f"\n  Summary saved → {out_path}")
    return summary


def run_uniform() -> pd.DataFrame:
    """
    Run TC diagnostics for the uniform kernel across all 36 cross-sections.
    """
    kernel_name = 'uniform'
    print(f"\n{'═'*60}")
    print(f"  Kernel: {kernel_name}")
    print(f"{'═'*60}")

    rows = []

    for feat1, feat2 in PAIRS:
        cs           = subdir(feat1, feat2)
        kernel_dir   = GRID_SEARCH_PATH / kernel_name / cs
        ports_path   = kernel_dir / f'Selected_Ports_{PORT_N}.csv'
        weights_path = kernel_dir / f'Selected_Ports_Weights_{PORT_N}.csv'

        if not ports_path.exists() or not weights_path.exists():
            print(f"  [{cs}] SKIP — ports/weights file not found")
            rows.append({
                'feat1': feat1, 'feat2': feat2, 'cross_section': cs,
                'gross_SR': None, 'net_SR': None,
                'SR_loss': None, 'mean_TC': None, 'status': 'missing',
            })
            continue

        print(f"\n  [{cs}]")
        try:
            result = compute_net_sharpe_uniform(
                ports_path    = ports_path,
                weights_path  = weights_path,
                panel_path    = PANEL_PATH,
                features      = ['LME', feat1, feat2],
                n_train_valid = N_TRAIN_VALID,
                label         = kernel_name,
            )
            rows.append({
                'feat1':        feat1,
                'feat2':        feat2,
                'cross_section': cs,
                'gross_SR':     result['gross_SR'],
                'net_SR':       result['net_SR'],
                'SR_loss':      result['gross_SR'] - result['net_SR'],
                'mean_TC':      float(result['tc_series'].mean()),
                'status':       'done',
            })
        except Exception as e:
            print(f"  [{cs}] ERROR: {e}")
            traceback.print_exc()
            rows.append({
                'feat1': feat1, 'feat2': feat2, 'cross_section': cs,
                'gross_SR': None, 'net_SR': None,
                'SR_loss': None, 'mean_TC': None, 'status': f'error: {e}',
            })

    summary = pd.DataFrame(rows)
    out_path = GRID_SEARCH_PATH / kernel_name / f'tc_summary_all_k{PORT_N}.csv'
    summary.to_csv(out_path, index=False)
    print(f"\n  Summary saved → {out_path}")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_summaries = {}

    for kernel in KERNEL_KERNELS:
        all_summaries[kernel] = run_kernel(kernel)

    all_summaries['uniform'] = run_uniform()

    # Print a quick comparison table across kernels for reference
    print(f"\n{'═'*60}")
    print("  DONE — median net SR per kernel:")
    for kernel, df in all_summaries.items():
        done = df[df['status'] == 'done']
        if len(done) > 0:
            print(f"    {kernel:15s}  median net SR = {done['net_SR'].median():.4f}"
                  f"  ({len(done)}/36 cross-sections)")