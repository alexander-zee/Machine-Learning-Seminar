"""
tc_batch.py
-----------
Run transaction cost diagnostics for all kernels × all 36 cross-sections.

All kernels (including uniform) use compute_net_sharpe with their
full_fit_detail_k{k}.csv — same format, same entry point.

Summary per kernel saved to:
    data/results/grid_search/tree/{kernel}/tc_summary_all_k{k}.csv

Usage
-----
    python -m part_3_metrics_collection.tc_batch_runner
"""

from itertools import combinations
from pathlib import Path
import traceback

import pandas as pd

from part_3_metrics_collection.transaction_costs import compute_net_sharpe

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

ALL_KERNELS = ['gaussian',  'uniform'] #'exponential', 'gaussian-tms',


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_kernel(kernel_name: str) -> pd.DataFrame:
    print(f"\n{'═'*60}")
    print(f"  Kernel: {kernel_name}")
    print(f"{'═'*60}")

    rows = []

    for feat1, feat2 in PAIRS:
        cs          = f'LME_{feat1}_{feat2}'
        detail_path = (
            GRID_SEARCH_PATH / kernel_name / cs / 'full_fit'
            / f'full_fit_detail_k{PORT_N}.csv'
        )

        if not detail_path.exists():
            print(f"  [{cs}] SKIP — detail file not found")
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
                'feat1':         feat1,
                'feat2':         feat2,
                'cross_section': cs,
                'gross_SR':      result['gross_SR'],
                'net_SR':        result['net_SR'],
                'SR_loss':       result['gross_SR'] - result['net_SR'],
                'mean_TC':       float(result['tc_series'].mean()),
                'status':        'done',
            })
        except ValueError as e:
            if 'months were skipped' in str(e):
                print(f"  [{cs}] SKIP — incomplete detail file")
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

    summary  = pd.DataFrame(rows)
    out_path = GRID_SEARCH_PATH / kernel_name / f'tc_summary_all_k{PORT_N}.csv'
    summary.to_csv(out_path, index=False)
    print(f"\n  Summary saved → {out_path}")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_summaries = {}

    for kernel in ALL_KERNELS:
        all_summaries[kernel] = run_kernel(kernel)

    print(f"\n{'═'*60}")
    print("  DONE — median net SR per kernel:")
    for kernel, df in all_summaries.items():
        done = df[df['status'] == 'done']
        if len(done) > 0:
            print(f"    {kernel:15s}  median net SR = {done['net_SR'].median():.4f}"
                  f"  ({len(done)}/36 cross-sections)")