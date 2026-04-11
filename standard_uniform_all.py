"""
standard_uniform_all.py — Uniform (static) AP-Pruning for all triplets.

Mirrors the structure of standard_gaussian_all.py but uses UniformKernel,
which runs the full fit inline (no separate kernel_full_fit step needed).

Pipeline per cross-section
--------------------------
1. AP_Pruning  — grid search over (lambda0, lambda2), writes:
       uniform/LME_f1_f2/results_cv_3_l0_{i}_l2_{j}_h_1.csv  (validation SR)
       uniform/LME_f1_f2/results_full_l0_{i}_l2_{j}_h_1.csv  (train + test SR + betas)

2. pick_best_lambda — reads those CSVs, selects best (l0, l2) at k=PORT_N,
   writes:
       uniform/LME_f1_f2/Selected_Ports_{k}.csv
       uniform/LME_f1_f2/Selected_Ports_Weights_{k}.csv
       uniform/LME_f1_f2/train_SR_{k}.csv  /  valid_SR_{k}.csv  /  test_SR_{k}.csv

3. uniform_full_fit — reconstructs the excess return time series for the
   winning portfolio at k=PORT_N and saves:
       uniform/LME_f1_f2/full_fit/full_fit_summary_k{k}.csv
       uniform/LME_f1_f2/full_fit/full_fit_detail_k{k}.csv
   These match exactly the format of the Gaussian full_fit output so that
   ff5_batch_regression.py and ledoit_wolf_sr_test.py work unchanged.

Usage
-----
    python standard_uniform_all.py          # from project root
"""

from __future__ import annotations

import traceback
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from part_2_AP_pruning.AP_Pruning import AP_Pruning
from part_2_AP_pruning.kernels.uniform import UniformKernel
from part_3_metrics_collection.pick_best_lambdas import pick_best_lambda
from part_3_metrics_collection.uniform_full_fit import uniform_full_fit

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CHARACTERISTICS = [
    'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover',
    'IdioVol',
]

LAMBDA0    = [0.5, 0.55, 0.6]
LAMBDA2    = [10**-7, 10**-7.25, 10**-7.5]
K_MIN      = 5
K_MAX      = 50
PORT_N     = 10
N_WORKERS  = 4

TREE_PORT_PATH   = Path('data/results/tree_portfolios')
GRID_SEARCH_PATH = Path('data/results/grid_search/tree')
PORT_FILE_NAME   = 'level_all_excess_combined_filtered.csv'
N_TRAIN_VALID    = 360

PAIRS = list(combinations(CHARACTERISTICS, 2))

PROGRESS_PATH = GRID_SEARCH_PATH / 'uniform' / 'progress_standard_uniform.csv'
SUMMARY_PATH  = GRID_SEARCH_PATH / 'uniform' / 'all_cross_sections_summary_standard_uniform.csv'


# ─────────────────────────────────────────────────────────────────────────────
# Progress helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_progress() -> pd.DataFrame:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PROGRESS_PATH.exists():
        df = pd.read_csv(PROGRESS_PATH)
        df['error'] = df['error'].astype(object)
        return df
    df = pd.DataFrame([
        {'feat1': f1, 'feat2': f2,
         'cross_section': f'LME_{f1}_{f2}',
         'status': 'pending',
         'train_SR': None, 'valid_SR': None, 'test_SR': None,
         'lambda0': None, 'lambda2': None,
         'error': None}
        for f1, f2 in PAIRS
    ])
    df['error'] = df['error'].astype(object)
    df.to_csv(PROGRESS_PATH, index=False)
    print(f"Progress file created: {PROGRESS_PATH}")
    return df


def save_progress(df: pd.DataFrame):
    df.to_csv(PROGRESS_PATH, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Per-combination worker
# ─────────────────────────────────────────────────────────────────────────────

def run_one(args: tuple) -> dict:
    feat1, feat2 = args
    subdir = f'LME_{feat1}_{feat2}'

    try:
        print(f"  [{subdir}] Starting...", flush=True)

        # Step 1: Grid search (uniform — full fit runs inline)
        AP_Pruning(
            feat1=feat1, feat2=feat2,
            input_path=TREE_PORT_PATH,
            input_file_name=PORT_FILE_NAME,
            output_path=GRID_SEARCH_PATH,
            n_train_valid=N_TRAIN_VALID, cvN=3, runFullCV=False,
            kmin=K_MIN, kmax=K_MAX,
            RunParallel=False, ParallelN=10, IsTree=True,
            lambda0=LAMBDA0, lambda2=LAMBDA2,
            kernel_cls=UniformKernel,
            state=None,
        )

        # Step 2: Pick best hyperparameters and write Selected_Ports files
        sr = pick_best_lambda(
            feat1=feat1, feat2=feat2,
            ap_prune_result_path=GRID_SEARCH_PATH,
            port_n=PORT_N,
            lambda0=LAMBDA0, lambda2=LAMBDA2,
            portfolio_path=TREE_PORT_PATH,
            port_name=PORT_FILE_NAME,
            full_cv=False, write_table=True,
        )
        train_SR, valid_SR, test_SR = float(sr[0]), float(sr[1]), float(sr[2])

        # Step 3: Reconstruct excess return time series for k=PORT_N
        uniform_full_fit(feat1, feat2, k=PORT_N)

        print(f"  [{subdir}] Done — test_SR={test_SR:.4f}", flush=True)

        return {
            'feat1': feat1, 'feat2': feat2,
            'cross_section': subdir,
            'status': 'done',
            'train_SR': train_SR,
            'valid_SR': valid_SR,
            'test_SR':  test_SR,
            'lambda0':  LAMBDA0[np.argmax([sr[1]])],
            'lambda2':  None,   # written per-file; summary has the winning values
            'error': None,
        }

    except Exception:
        tb = traceback.format_exc()
        print(f"  [{subdir}] FAILED:\n{tb}", flush=True)
        return {
            'feat1': feat1, 'feat2': feat2,
            'cross_section': subdir,
            'status': 'failed',
            'train_SR': None, 'valid_SR': None, 'test_SR': None,
            'lambda0': None, 'lambda2': None,
            'error': 'error',
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    progress = load_progress()
    pending  = progress[progress['status'] != 'done']
    n_done   = len(progress) - len(pending)
    print(f"{n_done}/{len(PAIRS)} already done, {len(pending)} remaining")
    print(f"Running with {N_WORKERS} parallel workers\n")

    if len(pending) == 0:
        print("All combinations already done.")
    else:
        args_list = [(row['feat1'], row['feat2']) for _, row in pending.iterrows()]

        with Pool(processes=N_WORKERS) as pool:
            for result in pool.imap_unordered(run_one, args_list):
                cs   = result['cross_section']
                mask = progress['cross_section'] == cs
                if result['error'] is not None:
                    result['error'] = str(result['error'])[:500]
                for col in ['status', 'train_SR', 'valid_SR', 'test_SR',
                            'lambda0', 'lambda2', 'error']:
                    progress.loc[mask, col] = result[col]
                save_progress(progress)
                print(f"  Saved progress for {cs} — status: {result['status']}",
                      flush=True)

    # Final summary
    done = progress[progress['status'] == 'done']
    done.to_csv(SUMMARY_PATH, index=False)
    print(f"\nAll done. {len(done)}/{len(PAIRS)} completed.")
    print(f"Summary: {SUMMARY_PATH}")
    if len(done) > 0:
        print(done[['cross_section', 'valid_SR', 'test_SR']].to_string(index=False))

    failed = progress[progress['status'] == 'failed']
    if len(failed) > 0:
        print(f"\n{len(failed)} failed:")
        print(failed[['cross_section', 'error']].to_string(index=False))