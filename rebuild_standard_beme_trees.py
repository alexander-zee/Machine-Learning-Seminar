"""
rebuild_beme_trees.py — Rebuild AP-tree portfolios for all BEME cross-sections.

Runs steps 2→3→4 (tree creation, combine, filter) for all 8 cross-sections
that include BEME, using 4 parallel workers.

Steps per cross-section
-----------------------
1. create_tree_portfolio  — builds all 81 trees (step2)
2. combine_trees          — dedup + subtract rf (step3)
3. filter_tree_ports      — remove single-sorted ports (step4)

panel.parquet is read-only and shared across workers — no race conditions.
Each cross-section writes to its own subfolder under data/results/tree_portfolios/
so parallelism is safe.

Usage
-----
    python rebuild_beme_trees.py          # from project root

Optionally re-run step1 (panel.parquet) first if needed — see bottom of file.
"""

from __future__ import annotations

import traceback
from multiprocessing import Pool
from pathlib import Path

from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import combine_trees
from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import filter_tree_ports

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

N_WORKERS = 4

# All 8 BEME cross-sections (BEME paired with each other characteristic)
BEME_PAIRS = [
    ('BEME', 'r12_2'),
    ('BEME', 'OP'),
    ('BEME', 'Investment'),
    ('BEME', 'ST_Rev'),
    ('BEME', 'LT_Rev'),
    ('BEME', 'AC'),
    ('BEME', 'LTurnover'),
    ('BEME', 'IdioVol'),
]

TREE_OUT    = Path('data/results/tree_portfolios')
FACTOR_PATH = Path('data/raw')


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def run_one(args: tuple) -> dict:
    feat1, feat2 = args
    subdir = f'LME_{feat1}_{feat2}'

    try:
        print(f"[{subdir}] Step 2: Building trees...", flush=True)
        create_tree_portfolio(feat1=feat1, feat2=feat2)

        print(f"[{subdir}] Step 3: Combining trees...", flush=True)
        combine_trees(feat1=feat1, feat2=feat2,
                      factor_path=FACTOR_PATH, tree_out=TREE_OUT)

        print(f"[{subdir}] Step 4: Filtering portfolios...", flush=True)
        filter_tree_ports(feat1=feat1, feat2=feat2, tree_out=TREE_OUT)

        print(f"[{subdir}] Done.", flush=True)
        return {'pair': subdir, 'status': 'done', 'error': None}

    except Exception:
        tb = traceback.format_exc()
        print(f"[{subdir}] FAILED:\n{tb}", flush=True)
        return {'pair': subdir, 'status': 'failed', 'error': tb}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Uncomment if you also need to regenerate panel.parquet from scratch:
    # from part_1_portfolio_creation.step1_prepare_data import prepare_data
    # print("Step 1: Rebuilding panel.parquet...")
    # prepare_data()
    # print("panel.parquet done.\n")

    print(f"Rebuilding AP-trees for {len(BEME_PAIRS)} BEME cross-sections "
          f"with {N_WORKERS} workers...\n")

    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(run_one, BEME_PAIRS)

    print("\n" + "="*60)
    print("Summary:")
    for r in results:
        status = "✓" if r['status'] == 'done' else "✗"
        print(f"  {status} {r['pair']}")

    failed = [r for r in results if r['status'] == 'failed']
    if failed:
        print(f"\n{len(failed)} cross-section(s) failed — check logs above.")
    else:
        print(f"\nAll {len(BEME_PAIRS)} cross-sections completed successfully.")
        print("\nNext step: run rerun_beme.py to rerun the AP-Pruning + kernel fitting.")