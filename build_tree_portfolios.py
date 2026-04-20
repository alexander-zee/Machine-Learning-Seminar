"""
build_tree_portfolios.py
========================
Builds the AP-tree candidate portfolios for all 36 LME-anchored triplets.

This is a prerequisite for all kernel runs (standard_uniform_all.py,
standard_gaussian_all.py, standard_exponential_all.py, etc.), which assume
the filtered portfolio CSVs already exist under:

    data/results/tree_portfolios/LME_<feat1>_<feat2>/
        level_all_excess_combined.csv
        level_all_excess_combined_filtered.csv

Pipeline stages
---------------
  Step 1  — prepare_data()
              Reads raw CRSP/Compustat CSV, rank-transforms characteristics,
              writes data/prepared/panel.parquet.
              Run once; skipped automatically if the parquet already exists.

  Step 2  — create_tree_portfolio(feat1, feat2)
              Builds all 81 (3^4) trees for the triplet (LME, feat1, feat2)
              and saves per-tree return CSVs.

  Step 3  — combine_trees(feat1, feat2)
              Stacks all tree orderings, deduplicates portfolios with
              identical return histories, subtracts the risk-free rate.
              Outputs level_all_excess_combined.csv.

  Step 4  — filter_tree_ports(feat1, feat2)
              Removes single-sort portfolios that are collinear with
              standard decile sorts.
              Outputs level_all_excess_combined_filtered.csv.

Usage
-----
  # Full run (all 36 triplets)
  python build_tree_portfolios.py

  # Skip step 1 if panel.parquet already exists
  python build_tree_portfolios.py --skip-step1

  # Skip triplets whose filtered CSV already exists
  python build_tree_portfolios.py --skip-existing

  # Run only a single triplet (useful for debugging)
  python build_tree_portfolios.py --feat1 BEME --feat2 OP

  # Combine flags
  python build_tree_portfolios.py --skip-step1 --skip-existing
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PANEL_PATH       = Path('data/prepared/panel.parquet')
TREE_PORT_PATH   = Path('data/results/tree_portfolios')
FACTOR_PATH      = Path('data/raw')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filtered_csv(feat1: str, feat2: str) -> Path:
    return TREE_PORT_PATH / f'LME_{feat1}_{feat2}' / 'level_all_excess_combined_filtered.csv'


def _combined_csv(feat1: str, feat2: str) -> Path:
    return TREE_PORT_PATH / f'LME_{feat1}_{feat2}' / 'level_all_excess_combined.csv'


def run_step1() -> None:
    from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import prepare_data
    print("─" * 60)
    print("Step 1: Preparing panel data → data/prepared/panel.parquet")
    print("─" * 60)
    prepare_data()
    print("Step 1 complete.\n")


def run_triplet(feat1: str, feat2: str) -> bool:
    """
    Run steps 2, 3, 4 for one triplet. Returns True on success, False on error.
    """
    from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
    from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import combine_trees
    from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import filter_tree_ports

    label = f'LME_{feat1}_{feat2}'
    try:
        # Step 2 — build trees
        create_tree_portfolio(
            feat1=feat1,
            feat2=feat2,
            output_path=TREE_PORT_PATH,
        )

        # Step 3 — combine & subtract rf
        combine_trees(
            feat1=feat1,
            feat2=feat2,
            factor_path=FACTOR_PATH,
            tree_out=TREE_PORT_PATH,
        )

        # Step 4 — filter single-sorts
        filter_tree_ports(
            feat1=feat1,
            feat2=feat2,
            tree_out=TREE_PORT_PATH,
        )

        return True

    except Exception:
        print(f"\n[ERROR] Triplet {label} failed:", file=sys.stderr)
        traceback.print_exc()
        return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build AP-tree portfolios (steps 1–4) for all 36 triplets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--skip-step1', action='store_true',
        help='Skip data preparation (step 1). Use if panel.parquet already exists.',
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip triplets whose filtered portfolio CSV already exists.',
    )
    parser.add_argument(
        '--feat1', type=str, default=None,
        help='Run only one triplet: first secondary characteristic (e.g. BEME).',
    )
    parser.add_argument(
        '--feat2', type=str, default=None,
        help='Run only one triplet: second secondary characteristic (e.g. OP).',
    )
    args = parser.parse_args()

    if (args.feat1 is None) ^ (args.feat2 is None):
        parser.error('Use --feat1 and --feat2 together, or neither.')

    # ── Resolve triplet list ──────────────────────────────────────────────────
    from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
        all_triplet_pairs,
        canonical_feat_pair,
    )

    if args.feat1 is not None:
        pairs = [canonical_feat_pair(args.feat1, args.feat2)]
    else:
        pairs = all_triplet_pairs()

    print("=" * 60)
    print("  AP-TREE PORTFOLIO BUILDER")
    print(f"  Triplets : {len(pairs)}")
    print(f"  Output   : {TREE_PORT_PATH}")
    print("=" * 60)

    # ── Step 1 ────────────────────────────────────────────────────────────────
    if args.skip_step1:
        if not PANEL_PATH.exists():
            print(f"[WARNING] --skip-step1 was set but {PANEL_PATH} does not exist. "
                  "Running step 1 anyway.")
            run_step1()
        else:
            print(f"Step 1 skipped — {PANEL_PATH} already exists.\n")
    else:
        run_step1()

    # ── Steps 2–4 per triplet ─────────────────────────────────────────────────
    skipped  = []
    failed   = []
    success  = []

    pbar = tqdm(pairs, desc='Building triplets', unit='triplet', file=sys.stderr)
    for feat1, feat2 in pbar:
        label = f'LME_{feat1}_{feat2}'
        pbar.set_postfix_str(label.replace('LME_', ''), refresh=False)

        if args.skip_existing and _filtered_csv(feat1, feat2).exists():
            skipped.append(label)
            continue

        print(f"\n{'─' * 60}")
        print(f"  Triplet: {label}")
        print(f"{'─' * 60}")

        ok = run_triplet(feat1, feat2)
        (success if ok else failed).append(label)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(pairs)
    print(f"\n{'=' * 60}")
    print(f"  DONE — {len(success)}/{total} completed  "
          f"| {len(skipped)} skipped  | {len(failed)} failed")
    print(f"{'=' * 60}")

    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")

    if failed:
        print(f"\nFailed ({len(failed)}) — check stderr above for tracebacks:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == '__main__':
    main()