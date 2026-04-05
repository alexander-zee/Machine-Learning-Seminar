#!/usr/bin/env python3
"""
Time-varying (kernel-weighted) moments on **AP-tree** returns — Part 2 for all 36 cross-sections.

Uses the same Part~1 outputs as standard trees (``data/results/tree_portfolios/`` filtered CSV).
``TV_Pruning`` writes ``data/results/ap_pruning/TV_LME_*_*`` (recency kernel on training rows;
override halflife with env ``TV_KERNEL_HALFLIFE_MONTHS``).

From repo root::

    python run_all_tv_cross_sections.py --help
    $env:AP_PRUNE_LAMBDA_GRID='fast'
    python run_all_tv_cross_sections.py --part2-only --skip-existing-part2 --pick-best

Expect **similar wall time to re-running AP Part~2 × 36** for the same λ grid (no extra Part~1).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _chdir_repo() -> None:
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def main() -> None:
    _chdir_repo()

    parser = argparse.ArgumentParser(
        description="TV (kernel-weighted moments) AP-pruning for all 36 cross-sections."
    )
    parser.add_argument(
        "--part2-only",
        action="store_true",
        help="Only Part 2 (expects tree_portfolios Part 1 already built).",
    )
    parser.add_argument("--skip-existing-part2", action="store_true")
    parser.add_argument("--no-clusters", action="store_true")
    parser.add_argument("--pick-best", action="store_true")
    args = parser.parse_args()

    from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
        all_triplet_pairs,
        triplet_subdir_name,
    )

    pairs = all_triplet_pairs()
    tree_out = Path("data/results/tree_portfolios")
    ap_out = Path("data/results/ap_pruning")

    print(f"TV cross-sections: {len(pairs)} triplets (same inputs as AP-trees)")

    import importlib.util

    p2 = REPO / "part_2_AP_pruning" / "run_part2.py"
    spec = importlib.util.spec_from_file_location("seminar_run_part2_tv", p2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def _part2_outputs_complete(sub_path: Path) -> bool:
        if not (sub_path / "lambda_grid_meta.json").is_file():
            return False
        return any(sub_path.glob("results_full_l0_*_l2_*.csv"))

    first = True
    for feat1, feat2 in pairs:
        sub = triplet_subdir_name(feat1, feat2)
        tv_sub = "TV_" + sub
        tv_sub_path = ap_out / tv_sub
        if args.skip_existing_part2 and _part2_outputs_complete(tv_sub_path):
            print(f"[skip part2 TV] {tv_sub} (complete grid outputs)")
            first = False
            continue
        tree_csv = tree_out / sub / mod.TREE_PORT_FILE
        if not tree_csv.is_file():
            print(f"[skip part2 TV] {sub}: missing {tree_csv}")
            first = False
            continue
        print(f"\n=== TV Part 2: {tv_sub} ===")
        mod.run_part2(
            run_trees=False,
            run_clusters=(first and not args.no_clusters),
            run_rp_trees=False,
            run_tv_trees=True,
            tree_feat1=feat1,
            tree_feat2=feat2,
            run_pick_best=False,
        )
        first = False

    if args.pick_best:
        from part_3_metrics_collection.pick_best_lambda import (
            print_ap_comparison,
            run_tv_picks_all,
        )

        print("\n--- pick_best_lambda (all TV_LME_* ) ---")
        picked = run_tv_picks_all(port_n=10)
        for r in picked:
            print(r)
        print_ap_comparison(picked)

    print("\nDone (TV pipeline).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
