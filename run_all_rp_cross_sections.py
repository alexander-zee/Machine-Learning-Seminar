#!/usr/bin/env python3
"""
Random-projection (RP) trees — Part 1 + Part 2 for all 36 cross-sections.

Mirrors ``run_all_tree_cross_sections.py`` but uses
``step2_RP_tree_portfolios.create_rp_tree_portfolio`` and ``step3_combine_RP_trees``,
writing to ``data/results/rp_tree_portfolios/``. Part 2 calls ``RP_Pruning`` → outputs
under ``data/results/ap_pruning/RP_LME_*_*`` (parallel to ``LME_*_*`` for standard trees).

From repo root::

    python run_all_rp_cross_sections.py --help
    $env:AP_PRUNE_LAMBDA_GRID='fast'
    python run_all_rp_cross_sections.py --part2-only --skip-existing-part2

**Speed vs. paper default:** RP Part~1 is the same *order* of work as AP Part~1 (81 trees × 36
triplets). There is no shortcut that reuses AP tree files. For **draft** runs, fewer trees::

    $env:RP_N_TREES='27'
    python run_all_rp_cross_sections.py --skip-existing-part1

or ``--rp-n-trees 9``. Final numbers should use the default **81**. If you change ``n_trees``,
delete that triplet folder under ``rp_tree_portfolios`` before rebuilding.
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

    parser = argparse.ArgumentParser(description="RP-tree pipeline for all cross-sections.")
    parser.add_argument("--part1-only", action="store_true")
    parser.add_argument("--part2-only", action="store_true")
    parser.add_argument("--skip-existing-part1", action="store_true")
    parser.add_argument("--skip-existing-part2", action="store_true")
    parser.add_argument("--no-clusters", action="store_true")
    parser.add_argument("--pick-best", action="store_true")
    parser.add_argument(
        "--rp-n-trees",
        type=int,
        default=None,
        metavar="N",
        help="Number of RP trees per triplet (default: env RP_N_TREES or 81). Lower = faster draft.",
    )
    args = parser.parse_args()
    if args.part1_only and args.part2_only:
        parser.error("Choose at most one of --part1-only / --part2-only")

    from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
        all_triplet_pairs,
        triplet_subdir_name,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step2_RP_tree_portfolios import (
        create_rp_tree_portfolio,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_RP_trees import (
        combine_rp_trees,
    )

    pairs = all_triplet_pairs()
    rp_out = Path("data/results/rp_tree_portfolios")
    ap_out = Path("data/results/ap_pruning")

    n_trees = args.rp_n_trees if args.rp_n_trees is not None else int(os.environ.get("RP_N_TREES", "81"))
    if n_trees < 1:
        parser.error("--rp-n-trees / RP_N_TREES must be >= 1")
    if n_trees != 81:
        print(
            f"NOTE: building {n_trees} RP trees per triplet (paper-style default is 81). "
            "Use 81 for final results; delete rp_tree_portfolios/<triplet> if you increase N later."
        )

    print(f"RP cross-sections: {len(pairs)} triplets (n_trees={n_trees})")

    if not args.part2_only:
        n_pairs = len(pairs)
        for i_triplet, (feat1, feat2) in enumerate(pairs, start=1):
            sub = triplet_subdir_name(feat1, feat2)
            pct_before = 100.0 * (i_triplet - 1) / n_pairs if n_pairs else 100.0
            print(
                f"\n[RP Part 1] Triplet {i_triplet}/{n_pairs} "
                f"({pct_before:.1f}% of list before this one) — {sub}"
            )
            filtered = rp_out / sub / "level_all_excess_combined.csv"
            if args.skip_existing_part1 and filtered.is_file():
                print(f"  [skip part1 RP] found {filtered.name}")
                pct_row = 100.0 * i_triplet / n_pairs if n_pairs else 100.0
                print(
                    f"  [RP Part 1] Triplet {i_triplet}/{n_pairs} done — "
                    f"{pct_row:.1f}% through triplet list"
                )
                continue
            print(f"=== RP Part 1 build: {sub} ===")
            create_rp_tree_portfolio(
                feat1=feat1,
                feat2=feat2,
                output_path=rp_out,
                n_trees=n_trees,
            )
            combine_rp_trees(feat1=feat1, feat2=feat2, tree_out=rp_out, n_trees=n_trees)
            pct_after = 100.0 * i_triplet / n_pairs if n_pairs else 100.0
            print(
                f"  [RP Part 1] Triplet {i_triplet}/{n_pairs} done — "
                f"{pct_after:.1f}% through triplet list"
            )

    if not args.part1_only:
        import importlib.util

        p2 = REPO / "part_2_AP_pruning" / "run_part2.py"
        spec = importlib.util.spec_from_file_location("seminar_run_part2_rp", p2)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def _part2_outputs_complete(sub_path: Path) -> bool:
            if not (sub_path / "lambda_grid_meta.json").is_file():
                return False
            return any(sub_path.glob("results_full_l0_*_l2_*.csv"))

        first = True
        for feat1, feat2 in pairs:
            sub = triplet_subdir_name(feat1, feat2)
            rp_sub = "RP_" + sub
            rp_sub_path = ap_out / rp_sub
            if args.skip_existing_part2 and _part2_outputs_complete(rp_sub_path):
                print(f"[skip part2 RP] {rp_sub} (complete grid outputs)")
                first = False
                continue
            tree_csv = rp_out / sub / mod.RP_TREE_PORT_FILE
            if not tree_csv.is_file():
                print(f"[skip part2 RP] {sub}: missing {tree_csv}")
                first = False
                continue
            print(f"\n=== RP Part 2: {rp_sub} ===")
            mod.run_part2(
                run_trees=False,
                run_clusters=(first and not args.no_clusters),
                run_rp_trees=True,
                tree_feat1=feat1,
                tree_feat2=feat2,
                run_pick_best=False,
            )
            first = False

    if args.pick_best and not args.part1_only:
        from part_3_metrics_collection.pick_best_lambda import (
            print_ap_comparison,
            run_rp_picks_all,
        )

        print("\n--- pick_best_lambda (all RP_LME_* ) ---")
        picked = run_rp_picks_all(port_n=10)
        for r in picked:
            print(r)
        print_ap_comparison(picked)

    print("\nDone (RP pipeline).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
