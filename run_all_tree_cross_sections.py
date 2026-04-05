#!/usr/bin/env python3
"""
Build AP-tree portfolios and run Part~2 LASSO grids for **every** characteristic triplet
``(LME, feat_i, feat_j)`` with ``1 <= i < j`` in ``FEATS_LIST`` (36 triplets: nine non-LME
characteristics including ``IdioVol`` from ``svar``).

From repository root::

    python run_all_tree_cross_sections.py --help
    python run_all_tree_cross_sections.py --part1-only
    python run_all_tree_cross_sections.py --part2-only --skip-existing-part2
    python run_all_tree_cross_sections.py --opt-quantile-trees --part1-only
    python run_all_tree_cross_sections.py --opt-quantile-trees --part2-only

``--opt-quantile-trees`` builds **extension-1** trees (causal optimal quantile splits) under
``data/results/tree_portfolios_optquantile/`` and Part~2 under ``ap_pruning_optquantile/``.
Baseline AP trees stay in ``tree_portfolios/`` and ``ap_pruning/``.

Part~1 (per triplet): ``create_tree_portfolio`` → ``combine_trees`` → ``filter_tree_ports``.

Part~2 (per triplet): same λ grid as ``part_2_AP_pruning/run_part2.py``; Ward clusters are
run **once** on the first triplet unless ``--no-clusters``.

After completion, ``pick_best_lambda.run_default_picks()`` discovers all ``LME_*`` folders
under ``data/results/ap_pruning/``, and ``generate_bpz_style_figures`` can plot the full
cross-section.

Expect **days** of CPU time for Part~2 × 36 if you use a dense λ grid (``AP_PRUNE_LAMBDA_GRID=paper``).
Use ``AP_PRUNE_LAMBDA_GRID=fast`` for dry runs.
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
        description="Run Part 1 (+ optional Part 2) for all AP-tree cross-sections."
    )
    parser.add_argument(
        "--part1-only",
        action="store_true",
        help="Only build tree CSVs, combine, and filter (no LASSO).",
    )
    parser.add_argument(
        "--part2-only",
        action="store_true",
        help="Only run Part 2 (expects Part 1 outputs already present).",
    )
    parser.add_argument(
        "--skip-existing-part1",
        action="store_true",
        help="Skip a triplet if level_all_excess_combined_filtered.csv already exists.",
    )
    parser.add_argument(
        "--skip-existing-part2",
        action="store_true",
        help="Skip Part 2 if ap_pruning/<triplet>/lambda_grid_meta.json exists.",
    )
    parser.add_argument(
        "--no-clusters",
        action="store_true",
        help="Do not run Ward cluster Part 2 even on the first triplet.",
    )
    parser.add_argument(
        "--pick-best",
        action="store_true",
        help="After Part 2, run pick_best_lambda for all discovered LME_* models (+ Ward).",
    )
    parser.add_argument(
        "--opt-quantile-trees",
        action="store_true",
        help="Extension 1: optimal-quantile splits → tree_portfolios_optquantile + ap_pruning_optquantile.",
    )
    parser.add_argument(
        "--part2-parallel",
        action="store_true",
        help="Enable parallel LASSO/CV inside each triplet's Part 2 (large speedup; needs RAM).",
    )
    parser.add_argument(
        "--part2-parallel-n",
        type=int,
        default=0,
        metavar="N",
        help="With --part2-parallel: worker count (default: min(16, cpu_count-1) if 0).",
    )
    args = parser.parse_args()
    if args.part1_only and args.part2_only:
        parser.error("Choose at most one of --part1-only / --part2-only")

    if args.opt_quantile_trees:
        tree_out = REPO / "data" / "results" / "tree_portfolios_optquantile"
        ap_out = REPO / "data" / "results" / "ap_pruning_optquantile"
    else:
        tree_out = REPO / "data" / "results" / "tree_portfolios"
        ap_out = REPO / "data" / "results" / "ap_pruning"

    from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
        all_triplet_pairs,
        triplet_subdir_name,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import (
        create_tree_portfolio,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import (
        combine_trees,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import (
        filter_tree_ports,
    )

    pairs = all_triplet_pairs()

    print(f"Cross-sections to process: {len(pairs)} triplets")
    print(f"  tree portfolios root: {tree_out}")
    print(f"  ap_pruning root: {ap_out}")
    print(f"  (see part_1_portfolio_creation/tree_portfolio_creation/cross_section_triplets.py)")

    if not args.part2_only:
        n_pairs = len(pairs)
        for i_triplet, (feat1, feat2) in enumerate(pairs, start=1):
            sub = triplet_subdir_name(feat1, feat2)
            pct_done = 100.0 * (i_triplet - 1) / n_pairs if n_pairs else 100.0
            print(
                f"\n[Part 1] Triplet {i_triplet}/{n_pairs} "
                f"({pct_done:.1f}% of list before this one) — {sub}"
            )
            filtered = tree_out / sub / "level_all_excess_combined_filtered.csv"
            if args.skip_existing_part1 and filtered.is_file():
                print(f"  [skip part1] found {filtered.name}")
                pct_row = 100.0 * i_triplet / n_pairs if n_pairs else 100.0
                print(
                    f"  [Part 1] Triplet {i_triplet}/{n_pairs} done — "
                    f"{pct_row:.1f}% through triplet list"
                )
                continue
            print(f"=== Part 1 build: {sub} ===")
            create_tree_portfolio(
                feat1=feat1,
                feat2=feat2,
                output_path=tree_out,
                split_mode="opt_quantile" if args.opt_quantile_trees else "ntile",
            )
            combine_trees(feat1=feat1, feat2=feat2, tree_out=tree_out)
            filter_tree_ports(feat1=feat1, feat2=feat2, tree_out=tree_out)
            pct_after = 100.0 * i_triplet / n_pairs if n_pairs else 100.0
            print(
                f"  [Part 1] Triplet {i_triplet}/{n_pairs} done — "
                f"{pct_after:.1f}% through triplet list"
            )

    if not args.part1_only:
        import importlib.util

        p2 = REPO / "part_2_AP_pruning" / "run_part2.py"
        spec = importlib.util.spec_from_file_location("seminar_run_part2_all", p2)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        first = True
        def _part2_outputs_complete(sub_path: Path) -> bool:
            """Meta alone is not enough (interrupted runs can lack results_full_*)."""
            if not (sub_path / "lambda_grid_meta.json").is_file():
                return False
            return any(sub_path.glob("results_full_l0_*_l2_*.csv"))

        n_pairs_p2 = len(pairs)
        for i_triplet, (feat1, feat2) in enumerate(pairs, start=1):
            sub = triplet_subdir_name(feat1, feat2)
            pct_done = 100.0 * (i_triplet - 1) / n_pairs_p2 if n_pairs_p2 else 100.0
            print(
                f"\n[Part 2] Triplet {i_triplet}/{n_pairs_p2} "
                f"({pct_done:.1f}% of list before this one) — {sub}"
            )
            sub_path = ap_out / sub
            if args.skip_existing_part2 and _part2_outputs_complete(sub_path):
                print(f"  [skip part2] complete grid outputs already present")
                first = False
                pct_row = 100.0 * i_triplet / n_pairs_p2 if n_pairs_p2 else 100.0
                print(
                    f"  [Part 2] Triplet {i_triplet}/{n_pairs_p2} done — "
                    f"{pct_row:.1f}% through triplet list"
                )
                continue
            tree_csv = tree_out / sub / mod.TREE_PORT_FILE
            if not tree_csv.is_file():
                print(f"  [skip part2] missing {tree_csv}")
                first = False
                pct_row = 100.0 * i_triplet / n_pairs_p2 if n_pairs_p2 else 100.0
                print(
                    f"  [Part 2] Triplet {i_triplet}/{n_pairs_p2} done — "
                    f"{pct_row:.1f}% through triplet list"
                )
                continue
            print(f"=== Part 2 run: {sub} ===")
            p2_n = args.part2_parallel_n
            if args.part2_parallel and p2_n <= 0:
                p2_n = min(16, max(2, (os.cpu_count() or 4) - 1))
            elif not args.part2_parallel:
                p2_n = 10
            mod.run_part2(
                run_trees=True,
                run_clusters=(first and not args.no_clusters and not args.opt_quantile_trees),
                tree_feat1=feat1,
                tree_feat2=feat2,
                run_pick_best=False,
                tree_input_root=str(tree_out),
                ap_output_root=str(ap_out),
                run_parallel=args.part2_parallel,
                parallel_n=p2_n,
            )
            first = False
            pct_after = 100.0 * i_triplet / n_pairs_p2 if n_pairs_p2 else 100.0
            print(
                f"  [Part 2] Triplet {i_triplet}/{n_pairs_p2} done — "
                f"{pct_after:.1f}% through triplet list"
            )

    if args.pick_best and not args.part1_only:
        from part_3_metrics_collection.pick_best_lambda import (
            print_ap_comparison,
            run_default_picks,
        )

        print("\n--- pick_best_lambda (all discovered LME_* + Ward) ---")
        picked = run_default_picks(
            port_n=10,
            ap_root=ap_out,
            tree_port_root=tree_out,
        )
        for r in picked:
            print(r)
        print_ap_comparison(picked)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
