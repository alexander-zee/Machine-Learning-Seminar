#!/usr/bin/env python3
"""
Random-projection (RP) trees — Part 1 + Part 2 for every cross-section on ``main``.

Uses the same layout as ``main.py`` (commented RP steps):

- Part 1: ``create_rp_tree_portfolio`` + ``combine_rp_trees`` →
  ``data/results/rp_tree_portfolios/LME_*/*``
- Part 2: ``RP_Pruning`` → ``data/results/grid_search/rp_tree/LME_*``
  (same subdir names as AP trees, but under the **rp_tree** grid root).

From repo root::

    python run_all_rp_cross_sections.py --help
    python run_all_rp_cross_sections.py --part2-only --skip-existing-part2 --part2-parallel

By default, **tqdm** bars show triplet-level progress (Part 1, Part 2, and ``--pick-best``).
Pass ``--no-progress`` for plain logging only.

Draft speed: fewer trees per triplet::

    set RP_N_TREES=27
    python run_all_rp_cross_sections.py --skip-existing-part1

Paper-style default is **81** trees per triplet. If you change ``n_trees``, delete that
triplet folder under ``rp_tree_portfolios`` before rebuilding.

Use ``--triplet-set no-idiovol`` to process only the **28** triplets that exclude IdioVol
(same set as ``export_table51_rp_uniform_vs_gaussian.py --rows no-idiovol``). Default is
**all** 36 triplets.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parent


def _chdir_repo() -> None:
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def main() -> None:
    _chdir_repo()

    parser = argparse.ArgumentParser(description="RP-tree pipeline for all cross-sections (main layout).")
    parser.add_argument("--part1-only", action="store_true")
    parser.add_argument("--part2-only", action="store_true")
    parser.add_argument("--skip-existing-part1", action="store_true")
    parser.add_argument("--skip-existing-part2", action="store_true")
    parser.add_argument(
        "--pick-best",
        action="store_true",
        help="After Part 2, run pick_best_lambda for every triplet with grid outputs.",
    )
    parser.add_argument(
        "--part2-parallel",
        action="store_true",
        help="Parallel LASSO/CV inside each triplet (joblib over lambda0 blocks; needs RAM).",
    )
    parser.add_argument(
        "--part2-parallel-n",
        type=int,
        default=0,
        metavar="N",
        help="With --part2-parallel: worker count (default: min(16, cpu_count-1) if 0).",
    )
    parser.add_argument(
        "--rp-n-trees",
        type=int,
        default=None,
        metavar="N",
        help="RP trees per triplet (default: env RP_N_TREES or 81).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars (Part 1/2 triplet loops and pick-best).",
    )
    parser.add_argument(
        "--triplet-set",
        choices=("all", "no-idiovol"),
        default="all",
        help="all=36 triplets; no-idiovol=28 (exclude any pair using IdioVol).",
    )
    parser.add_argument(
        "--feat1",
        type=str,
        default=None,
        help="Run only one triplet: first secondary characteristic (e.g. BEME).",
    )
    parser.add_argument(
        "--feat2",
        type=str,
        default=None,
        help="Run only one triplet: second secondary characteristic (e.g. OP).",
    )
    args = parser.parse_args()
    if args.part1_only and args.part2_only:
        parser.error("Choose at most one of --part1-only / --part2-only")

    from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
        all_triplet_pairs,
        all_triplet_pairs_excluding_secondary,
        canonical_feat_pair,
        triplet_subdir_name,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step2_RP_tree_portfolios import (
        create_rp_tree_portfolio,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_RP_trees import (
        combine_rp_trees,
    )
    from part_2_AP_pruning.RP_Pruning import RP_Pruning

    if args.triplet_set == "no-idiovol":
        pairs = all_triplet_pairs_excluding_secondary("IdioVol")
    else:
        pairs = all_triplet_pairs()
    if (args.feat1 is None) ^ (args.feat2 is None):
        parser.error("Use --feat1 and --feat2 together (or neither).")
    if args.feat1 is not None and args.feat2 is not None:
        f1, f2 = canonical_feat_pair(args.feat1, args.feat2)
        pairs = [(f1, f2)]

    rp_out = Path("data/results/rp_tree_portfolios")
    grid_rp = Path("data/results/grid_search/rp_tree")
    grid_rp.mkdir(parents=True, exist_ok=True)

    n_trees = args.rp_n_trees if args.rp_n_trees is not None else int(os.environ.get("RP_N_TREES", "81"))
    if n_trees < 1:
        parser.error("--rp-n-trees / RP_N_TREES must be >= 1")
    if n_trees != 81:
        print(
            f"NOTE: building {n_trees} RP trees per triplet (paper-style default is 81). "
            "Use 81 for final results; delete rp_tree_portfolios/<triplet> if you increase N later."
        )

    print(
        f"RP cross-sections: {len(pairs)} triplets "
        f"(triplet-set={args.triplet_set}, n_trees={n_trees})"
    )

    if not args.part2_only:
        n_pairs = len(pairs)
        pbar = tqdm(
            pairs,
            desc="RP Part 1 (portfolios)",
            unit="triplet",
            file=sys.stderr,
            disable=args.no_progress,
        )
        for i_triplet, (feat1, feat2) in enumerate(pbar, start=1):
            sub = triplet_subdir_name(feat1, feat2)
            pbar.set_postfix_str(sub.replace("LME_", "")[:40], refresh=False)
            filtered = rp_out / sub / "level_all_excess_combined.csv"
            if args.skip_existing_part1 and filtered.is_file():
                if args.no_progress:
                    print(f"[RP Part 1] {i_triplet}/{n_pairs} skip {sub} (exists)")
                continue
            if args.no_progress:
                print(f"\n[RP Part 1] Triplet {i_triplet}/{n_pairs} — {sub}")
            print(f"=== RP Part 1 build: {sub} ===")
            create_rp_tree_portfolio(
                feat1=feat1,
                feat2=feat2,
                output_path=rp_out,
                n_trees=n_trees,
            )
            combine_rp_trees(feat1=feat1, feat2=feat2, tree_out=rp_out, n_trees=n_trees)

    if not args.part1_only:

        def _part2_outputs_complete(sub_path: Path) -> bool:
            return any(sub_path.glob("results_full_l0_*_l2_*.csv"))

        p2_n = args.part2_parallel_n
        if args.part2_parallel and p2_n <= 0:
            p2_n = min(16, max(2, (os.cpu_count() or 4) - 1))
        elif not args.part2_parallel:
            p2_n = 10

        n_pairs_p2 = len(pairs)
        pbar2 = tqdm(
            pairs,
            desc="RP Part 2 (LASSO grid)",
            unit="triplet",
            file=sys.stderr,
            disable=args.no_progress,
        )
        for i_triplet, (feat1, feat2) in enumerate(pbar2, start=1):
            sub = triplet_subdir_name(feat1, feat2)
            out_sub = grid_rp / sub
            pbar2.set_postfix_str(sub.replace("LME_", "")[:40], refresh=False)
            if args.skip_existing_part2 and _part2_outputs_complete(out_sub):
                if args.no_progress:
                    print(f"[RP Part 2] {i_triplet}/{n_pairs_p2} skip {sub} (grid complete)")
                continue
            tree_csv = rp_out / sub / "level_all_excess_combined.csv"
            if not tree_csv.is_file():
                if args.no_progress:
                    print(f"  [skip part2 RP] missing {tree_csv}")
                continue
            if args.no_progress:
                print(f"\n[RP Part 2] Triplet {i_triplet}/{n_pairs_p2} — {sub}")
            print(f"=== RP Part 2 run: {sub} ===")
            RP_Pruning(
                feat1=feat1,
                feat2=feat2,
                input_path=rp_out,
                input_file_name="level_all_excess_combined.csv",
                output_path=grid_rp,
                n_train_valid=360,
                cvN=3,
                runFullCV=False,
                kmin=5,
                kmax=50,
                RunParallel=args.part2_parallel,
                ParallelN=p2_n,
                IsTree=True,
            )

    if args.pick_best and not args.part1_only:
        from part_3_metrics_collection.pick_best_lambdas import run_rp_picks_all

        print("\n--- pick_best_lambda (all RP triplets under grid_search/rp_tree) ---")
        picked = run_rp_picks_all(
            port_n=10,
            show_progress=not args.no_progress,
            pairs=pairs,
        )
        for r in picked:
            print(r)

    print("\nDone (RP pipeline).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
