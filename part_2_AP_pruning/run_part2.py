"""
Part 2: AP-style LASSO pruning (train/valid/test Sharpe grid).

Run from repository root:
    python part_2_AP_pruning/run_part2.py

Extension 1 (opt-quantile trees built under ``data/results/tree_portfolios_optquantile/``)::

    python part_2_AP_pruning/run_part2.py --opt-quantile

That reads filtered CSVs from ``tree_portfolios_optquantile`` and writes
``data/results/ap_pruning_optquantile/LME_*``. Baseline AP trees stay in
``tree_portfolios/`` + ``ap_pruning/`` — they are not overwritten.

Or import ``run_part2`` from main (see main.py); pass ``tree_input_root`` /
``ap_output_root`` to override paths programmatically.

Tree pruning on ~1500 portfolios can take a long time; cluster pruning (10 columns) is fast.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Local imports (lasso.py, lasso_valid_par_full.py)
_P2 = Path(__file__).resolve().parent
if str(_P2) not in sys.path:
    sys.path.insert(0, str(_P2))

from AP_Pruning import AP_Pruning, AP_Pruning_clusters, RP_Pruning, TV_Pruning  # noqa: E402
from lambda_grids import ap_lambda_grid_mode, get_lambda_grids  # noqa: E402

REPO_ROOT = _P2.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (  # noqa: E402
    FEATS_LIST,
)

TREE_INPUT_ROOT = REPO_ROOT / "data" / "results" / "tree_portfolios"
TREE_INPUT_OPT_QUANT = REPO_ROOT / "data" / "results" / "tree_portfolios_optquantile"
TREE_PORT_FILE = "level_all_excess_combined_filtered.csv"
RP_TREE_INPUT_ROOT = REPO_ROOT / "data" / "results" / "rp_tree_portfolios"
RP_TREE_PORT_FILE = "level_all_excess_combined.csv"
AP_OUT = REPO_ROOT / "data" / "results" / "ap_pruning"
AP_OUT_OPT_QUANT = REPO_ROOT / "data" / "results" / "ap_pruning_optquantile"

CLUSTER_RETURNS = REPO_ROOT / "data" / "portfolios" / "clusters" / "cluster_returns.csv"


def _run_part3_pick_best(port_n: int = 10, run_rp: bool = False, run_tv: bool = False) -> None:
    p = REPO_ROOT / "part_3_metrics_collection" / "pick_best_lambda.py"
    spec = importlib.util.spec_from_file_location("seminar_pick_best", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("--- PART 3: λ* from max validation Sharpe (Pick_Best_Lambda.R logic) ---")
    picked = list(mod.run_default_picks(port_n=port_n))
    if run_rp:
        picked += list(mod.run_rp_picks_all(port_n=port_n))
    if run_tv:
        picked += list(mod.run_tv_picks_all(port_n=port_n))
    for r in picked:
        print(r)
    mod.print_ap_comparison(picked)


def run_part2(
    run_trees: bool = True,
    run_clusters: bool = True,
    run_rp_trees: bool = False,
    run_tv_trees: bool = False,
    tree_feat1: str | int = "OP",
    tree_feat2: str | int = "Investment",
    n_train_valid: int = 360,
    cvN: int = 3,
    run_full_cv: bool = False,
    run_parallel: bool = False,
    parallel_n: int = 10,
    run_pick_best: bool = True,
    port_n: int = 10,
    tree_input_root: str | Path | None = None,
    ap_output_root: str | Path | None = None,
) -> None:
    tree_root = Path(tree_input_root) if tree_input_root is not None else TREE_INPUT_ROOT
    ap_root = Path(ap_output_root) if ap_output_root is not None else AP_OUT
    ap_root.mkdir(parents=True, exist_ok=True)
    out_str = str(ap_root)
    g0, g2 = get_lambda_grids()
    print(
        f"AP λ grid: mode={ap_lambda_grid_mode()} "
        f"(|λ0|={len(g0)}, |λ2|={len(g2)}; set AP_PRUNE_LAMBDA_GRID=fast|paper|paper_full)"
    )

    if run_clusters:
        if not CLUSTER_RETURNS.is_file():
            print(f"Skipping clusters: file not found: {CLUSTER_RETURNS}")
        else:
            print("--- PART 2: AP pruning (Ward clusters, 10 portfolios) ---")
            AP_Pruning_clusters(
                str(CLUSTER_RETURNS),
                out_str,
                sub_dir="Ward_clusters_10",
                n_train_valid=n_train_valid,
                cvN=cvN,
                runFullCV=run_full_cv,
                kmin=3,
                kmax=10,
                RunParallel=run_parallel,
                ParallelN=parallel_n,
            )
            print(f"Cluster grid search saved under: {AP_OUT / 'Ward_clusters_10'}")

    if run_trees:
        i1 = _resolve_i(tree_feat1)
        i2 = _resolve_i(tree_feat2)
        sub = "_".join(["LME", FEATS_LIST[i1], FEATS_LIST[i2]])
        tree_csv = tree_root / sub / TREE_PORT_FILE
        if not tree_csv.is_file():
            print(f"Skipping trees: missing {tree_csv}")
        else:
            print("--- PART 2: AP pruning (AP-trees, filtered excess returns) ---")
            print("This can take substantial time (many portfolios × λ grid).")
            AP_Pruning(
                FEATS_LIST,
                tree_feat1,
                tree_feat2,
                str(tree_root),
                TREE_PORT_FILE,
                out_str,
                n_train_valid=n_train_valid,
                cvN=cvN,
                runFullCV=run_full_cv,
                kmin=5,
                kmax=50,
                RunParallel=run_parallel,
                ParallelN=parallel_n,
                IsTree=True,
            )
            print(f"Tree grid search saved under: {AP_OUT / sub}")

    if run_rp_trees:
        i1 = _resolve_i(tree_feat1)
        i2 = _resolve_i(tree_feat2)
        sub = "_".join(["LME", FEATS_LIST[i1], FEATS_LIST[i2]])
        rp_csv = RP_TREE_INPUT_ROOT / sub / RP_TREE_PORT_FILE
        if not rp_csv.is_file():
            print(f"Skipping RP trees: missing {rp_csv}")
        else:
            print("--- PART 2: AP pruning (RP-trees, combined excess returns) ---")
            RP_Pruning(
                FEATS_LIST,
                tree_feat1,
                tree_feat2,
                str(RP_TREE_INPUT_ROOT),
                out_str,
                n_train_valid=n_train_valid,
                cvN=cvN,
                runFullCV=run_full_cv,
                kmin=5,
                kmax=50,
                RunParallel=run_parallel,
                ParallelN=parallel_n,
            )
            print(f"RP-tree grid search saved under: {AP_OUT / ('RP_' + sub)}")

    if run_tv_trees:
        i1 = _resolve_i(tree_feat1)
        i2 = _resolve_i(tree_feat2)
        sub = "_".join(["LME", FEATS_LIST[i1], FEATS_LIST[i2]])
        tv_csv = tree_root / sub / TREE_PORT_FILE
        if not tv_csv.is_file():
            print(f"Skipping TV trees: missing {tv_csv}")
        else:
            print("--- PART 2: AP pruning (TV / kernel-weighted moments on AP-tree returns) ---")
            TV_Pruning(
                FEATS_LIST,
                tree_feat1,
                tree_feat2,
                str(tree_root),
                TREE_PORT_FILE,
                out_str,
                n_train_valid=n_train_valid,
                cvN=cvN,
                runFullCV=run_full_cv,
                kmin=5,
                kmax=50,
                RunParallel=run_parallel,
                ParallelN=parallel_n,
            )
            print(f"TV-tree grid search saved under: {AP_OUT / ('TV_' + sub)}")

    if run_pick_best and (run_clusters or run_trees or run_rp_trees or run_tv_trees):
        try:
            _run_part3_pick_best(port_n=port_n, run_rp=run_rp_trees, run_tv=run_tv_trees)
        except Exception as e:
            print(f"Part 3 pick_best_lambda skipped or failed: {e}")


def _resolve_i(f: str | int) -> int:
    if isinstance(f, int):
        return f
    return FEATS_LIST.index(f)


if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="Part 2: AP-style LASSO pruning.")
    pa.add_argument(
        "--opt-quantile",
        action="store_true",
        help="Use tree_portfolios_optquantile and write ap_pruning_optquantile (extension-1 trees).",
    )
    pa.add_argument("--no-clusters", action="store_true", help="Skip Ward cluster grid.")
    pa.add_argument("--no-pick-best", action="store_true", help="Skip pick_best_lambda at end.")
    args = pa.parse_args()
    run_part2(
        run_trees=True,
        run_clusters=not args.no_clusters,
        run_pick_best=not args.no_pick_best,
        tree_input_root=str(TREE_INPUT_OPT_QUANT) if args.opt_quantile else None,
        ap_output_root=str(AP_OUT_OPT_QUANT) if args.opt_quantile else None,
    )
