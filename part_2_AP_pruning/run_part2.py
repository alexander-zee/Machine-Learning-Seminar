"""
Part 2: AP-style LASSO pruning (train/valid/test Sharpe grid).

Run from repository root:
    python part_2_AP_pruning/run_part2.py

Or import `run_part2` from main (see main.py).

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

from AP_Pruning import AP_Pruning, AP_Pruning_clusters  # noqa: E402

REPO_ROOT = _P2.parent

FEATS_LIST = [
    "LME",
    "BEME",
    "r12_2",
    "OP",
    "Investment",
    "ST_Rev",
    "LT_Rev",
    "AC",
    "LTurnover",
]

TREE_INPUT_ROOT = REPO_ROOT / "data" / "results" / "tree_portfolios"
TREE_PORT_FILE = "level_all_excess_combined_filtered.csv"
AP_OUT = REPO_ROOT / "data" / "results" / "ap_pruning"

CLUSTER_RETURNS = REPO_ROOT / "data" / "portfolios" / "clusters" / "cluster_returns.csv"


def _run_part3_pick_best(port_n: int = 10) -> None:
    p = REPO_ROOT / "part_3_metrics_collection" / "pick_best_lambda.py"
    spec = importlib.util.spec_from_file_location("seminar_pick_best", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("--- PART 3: λ* from max validation Sharpe (Pick_Best_Lambda.R logic) ---")
    picked = list(mod.run_default_picks(port_n=port_n))
    for r in picked:
        print(r)
    mod.print_ap_comparison(picked)


def run_part2(
    run_trees: bool = True,
    run_clusters: bool = True,
    tree_feat1: str | int = "OP",
    tree_feat2: str | int = "Investment",
    n_train_valid: int = 360,
    cvN: int = 3,
    run_full_cv: bool = False,
    run_parallel: bool = False,
    parallel_n: int = 10,
    run_pick_best: bool = True,
    port_n: int = 10,
) -> None:
    AP_OUT.mkdir(parents=True, exist_ok=True)
    out_str = str(AP_OUT)

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
        tree_csv = TREE_INPUT_ROOT / sub / TREE_PORT_FILE
        if not tree_csv.is_file():
            print(f"Skipping trees: missing {tree_csv}")
        else:
            print("--- PART 2: AP pruning (AP-trees, filtered excess returns) ---")
            print("This can take substantial time (many portfolios × λ grid).")
            AP_Pruning(
                FEATS_LIST,
                tree_feat1,
                tree_feat2,
                str(TREE_INPUT_ROOT),
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

    if run_pick_best and (run_clusters or run_trees):
        try:
            _run_part3_pick_best(port_n=port_n)
        except Exception as e:
            print(f"Part 3 pick_best_lambda skipped or failed: {e}")


def _resolve_i(f: str | int) -> int:
    if isinstance(f, int):
        return f
    return FEATS_LIST.index(f)


if __name__ == "__main__":
    run_part2(run_trees=True, run_clusters=True)
