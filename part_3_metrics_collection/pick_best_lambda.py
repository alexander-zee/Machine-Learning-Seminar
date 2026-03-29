"""
Select (λ0, λ2) by maximum validation Sharpe; export SR grids and chosen portfolios.

Matches part_3_metrics_collection/Pick_Best_Lambda.R logic:
  - train/test Sharpe from results_full_*; validation from results_cv_* (fold 3, or mean of 1–3 if full_cv).
  - Chooses the grid point with highest valid_SR; ties → first occurrence (numpy argmax order).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
AP_PRUNE_DEFAULT = REPO_ROOT / "data" / "results" / "ap_pruning"
TREE_PORT_ROOT = REPO_ROOT / "data" / "results" / "tree_portfolios"
CLUSTER_RETURNS = REPO_ROOT / "data" / "portfolios" / "clusters" / "cluster_returns.csv"

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


def _triplet_subdir(feat1: str | int, feat2: str | int) -> str:
    if isinstance(feat1, str):
        i1 = FEATS_LIST.index(feat1)
    else:
        i1 = feat1
    if isinstance(feat2, str):
        i2 = FEATS_LIST.index(feat2)
    else:
        i2 = feat2
    return "_".join(["LME", FEATS_LIST[i1], FEATS_LIST[i2]])


def pick_best_lambda(
    ap_prune_root: str | Path,
    sub_dir: str,
    port_n: int,
    portfolio_csv: str | Path,
    *,
    n_lambda0: int = 3,
    n_lambda2: int = 3,
    full_cv: bool = False,
    returns_index_col: int | None = None,
    write_tables: bool = True,
) -> dict:
    """
    Parameters
    ----------
    ap_prune_root
        Directory containing sub_dir with results_full_* and results_cv_* CSVs.
    sub_dir
        e.g. ``LME_OP_Investment`` or ``Ward_clusters_10``.
    port_n
        Target number of selected portfolios (``portsN`` in LASSO output), e.g. 10.
    portfolio_csv
        Same return panel used in Part 2 (trees: wide CSV of excess returns; clusters: cluster_returns.csv).
    returns_index_col
        Pass ``0`` if the first CSV column is a date index (cluster returns); ``None`` for tree matrices.
    """
    root = Path(ap_prune_root)
    sub = root / sub_dir
    if not sub.is_dir():
        raise FileNotFoundError(f"Missing AP-pruning folder: {sub}")

    train_sr = np.zeros((n_lambda0, n_lambda2))
    valid_sr = np.zeros((n_lambda0, n_lambda2))
    test_sr = np.zeros((n_lambda0, n_lambda2))

    for i in range(1, n_lambda0 + 1):
        for j in range(1, n_lambda2 + 1):
            full_path = sub / f"results_full_l0_{i}_l2_{j}.csv"
            full_data = pd.read_csv(full_path)
            row_f = full_data.loc[full_data["portsN"] == port_n]
            if row_f.empty:
                raise ValueError(f"No row with portsN=={port_n} in {full_path}")
            row_f = row_f.iloc[0]
            train_sr[i - 1, j - 1] = row_f["train_SR"]
            test_sr[i - 1, j - 1] = row_f["test_SR"]

            if full_cv:
                vsum = 0.0
                for fold in (1, 2, 3):
                    cv_path = sub / f"results_cv_{fold}_l0_{i}_l2_{j}.csv"
                    cv_data = pd.read_csv(cv_path)
                    row_c = cv_data.loc[cv_data["portsN"] == port_n].iloc[0]
                    vsum += row_c["valid_SR"]
                valid_sr[i - 1, j - 1] = vsum / 3.0
            else:
                cv_path = sub / f"results_cv_3_l0_{i}_l2_{j}.csv"
                cv_data = pd.read_csv(cv_path)
                row_c = cv_data.loc[cv_data["portsN"] == port_n].iloc[0]
                valid_sr[i - 1, j - 1] = row_c["valid_SR"]

    flat = np.argmax(valid_sr)
    bi, bj = np.unravel_index(flat, valid_sr.shape)
    i_star, j_star = int(bi) + 1, int(bj) + 1

    best_full = sub / f"results_full_l0_{i_star}_l2_{j_star}.csv"
    full_data = pd.read_csv(best_full)
    row_best = full_data.loc[full_data["portsN"] == port_n].iloc[0]
    meta = {"train_SR", "test_SR", "valid_SR", "portsN"}
    wcols = [c for c in full_data.columns if c not in meta]
    w = row_best[wcols].astype(float)
    weights_out = w[w != 0]
    selected_names = weights_out.index.tolist()
    weights_values = weights_out.values

    pcsv = Path(portfolio_csv)
    if returns_index_col is not None:
        ports = pd.read_csv(pcsv, index_col=returns_index_col)
    else:
        ports = pd.read_csv(pcsv)
    missing = [c for c in selected_names if c not in ports.columns]
    if missing:
        raise KeyError(f"Portfolio CSV missing columns {missing}")
    selected_ports = ports[selected_names]

    if write_tables:
        np.savetxt(sub / f"train_SR_{port_n}.csv", train_sr, delimiter=",")
        np.savetxt(sub / f"valid_SR_{port_n}.csv", valid_sr, delimiter=",")
        np.savetxt(sub / f"test_SR_{port_n}.csv", test_sr, delimiter=",")
        selected_ports.to_csv(sub / f"Selected_Ports_{port_n}.csv", index=True)
        pd.DataFrame({"weight": weights_values}).to_csv(
            sub / f"Selected_Ports_Weights_{port_n}.csv", index=False
        )

    return {
        "subdir": sub_dir,
        "best_i_lambda0": i_star,
        "best_j_lambda2": j_star,
        "train_SR": float(train_sr[bi, bj]),
        "valid_SR": float(valid_sr[bi, bj]),
        "test_SR": float(test_sr[bi, bj]),
        "selected_columns": selected_names,
        "weights": weights_values.tolist(),
    }


def run_default_picks(port_n: int = 10, ap_root: Path | None = None) -> list[dict]:
    """Run picks for Ward clusters and (if outputs exist) LME_OP_Investment trees."""
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    out: list[dict] = []

    ward = ap / "Ward_clusters_10"
    if ward.is_dir() and CLUSTER_RETURNS.is_file():
        print(f"pick_best_lambda: Ward_clusters_10, port_n={port_n}")
        out.append(
            pick_best_lambda(
                ap,
                "Ward_clusters_10",
                port_n,
                CLUSTER_RETURNS,
                returns_index_col=0,
                full_cv=False,
            )
        )

    sub_tree = _triplet_subdir("OP", "Investment")
    tree_dir = ap / sub_tree
    tree_csv = TREE_PORT_ROOT / sub_tree / "level_all_excess_combined_filtered.csv"
    if tree_dir.is_dir() and tree_csv.is_file():
        print(f"pick_best_lambda: {sub_tree}, port_n={port_n}")
        out.append(
            pick_best_lambda(
                ap,
                sub_tree,
                port_n,
                tree_csv,
                returns_index_col=None,
                full_cv=False,
            )
        )

    return out


def print_ap_comparison(rows: list[dict]) -> None:
    """Print a compact table; use test_SR to rank models (valid was used to tune λ*)."""
    print("\n" + "=" * 66)
    print("AP comparison (lambda* = max validation Sharpe within each model)")
    print("Rank models on test_SR, not valid_SR (avoid double-counting tuning).")
    print("=" * 66)
    if not rows:
        print("No pick_best results.")
        return
    if len(rows) == 1:
        r0 = rows[0]
        sub = str(r0.get("subdir", "?"))
        print(f"Only one model: {sub}")
        print(
            f"  train_SR={r0['train_SR']:.6f}  valid_SR={r0['valid_SR']:.6f}  "
            f"test_SR={r0['test_SR']:.6f}"
        )
        if sub == "Ward_clusters_10":
            print(
                "To compare vs benchmark AP-trees: set RUN_PART2_TREES=1 and rerun main.py, "
                "or: python part_2_AP_pruning/run_part2.py"
            )
        return

    w = max(len(str(r["subdir"])) for r in rows)
    line = f"{'model':<{w}}  {'train_SR':>10}  {'valid_SR':>10}  {'test_SR':>10}"
    print(line)
    print("-" * len(line))
    for r in sorted(rows, key=lambda x: -float(x["test_SR"])):
        print(
            f"{str(r['subdir']):<{w}}  {float(r['train_SR']):10.4f}  "
            f"{float(r['valid_SR']):10.4f}  {float(r['test_SR']):10.4f}"
        )
    best = max(rows, key=lambda x: float(x["test_SR"]))
    print(f"\nHighest test_SR → {best['subdir']} ({float(best['test_SR']):.6f})")


if __name__ == "__main__":
    picked = run_default_picks()
    for r in picked:
        print(r)
    print_ap_comparison(picked)
