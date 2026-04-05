"""
Select (λ0, λ2) by maximum validation Sharpe; export SR grids and chosen portfolios.

Matches part_3_metrics_collection/Pick_Best_Lambda.R logic:
  - train/test Sharpe from results_full_*; validation from results_cv_* (fold 3, or mean of 1–3 if full_cv).
  - Chooses the grid point with highest valid_SR; ties → first occurrence (numpy argmax order).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
AP_PRUNE_DEFAULT = REPO_ROOT / "data" / "results" / "ap_pruning"
AP_PRUNE_OPT_QUANT = REPO_ROOT / "data" / "results" / "ap_pruning_optquantile"
TREE_PORT_ROOT = REPO_ROOT / "data" / "results" / "tree_portfolios"
TREE_PORT_OPT_QUANT = REPO_ROOT / "data" / "results" / "tree_portfolios_optquantile"
RP_TREE_PORT_ROOT = REPO_ROOT / "data" / "results" / "rp_tree_portfolios"
CLUSTER_RETURNS = REPO_ROOT / "data" / "portfolios" / "clusters" / "cluster_returns.csv"

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (  # noqa: E402
    FEATS_LIST,
)


def load_lambda_grid_meta(ap_subdir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read lambda0/lambda2 vectors written by Part 2 (lasso_valid_full)."""
    p = ap_subdir / "lambda_grid_meta.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return np.asarray(d["lambda0"], dtype=float), np.asarray(d["lambda2"], dtype=float)


def _feasible_common_port_n(
    sub: Path, n_lambda0: int, n_lambda2: int, port_n_target: int
) -> int:
    """
    Largest n <= port_n_target such that every results_full_l0_*_l2_*.csv contains a row
    with portsN == n. Needed when LARS never reaches the requested count (e.g. Ward + tiny λ₂).
    """
    caps: list[int] = []
    for i in range(1, n_lambda0 + 1):
        for j in range(1, n_lambda2 + 1):
            p = sub / f"results_full_l0_{i}_l2_{j}.csv"
            df = pd.read_csv(p, usecols=["portsN"])
            le = df.loc[df["portsN"] <= port_n_target, "portsN"]
            if le.empty:
                raise ValueError(
                    f"No LASSO row with portsN<={port_n_target} in {p} "
                    f"(max in file={int(df['portsN'].max())})."
                )
            caps.append(int(le.max()))
    return min(caps)


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


def discover_lme_tree_ap_subdirs(
    ap_root: Path | None = None,
    tree_port_root: Path | None = None,
) -> list[str]:
    """
    Subdirectories ``LME_*`` under ap_pruning that have Part~1 filtered tree returns
    (so Part~2 / pick_best can run).
    """
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    tpr = Path(tree_port_root) if tree_port_root is not None else TREE_PORT_ROOT
    if not ap.is_dir():
        return []
    out: list[str] = []
    for p in sorted(ap.iterdir()):
        if not p.is_dir() or not p.name.startswith("LME_"):
            continue
        tree_csv = tpr / p.name / "level_all_excess_combined_filtered.csv"
        if tree_csv.is_file():
            out.append(p.name)
    return out


def discover_rp_tree_ap_subdirs(ap_root: Path | None = None) -> list[str]:
    """``RP_LME_*`` folders under ap_pruning with matching RP Part~1 combined CSV."""
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    if not ap.is_dir():
        return []
    out: list[str] = []
    for p in sorted(ap.iterdir()):
        if not p.is_dir() or not p.name.startswith("RP_LME_"):
            continue
        core = p.name[3:]  # LME_feat_feat
        tree_csv = RP_TREE_PORT_ROOT / core / "level_all_excess_combined.csv"
        if tree_csv.is_file():
            out.append(p.name)
    return out


def discover_tv_tree_ap_subdirs(ap_root: Path | None = None) -> list[str]:
    """``TV_LME_*`` folders under ap_pruning with matching AP-tree filtered returns."""
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    if not ap.is_dir():
        return []
    out: list[str] = []
    for p in sorted(ap.iterdir()):
        if not p.is_dir() or not p.name.startswith("TV_LME_"):
            continue
        core = p.name[3:]  # LME_feat_feat
        tree_csv = TREE_PORT_ROOT / core / "level_all_excess_combined_filtered.csv"
        if tree_csv.is_file():
            out.append(p.name)
    return out


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

    meta = load_lambda_grid_meta(sub)
    if meta is not None:
        lambda0_vals, lambda2_vals = meta
        n_lambda0, n_lambda2 = int(lambda0_vals.size), int(lambda2_vals.size)
    else:
        if n_lambda0 != 3 or n_lambda2 != 3:
            raise ValueError(
                f"{sub}: missing lambda_grid_meta.json; cannot infer grid size "
                f"(expected 3x3 legacy or re-run Part 2). Got n_lambda0={n_lambda0}, n_lambda2={n_lambda2}."
            )
        lambda0_vals = np.array([0.0, 0.1, 0.2], dtype=float)
        lambda2_vals = np.array([0.01, 0.05, 0.1], dtype=float)

    port_n_use = _feasible_common_port_n(sub, n_lambda0, n_lambda2, port_n)
    if port_n_use < port_n:
        print(
            f"pick_best_lambda [{sub_dir}]: requested portsN={port_n}, but the LARS path "
            f"does not reach that count in every lambda cell; using portsN={port_n_use} "
            f"for the Sharpe grid and lambda* selection."
        )

    train_sr = np.zeros((n_lambda0, n_lambda2))
    valid_sr = np.zeros((n_lambda0, n_lambda2))
    test_sr = np.zeros((n_lambda0, n_lambda2))

    for i in range(1, n_lambda0 + 1):
        for j in range(1, n_lambda2 + 1):
            full_path = sub / f"results_full_l0_{i}_l2_{j}.csv"
            full_data = pd.read_csv(full_path)
            row_f = full_data.loc[full_data["portsN"] == port_n_use]
            if row_f.empty:
                raise ValueError(
                    f"No row with portsN=={port_n_use} in {full_path} "
                    f"(feasible scan should have prevented this)."
                )
            row_f = row_f.iloc[0]
            train_sr[i - 1, j - 1] = row_f["train_SR"]
            test_sr[i - 1, j - 1] = row_f["test_SR"]

            if full_cv:
                vsum = 0.0
                for fold in (1, 2, 3):
                    cv_path = sub / f"results_cv_{fold}_l0_{i}_l2_{j}.csv"
                    cv_data = pd.read_csv(cv_path)
                    row_c = cv_data.loc[cv_data["portsN"] == port_n_use].iloc[0]
                    vsum += row_c["valid_SR"]
                valid_sr[i - 1, j - 1] = vsum / 3.0
            else:
                cv_path = sub / f"results_cv_3_l0_{i}_l2_{j}.csv"
                cv_data = pd.read_csv(cv_path)
                row_c = cv_data.loc[cv_data["portsN"] == port_n_use].iloc[0]
                valid_sr[i - 1, j - 1] = row_c["valid_SR"]

    flat = np.argmax(valid_sr)
    bi, bj = np.unravel_index(flat, valid_sr.shape)
    i_star, j_star = int(bi) + 1, int(bj) + 1

    best_full = sub / f"results_full_l0_{i_star}_l2_{j_star}.csv"
    full_data = pd.read_csv(best_full)
    row_best = full_data.loc[full_data["portsN"] == port_n_use].iloc[0]
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
        np.savetxt(sub / f"train_SR_{port_n_use}.csv", train_sr, delimiter=",")
        np.savetxt(sub / f"valid_SR_{port_n_use}.csv", valid_sr, delimiter=",")
        np.savetxt(sub / f"test_SR_{port_n_use}.csv", test_sr, delimiter=",")
        selected_ports.to_csv(sub / f"Selected_Ports_{port_n_use}.csv", index=True)
        pd.DataFrame({"weight": weights_values}).to_csv(
            sub / f"Selected_Ports_Weights_{port_n_use}.csv", index=False
        )

    out = {
        "subdir": sub_dir,
        "port_n": int(port_n_use),
        "best_i_lambda0": i_star,
        "best_j_lambda2": j_star,
        "train_SR": float(train_sr[bi, bj]),
        "valid_SR": float(valid_sr[bi, bj]),
        "test_SR": float(test_sr[bi, bj]),
        "selected_columns": selected_names,
        "weights": weights_values.tolist(),
        "lambda0": lambda0_vals.tolist(),
        "lambda2": lambda2_vals.tolist(),
    }
    if port_n_use < port_n:
        out["port_n_requested"] = int(port_n)
    return out


def run_default_picks(
    port_n: int = 10,
    ap_root: Path | None = None,
    tree_port_root: Path | None = None,
    *,
    include_ward: bool = True,
) -> list[dict]:
    """Run picks for Ward clusters and every AP-tree cross-section that has Part~1+2 outputs."""
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    tpr = Path(tree_port_root) if tree_port_root is not None else TREE_PORT_ROOT
    out: list[dict] = []

    if include_ward:
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

    for sub_tree in discover_lme_tree_ap_subdirs(ap, tree_port_root=tpr):
        tree_dir = ap / sub_tree
        tree_csv = tpr / sub_tree / "level_all_excess_combined_filtered.csv"
        if not tree_dir.is_dir() or not tree_csv.is_file():
            continue
        print(f"pick_best_lambda: {sub_tree}, port_n={port_n}")
        try:
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
        except Exception as e:
            print(f"  skipped {sub_tree}: {e}")

    return out


def run_rp_picks_all(port_n: int = 10, ap_root: Path | None = None) -> list[dict]:
    """Run ``pick_best_lambda`` for every discovered RP-tree cross-section."""
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    out: list[dict] = []
    for rp_sub in discover_rp_tree_ap_subdirs(ap):
        core = rp_sub[3:]
        rp_csv = RP_TREE_PORT_ROOT / core / "level_all_excess_combined.csv"
        print(f"pick_best_lambda: {rp_sub}, port_n={port_n}")
        try:
            out.append(
                pick_best_lambda(
                    ap,
                    rp_sub,
                    port_n,
                    rp_csv,
                    returns_index_col=None,
                    full_cv=False,
                )
            )
        except Exception as e:
            print(f"  skipped {rp_sub}: {e}")
    return out


def run_tv_picks_all(port_n: int = 10, ap_root: Path | None = None) -> list[dict]:
    """Run ``pick_best_lambda`` for every discovered TV-tree cross-section."""
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    out: list[dict] = []
    for tv_sub in discover_tv_tree_ap_subdirs(ap):
        core = tv_sub[3:]
        tree_csv = TREE_PORT_ROOT / core / "level_all_excess_combined_filtered.csv"
        print(f"pick_best_lambda: {tv_sub}, port_n={port_n}")
        try:
            out.append(
                pick_best_lambda(
                    ap,
                    tv_sub,
                    port_n,
                    tree_csv,
                    returns_index_col=None,
                    full_cv=False,
                )
            )
        except Exception as e:
            print(f"  skipped {tv_sub}: {e}")
    return out


def run_full_paper_picks(
    ap_root: Path | None = None,
    ward_port_n: int = 10,
    tree_port_ns: tuple[int, ...] = (10, 40),
    feat1_tree: str = "OP",
    feat2_tree: str = "Investment",
    all_tree_triplets: bool = False,
    tree_port_root: Path | None = None,
) -> list[dict]:
    """
    All picks typically reported in the paper-style bundle: Ward (N<=10), AP-trees for
    several N (e.g. 10 and 40). Skips tree N if no row exists in LASSO output.

    If ``all_tree_triplets`` is True, runs tree picks for every discovered ``LME_*``
    cross-section (see ``discover_lme_tree_ap_subdirs``) instead of only
    ``feat1_tree`` × ``feat2_tree`` (can be very many figures/tables).
    """
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    tpr = Path(tree_port_root) if tree_port_root is not None else TREE_PORT_ROOT
    out: list[dict] = []

    ward = ap / "Ward_clusters_10"
    if ward.is_dir() and CLUSTER_RETURNS.is_file():
        print(f"pick_best_lambda: Ward_clusters_10, port_n={ward_port_n}")
        out.append(
            pick_best_lambda(
                ap,
                "Ward_clusters_10",
                ward_port_n,
                CLUSTER_RETURNS,
                returns_index_col=0,
                full_cv=False,
            )
        )

    if all_tree_triplets:
        tree_subs = discover_lme_tree_ap_subdirs(ap, tree_port_root=tpr)
    else:
        tree_subs = [_triplet_subdir(feat1_tree, feat2_tree)]

    for sub_tree in tree_subs:
        tree_dir = ap / sub_tree
        tree_csv = tpr / sub_tree / "level_all_excess_combined_filtered.csv"
        if not (tree_dir.is_dir() and tree_csv.is_file()):
            continue

        for pn in tree_port_ns:
            print(f"pick_best_lambda: {sub_tree}, port_n={pn}")
            try:
                out.append(
                    pick_best_lambda(
                        ap,
                        sub_tree,
                        pn,
                        tree_csv,
                        returns_index_col=None,
                        full_cv=False,
                    )
                )
            except (ValueError, KeyError) as e:
                print(f"  skipped {sub_tree} port_n={pn}: {e}")

    return out


def _pick_row_label(r: dict) -> str:
    sub = str(r.get("subdir", "?"))
    pn = r.get("port_n")
    return f"{sub} (N={pn})" if pn is not None else sub


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
        sub = _pick_row_label(r0)
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

    labels = [_pick_row_label(r) for r in rows]
    w = max(len(s) for s in labels)
    line = f"{'model':<{w}}  {'train_SR':>10}  {'valid_SR':>10}  {'test_SR':>10}"
    print(line)
    print("-" * len(line))
    order = sorted(range(len(rows)), key=lambda i: -float(rows[i]["test_SR"]))
    for i in order:
        r = rows[i]
        lab = labels[i]
        print(
            f"{lab:<{w}}  {float(r['train_SR']):10.4f}  "
            f"{float(r['valid_SR']):10.4f}  {float(r['test_SR']):10.4f}"
        )
    best = max(rows, key=lambda x: float(x["test_SR"]))
    print(f"\nHighest test_SR -> {_pick_row_label(best)} ({float(best['test_SR']):.6f})")


if __name__ == "__main__":
    picked = run_default_picks()
    for r in picked:
        print(r)
    print_ap_comparison(picked)
