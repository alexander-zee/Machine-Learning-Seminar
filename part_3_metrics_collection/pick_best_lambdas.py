"""
pick_best_lambda.py — Select best hyperparameters from AP-Pruning grid search.

Supports both:
    - Uniform kernel (original): 2D grid over (lambda0, lambda2), h_idx=1
      Reads betas from full-fit CSV to extract selected portfolios.
    - Non-uniform kernel: 3D grid over (lambda0, lambda2, h)
      SR-only CSVs → select best combo → call kernel_full_fit for the winner.

Functions
---------
pick_best_lambda        : original interface, uniform kernel (unchanged logic)
pick_best_lambda_kernel : 3D grid search for kernel case
pick_sr_n               : sweep over k, auto-routes to uniform or kernel
pick_sr_n_kernel        : sweep over k for kernel case
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def pruning_results_base(grid_dir: Path | str, feat1: str, feat2: str) -> Path:
    """
    Folder that holds ``Selected_Ports_{k}.csv`` / weights for a triplet.

    AP (tree) grids use ``{grid_dir}/uniform/LME_{feat1}_{feat2}/``.
    RP grids use ``{grid_dir}/LME_{feat1}_{feat2}/`` (no ``uniform`` layer).
    """
    g = Path(grid_dir)
    sub = "_".join(["LME", feat1, feat2])
    uniform_branch = g / "uniform" / sub
    if uniform_branch.is_dir():
        return uniform_branch
    return g / sub


def master_excess_returns_csv_name(k: int, subdir_name: str) -> str:
    return f"{subdir_name}_Master_Excess_Returns_{k}.csv"


def write_master_excess_returns_csv(
    grid_triplet_base: Path,
    k: int,
    *,
    n_train_valid: int = 360,
    portfolio_path: Path | str | None = None,
    port_name: str = "level_all_excess_combined.csv",
) -> Path | None:
    """
    Write ``{subdir}_Master_Excess_Returns_{k}.csv`` under ``grid_triplet_base``.

    Uses ``Selected_Ports_{k}.csv`` / ``Selected_Ports_Weights_{k}.csv`` on the grid
    side and the full primitive excess panel under ``portfolio_path``.
    """
    from part_3_metrics_collection.ff5 import Y_MAX, Y_MIN, generate_dates

    sub = Path(grid_triplet_base).name
    ports_root = Path(portfolio_path) if portfolio_path is not None else Path("data/results/rp_tree_portfolios")

    sel_path = Path(grid_triplet_base) / f"Selected_Ports_{k}.csv"
    w_path = Path(grid_triplet_base) / f"Selected_Ports_Weights_{k}.csv"
    panel_path = ports_root / sub / port_name
    if not sel_path.is_file() or not w_path.is_file() or not panel_path.is_file():
        return None

    ports_sel = pd.read_csv(sel_path)
    weights = pd.read_csv(w_path).values.flatten().astype(float)
    all_ports = pd.read_csv(panel_path)

    cols = list(ports_sel.columns)
    if len(weights) != len(cols):
        raise ValueError(
            f"Selected_Ports_{k} has {len(cols)} columns but weights length is {len(weights)} ({sub})"
        )
    missing = [c for c in cols if c not in all_ports.columns]
    if missing:
        raise ValueError(
            f"Columns from Selected_Ports_{k} missing in {panel_path}: {missing[:10]}"
        )

    r_master = (all_ports[cols].to_numpy(dtype=float) @ weights.reshape(-1, 1)).ravel()
    dates = generate_dates(Y_MIN, Y_MAX)
    if len(r_master) != len(dates):
        raise ValueError(
            f"Return length {len(r_master)} != calendar {len(dates)} for {sub} ({panel_path.name})"
        )

    window = np.where(np.arange(len(r_master)) < n_train_valid, "train_valid", "test")
    out = Path(grid_triplet_base) / master_excess_returns_csv_name(k, sub)
    pd.DataFrame(
        {"Date": dates.astype(int), "r_master_excess": r_master, "window": window}
    ).to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Uniform kernel (original, unchanged logic)
# ---------------------------------------------------------------------------

def _ap_grid_csv(base: Path, kind: str, i: int, j: int, fold: int | None = None) -> Path:
    """
    Resolve AP / legacy grid CSV path. Tree outputs use ``_h_1``; older exports omit it.
    """
    if kind == "full":
        stem = f"results_full_l0_{i}_l2_{j}"
    elif kind == "cv":
        if fold is None:
            raise ValueError("fold required for cv")
        stem = f"results_cv_{fold}_l0_{i}_l2_{j}"
    else:
        raise ValueError(kind)
    p_h = base / f"{stem}_h_1.csv"
    if p_h.is_file():
        return p_h
    p_legacy = base / f"{stem}.csv"
    if p_legacy.is_file():
        return p_legacy
    return p_h


def pick_best_lambda(feat1, feat2, ap_prune_result_path, port_n, lambda0, lambda2,
                     portfolio_path, port_name, full_cv=False, write_table=True):
    """
    Find best (lambda0, lambda2) at fixed portfolio count port_n.
    Grid CSVs are read from ``pruning_results_base(ap_prune_result_path, ...)``. h_idx is always 1.
    """
    subdir = '_'.join(['LME', feat1, feat2])
    base = pruning_results_base(ap_prune_result_path, feat1, feat2)
    n_l0, n_l2 = len(lambda0), len(lambda2)

    train_SR = np.zeros((n_l0, n_l2))
    valid_SR = np.zeros((n_l0, n_l2))
    test_SR  = np.zeros((n_l0, n_l2))

    for i in range(n_l0):
        for j in range(n_l2):
            full_data = pd.read_csv(_ap_grid_csv(base, "full", i + 1, j + 1))
            cv_data = pd.read_csv(_ap_grid_csv(base, "cv", i + 1, j + 1, fold=3))
            full_row = full_data[full_data['portsN'] == port_n].iloc[0]
            cv_row = cv_data[cv_data['portsN'] == port_n].iloc[0]

            train_SR[i, j] = full_row['train_SR']
            valid_SR[i, j] = cv_row['valid_SR']
            test_SR[i, j] = full_row['test_SR']

            if full_cv:
                for fold in [1, 2]:
                    fold_data = pd.read_csv(_ap_grid_csv(base, "cv", i + 1, j + 1, fold=fold))
                    valid_SR[i, j] += fold_data[fold_data['portsN'] == port_n].iloc[0]['valid_SR']
                valid_SR[i, j] /= 3.0

    i_best, j_best = np.unravel_index(np.argmax(valid_SR), valid_SR.shape)

    # Extract weights from full fit for winning (lambda0, lambda2)
    full_data = pd.read_csv(_ap_grid_csv(base, "full", i_best + 1, j_best + 1))
    best_row  = full_data[full_data['portsN'] == port_n].iloc[0]
    meta_cols = ['train_SR', 'test_SR', 'portsN']
    if 'valid_SR' in full_data.columns:
        meta_cols.append('valid_SR')
    weights   = best_row[[c for c in full_data.columns if c not in meta_cols]].values.astype(float)

    nonzero_mask = weights != 0
    ports_df     = pd.read_csv(portfolio_path / subdir / port_name)
    selected_ports = ports_df.iloc[:, np.where(nonzero_mask)[0]]

    if write_table:
        pd.DataFrame(train_SR).to_csv(base / f'train_SR_{port_n}.csv', index=False)
        pd.DataFrame(valid_SR).to_csv(base / f'valid_SR_{port_n}.csv', index=False)
        pd.DataFrame(test_SR).to_csv(base / f'test_SR_{port_n}.csv', index=False)
        selected_ports.to_csv(base / f'Selected_Ports_{port_n}.csv', index=False)
        pd.DataFrame(weights[nonzero_mask]).to_csv(base / f'Selected_Ports_Weights_{port_n}.csv', index=False)

    return np.array([train_SR[i_best, j_best], valid_SR[i_best, j_best], test_SR[i_best, j_best]])


# ---------------------------------------------------------------------------
# Kernel case: 3D grid search over (lambda0, lambda2, h)
# ---------------------------------------------------------------------------

def pick_best_lambda_kernel(feat1, feat2, ap_prune_result_path, port_n,
                            lambda0, lambda2, n_bandwidths,
                            kernel_name='gaussian',
                            full_cv=False, write_table=True):
    """
    Find best (lambda0, lambda2, h) at fixed portfolio count port_n.

    Reads validation-only CSVs from the kernel subfolder.
    Returns the best (i_best, j_best, h_best) indices and validation SR.
    test_SR is not available yet — call kernel_full_fit for the winner.

    Parameters
    ----------
    n_bandwidths : int — number of bandwidth candidates (len of bandwidth_grid)
    kernel_name  : str — subfolder name, e.g. 'gaussian'

    Returns
    -------
    dict with keys:
        'best_idx'     : (i_best, j_best, h_best) — 0-indexed
        'valid_SR'     : float (from cv fold)
        'all_valid_SR' : 3D array (n_l0, n_l2, n_h) of validation SRs
    """
    subdir = '_'.join(['LME', feat1, feat2])
    base = ap_prune_result_path / kernel_name / subdir
    n_l0, n_l2, n_h = len(lambda0), len(lambda2), n_bandwidths

    valid_SR = np.full((n_l0, n_l2, n_h), np.nan)

    for i in range(n_l0):
        for j in range(n_l2):
            for h in range(n_h):
                cv_path = base / f'results_cv_3_l0_{i+1}_l2_{j+1}_h_{h+1}.csv'
                if not cv_path.exists():
                    continue
                cv_data = pd.read_csv(cv_path)
                cv_match = cv_data[cv_data['portsN'] == port_n]
                if len(cv_match) == 0:
                    continue
                valid_SR[i, j, h] = cv_match.iloc[0]['valid_SR']

                if full_cv:
                    for fold in [1, 2]:
                        fold_path = base / f'results_cv_{fold}_l0_{i+1}_l2_{j+1}_h_{h+1}.csv'
                        if fold_path.exists():
                            fold_data = pd.read_csv(fold_path)
                            fold_match = fold_data[fold_data['portsN'] == port_n]
                            if len(fold_match) > 0:
                                valid_SR[i, j, h] += fold_match.iloc[0]['valid_SR']
                    valid_SR[i, j, h] /= 3.0

    # Find best by validation SR (ignoring NaN)
    valid_SR_flat = np.where(np.isnan(valid_SR), -np.inf, valid_SR)
    best_flat = np.argmax(valid_SR_flat)
    i_best, j_best, h_best = np.unravel_index(best_flat, valid_SR.shape)

    result = {
        'best_idx':     (int(i_best), int(j_best), int(h_best)),
        'valid_SR':     float(valid_SR[i_best, j_best, h_best]),
        'all_valid_SR': valid_SR,
    }

    if write_table:
        rows = []
        for i in range(n_l0):
            for j in range(n_l2):
                for h in range(n_h):
                    rows.append({
                        'l0_idx': i+1, 'l2_idx': j+1, 'h_idx': h+1,
                        'valid_SR': valid_SR[i, j, h],
                    })
        pd.DataFrame(rows).to_csv(
            base / f'SR_grid_{port_n}.csv', index=False)

    print(f"  Best for k={port_n}: l0_idx={i_best+1}, l2_idx={j_best+1}, "
          f"h_idx={h_best+1} -> valid_SR={result['valid_SR']:.4f}")

    return result


# ---------------------------------------------------------------------------
# SR_N collection: sweep over k
# ---------------------------------------------------------------------------

def pick_sr_n(feat1, feat2, grid_search_path, mink, maxk, lambda0, lambda2,
              port_path, port_file_name):
    """
    Collect best [train_SR, valid_SR, test_SR] for every k from mink to maxk.
    Uses ``pruning_results_base`` for the triplet folder (``uniform/LME_*`` when present).
    """
    subdir = '_'.join(['LME', feat1, feat2])
    base = pruning_results_base(grid_search_path, feat1, feat2)
    sr_n = None
    for k in range(mink, maxk + 1):
        print(f"  k={k}")
        full_data = pd.read_csv(_ap_grid_csv(base, "full", 1, 1))
        if full_data[full_data['portsN'] == k].empty:
            print(f"  k={k} not found in results, skipping.")
            continue

        sr = pick_best_lambda(feat1, feat2, grid_search_path, k, lambda0, lambda2,
                              port_path, port_file_name, full_cv=False, write_table=True)
        

        sr_n = sr.reshape(-1, 1) if sr_n is None else np.hstack([sr_n, sr.reshape(-1, 1)])

    pd.DataFrame(sr_n, index=['train_SR', 'valid_SR', 'test_SR']).to_csv(
        base / 'SR_N.csv', index=True)




def get_mu_sigma(feat1, feat2, ap_prune_result_path, portfolio_path,
                      port_name, port_n, n_train_valid=360):
    """
    Return mu (mean) and sigma (std) of the SDF for the selected portfolio
    at a given k, on both the train and test windows.

    These match exactly the values used to compute train_SR and test_SR
    in _run_one_lambda0: SR = sdf.mean() / sdf.std(ddof=1)
    """
    subdir = '_'.join(['LME', feat1, feat2])
    base = pruning_results_base(ap_prune_result_path, feat1, feat2)

    ports_df = pd.read_csv(base / f'Selected_Ports_{port_n}.csv')
    weights  = pd.read_csv(base / f'Selected_Ports_Weights_{port_n}.csv').values.flatten()

    all_ports   = pd.read_csv(Path(portfolio_path) / subdir / port_name)
    ports_train = all_ports.iloc[:n_train_valid][ports_df.columns]
    ports_test  = all_ports.iloc[n_train_valid:][ports_df.columns]

    sdf_train = ports_train.values @ weights
    sdf_test  = ports_test.values  @ weights

    return {
        'train': {
            'mu':    sdf_train.mean(),
            'sigma': sdf_train.std(ddof=1),
            'SR':    sdf_train.mean() / sdf_train.std(ddof=1),
        },
        'test': {
            'mu':    sdf_test.mean(),
            'sigma': sdf_test.std(ddof=1),
            'SR':    sdf_test.mean() / sdf_test.std(ddof=1),
        },
    }


def run_rp_picks_all(
    port_n: int = 10,
    grid_search_path: Path | str | None = None,
    portfolio_path: Path | str | None = None,
    port_name: str = "level_all_excess_combined.csv",
    lambda0: list | None = None,
    lambda2: list | None = None,
    full_cv: bool = False,
    write_table: bool = True,
    pairs: list[tuple[str, str]] | None = None,
    *,
    show_progress: bool = True,
) -> list[dict]:
    """
    Run ``pick_best_lambda`` for every cross-section that has RP Part~2 grid outputs.

    Expects the ``main`` layout: ``grid_search_path/LME_feat_feat/`` with
    ``results_full_*.csv``, and matching ``portfolio_path/LME_feat_feat/port_name``.

    If ``pairs`` is given (e.g. from ``run_all_rp_cross_sections --feat1/--feat2``),
    only those ``(feat1, feat2)`` are processed; otherwise all ``C(9,2)`` triplets.

    ``show_progress`` — optional tqdm bar (used by ``run_all_rp_cross_sections --pick-best``).
    """
    from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
        all_triplet_pairs,
        triplet_subdir_name,
    )

    grid = Path(grid_search_path) if grid_search_path is not None else Path("data/results/grid_search/rp_tree")
    ports_root = Path(portfolio_path) if portfolio_path is not None else Path("data/results/rp_tree_portfolios")
    if lambda0 is None:
        lambda0 = [0.5, 0.55, 0.6]
    if lambda2 is None:
        lambda2 = [10**-7, 10**-7.25, 10**-7.5]

    feat_pairs = list(pairs) if pairs is not None else list(all_triplet_pairs())

    out: list[dict] = []
    loop = feat_pairs
    if show_progress:
        try:
            from tqdm import tqdm

            loop = tqdm(feat_pairs, desc="pick_best_lambda (RP)", unit="triplet")
        except Exception:
            loop = feat_pairs
    for feat1, feat2 in loop:
        sub = triplet_subdir_name(feat1, feat2)
        base = grid / sub
        if not _ap_grid_csv(base, "full", 1, 1).is_file():
            continue
        rp_csv = ports_root / sub / port_name
        if not rp_csv.is_file():
            print(f"  skip {sub}: missing {rp_csv}")
            continue
        print(f"pick_best_lambda (RP): {sub}, port_n={port_n}")
        try:
            sr = pick_best_lambda(
                feat1,
                feat2,
                grid,
                port_n,
                lambda0,
                lambda2,
                ports_root,
                port_name,
                full_cv=full_cv,
                write_table=write_table,
            )
            out.append(
                {
                    "feat1": feat1,
                    "feat2": feat2,
                    "subdir": sub,
                    "train_SR": float(sr[0]),
                    "valid_SR": float(sr[1]),
                    "test_SR": float(sr[2]),
                }
            )
        except Exception as e:
            print(f"  skipped {sub}: {e}")
    return out


