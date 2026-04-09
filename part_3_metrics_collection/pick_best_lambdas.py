"""
pick_best_lambda.py — Select best hyperparameters from AP-Pruning grid search.

pick_best_lambda : for a fixed k, find best (lambda0, lambda2) by validation
                   Sharpe, extract weights from the full fit, save selected ports
pick_sr_n        : sweep pick_best_lambda over k=mink..maxk, save SR_N.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def pick_best_lambda(feat1, feat2, ap_prune_result_path, port_n, lambda0, lambda2,
                     portfolio_path, port_name, full_cv=False, write_table=True):
    """
    Find best (lambda0, lambda2) at fixed portfolio count port_n.
    Input  : feat1/feat2 strings, result path, port_n (fixed k), lambda grids,
             portfolio_path to original filtered CSV, full_cv to average all folds.
    Output : returns np.ndarray [train_SR, valid_SR, test_SR] for the best combo.
             If write_table=True, also saves SR matrices and Selected_Ports_{k}.csv
             and Selected_Ports_Weights_{k}.csv to the triplet result folder.
    """
    subdir = '_'.join(['LME', feat1, feat2])
    base = ap_prune_result_path / subdir
    n_l0, n_l2 = len(lambda0), len(lambda2)

    train_SR = np.zeros((n_l0, n_l2))
    valid_SR = np.zeros((n_l0, n_l2))
    test_SR  = np.zeros((n_l0, n_l2))

    for i in range(n_l0):
        for j in range(n_l2):
            full_data = pd.read_csv(base / f'results_full_l0_{i+1}_l2_{j+1}.csv')
            cv_data   = pd.read_csv(base / f'results_cv_3_l0_{i+1}_l2_{j+1}.csv')
            full_row  = full_data[full_data['portsN'] == port_n].iloc[0]
            cv_row    = cv_data[cv_data['portsN']     == port_n].iloc[0]

            train_SR[i, j] = full_row['train_SR']
            valid_SR[i, j] = cv_row['valid_SR']
            test_SR[i, j]  = full_row['test_SR']

            if full_cv:
                for fold in [1, 2]:
                    fold_data = pd.read_csv(base / f'results_cv_{fold}_l0_{i+1}_l2_{j+1}.csv')
                    valid_SR[i, j] += fold_data[fold_data['portsN'] == port_n].iloc[0]['valid_SR']
                valid_SR[i, j] /= 3.0

    i_best, j_best = np.unravel_index(np.argmax(valid_SR), valid_SR.shape)

    # Extract weights from full fit for winning (lambda0, lambda2)
    full_data = pd.read_csv(base / f'results_full_l0_{i_best+1}_l2_{j_best+1}.csv')
    best_row  = full_data[full_data['portsN'] == port_n].iloc[0]
    meta_cols = ['train_SR', 'test_SR', 'portsN']
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


def pick_sr_n(feat1, feat2, grid_search_path, mink, maxk, lambda0, lambda2, port_path, port_file_name):
    """
    Collect best [train_SR, valid_SR, test_SR] for every k from mink to maxk.
    Input  : feat1/feat2 strings, grid_search_path, k range, lambda grids,
             port_path to original filtered CSV.
    Output : saves SR_N.csv to grid_search_path/LME_feat1_feat2/ —
             3 rows (train/valid/test) x (maxk-mink+1) columns, one per k.
             For each k the best (lambda0, lambda2) is selected independently.
    """
    subdir = '_'.join(['LME', feat1, feat2])
    sr_n = None
    for k in range(mink, maxk + 1):
        print(f"  k={k}")
        base = grid_search_path / subdir
        full_data = pd.read_csv(base / f'results_full_l0_1_l2_1.csv')
        if full_data[full_data['portsN'] == k].empty:
            print(f"  k={k} not found in results, skipping.")
            continue

        sr = pick_best_lambda(feat1, feat2, grid_search_path, k, lambda0, lambda2,
                              port_path, port_file_name, full_cv=False, write_table=True)
        

        sr_n = sr.reshape(-1, 1) if sr_n is None else np.hstack([sr_n, sr.reshape(-1, 1)])

    pd.DataFrame(sr_n, index=['train_SR', 'valid_SR', 'test_SR']).to_csv(
        grid_search_path / subdir / 'SR_N.csv', index=True)

def get_mu_sigma(feat1, feat2, ap_prune_result_path, portfolio_path,
                      port_name, port_n, n_train_valid=360):
    """
    Return mu (mean) and sigma (std) of the SDF for the selected portfolio
    at a given k, on both the train and test windows.

    These match exactly the values used to compute train_SR and test_SR
    in _run_one_lambda0: SR = sdf.mean() / sdf.std(ddof=1)
    """
    subdir   = '_'.join(['LME', feat1, feat2])
    base     = Path(ap_prune_result_path) / subdir

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
) -> list[dict]:
    """
    Run ``pick_best_lambda`` for every cross-section that has RP Part~2 grid outputs.

    Expects the ``main`` layout: ``grid_search_path/LME_feat_feat/`` with
    ``results_full_*.csv``, and matching ``portfolio_path/LME_feat_feat/port_name``.
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

    out: list[dict] = []
    for feat1, feat2 in all_triplet_pairs():
        sub = triplet_subdir_name(feat1, feat2)
        base = grid / sub
        if not (base / "results_full_l0_1_l2_1.csv").is_file():
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


if __name__ == '__main__':
    GRID = Path('data/results/grid_search/tree')
    PORTS = Path('data/results/tree_portfolios')
    L0 = [0.5, 0.55, 0.6]
    L2 = [10**-7, 10**-7.25, 10**-7.5]

    result = pick_best_lambda('OP', 'Investment', GRID, 10, L0, L2, PORTS,
                              'level_all_excess_combined_filtered.csv')
    print(f"k=10: train={result[0]:.4f}, valid={result[1]:.4f}, test={result[2]:.4f}")

    pick_sr_n('OP', 'Investment', GRID, 5, 50, L0, L2, PORTS,
              'level_all_excess_combined_filtered.csv')