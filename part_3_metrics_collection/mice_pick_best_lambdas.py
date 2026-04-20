"""
mice_pick_best_lambda.py — Select best hyperparameters from MICE RP-Pruning grid search.
 
Based directly on pick_best_lambda.py (AP trees). The only structural differences are:
  - subdir encodes all_features and n_features_per_split instead of a triplet name
  - _base_path resolves the kernel subfolder from kernel_cls
  - mice_pick_best_lambda handles missing port_n rows gracefully (returns nan)
  - mice_pick_sr_n skips k values not present in results instead of crashing
"""
 
from __future__ import annotations
 
import numpy as np
import pandas as pd
from pathlib import Path
 
 
def _base_path(ap_prune_result_path: Path, subdir: str, kernel_cls=None) -> Path:
    """
    Resolve the results directory for a given kernel and subdir.
 
    Mirrors the output layout of Mice_RP_Pruning:
        output_path / {kernel_name} / {subdir}
 
    kernel_cls=None resolves to 'uniform'.
    """
    kernel_name = (
        kernel_cls.__name__.lower().replace('kernel', '')
        if kernel_cls is not None
        else 'uniform'
    )
    return Path(ap_prune_result_path) / kernel_name / subdir
 
 
# ---------------------------------------------------------------------------
# Uniform kernel: 2D grid over (lambda0, lambda2), h_idx always 1
# ---------------------------------------------------------------------------
 
def mice_pick_best_lambda(all_features, n_features_per_split,
                          ap_prune_result_path, port_n, lambda0, lambda2,
                          portfolio_path, port_name, full_cv=False,
                          write_table=True, kernel_cls=None):
    """
    Find best (lambda0, lambda2) at fixed portfolio count port_n.
 
    Mirrors pick_best_lambda (AP trees). Differences:
      - subdir = '_'.join(all_features) + '__nf{n_features_per_split}'
      - reads from {kernel_name}/ subfolder via _base_path
      - returns [nan, nan, nan] if port_n is absent from any results file
 
    Parameters
    ----------
    all_features         : list of all characteristic names
    n_features_per_split : int, used to locate the correct subdirectory
    ap_prune_result_path : Path to grid search results root
    port_n               : fixed k (portfolio count) to evaluate
    lambda0, lambda2     : lists of penalty values
    portfolio_path       : Path to combined portfolio CSVs root
    port_name            : filename of the combined excess return CSV
    full_cv              : if True, average valid_SR across all 3 folds
    write_table          : if True, save SR matrices and selected ports to disk
    kernel_cls           : kernel class used during pruning (default None → uniform)
 
    Returns
    -------
    np.ndarray [train_SR, valid_SR, test_SR] for the best (lambda0, lambda2),
    or np.ndarray [nan, nan, nan] if port_n not found in any results file.
    """
    subdir = f"{'_'.join(all_features)}__nf{n_features_per_split}"
    base   = _base_path(ap_prune_result_path, subdir, kernel_cls)
    n_l0, n_l2 = len(lambda0), len(lambda2)
 
    train_SR = np.full((n_l0, n_l2), np.nan)
    valid_SR = np.full((n_l0, n_l2), np.nan)
    test_SR  = np.full((n_l0, n_l2), np.nan)
 
    for i in range(n_l0):
        for j in range(n_l2):
            full_data = pd.read_csv(base / f'results_full_l0_{i+1}_l2_{j+1}_h_1.csv')
            cv_data   = pd.read_csv(base / f'results_cv_3_l0_{i+1}_l2_{j+1}_h_1.csv')
 
            full_match = full_data[full_data['portsN'] == port_n]
            cv_match   = cv_data[cv_data['portsN']     == port_n]
 
            if full_match.empty or cv_match.empty:
                continue
 
            train_SR[i, j] = full_match.iloc[0]['train_SR']
            valid_SR[i, j] = cv_match.iloc[0]['valid_SR']
            test_SR[i, j]  = full_match.iloc[0]['test_SR']
 
            if full_cv:
                for fold in [1, 2]:
                    fold_data  = pd.read_csv(base / f'results_cv_{fold}_l0_{i+1}_l2_{j+1}_h_1.csv')
                    fold_match = fold_data[fold_data['portsN'] == port_n]
                    if not fold_match.empty:
                        valid_SR[i, j] += fold_match.iloc[0]['valid_SR']
                valid_SR[i, j] /= 3.0
 
    if np.all(np.isnan(valid_SR)):
        print(f"  k={port_n} not found in any results file, skipping.")
        return np.array([np.nan, np.nan, np.nan])
 
    i_best, j_best = np.unravel_index(np.nanargmax(valid_SR), valid_SR.shape)
 
    full_data = pd.read_csv(base / f'results_full_l0_{i_best+1}_l2_{j_best+1}_h_1.csv')
    best_row  = full_data[full_data['portsN'] == port_n].iloc[0]
    meta_cols = ['train_SR', 'test_SR', 'portsN']
    if 'valid_SR' in full_data.columns:
        meta_cols.append('valid_SR')
    weights = best_row[[c for c in full_data.columns if c not in meta_cols]].values.astype(float)
 
    nonzero_mask   = weights != 0
    ports_df       = pd.read_csv(Path(portfolio_path) / subdir / port_name)
    selected_ports = ports_df.iloc[:, np.where(nonzero_mask)[0]]
 
    if write_table:
        pd.DataFrame(train_SR).to_csv(base / f'train_SR_{port_n}.csv', index=False)
        pd.DataFrame(valid_SR).to_csv(base / f'valid_SR_{port_n}.csv', index=False)
        pd.DataFrame(test_SR).to_csv(base / f'test_SR_{port_n}.csv', index=False)
        selected_ports.to_csv(base / f'Selected_Ports_{port_n}.csv', index=False)
        pd.DataFrame(weights[nonzero_mask]).to_csv(
            base / f'Selected_Ports_Weights_{port_n}.csv', index=False)
        
        pd.DataFrame([{
        'lambda0': lambda0[i_best],
        'lambda2': lambda2[j_best],
        }]).to_csv(base / f'best_hyperparams_{port_n}.csv', index=False)
 
    return np.array([train_SR[i_best, j_best], valid_SR[i_best, j_best], test_SR[i_best, j_best]])
 
 
# ---------------------------------------------------------------------------
# Non-uniform kernel: 3D grid search over (lambda0, lambda2, h)
# ---------------------------------------------------------------------------
 
def mice_pick_best_lambda_kernel(all_features, n_features_per_split,
                                 ap_prune_result_path, port_n,
                                 lambda0, lambda2, n_bandwidths,
                                 kernel_cls, portfolio_path, port_name,
                                 full_cv=False, write_table=True):
    """
    Find best (lambda0, lambda2, h) for the non-uniform kernel case.
 
    Mirrors pick_best_lambda_kernel (AP trees). Differences:
      - subdir encodes all_features and n_features_per_split
      - reads from {kernel_name}/ subfolder via _base_path
 
    Parameters
    ----------
    all_features         : list of all characteristic names
    n_features_per_split : int, used to locate the correct subdirectory
    ap_prune_result_path : Path to grid search results root
    port_n               : fixed k (portfolio count) to evaluate
    lambda0, lambda2     : lists of penalty values
    n_bandwidths         : number of bandwidth candidates
    kernel_cls           : kernel class used during pruning
    portfolio_path       : Path to combined portfolio CSVs root
    port_name            : filename of the combined excess return CSV
    full_cv              : if True, average valid_SR across all 3 folds
    write_table          : if True, save SR_grid_{port_n}.csv to disk
 
    Returns
    -------
    dict with keys:
        'best_idx'     : (i_best, j_best, h_best) — 0-indexed
        'valid_SR'     : float
        'all_valid_SR' : 3D array (n_l0, n_l2, n_h) of validation SRs
    """
    subdir = f"{'_'.join(all_features)}__nf{n_features_per_split}"
    base   = _base_path(ap_prune_result_path, subdir, kernel_cls)
    n_l0, n_l2, n_h = len(lambda0), len(lambda2), n_bandwidths
 
    valid_SR = np.full((n_l0, n_l2, n_h), np.nan)
 
    for i in range(n_l0):
        for j in range(n_l2):
            for h in range(n_h):
                cv_path = base / f'results_cv_3_l0_{i+1}_l2_{j+1}_h_{h+1}.csv'
                if not cv_path.exists():
                    continue
                cv_data  = pd.read_csv(cv_path)
                cv_match = cv_data[cv_data['portsN'] == port_n]
                if cv_match.empty:
                    continue
                valid_SR[i, j, h] = cv_match.iloc[0]['valid_SR']
 
                if full_cv:
                    for fold in [1, 2]:
                        fold_path  = base / f'results_cv_{fold}_l0_{i+1}_l2_{j+1}_h_{h+1}.csv'
                        if fold_path.exists():
                            fold_data  = pd.read_csv(fold_path)
                            fold_match = fold_data[fold_data['portsN'] == port_n]
                            if not fold_match.empty:
                                valid_SR[i, j, h] += fold_match.iloc[0]['valid_SR']
                    valid_SR[i, j, h] /= 3.0
 
    valid_SR_flat = np.where(np.isnan(valid_SR), -np.inf, valid_SR)
    i_best, j_best, h_best = np.unravel_index(np.argmax(valid_SR_flat), valid_SR.shape)
 
    result = {
        'best_idx':     (int(i_best), int(j_best), int(h_best)),
        'valid_SR':     float(valid_SR[i_best, j_best, h_best]),
        'all_valid_SR': valid_SR,
    }
 
    if write_table:
        rows = [
            {'l0_idx': i+1, 'l2_idx': j+1, 'h_idx': h+1,
             'valid_SR': valid_SR[i, j, h]}
            for i in range(n_l0)
            for j in range(n_l2)
            for h in range(n_h)
        ]
        pd.DataFrame(rows).to_csv(base / f'SR_grid_{port_n}.csv', index=False)
 
    print(f"  Best for k={port_n}: l0_idx={i_best+1}, l2_idx={j_best+1}, "
          f"h_idx={h_best+1}  valid_SR={result['valid_SR']:.4f}")
 
    return result
 
 
# ---------------------------------------------------------------------------
# SR_N sweep over k
# ---------------------------------------------------------------------------
 
def mice_pick_sr_n(all_features, n_features_per_split, grid_search_path,
                   mink, maxk, lambda0, lambda2, port_path, port_file_name,
                   kernel_cls=None):
    """
    Collect best [train_SR, valid_SR, test_SR] for every k from mink to maxk.
 
    Mirrors pick_sr_n (AP trees). Differences:
      - subdir encodes all_features and n_features_per_split
      - reads from {kernel_name}/ subfolder via _base_path
      - skips k values not present in results instead of crashing
 
    Parameters
    ----------
    all_features         : list of all characteristic names
    n_features_per_split : int, used to locate the correct subdirectory
    grid_search_path     : Path to grid search results root
    mink, maxk           : range of portfolio counts to sweep
    lambda0, lambda2     : lists of penalty values
    port_path            : Path to combined portfolio CSVs root
    port_file_name       : filename of the combined excess return CSV
    kernel_cls           : kernel class used during pruning (default None → uniform)
    """
    subdir = f"{'_'.join(all_features)}__nf{n_features_per_split}"
    base   = _base_path(grid_search_path, subdir, kernel_cls)
    sr_n   = None
 
    results_file = base / 'results_full_l0_1_l2_1_h_1.csv'
    if not results_file.is_file():
        print(f"  Results not found at {results_file}, aborting.")
        return
 
    full_data = pd.read_csv(results_file)
 
    for k in range(mink, maxk + 1):
        print(f"  k={k}")
        if full_data[full_data['portsN'] == k].empty:
            print(f"  k={k} not found in results, skipping.")
            continue
 
        sr = mice_pick_best_lambda(
            all_features, n_features_per_split, grid_search_path, k,
            lambda0, lambda2, port_path, port_file_name,
            full_cv=False, write_table=True, kernel_cls=kernel_cls,
        )
 
        if np.any(np.isnan(sr)):
            print(f"  k={k} skipped (no valid results).")
            continue
 
        sr_n = sr.reshape(-1, 1) if sr_n is None else np.hstack([sr_n, sr.reshape(-1, 1)])
 
    if sr_n is not None:
        pd.DataFrame(sr_n, index=['train_SR', 'valid_SR', 'test_SR']).to_csv(
            base / 'SR_N.csv', index=True)
 
 
# ---------------------------------------------------------------------------
# Mu / sigma of the SDF
# ---------------------------------------------------------------------------
 
def mice_get_mu_sigma(all_features, n_features_per_split, ap_prune_result_path,
                      portfolio_path, port_name, port_n, n_train_valid=360,
                      kernel_cls=None):
    """
    Return mu (mean) and sigma (std) of the SDF for the selected portfolio
    at a given k, on both the train and test windows.
 
    The SDF is normalised to unit variance on the training window before
    computing mu and sigma, making results comparable across n_features values.
    The SR is scale-invariant and unaffected by this normalisation.
 
    Mirrors get_mu_sigma (AP trees). Differences:
      - subdir encodes all_features and n_features_per_split
      - reads from {kernel_name}/ subfolder via _base_path
      - normalises SDF to unit train-window variance
 
    Parameters
    ----------
    all_features         : list of all characteristic names
    n_features_per_split : int, used to locate the correct subdirectory
    ap_prune_result_path : Path to grid search results root
    portfolio_path       : Path to combined portfolio CSVs root
    port_name            : filename of the combined excess return CSV
    port_n               : fixed k (portfolio count) to evaluate
    n_train_valid        : number of months in the train + validation window
    kernel_cls           : kernel class used during pruning (default None → uniform)
 
    Returns
    -------
    dict with keys 'train' and 'test', each containing mu, sigma, SR
    """
    subdir = f"{'_'.join(all_features)}__nf{n_features_per_split}"
    base   = _base_path(ap_prune_result_path, subdir, kernel_cls)
 
    ports_df = pd.read_csv(base / f'Selected_Ports_{port_n}.csv')
    weights  = pd.read_csv(base / f'Selected_Ports_Weights_{port_n}.csv').values.flatten()
 
    all_ports   = pd.read_csv(Path(portfolio_path) / subdir / port_name)
    ports_train = all_ports.iloc[:n_train_valid][ports_df.columns]
    ports_test  = all_ports.iloc[n_train_valid:][ports_df.columns]
 
    sdf_train = ports_train.values @ weights
    sdf_test  = ports_test.values  @ weights
 
    # Normalise to unit train-window variance so mu and sigma are comparable
    # across n_features values. SR is scale-invariant and unaffected.
    scale     = sdf_train.std(ddof=1)
    sdf_train = sdf_train / scale
    sdf_test  = sdf_test  / scale
 
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
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == '__main__':
    GRID  = Path('data/results/grid_search/mice_rp_tree')
    PORTS = Path('data/results/mice_rp_tree_portfolios')
    L0    = [0.5, 0.55, 0.6]
    L2    = [10**-7, 10**-7.25, 10**-7.5]
    ALL_FEATURES = [
        'LME', 'BEME', 'r12_2', 'OP', 'Investment',
        'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
    ]
    N_FEATURES_PER_SPLIT = 3
 
    result = mice_pick_best_lambda(
        ALL_FEATURES, N_FEATURES_PER_SPLIT, GRID, 10, L0, L2, PORTS,
        'level_all_excess_combined.csv',
    )
    print(f"k=10: train={result[0]:.4f}, valid={result[1]:.4f}, test={result[2]:.4f}")
 
    mice_pick_sr_n(
        ALL_FEATURES, N_FEATURES_PER_SPLIT, GRID, 5, 50, L0, L2, PORTS,
        'level_all_excess_combined.csv',
    )