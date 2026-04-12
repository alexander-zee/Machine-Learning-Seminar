"""
mice_rp_pruning.py — RP-Pruning for randomly generated triplets.
 
Mirrors AP_Pruning exactly, with two differences:
  1. Column names follow the RP tree convention '<tree_id>.<node>'
     (e.g. '07.1111') rather than the AP convention '<feat_combo>.<node>'
     (e.g. '1111.11111'). Depth extraction is identical once the node
     string is isolated: depth = len(col.split('.')[1]) - 1.
  2. The subdirectory encodes all_features and n_features_per_split rather
     than a triplet name: LME_BEME_..._IdioVol__nf3/
 
Output layout (mirrors AP_Pruning):
    output_path / {kernel_name} / {subdir} /
        results_cv_{fold}_l0_{i}_l2_{j}_h_{h}.csv
        results_full_l0_{i}_l2_{j}_h_{h}.csv   (UniformKernel only)
        grid_manifest.json
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from .lasso_valid_par_full import lasso_valid_full
from .kernels.uniform import UniformKernel
 
 
def Mice_RP_Pruning(all_features, n_features_per_split, input_path, input_file_name,
                    output_path, n_train_valid=360, cvN=3, runFullCV=False,
                    kmin=5, kmax=50, RunParallel=False, ParallelN=10, IsTree=True,
                    lambda0=None, lambda2=None,
                    kernel_cls=None, state=None, n_bandwidths=None):
    """
    Run RP-Pruning grid search across all (lambda0, lambda2, h).
 
    Parameters
    ----------
    all_features         : list of all characteristic names
    n_features_per_split : int, determines subdirectory alongside all_features
    input_path           : Path to the directory containing the subdir
    input_file_name      : CSV filename, e.g. 'level_all_excess_combined.csv'
    output_path          : root Path for results
    n_train_valid        : total train+validation window length (default 360)
    cvN                  : number of cross-validation folds (default 3)
    runFullCV            : if True, also run on the full training sample
    kmin, kmax           : range of portfolio-count targets for LARS
    RunParallel          : use joblib parallelism across (lambda0, lambda2) grid
    ParallelN            : number of parallel workers
    IsTree               : if False, skip depth weighting (flat weight = 1)
    lambda0              : list of l0 penalty values (default [0.50, 0.55, 0.60])
    lambda2              : list of l2 penalty values (default [1e-7, 10**-7.25, 1e-7.5])
    kernel_cls           : kernel class (not instance), e.g. GaussianKernel.
                           Defaults to UniformKernel.
    state                : pd.Series (T,) of monthly state variable values aligned
                           with portfolio return rows. None for UniformKernel.
    n_bandwidths         : number of bandwidth candidates. None uses kernel default.
 
    Output
    ------
    output_path / {kernel_name} / {subdir} /
        results_cv_{fold}_l0_{i}_l2_{j}_h_{h}.csv
        results_full_l0_{i}_l2_{j}_h_{h}.csv   (UniformKernel only)
        grid_manifest.json
    """
    if lambda0 is None:
        lambda0 = [0.5, 0.55, 0.6]
    if lambda2 is None:
        lambda2 = [10**-7, 10**-7.25, 10**-7.5]
    if kernel_cls is None:
        kernel_cls = UniformKernel
 
    kernel_name = kernel_cls.__name__.lower().replace('kernel', '')
    sub_dir     = f"{'_'.join(all_features)}__nf{n_features_per_split}"
    print(f"RP_Pruning: {sub_dir}  kernel={kernel_name}")
 
    ports = pd.read_csv(input_path / sub_dir / input_file_name)
 
    if IsTree:
        depths = np.array([len(col.split('.')[1]) - 1 for col in ports.columns])
        adj_w  = 1.0 / np.sqrt(2.0 ** depths)
    else:
        adj_w = np.ones(ports.shape[1])
 
    adj_ports = ports * adj_w if IsTree else ports.copy()
 
    if state is None and not issubclass(kernel_cls, UniformKernel):
        raise ValueError(
            f"{kernel_cls.__name__} requires a state variable but state=None. "
            "Pass a pd.Series of monthly state values aligned with the portfolio returns."
        )
    bandwidths = kernel_cls.bandwidth_grid_from_state(state, n_train_valid, n=n_bandwidths)
    assert bandwidths is not None, "bandwidth_grid_from_state must return a list"
 
    kernel_output_path = Path(output_path) / kernel_name
    out_subdir         = kernel_output_path / sub_dir
    out_subdir.mkdir(parents=True, exist_ok=True)
 
    manifest = {
        'kernel':        kernel_name,
        'lambda0':       list(lambda0),
        'lambda2':       list(lambda2),
        'bandwidths':    [h if h is not None else 'uniform' for h in bandwidths],
        'n_train_valid': n_train_valid,
        'kmin':          kmin,
        'kmax':          kmax,
    }
    manifest_path = out_subdir / 'grid_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest written → {manifest_path}", flush=True)
 
    lasso_valid_full(
        adj_ports, lambda0, lambda2,
        str(kernel_output_path), sub_dir, adj_w,
        n_train_valid, cvN, runFullCV, kmin, kmax,
        RunParallel, ParallelN,
        kernel_cls=kernel_cls, bandwidths=bandwidths, state=state,
    )
 
 
if __name__ == '__main__':
    Mice_RP_Pruning(
        all_features         = ['LME', 'BEME', 'r12_2', 'OP', 'Investment',
                                 'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol'],
        n_features_per_split = 3,
        input_path           = Path('data/results/mice_rp_tree_portfolios'),
        input_file_name      = 'level_all_excess_combined.csv',
        output_path          = Path('data/results/grid_search/mice_rp_tree'),
        lambda0              = [0.5, 0.55, 0.6],
        lambda2              = [10**-7, 10**-7.25, 10**-7.5],
    )