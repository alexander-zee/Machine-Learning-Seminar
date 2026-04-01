"""
ap_pruning.py — AP-Pruning entry point for one triplet (LME, feat1, feat2).

Reads the filtered portfolio matrix, computes depth-based pre-weights,
pre-scales returns, then runs the full (lambda0, lambda2) grid search.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .lasso_valid_par_full import lasso_valid_full


def AP_Pruning(feat1, feat2, input_path, input_file_name, output_path,
               n_train_valid=360, cvN=3, runFullCV=False, kmin=5, kmax=50,
               RunParallel=False, ParallelN=10, IsTree=True,
               lambda0=None, lambda2=None):
    """
    Run AP-Pruning grid search for one triplet (LME, feat1, feat2).
    Input  : feat1/feat2 strings, input_path to filtered portfolio CSV,
             output_path for results, lambda grids, train/valid split params.
    Output : writes results_cv_{fold}_l0_{i}_l2_{j}.csv and
             results_full_l0_{i}_l2_{j}.csv to output_path/LME_feat1_feat2/.
             One CSV per (lambda0, lambda2) pair, each containing SR metrics
             and weights for all k in [kmin, kmax].
    """
    if lambda0 is None:
        lambda0 = [0.5, 0.55, 0.6]
    if lambda2 is None:
        lambda2 = [10**-7, 10**-7.25, 10**-7.5]

    subdir = '_'.join(['LME', feat1, feat2])
    print(f"AP_Pruning: {subdir}")

    ports = pd.read_csv(input_path / subdir / input_file_name)

    # Depth weight per portfolio: '1111.11111' -> depth = len('11111') - 1 = 4
    # adj_w = 1/sqrt(2^depth): depth-4 gets 0.25, root gets 1.0
    if IsTree:
        depths = np.array([len(col.split('.')[1]) - 1 for col in ports.columns])
        adj_w = 1.0 / np.sqrt(2.0 ** depths)
    else:
        adj_w = np.ones(ports.shape[1])

    # Pre-scale returns by depth weight so LARS implicitly penalises deeper nodes
    adj_ports = ports * adj_w if IsTree else ports.copy()

    lasso_valid_full(adj_ports, lambda0, lambda2, str(output_path), subdir, adj_w,
                     n_train_valid, cvN, runFullCV, kmin, kmax, RunParallel, ParallelN)


if __name__ == '__main__':
    AP_Pruning('OP', 'Investment',
               input_path=Path('data/results/tree_portfolios'),
               input_file_name='level_all_excess_combined_filtered.csv',
               output_path=Path('data/results/grid_search/tree'),
               lambda0=[0.5, 0.55, 0.6], lambda2=[10**-7, 10**-7.25, 10**-7.5])