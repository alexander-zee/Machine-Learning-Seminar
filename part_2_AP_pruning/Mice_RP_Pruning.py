"""
mice_rp_pruning.py — RP-Pruning for randomly generated triplets

Mirrors AP_Pruning exactly, with one change: column names follow the RP tree
convention  '<tree_id>.<node>'  (e.g. '07.1111') rather than the AP convention
'<feat_combo>.<node>' (e.g. '1111.11111').  Depth is therefore read from the
node part (after the dot) in the same way.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .lasso_valid_par_full import lasso_valid_full


def Mice_RP_Pruning(allfeatures, input_path, input_file_name, output_path,
               n_train_valid=360, cvN=3, runFullCV=False, kmin=5, kmax=50,
               RunParallel=False, ParallelN=10, IsTree=True,
               lambda0=None, lambda2=None):
    """
    Run RP-Pruning grid search for one triplet (LME, feat1, feat2).

    Identical to AP_Pruning except for how node depth is extracted from column
    names.  RP tree columns look like '<tree_id>.<node>', e.g. '07.1' (root,
    depth 0) or '07.1122' (depth 3).  AP tree columns look like
    '<feat_combo>.<node>', e.g. '1111.11111' (depth 4).

    Depth extraction
    ----------------
    AP:  col = '1111.11111'  →  node = '11111'  →  depth = len('11111') - 1 = 4
    RP:  col = '07.1111'     →  node = '1111'   →  depth = len('1111')  - 1 = 3

    Both cases: depth = len(col.split('.')[1]) - 1   ← same formula, works for both.

    Parameters
    ----------
    feat1, feat2      : second and third characteristics (first is always LME)
    input_path        : Path to the directory that contains LME_feat1_feat2/
    input_file_name   : CSV filename inside that subdirectory, e.g.
                        'level_all_excess_combined.csv'
    output_path       : root Path for results; one subdir per triplet is created
    n_train_valid     : total train+validation window length (default 360 months)
    cvN               : number of cross-validation folds (default 3)
    runFullCV         : if True, also run on the full training sample
    kmin, kmax        : range of portfolio-count targets for LARS
    RunParallel       : use joblib parallelism across (lambda0, lambda2) grid
    ParallelN         : number of parallel workers
    IsTree            : if False, skip depth weighting (flat weight = 1)
    lambda0           : list of l0 penalty values  (default [0.50, 0.55, 0.60])
    lambda2           : list of l2 penalty values  (default [1e-7, 10**-7.25, 1e-7.5])

    Output
    ------
    Writes results_cv_{fold}_l0_{i}_l2_{j}.csv  and
           results_full_l0_{i}_l2_{j}.csv
    to  output_path / 'LME_{feat1}_{feat2}' /
    One CSV per (lambda0, lambda2) pair, each containing SR metrics and
    portfolio weights for every k in [kmin, kmax].
    """
    if lambda0 is None:
        lambda0 = [0.5, 0.55, 0.6]
    if lambda2 is None:
        lambda2 = [10**-7, 10**-7.25, 10**-7.5]

    subdir = '_'.join(allfeatures)
    print(f"RP_Pruning: {subdir}")

    ports = pd.read_csv(input_path / subdir / input_file_name)

    # Depth weight per portfolio.
    # Column format: '<tree_id>.<node>'  e.g. '07.1111'
    #   node part  = col.split('.')[1]   e.g. '1111'
    #   depth      = len(node) - 1       e.g. 3
    #   adj_w      = 1 / sqrt(2^depth)
    # This is identical to the AP formula — only the column *format* differs;
    # the depth arithmetic is the same once the node string is isolated.
    if IsTree:
        depths = np.array([len(col.split('.')[1]) - 1 for col in ports.columns])
        adj_w  = 1.0 / np.sqrt(2.0 ** depths)
    else:
        adj_w = np.ones(ports.shape[1])

    # Pre-scale returns by depth weight so LARS implicitly penalises deeper nodes
    adj_ports = ports * adj_w if IsTree else ports.copy()

    lasso_valid_full(adj_ports, lambda0, lambda2, str(output_path), subdir, adj_w,
                     n_train_valid, cvN, runFullCV, kmin, kmax, RunParallel, ParallelN)


if __name__ == '__main__':
    Mice_RP_Pruning( allfeatures=['LME', 'BEME', 'r12_2', 'OP', 'Investment', 'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol'],
               input_path=Path('data/results/mice_rp_tree_portfolios'),
               input_file_name='level_all_excess_combined.csv',
               output_path=Path('data/results/grid_search/mice_rp_tree'),
               lambda0=[0.5, 0.55, 0.6],
               lambda2=[10**-7, 10**-7.25, 10**-7.5])