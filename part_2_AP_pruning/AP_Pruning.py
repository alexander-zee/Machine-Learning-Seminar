"""
AP_Pruning.py — AP-Pruning entry point for one triplet (LME, feat1, feat2).

Reads the filtered portfolio matrix, computes depth-based pre-weights,
pre-scales returns, then runs the full (lambda0, lambda2, h) grid search.

Changes vs original:
    - Accepts kernel_cls and state (no bandwidths — derived from kernel)
    - Calls kernel_cls.bandwidth_grid() to get the h grid internally
    - Output folder derived from kernel class name:
          output_path/uniform/LME_OP_Investment/
          output_path/gaussian/LME_OP_Investment/
    - Passes lambda0, lambda2, bandwidths as vectors to lasso_valid_full
      so the grid search is fully symmetric across all three
    - Everything else (depth weights, adj_ports) unchanged
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .lasso_valid_par_full import lasso_valid_full
from .kernels.uniform import UniformKernel


def AP_Pruning(feat1, feat2, input_path, input_file_name, output_path,
               n_train_valid=360, cvN=3, runFullCV=False, kmin=5, kmax=50,
               RunParallel=False, ParallelN=10, IsTree=True,
               lambda0=None, lambda2=None,
               kernel_cls=None, state=None):
    """
    Run AP-Pruning grid search for one triplet across all (lambda0, lambda2, h).

    Parameters
    ----------
    kernel_cls : kernel class (not instance), e.g. GaussianKernel.
                 Defaults to UniformKernel (original behavior).
                 The bandwidth grid is derived from kernel_cls.bandwidth_grid().
    state      : pd.Series (T,) of monthly state variable values aligned with
                 portfolio return rows. None for UniformKernel.

    Output
    ------
    output_path / {kernel_name} / LME_{feat1}_{feat2} /
        results_cv_{fold}_l0_{i}_l2_{j}_h_{h_idx}.csv
        results_full_l0_{i}_l2_{j}_h_{h_idx}.csv
    One CSV per (lambda0, lambda2, h) combination, all k in [kmin, kmax].
    """
    if lambda0 is None:
        lambda0 = [0.5, 0.55, 0.6]
    if lambda2 is None:
        lambda2 = [10**-7, 10**-7.25, 10**-7.5]
    if kernel_cls is None:
        kernel_cls = UniformKernel

    # Output subfolder derived from kernel name
    # e.g. UniformKernel -> "uniform", GaussianKernel -> "gaussian"
    kernel_name = kernel_cls.__name__.lower().replace('kernel', '')

    subdir = '_'.join(['LME', feat1, feat2])
    print(f"AP_Pruning: {subdir}  kernel={kernel_name}")

    ports = pd.read_csv(input_path / subdir / input_file_name)

    if IsTree:
        depths = np.array([len(col.split('.')[1]) - 1 for col in ports.columns])
        adj_w  = 1.0 / np.sqrt(2.0 ** depths)
    else:
        adj_w = np.ones(ports.shape[1])

    adj_ports = ports * adj_w if IsTree else ports.copy()

    # Derive bandwidth grid from the kernel class itself — hardcoded per kernel
    # UniformKernel.bandwidth_grid()         -> [None]  (one run)
    # GaussianKernel.bandwidth_grid(sigma_s) -> [0.05*s, 0.1*s, ...]
    if issubclass(kernel_cls, UniformKernel):
        bandwidths = kernel_cls.bandwidth_grid()
    else:
        if state is None:
            raise ValueError(
                f"{kernel_cls.__name__} requires a state variable but state=None. "
                "Pass a pd.Series of monthly state values aligned with the portfolio returns."
            )
        sigma_s    = state.iloc[:n_train_valid].std()
        bandwidths = kernel_cls.bandwidth_grid(sigma_s)

    kernel_output_path = Path(output_path) / kernel_name

    lasso_valid_full(
        adj_ports, lambda0, lambda2,
        str(kernel_output_path), subdir, adj_w,
        n_train_valid, cvN, runFullCV, kmin, kmax,
        RunParallel, ParallelN,
        kernel_cls=kernel_cls, bandwidths=bandwidths, state=state,
    )


if __name__ == '__main__':
    # Baseline — identical to original behavior
    AP_Pruning('OP', 'Investment',
               input_path=Path('data/results/tree_portfolios'),
               input_file_name='level_all_excess_combined_filtered.csv',
               output_path=Path('data/results/grid_search/tree'),
               lambda0=[0.5, 0.55, 0.6],
               lambda2=[10**-7, 10**-7.25, 10**-7.5],
               kernel_cls=UniformKernel)
