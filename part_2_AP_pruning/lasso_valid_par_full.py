"""
lasso_valid_par_full.py — Grid search orchestration and routing.

Thin entry point that loops over bandwidths and CV folds, routing each
call to the appropriate implementation:
    - UniformKernel  → lasso_uniform.static_cv_helper (original code)
    - Non-uniform    → lasso_kernel_validation.kernel_cv_helper (per-month)

For non-uniform kernels, the full fit is NOT run during the grid search.
Call kernel_full_fit() separately after selecting the winning hyperparameters.

Functions
---------
lasso_valid_full : orchestrate CV folds across all (lambda0, lambda2, h)
lasso_cv_helper  : route one call to uniform or kernel path
"""

import os

import pandas as pd

from .kernels.uniform import UniformKernel
from .lasso_uniform import static_cv_helper
from .lasso_kernel_validation import kernel_cv_helper


def lasso_valid_full(ports, lambda0, lambda2, main_dir, sub_dir, adj_w,
                     n_train_valid=360, cvN=3, runFullCV=False,
                     kmin=5, kmax=50, RunParallel=False, ParallelN=10,
                     kernel_cls=None, bandwidths=None, state=None):
    """
    Orchestrate CV folds (and full fit for uniform) across all (lambda0, lambda2, h).

    For UniformKernel: runs both CV folds and full fit (cheap — one-shot).
    For non-uniform kernels: runs only CV folds. The full fit is deferred
        to kernel_full_fit() after the winning (l0*, l2*, h*) are selected.

    Parameters
    ----------
    kernel_cls  : kernel class. Defaults to UniformKernel.
    bandwidths  : list of h values. UniformKernel: [None].
                  GaussianKernel: list of floats from bandwidth_grid().
    state       : (T,) pd.Series of monthly state variable values.
                  None for UniformKernel.
    """
    if kernel_cls is None:
        kernel_cls = UniformKernel
    if bandwidths is None:
        bandwidths = [None]

    os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)

    ports_test = ports.iloc[n_train_valid:]
    state_test = None if state is None else state.iloc[n_train_valid:]

    n_valid    = n_train_valid // cvN
    fold_range = range(1, cvN + 1) if runFullCV else range(cvN, cvN + 1)

    is_uniform = issubclass(kernel_cls, UniformKernel)

    for h_idx, h in enumerate(bandwidths, start=1):

        if is_uniform:
            kernel = UniformKernel()
        else:
            kernel = kernel_cls(h=h)

        print(f"  bandwidth {h_idx}/{len(bandwidths)}: {kernel}", flush=True)

        for fold in fold_range:
            val_start, val_end = (fold - 1) * n_valid, fold * n_valid
            ports_valid = ports.iloc[val_start:val_end]
            ports_train = pd.concat([ports.iloc[:val_start],
                                      ports.iloc[val_end:n_train_valid]])

            state_train = (None if state is None else
                           pd.concat([state.iloc[:val_start],
                                       state.iloc[val_end:n_train_valid]]))
            state_valid = None if state is None else state.iloc[val_start:val_end]

            lasso_cv_helper(
                ports_train, ports_valid, ports_test, lambda0, lambda2,
                main_dir, sub_dir, adj_w, f'cv_{fold}', kmin, kmax,
                RunParallel, ParallelN,
                kernel=kernel, h_idx=h_idx,
                state_train=state_train, state_valid=state_valid,
                state_test=state_test)

        # Full fit: only for uniform (cheap). Kernel defers to kernel_full_fit.
        if is_uniform:
            state_train_full = None if state is None else state.iloc[:n_train_valid]
            lasso_cv_helper(
                ports.iloc[:n_train_valid], None, ports_test, lambda0, lambda2,
                main_dir, sub_dir, adj_w, 'full', kmin, kmax,
                RunParallel, ParallelN,
                kernel=kernel, h_idx=h_idx,
                state_train=state_train_full, state_valid=None,
                state_test=state_test)

    if not is_uniform:
        print(f"  Grid search done. Run kernel_full_fit() for the winning combo.",
              flush=True)


def lasso_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
                    main_dir, sub_dir, adj_w, cv_name, kmin=5, kmax=50,
                    RunParallel=False, ParallelN=10,
                    kernel=None, h_idx=1,
                    state_train=None, state_valid=None, state_test=None):
    """
    Route one CV-fold call to the uniform or kernel path.

    ports_test / state_test are only used by the uniform path (static_cv_helper
    runs the full fit inline).  The kernel path ignores them — full fit is
    handled separately by kernel_full_fit() after hyperparameter selection.
    """
    if kernel is None:
        kernel = UniformKernel()

    if isinstance(kernel, UniformKernel):
        static_cv_helper(
            ports_train, ports_valid, ports_test, lambda0, lambda2,
            main_dir, sub_dir, adj_w, cv_name, kmin, kmax,
            RunParallel, ParallelN, h_idx=h_idx)
    else:
        # Kernel path: validation fold only — no test data needed here.
        kernel_cv_helper(
            ports_train, ports_valid, lambda0, lambda2,
            main_dir, sub_dir, adj_w, cv_name, kmin, kmax,
            kernel=kernel, h_idx=h_idx,
            state_train=state_train, state_valid=state_valid)