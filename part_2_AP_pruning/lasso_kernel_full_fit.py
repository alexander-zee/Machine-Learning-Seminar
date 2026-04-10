"""
lasso_kernel_full_fit.py — Final test evaluation for a single (k, l0*, l2*, h*) combo.

Called after pick_best_lambda_kernel selects the winning hyperparameters for a
specific target portfolio count k.  All test-period logic lives here — nothing
is delegated back to kernel_cv_helper, which handles validation folds only.

Workflow
--------
1.  pick_best_lambda_kernel reads the validation CSVs and returns:
        (i_best, j_best, h_best), lambda0_star, lambda2_star, kernel_with_h

2.  kernel_full_fit is called once per target k:
    - Trains on the full n_train_valid months (with kernel weights per test month)
    - For each test month, runs one_month_lars and extracts only the k-specific weights
    - Saves two CSVs to output_dir, both named with the target k:

        full_fit_summary_k{k}.csv
            One row: k, test_SR, lambda0, lambda2, h, kernel

        full_fit_detail_k{k}.csv
            T_test rows × (1 + N) columns:
                excess_return  — SDF return that month (portfolio weights · returns)
                <port_col_1>   — SDF weight for portfolio 1
                ...
                <port_col_N>   — SDF weight for portfolio N

    Weights stored are the actual portfolio weights w = b / adj_w (undone depth
    scaling), consistent with: excess_return = returns_row @ w.

Notes
-----
- The full covariance / mean is re-estimated locally for every test month via
  the kernel — there is no single global beta vector to summarise.  The detail
  CSV is therefore the primary output; the summary CSV contains the SR and the
  hyperparameters used.
- If the LARS path does not produce a solution at exactly k for some months,
  those months are skipped with a warning.  The SR is computed over the months
  that do have a solution.
"""

import os
import warnings

import numpy as np
import pandas as pd

from .lasso_core import one_month_lars


def kernel_full_fit(
    ports,
    k_target,
    lambda0_star,
    lambda2_star,
    kernel,
    state,
    adj_w,
    output_dir,
    *,
    n_train_valid=360,
    kmin=5,
    kmax=50,
    kernel_name=None,
):
    """
    Test evaluation for the winning (lambda0*, lambda2*, h*) at a given k.

    Parameters
    ----------
    ports         : DataFrame (T_total, N) of pre-scaled portfolio excess returns
    k_target      : int — the specific number of portfolios to evaluate
    lambda0_star  : float — winning l0 hyperparameter
    lambda2_star  : float — winning l2 hyperparameter
    kernel        : kernel instance already initialised with bandwidth h*
    state         : pd.Series (T_total,) — monthly state variable
    adj_w         : (N,) — depth-based pre-weights (from AP-Tree depths)
    output_dir    : str or Path — directory for output CSVs
    n_train_valid : int — number of months in the training+validation window
    kmin, kmax    : int — LARS path bounds (must bracket k_target)
    kernel_name   : str or None — label written to the summary CSV
                    (e.g. 'gaussian', 'exponential').  Inferred from repr if None.

    Output files (in output_dir)
    ----------------------------
    full_fit_summary_k{k_target}.csv
        Columns: k, test_SR, lambda0, lambda2, h, kernel

    full_fit_detail_k{k_target}.csv
        Columns: excess_return, <port_col_1>, ..., <port_col_N>
        One row per test month (months without a k_target solution are skipped).

    Returns
    -------
    dict with keys:
        'test_SR'       : float — monthly Sharpe ratio over the test period
        'excess_returns': list of floats — per-month SDF returns
        'weights'       : list of (N,) arrays — per-month portfolio weight vectors
        'months_used'   : int — number of test months that produced a k solution
        'months_total'  : int — total number of test months
    """
    os.makedirs(output_dir, exist_ok=True)

    if not (kmin <= k_target <= kmax):
        raise ValueError(
            f"k_target={k_target} is outside the LARS path bounds "
            f"[kmin={kmin}, kmax={kmax}]."
        )

    # ------------------------------------------------------------------ #
    # Split into train and test
    # ------------------------------------------------------------------ #
    ports_train_arr = ports.iloc[:n_train_valid].values
    ports_test      = ports.iloc[n_train_valid:]
    port_cols       = ports.columns.tolist()

    state_train_arr = np.asarray(state.iloc[:n_train_valid])
    state_test_arr  = np.asarray(state.iloc[n_train_valid:])

    T_test         = len(ports_test)
    ports_test_arr = ports_test.values

    # LARS is run for a single (l0, l2) combo — wrap in lists so one_month_lars
    # iterates them, then we index back to (0, 0, k_target).
    lambda0_list = [lambda0_star]
    lambda2_list = [lambda2_star]

    # ------------------------------------------------------------------ #
    # Per-month test loop
    # ------------------------------------------------------------------ #
    excess_returns = []
    weights_list   = []
    skipped        = 0

    print(
        f"  kernel_full_fit: k={k_target}, "
        f"l0={lambda0_star}, l2={lambda2_star:.2e}, "
        f"kernel={kernel!r}",
        flush=True,
    )
    print(f"  Running {T_test} test months...", flush=True)

    for t in range(T_test):
        if (t + 1) % 20 == 0 or t == 0:
            print(f"    test month {t+1}/{T_test}", flush=True)

        s_current = float(state_test_arr[t])

        # one_month_lars returns {(i, j, k): sdf_weights}
        # With single-element lambda lists, the only combo is (0, 0, *)
        monthly = one_month_lars(
            ports_train_arr, state_train_arr, s_current,
            kernel, lambda0_list, lambda2_list, adj_w,
            kmin, kmax,
        )

        key = (0, 0, k_target)
        if key not in monthly:
            skipped += 1
            continue

        sdf_w   = monthly[key]                         # (N,)  portfolio weights
        sdf_ret = float(ports_test_arr[t] @ sdf_w)    # scalar excess return

        excess_returns.append(sdf_ret)
        weights_list.append(sdf_w)

    months_used = len(excess_returns)
    print(
        f"  Done — {months_used}/{T_test} months used "
        f"({skipped} skipped: LARS path lacked k={k_target})",
        flush=True,
    )

    if months_used == 0:
        warnings.warn(
            f"kernel_full_fit: no test months produced k={k_target}. "
            "Check kmin/kmax bounds and hyperparameter selection.",
            RuntimeWarning,
        )
        return {
            'test_SR':        np.nan,
            'excess_returns': [],
            'weights':        [],
            'months_used':    0,
            'months_total':   T_test,
        }

    # ------------------------------------------------------------------ #
    # Sharpe ratio
    # ------------------------------------------------------------------ #
    r_arr    = np.array(excess_returns)
    mean_ret = float(r_arr.mean())
    std_ret  = float(r_arr.std(ddof=1))
    test_SR  = float(mean_ret / std_ret) if std_ret > 0 else np.nan

    # ------------------------------------------------------------------ #
    # Derive bandwidth value for the summary (None for uniform kernel)
    # ------------------------------------------------------------------ #
    h_val = getattr(kernel, 'h', None)

    # Human-readable kernel label
    if kernel_name is None:
        kernel_name = type(kernel).__name__

    # ------------------------------------------------------------------ #
    # Save summary CSV  (1 row, fully labelled)
    # ------------------------------------------------------------------ #
    summary_path = os.path.join(output_dir, f'full_fit_summary_k{k_target}.csv')
    pd.DataFrame([{
        'k':        k_target,
        'test_SR':  test_SR,
        'mean_ret': mean_ret,
        'std_ret':  std_ret,
        'lambda0':  lambda0_star,
        'lambda2':  lambda2_star,
        'h':        h_val,
        'kernel':   kernel_name,
    }]).to_csv(summary_path, index=False)
    print(f"  Summary saved → {summary_path}  (test_SR={test_SR:.4f})", flush=True)

    # ------------------------------------------------------------------ #
    # Save detail CSV  (T_used rows × [excess_return + N weight columns])
    # ------------------------------------------------------------------ #
    weights_arr = np.stack(weights_list, axis=0)          # (T_used, N)
    detail_df   = pd.DataFrame(weights_arr, columns=port_cols)
    detail_df.insert(0, 'excess_return', excess_returns)

    detail_path = os.path.join(output_dir, f'full_fit_detail_k{k_target}.csv')
    detail_df.to_csv(detail_path, index=False)
    print(f"  Detail saved  → {detail_path}  ({months_used} rows, {len(port_cols)+1} cols)",
          flush=True)

    return {
        'test_SR':        test_SR,
        'excess_returns': excess_returns,
        'weights':        weights_list,
        'months_used':    months_used,
        'months_total':   T_test,
    }