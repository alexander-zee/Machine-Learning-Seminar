import os
from pathlib import Path

import numpy as np
import pandas as pd

from lambda_grids import get_lambda_grids
from lasso_valid_par_full import lasso_valid_full


def _tv_recency_kernel_weights(n_rows: int, halflife_months: float) -> pd.DataFrame:
    """
    Upweight recent months for kernel-weighted train moments (``lasso_valid_full``).

    Weight ∝ exp((t - (T-1)) / h) with h = halflife; normalized per CV fold inside LASSO.
    Set ``TV_KERNEL_HALFLIFE_MONTHS`` (default 72) in the environment.
    """
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    t = np.arange(n_rows, dtype=float)
    h = max(float(halflife_months), 1e-6)
    w = np.exp((t - (n_rows - 1)) / h)
    return pd.DataFrame({"Kernel_Weight": w})


def _resolve_feat_idx(feats_list, f):
    if isinstance(f, int):
        return f
    if isinstance(f, str):
        return feats_list.index(f)
    raise TypeError("feat index must be int or str matching feats_list")


def AP_Pruning(
    feats_list,
    feat1,
    feat2,
    input_path,
    input_file_name,
    output_path,
    n_train_valid=360,
    cvN=3,
    runFullCV=False,
    kmin=5,
    kmax=50,
    RunParallel=False,
    ParallelN=10,
    IsTree=True,
    lambda0=None,
    lambda2=None,
    weights_dict_df=None,
):
    if lambda0 is None or lambda2 is None:
        g0, g2 = get_lambda_grids()
        lambda0 = g0 if lambda0 is None else lambda0
        lambda2 = g2 if lambda2 is None else lambda2

    i1 = _resolve_feat_idx(feats_list, feat1)
    i2 = _resolve_feat_idx(feats_list, feat2)
    feats_chosen = ["LME", feats_list[i1], feats_list[i2]]
    sub_dir = "_".join(feats_chosen)

    csv_path = Path(input_path) / sub_dir / input_file_name.lstrip("/\\")
    ports = pd.read_csv(csv_path)
    
    # --- FIX: Remove the date column before math begins ---
    if 'yyyymm' in ports.columns:
        ports = ports.drop(columns=['yyyymm'])
    # ------------------------------------------------------
    
    if IsTree:
        # Parse the tree sorting weights from column names
        adj_w = [float(col.split('_')[-1]) for col in ports.columns]
    else:
        adj_w = [1.0] * ports.shape[1]
    
    lasso_valid_full(
        ports,
        lambda0,
        lambda2,
        output_path,
        sub_dir,
        adj_w,
        n_train_valid,
        cvN,
        runFullCV,
        kmin,
        kmax,
        RunParallel,
        ParallelN,
        weights_dict_df,
    )


def AP_Pruning_clusters(
    input_csv,
    output_path,
    sub_dir="Ward_clusters_10",
    n_train_valid=360,
    cvN=3,
    runFullCV=False,
    kmin=3,
    kmax=10,
    RunParallel=False,
    ParallelN=10,
    lambda0=None,
    lambda2=None,
    weights_dict_df=None,
):
    """
    LASSO / AP-style pruning on Ward cluster portfolio returns (no tree depth weights).
    Expects CSV with dates in the first column (or index) and numeric cluster return columns.
    """
    if lambda0 is None or lambda2 is None:
        g0, g2 = get_lambda_grids()
        lambda0 = g0 if lambda0 is None else lambda0
        lambda2 = g2 if lambda2 is None else lambda2

    path = Path(input_csv)
    ports = pd.read_csv(path, index_col=0)
    ports = ports.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    ports = ports.dropna(axis=0, how="any")
    p = ports.shape[1]
    kmax_eff = min(kmax, p)
    kmin_eff = min(kmin, kmax_eff)
    if kmin_eff < 1:
        raise ValueError("No numeric return columns after cleaning cluster CSV.")

    adj_w = [1.0] * p
    out = str(Path(output_path))
    os.makedirs(Path(out) / sub_dir, exist_ok=True)

    lasso_valid_full(
        ports,
        lambda0,
        lambda2,
        out,
        sub_dir,
        adj_w,
        n_train_valid,
        cvN,
        runFullCV,
        kmin_eff,
        kmax_eff,
        RunParallel,
        ParallelN,
        weights_dict_df,
    )


def RP_Pruning(
    feats_list,
    feat1,
    feat2,
    input_path,
    output_path,
    n_train_valid=360,
    cvN=3,
    runFullCV=False,
    kmin=5,
    kmax=50,
    RunParallel=False,
    ParallelN=10,
    lambda0=None,
    lambda2=None,
    weights_dict_df=None,
):
    """
    AP pruning on **random-projection** trees (``origin/rp-trees``).

    Reads ``level_all_excess_combined.csv`` (no depth-weight column names). Writes to
    ``RP_LME_*_*`` under ``output_path`` so ordinary AP-trees stay in ``LME_*_*``.
    """
    if lambda0 is None or lambda2 is None:
        g0, g2 = get_lambda_grids()
        lambda0 = g0 if lambda0 is None else lambda0
        lambda2 = g2 if lambda2 is None else lambda2

    i1 = _resolve_feat_idx(feats_list, feat1)
    i2 = _resolve_feat_idx(feats_list, feat2)
    feats_chosen = ["LME", feats_list[i1], feats_list[i2]]
    sub_dir = "RP_" + "_".join(feats_chosen)

    csv_path = Path(input_path) / "_".join(feats_chosen) / "level_all_excess_combined.csv"
    ports = pd.read_csv(csv_path)

    if "yyyymm" in ports.columns:
        ports = ports.drop(columns=["yyyymm"])

    adj_w = [1.0] * ports.shape[1]

    lasso_valid_full(
        ports,
        lambda0,
        lambda2,
        output_path,
        sub_dir,
        adj_w,
        n_train_valid,
        cvN,
        runFullCV,
        kmin,
        kmax,
        RunParallel,
        ParallelN,
        weights_dict_df,
    )


def TV_Pruning(
    feats_list,
    feat1,
    feat2,
    input_path,
    input_file_name,
    output_path,
    n_train_valid=360,
    cvN=3,
    runFullCV=False,
    kmin=5,
    kmax=50,
    RunParallel=False,
    ParallelN=10,
    lambda0=None,
    lambda2=None,
    weights_dict_df=None,
    kernel_halflife_months: float | None = None,
):
    """
    **Time-varying moments** variant of AP pruning on **standard** characteristic trees.

    Uses the same filtered return panel as ``AP_Pruning`` (tree depth weights ``adj_w``) but
    passes row weights into ``lasso_valid_full`` so train/valid moments are **kernel-weighted**
    toward **recent** observations (recency kernel). Outputs live under ``TV_LME_*_*``.

    Halflife (months): env ``TV_KERNEL_HALFLIFE_MONTHS`` or ``kernel_halflife_months`` (default 72).
    For state-variable kernels, pass a custom ``weights_dict_df`` (column ``Kernel_Weight``).
    """
    if lambda0 is None or lambda2 is None:
        g0, g2 = get_lambda_grids()
        lambda0 = g0 if lambda0 is None else lambda0
        lambda2 = g2 if lambda2 is None else lambda2

    i1 = _resolve_feat_idx(feats_list, feat1)
    i2 = _resolve_feat_idx(feats_list, feat2)
    feats_chosen = ["LME", feats_list[i1], feats_list[i2]]
    sub_dir = "TV_" + "_".join(feats_chosen)

    csv_path = Path(input_path) / "_".join(feats_chosen) / input_file_name.lstrip("/\\")
    ports = pd.read_csv(csv_path)

    if "yyyymm" in ports.columns:
        ports = ports.drop(columns=["yyyymm"])

    adj_w = [float(col.split("_")[-1]) for col in ports.columns]

    if weights_dict_df is None:
        h = kernel_halflife_months
        if h is None:
            h = float(os.environ.get("TV_KERNEL_HALFLIFE_MONTHS", "72"))
        weights_dict_df = _tv_recency_kernel_weights(len(ports), h)

    lasso_valid_full(
        ports,
        lambda0,
        lambda2,
        output_path,
        sub_dir,
        adj_w,
        n_train_valid,
        cvN,
        runFullCV,
        kmin,
        kmax,
        RunParallel,
        ParallelN,
        weights_dict_df,
    )
