import os
from pathlib import Path

import pandas as pd

from lambda_grids import get_lambda_grids
from lasso_valid_par_full import lasso_valid_full


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
