"""
Step 3: Combine Trees — subtract rf, deduplicate, save combined matrix.

Matches R's combinetrees() in Step3_RmRf_Combine_Trees.R.

For a triplet (LME, feat1, feat2):
  - Reads all 81 tree ret CSVs (each T×31)
  - Renames columns with tree-id prefix: e.g. '1111.1', '1111.11', ...
  - Concatenates horizontally → T×2511
  - Deduplicates identical return series (columns) → T×~450
  - Subtracts monthly risk-free rate (stored as % in rf_factor.csv)
  - Applies same dedup mask to the min/max quantile tables
  - Saves 7 output files per triplet

Output files (in data/results/tree_portfolios/LME_feat1_feat2/):
  level_all_excess_combined.csv          (T × N_dedup)
  level_all_{feat}_min.csv  × 3 feats
  level_all_{feat}_max.csv  × 3 feats
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product


# ── Defaults (override via function arguments) ────────────────────────────────
TREE_DEPTH   = 4
Q_NUM        = 2          # binary splits → 3 feats → 3^4 = 81 trees
FACTOR_PATH  = Path('data/raw')
TREE_OUT     = Path('data/results/tree_portfolios')

# Node names matching step2 CNAMES (31 nodes for depth=4)
CNAMES = [
    '1',
    '11', '12',
    '111', '112', '121', '122',
    '1111', '1112', '1121', '1122', '1211', '1212', '1221', '1222',
    '11111', '11112', '11121', '11122', '11211', '11212', '11221', '11222',
    '12111', '12112', '12121', '12122', '12211', '12212', '12221', '12222',
]


# ── Risk-free rate ─────────────────────────────────────────────────────────────

def load_rf(factor_path: Path) -> np.ndarray:
    """
    Load rf_factor.csv — a single column of monthly risk-free rates in percent.
    Returns a 1-D array of length T with rates already divided by 100.

    R stores them as percentages so the subtraction is:
        port_ret[,i] = port_ret[,i] - rf / 100
    """
    rf_file = factor_path / 'rf_factor.csv'
    rf_df = pd.read_csv(rf_file, header=None)
    if rf_df.shape[1] != 1:
        raise ValueError("rf_factor.csv should have exactly one column")
    # Directly extract the first column as a numpy array
    rf = rf_df.iloc[:, 0].to_numpy(dtype=float) / 100.0
    return rf


# ── Column renaming ────────────────────────────────────────────────────────────

def make_col_names(tree_id: str, cnames: list[str]) -> list[str]:
    """
    Rename node columns with tree-id prefix, matching R's convention:
        colnames(port_ret0) = paste(tree_id, substring(colnames, 2), sep=".")
    R column names start with 'X' (e.g. 'X1', 'X11') and substring(...,2)
    strips the leading 'X', leaving '1', '11', etc.
    We produce: '1111.1', '1111.11', '1111.111', ...
    """
    return [f'{tree_id}.{c}' for c in cnames]


# ── Main combine function ──────────────────────────────────────────────────────

def combine_trees(
    feat1: str,
    feat2: str,
    tree_depth: int  = TREE_DEPTH,
    q_num: int       = Q_NUM,
    factor_path: Path = FACTOR_PATH,
    tree_out: Path   = TREE_OUT,
) -> None:
    """
    Combine all q_num^tree_depth trees for triplet (LME, feat1, feat2).

    Parameters
    ----------
    feat1, feat2    : second and third characteristics (LME is always first)
    tree_depth      : depth of each tree (default 4)
    q_num           : number of bins per split (default 2 → binary)
    factor_path     : directory containing rf_factor.csv
    tree_out        : root directory where step2 saved per-tree CSVs
    """
    feats   = ['LME', feat1, feat2]
    sub_dir = '_'.join(feats)
    data_dir = tree_out / sub_dir

    print(f"Combining trees for triplet: {feats}")

    # All tree ids: tuples of 1-indexed feature choices, length tree_depth
    # Matches R's expand.grid(rep(list(1:n_feats), tree_depth))
    n_feats    = len(feats)
    all_combos = r_expand_grid_order(n_feats, tree_depth) 
    n_trees    = len(all_combos)   # 3^4 = 81
    print(f"  Reading {n_trees} trees...")

    # ── 1. Load and horizontally concatenate all ret CSVs ─────────────────────
    ret_frames = []
    for combo in all_combos:
        tree_id  = ''.join(str(i) for i in combo)
        ret_path = data_dir / f'{tree_id}ret.csv'
        df       = pd.read_csv(ret_path)                     # T × 31
        df.columns = make_col_names(tree_id, CNAMES)
        ret_frames.append(df)

    port_ret = pd.concat(ret_frames, axis=1)                 # T × 2511
    print(f"  Before dedup: {port_ret.shape[1]} columns")

    # ── 2. Deduplicate identical return series ─────────────────────────────────
    # R: port_transpose = t(as.matrix(port_ret)); keep = !duplicated(port_transpose)
    # duplicated() on transposed matrix → finds duplicate ROWS of transpose
    #   = duplicate COLUMNS of original
    arr  = port_ret.to_numpy()                               # T × 2511
    _, keep_idx = np.unique(arr, axis=1, return_index=True)
    keep_idx    = np.sort(keep_idx)                          # preserve order

    # Boolean mask (same length as columns) for applying to min/max tables
    keep_mask = np.zeros(arr.shape[1], dtype=bool)
    keep_mask[keep_idx] = True

    port_dedup = port_ret.iloc[:, keep_idx]                  # T × ~450
    print(f"  After dedup:  {port_dedup.shape[1]} columns")

    # ── 3. Subtract risk-free rate ─────────────────────────────────────────────
    rf = load_rf(factor_path)                                # length T
    if len(rf) != len(port_dedup):
        raise ValueError(
            f"rf length {len(rf)} does not match portfolio length {len(port_dedup)}. "
            "Check that rf_factor.csv covers the same time period."
        )
    port_excess = port_dedup.subtract(rf, axis=0)

    # ── 4. Save excess return matrix ──────────────────────────────────────────
    out_file = data_dir / 'level_all_excess_combined.csv'
    port_excess.to_csv(out_file, index=False)
    print(f"  Saved: {out_file.name}  shape={port_excess.shape}")

    # ── 5. Load, dedup, and save min/max quantile tables ─────────────────────
    for feat in feats:
        for suffix in ('min', 'max'):
            frames = []
            for combo in all_combos:
                tree_id   = ''.join(str(i) for i in combo)
                file_path = data_dir / f'{tree_id}{feat}_{suffix}.csv'
                df        = pd.read_csv(file_path)           # T × 31
                df.columns = make_col_names(tree_id, CNAMES)
                frames.append(df)

            combined = pd.concat(frames, axis=1)             # T × 2511
            deduped  = combined.iloc[:, keep_idx]            # T × ~450
            out_name = data_dir / f'level_all_{feat}_{suffix}.csv'
            deduped.to_csv(out_name, index=False)
            print(f"  Saved: {out_name.name}  shape={deduped.shape}")

    print(f"Done: {sub_dir}\n")

def r_expand_grid_order(n_feats: int, depth: int) -> list[tuple]:
    """
    Return list of tuples of length depth, each element in 1..n_feats,
    in the order that R's expand.grid(rep(list(1:n_feats), depth)) produces.
    """
    total = n_feats ** depth
    combos = []
    for i in range(total):
        digits = []
        tmp = i
        for _ in range(depth):
            digits.append(tmp % n_feats)
            tmp //= n_feats
        combos.append(tuple(d + 1 for d in digits))
    return combos


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    combine_trees(
        feat1       = 'OP',
        feat2       = 'Investment',
        factor_path = FACTOR_PATH,
        tree_out    = TREE_OUT,
    )