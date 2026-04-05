"""
Step 3 (RP variant): Combine Trees — subtract rf, deduplicate, save combined matrix.
 
Identical logic to the original step3, but adapted for RP tree file naming:
  - Tree IDs are zero-padded integers ('00', '01', ..., '80')
    instead of feature-combo strings ('1111', '1112', ...)
  - Everything else (dedup, rf subtraction, output files) is unchanged,
    so downstream steps work without modification.
 
Output files (in data/results/rp_tree_portfolios/LME_feat1_feat2/):
  level_all_excess_combined.csv          (T × N_dedup)
  level_all_{feat}_min.csv  × 3 feats
  level_all_{feat}_max.csv  × 3 feats
"""
 
import numpy as np
import pandas as pd
from pathlib import Path
 
 
# ── Defaults ──────────────────────────────────────────────────────────────────
TREE_DEPTH   = 4
Q_NUM        = 2
N_TREES      = 81
FACTOR_PATH  = Path('data/raw')
TREE_OUT     = Path('data/results/rp_tree_portfolios')
 
_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
_SHIPPED_RF  = _REPO_ROOT / 'paper_data' / 'factor' / 'rf_factor.csv'
 
CNAMES = [
    '1',
    '11', '12',
    '111', '112', '121', '122',
    '1111', '1112', '1121', '1122', '1211', '1212', '1221', '1222',
    '11111', '11112', '11121', '11122', '11211', '11212', '11221', '11222',
    '12111', '12112', '12121', '12122', '12211', '12212', '12221', '12222',
]
 
 
# ── Risk-free rate (unchanged) ────────────────────────────────────────────────
 
def load_rf(factor_path: Path) -> np.ndarray:
    rf_file = factor_path / 'rf_factor.csv'
    if not rf_file.is_file():
        if _SHIPPED_RF.is_file():
            rf_file = _SHIPPED_RF
        else:
            raise FileNotFoundError(
                f"rf_factor.csv not found at {factor_path / 'rf_factor.csv'}."
            )
    rf_df = pd.read_csv(rf_file, header=None)
    if rf_df.shape[1] != 1:
        raise ValueError("rf_factor.csv should have exactly one column")
    return rf_df.iloc[:, 0].to_numpy(dtype=float) / 100.0
 
 
# ── Column renaming ───────────────────────────────────────────────────────────
 
def make_col_names(tree_id: str, cnames: list) -> list:
    """Prefix each node name with the tree id: e.g. '07.1', '07.11', ..."""
    return [f'{tree_id}.{c}' for c in cnames]
 
 
# ── Main combine function ─────────────────────────────────────────────────────
 
def combine_rp_trees(
    feat1: str,
    feat2: str,
    tree_depth: int   = TREE_DEPTH,
    q_num: int        = Q_NUM,
    n_trees: int      = N_TREES,
    factor_path: Path = FACTOR_PATH,
    tree_out: Path    = TREE_OUT,
) -> None:
    """
    Combine all n_trees RP trees for triplet (LME, feat1, feat2).
 
    Parameters
    ----------
    feat1, feat2  : second and third characteristics
    tree_depth    : depth of each tree
    q_num         : bins per split
    n_trees       : number of RP trees (must match step2)
    factor_path   : directory containing rf_factor.csv
    tree_out      : root directory where step2_rp saved per-tree CSVs
    """
    feats    = ['LME', feat1, feat2]
    sub_dir  = '_'.join(feats)
    data_dir = tree_out / sub_dir
 
    print(f"Combining {n_trees} RP trees for triplet: {feats}")
 
    id_width = len(str(n_trees - 1))
    tree_ids = [str(i).zfill(id_width) for i in range(n_trees)]
 
    # ── 1. Load and horizontally concatenate all ret CSVs ─────────────────────
    ret_frames = []
    for tree_id in tree_ids:
        ret_path = data_dir / f'{tree_id}ret.csv'
        df       = pd.read_csv(ret_path)                   # T × 31
        df.columns = make_col_names(tree_id, CNAMES)
        ret_frames.append(df)
 
    port_ret = pd.concat(ret_frames, axis=1)               # T × (81*31 = 2511)
    print(f"  Before dedup: {port_ret.shape[1]} columns")
 
    # ── 2. Deduplicate identical return series ─────────────────────────────────
    arr  = port_ret.to_numpy()
    _, keep_idx = np.unique(arr, axis=1, return_index=True)
    keep_idx    = np.sort(keep_idx)
 
    keep_mask = np.zeros(arr.shape[1], dtype=bool)
    keep_mask[keep_idx] = True
 
    port_dedup = port_ret.iloc[:, keep_idx]
    print(f"  After dedup:  {port_dedup.shape[1]} columns")
 
    # ── 3. Subtract risk-free rate ─────────────────────────────────────────────
    rf = load_rf(factor_path)
    if len(rf) != len(port_dedup):
        raise ValueError(
            f"rf length {len(rf)} does not match portfolio length {len(port_dedup)}."
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
            for tree_id in tree_ids:
                file_path = data_dir / f'{tree_id}{feat}_{suffix}.csv'
                df        = pd.read_csv(file_path)         # T × 31
                df.columns = make_col_names(tree_id, CNAMES)
                frames.append(df)
 
            combined = pd.concat(frames, axis=1)
            deduped  = combined.iloc[:, keep_idx]
            out_name = data_dir / f'level_all_{feat}_{suffix}.csv'
            deduped.to_csv(out_name, index=False)
            print(f"  Saved: {out_name.name}  shape={deduped.shape}")
 
    print(f"Done: {sub_dir}\n")
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    combine_rp_trees(
        feat1       = 'OP',
        feat2       = 'Investment',
        factor_path = FACTOR_PATH,
        tree_out    = TREE_OUT,
    )