"""
Step 3 (RP variant): Combine Trees — subtract rf, deduplicate, save combined matrix.
 
Adapted for the all-features RP tree setup from step2_RP_tree_portfolios.py:
  - Subdirectory is named after all features joined by '_'
    (e.g. LME_..._LTurnover_IdioVol)
  - Tree IDs are zero-padded integers ('00', '01', ..., '80')
  - Min/max tables are saved for ALL features, not just three
  - Everything else (dedup, rf subtraction, output files) is unchanged
 
Output files (in output_path / '_'.join(all_features) /):
  level_all_excess_combined.csv         (T × N_dedup)
  level_all_{feat}_min.csv  × n_feats
  level_all_{feat}_max.csv  × n_feats
"""
 
import numpy as np
import pandas as pd
from pathlib import Path
 
 
# ── Defaults ──────────────────────────────────────────────────────────────────
TREE_DEPTH  = 4
Q_NUM       = 2
N_TREES     = 81
FACTOR_PATH = Path('data/raw')
TREE_OUT    = Path('data/results/mice_rp_tree_portfolios')
 
ALL_FEATURES = [
    'LME', 'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]
N_FEATURES_PER_SPLIT = 3
 
_REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
_SHIPPED_RF = _REPO_ROOT / 'paper_data' / 'factor' / 'rf_factor.csv'
 
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
    """
    Load rf_factor.csv — single column of monthly rates in percent.
    Returns 1-D array of length T with rates divided by 100.
    """
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
 
 
# ── Column renaming (unchanged) ───────────────────────────────────────────────
 
def make_col_names(tree_id: str, cnames: list) -> list:
    """Prefix each node name with the tree id: e.g. '07.1', '07.11', ..."""
    return [f'{tree_id}.{c}' for c in cnames]
 
 
# ── Main combine function ─────────────────────────────────────────────────────
 
def combine_mice_rp_trees(
    all_features: list  = ALL_FEATURES,
    n_trees: int        = N_TREES,
    factor_path: Path   = FACTOR_PATH,
    tree_out: Path      = TREE_OUT,
    n_features_per_split: int = N_FEATURES_PER_SPLIT
) -> None:
    """
    Combine all n_trees RP trees built over all_features.
 
    Parameters
    ----------
    all_features : list of all characteristic names used in step2
                   (must match exactly — determines the subdirectory name)
    n_trees      : number of RP trees (must match step2)
    factor_path  : directory containing rf_factor.csv
    tree_out     : root directory where step2 saved per-tree CSVs
 
    Output
    ------
    Writes to tree_out / '_'.join(all_features) /:
        level_all_excess_combined.csv         (T × N_dedup)
        level_all_{feat}_min.csv              (T × N_dedup)  — one per feature
        level_all_{feat}_max.csv              (T × N_dedup)  — one per feature
    """
    sub_dir = f"{'_'.join(all_features)}__nf{n_features_per_split}"
    data_dir = tree_out / sub_dir
 
    print(f"Combining {n_trees} RP trees")
    print(f"  Features ({len(all_features)}): {all_features}")
    print(f"  Reading from: {data_dir}")
 
    id_width = len(str(n_trees - 1))
    tree_ids = [str(i).zfill(id_width) for i in range(n_trees)]
 
    # ── 1. Load and horizontally concatenate all ret CSVs ─────────────────────
    ret_frames = []
    for tree_id in tree_ids:
        ret_path = data_dir / f'{tree_id}ret.csv'
        df       = pd.read_csv(ret_path)            # T × 31
        df.columns = make_col_names(tree_id, CNAMES)
        ret_frames.append(df)
 
    port_ret = pd.concat(ret_frames, axis=1)        # T × (N_TREES * 31)
    print(f"  Before dedup: {port_ret.shape[1]} columns")
 
    # ── 2. Deduplicate identical return series ─────────────────────────────────
    arr = port_ret.to_numpy()
    _, keep_idx = np.unique(arr, axis=1, return_index=True)
    keep_idx    = np.sort(keep_idx)
 
    port_dedup = port_ret.iloc[:, keep_idx]
    print(f"  After dedup:  {port_dedup.shape[1]} columns")
 
    # ── 3. Subtract risk-free rate ─────────────────────────────────────────────
    rf = load_rf(factor_path)
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
 
    # ── 5. Load, dedup, and save min/max tables for ALL features ──────────────
    for feat in all_features:
        for suffix in ('min', 'max'):
            frames = []
            for tree_id in tree_ids:
                file_path = data_dir / f'{tree_id}{feat}_{suffix}.csv'
                df        = pd.read_csv(file_path)  # T × 31
                df.columns = make_col_names(tree_id, CNAMES)
                frames.append(df)
 
            combined = pd.concat(frames, axis=1)
            deduped  = combined.iloc[:, keep_idx]
            out_name = data_dir / f'level_all_{feat}_{suffix}.csv'
            deduped.to_csv(out_name, index=False)
            print(f"  Saved: {out_name.name}  shape={deduped.shape}")
 
    print(f"\nDone: {sub_dir}")
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    combine_mice_rp_trees(
        all_features = ALL_FEATURES,
        factor_path  = FACTOR_PATH,
        tree_out     = TREE_OUT,
        n_features_per_split= N_FEATURES_PER_SPLIT
    )