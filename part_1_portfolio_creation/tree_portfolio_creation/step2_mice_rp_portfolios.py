"""
Step 2 (RP variant): Random Projection Tree Portfolios — all features, random
selection per split.
 
Instead of a fixed triplet (LME, feat1, feat2), this module uses ALL available
characteristics. At each depth level of each tree, N_FEATURES_PER_SPLIT features
are drawn at random and a unit vector in that subspace defines the split direction.
 
Output structure (mirrors step2 so step3 and downstream work unchanged):
    output_path/LME_BEME_r12_2_OP_Investment_ST_Rev_LT_Rev_AC_LTurnover_IdioVol/
        {file_id}ret.csv               (T × 31)
        {file_id}{feat}_min.csv        (T × 31)  — one file per feature
        {file_id}{feat}_max.csv        (T × 31)  — one file per feature
        projection_metadata.json       — feature selection + vectors per tree/depth
"""
 
import json
import numpy as np
import pandas as pd
from pathlib import Path
 
 
# ── Config ────────────────────────────────────────────────────────────────────
PANEL_PATH = Path('data/prepared/panel_clustering_mice.parquet')
OUTPUT_PATH = Path('data/results/mice_rp_tree_portfolios')
TREE_DEPTH = 4
Q_NUM = 2                # binary median splits
Y_MIN = 1964
Y_MAX = 2016
N_TREES = 81
GLOBAL_SEED = 42
 
ALL_FEATURES = [
    'LME', 'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]
N_FEATURES_PER_SPLIT = 3   # features randomly selected per depth level per tree
 
# Node names for depth=4 — identical encoding to original step2
CNAMES = [
    '1',
    '11', '12',
    '111', '112', '121', '122',
    '1111', '1112', '1121', '1122', '1211', '1212', '1221', '1222',
    '11111', '11112', '11121', '11122', '11211', '11212', '11221', '11222',
    '12111', '12112', '12121', '12122', '12211', '12212', '12221', '12222',
]
assert len(CNAMES) == 31
 
 
# ── Data loading ──────────────────────────────────────────────────────────────
 
def load_panel(all_features: list) -> pd.DataFrame:
    """
    Load panel and filter to complete cases for all features.
 
    Uses the MICE-imputed panel so NaNs in characteristics are already filled;
    only ret and size are hard requirements for a row to be usable.
 
    Parameters
    ----------
    all_features : list of characteristic column names
 
    Returns
    -------
    DataFrame with columns: permno, date, yy, mm, ret, size, *all_features
    """
    required = ['permno', 'date', 'ret', 'size'] + all_features
    df = pd.read_parquet(PANEL_PATH, columns=required)
    df['date'] = pd.to_datetime(df['date'])
    df['yy'] = df['date'].dt.year
    df['mm'] = df['date'].dt.month
    before = len(df)
    df = df.dropna(subset=['ret', 'size'] + all_features)
    after = len(df)
    print(f"  Panel loaded: {after:,} rows kept ({before - after:,} dropped)")


    return df.reset_index(drop=True)
 
 
# ── Random projection matrix ──────────────────────────────────────────────────
 
def make_projection_matrix(all_features: list,
                            n_select: int,
                            tree_depth: int,
                            rng: np.random.Generator) -> list:
    """
    For each depth level, randomly select n_select features and draw a unit
    vector in that subspace.
 
    Parameters
    ----------
    all_features : full list of feature names to sample from
    n_select     : number of features to use per split
    tree_depth   : number of split levels
    rng          : per-tree RNG for reproducibility
 
    Returns
    -------
    proj_matrix : list of tree_depth dicts, each containing:
        'features' : list[str]  — selected feature names at this depth
        'vector'   : np.ndarray — unit vector of length n_select
    """
    proj_matrix = []
    for _ in range(tree_depth):
        selected = list(rng.choice(all_features, size=n_select, replace=False))
        raw = rng.standard_normal(n_select)
        vec = raw / np.linalg.norm(raw)
        proj_matrix.append({'features': selected, 'vector': vec})
    return proj_matrix
 
 
# ── ntile (unchanged from original step2) ────────────────────────────────────
 
def _ntile(series: pd.Series, q: int) -> pd.Series:
    """
    R's ntile: splits into q roughly equal groups, stable under ties.
    """
    n = len(series)
    order = np.argsort(series.to_numpy(), kind='stable')
    size = n // q
    remainder = n % q
    sizes = [size + 1] * remainder + [size] * (q - remainder)
    bucket = np.empty(n, dtype=int)
    pos = 0
    for i, sz in enumerate(sizes):
        bucket[order[pos:pos + sz]] = i + 1
        pos += sz
    return pd.Series(bucket, index=series.index)
 
 
# ── Node assignment for one month ─────────────────────────────────────────────
 
def assign_nodes_month_rp(df_month: pd.DataFrame,
                           proj_matrix: list,
                           tree_depth: int,
                           q_num: int) -> pd.DataFrame:
    """
    Assign splits and portfolio indices for one month using random projections.
 
    At depth d the scalar projection
        z = X_d @ proj_matrix[d-1]['vector']
    is computed where X_d contains only the features selected for that depth.
    The within-node median of z determines the split.
 
    Parameters
    ----------
    df_month    : rows for a single (year, month)
    proj_matrix : list of dicts from make_projection_matrix
    tree_depth  : number of split levels
    q_num       : bins per split (2 = binary)
 
    Returns
    -------
    df_month with new columns split_1..split_{tree_depth} and port0..port{tree_depth}
    """
    df = df_month.copy()
    for k in range(1, tree_depth + 1):
        df[f'split_{k}'] = 0
    for i in range(0, tree_depth + 1):
        df[f'port{i}'] = 1
 
    if len(df) == 0:
        return df
 
    # Depth-1 split on the full sample
    feat_d1 = proj_matrix[0]['features']
    vec_d1  = proj_matrix[0]['vector']
    z1 = df[feat_d1].to_numpy(dtype=float) @ vec_d1
    df['split_1'] = _ntile(pd.Series(z1, index=df.index), q_num)
 
    # Recursively split each bin
    for bin_val in range(1, q_num + 1):
        mask = df['split_1'] == bin_val
        sub  = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        sub = _recurse_subset_rp(sub, proj_matrix, depth=1,
                                  tree_depth=tree_depth, q_num=q_num)
        for k in range(2, tree_depth + 1):
            df.loc[mask, f'split_{k}'] = sub[f'split_{k}'].values
 
    # Compute port indices (identical formula to original step2)
    for i in range(1, tree_depth + 1):
        port = np.ones(len(df), dtype=int)
        for k in range(1, i + 1):
            split_vals = df[f'split_{k}'].to_numpy(dtype=int)
            port += (split_vals - 1) * (q_num ** (i - k))
        df[f'port{i}'] = port
 
    return df
 
 
def _recurse_subset_rp(df: pd.DataFrame,
                        proj_matrix: list,
                        depth: int,
                        tree_depth: int,
                        q_num: int) -> pd.DataFrame:
    """Recursively assign RP splits for a node subset."""
    if depth >= tree_depth:
        return df
    next_depth = depth + 1
    feat_cols  = proj_matrix[next_depth - 1]['features']
    vec        = proj_matrix[next_depth - 1]['vector']
    z = df[feat_cols].to_numpy(dtype=float) @ vec
    df[f'split_{next_depth}'] = _ntile(pd.Series(z, index=df.index), q_num)
 
    for bin_val in range(1, q_num + 1):
        mask = df[f'split_{next_depth}'] == bin_val
        sub  = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        sub = _recurse_subset_rp(sub, proj_matrix, next_depth, tree_depth, q_num)
        for k in range(next_depth + 1, tree_depth + 1):
            df.loc[mask, f'split_{k}'] = sub[f'split_{k}'].values
    return df
 
 
# ── Single RP tree: compute all 31 node portfolios ────────────────────────────
 
def compute_one_rp_tree(panel: pd.DataFrame,
                         all_features: list,
                         proj_matrix: list,
                         tree_depth: int,
                         q_num: int,
                         y_min: int,
                         y_max: int):
    """
    Compute value-weighted returns and characteristic min/max for all 31 nodes
    of one RP tree.
 
    Parameters
    ----------
    panel       : full panel (all years, NaN-filtered)
    all_features: list of all characteristic names
    proj_matrix : list of dicts from make_projection_matrix
    tree_depth, q_num, y_min, y_max : as in config
 
    Returns
    -------
    ret_table        : np.ndarray (T, 31)
    feat_min_tables  : list of np.ndarray (T, 31), one per feature
    feat_max_tables  : list of np.ndarray (T, 31), one per feature
    """
    n_months = (y_max - y_min + 1) * 12
    n_nodes  = 2 * (2 ** tree_depth) - 1      # 31 for depth=4
    n_feats  = len(all_features)
 
    ret_table       = np.zeros((n_months, n_nodes))
    feat_min_tables = [np.zeros((n_months, n_nodes)) for _ in range(n_feats)]
    feat_max_tables = [np.zeros((n_months, n_nodes)) for _ in range(n_feats)]
 
    for y in range(y_min, y_max + 1):
        if y % 5 == 0:
            print(f"    Year {y}")
 
        df_year = panel[panel['yy'] == y]
        if len(df_year) == 0:
            continue
 
        month_frames = []
        for m in range(1, 13):
            df_month = df_year[df_year['mm'] == m].copy()
            if len(df_month) == 0:
                continue
            df_month = assign_nodes_month_rp(
                df_month, proj_matrix, tree_depth, q_num
            )
            month_frames.append(df_month)
 
        if not month_frames:
            continue
 
        df_assigned = pd.concat(month_frames, ignore_index=True)
 
        for i in range(0, tree_depth + 1):
            n_nodes_at_depth = q_num ** i
            for m in range(1, 13):
                t_idx = 12 * (y - y_min) + (m - 1)
                for k in range(1, n_nodes_at_depth + 1):
                    col_idx = (2 ** i - 1) + (k - 1)
                    mask    = (df_assigned['mm'] == m) & (df_assigned[f'port{i}'] == k)
                    subset  = df_assigned[mask]
 
                    if len(subset) == 0:
                        continue
                    total_size = subset['size'].sum()
                    if total_size == 0:
                        continue
 
                    ret_table[t_idx, col_idx] = (
                        (subset['ret'] * subset['size']).sum() / total_size
                    )
                    for f_idx, feat in enumerate(all_features):
                        feat_min_tables[f_idx][t_idx, col_idx] = subset[feat].min()
                        feat_max_tables[f_idx][t_idx, col_idx] = subset[feat].max()
 
    return ret_table, feat_min_tables, feat_max_tables
 
 
# ── Build all N_TREES RP trees ────────────────────────────────────────────────
 
def create_mice_rp_tree_portfolio(tree_depth: int        = TREE_DEPTH,
                              q_num: int             = Q_NUM,
                              y_min: int             = Y_MIN,
                              y_max: int             = Y_MAX,
                              n_trees: int           = N_TREES,
                              global_seed: int       = GLOBAL_SEED,
                              all_features: list     = ALL_FEATURES,
                              n_features_per_split: int = N_FEATURES_PER_SPLIT,
                              output_path: Path      = OUTPUT_PATH) -> None:
    """
    Build n_trees random-projection trees over all features.
 
    Each tree independently draws N_FEATURES_PER_SPLIT features per depth level
    and a random unit vector in that subspace. All stocks are used in every tree
    (no row sampling); diversity comes from the random split directions.
 
    Output is written to:
        output_path / '_'.join(all_features) /
 
    Files per tree (zero-padded id, e.g. '00'..'80'):
        {id}ret.csv              (T × 31) value-weighted returns
        {id}{feat}_min.csv       (T × 31) per feature
        {id}{feat}_max.csv       (T × 31) per feature
 
    Additionally saves projection_metadata.json for auditability.
    """
    sub_dir = '_'.join(all_features)
    out_dir = output_path / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"\nBuilding {n_trees} RP trees")
    print(f"  Features ({len(all_features)}): {all_features}")
    print(f"  Features per split: {n_features_per_split}")
 
    panel = load_panel(all_features)
 
    master_rng  = np.random.default_rng(global_seed)
    child_seeds = master_rng.integers(0, 2**31, size=n_trees)
    id_width    = len(str(n_trees - 1))
 
    metadata = {}
 
    for t_idx in range(n_trees):
        file_id = str(t_idx).zfill(id_width)
        rng     = np.random.default_rng(int(child_seeds[t_idx]))
 
        proj_matrix = make_projection_matrix(
            all_features, n_features_per_split, tree_depth, rng
        )
 
        # Store metadata for auditability
        metadata[file_id] = {
            f'depth{d + 1}': {
                'features': proj_matrix[d]['features'],
                'vector':   proj_matrix[d]['vector'].tolist(),
            }
            for d in range(tree_depth)
        }
 
        print(f"\n  Tree {t_idx + 1:02d}/{n_trees}  id={file_id}")
        for d, proj in enumerate(proj_matrix):
            coords = '  '.join(
                f"{f}:{v:+.4f}"
                for f, v in zip(proj['features'], proj['vector'])
            )
            print(f"    depth {d + 1}: [{coords}]")
 
        ret_table, feat_min_tables, feat_max_tables = compute_one_rp_tree(
            panel, all_features, proj_matrix, tree_depth, q_num, y_min, y_max
        )
 
        pd.DataFrame(ret_table, columns=CNAMES).to_csv(
            out_dir / f'{file_id}ret.csv', index=False
        )
        for f_idx, feat in enumerate(all_features):
            pd.DataFrame(feat_min_tables[f_idx], columns=CNAMES).to_csv(
                out_dir / f'{file_id}{feat}_min.csv', index=False
            )
            pd.DataFrame(feat_max_tables[f_idx], columns=CNAMES).to_csv(
                out_dir / f'{file_id}{feat}_max.csv', index=False
            )
 
    with open(out_dir / 'projection_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
 
    print(f"\n  Projection metadata saved to projection_metadata.json")
    print(f"  Done. Results saved to {out_dir}")
 
 
# ── Convenience: inspect a saved projection ───────────────────────────────────
 
def load_projection(tree_id: str,
                    all_features: list   = ALL_FEATURES,
                    output_path: Path    = OUTPUT_PATH) -> dict:
    """
    Load and display the projection metadata for a specific tree.
 
    Parameters
    ----------
    tree_id      : zero-padded index string, e.g. '07'
    all_features : same list used during creation
 
    Returns
    -------
    dict with depth-level feature selections and unit vectors
    """
    sub_dir   = '_'.join(all_features)
    meta_path = output_path / sub_dir / 'projection_metadata.json'
    with open(meta_path) as f:
        metadata = json.load(f)
    proj = metadata[tree_id]
    print(f"Projection metadata for tree {tree_id}:")
    for depth_key, info in proj.items():
        coords = '  '.join(
            f"{feat}:{v:+.4f}"
            for feat, v in zip(info['features'], info['vector'])
        )
        print(f"  {depth_key}: features={info['features']}  [{coords}]")
    return proj
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    create_mice_rp_tree_portfolio(
        n_trees              = N_TREES,
        global_seed          = GLOBAL_SEED,
        all_features         = ALL_FEATURES,
        n_features_per_split = N_FEATURES_PER_SPLIT,
        output_path          = OUTPUT_PATH,
    )