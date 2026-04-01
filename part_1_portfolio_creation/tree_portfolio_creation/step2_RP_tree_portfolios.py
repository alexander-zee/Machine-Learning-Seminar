import numpy as np
import pandas as pd
from pathlib import Path
 
 
# ── Config ────────────────────────────────────────────────────────────────────
PANEL_PATH  = Path('data/prepared/panel.parquet')
OUTPUT_PATH = Path('data/results/rp_tree_portfolios')
TREE_DEPTH  = 4
Q_NUM       = 2        # binary median splits
Y_MIN       = 1964
Y_MAX       = 2016
N_TREES     = 81       # replaces the 3^4 feature-combo trees
GLOBAL_SEED = 42       # for reproducibility
 
# Node names for depth=4 — identical encoding to step2 (first digit always 1)
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
 
def load_triplet(feat1: str, feat2: str) -> pd.DataFrame:
    """
    Load panel and filter to complete cases for this triplet.
 
    Parameters
    ----------
    feat1, feat2 : characteristic column names (quantile-ranked in parquet)
 
    Returns
    -------
    DataFrame with columns: permno, date, yy, mm, ret, size, LME, feat1, feat2
    """
    df = pd.read_parquet(
        PANEL_PATH,
        columns=['permno', 'date', 'yy', 'mm', 'ret', 'size', 'LME', feat1, feat2]
    )
    before = len(df)
    df = df.dropna(subset=['LME', feat1, feat2, 'ret', 'size'])
    after  = len(df)
    print(f"  Triplet (LME, {feat1}, {feat2}): {after:,} rows kept ({before-after:,} dropped)")
    return df.reset_index(drop=True)
 
 
# ── Random projection vectors ─────────────────────────────────────────────────
 
def make_projection_matrix(n_feats: int,
                            tree_depth: int,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Draw one random unit vector per depth level.
 
    Returns
    -------
    proj : np.ndarray of shape (tree_depth, n_feats)
        proj[d] is the unit vector used to project features at depth d+1.
 
    Notes
    -----
    Gaussian entries then L2-normalised → uniformly distributed direction
    on the unit sphere (standard RP-tree construction).
    Each tree gets its own RNG state so trees are independent but the whole
    run is reproducible via GLOBAL_SEED.
    """
    raw  = rng.standard_normal((tree_depth, n_feats))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms
 
 
# ── ntile (unchanged from step2) ─────────────────────────────────────────────
 
def _ntile(series: pd.Series, q: int) -> pd.Series:
    """
    R's ntile: splits into q roughly equal groups,
    even when values are tied.
    """
    n = len(series)
    order     = np.argsort(series.to_numpy(), kind='stable')
    size      = n // q
    remainder = n % q
    sizes     = [size + 1] * remainder + [size] * (q - remainder)
    bucket    = np.empty(n, dtype=int)
    pos = 0
    for i, sz in enumerate(sizes):
        bucket[order[pos:pos + sz]] = i + 1
        pos += sz
    return pd.Series(bucket, index=series.index)
 
 
# ── Node assignment for one month ─────────────────────────────────────────────
 
def assign_nodes_month_rp(df_month: pd.DataFrame,
                           feat_cols: list,
                           proj_matrix: np.ndarray,
                           tree_depth: int,
                           q_num: int) -> pd.DataFrame:
    """
    Assign splits and portfolio indices for one month using random projections.
 
    The split at depth d is made on the *scalar projection*
        z = X @ proj_matrix[d-1]
    where X is the (n_stocks × n_feats) matrix of standardised quantile ranks.
    The median of z within the current node determines the split.
 
    Parameters
    ----------
    df_month    : rows for a single (year, month)
    feat_cols   : list of feature column names, e.g. ['LME', 'OP', 'Investment']
    proj_matrix : (tree_depth, n_feats) unit vectors, one per depth level
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
 
    # Features matrix: (n_stocks, n_feats), already quantile-ranked → [0,1]
    X = df[feat_cols].to_numpy(dtype=float)
 
    # Depth-1 split on the full sample
    z1 = X @ proj_matrix[0]                      # scalar projection, shape (n,)
    df['split_1'] = _ntile(pd.Series(z1, index=df.index), q_num)
 
    # Recursively split each bin
    for bin_val in range(1, q_num + 1):
        mask = df['split_1'] == bin_val
        sub  = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        sub = _recurse_subset_rp(sub, feat_cols, proj_matrix,
                                  depth=1, tree_depth=tree_depth, q_num=q_num)
        for k in range(2, tree_depth + 1):
            df.loc[mask, f'split_{k}'] = sub[f'split_{k}'].values
 
    # Compute port indices (identical formula to original)
    for i in range(1, tree_depth + 1):
        port = np.ones(len(df), dtype=int)
        for k in range(1, i + 1):
            split_vals = df[f'split_{k}'].to_numpy(dtype=int)
            port += (split_vals - 1) * (q_num ** (i - k))
        df[f'port{i}'] = port
 
    return df
 
 
def _recurse_subset_rp(df: pd.DataFrame,
                        feat_cols: list,
                        proj_matrix: np.ndarray,
                        depth: int,
                        tree_depth: int,
                        q_num: int) -> pd.DataFrame:
    """Recursively assign RP splits for a node subset."""
    if depth >= tree_depth:
        return df
    next_depth = depth + 1
    X = df[feat_cols].to_numpy(dtype=float)
    z = X @ proj_matrix[next_depth - 1]          # project onto next vector
    df[f'split_{next_depth}'] = _ntile(pd.Series(z, index=df.index), q_num)
 
    for bin_val in range(1, q_num + 1):
        mask = df[f'split_{next_depth}'] == bin_val
        sub  = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        sub = _recurse_subset_rp(sub, feat_cols, proj_matrix,
                                  next_depth, tree_depth, q_num)
        for k in range(next_depth + 1, tree_depth + 1):
            df.loc[mask, f'split_{k}'] = sub[f'split_{k}'].values
    return df
 
 
# ── Single RP tree: compute all 31 node portfolios ────────────────────────────
 
def compute_one_rp_tree(panel: pd.DataFrame,
                         feat_cols: list,
                         proj_matrix: np.ndarray,
                         tree_depth: int,
                         q_num: int,
                         y_min: int,
                         y_max: int):
    """
    Compute value-weighted returns and characteristic min/max for all 31 nodes
    of one RP tree.
 
    Parameters
    ----------
    panel       : full triplet panel (all years, NaN-filtered)
    feat_cols   : list of feature column names ['LME', feat1, feat2]
    proj_matrix : (tree_depth, n_feats) projection matrix for this tree
    tree_depth, q_num, y_min, y_max : as in config
 
    Returns
    -------
    ret_table        : np.ndarray (T, 31)  value-weighted excess returns
    feat_min_tables  : list of 3 np.ndarray (T, 31)
    feat_max_tables  : list of 3 np.ndarray (T, 31)
    """
    n_months = (y_max - y_min + 1) * 12
    n_nodes  = 2 * (2 ** tree_depth) - 1         # 31 for depth=4
    n_feats  = len(feat_cols)
 
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
                df_month, feat_cols, proj_matrix, tree_depth, q_num
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
 
                    mask   = (df_assigned['mm'] == m) & (df_assigned[f'port{i}'] == k)
                    subset = df_assigned[mask]
 
                    if len(subset) == 0:
                        continue
 
                    total_size = subset['size'].sum()
                    if total_size == 0:
                        continue
 
                    vw_ret = (subset['ret'] * subset['size']).sum() / total_size
                    ret_table[t_idx, col_idx] = vw_ret
 
                    for f_idx, feat in enumerate(feat_cols):
                        feat_min_tables[f_idx][t_idx, col_idx] = subset[feat].min()
                        feat_max_tables[f_idx][t_idx, col_idx] = subset[feat].max()
 
    return ret_table, feat_min_tables, feat_max_tables
 
 
# ── All N_TREES RP trees for one triplet ─────────────────────────────────────
 
def create_rp_tree_portfolio(feat1: str,
                              feat2: str,
                              tree_depth: int  = TREE_DEPTH,
                              q_num: int       = Q_NUM,
                              y_min: int       = Y_MIN,
                              y_max: int       = Y_MAX,
                              n_trees: int     = N_TREES,
                              global_seed: int = GLOBAL_SEED,
                              output_path: Path = OUTPUT_PATH) -> None:
    """
    Build n_trees random-projection trees for the triplet (LME, feat1, feat2).
 
    Each tree gets its own reproducible sub-RNG (seeded from global_seed + tree
    index) so results are fully reproducible and individual trees can be
    regenerated independently.
 
    Output structure mirrors step2 so step3 (combine_trees) works unchanged:
        output_path/LME_feat1_feat2/{file_id}ret.csv          (T × 31)
        output_path/LME_feat1_feat2/{file_id}{feat}_min.csv   (T × 31)
        output_path/LME_feat1_feat2/{file_id}{feat}_max.csv   (T × 31)
 
    file_id is zero-padded tree index, e.g. '000', '001', ..., '080'.
 
    Additionally saves projection_matrices.npz so any tree can be inspected or
    reproduced later:
        proj['{file_id}'] → shape (tree_depth, n_feats)
    """
    feat_cols = ['LME', feat1, feat2]
    n_feats   = len(feat_cols)
    sub_dir   = '_'.join(feat_cols)
    out_dir   = output_path / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"\nBuilding {n_trees} RP trees for triplet: {feat_cols}")
 
    panel = load_triplet(feat1, feat2)
 
    # Master RNG — spawn one child RNG per tree for independence + reproducibility
    master_rng   = np.random.default_rng(global_seed)
    child_seeds  = master_rng.integers(0, 2**31, size=n_trees)
 
    saved_projections = {}
    id_width = len(str(n_trees - 1))   # e.g. 2 for 81 trees → '00'..'80'
 
    for t_idx in range(n_trees):
        file_id = str(t_idx).zfill(id_width)
        rng     = np.random.default_rng(int(child_seeds[t_idx]))
 
        proj_matrix = make_projection_matrix(n_feats, tree_depth, rng)
        saved_projections[file_id] = proj_matrix
 
        print(f"  Tree {t_idx+1:02d}/{n_trees}: id={file_id}")
        print(f"    Projection vectors (rows = depth levels):")
        for d, v in enumerate(proj_matrix):
            coords = "  ".join(f"{feat_cols[j]}:{v[j]:+.4f}" for j in range(n_feats))
            print(f"      depth {d+1}: [{coords}]")
 
        ret_table, feat_min_tables, feat_max_tables = compute_one_rp_tree(
            panel, feat_cols, proj_matrix, tree_depth, q_num, y_min, y_max
        )
 
        # Save returns
        pd.DataFrame(ret_table, columns=CNAMES).to_csv(
            out_dir / f'{file_id}ret.csv', index=False
        )
        # Save characteristic min/max ranges
        for f_idx, feat in enumerate(feat_cols):
            pd.DataFrame(feat_min_tables[f_idx], columns=CNAMES).to_csv(
                out_dir / f'{file_id}{feat}_min.csv', index=False
            )
            pd.DataFrame(feat_max_tables[f_idx], columns=CNAMES).to_csv(
                out_dir / f'{file_id}{feat}_max.csv', index=False
            )
 
    # Save all projection matrices for auditability / reproducibility
    np.savez(out_dir / 'projection_matrices.npz', **saved_projections)
    print(f"\n  Projection matrices saved to projection_matrices.npz")
    print(f"  Done. Results saved to {out_dir}")
 
 
# ── Convenience: inspect a saved projection matrix ────────────────────────────
 
def load_projection(feat1: str,
                    feat2: str,
                    tree_id: str,
                    output_path: Path = OUTPUT_PATH) -> np.ndarray:
    """
    Load and display the projection matrix for a specific tree.
 
    Parameters
    ----------
    feat1, feat2 : same as used during creation
    tree_id      : zero-padded index string, e.g. '07'
 
    Returns
    -------
    proj : np.ndarray of shape (tree_depth, n_feats)
    """
    feat_cols = ['LME', feat1, feat2]
    npz_path  = output_path / '_'.join(feat_cols) / 'projection_matrices.npz'
    data      = np.load(npz_path)
    proj      = data[tree_id]
    print(f"Projection matrix for tree {tree_id}  (feat_cols={feat_cols}):")
    for d, v in enumerate(proj):
        coords = "  ".join(f"{feat_cols[j]}:{v[j]:+.4f}" for j in range(len(feat_cols)))
        print(f"  depth {d+1}: [{coords}]")
    return proj
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    create_rp_tree_portfolio(
        feat1       = 'OP',
        feat2       = 'Investment',
        n_trees     = N_TREES,
        global_seed = GLOBAL_SEED,
        output_path = OUTPUT_PATH,
    )