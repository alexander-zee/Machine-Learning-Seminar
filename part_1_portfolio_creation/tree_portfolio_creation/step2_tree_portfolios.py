import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product


# ── Config ────────────────────────────────────────────────────────────────────
PANEL_PATH  = Path('data/prepared/panel.parquet')
OUTPUT_PATH = Path('data/results/tree_portfolios')
TREE_DEPTH  = 4
Q_NUM       = 2        # binary median splits
Y_MIN       = 1964
Y_MAX       = 2016

# Node names for depth=4, matches R's cnames exactly
# Encoding: first digit always 1 (root), subsequent digits 1=low 2=high
CNAMES = [
    '1',
    '11', '12',
    '111', '112', '121', '122',
    '1111', '1112', '1121', '1122', '1211', '1212', '1221', '1222',
    '11111', '11112', '11121', '11122', '11211', '11212', '11221', '11222',
    '12111', '12112', '12121', '12122', '12211', '12212', '12221', '12222',
]
# Sanity check: 1 + 2 + 4 + 8 + 16 = 31 nodes
assert len(CNAMES) == 31


# ── Data loading ──────────────────────────────────────────────────────────────

def load_triplet(feat1: str, feat2: str) -> pd.DataFrame:
    """
    Load panel and filter to complete cases for this triplet.
    Equivalent to R's intersect operation per (year, month).

    Parameters
    ----------
    feat1, feat2 : characteristic column names (already quantile-ranked in parquet)

    Returns
    -------
    DataFrame with columns: permno, date, yy, mm, ret, size, LME, feat1, feat2
    NaN rows for any of the three characteristics are dropped.
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


# ── Node assignment for one month ─────────────────────────────────────────────

def assign_nodes_month(df_month: pd.DataFrame,
                        feat_list: list,
                        tree_depth: int,
                        q_num: int) -> pd.DataFrame:
    """Assign splits and compute port indices for one month."""
    df = df_month.copy()
    # Initialize split and port columns
    for k in range(1, tree_depth + 1):
        df[f'split_{k}'] = 0
    for i in range(0, tree_depth + 1):
        df[f'port{i}'] = 1

    if len(df) == 0:
        return df

    # Depth‑1 split
    df['split_1'] = _ntile(df[feat_list[0]], q_num)

    # Recursively assign deeper splits, bin by bin
    for bin_val in range(1, q_num + 1):
        mask = df['split_1'] == bin_val
        sub = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        sub = _recurse_subset(sub, feat_list, depth=1,
                              tree_depth=tree_depth, q_num=q_num)
        # Copy back split columns for depths > 1
        for k in range(2, tree_depth + 1):
            df.loc[mask, f'split_{k}'] = sub[f'split_{k}'].values

    # Compute port indices
    for i in range(1, tree_depth + 1):
        port = np.ones(len(df), dtype=int)
        for k in range(1, i + 1):
            split_vals = df[f'split_{k}'].to_numpy(dtype=int)
            port += (split_vals - 1) * (q_num ** (i - k))
        df[f'port{i}'] = port

    return df

def _recurse_subset(df: pd.DataFrame,
                    feat_list: list,
                    depth: int,
                    tree_depth: int,
                    q_num: int) -> pd.DataFrame:
    """Recursively assign splits for a subset; returns updated subset."""
    if depth >= tree_depth:
        return df
    next_depth = depth + 1
    # Assign split at next_depth
    df[f'split_{next_depth}'] = _ntile(df[feat_list[next_depth - 1]], q_num)
    # Recurse into each bin
    for bin_val in range(1, q_num + 1):
        mask = df[f'split_{next_depth}'] == bin_val
        sub = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        sub = _recurse_subset(sub, feat_list, next_depth,
                              tree_depth, q_num)
        # Copy back deeper splits
        for k in range(next_depth + 1, tree_depth + 1):
            df.loc[mask, f'split_{k}'] = sub[f'split_{k}'].values
    return df

def _ntile(series: pd.Series, q: int) -> pd.Series:
    """
    R's ntile: splits into q roughly equal groups,
    even when values are tied.
    """
    n = len(series)
    # Stable sort order (preserves original order for ties)
    order = np.argsort(series.to_numpy(), kind='stable')
    
    # Calculate bucket sizes
    size = n // q
    remainder = n % q
    sizes = [size + 1] * remainder + [size] * (q - remainder)
    
    # Assign bucket numbers to sorted positions
    bucket = np.empty(n, dtype=int)
    pos = 0
    for i, sz in enumerate(sizes):
        bucket[order[pos:pos+sz]] = i + 1
        pos += sz
    
    return pd.Series(bucket, index=series.index)


# ── Single tree: compute all 31 node portfolios ───────────────────────────────

def compute_one_tree(panel: pd.DataFrame,
                      feat_list: list,
                      feats: list,
                      tree_depth: int,
                      q_num: int,
                      y_min: int,
                      y_max: int):
    """
    Compute value-weighted returns and characteristic min/max for all 31 nodes
    of one tree defined by feat_list.

    Matches R's tree_portfolio function.

    Parameters
    ----------
    panel     : full triplet panel (all years, already filtered for NaNs)
    feat_list : list of length tree_depth, which characteristic to split on
                at each depth level e.g. ['LME', 'OP', 'LME', 'Investment']
    feats     : the three base characteristics ['LME', feat1, feat2]
                used to track min/max quantile ranges per node

    Returns
    -------
    ret_table        : np.ndarray (T, 31)  value-weighted excess returns
    feat_min_tables  : list of 3 np.ndarray (T, 31)  min quantile per node
    feat_max_tables  : list of 3 np.ndarray (T, 31)  max quantile per node
    """
    n_months = (y_max - y_min + 1) * 12
    n_nodes  = q_num ** (tree_depth + 1) - 1    # 31 for depth=4
    n_feats  = len(feats)

    ret_table        = np.zeros((n_months, n_nodes))
    feat_min_tables  = [np.zeros((n_months, n_nodes)) for _ in range(n_feats)]
    feat_max_tables  = [np.zeros((n_months, n_nodes)) for _ in range(n_feats)]

    for y in range(y_min, y_max + 1):
        if y % 5 == 0:
            print(f"    Year {y}")

        df_year = panel[panel['yy'] == y]
        if len(df_year) == 0:
            continue

        # Assign node memberships month by month (fresh splits each month)
        month_frames = []
        for m in range(1, 13):
            df_month = df_year[df_year['mm'] == m].copy()
            if len(df_month) == 0:
                continue
            df_month = assign_nodes_month(df_month, feat_list, tree_depth, q_num)
            month_frames.append(df_month)

        if len(month_frames) == 0:
            continue

        df_assigned = pd.concat(month_frames, ignore_index=True)

        # ── For each depth level, month, and node: compute VW return + ranges ─
        for i in range(0, tree_depth + 1):     # depth 0 (root) to tree_depth
            n_nodes_at_depth = q_num ** i       # 1, 2, 4, 8, 16

            for m in range(1, 13):
                t_idx = 12 * (y - y_min) + (m - 1)   # 0-indexed time row

                for k in range(1, n_nodes_at_depth + 1):   # 1-indexed node
                    # Column index matches R's: 2^i - 1 + k  (converted to 0-indexed)
                    col_idx = (2 ** i - 1) + (k - 1)

                    mask   = (df_assigned['mm'] == m) & (df_assigned[f'port{i}'] == k)
                    subset = df_assigned[mask]

                    if len(subset) == 0:
                        continue

                    total_size = subset['size'].sum()
                    if total_size == 0:
                        continue

                    # Value-weighted return: sum(ret * size) / sum(size)
                    vw_ret = (subset['ret'] * subset['size']).sum() / total_size
                    ret_table[t_idx, col_idx] = vw_ret

                    # Min/max quantile of each characteristic within this node
                    # These are the quantile-ranked values (0-1) from parquet
                    for f_idx, feat in enumerate(feats):
                        feat_min_tables[f_idx][t_idx, col_idx] = subset[feat].min()
                        feat_max_tables[f_idx][t_idx, col_idx] = subset[feat].max()

    return ret_table, feat_min_tables, feat_max_tables


# ── All 81 trees for one triplet ──────────────────────────────────────────────

def create_tree_portfolio(feat1: str,
                           feat2: str,
                           tree_depth: int = TREE_DEPTH,
                           q_num: int = Q_NUM,
                           y_min: int = Y_MIN,
                           y_max: int = Y_MAX,
                           output_path: Path = OUTPUT_PATH) -> None:
    """
    Build all q_num^tree_depth = 81 trees for the triplet (LME, feat1, feat2).
    Saves results to output_path/LME_feat1_feat2/

    Matches R's create_tree_portfolio function.
    For each of the 81 trees, saves:
        {file_id}ret.csv              : (T, 31) value-weighted returns
        {file_id}{feat}_min.csv       : (T, 31) min quantile per node
        {file_id}{feat}_max.csv       : (T, 31) max quantile per node
    """
    feats   = ['LME', feat1, feat2]
    sub_dir = '_'.join(feats)
    out_dir = output_path / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding trees for triplet: {feats}")

    # Load triplet once — all workers share this filtered panel
    panel = load_triplet(feat1, feat2)

    # All 3^4 = 81 combinations of which feature to split on at each depth
    # Matches R's expand.grid(rep(list(1:n_feats), tree_depth))
    # Each combo is a tuple of 0-based indices into feats
    all_combos = list(product(range(len(feats)), repeat=tree_depth))
    print(f"  Processing {len(all_combos)} trees ({len(feats)}^{tree_depth})...")

    for k, combo in enumerate(all_combos):
        # feat_list: which characteristic to split on at each depth level
        feat_list = [feats[i] for i in combo]

        file_id   = ''.join(str(i + 1) for i in combo)

        print(f"  Tree {k+1:02d}/{len(all_combos)}: id={file_id}  splits={feat_list}")

        ret_table, feat_min_tables, feat_max_tables = compute_one_tree(
            panel, feat_list, feats, tree_depth, q_num, y_min, y_max
        )

        # ── Save results ──────────────────────────────────────────────────────
        # Returns
        pd.DataFrame(ret_table, columns=CNAMES).to_csv(
            out_dir / f'{file_id}ret.csv', index=False
        )
        # Characteristic min/max quantile ranges per node
        for f_idx, feat in enumerate(feats):
            pd.DataFrame(feat_min_tables[f_idx], columns=CNAMES).to_csv(
                out_dir / f'{file_id}{feat}_min.csv', index=False
            )
            pd.DataFrame(feat_max_tables[f_idx], columns=CNAMES).to_csv(
                out_dir / f'{file_id}{feat}_max.csv', index=False
            )

    print(f"  Done. Results saved to {out_dir}")