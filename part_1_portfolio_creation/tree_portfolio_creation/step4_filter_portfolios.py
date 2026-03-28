"""
Step 4: Filter Single-Sorted Tree Portfolios.

Matches R's filterTreePorts() in Step4_Filter_SingleSorted_Tree_Ports.R.

Removes portfolios that correspond to single-characteristic sorts —
those are already spanned by standard factors and add no pricing information.

Input  (from step3 output, per triplet):
    level_all_excess_combined.csv
    level_all_{feat}_min.csv  × 3
    level_all_{feat}_max.csv  × 3

Output (in same directory):
    level_all_excess_combined_filtered.csv
    level_all_{feat}_min_filtered.csv  × 3
    level_all_{feat}_max_filtered.csv  × 3
"""

import pandas as pd
from pathlib import Path


# ── Defaults ──────────────────────────────────────────────────────────────────
TREE_OUT = Path('data/results/tree_portfolios')

# Tree IDs that represent single-characteristic sorts at every depth level.
# In a 3-feature tree (LME=1, feat1=2, feat2=3):
#   '1111' = split on LME at all 4 levels
#   '2222' = split on feat1 at all 4 levels
#   '3333' = split on feat2 at all 4 levels
SINGLE_SORT_PREFIXES = ('1111.', '2222.', '3333.')

# Depth-4 leaf node names have 5 digits: '11111'..'12222'
# Full column name: e.g. '1111.11111' → length 10 in Python naming
SINGLE_SORT_COL_LEN = 10


# ── Filter logic ──────────────────────────────────────────────────────────────

def _is_single_sort(col_name: str) -> bool:
    """
    Return True if this column should be removed.

    Criterion (matching R's logic adapted for Python naming without 'X' prefix):
      - Starts with one of SINGLE_SORT_PREFIXES  ('1111.', '2222.', '3333.')
      - AND total length is SINGLE_SORT_COL_LEN  (10 chars)

    Examples removed  : '1111.11111', '2222.12222', '3333.11112'
    Examples kept     : '1111.1', '1112.11111', '2131.12121'
    """
    return (
        any(col_name.startswith(p) for p in SINGLE_SORT_PREFIXES)
        and len(col_name) == SINGLE_SORT_COL_LEN
    )


# ── Main filter function ───────────────────────────────────────────────────────

def filter_tree_ports(
    feat1: str,
    feat2: str,
    tree_out: Path = TREE_OUT,
) -> None:
    """
    Remove single-sorted portfolios from combined tree output for one triplet.

    Parameters
    ----------
    feat1, feat2 : second and third characteristics (LME is always first)
    tree_out     : root directory where step3 saved combined CSVs
    """
    feats    = ['LME', feat1, feat2]
    sub_dir  = '_'.join(feats)
    data_dir = tree_out / sub_dir

    print(f"Filtering single-sorted ports for triplet: {feats}")

    # ── Load combined excess return matrix ────────────────────────────────────
    ret_path = data_dir / 'level_all_excess_combined.csv'
    port_ret = pd.read_csv(ret_path)

    print(f"  Before filter: {port_ret.shape[1]} columns")

    # ── Build filter mask ─────────────────────────────────────────────────────
    # True = keep, False = remove
    keep_mask = [not _is_single_sort(c) for c in port_ret.columns]
    n_removed = sum(not k for k in keep_mask)
    print(f"  Removing {n_removed} single-sort portfolios")

    # ── Filter and save return matrix ─────────────────────────────────────────
    port_filtered = port_ret.loc[:, keep_mask]
    print(f"  After filter:  {port_filtered.shape[1]} columns")

    out_ret = data_dir / 'level_all_excess_combined_filtered.csv'
    port_filtered.to_csv(out_ret, index=False)
    print(f"  Saved: {out_ret.name}")

    # ── Apply same mask to all min/max quantile tables ────────────────────────
    for feat in feats:
        for suffix in ('min', 'max'):
            in_path  = data_dir / f'level_all_{feat}_{suffix}.csv'
            out_path = data_dir / f'level_all_{feat}_{suffix}_filtered.csv'

            df = pd.read_csv(in_path)
            df_filtered = df.loc[:, keep_mask]
            df_filtered.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}  shape={df_filtered.shape}")

    print(f"Done: {sub_dir}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    filter_tree_ports(
        feat1    = 'OP',
        feat2    = 'Investment',
        tree_out = TREE_OUT,
    )