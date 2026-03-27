# tests/test_tree_portfolio_vs_r.py

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our functions directly — no parquet needed
from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import (
    assign_nodes_month,
    compute_one_tree,
    CNAMES,
    TREE_DEPTH,
    Q_NUM,
)

# ── Paths to R output ─────────────────────────────────────────────────────────
# Point these at wherever your R output lives
TEST_DIR = Path(__file__).parent
# Root directory is one level up
ROOT_DIR = TEST_DIR.parent

R_CHUNK_DIR  = ROOT_DIR / 'paper_data' / 'data_chunk_files_quantile' / 'LME_OP_Investment'
R_TREE_DIR   = ROOT_DIR / 'paper_data' / 'tree_portfolio_quantile' / 'LME_OP_Investment'

FEAT1 = 'OP'
FEAT2 = 'Investment'
FEATS = ['LME', FEAT1, FEAT2]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_r_yearly_csv(year: int) -> pd.DataFrame:
    """Load the R-generated yearly chunk CSV for one year."""
    path = R_CHUNK_DIR / f'y{year}.csv'
    df   = pd.read_csv(path)
    # R uses lowercase 'size', our code also uses 'size'
    # R column names: yy, mm, date, permno, ret, LME, OP, Investment, size
    return df


def load_r_tree_output(file_id: str, suffix: str) -> pd.DataFrame:
    """
    Load one of R's output CSVs for a specific tree.
    suffix examples: 'ret', 'LME_min', 'OP_max'
    """
    path = R_TREE_DIR / f'{file_id}{suffix}.csv'
    return pd.read_csv(path)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_node_assignment_single_month():
    """
    For January 1964, check that our node assignments match R's.
    We verify by checking that the value-weighted return for each node
    matches R's ret output for tree 1111.
    """
    # Load R's input data for 1964
    df_year = load_r_yearly_csv(1964)

    # Run our node assignment for January only
    df_jan   = df_year[df_year['mm'] == 1].copy()
    feat_list = ['LME', 'LME', 'LME', 'LME']   # tree 1111
    df_result = assign_nodes_month(df_jan, feat_list, TREE_DEPTH, Q_NUM)

    # Load R's ret output for tree 1111
    r_ret = load_r_tree_output('1111', 'ret')

    # R's first row (January 1964) across all 31 nodes
    r_jan = r_ret.iloc[0].values.astype(float)

    # Compute our VW returns for each node in January
    our_jan = np.zeros(31)
    for i in range(0, TREE_DEPTH + 1):
        for k in range(1, Q_NUM**i + 1):
            col_idx = (2**i - 1) + (k - 1)
            mask    = (df_result['mm'] == 1) & (df_result[f'port{i}'] == k)
            subset  = df_result[mask]
            if len(subset) == 0 or subset['size'].sum() == 0:
                continue
            our_jan[col_idx] = (
                (subset['ret'] * subset['size']).sum() / subset['size'].sum()
            )

    # Compare with tolerance for floating point differences
    np.testing.assert_allclose(
        our_jan, r_jan, # type: ignore
        rtol=1e-5, atol=1e-8,
        err_msg="Node returns for tree 1111, January 1964 do not match R output"
    ) # type: ignore
    print("✓ January 1964 node returns match R output")


def test_full_year_single_tree():
    """
    For all 12 months of 1964, check that tree 1111 returns match R exactly.
    """
    df_year   = load_r_yearly_csv(1964)
    feat_list = ['LME', 'LME', 'LME', 'LME']

    # Build a minimal panel with just 1964
    panel = df_year.copy()
    panel['yy'] = 1964

    ret_table, _, _ = compute_one_tree(
        panel     = panel,
        feat_list = feat_list,
        feats     = FEATS,
        tree_depth = TREE_DEPTH,
        q_num      = Q_NUM,
        y_min      = 1964,
        y_max      = 1964,   # only 1 year
    )

    # R's output: 12 rows (one per month), 31 columns (one per node)
    r_ret = load_r_tree_output('1111', 'ret')
    r_1964 = r_ret.iloc[0:12].values.astype(float)

    np.testing.assert_allclose(
        ret_table, r_1964,
        rtol=1e-5, atol=1e-8,
        err_msg="Full year 1964 returns for tree 1111 do not match R output"
    )
    print("✓ Full year 1964 (12 months × 31 nodes) matches R output for tree 1111")


def test_different_tree_ordering():
    """
    Test tree 2131 (OP, LME, Investment, LME) to verify
    that different split orderings also match.
    """
    df_year   = load_r_yearly_csv(1964)
    feat_list = ['OP', 'LME', 'Investment', 'LME']   # tree 2131

    panel = df_year.copy()
    panel['yy'] = 1964

    ret_table, _, _ = compute_one_tree(
        panel      = panel,
        feat_list  = feat_list,
        feats      = FEATS,
        tree_depth = TREE_DEPTH,
        q_num      = Q_NUM,
        y_min      = 1964,
        y_max      = 1964,
    )

    r_ret   = load_r_tree_output('2131', 'ret')
    r_1964  = r_ret.iloc[0:12].values.astype(float)

    np.testing.assert_allclose(
        ret_table, r_1964,
        rtol=1e-5, atol=1e-8,
        err_msg="Full year 1964 returns for tree 2131 do not match R output"
    )
    print("✓ Tree 2131 (OP → LME → Investment → LME) matches R output")


def test_characteristic_min_max():
    """
    Verify that the min/max quantile tables also match R's output.
    Tests LME_min for tree 1111.
    """
    df_year   = load_r_yearly_csv(1964)
    feat_list = ['LME', 'LME', 'LME', 'LME']

    panel = df_year.copy()
    panel['yy'] = 1964

    _, feat_min_tables, feat_max_tables = compute_one_tree(
        panel      = panel,
        feat_list  = feat_list,
        feats      = FEATS,
        tree_depth = TREE_DEPTH,
        q_num      = Q_NUM,
        y_min      = 1964,
        y_max      = 1964,
    )

    r_lme_min = load_r_tree_output('1111', 'LME_min').iloc[0:12].values.astype(float)
    r_lme_max = load_r_tree_output('1111', 'LME_max').iloc[0:12].values.astype(float)

    # feat_min_tables[0] is LME (index 0 in feats)
    np.testing.assert_allclose(
        feat_min_tables[0], r_lme_min,
        rtol=1e-5, atol=1e-8,
        err_msg="LME min quantile ranges do not match R output"
    )
    np.testing.assert_allclose(
        feat_max_tables[0], r_lme_max,
        rtol=1e-5, atol=1e-8,
        err_msg="LME max quantile ranges do not match R output"
    )
    print("✓ LME min/max quantile ranges match R output")

def test_node_assignment_counts():
    df_year = load_r_yearly_csv(1964)
    df_jan = df_year[df_year['mm'] == 1].copy()
    feat_list = ['LME', 'LME', 'LME', 'LME']
    df_result = assign_nodes_month(df_jan, feat_list, TREE_DEPTH, Q_NUM)

    # Print number of stocks per node at each depth
    for i in range(0, TREE_DEPTH + 1):
        counts = df_result.groupby(f'port{i}').size()
        print(f"Depth {i}: {counts.to_dict()}")


if __name__ == '__main__':
    # Run without pytest for quick manual checking
    test_node_assignment_counts()
    test_node_assignment_single_month()
    test_full_year_single_tree()
    test_different_tree_ordering()
    test_characteristic_min_max()
    print("\n✓ All tests passed")