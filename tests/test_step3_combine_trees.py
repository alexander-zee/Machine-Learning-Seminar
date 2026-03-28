# tests/test_step3_combine_trees.py

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import (
    make_col_names,
    CNAMES,
    load_rf,
    r_expand_grid_order
)

# Paths
TEST_DIR = Path(__file__).parent
ROOT_DIR = TEST_DIR.parent
R_TREE_DIR = ROOT_DIR / "paper_data" / "tree_portfolio_quantile" / "LME_OP_Investment"
FACTOR_DIR = ROOT_DIR / "paper_data" / "factor"

FEATS = ["LME", "OP", "Investment"]


def load_r_tree_ret(combo: tuple) -> pd.DataFrame:
    """Load R's tree ret CSV for a given combo (e.g., (1,1,1,1))"""
    tree_id = "".join(str(i) for i in combo)
    path = R_TREE_DIR / f"{tree_id}ret.csv"
    df = pd.read_csv(path)
    # R adds 'X' to column names, e.g., X1, X11
    df.columns = [col[1:] if col.startswith("X") else col for col in df.columns]
    return df


def test_combine_trees_ret():
    """Compare combined excess returns from our logic with R's output."""
    # Build all combos
    n_feats = len(FEATS)
    tree_depth = 4
    all_combos = r_expand_grid_order(n_feats, tree_depth)

    # 1. Load all ret CSVs and rename columns
    ret_frames = []
    for combo in all_combos:
        df = load_r_tree_ret(combo)
        # Make sure columns are exactly CNAMES after stripping X
        if not all(col in CNAMES for col in df.columns):
            raise ValueError(f"Unexpected columns in {combo}: {df.columns.tolist()}")
        tree_id = "".join(str(i) for i in combo)
        df.columns = make_col_names(tree_id, CNAMES)
        ret_frames.append(df)

    combined_ret = pd.concat(ret_frames, axis=1)  # T x (81*31 = 2511)

    # 2. Deduplicate
    arr = combined_ret.to_numpy()
    _, keep_idx = np.unique(arr, axis=1, return_index=True)
    keep_idx = np.sort(keep_idx)
    dedup_ret = combined_ret.iloc[:, keep_idx]

    # 3. Subtract risk-free rate
    rf = load_rf(FACTOR_DIR)
    # Ensure length matches
    if len(rf) != len(dedup_ret):
        raise ValueError(f"RF length {len(rf)} != portfolio rows {len(dedup_ret)}")
    excess = dedup_ret.subtract(rf, axis=0)

    # 4. Load R's combined excess file
    r_excess = pd.read_csv(R_TREE_DIR / "level_all_excess_combined.csv")

    # Compare
    np.testing.assert_allclose(
        excess.values, r_excess.values,
        rtol=1e-5, atol=1e-8,
        err_msg="Combined excess returns do not match R output"
    )
    print("✓ Combined excess returns match R output")


def test_combine_trees_minmax():
    """Test min and max tables for each feature."""
    n_feats = len(FEATS)
    tree_depth = 4
    all_combos = r_expand_grid_order(n_feats, tree_depth)   # consistent order

    # First, compute the dedup mask from returns using the same combo order
    ret_frames = []
    for combo in all_combos:
        df = load_r_tree_ret(combo)
        df.columns = [col[1:] if col.startswith("X") else col for col in df.columns]
        df.columns = make_col_names("".join(str(i) for i in combo), CNAMES)
        ret_frames.append(df)
    combined_ret = pd.concat(ret_frames, axis=1)
    arr_ret = combined_ret.to_numpy()
    _, keep_idx = np.unique(arr_ret, axis=1, return_index=True)
    keep_idx = np.sort(keep_idx)

    for feat in FEATS:
        for suffix in ("min", "max"):
            frames = []
            for combo in all_combos:
                tree_id = "".join(str(i) for i in combo)
                path = R_TREE_DIR / f"{tree_id}{feat}_{suffix}.csv"
                df = pd.read_csv(path)
                df.columns = [col[1:] if col.startswith("X") else col for col in df.columns]
                df.columns = make_col_names(tree_id, CNAMES)
                frames.append(df)

            combined = pd.concat(frames, axis=1)
            deduped = combined.iloc[:, keep_idx]

            r_combined = pd.read_csv(R_TREE_DIR / f"level_all_{feat}_{suffix}.csv")
            np.testing.assert_allclose(
                deduped.values, r_combined.values,
                rtol=1e-5, atol=1e-8,
                err_msg=f"{feat}_{suffix} table does not match R output"
            )
            print(f"✓ {feat}_{suffix} table matches R output")

if __name__ == "__main__":
    test_combine_trees_ret()
    test_combine_trees_minmax()
    print("\nAll combine_trees tests passed")