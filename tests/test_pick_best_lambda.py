# tests/test_pick_best_lambda.py
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from part_3_metrics_collection.pick_best_lambdas import pick_best_lambda, pick_sr_n

TEST_DIR = Path(__file__).parent
ROOT_DIR = TEST_DIR.parent

# Paths to R outputs
R_GRID_DIR = ROOT_DIR / "paper_data" / "TreeGridSearch" / "LME_OP_Investment"
R_FILTERED_CSV = ROOT_DIR / "paper_data" / "tree_portfolio_quantile" / "LME_OP_Investment" / "level_all_excess_combined_filtered.csv"

FEAT1 = "OP"
FEAT2 = "Investment"
LAMBDA0 = [0.5, 0.55, 0.6]
LAMBDA2 = [10**-7, 10**-7.25, 10**-7.5]
K_MIN = 5
K_MAX = 50


def compare_dataframes(our_df, r_df, rtol=1e-5, atol=1e-8):
    """Compare two DataFrames, ignoring column names and index labels."""
    our_vals = our_df.values
    r_vals = r_df.values
    np.testing.assert_allclose(our_vals, r_vals, rtol=rtol, atol=atol)


def test_pick_best_lambda():
    """Test pick_best_lambda for a fixed k (10) using R's grid results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        subdir = "_".join(["LME", FEAT1, FEAT2])
        target_dir = tmp_path / subdir
        target_dir.mkdir(parents=True)

        # Copy R's grid result files (full and cv) into temp directory
        for file in R_GRID_DIR.glob("results_*.csv"):
            shutil.copy(file, target_dir / file.name)

        # Copy filtered portfolio CSV
        shutil.copy(R_FILTERED_CSV, target_dir / "level_all_excess_combined_filtered.csv")

        # Run pick_best_lambda for k=10
        pick_best_lambda(
            feat1=FEAT1, feat2=FEAT2,
            ap_prune_result_path=tmp_path,
            port_n=10,
            lambda0=LAMBDA0,
            lambda2=LAMBDA2,
            portfolio_path=tmp_path,
            port_name="level_all_excess_combined_filtered.csv",
            full_cv=False,
            write_table=True
        )

        # Compare each generated file with R's counterpart
        for fname in ["train_SR_10.csv", "valid_SR_10.csv", "test_SR_10.csv",
                      "Selected_Ports_10.csv", "Selected_Ports_Weights_10.csv"]:
            our_file = target_dir / fname
            r_file = R_GRID_DIR / fname
            if not r_file.exists():
                # R might have only created these for some ks; skip if not present
                continue

            our_df = pd.read_csv(our_file)
            r_df = pd.read_csv(r_file)

            # For Selected_Ports_Weights, R writes without header, so r_df has a single column with no name.
            # Our file has a header "0". We'll just compare values.
            compare_dataframes(our_df, r_df)
            print(f"✓ {fname} matches R output")


def test_pick_sr_n():
    """Test pick_sr_n for k=5..50 using R's grid results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        subdir = "_".join(["LME", FEAT1, FEAT2])
        target_dir = tmp_path / subdir
        target_dir.mkdir(parents=True)

        # Copy R's grid result files and filtered CSV
        for file in R_GRID_DIR.glob("results_*.csv"):
            shutil.copy(file, target_dir / file.name)
        shutil.copy(R_FILTERED_CSV, target_dir / "level_all_excess_combined_filtered.csv")

        # Run pick_sr_n
        pick_sr_n(
            feat1=FEAT1, feat2=FEAT2,
            grid_search_path=tmp_path,
            mink=K_MIN, maxk=K_MAX,
            lambda0=LAMBDA0, lambda2=LAMBDA2,
            port_path=tmp_path,
            port_file_name="level_all_excess_combined_filtered.csv"
        )

        # Compare SR_N.csv
        our_srn = pd.read_csv(target_dir / "SR_N.csv")
        r_srn = pd.read_csv(R_GRID_DIR / "SR_N.csv")

        # Our file has an index column (row names). R file does not.
        # Compare only the numeric values, ignoring the index column.
        our_vals = our_srn.iloc[:, 1:].values  # skip first column (index)
        r_vals = r_srn.values
        np.testing.assert_allclose(our_vals, r_vals, rtol=1e-5, atol=1e-8)
        print("✓ SR_N.csv matches R output")

        # Optionally, also check a few of the per-k files (like Selected_Ports_10.csv)
        # We can rely on test_pick_best_lambda for that.


if __name__ == "__main__":
    test_pick_best_lambda()
    test_pick_sr_n()
    print("\nAll pick_best_lambda tests passed")