"""
rerun_beme.py — Reset BEME cross-sections to pending and rerun all 4 kernels.

Steps
-----
1. For each kernel's progress CSV, set all rows where feat1=='BEME' or
   feat2=='BEME' to status='pending' and clear the result columns.
2. Run each standard_*_all.py in sequence. Since all non-BEME rows are
   already 'done', each script will only process the 16 BEME cross-sections.

Usage
-----
    python rerun_beme.py          # from project root
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — mirrors paths in each standard_*_all.py
# ─────────────────────────────────────────────────────────────────────────────

GRID_SEARCH_PATH = Path("data/results/grid_search/tree")

# (subfolder, progress_csv, columns_to_clear)
KERNELS = [
    #(
    #    "uniform",
    #    "progress_standard_uniform.csv",
    #    ["status", "train_SR", "valid_SR", "test_SR", "lambda0", "lambda2", "error"],
    #),
    #(
    #    "gaussian",
    #    "progress_standard_gaussian.csv",
    #    ["status", "test_SR", "valid_SR", "months_used", "lambda0", "lambda2", "h", "error"],
    #),
    (
        "gaussian-tms",
        "progress_standard_gaussian_tms.csv",
        ["status", "test_SR", "valid_SR", "months_used", "lambda0", "lambda2", "h", "error"],
    ),
    (
        "exponential",
        "progress_standard_exponential.csv",
        ["status", "test_SR", "valid_SR", "months_used", "lambda0", "lambda2", "h", "error"],
    ),
]

SCRIPTS = [
    "standard_gaussian_tms_all.py",
    "standard_exponential_all.py",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Reset BEME rows
# ─────────────────────────────────────────────────────────────────────────────

def reset_beme_rows() -> None:
    for kernel_dir, progress_file, cols_to_clear in KERNELS:
        progress_path = GRID_SEARCH_PATH / kernel_dir / progress_file

        if not progress_path.exists():
            print(f"  [SKIP] Not found: {progress_path}")
            continue

        df = pd.read_csv(progress_path, dtype=object)
        beme_mask = (df["feat1"] == "BEME") | (df["feat2"] == "BEME")
        n = beme_mask.sum()

        df.loc[beme_mask, "status"] = "pending"
        for col in cols_to_clear:
            if col != "status" and col in df.columns:
                df.loc[beme_mask, col] = None

        df.to_csv(progress_path, index=False)
        print(f"  [{kernel_dir}] Reset {n} BEME rows → {progress_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run each script
# ─────────────────────────────────────────────────────────────────────────────

def run_scripts() -> None:
    for script in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  Running: {script}")
        print(f"{'='*60}\n")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            print(f"\n  WARNING: {script} exited with code {result.returncode}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Step 1: Resetting BEME rows to pending...\n")
    reset_beme_rows()

    print("\nStep 2: Running all kernels...\n")
    run_scripts()

    print("\nDone.")