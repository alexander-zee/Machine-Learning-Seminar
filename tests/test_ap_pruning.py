"""
diagnostic_portfolio_overlap.py

Prints portfolio overlap between our AP-Pruning output and R's for every
(lambda0, lambda2) combination and every k value. No assertions — just output.

Run with: python diagnostic_portfolio_overlap.py
"""

import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR      = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

FILTERED_CSV  = ROOT_DIR / "paper_data" / "tree_portfolio_quantile" / "LME_OP_Investment" / "level_all_excess_combined_filtered.csv"
R_RESULTS_DIR = ROOT_DIR / "paper_data" / "TreeGridSearch" / "LME_OP_Investment"

FEAT1 = "OP"
FEAT2 = "Investment"
LAMBDA0 = [0.5, 0.55, 0.6]
LAMBDA2 = [10**-7, 10**-7.25, 10**-7.5]
N_TRAIN_VALID = 360
CVN   = 3
KMIN  = 5
KMAX  = 50

META_COLS = {'train_SR', 'valid_SR', 'test_SR', 'portsN'}


def get_beta_values(row, df):
    beta_cols = [c for c in df.columns if c not in META_COLS]
    return row[beta_cols].values.astype(float)


def selected_return_series(result_df, k, filtered_returns):
    row = result_df[result_df['portsN'] == k].iloc[0]
    betas = get_beta_values(row, result_df)
    nonzero_idx = np.where(betas != 0)[0]
    return frozenset(tuple(filtered_returns[:, idx]) for idx in nonzero_idx)


def run_ap_pruning():
    from part_2_AP_pruning.AP_Pruning import AP_Pruning
    from part_2_AP_pruning.kernels.uniform import UniformKernel

    tmpdir = Path(tempfile.mkdtemp())
    triplet_dir = tmpdir / "LME_OP_Investment"
    triplet_dir.mkdir()
    shutil.copy(FILTERED_CSV, triplet_dir / "level_all_excess_combined_filtered.csv")
    print("Running AP_Pruning... (this takes a few minutes)")
    AP_Pruning(
        feat1=FEAT1, feat2=FEAT2,
        input_path=tmpdir,
        input_file_name="level_all_excess_combined_filtered.csv",
        output_path=tmpdir,
        n_train_valid=N_TRAIN_VALID, cvN=CVN, runFullCV=False,
        kmin=KMIN, kmax=KMAX, RunParallel=False,
        IsTree=True, lambda0=LAMBDA0, lambda2=LAMBDA2,
        kernel_cls=UniformKernel,
    )
    # AP_Pruning adds kernel name subfolder internally -> tmpdir/uniform/LME_OP_Investment
    return tmpdir / "uniform" / "LME_OP_Investment"


def main():
    filtered_returns = pd.read_csv(FILTERED_CSV).values   # (T, N)

    our_dir = run_ap_pruning()

    print("\n" + "="*70)
    print("PORTFOLIO OVERLAP: our LARS vs R's LARS")
    print("(number of differing portfolios at each k, for each lambda pair)")
    print("="*70)

    for i, l0 in enumerate(LAMBDA0, 1):
        for j, l2 in enumerate(LAMBDA2, 1):
            # Our files now have _h_1 suffix; R files use the original naming
            our_fname = f"results_full_l0_{i}_l2_{j}_h_1.csv"
            r_fname   = f"results_full_l0_{i}_l2_{j}.csv"

            our = pd.read_csv(our_dir       / our_fname)
            r   = pd.read_csv(R_RESULTS_DIR / r_fname)

            common_k = sorted(set(our['portsN']) & set(r['portsN']))

            print(f"\nλ0={l0:.2f}  λ2={l2:.2e}  ({len(common_k)} common k values)")
            print(f"  {'k':>4}  {'match':>6}  {'total':>6}  {'overlap':>8}  {'differ':>6}")
            print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}")

            any_diff = False
            for k in common_k:
                our_series = selected_return_series(our, k, filtered_returns)
                r_series   = selected_return_series(r,   k, filtered_returns)
                if not r_series:
                    continue
                matched = len(our_series & r_series)
                total   = len(r_series)
                overlap = matched / total
                n_diff  = total - matched
                flag    = " ←" if n_diff > 0 else ""
                print(f"  {k:>4}  {matched:>6}  {total:>6}  {overlap:>7.1%}  {n_diff:>6}{flag}")
                if n_diff > 0:
                    any_diff = True

            if not any_diff:
                print("  All k values: 100% overlap ✓")

    print("\n" + "="*70)
    print("Done.")

    import shutil as _shutil
    _shutil.rmtree(our_dir.parent.parent, ignore_errors=True)


if __name__ == "__main__":
    main()