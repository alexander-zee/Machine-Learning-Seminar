from pathlib import Path
import pandas as pd

# Portfolio creation (already done, commented out)
# from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import prepare_data, build_state_variables
# from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
# from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import combine_trees
# from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import filter_tree_ports

# AP-Pruning modules
from part_2_AP_pruning.AP_Pruning import AP_Pruning
from part_2_AP_pruning.kernels.dummy_uniform import DummyUniformKernel as UniformKernel
from part_2_AP_pruning.lasso_kernel_full_fit import kernel_full_fit

# Metrics
from part_3_metrics_collection.pick_best_lambdas import pick_best_lambda_kernel

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
FEAT1   = 'OP'
FEAT2   = 'Investment'
LAMBDA0 = [0.5, 0.55, 0.6]
LAMBDA2 = [10**-7, 10**-7.25, 10**-7.5]
K_MIN   = 5
K_MAX   = 50
PORT_N  = 10   # the k we want to evaluate

TREE_PORT_PATH  = Path('data/results/tree_portfolios')
GRID_SEARCH_PATH = Path('data/results/grid_search/tree')
PORT_FILE_NAME  = 'level_all_excess_combined_filtered.csv'


if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────────
    # Load state variable
    # ─────────────────────────────────────────────────────────────────
    state_df = pd.read_csv('data/state_variables.csv',
                           index_col='MthCalDt', parse_dates=True)
    state = state_df['svar']
    print(f"State variable loaded: {len(state)} months")

    # ─────────────────────────────────────────────────────────────────
    # Step 1: Grid search — validation only, no full fit
    #
    # For each bandwidth h (7 candidates):
    #     For CV fold 3:
    #         120 validation months × 9 (l0, l2) combos
    #         → 9 CSVs with valid_SR per k
    # Total: 7 × 120 = 840 per-month LARS calls
    # Output: 63 CSVs in gaussian/LME_OP_Investment/
    # ─────────────────────────────────────────────────────────────────
    print("\n=== STEP 1: Gaussian Kernel Grid Search (validation only) ===")
    AP_Pruning(
        feat1=FEAT1,
        feat2=FEAT2,
        input_path=TREE_PORT_PATH,
        input_file_name=PORT_FILE_NAME,
        output_path=GRID_SEARCH_PATH,
        n_train_valid=360, cvN=3, runFullCV=False,
        kmin=K_MIN, kmax=K_MAX,
        RunParallel=False, ParallelN=10, IsTree=True,
        lambda0=LAMBDA0, lambda2=LAMBDA2,
        kernel_cls=UniformKernel,
        state=state,
    )

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Pick best hyperparameters for k=PORT_N
    #
    # Scans all 63 validation CSVs, finds the (l0, l2, h) combo
    # that maximises valid_SR at k=PORT_N.
    # ─────────────────────────────────────────────────────────────────
    print(f"\n=== STEP 2: Pick Best (lambda0, lambda2, h) for k={PORT_N} ===")

    # Need n_bandwidths to tell pick_best how many h files to look for
    sigma_s      = state.iloc[:360].std()
    bandwidths   = UniformKernel.bandwidth_grid()
    n_bandwidths = len(bandwidths)

    res = pick_best_lambda_kernel(
        feat1=FEAT1, feat2=FEAT2,
        ap_prune_result_path=GRID_SEARCH_PATH,
        port_n=PORT_N,
        lambda0=LAMBDA0, lambda2=LAMBDA2,
        n_bandwidths=n_bandwidths,
        kernel_name='gaussian',
        full_cv=False, write_table=True,
    )

    i_best, j_best, h_best = res['best_idx']
    print(f"  Winner: lambda0={LAMBDA0[i_best]}, lambda2={LAMBDA2[j_best]:.2e}, "
          f"h={bandwidths[h_best]:.6f} (h_idx={h_best+1})")
    print(f"  Validation SR: {res['valid_SR']:.4f}")

    # ─────────────────────────────────────────────────────────────────
    # Step 3: Full fit for the winning combo
    #
    # Runs 276 test months for just the one winning (l0*, l2*, h*).
    # The resulting CSV has test_SR for ALL k values (5..50),
    # so this single run covers any k you might want later.
    # ─────────────────────────────────────────────────────────────────
    print(f"\n=== STEP 3: Full Fit for Winning Combo ===")

    # Load the adj_ports (same preprocessing as AP_Pruning does internally)
    subdir = f'LME_{FEAT1}_{FEAT2}'
    ports  = pd.read_csv(TREE_PORT_PATH / subdir / PORT_FILE_NAME)
    depths = np.array([len(col.split('.')[1]) - 1 for col in ports.columns])
    adj_w  = 1.0 / np.sqrt(2.0 ** depths)
    adj_ports = ports * adj_w

    # Instantiate the winning kernel
    kernel_star = UniformKernel(h=bandwidths[h_best])

    kernel_full_fit(
        ports=adj_ports,
        lambda0_star=LAMBDA0[i_best],
        lambda2_star=LAMBDA2[j_best],
        main_dir=str(GRID_SEARCH_PATH / 'gaussian'),
        sub_dir=subdir,
        adj_w=adj_w,
        kernel=kernel_star,
        h_idx=h_best + 1,
        state=state,
        n_train_valid=360,
        kmin=K_MIN, kmax=K_MAX,
    )

    # ─────────────────────────────────────────────────────────────────
    # Step 4: Read test_SR from the full fit
    # ─────────────────────────────────────────────────────────────────
    print(f"\n=== STEP 4: Results ===")
    full_csv = (GRID_SEARCH_PATH / 'gaussian' / subdir /
                f'results_full_l0_1_l2_1_h_{h_best+1}.csv')
    full_data = pd.read_csv(full_csv)
    print(full_data.to_string(index=False))

    # Extract test_SR for the specific k we care about
    row = full_data[full_data['portsN'] == PORT_N]
    if len(row) > 0:
        test_sr = row.iloc[0]['test_SR']
        print(f"\n  k={PORT_N}: test_SR = {test_sr:.4f}")
    else:
        print(f"\n  k={PORT_N} not found in full fit results.")
        print(f"  Available k values: {sorted(full_data['portsN'].tolist())}")

    print("\nDone.")