from pathlib import Path
import pandas as pd

# Portfolio creation (already done, commented out)
# from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import prepare_data, build_state_variables
# from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
# from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import combine_trees
# from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import filter_tree_ports

# AP-Pruning modules
from part_2_AP_pruning.AP_Pruning import AP_Pruning
from part_2_AP_pruning.kernels.gaussian import GaussianKernel
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
K_MIN        = 5
K_MAX        = 50
PORT_N       = 10   # the k we want to evaluate
N_BANDWIDTHS = 5    # number of bandwidth candidates for the Gaussian kernel

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
    # ─────────────────────────────────────────────────────────────────
    #print("\n=== STEP 1: Gaussian Kernel Grid Search (validation only) ===")
    #AP_Pruning(
    #    feat1=FEAT1,
    #    feat2=FEAT2,
    #    input_path=TREE_PORT_PATH,
    #    input_file_name=PORT_FILE_NAME,
    ##    output_path=GRID_SEARCH_PATH,
    #    n_train_valid=360, cvN=3, runFullCV=False,
    #    kmin=K_MIN, kmax=K_MAX,
    #    RunParallel=False, ParallelN=10, IsTree=True,
    #    lambda0=LAMBDA0, lambda2=LAMBDA2,
    #    kernel_cls=GaussianKernel,
    #    state=state,
    #    n_bandwidths=N_BANDWIDTHS,
    #)

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Pick best hyperparameters for k=PORT_N
    # Scans all validation CSVs, finds the (l0, l2, h) combo
    # that maximises valid_SR at k=PORT_N.
    # ─────────────────────────────────────────────────────────────────
    print(f"\n=== STEP 2: Pick Best (lambda0, lambda2, h) for k={PORT_N} ===")

    # Need n_bandwidths to tell pick_best how many h files to look for
    sigma_s      = state.iloc[:360].std()
    bandwidths   = GaussianKernel.bandwidth_grid(sigma_s, n=N_BANDWIDTHS)
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
    # Step 3: Full fit for the winning (l0*, l2*, h*) at k=PORT_N
    #
    # Runs T_test months using the winning hyperparameters selected for
    # k=PORT_N specifically.  Only k=PORT_N weights are extracted and
    # saved, giving:
    #   full_fit_summary_k{PORT_N}.csv  — SR + hyperparameters (1 row)
    #   full_fit_detail_k{PORT_N}.csv   — per-month weights & excess return
    # ─────────────────────────────────────────────────────────────────
    print(f"\n=== STEP 3: Full Fit for k={PORT_N} ===")

    # Load the adj_ports (same preprocessing as AP_Pruning does internally)
    subdir    = f'LME_{FEAT1}_{FEAT2}'
    ports     = pd.read_csv(TREE_PORT_PATH / subdir / PORT_FILE_NAME)
    depths    = np.array([len(col.split('.')[1]) - 1 for col in ports.columns])
    adj_w     = 1.0 / np.sqrt(2.0 ** depths)
    adj_ports = ports * adj_w

    # Instantiate the winning kernel
    kernel_star = GaussianKernel(h=bandwidths[h_best])

    # Output directory for this run
    full_fit_dir = GRID_SEARCH_PATH / 'gaussian' / subdir / 'full_fit'

    result = kernel_full_fit(
        ports=adj_ports,
        k_target=PORT_N,
        lambda0_star=LAMBDA0[i_best],
        lambda2_star=LAMBDA2[j_best],
        kernel=kernel_star,
        state=state,
        adj_w=adj_w,
        output_dir=str(full_fit_dir),
        n_train_valid=360,
        kmin=K_MIN, kmax=K_MAX,
        kernel_name='gaussian',
    )

    # ─────────────────────────────────────────────────────────────────
    # Step 4: Show results
    # ─────────────────────────────────────────────────────────────────
    print(f"\n=== STEP 4: Results ===")
    print(f"  k={PORT_N}  test_SR={result['test_SR']:.4f}  "
          f"({result['months_used']}/{result['months_total']} test months used)")
    print(f"  Summary CSV : {full_fit_dir}/full_fit_summary_k{PORT_N}.csv")
    print(f"  Detail CSV  : {full_fit_dir}/full_fit_detail_k{PORT_N}.csv")
    print("\nDone.")