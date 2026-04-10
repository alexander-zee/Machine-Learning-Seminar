from pathlib import Path
# Importeer je nieuwe imputatie script
from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import prepare_data, build_state_variables
from part_1_portfolio_creation.tree_portfolio_creation.step1b_impute_data import run_mice_imputation 
from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step2_cluster_portfolios import create_cluster_portfolios
from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import combine_trees
from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import filter_tree_ports

# AP‑Pruning and metric collection modules
from part_2_AP_pruning.AP_Pruning import AP_Pruning
from part_2_AP_pruning.kernels.uniform import UniformKernel
from part_2_AP_pruning.kernels.gaussian import GaussianKernel
from part_3_metrics_collection.pick_best_lambdas import pick_best_lambda, pick_sr_n

import pandas as pd

# Configuration (same as in R for the demonstration)
FEAT1 = 'OP'
FEAT2 = 'Investment'
LAMBDA0 = [0.5, 0.55, 0.6]
LAMBDA2 = [10**-7, 10**-7.25, 10**-7.5]
PORT_N = 10               # fixed k for best lambda selection
K_MIN = 5
K_MAX = 50


if __name__ == "__main__":
    # --- STAP 1: DATA PREPARATION (Basis) ---
    # Slaat 'panel_benchmark.parquet' op (met NaNs, zoals het paper)
    #print("--- STARTING STEP 1: DATA PREPARATION ---")
    #prepare_data()
    
    # --- STAP 1B: IMPUTATIE (Specifiek voor Clustering) ---
    # Slaat 'panel_clustering.parquet' op (met Forward Fill + 0.5)
    #print("--- STARTING STEP 1B: IMPUTATION FOR CLUSTERING ---")
    #impute_characteristics()
    
    print("\n" + "="*50 + "\n")

    # --- STAP 2A: BENCHMARK (De Bomen van je collega) ---
    # Deze functie leest intern 'panel_benchmark.parquet' (zonder jouw wijzigingen)
    print("--- STARTING STEP 2A: TREE PORTFOLIOS (Benchmark) ---")
    #create_tree_portfolio(
    #    feat1       = 'OP',
    #    feat2       = 'Investment',
    #    output_path = Path('data/results/tree_portfolios')
    #)

    print("\n" + "="*50 + "\n")

    # --- STAP 2B: EXTENSION (Jouw Ward Clustering) ---
    # Deze functie leest intern 'panel_clustering.parquet'
    #print("--- STARTING STEP 2B: CLUSTER PORTFOLIOS (Extension) ---")
    #create_cluster_portfolios()

    print("\n=== STEP 3: Combine Trees ===")
    combine_trees(
        feat1=FEAT1,
        feat2=FEAT2,
        factor_path=Path('data/raw'),          # directory containing rf_factor.csv
        tree_out=Path('data/results/tree_portfolios')
    )

    print("\n=== STEP 4: Filter Single‑Sorted Portfolios ===")
    filter_tree_ports(
        feat1=FEAT1,
        feat2=FEAT2,
        tree_out=Path('data/results/tree_portfolios')
    )
    
    #Create the state variable which we want, which is in long format. Create the csv, and query from it. 
    #build_state_variables(
    #    final_dataset_path=Path('data/raw/FinalDataset.csv'),
    #    output_path=Path('data/state_variables.csv'),
    #)

    state_df = pd.read_csv('data/state_variables.csv', index_col='MthCalDt', parse_dates=True)
    state    = state_df['svar']   # (636,) one value per month


    print(state)
    print("\n=== STEP 5: AP‑Pruning Grid Search ===")
    # The filtered output from step4 is used as input for AP pruning
     # --- Baseline: uniform kernel (original behavior) ---
    # --- Baseline: uniform kernel (original behavior) ---
    # Results -> data/results/grid_search/tree/uniform/LME_OP_Investment/
    AP_Pruning(
        feat1=FEAT1,
        feat2=FEAT2,
        input_path=Path('data/results/tree_portfolios'),
        input_file_name='level_all_excess_combined_filtered.csv',
        output_path=Path('data/results/grid_search/tree'),
        n_train_valid=360, cvN=3, runFullCV=False,
        kmin=K_MIN, kmax=K_MAX,
        RunParallel=False, ParallelN=10, IsTree=True,
        lambda0=LAMBDA0, lambda2=LAMBDA2,
        kernel_cls=GaussianKernel,
        state = state
    )
 
    # --- Gaussian kernel (svar state variable) ---
    # Results -> data/results/grid_search/tree/gaussian/LME_OP_Investment/
    # TODO: build monthly state variable file and uncomment
    # state = pd.read_csv('data/state_variables.csv',
    #                     index_col='date', parse_dates=True)['svar']
    # AP_Pruning(
    #     feat1=FEAT1,
    #     feat2=FEAT2,
    #     input_path=Path('data/results/tree_portfolios'),
    #     input_file_name='level_all_excess_combined_filtered.csv',
    #     output_path=Path('data/results/grid_search/tree'),
    #     n_train_valid=360, cvN=3, runFullCV=False,
    #     kmin=K_MIN, kmax=K_MAX,
    #     RunParallel=False, ParallelN=10, IsTree=True,
    #     lambda0=LAMBDA0, lambda2=LAMBDA2,
    #     kernel_cls=GaussianKernel,
    #     state=state,
    # )

    print("\n=== STEP 6: Pick Best Lambda (for k = 10) ===")
    # This will generate the files Selected_Ports_10.csv, etc.
    best_sr = pick_best_lambda(
        feat1=FEAT1,
        feat2=FEAT2,
        ap_prune_result_path=Path('data/results/grid_search/tree'),
        port_n=PORT_N,
        lambda0=LAMBDA0,
        lambda2=LAMBDA2,
        portfolio_path=Path('data/results/tree_portfolios'),
        port_name='level_all_excess_combined_filtered.csv',
        full_cv=False,
        write_table=True
    )
    print(f"Best SR for k={PORT_N}: train={best_sr[0]:.4f}, valid={best_sr[1]:.4f}, test={best_sr[2]:.4f}")

    print("\n=== STEP 7: Collect SR_N for k = 5..50 ===")
    pick_sr_n(
        feat1=FEAT1,
        feat2=FEAT2,
        grid_search_path=Path('data/results/grid_search/tree'),
        mink=K_MIN,
        maxk=K_MAX,
        lambda0=LAMBDA0,
        lambda2=LAMBDA2,
        port_path=Path('data/results/tree_portfolios'),
        port_file_name='level_all_excess_combined_filtered.csv'
    )

    print("\nPipeline complete. All results stored under data/results/")

