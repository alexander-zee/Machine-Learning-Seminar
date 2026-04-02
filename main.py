from pathlib import Path
# Importeer je nieuwe imputatie script
from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import prepare_data
#from part_1_portfolio_creation.tree_portfolio_creation.step1b_impute_data import run_mice_imputation 
from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step2_RP_tree_portfolios import create_rp_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step2_cluster_portfolios import create_cluster_portfolios
from part_1_portfolio_creation.tree_portfolio_creation.step2_mice_rp_portfolios import create_mice_rp_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import combine_trees
from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_RP_trees import combine_rp_trees
from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_mice_rp import combine_mice_rp_trees
from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import filter_tree_ports

# AP‑Pruning and metric collection modules
from part_2_AP_pruning.AP_Pruning import AP_Pruning
from part_2_AP_pruning.RP_Pruning import RP_Pruning
from part_2_AP_pruning.Mice_RP_Pruning import Mice_RP_Pruning
from part_3_metrics_collection.pick_best_lambdas import pick_best_lambda, pick_sr_n, get_mu_sigma
from part_3_metrics_collection.mice_pick_best_lambdas import mice_pick_best_lambda, mice_pick_sr_n, mice_get_mu_sigma

# Configuration (same as in R for the demonstration)
FEAT1 = 'OP'
FEAT2 = 'Investment'
LAMBDA0 = [0.5, 0.55, 0.6]
LAMBDA2 = [10**-7, 10**-7.25, 10**-7.5]
PORT_N = 10               # fixed k for best lambda selection
K_MIN = 5
K_MAX = 50
ALL_FEATURES = ['LME', 'BEME', 'r12_2', 'OP', 'Investment', 'ST_Rev', 'LT_Rev', 'AC', 'LTurnover']
N_FEATURES_PER_SPLIT = 3   # number of features randomly selected per split level

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
    #print("--- STARTING STEP 2A: TREE PORTFOLIOS (Benchmark) ---")
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

    #print("\n=== STEP 3: Combine Trees ===")
    #combine_trees(
    #   feat1=FEAT1,
    #   feat2=FEAT2,
    #    factor_path=Path('data/raw'),          # directory containing rf_factor.csv
    #    tree_out=Path('data/results/tree_portfolios')
    #)

    #print("\n=== STEP 4: Filter Single‑Sorted Portfolios ===")
    #filter_tree_ports(
    #    feat1=FEAT1,
    #    feat2=FEAT2,
    #    tree_out=Path('data/results/tree_portfolios')
    #)

    #print("\n=== STEP 5: AP‑Pruning Grid Search ===")
    # The filtered output from step4 is used as input for AP pruning
    #AP_Pruning(
    #    feat1=FEAT1,
    #    feat2=FEAT2,
    #    input_path=Path('data/results/tree_portfolios'),
    #    input_file_name='level_all_excess_combined_filtered.csv',
    #    output_path=Path('data/results/grid_search/tree'),
    #    n_train_valid=360,         # same as in R
    #    cvN=3,
    #    runFullCV=False,
    #    kmin=K_MIN,
    #    kmax=K_MAX,
    #    RunParallel=False,         # set to True if you have many cores
    #    ParallelN=10,
    #    IsTree=True,
    #    lambda0=LAMBDA0,
    #    lambda2=LAMBDA2
    #)

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

    print("\n=== STEP 6D cont.: Mu and Sigma for k=10 ===")
    stats_tree = get_mu_sigma(
        feat1                = FEAT1,
        feat2                = FEAT2,
        ap_prune_result_path = Path('data/results/grid_search/tree'),
        portfolio_path       = Path('data/results/tree_portfolios'),
        port_name            = 'level_all_excess_combined.csv',
        port_n               = PORT_N,
        n_train_valid        = 360
    )
    print(f"Train — mu={stats_tree['train']['mu']:.6f}  sigma={stats_tree['train']['sigma']:.6f}  SR={stats_tree['train']['SR']:.4f}")
    print(f"Test  — mu={stats_tree['test']['mu']:.6f}   sigma={stats_tree['test']['sigma']:.6f}   SR={stats_tree['test']['SR']:.4f}")
    # Verify test SR matches what was saved in step 6D
    print(f"Cross-check: saved test_SR={best_sr[2]:.4f}  recomputed={stats_tree['test']['SR']:.4f}  match={abs(best_sr[2] - stats_tree['test']['SR']) < 1e-8}")

    #print("\n=== STEP 7: Collect SR_N for k = 5..50 ===")
    #pick_sr_n(
    #    feat1=FEAT1,
    #    feat2=FEAT2,
    #    grid_search_path=Path('data/results/grid_search/tree'),
    #    mink=K_MIN,
    #    maxk=K_MAX,
    #    lambda0=LAMBDA0,
    #    lambda2=LAMBDA2,
    #    port_path=Path('data/results/tree_portfolios'),
    #    port_file_name='level_all_excess_combined_filtered.csv'
    #)

    #print("\n=== STEP 2C: RP Tree Portfolios ===")
    #create_rp_tree_portfolio(
    #    feat1       = FEAT1,
    #    feat2       = FEAT2,
    #    output_path = Path('data/results/rp_tree_portfolios')
    #)

    #print("\n=== STEP 3B: Combine RP Trees ===")
    #combine_rp_trees(
    #    feat1       = FEAT1,
    #    feat2       = FEAT2,
    #    factor_path = Path('data/raw'),
    #    tree_out    = Path('data/results/rp_tree_portfolios')
    #)

    # Note: no filter step for RP trees — combine already saves the final CSV

    #print("\n=== STEP 5B: RP-Pruning Grid Search ===")
    #RP_Pruning(
    #    feat1          = FEAT1,
    #    feat2          = FEAT2,
    #    input_path     = Path('data/results/rp_tree_portfolios'),
    #    input_file_name= 'level_all_excess_combined.csv',
    #    output_path    = Path('data/results/grid_search/rp_tree'),
    #    n_train_valid  = 360,
    #    cvN            = 3,
    #    runFullCV      = False,
    #    kmin           = K_MIN,
    #    kmax           = K_MAX,
    #    RunParallel    = False,
    #    ParallelN      = 10,
    #    IsTree         = True,
    #    lambda0        = LAMBDA0,
    #    lambda2        = LAMBDA2
    #)

    print("\n=== STEP 6B: Pick Best Lambda for RP Trees (k=10) ===")
    best_sr_rp = pick_best_lambda(
        feat1               = FEAT1,
        feat2               = FEAT2,
        ap_prune_result_path= Path('data/results/grid_search/rp_tree'),
        port_n              = PORT_N,
        lambda0             = LAMBDA0,
        lambda2             = LAMBDA2,
        portfolio_path      = Path('data/results/rp_tree_portfolios'),
        port_name           = 'level_all_excess_combined.csv',
        full_cv             = False,
        write_table         = True
    )
    print(f"RP Best SR for k={PORT_N}: train={best_sr_rp[0]:.4f}, valid={best_sr_rp[1]:.4f}, test={best_sr_rp[2]:.4f}")

    print("\n=== STEP 6B cont.: Mu and Sigma for k=10 ===")
    stats_rp = get_mu_sigma(
        feat1                = FEAT1,
        feat2                = FEAT2,
        ap_prune_result_path = Path('data/results/grid_search/rp_tree'),
        portfolio_path       = Path('data/results/rp_tree_portfolios'),
        port_name            = 'level_all_excess_combined.csv',
        port_n               = PORT_N,
        n_train_valid        = 360
    )
    print(f"Train — mu={stats_rp['train']['mu']:.6f}  sigma={stats_rp['train']['sigma']:.6f}  SR={stats_rp['train']['SR']:.4f}")
    print(f"Test  — mu={stats_rp['test']['mu']:.6f}   sigma={stats_rp['test']['sigma']:.6f}   SR={stats_rp['test']['SR']:.4f}")
    # Verify test SR matches what was saved in step 6D
    print(f"Cross-check: saved test_SR={best_sr_rp[2]:.4f}  recomputed={stats_rp['test']['SR']:.4f}  match={abs(best_sr_rp[2] - stats_rp['test']['SR']) < 1e-8}")


    #print("\n=== STEP 7B: Collect SR_N for RP Trees (k=5..50) ===")
    #pick_sr_n(
    #    feat1           = FEAT1,
    #    feat2           = FEAT2,
    #    grid_search_path= Path('data/results/grid_search/rp_tree'),
    #    mink            = K_MIN,
    #    maxk            = K_MAX,
    #    lambda0         = LAMBDA0,
    #    lambda2         = LAMBDA2,
    #    port_path       = Path('data/results/rp_tree_portfolios'),
    #    port_file_name  = 'level_all_excess_combined.csv'
    #)

    #print("\n=== STEP 2D: Mice RP Tree Portfolios ===")
    #create_mice_rp_tree_portfolio(n_features_per_split=N_FEATURES_PER_SPLIT, all_features=ALL_FEATURES)

    #print("\n=== STEP 3D: Combine Mice RP Trees ===")
    #combine_mice_rp_trees(all_features=ALL_FEATURES)

    #print("\n=== STEP 5D: Mice RP-Pruning Grid Search ===")
    #Mice_RP_Pruning(allfeatures=ALL_FEATURES, 
    #                input_path=Path('data/results/mice_rp_tree_portfolios'), 
    #                input_file_name='level_all_excess_combined.csv', 
    #                output_path=Path('data/results/grid_search/mice_rp_tree'),
    #                n_train_valid  = 360,
    #                cvN            = 3,
    #                runFullCV      = False,
    #                kmin           = K_MIN,
    #                kmax           = K_MAX,
    #                RunParallel    = False,
    #                ParallelN      = 10,
    #                IsTree         = True,
    #                lambda0        = LAMBDA0,
    #                lambda2        = LAMBDA2)

    print("\n=== STEP 6D: Pick Best Lambda for RP Trees (k=10) ===")
    best_sr_mice_rp = mice_pick_best_lambda( allfeatures=ALL_FEATURES, 
        port_n=PORT_N, 
        ap_prune_result_path='data/results/grid_search/mice_rp_tree',
        lambda0 = LAMBDA0,
        lambda2 = LAMBDA2,
        portfolio_path  = Path('data/results/mice_rp_tree_portfolios'),
        port_name  = 'level_all_excess_combined.csv',
        full_cv  = False,
        write_table = True)
    print(f"RP Best SR for k={PORT_N}: train={best_sr_mice_rp[0]:.4f}, valid={best_sr_mice_rp[1]:.4f}, test={best_sr_mice_rp[2]:.4f}")

    print("\n=== STEP 6D cont.: Mu and Sigma for k=10 ===")
    stats_mice_rp = mice_get_mu_sigma(
        allfeatures          = ALL_FEATURES,
        ap_prune_result_path = Path('data/results/grid_search/mice_rp_tree'),
        portfolio_path       = Path('data/results/mice_rp_tree_portfolios'),
        port_name            = 'level_all_excess_combined.csv',
        port_n               = PORT_N,
        n_train_valid        = 360
    )
    print(f"Train — mu={stats_mice_rp['train']['mu']:.6f}  sigma={stats_mice_rp['train']['sigma']:.6f}  SR={stats_mice_rp['train']['SR']:.4f}")
    print(f"Test  — mu={stats_mice_rp['test']['mu']:.6f}   sigma={stats_mice_rp['test']['sigma']:.6f}   SR={stats_mice_rp['test']['SR']:.4f}")
    # Verify test SR matches what was saved in step 6D
    print(f"Cross-check: saved test_SR={best_sr_mice_rp[2]:.4f}  recomputed={stats_mice_rp['test']['SR']:.4f}  match={abs(best_sr_mice_rp[2] - stats_mice_rp['test']['SR']) < 1e-8}")

    print("\n=== STEP 7D: Collect SR_N for RP Trees (k=5..50) ===")
    #mice_pick_sr_n(allfeatures=ALL_FEATURES, grid_search_path= Path('data/results/grid_search/mice_rp_tree'),
    #    mink            = K_MIN,
    #    maxk            = K_MAX,
    #    lambda0         = LAMBDA0,
    #    lambda2         = LAMBDA2,
    #    port_path       = Path('data/results/mice_rp_tree_portfolios'),
    #    port_file_name  = 'level_all_excess_combined.csv'
    #)   

    print("\nPipeline complete. All results stored under data/results/")
