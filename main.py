from pathlib import Path
# Importeer je nieuwe imputatie script
from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import prepare_data
from part_1_portfolio_creation.tree_portfolio_creation.step1b_impute_data import impute_characteristics 
from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step2_cluster_portfolios import create_cluster_portfolios
from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import combine_trees
from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import filter_tree_ports



if __name__ == "__main__":
    # --- STAP 1: DATA PREPARATION (Basis) ---
    # Slaat 'panel_benchmark.parquet' op (met NaNs, zoals het paper)
    print("--- STARTING STEP 1: DATA PREPARATION ---")
    #prepare_data()
    
    # --- STAP 1B: IMPUTATIE (Specifiek voor Clustering) ---
    # Slaat 'panel_clustering.parquet' op (met Forward Fill + 0.5)
    print("--- STARTING STEP 1B: IMPUTATION FOR CLUSTERING ---")
    impute_characteristics()
    
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
    print("--- STARTING STEP 2B: CLUSTER PORTFOLIOS (Extension) ---")
    #create_cluster_portfolios()

    print("\nPipeline complete! Results saved in /data/results/ and /data/portfolios/clusters/")

    #Just these for now, full run we need to loop over al 36 combinations of possible characteristics. 
    combine_trees(feat1='OP', feat2='Investment')
    filter_tree_ports(feat1='OP', feat2='Investment')
