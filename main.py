from pathlib import Path
from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import prepare_data
from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step2_cluster_portfolios import create_cluster_portfolios

if __name__ == "__main__":
    # --- STAP 1: DATA PREPARATION (Gezamenlijk) ---
    # Dit zorgt voor de 0.5 interpolatie waar beide methodes profijt van hebben
    print("--- STARTING STEP 1: DATA PREPARATION ---")
    prepare_data()
    
    print("\n" + "="*50 + "\n")

    # --- STAP 2A: BENCHMARK (Trees van je collega) ---
    # Later this becomes a loop over all 36 pairs
    print("--- STARTING STEP 2A: TREE PORTFOLIOS (Benchmark) ---")
    create_tree_portfolio(
        feat1       = 'OP',
        feat2       = 'Investment',
        output_path = Path('data/results/tree_portfolios')
    )

    print("\n" + "="*50 + "\n")

    # --- STAP 2B: EXTENSION (Jouw Ward Clustering) ---
    print("--- STARTING STEP 2B: CLUSTER PORTFOLIOS (Extension) ---")
    create_cluster_portfolios()

    print("\nPipeline complete! Results saved in /data/results/ and /data/portfolios/clusters/")
