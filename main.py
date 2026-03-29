import importlib.util
import os
from pathlib import Path

# Importeer je nieuwe imputatie script
from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import prepare_data
from part_1_portfolio_creation.tree_portfolio_creation.step1b_impute_data import impute_characteristics 
from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step2_cluster_portfolios import create_cluster_portfolios
from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import combine_trees
from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import filter_tree_ports


def _load_run_part2():
    p = Path(__file__).resolve().parent / "part_2_AP_pruning" / "run_part2.py"
    spec = importlib.util.spec_from_file_location("seminar_run_part2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    # Leest 'panel_clustering_mice.parquet'; schrijft cluster_returns.csv + dendrogram
    print("--- STARTING STEP 2B: CLUSTER PORTFOLIOS (Extension) ---")
    create_cluster_portfolios()

    print("\nPipeline complete! Results saved in /data/results/ and /data/portfolios/clusters/")

    #Just these for now, full run we need to loop over al 36 combinations of possible characteristics. 
    combine_trees(feat1='OP', feat2='Investment')
    filter_tree_ports(feat1='OP', feat2='Investment')

    print("\n" + "=" * 50 + "\n")
    print("--- PART 2: AP pruning (LASSO λ grid → train / valid / test Sharpe) ---")
    run_trees = os.environ.get("RUN_PART2_TREES", "").lower() in ("1", "true", "yes")
    if not run_trees:
        print(
            "Running cluster portfolios only (fast). "
            "For benchmark AP-trees (~1500 cols, slow) set RUN_PART2_TREES=1 — "
            "Part 3 then prints a Ward vs LME_OP_Investment test_SR comparison. "
            "Or: python part_2_AP_pruning/run_part2.py"
        )
    _load_run_part2().run_part2(run_trees=run_trees, run_clusters=True)
