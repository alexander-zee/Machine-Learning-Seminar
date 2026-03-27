from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import prepare_data
from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
from pathlib import Path

FEATS_LIST = [
    'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'IdioVol', 'LTurnover'
]

if __name__ == "__main__":
    # Step 1: prepare data once
    prepare_data()
    
    # Step 2: build trees for one triplet to start
    # Later this becomes a loop over all 36 pairs
    create_tree_portfolio(
        feat1       = 'OP',
        feat2       = 'Investment',
        output_path = Path('data/results/tree_portfolios')
    )