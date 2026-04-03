import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web
from pathlib import Path
import warnings

# Verberg de waarschuwingen
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_ff5_regression(portfolio_returns, dates):
    # 1. Download Fama-French 5-Factor data (brede datumreeks gepakt voor de zekerheid)
    ff_dict = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='1960-01-01', end='2026-12-31')
    ff5 = ff_dict[0]
    ff5.index = ff5.index.to_timestamp().strftime('%Y%m').astype(int)
    
    # 2. Zet returns en datums in een DataFrame
    portfolio_df = pd.DataFrame({
        'Date': pd.Series(dates).astype(int).values,
        'Port_Return': pd.Series(portfolio_returns).values
    })
    
    # 3. Merge op basis van de datums
    merged = pd.merge(portfolio_df, ff5, left_on='Date', right_index=True, how='inner')
    
    if merged.empty:
        print("Error: Merge failed. Check your date formats.")
        return None, None
        
    # 4. Voer de OLS regressie uit
    X = merged[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    X = sm.add_constant(X) 
    Y = merged['Port_Return']
    
    model = sm.OLS(Y, X).fit()
    
    # 5. Pak alleen de Alpha en P-waarde
    alpha = model.params['const']
    p_val = model.pvalues['const']
    
    print(f"\nAlpha (Intercept): {alpha:.6f}")
    print(f"P-value:           {p_val:.6f}")
    
    return alpha, p_val


def evaluate_master_portfolio(feat1, feat2, k, grid_dir, ports_dir, file_name):
    # Let op: de underscores in subdir en ports_csv zijn nu correct toegevoegd
    subdir = f"LME_{feat1}_{feat2}"
    ports_csv = Path(grid_dir) / subdir / f'Selected_Ports_{k}.csv'
    weights_csv = Path(grid_dir) / subdir / f'Selected_Ports_Weights_{k}.csv'
    original_data_csv = Path(ports_dir) / subdir / file_name
    
    # Laad CSV's
    ports = pd.read_csv(ports_csv)
    weights = pd.read_csv(weights_csv)
    original_data = pd.read_csv(original_data_csv)

    # Bereken master portfolio excess returns
    master_portfolio_returns = ports.dot(weights.values).squeeze()

    # Haal de datumkolom op
    date_col = original_data.columns[0]
    dates = original_data[date_col]

    # Voer Fama-French uit op deze vector
    return run_ff5_regression(master_portfolio_returns, dates)

