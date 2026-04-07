import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web
from pathlib import Path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

Y_MIN = 1964
Y_MAX = 2016


def generate_dates(y_min: int = Y_MIN, y_max: int = Y_MAX) -> np.ndarray:
    dates = []
    for y in range(y_min, y_max + 1):
        for m in range(1, 13):
            dates.append(int(f'{y}{m:02d}'))
    return np.array(dates)


def run_ff5_regression(portfolio_returns, dates):
    ff_dict = web.DataReader(
        'F-F_Research_Data_5_Factors_2x3', 'famafrench',
        start='1960-01-01', end='2026-12-31'
    )
    ff5 = ff_dict[0]
    ff5.index = ff5.index.to_timestamp().strftime('%Y%m').astype(int)

    portfolio_df = pd.DataFrame({
        'Date':        np.array(dates, dtype=int),
        'Port_Return': np.array(portfolio_returns),
    })

    merged = pd.merge(portfolio_df, ff5,
                      left_on='Date', right_index=True, how='inner')

    if merged.empty:
        print("Error: Merge failed — check date range vs FF5 availability.")
        return None, None

    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    merged[factor_cols] = merged[factor_cols] / 100.0

    X = sm.add_constant(merged[factor_cols])
    Y = merged['Port_Return']

    model = sm.OLS(Y, X).fit()

    alpha = model.params['const']
    p_val = model.pvalues['const']

    print(f"\nAlpha (intercept): {alpha:.6f}")
    print(f"P-value:           {p_val:.6f}")
    print(f"N obs:             {int(model.nobs)}")

    return alpha, p_val


def evaluate_master_portfolio(feat1, feat2, k, grid_dir, ports_dir,
                               file_name, n_train_valid=360,
                               y_min=Y_MIN, y_max=Y_MAX):
    """
    Compute FF5 alpha and p-value for the master portfolio on the test window.

    Parameters
    ----------
    feat1, feat2   : characteristic names (e.g. 'OP', 'Investment')
    k              : portfolio count (e.g. 10)
    grid_dir       : path to grid search results
    ports_dir      : path to portfolio returns
    file_name      : filename of the combined excess returns CSV
    n_train_valid  : months in train+valid window (default 360)
    y_min, y_max   : year range used during portfolio creation
    """
    subdir      = f"LME_{feat1}_{feat2}"
    ports_csv   = Path(grid_dir) / subdir / f'Selected_Ports_{k}.csv'
    weights_csv = Path(grid_dir) / subdir / f'Selected_Ports_Weights_{k}.csv'

    ports   = pd.read_csv(ports_csv)
    weights = pd.read_csv(weights_csv).values.flatten()

    all_dates  = generate_dates(y_min, y_max)
    ports_test = ports.iloc[n_train_valid:]
    test_dates = all_dates[n_train_valid:]

    test_returns = ports_test.values @ weights

    if len(test_dates) != len(test_returns):
        raise ValueError(
            f"Date length {len(test_dates)} does not match "
            f"return length {len(test_returns)}. "
            "Check Y_MIN, Y_MAX and n_train_valid."
        )

    print(f"\nEvaluating portfolio  LME_{feat1}_{feat2}  k={k}  ({len(test_returns)} test months)")
    return run_ff5_regression(test_returns, test_dates)