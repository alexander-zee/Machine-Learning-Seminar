import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web
from pathlib import Path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

Y_MIN = 1964
Y_MAX = 2016

# One download per process — ``run_ff5_regression_detailed`` is called many times from table exporters.
_ff5_research_panel_cache: pd.DataFrame | None = None
_ff3_research_panel_cache: pd.DataFrame | None = None
_ff11_research_panel_cache: pd.DataFrame | None = None

FF11_FACTORS = [
    "Mkt-RF",
    "LME",
    "BEME",
    "OP",
    "Investment",
    "r12_2",
    "ST_REV",
    "LT_REV",
    "AC",
    "IdioVol",
    "Lturnover",
]

def load_xsf_research_panel():
    """
    Cross-sectional factor model:
    Market + Size + Investment + Operating Profitability
    extracted from FF11 tradable factor file.
    """

    df = load_ff11_research_panel().copy()

    # keep only needed factors
    cols = ["Mkt-RF", "LME", "Investment", "OP"]

    df = df[cols]

    return df

def load_ff5_research_panel() -> pd.DataFrame:
    """
    Fama–French 5 (2x3) monthly factors, index = YYYYMM int.
    Fetched at most once per Python process; reused by all FF5/CAPM regressions below.
    """
    global _ff5_research_panel_cache
    if _ff5_research_panel_cache is not None:
        return _ff5_research_panel_cache
    ff_dict = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3",
        "famafrench",
        start="1960-01-01",
        end="2026-12-31",
    )
    ff5 = ff_dict[0]
    ff5.index = ff5.index.to_timestamp().strftime("%Y%m").astype(int)
    _ff5_research_panel_cache = ff5
    return ff5

def load_ff3_research_panel() -> pd.DataFrame:
    """
    Fama–French 3-factor monthly factors (Mkt-RF, SMB, HML), index = YYYYMM int.
    Fetched at most once per Python process.
    """
    global _ff3_research_panel_cache
    if _ff3_research_panel_cache is not None:
        return _ff3_research_panel_cache
    ff_dict = web.DataReader(
        "F-F_Research_Data_Factors",
        "famafrench",
        start="1960-01-01",
        end="2026-12-31",
    )
    ff3 = ff_dict[0]
    ff3.index = ff3.index.to_timestamp().strftime("%Y%m").astype(int)
    _ff3_research_panel_cache = ff3
    return ff3


def load_ff11_research_panel(path="data/factor/tradable_factors.csv") -> pd.DataFrame:
    """
    Load the paper-provided FF11 (tradable factor) panel.

    This replaces any manual construction of characteristic-sorted factors.

    Expected format:
    - One column = Date (YYYYMM or datetime)
    - Remaining columns = factor-mimicking long–short returns
    - Optional RF column (will be dropped if present)
    """

    global _ff11_research_panel_cache
    if _ff11_research_panel_cache is not None:
        return _ff11_research_panel_cache

    df = pd.read_csv(path)

    # --- standardize date ---
    if "Date" not in df.columns:
        raise ValueError("tradable_factors.csv must contain a 'Date' column")

    df["Date"] = df["Date"].astype(int)
    df = df.set_index("Date").sort_index()

    # --- drop RF if present ---
    if "rf" in df.columns:
        df = df.drop(columns=["rf"])

    _ff11_research_panel_cache = df
    return df

def clear_ff5_research_panel_cache() -> None:
    """Tests or long-running jobs can reset the cache if needed."""
    global _ff5_research_panel_cache
    _ff5_research_panel_cache = None

def clear_all_panel_caches() -> None:
    """Reset all factor-panel caches (FF3, FF5, FF11)."""
    global _ff3_research_panel_cache, _ff5_research_panel_cache, _ff11_research_panel_cache
    _ff3_research_panel_cache = None
    _ff5_research_panel_cache = None
    _ff11_research_panel_cache = None

def generate_dates(y_min: int = Y_MIN, y_max: int = Y_MAX) -> np.ndarray:
    dates = []
    for y in range(y_min, y_max + 1):
        for m in range(1, 13):
            dates.append(int(f'{y}{m:02d}'))
    return np.array(dates)


def run_ff5_regression(portfolio_returns, dates):
    ff5 = load_ff5_research_panel()

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


def run_capm_regression(portfolio_returns, dates):
    """
    CAPM: regress master-portfolio (excess) returns on Mkt-RF only.
    FF factors are in percent per month in the raw table; we divide by 100.
    Returns (alpha, beta_mkt, p_alpha, p_beta, nobs) or Nones if merge fails.
    """
    ff_dict = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3",
        "famafrench",
        start="1960-01-01",
        end="2026-12-31",
    )
    ff5 = ff_dict[0]
    ff5.index = ff5.index.to_timestamp().strftime("%Y%m").astype(int)

    portfolio_df = pd.DataFrame(
        {
            "Date": np.array(dates, dtype=int),
            "Port_Return": np.array(portfolio_returns),
        }
    )

    merged = pd.merge(portfolio_df, ff5, left_on="Date", right_index=True, how="inner")
    if merged.empty:
        print("Error: CAPM merge failed — check date range vs FF availability.")
        return None, None, None, None, None

    merged["Mkt-RF"] = merged["Mkt-RF"] / 100.0
    X = sm.add_constant(merged[["Mkt-RF"]])
    Y = merged["Port_Return"]
    model = sm.OLS(Y, X).fit()
    alpha = float(model.params["const"])
    beta = float(model.params["Mkt-RF"])
    p_alpha = float(model.pvalues["const"])
    p_beta = float(model.pvalues["Mkt-RF"])
    nobs = int(model.nobs)
    return alpha, beta, p_alpha, p_beta, nobs


def run_ff5_regression_detailed(portfolio_returns, dates):
    """
    Full FF5 regression; returns dict of alpha, factor loadings, p-values, R2, nobs.
    """
    ff5 = load_ff5_research_panel()

    portfolio_df = pd.DataFrame(
        {
            "Date": np.array(dates, dtype=int),
            "Port_Return": np.array(portfolio_returns),
        }
    )

    merged = pd.merge(portfolio_df, ff5, left_on="Date", right_index=True, how="inner")
    if merged.empty:
        return None

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    merged[factor_cols] = merged[factor_cols] / 100.0

    X = sm.add_constant(merged[factor_cols])
    Y = merged["Port_Return"]
    model = sm.OLS(Y, X).fit()

    out = {
        "alpha": float(model.params["const"]),
        "se_alpha": float(model.bse["const"]),
        "t_alpha": float(model.tvalues["const"]),
        "p_alpha": float(model.pvalues["const"]),
        "r2": float(model.rsquared),
        "nobs": int(model.nobs),
    }
    for c in factor_cols:
        out[f"beta_{c}"] = float(model.params[c])
        out[f"p_{c}"] = float(model.pvalues[c])
    return out

def run_ff3_regression(portfolio_returns, dates):
    """
    Fama–French 3-factor regression (Mkt-RF, SMB, HML).
 
    Raw FF3 factors are in percent per month; divided by 100 before regression.
    Returns (alpha, p_alpha) or (None, None) if the date-merge fails.
    """
    ff3 = load_ff3_research_panel()
 
    portfolio_df = pd.DataFrame(
        {
            "Date": np.array(dates, dtype=int),
            "Port_Return": np.array(portfolio_returns),
        }
    )
 
    merged = pd.merge(portfolio_df, ff3, left_on="Date", right_index=True, how="inner")
    if merged.empty:
        print("Error: FF3 merge failed — check date range vs FF3 availability.")
        return None, None
 
    factor_cols = ["Mkt-RF", "SMB", "HML"]
    merged[factor_cols] = merged[factor_cols] / 100.0
 
    X = sm.add_constant(merged[factor_cols])
    Y = merged["Port_Return"]
    model = sm.OLS(Y, X).fit()
 
    alpha = float(model.params["const"])
    p_val = float(model.pvalues["const"])
 
    print(f"\nFF3  Alpha (intercept): {alpha:.6f}")
    print(f"FF3  P-value:           {p_val:.6f}")
    print(f"FF3  N obs:             {int(model.nobs)}")
 
    return alpha, p_val
 
 
def run_ff3_regression_detailed(portfolio_returns, dates):
    """
    Full FF3 regression; returns a dict with alpha, factor loadings,
    p-values, R², and nobs — mirroring ``run_ff5_regression_detailed``.
 
    Keys
    ----
    alpha, se_alpha, t_alpha, p_alpha, r2, nobs,
    beta_Mkt-RF, p_Mkt-RF, beta_SMB, p_SMB, beta_HML, p_HML
    """
    ff3 = load_ff3_research_panel()
 
    portfolio_df = pd.DataFrame(
        {
            "Date": np.array(dates, dtype=int),
            "Port_Return": np.array(portfolio_returns),
        }
    )
 
    merged = pd.merge(portfolio_df, ff3, left_on="Date", right_index=True, how="inner")
    if merged.empty:
        return None
 
    factor_cols = ["Mkt-RF", "SMB", "HML"]
    merged[factor_cols] = merged[factor_cols] / 100.0
 
    X = sm.add_constant(merged[factor_cols])
    Y = merged["Port_Return"]
    model = sm.OLS(Y, X).fit()
 
    out = {
        "alpha":    float(model.params["const"]),
        "se_alpha": float(model.bse["const"]),
        "t_alpha":  float(model.tvalues["const"]),
        "p_alpha":  float(model.pvalues["const"]),
        "r2":       float(model.rsquared),
        "nobs":     int(model.nobs),
    }
    for c in factor_cols:
        out[f"beta_{c}"] = float(model.params[c])
        out[f"p_{c}"]    = float(model.pvalues[c])
    return out

def run_ff11_regression(portfolio_returns, dates):

    panel = load_ff11_research_panel()

    portfolio_df = pd.DataFrame({
        "Date": np.array(dates, dtype=int),
        "Port_Return": np.array(portfolio_returns),
    })

    merged = pd.merge(
        portfolio_df,
        panel,
        left_on="Date",
        right_index=True,
        how="inner",
    )

    if merged.empty:
        print("FF11 merge failed.")
        return None, None

    X = sm.add_constant(merged[FF11_FACTORS])
    Y = merged["Port_Return"]

    model = sm.OLS(Y, X).fit()

    print("\nFF11 regression (paper tradable factors)")
    print("alpha:", float(model.params["const"]))
    print("p-value:", float(model.pvalues["const"]))
    print("N:", int(model.nobs))

    return float(model.params["const"]), float(model.pvalues["const"])

def run_ff11_regression_detailed(portfolio_returns, dates):

    panel = load_ff11_research_panel()

    portfolio_df = pd.DataFrame({
        "Date": np.array(dates, dtype=int),
        "Port_Return": np.array(portfolio_returns),
    })

    merged = pd.merge(portfolio_df, panel,
                      left_on="Date", right_index=True, how="inner")

    if merged.empty:
        return None

    X = sm.add_constant(merged[FF11_FACTORS])
    Y = merged["Port_Return"]

    model = sm.OLS(Y, X).fit()

    out = {
        "alpha": float(model.params["const"]),
        "se_alpha": float(model.bse["const"]),
        "t_alpha": float(model.tvalues["const"]),
        "p_alpha": float(model.pvalues["const"]),
        "r2": float(model.rsquared),
        "nobs": int(model.nobs),
    }

    for c in FF11_FACTORS:
        out[f"beta_{c}"] = float(model.params[c])
        out[f"p_{c}"] = float(model.pvalues[c])

    return out


def run_xsf_regression(portfolio_returns, dates):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    panel = load_xsf_research_panel()

    portfolio_df = pd.DataFrame({
        "Date": np.array(dates, dtype=int),
        "Ret": np.array(portfolio_returns),
    })

    merged = pd.merge(
        portfolio_df,
        panel,
        left_on="Date",
        right_index=True,
        how="inner",
    )

    if merged.empty:
        raise ValueError("No overlap between returns and xsf factors")

    factor_cols = ["Mkt-RF", "LME", "Investment", "OP"]

    X = sm.add_constant(merged[factor_cols])
    Y = merged["Ret"]

    # HAC is strongly recommended
    model = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    return {
        "model": "xsf",
        "alpha": float(model.params["const"]),
        "t_alpha": float(model.tvalues["const"]),
        "p_alpha": float(model.pvalues["const"]),
        "r2": float(model.rsquared),
        "nobs": int(model.nobs),
    }



def load_master_test_returns(
    feat1,
    feat2,
    k,
    grid_dir,
    ports_dir,
    file_name,
    n_train_valid=360,
    y_min=Y_MIN,
    y_max=Y_MAX,
):
    """
    Master (SDF) portfolio returns on the hold-out window: Selected_Ports_k @ weights_k.

    Returns (test_returns, test_dates_yyyymm) aligned with Fama–French monthly dates.
    """
    from part_3_metrics_collection.pick_best_lambdas import pruning_results_base

    subdir = f"LME_{feat1}_{feat2}"
    gbase = pruning_results_base(grid_dir, feat1, feat2)
    ports_csv = gbase / f"Selected_Ports_{k}.csv"
    weights_csv = gbase / f"Selected_Ports_Weights_{k}.csv"
    all_ports_path = Path(ports_dir) / subdir / file_name

    ports_sel = pd.read_csv(ports_csv)
    weights = pd.read_csv(weights_csv).values.flatten()
    all_ports = pd.read_csv(all_ports_path)

    ports_test = all_ports.iloc[n_train_valid:][ports_sel.columns]
    test_returns = ports_test.values @ weights

    all_dates = generate_dates(y_min, y_max)
    test_dates = all_dates[n_train_valid:]

    if len(test_dates) != len(test_returns):
        raise ValueError(
            f"Date length {len(test_dates)} does not match "
            f"return length {len(test_returns)}. "
            "Check Y_MIN, Y_MAX and n_train_valid."
        )

    return test_returns, test_dates

def run_factor_regression(portfolio_returns, dates, model="FF5"):
    """
    Unified regression interface for FF3 / FF5 / FF11.

    Parameters
    ----------
    portfolio_returns : array-like
    dates : array-like (YYYYMM int)
    model : str
        "FF3", "FF5", or "FF11"

    Returns
    -------
    dict with alpha, p-value, betas, r2, nobs
    """

    portfolio_df = pd.DataFrame({
        "Date": np.array(dates, dtype=int),
        "Port_Return": np.array(portfolio_returns),
    })

    # -----------------------------
    # Load factor panel
    # -----------------------------
    if model == "FF3":
        from part_3_metrics_collection.ff5 import load_ff3_research_panel
        panel = load_ff3_research_panel()

        factor_cols = ["Mkt-RF", "SMB", "HML"]

    elif model == "FF5":
        from part_3_metrics_collection.ff5 import load_ff5_research_panel
        panel = load_ff5_research_panel()

        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    elif model == "FF11":
        from part_3_metrics_collection.ff5 import load_ff11_research_panel
        panel = load_ff11_research_panel()

        # IMPORTANT: drop RF if present
        panel = panel.copy()
        if "rf" in panel.columns:
            panel = panel.drop(columns=["rf"])

        factor_cols = list(panel.columns)

    else:
        raise ValueError("model must be 'FF3', 'FF5', or 'FF11'")

    # -----------------------------
    # Merge
    # -----------------------------
    merged = pd.merge(
        portfolio_df,
        panel,
        left_on="Date",
        right_index=True,
        how="inner",
    )

    if merged.empty:
        raise ValueError("No overlap between portfolio returns and factor data")

    # -----------------------------
    # Regression
    # -----------------------------
    X = sm.add_constant(merged[factor_cols])
    Y = merged["Port_Return"]

    res = sm.OLS(Y, X).fit()

    return {
        "model": model,
        "alpha": float(res.params["const"]),
        "t_alpha": float(res.tvalues["const"]),
        "p_alpha": float(res.pvalues["const"]),
        "r2": float(res.rsquared),
        "nobs": int(res.nobs),
    }

def evaluate_master_portfolio(
    feat1,
    feat2,
    k,
    grid_dir,
    ports_dir,
    file_name,
    n_train_valid=360,
    y_min=Y_MIN,
    y_max=Y_MAX,
):
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
    test_returns, test_dates = load_master_test_returns(
        feat1,
        feat2,
        k,
        grid_dir,
        ports_dir,
        file_name,
        n_train_valid=n_train_valid,
        y_min=y_min,
        y_max=y_max,
    )
    print(
        f"\nEvaluating portfolio  LME_{feat1}_{feat2}  k={k}  ({len(test_returns)} test months)"
    )
    return run_ff5_regression(test_returns, test_dates)