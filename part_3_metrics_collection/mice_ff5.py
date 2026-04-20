"""
mice_evaluate_master_portfolio.py — FF5 alpha for MICE RP tree portfolios.
 
Mirrors the batch ff5_batch_regression.py approach:
  - Reads FF5 factors from a local CSV (no internet required)
  - Routes to the correct kernel subfolder via _base_path
  - Supports both uniform (Selected_Ports_{k}.csv @ weights) and
    Gaussian (full_fit_detail_k{k}.csv pre-computed returns) cases
  - Subdirectory encodes all_features and n_features_per_split
"""
 
from __future__ import annotations
 
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
import statsmodels.api as sm
 
warnings.simplefilter(action='ignore', category=FutureWarning)
 
# ── Config ────────────────────────────────────────────────────────────────────
 
Y_MIN = 1964
Y_MAX = 2016
 
FF5_CSV = Path('data/raw/F-F_Research_Data_5_Factors_2x3.csv')
 
_ff5_cache: pd.DataFrame | None = None
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _base_path(grid_dir: Path, subdir: str, kernel_cls=None) -> Path:
    """
    Resolve the results directory for a given kernel and subdir.
    kernel_cls=None → 'uniform'.
    """
    kernel_name = (
        kernel_cls.__name__.lower().replace('kernel', '')
        if kernel_cls is not None
        else 'uniform'
    )
    return Path(grid_dir) / kernel_name / subdir
 
 
def _make_subdir(all_features: list, n_features_per_split: int) -> str:
    return f"{'_'.join(all_features)}__nf{n_features_per_split}"
 
 
def _generate_dates(y_min: int = Y_MIN, y_max: int = Y_MAX) -> np.ndarray:
    dates = []
    for y in range(y_min, y_max + 1):
        for m in range(1, 13):
            dates.append(int(f'{y}{m:02d}'))
    return np.array(dates)
 
 
def _load_ff5() -> pd.DataFrame:
    global _ff5_cache
    if _ff5_cache is not None:
        return _ff5_cache

    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    if FF5_CSV.is_file():
        print(f"Loading FF5 factors from {FF5_CSV}...", flush=True)
        # Try reading as our saved format first (no skiprows, index is Date)
        ff5 = pd.read_csv(FF5_CSV, index_col=0)
        # If factor columns aren't present, it's the original French format
        if not all(c in ff5.columns for c in factor_cols):
            ff5 = pd.read_csv(FF5_CSV, skiprows=4, index_col=0)
            ff5 = ff5[ff5.index.astype(str).str.strip().str.match(r'^\d{6}$')]
            ff5.index = ff5.index.astype(int)
            ff5.index.name = 'Date'
            ff5[factor_cols] = ff5[factor_cols].apply(pd.to_numeric, errors='coerce') / 100.0
        else:
            ff5.index = ff5.index.astype(int)
            ff5.index.name = 'Date'
            # Already in decimal form if saved by us — check and skip dividing
            if ff5[factor_cols].abs().max().max() > 1.0:
                ff5[factor_cols] = ff5[factor_cols].apply(pd.to_numeric, errors='coerce') / 100.0
    else:
        print("FF5 local CSV not found, downloading via pandas_datareader...", flush=True)
        import pandas_datareader.data as web
        ff_dict = web.DataReader(
            'F-F_Research_Data_5_Factors_2x3', 'famafrench',
            start='1960-01-01', end='2030-12-31',
        )
        ff5 = ff_dict[0].copy()
        ff5.index = ff5.index.to_timestamp().strftime('%Y%m').astype(int)
        ff5.index.name = 'Date'
        ff5[factor_cols] = ff5[factor_cols] / 100.0
        FF5_CSV.parent.mkdir(parents=True, exist_ok=True)
        ff5.to_csv(FF5_CSV)
        print(f"  Saved to {FF5_CSV} for future use.", flush=True)

    print(f"  FF5 loaded: {len(ff5)} months ({ff5.index[0]} – {ff5.index[-1]})",
          flush=True)
    _ff5_cache = ff5
    return ff5
 
def _run_ff5_regression(
    portfolio_returns: np.ndarray,
    dates: np.ndarray,
) -> tuple[float | None, float | None]:
    """
    Regress portfolio excess returns on FF5 factors.
    Returns (alpha, p_value) or (None, None) if merge fails.
    """
    ff5 = _load_ff5()
 
    port_df = pd.DataFrame({
        'Date':        dates.astype(int),
        'Port_Return': portfolio_returns,
    })
    merged = pd.merge(port_df, ff5, left_on='Date', right_index=True, how='inner')
 
    if merged.empty:
        print("  Warning: FF5 merge failed — check date range.", flush=True)
        return None, None
 
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    X     = sm.add_constant(merged[factor_cols])
    Y     = merged['Port_Return']
    model = sm.OLS(Y, X).fit()
 
    alpha = float(model.params['const'])
    p_val = float(model.pvalues['const'])
    
    print(f"  Alpha: {alpha:.6f}  p={p_val:.4f}  N={int(model.nobs)}", flush=True)
    return alpha, p_val
 
 
# ── Main evaluation function ──────────────────────────────────────────────────
 
def mice_evaluate_master_portfolio(
    all_features: list,
    n_features_per_split: int,
    k: int,
    grid_dir: Path,
    ports_dir: Path,
    file_name: str,
    n_train_valid: int = 360,
    y_min: int = Y_MIN,
    y_max: int = Y_MAX,
    kernel_cls=None,
) -> tuple[float | None, float | None]:
    """
    Evaluate the FF5 alpha of the master portfolio at a given k.
 
    For the uniform kernel: reads Selected_Ports_{k}.csv and
    Selected_Ports_Weights_{k}.csv from the grid results directory,
    then computes test returns as ports @ weights.
 
    For the Gaussian kernel: reads full_fit_detail_k{k}.csv from the
    full_fit/ subfolder, which already contains per-month excess returns
    computed by kernel_full_fit (kernel-weighted rolling fit).
 
    Parameters
    ----------
    all_features         : list of all characteristic names
    n_features_per_split : int, used to locate the correct subdirectory
    k                    : portfolio count to evaluate
    grid_dir             : Path to grid search results root
    ports_dir            : Path to combined portfolio CSVs root
    file_name            : filename of the combined excess return CSV
    n_train_valid        : number of months in the train + validation window
    y_min, y_max         : year range matching the panel used in step 2
    kernel_cls           : kernel class used during pruning (default None → uniform)
 
    Returns
    -------
    (alpha, p_value) or (None, None) if results files are missing
    """
    subdir    = _make_subdir(all_features, n_features_per_split)
    base      = _base_path(grid_dir, subdir, kernel_cls)
    all_dates = _generate_dates(y_min, y_max)
 
    if kernel_cls is None:
        # ── Uniform: reconstruct returns from weights ──────────────────────
        ports_csv   = base / f'Selected_Ports_{k}.csv'
        weights_csv = base / f'Selected_Ports_Weights_{k}.csv'
 
        if not ports_csv.exists() or not weights_csv.exists():
            print(f"  k={k}: results not found at {base}, skipping.", flush=True)
            return None, None
 
        ports   = pd.read_csv(ports_csv)
        weights = pd.read_csv(weights_csv).values.flatten()
 
        # Selected_Ports contains the full period — slice to test window
        ports_test   = ports.iloc[n_train_valid:]
        test_dates   = all_dates[n_train_valid:]
        test_returns = ports_test.values @ weights
 
        if len(test_dates) != len(test_returns):
            raise ValueError(
                f"Date length {len(test_dates)} does not match "
                f"return length {len(test_returns)}. "
                "Check Y_MIN, Y_MAX and n_train_valid."
            )
 
    else:
        # ── Gaussian: use pre-computed per-month returns from full_fit ─────
        kernel_name = kernel_cls.__name__.lower().replace('kernel', '')
        detail_path = base / 'full_fit' / f'full_fit_detail_k{k}.csv'
 
        if not detail_path.exists():
            print(f"  k={k}: full_fit_detail not found at {detail_path}, skipping.",
                  flush=True)
            return None, None
 
        detail       = pd.read_csv(detail_path)
        test_returns = detail['excess_return'].values
 
        # detail CSV starts at n_train_valid; may be shorter if LARS skipped months
        test_dates = all_dates[n_train_valid : n_train_valid + len(test_returns)]
 
    print(
        f"\n  Evaluating k={k}  subdir={subdir}  "
        f"kernel={'uniform' if kernel_cls is None else kernel_cls.__name__}  "
        f"({len(test_returns)} test months)",
        flush=True,
    )
    print(f"  test_returns: mean={test_returns.mean():.6f}  std={test_returns.std():.6f}  min={test_returns.min():.6f}  max={test_returns.max():.6f}")
    return _run_ff5_regression(test_returns, test_dates)