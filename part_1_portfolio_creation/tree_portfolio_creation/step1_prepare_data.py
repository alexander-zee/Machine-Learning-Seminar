import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path('data/raw/FINALdataset.csv')
OUTPUT_PATH = Path('data/prepared/panel.parquet')

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FF3_CSV = _REPO_ROOT / 'paper_data' / 'factor' / 'ff3_factors.csv'

COLUMN_MAP = {
    'PERMNO':      'permno',
    'MthCalDt':    'date',
    'ExRet':       'ret',
    'LME':         'LME',
    'BEME':        'BEME',
    'OP':          'OP',
    'Investment':  'Investment',
    'Accrual':     'AC',
    'r12_2':       'r12_2',
    'ST_Rev':      'ST_Rev',
    'LT_Rev':      'LT_Rev',
    'Lturnover':   'LTurnover',
}

# All characteristics that will be quantile ranked
CHARACTERISTICS = [
    'LME', 'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]

Y_MIN = 1964
Y_MAX = 2016

# IdioVol: std of residuals from FF3 regression of excess returns (past window, no look-ahead)
IDIOVOL_WINDOW = 60
IDIOVOL_MIN_PERIODS = 36
_FF3_COLS = ['Mkt-RF', 'SMB', 'HML']


def apply_beme_december_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace monthly BEME with an annual snapshot based on previous December.

    Rationale
    ---------
    Bryzgalova et al. and related AP-tree work define BEME using accounting data with
    market equity as of December t-1, and use characteristics known at t-1 for returns in t.
    Using rapidly-updating same-month BEME can inject look-ahead-like timing artifacts.

    Construction
    ------------
    For each stock (permno) and calendar year t, use BEME observed in December of t-1 for
    all months in year t.
    """
    if "BEME" not in df.columns:
        return df

    out = df.copy()
    before_non_na = int(out["BEME"].notna().sum())

    dec_snap = out.loc[out["mm"] == 12, ["permno", "yy", "BEME"]].copy()
    dec_snap["yy"] = dec_snap["yy"] + 1
    dec_snap = dec_snap.rename(columns={"BEME": "BEME_dec_lag"})

    out = out.merge(dec_snap, on=["permno", "yy"], how="left")
    out["BEME"] = out["BEME_dec_lag"]
    out = out.drop(columns=["BEME_dec_lag"])

    after_non_na = int(out["BEME"].notna().sum())
    print(
        "Applied BEME annual timing: using December t-1 snapshot for year t "
        f"(non-NaN before={before_non_na:,}, after={after_non_na:,})"
    )
    return out


def _load_ff3_monthly() -> pd.DataFrame:
    """
    Monthly Fama–French 3 factors, index int yyyymm, columns in decimals (not %).
    Tries pandas_datareader first, then paper_data/factor/ff3_factors.csv.
    """
    try:
        import pandas_datareader.data as web
        warnings.simplefilter(action='ignore', category=FutureWarning)
        ff_dict = web.DataReader(
            'F-F_Research_Data_Factors', 'famafrench',
            start='1926-01-01', end='2030-12-31',
        )
        ff3 = ff_dict[0].copy()
        ff3.index = ff3.index.to_timestamp().strftime('%Y%m').astype(int)
        for c in _FF3_COLS:
            ff3[c] = ff3[c] / 100.0
        return ff3[_FF3_COLS]
    except Exception as e:
        if not _FF3_CSV.is_file():
            raise RuntimeError(
                "Could not load FF3 factors (network/datareader) and "
                f"missing local file {_FF3_CSV}. Original error: {e}"
            ) from e
        raw = pd.read_csv(_FF3_CSV)
        ff3 = raw.set_index('yyyymm')
        for c in _FF3_COLS:
            ff3[c] = ff3[c].astype(float) / 100.0
        return ff3[_FF3_COLS]


def _roll_idiovol_one_stock(
    g: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.Series:
    """
    For each month t, regress excess returns in (t-window, t) — excluding t —
    on FF3; IdioVol_t = sample std of residuals (ddof=1).
    """
    g = g.sort_values('date')
    y = g['ret'].to_numpy(dtype=float)
    X = g[_FF3_COLS].to_numpy(dtype=float)
    n = len(g)
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - window), i
        if hi - lo < min_periods:
            continue
        yw = y[lo:hi]
        Xw = X[lo:hi]
        mask = np.isfinite(yw) & np.isfinite(Xw).all(axis=1)
        yw = yw[mask]
        Xw = Xw[mask]
        if len(yw) < min_periods:
            continue
        Xd = np.column_stack([np.ones(len(Xw)), Xw])
        beta, *_ = np.linalg.lstsq(Xd, yw, rcond=None)
        resid = yw - Xd @ beta
        out[i] = float(np.std(resid, ddof=1))
    return pd.Series(out, index=g.index)


def attach_idiovol_ff3(
    df: pd.DataFrame,
    window: int = IDIOVOL_WINDOW,
    min_periods: int = IDIOVOL_MIN_PERIODS,
) -> pd.DataFrame:
    """Merge FF3, compute IdioVol per stock-month, drop factor columns."""
    ff3 = _load_ff3_monthly()
    df = df.copy()
    df['_ym'] = df['yy'] * 100 + df['mm']
    for c in _FF3_COLS:
        df[c] = df['_ym'].map(ff3[c])
    df = df.sort_values(['permno', 'date']).reset_index(drop=True)
    print(
        f"Computing IdioVol (FF3 residual std, window={window}, "
        f"min_periods={min_periods}, past months only)..."
    )
    df['IdioVol'] = df.groupby('permno', group_keys=False).apply(
        lambda g: _roll_idiovol_one_stock(g, window, min_periods)
    )
    df = df.drop(columns=['_ym'] + _FF3_COLS)
    return df


def convert_quantile(series):
    """
    Rank non-NaN values to [0,1], leave NaN as NaN.
    Computed only among stocks with available values,
    exactly matching R: (rank(na.omit(x))-1)/(length(na.omit(x))-1)
    """
    result = series.copy().astype(float)
    mask = series.notna()
    valid = series[mask]
    if len(valid) <= 1:
        return result
    ranks = valid.rank(method='average')
    result[mask] = (ranks - 1) / (len(valid) - 1)
    return result


def prepare_data(*, use_beme_december_lag: bool = True):
    print("Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    df = df.rename(columns=COLUMN_MAP)

    df['date'] = pd.to_datetime(df['date'])
    df['yy'] = df['date'].dt.year
    df['mm'] = df['date'].dt.month

    df = df[(df['yy'] >= Y_MIN) & (df['yy'] <= Y_MAX)].copy()

    df['size'] = df['LME'].copy()

    if use_beme_december_lag:
        df = apply_beme_december_lag(df)

    df = attach_idiovol_ff3(df)

    before = len(df)
    df = df.dropna(subset=['ret', 'LME'])
    after = len(df)
    print(f"Dropped {before - after} rows missing ret or LME ({after} remaining)")

    print("Computing cross-sectional quantile ranks...")
    for feat in CHARACTERISTICS:
        before_nulls = df[feat].isna().sum()
        df[feat] = (
            df.groupby(['yy', 'mm'])[feat]
            .transform(convert_quantile)
        )
        print(f"  Ranked {feat} ({before_nulls} NaN values kept as NaN)")

    cols = (['permno', 'date', 'yy', 'mm', 'ret', 'size'] + CHARACTERISTICS)
    df = df[cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("\nMissing values per characteristic:")
    print(df[CHARACTERISTICS].isna().sum())

def build_state_variables(
    final_dataset_path: Path,
    output_path: Path,
    date_col: str = 'MthCalDt',
    state_cols: list = ['svar', 'DEF', 'TMS']
):
    """
    Extract monthly state variables from the long-format FinalDataset.csv.
    Deduplicates on date — all stocks share the same macro value per month.
    
    Saves a small CSV: date + state_cols, one row per month.
    """
    df = pd.read_csv(final_dataset_path, usecols=[date_col] + state_cols)
    
    # Deduplicate — any row per month is fine since values are identical
    monthly = df.drop_duplicates(subset=date_col).sort_values(date_col)
    monthly = monthly.reset_index(drop=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_path, index=False)
    print(f"Saved {len(monthly)} monthly observations to {output_path}")
    return monthly

if __name__ == '__main__':
    prepare_data()
