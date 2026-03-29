import pandas as pd
import numpy as np
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Point to your local FINALdataset.csv (see README); never commit the CSV if it is restricted.
RAW_PATH = _REPO_ROOT / "data" / "raw" / "FINALdataset.csv"
OUTPUT_PATH = _REPO_ROOT / "data" / "prepared" / "panel_benchmark.parquet"

COLUMN_MAP = {
    'PERMNO':      'permno',
    'MthCalDt':    'date',
    'ExRet':      'ret',
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
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover'
    # IdioVol added later once constructed
]

Y_MIN = 1964
Y_MAX = 2016

def convert_quantile(series):
    """
    Rank non-NaN values to [0,1], leave NaN as NaN.
    Computed only among stocks with available values,
    exactly matching R: (rank(na.omit(x))-1)/(length(na.omit(x))-1)
    """
    result = series.copy().astype(float)
    mask   = series.notna()
    valid  = series[mask]
    if len(valid) <= 1:
        return result
    ranks        = valid.rank(method='average')
    result[mask] = (ranks - 1) / (len(valid) - 1)
    # NaN positions untouched
    return result

def prepare_data():
    print("Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    df = df.rename(columns=COLUMN_MAP)

    # ── Parse dates ───────────────────────────────────────────────────────────
    df['date'] = pd.to_datetime(df['date'])
    df['yy']   = df['date'].dt.year
    df['mm']   = df['date'].dt.month

    # ── Filter year range ─────────────────────────────────────────────────────
    df = df[(df['yy'] >= Y_MIN) & (df['yy'] <= Y_MAX)].copy()

    # ── Save raw LME as size BEFORE quantile ranking ──────────────────────────
    # Critical: size needs raw dollar value for value weighting
    # LME as characteristic will be quantile ranked to [0,1]
    df['size'] = df['LME'].copy()

    # ── Only upfront filtering: ret and LME ───────────────────────────────────
    # ret:  useless for every triplet if missing
    # LME:  needed for value weighting in every triplet AND as characteristic
    before = len(df)
    df = df.dropna(subset=['ret', 'LME'])
    after  = len(df)
    print(f"Dropped {before - after} rows missing ret or LME ({after} remaining)")
    # All other characteristics keep NaN through to parquet
    # They will be filtered per triplet later

    # ── Quantile rank all characteristics per month ───────────────────────────
    # Ranks computed among all stocks with that characteristic available
    # regardless of whether other characteristics are present
    # This matches original R behavior exactly
    print("Computing cross-sectional quantile ranks...")
    for feat in CHARACTERISTICS:
        before_nulls = df[feat].isna().sum()
        df[feat] = (
            df.groupby(['yy', 'mm'])[feat]
              .transform(convert_quantile)
        )
        print(f"  Ranked {feat} ({before_nulls} NaN values kept as NaN)")

    # ── Select and order columns ──────────────────────────────────────────────
    cols = (['permno', 'date', 'yy', 'mm', 'ret', 'size']
            + CHARACTERISTICS)
    df = df[cols]

    # ── Save to parquet ───────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values per characteristic:")
    print(df[CHARACTERISTICS].isna().sum())

if __name__ == '__main__':
    prepare_data()
