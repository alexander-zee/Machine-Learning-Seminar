import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm
 
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR   = SCRIPT_DIR.parent.parent
PREPARED_DIR = ROOT_DIR / 'data' / 'prepared'
 
PATH_INPUT  = PREPARED_DIR / 'panel.parquet'
PATH_OUTPUT = PREPARED_DIR / 'panel_clustering_mice.parquet'
 
CHARS = [
    'LME', 'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]
 
 
def run_mice_imputation() -> None:
    print("=" * 65)
    print("   MULTIVARIATE CORRELATION-BASED IMPUTATION (MICE)")
    print("=" * 65)
 
    print(f"Loading {PATH_INPUT.name}...")
    df = pd.read_parquet(PATH_INPUT)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  {len(df):,} rows, {df.shape[1]} columns")
 
    available_chars = [c for c in CHARS if c in df.columns]
    missing = [c for c in CHARS if c not in df.columns]
    if missing:
        print(f"  WARNING: characteristics not found in panel and will be skipped: {missing}")
    print(f"  Imputing: {available_chars}")
 
    imputer = IterativeImputer(
        max_iter          = 10,
        random_state      = 42,
        initial_strategy  = 'median',
    )
 
    imputed_frames = []
    for d in tqdm(sorted(df['date'].unique()), desc="Processing months"):
        month = df[df['date'] == d].copy()
 
        has_missing = month[available_chars].isnull().any(axis=1).any()
        enough_rows = len(month) > len(available_chars) + 10
 
        if has_missing and enough_rows:
            try:
                month[available_chars] = imputer.fit_transform(month[available_chars])
            except Exception:
                # Fallback: fill with column median if imputer fails (singular matrix)
                month[available_chars] = month[available_chars].fillna(
                    month[available_chars].median()
                )
 
        imputed_frames.append(month)
 
    df_out = pd.concat(imputed_frames, ignore_index=True)
 
    # Fill any remaining NaNs (e.g. early IdioVol observations) with cross-sectional midpoint
    remaining = df_out[available_chars].isnull().sum().sum()
    if remaining > 0:
        print(f"  Filling {remaining:,} remaining NaNs with 0.5")
        df_out[available_chars] = df_out[available_chars].fillna(0.5)
 
    df_out.to_parquet(PATH_OUTPUT, index=False)
 
    print("-" * 65)
    print(f"Saved: {PATH_OUTPUT.name}")
    print(f"Rows:  {len(df_out):,}  (input: {len(df):,})  match: {'YES' if len(df_out) == len(df) else 'NO'}")
    print(f"Missing after imputation: {df_out[available_chars].isnull().sum().sum()}")
    print("=" * 65)
 
 
if __name__ == "__main__":
    run_mice_imputation()