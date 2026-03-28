import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm

# Paden instellen
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent.parent
PREPARED_DIR = ROOT_DIR / 'data' / 'prepared'

PATH_ORIGINAL = PREPARED_DIR / 'FINALdataset.parquet'
PATH_YOUR_DATA = PREPARED_DIR / 'panel_benchmark.parquet'
PATH_OUTPUT = PREPARED_DIR / 'panel_clustering_mice.parquet'

def run_mice_imputation():
    print("="*65)
    print("   MULTIVARIATE CORRELATION-BASED IMPUTATION (MICE)")
    print("="*65)

    # 1. Inladen
    print("Laden van datasets...")
    df_orig = pd.read_parquet(PATH_ORIGINAL)
    df_yours = pd.read_parquet(PATH_YOUR_DATA)

    # --- KOLOM-NORMALISATIE ---
    # Zoek de datumkolom van de collega
    date_col_orig = next((c for c in df_orig.columns if 'date' in c.lower() or 'caldt' in c.lower()), df_orig.columns[1])
    # Zoek de PERMNO kolom (hoofdlettergevoeligheid fix)
    permno_col_orig = next((c for c in df_orig.columns if 'permno' in c.lower()), df_orig.columns[0])
    # Zoek de RET kolom (de boosdoener van de vorige error)
    ret_col_orig = next((c for c in df_orig.columns if 'ret' in c.lower()), None)
    
    if not ret_col_orig:
        print("WAARSCHUWING: Geen rendementskolom gevonden in origineel. We gebruiken jouw rendementen.")
        df_orig = df_orig.rename(columns={date_col_orig: 'date', permno_col_orig: 'permno'})
    else:
        df_orig = df_orig.rename(columns={date_col_orig: 'date', permno_col_orig: 'permno', ret_col_orig: 'ret'})

    df_orig['date'] = pd.to_datetime(df_orig['date'])
    df_yours['date'] = pd.to_datetime(df_yours['date'])

    # 2. De JOIN (Exact 3.406.794 rijen behouden)
    print(f"Samenvoegen op basis van collega-template ({len(df_orig):,} rijen)...")
    
    # We pakken de ID's en het rendement van de collega als basis
    base_cols = ['permno', 'date']
    if 'ret' in df_orig.columns: base_cols.append('ret')
    
    # Merge jouw characteristics op zijn template
    df_combined = pd.merge(df_orig[base_cols], 
                           df_yours.drop(columns=['ret', 'yy', 'mm'], errors='ignore'), 
                           on=['permno', 'date'], how='left')

    # 3. MICE Imputatie (Per maand)
    chars = ['LME', 'BEME', 'r12_2', 'OP', 'Investment', 'ST_Rev', 'LT_Rev', 'AC', 'LTurnover']
    available_chars = [c for c in chars if c in df_combined.columns]
    
    print(f"Start MICE correlatie-imputatie op: {available_chars}")
    
    imputed_frames = []
    unique_dates = sorted(df_combined['date'].unique())

    # De imputer: BayesianRidge zorgt voor behoud van variantie en correlatie
    imputer = IterativeImputer(max_iter=10, random_state=42, initial_strategy='median')

    for d in tqdm(unique_dates, desc="Maanden verwerken"):
        month_data = df_combined[df_combined['date'] == d].copy()
        
        # Alleen imputeren als er missende waarden zijn en genoeg data (N > aantal kolommen)
        mask = month_data[available_chars].isnull().any(axis=1)
        if mask.any() and len(month_data) > len(available_chars) + 10:
            try:
                # We trainen de imputer op de data van deze specifieke maand
                month_data[available_chars] = imputer.fit_transform(month_data[available_chars])
            except:
                # Fallback naar mediaan als de matrix singulier is (te weinig unieke data)
                month_data[available_chars] = month_data[available_chars].fillna(month_data[available_chars].median())
        
        imputed_frames.append(month_data)

    df_final = pd.concat(imputed_frames)

    # 4. Final Touch: Alles wat nog mist (bijv. extreme uitschieters) naar 0.5
    df_final[available_chars] = df_final[available_chars].fillna(0.5)
    
    # Opslaan
    df_final.to_parquet(PATH_OUTPUT)
    
    print("-" * 65)
    print(f"SUCCES! Dataset opgeslagen als: {PATH_OUTPUT.name}")
    print(f"Rijen match: {'JA' if len(df_final) == len(df_orig) else 'NEE'}")
    print(f"Totaal rijen: {len(df_final):,}")
    print("="*65)

if __name__ == "__main__":
    run_mice_imputation()