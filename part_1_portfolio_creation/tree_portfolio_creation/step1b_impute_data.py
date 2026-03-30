import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from tqdm import tqdm

# Paden instellen
SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    """
    Robust repo-root detection for runs from VSCode/PowerShell on Windows.
    Walk up from this file until we find project markers.
    """
    for p in [start, *start.parents]:
        if (p / "requirements.txt").is_file() and (p / "part_1_portfolio_creation").is_dir():
            return p
    # Safe fallback for expected layout:
    # <repo>/part_1_portfolio_creation/tree_portfolio_creation/step1b_impute_data.py
    return start.parents[2]


ROOT_DIR = _find_repo_root(SCRIPT_DIR)
PREPARED_DIR = ROOT_DIR / 'data' / 'prepared'

PATH_ORIGINAL = PREPARED_DIR / 'FINALdataset.parquet'
PATH_YOUR_DATA = PREPARED_DIR / 'panel_benchmark.parquet'
PATH_OUTPUT = PREPARED_DIR / 'panel_clustering_mice.parquet'

CHAR_LIST = ['LME', 'BEME', 'r12_2', 'OP', 'Investment', 'ST_Rev', 'LT_Rev', 'AC', 'LTurnover']


def run_mice_imputation():
    print("=" * 65)
    print("   MULTIVARIATE CORRELATION-BASED IMPUTATION (MICE)")
    print("=" * 65)

    print("Laden van datasets...")
    df_yours = pd.read_parquet(PATH_YOUR_DATA)
    if PATH_ORIGINAL.is_file():
        df_orig = pd.read_parquet(PATH_ORIGINAL)
        using_final_template = True
    else:
        # New-user fallback: use benchmark panel as template if FINALdataset.parquet is absent.
        print(
            f"Info: {PATH_ORIGINAL.name} niet gevonden in data/prepared; "
            "gebruik panel_benchmark.parquet als template."
        )
        df_orig = df_yours.copy()
        using_final_template = False

    # --- KOLOM-NORMALISATIE (FINALdataset) ---
    date_col_orig = next(
        (c for c in df_orig.columns if 'date' in c.lower() or 'caldt' in c.lower()),
        df_orig.columns[1],
    )
    permno_col_orig = next(
        (c for c in df_orig.columns if 'permno' in c.lower()),
        df_orig.columns[0],
    )
    ret_col_orig = next((c for c in df_orig.columns if 'ret' in c.lower()), None)

    if not ret_col_orig:
        print("WAARSCHUWING: Geen rendementskolom gevonden in origineel.")
        df_orig = df_orig.rename(columns={date_col_orig: 'date', permno_col_orig: 'permno'})
    else:
        df_orig = df_orig.rename(
            columns={date_col_orig: 'date', permno_col_orig: 'permno', ret_col_orig: 'ret'}
        )

    df_orig['date'] = pd.to_datetime(df_orig['date'])
    df_yours['date'] = pd.to_datetime(df_yours['date'])

    n_dup_o = int(df_orig.duplicated(subset=['permno', 'date']).sum())
    if n_dup_o:
        print(
            f"WAARSCHUWING: {n_dup_o} dubbele (permno, date) in template-dataset — "
            "rijen in output volgen dit bestand exact."
        )

    # --- JOIN: één rij per (permno, date) aan panel-kant; geen cartesische explosie ---
    base_cols = ['permno', 'date']
    if 'ret' in df_orig.columns:
        base_cols.append('ret')

    panel_side = df_yours.drop(columns=['ret', 'yy', 'mm'], errors='ignore')
    n_dup_p = int(panel_side.duplicated(subset=['permno', 'date']).sum())
    if n_dup_p:
        print(
            f"Info: {n_dup_p} dubbele (permno, date) in panel_benchmark — "
            "voor merge: eerste rij per sleutel behouden."
        )
    panel_side = panel_side.drop_duplicates(subset=['permno', 'date'], keep='first')

    template_name = "FINALdataset-template" if using_final_template else "panel_benchmark-template"
    print(f"Samenvoegen op basis van {template_name} ({len(df_orig):,} rijen)...")
    df_combined = df_orig[base_cols].merge(
        panel_side,
        on=['permno', 'date'],
        how='left',
        validate='many_to_one',
    )

    if len(df_combined) != len(df_orig):
        raise ValueError(
            f"Rijenaantal na merge ({len(df_combined):,}) wijkt af van template "
            f"({len(df_orig):,}). Controleer sleutels en duplicaten."
        )

    df_combined = df_combined.reset_index(drop=True)

    available_chars = [c for c in CHAR_LIST if c in df_combined.columns]
    if not available_chars:
        raise ValueError("Geen van de verwachte characteristic-kolommen staat in de merge-output.")

    print(f"MICE (per kalendermaand, cross-sectioneel) op: {available_chars}")

    unique_dates = sorted(df_combined['date'].unique())

    # Per maand: zelfde rij-volgorde als FINAL × merge; geen concat op datum (die herschikt rijen).
    for d in tqdm(unique_dates, desc="Maanden verwerken"):
        idx = df_combined['date'] == d
        X = df_combined.loc[idx, available_chars].copy()

        # Kolommen die deze maand nergens geobserveerd zijn: geen MICE mogelijk → neutraal midden quantiel.
        all_nan_cols = [c for c in available_chars if X[c].notna().sum() == 0]
        for c in all_nan_cols:
            X[c] = 0.5

        fit_cols = [c for c in available_chars if c not in all_nan_cols]
        if not fit_cols:
            df_combined.loc[idx, available_chars] = X
            continue

        need_impute = X[fit_cols].isnull().any(axis=1)
        # Minimaal voldoende cross-sectie om MICE te stabiliseren
        if need_impute.any() and len(X) > len(fit_cols) + 10:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    imputer = IterativeImputer(
                        max_iter=10,
                        random_state=42,
                        initial_strategy='median',
                    )
                    X[fit_cols] = imputer.fit_transform(X[fit_cols])
            except (ValueError, RuntimeError) as e:
                print(f"  MICE fallback voor {d.date()}: {e}")
                X[fit_cols] = X[fit_cols].fillna(X[fit_cols].median())

        df_combined.loc[idx, available_chars] = X

    df_combined[available_chars] = df_combined[available_chars].fillna(0.5)

    df_combined.to_parquet(PATH_OUTPUT, index=False)

    print("-" * 65)
    print(f"SUCCES! Dataset opgeslagen als: {PATH_OUTPUT.name}")
    print(f"Rijen match template: {'JA' if len(df_combined) == len(df_orig) else 'NEE'}")
    print(f"Totaal rijen: {len(df_combined):,}")
    print("=" * 65)


# Entry point used by main.py pipeline
impute_characteristics = run_mice_imputation

if __name__ == "__main__":
    run_mice_imputation()
