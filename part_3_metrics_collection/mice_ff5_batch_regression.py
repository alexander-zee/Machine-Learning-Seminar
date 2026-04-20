"""
mice_ff5_batch.py — FF5 alpha for MICE RP tree portfolios across all n_features values.
 
Produces a summary table analogous to Table B.1 of Bryzgalova et al. (2025):
 
    Id | n_features | SR | αFF5 [t-stat] | λ0 | λ2
 
Rows are ordered by SR ascending (matching the paper's convention).
 
For the uniform kernel: returns are reconstructed as raw_ports @ saved_betas,
normalised to unit train-window variance before regression so alphas are
comparable across n_features values.
 
For the Gaussian kernel: returns are read from full_fit_detail_k{k}.csv
(pre-computed per-month returns from kernel_full_fit), also normalised.
 
Usage
-----
    python mice_ff5_batch.py
 
Adjust CONFIG below. Outputs:
    <OUTPUT_PATH>/mice_ff5_results_{kernel}_{n_features_label}_k{K}.csv
    <OUTPUT_PATH>/mice_ff5_table_{kernel}_{n_features_label}_k{K}.csv
"""
 
from __future__ import annotations
 
import warnings
from pathlib import Path
 
import numpy as np
import pandas as pd
import statsmodels.api as sm
 
warnings.simplefilter(action='ignore', category=FutureWarning)
 
 
# ── Config ────────────────────────────────────────────────────────────────────
 
ALL_FEATURES = [
    'LME', 'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]
 
N_FEATURES_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
 
K             = 10
N_TRAIN_VALID = 360
Y_MIN, Y_MAX  = 1964, 2016
 
GRID_PATH      = Path('data/results/grid_search/mice_rp_tree')
PORTFOLIO_PATH = Path('data/results/mice_rp_tree_portfolios')
OUTPUT_PATH    = Path('data/results/diagnostics')
PORT_FILE      = 'level_all_excess_combined.csv'
FF5_CSV        = Path('data/raw/F-F_Research_Data_5_Factors_2x3.csv')
 
_ff5_cache: pd.DataFrame | None = None
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _make_subdir(n_features: int) -> str:
    return f"{'_'.join(ALL_FEATURES)}__nf{n_features}"
 
 
def _base_path(grid_dir: Path, subdir: str, kernel_cls=None) -> Path:
    kernel_name = (
        kernel_cls.__name__.lower().replace('kernel', '')
        if kernel_cls is not None
        else 'uniform'
    )
    return Path(grid_dir) / kernel_name / subdir
 
 
def _generate_dates(y_min: int = Y_MIN, y_max: int = Y_MAX) -> np.ndarray:
    dates = []
    for y in range(y_min, y_max + 1):
        for m in range(1, 13):
            dates.append(int(f'{y}{m:02d}'))
    return np.array(dates)
 
 
def _load_ff5() -> pd.DataFrame:
    """Load FF5 factors. Cached after first call. Returns decimal form."""
    global _ff5_cache
    if _ff5_cache is not None:
        return _ff5_cache
 
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
 
    if FF5_CSV.is_file():
        print(f"Loading FF5 factors from {FF5_CSV}...", flush=True)
        ff5 = pd.read_csv(FF5_CSV, index_col=0)
        if not all(c in ff5.columns for c in factor_cols):
            ff5 = pd.read_csv(FF5_CSV, skiprows=4, index_col=0)
            ff5 = ff5[ff5.index.astype(str).str.strip().str.match(r'^\d{6}$')]
            ff5.index = ff5.index.astype(int)
            ff5.index.name = 'Date'
            ff5[factor_cols] = ff5[factor_cols].apply(
                pd.to_numeric, errors='coerce') / 100.0
        else:
            ff5.index = ff5.index.astype(int)
            ff5.index.name = 'Date'
            if ff5[factor_cols].abs().max().max() > 1.0:
                ff5[factor_cols] = ff5[factor_cols].apply(
                    pd.to_numeric, errors='coerce') / 100.0
    else:
        print('FF5 local CSV not found, downloading via pandas_datareader...',
              flush=True)
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
        print(f'  Saved to {FF5_CSV} for future use.', flush=True)
 
    print(f'  FF5 loaded: {len(ff5)} months ({ff5.index[0]} – {ff5.index[-1]})',
          flush=True)
    _ff5_cache = ff5
    return ff5
 
 
def _load_hyperparams(base: Path, k: int, kernel_cls=None) -> dict:
    """
    Read lambda0, lambda2 (and h for non-uniform) from results files.
    For uniform: reads from train_SR_{k}.csv (lambda indices from best fit).
    For Gaussian: reads from full_fit/full_fit_summary_k{k}.csv.
    Returns dict with None values if files are missing.
    """
    if kernel_cls is None:
        hp_path = base / f'best_hyperparams_{k}.csv'
        if not hp_path.exists():
            return {'lambda0': None, 'lambda2': None, 'h': None}
        row = pd.read_csv(hp_path).iloc[0]
        return {
            'lambda0': float(row['lambda0']),
            'lambda2': float(row['lambda2']),
            'h':       None,   # uniform has no bandwidth
        }
    else:
        summary_path = base / 'full_fit' / f'full_fit_summary_k{k}.csv'
        if not summary_path.exists():
            return {'lambda0': None, 'lambda2': None, 'h': None}
        row = pd.read_csv(summary_path).iloc[0]
        return {
            'lambda0': float(row['lambda0']),
            'lambda2': float(row['lambda2']),
            'h': float(row['h']) if 'h' in row and pd.notna(row['h']) else None,
        }
 
 
# ── Core regression for one n_features value ──────────────────────────────────
 
def _regress_one(
    n_features: int,
    ff5: pd.DataFrame,
    all_dates: np.ndarray,
    kernel_cls=None,
    k: int = K,
) -> dict:
    """
    Run FF5 regression for one n_features_per_split value.
 
    Uniform kernel: reconstructs SDF as raw_ports @ saved_betas, then
    normalises to unit train-window variance.
    Gaussian kernel: reads pre-computed returns from full_fit_detail_k{k}.csv,
    then normalises to unit train-window variance.
    """
    subdir     = _make_subdir(n_features)
    base       = _base_path(GRID_PATH, subdir, kernel_cls)
    kernel_name = (
        kernel_cls.__name__.lower().replace('kernel', '')
        if kernel_cls is not None else 'uniform'
    )
 
    base_result = {
        'n_features': n_features,
        'subdir':     subdir,
        'kernel':     kernel_name,
    }
 
    _missing = {
        **base_result, 'status': 'missing',
        'sr': None, 'mean_ret': None, 'std_ret': None,
        'alpha_ff5': None, 'alpha_ff5_tstat': None, 'alpha_ff5_pval': None,
        'r2': None,
        'beta_MktRF': None, 'beta_SMB': None,
        'beta_HML': None, 'beta_RMW': None, 'beta_CMA': None,
        'lambda0': None, 'lambda2': None, 'h': None, 'n_obs': None,
    }
 
    if kernel_cls is None:
        ports_csv   = base / f'Selected_Ports_{k}.csv'
        weights_csv = base / f'Selected_Ports_Weights_{k}.csv'

        if not ports_csv.exists() or not weights_csv.exists():
            return _missing

        ports   = pd.read_csv(ports_csv)
        weights = pd.read_csv(weights_csv).values.flatten()

        sdf_full  = ports.values @ weights
        sdf_test  = sdf_full[N_TRAIN_VALID:]
        test_dates = all_dates[N_TRAIN_VALID : N_TRAIN_VALID + len(sdf_test)]
 
    else:
        # ── Gaussian: read pre-computed returns from kernel_full_fit ──────
        detail_path = base / 'full_fit' / f'full_fit_detail_k{k}.csv'
        if not detail_path.exists():
            return _missing
 
        # Load train returns for normalisation scale
        ports_csv   = base / f'Selected_Ports_{k}.csv'
        weights_csv = base / f'Selected_Ports_Weights_{k}.csv'
        if ports_csv.exists() and weights_csv.exists():
            ports_all = pd.read_csv(ports_csv)
            weights   = pd.read_csv(weights_csv).values.flatten()
            sdf_train = (ports_all.values[:N_TRAIN_VALID]) @ weights
            #scale     = sdf_train.std(ddof=1)
            scale= 1.0
        else:
            scale = 1.0   # fallback: no normalisation if train data unavailable

        detail   = pd.read_csv(detail_path)
        sdf_test = detail['excess_return'].values / scale
 
        test_dates = all_dates[N_TRAIN_VALID : N_TRAIN_VALID + len(sdf_test)]
 
    # ── FF5 regression ────────────────────────────────────────────────────
    port_df = pd.DataFrame({'Date': test_dates.astype(int), 'ret': sdf_test})
    merged  = pd.merge(port_df, ff5, left_on='Date', right_index=True, how='inner')
 
    if merged.empty:
        return {**_missing, 'status': 'merge_failed', 'n_obs': 0}
 
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    X     = sm.add_constant(merged[factor_cols])
    Y     = merged['ret']
    model = sm.OLS(Y, X).fit()
 
    r     = merged['ret']
    std_r = r.std(ddof=1)
    sr    = float(r.mean() / std_r) if std_r > 0 else np.nan
 
    hp = _load_hyperparams(base, k, kernel_cls)
 
    return {
        **base_result,
        'status':          'ok',
        'n_obs':           int(model.nobs),
        'sr':              sr,
        'mean_ret':        float(r.mean()),
        'std_ret':         float(std_r),
        'alpha_ff5':       float(model.params['const']),
        'alpha_ff5_tstat': float(model.tvalues['const']),
        'alpha_ff5_pval':  float(model.pvalues['const']),
        'r2':              float(model.rsquared),
        'beta_MktRF':      float(model.params['Mkt-RF']),
        'beta_SMB':        float(model.params['SMB']),
        'beta_HML':        float(model.params['HML']),
        'beta_RMW':        float(model.params['RMW']),
        'beta_CMA':        float(model.params['CMA']),
        'lambda0':         hp['lambda0'],
        'lambda2':         hp['lambda2'],
        'h':               hp['h'],
    }
 
 
# ── Display table ─────────────────────────────────────────────────────────────
 
def _build_display_table(results_df: pd.DataFrame, kernel_name: str) -> pd.DataFrame:
    """
    Summary table:
        Id | n_features | SR | αFF5 [t-stat] | λ0 | λ2 [| h]
 
    Rows sorted by SR ascending (lowest SR first, as in the paper).
    """
    ok = results_df[results_df['status'] == 'ok'].copy()
    ok = ok.sort_values('sr', ascending=True).reset_index(drop=True)
    ok.index = ok.index + 1
 
    rows = []
    for idx, row in ok.iterrows():
        def fmt(val, fmt_str):
            return format(val, fmt_str) if pd.notna(val) else '—'
 
        alpha_str = (
            f"{fmt(row['alpha_ff5'], '.4f')} [{fmt(row['alpha_ff5_tstat'], '.2f')}]"
            if pd.notna(row['alpha_ff5']) else '—'
        )
 
        entry = {
            'Id':         idx,
            'n_features': int(row['n_features']),
            'SR':         fmt(row['sr'], '.4f'),
            'αFF5 [t]':   alpha_str,
            'λ0':         fmt(row['lambda0'], '.2f'),
            'λ2':         fmt(row['lambda2'], '.2e'),
        }
 
        if kernel_name != 'uniform':
            entry['h'] = fmt(row.get('h'), '.4f') if pd.notna(row.get('h')) else '—'
 
        rows.append(entry)
 
    return pd.DataFrame(rows)
 
 
# ── Main batch runner ─────────────────────────────────────────────────────────
 
def run_mice_ff5_batch(
    n_features_grid: list[int] = N_FEATURES_GRID,
    k: int                     = K,
    kernel_cls                 = None,
    output_path: Path          = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Run FF5 regressions for all n_features values.
 
    Parameters
    ----------
    n_features_grid : list of int — n_features_per_split values to evaluate
    k               : portfolio count to evaluate
    kernel_cls      : kernel class (None → uniform)
    output_path     : directory for output CSVs
 
    Returns
    -------
    DataFrame of numeric results, one row per n_features value
    """
    output_path.mkdir(parents=True, exist_ok=True)
    all_dates   = _generate_dates()
    ff5         = _load_ff5()
    kernel_name = (
        kernel_cls.__name__.lower().replace('kernel', '')
        if kernel_cls is not None else 'uniform'
    )
    nf_label = f"nf{'_'.join(str(n) for n in n_features_grid)}"
 
    print(
        f"\nMICE RP tree FF5 regressions — kernel={kernel_name}, k={k}, "
        f"n_features={n_features_grid}\n",
        flush=True,
    )
 
    records = []
    for nf in n_features_grid:
        row = _regress_one(nf, ff5, all_dates, kernel_cls=kernel_cls, k=k)
        records.append(row)
 
        if row['status'] == 'ok':
            h_part = (
                f"  h={row['h']:.4f}"
                if kernel_name != 'uniform' and row['h'] is not None
                else ''
            )
            print(
                f"  nf={nf:<3}  SR={row['sr']:+.4f}  "
                f"αFF5={row['alpha_ff5']:+.6f} [t={row['alpha_ff5_tstat']:+.2f}]"
                + h_part,
                flush=True,
            )
        else:
            print(f"  nf={nf:<3}  [{row['status']}]", flush=True)
 
    results_df = pd.DataFrame(records)
 
    # Full numeric CSV
    numeric_csv = output_path / f'mice_ff5_results_{kernel_name}_{nf_label}_k{k}.csv'
    results_df.to_csv(numeric_csv, index=False)
    print(f"\nNumeric results → {numeric_csv}", flush=True)
 
    # Display table sorted by SR
    display_df  = _build_display_table(results_df, kernel_name)
    display_csv = output_path / f'mice_ff5_table_{kernel_name}_{nf_label}_k{k}.csv'
    display_df.to_csv(display_csv, index=False)
    print(f"Display table   → {display_csv}", flush=True)
 
    print(f"\n{'─' * 70}")
    print(display_df.to_string(index=False))
    print(f"{'─' * 70}\n")
 
    ok = results_df[results_df['status'] == 'ok']
    if len(ok) > 0:
        sig05 = ok[ok['alpha_ff5_pval'] < 0.05]
        print(f"  Completed:           {len(ok)}/{len(n_features_grid)}")
        print(f"  Missing / failed:    {len(n_features_grid) - len(ok)}")
        print(f"  Mean SR (monthly):   {ok['sr'].mean():.4f}")
        print(f"  Mean αFF5:           {ok['alpha_ff5'].mean():.6f}")
        print(f"  Sig. alpha (p<.05):  {len(sig05)}/{len(ok)}")
 
    return results_df
 
 
if __name__ == '__main__':
    run_mice_ff5_batch()