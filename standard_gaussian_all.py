from itertools import combinations
from pathlib import Path
from multiprocessing import Pool
import traceback
import pandas as pd

from part_2_AP_pruning.AP_Pruning import AP_Pruning
from part_2_AP_pruning.kernels.gaussian import GaussianKernel
from part_2_AP_pruning.lasso_kernel_full_fit import kernel_full_fit
from part_3_metrics_collection.pick_best_lambdas import pick_best_lambda_kernel

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
CHARACTERISTICS = [
    'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]  # LME is always feat1

LAMBDA0      = [0.5, 0.55, 0.6]
LAMBDA2      = [10**-7, 10**-7.25, 10**-7.5]
K_MIN        = 5
K_MAX        = 50
PORT_N       = 10
N_BANDWIDTHS = 5
N_WORKERS    = 8

TREE_PORT_PATH   = Path('data/results/tree_portfolios')
GRID_SEARCH_PATH = Path('data/results/grid_search/tree')
PORT_FILE_NAME   = 'level_all_excess_combined_filtered.csv'

PAIRS = list(combinations(CHARACTERISTICS, 2))

PROGRESS_PATH = GRID_SEARCH_PATH / 'gaussian' / 'progress_standard_gaussian.csv'
SUMMARY_PATH  = GRID_SEARCH_PATH / 'gaussian' / 'all_cross_sections_summary_standard_gaussian.csv'


def load_progress():
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PROGRESS_PATH.exists():
        df = pd.read_csv(PROGRESS_PATH)
        df['error'] = df['error'].astype(object)
        return df
    df = pd.DataFrame([
        {'feat1': f1, 'feat2': f2,
         'cross_section': f'LME_{f1}_{f2}',
         'status': 'pending',
         'test_SR': None, 'valid_SR': None,
         'lambda0': None, 'lambda2': None, 'h': None,
         'months_used': None, 'error': None}
        for f1, f2 in PAIRS
    ])
    df['error'] = df['error'].astype(object)
    df.to_csv(PROGRESS_PATH, index=False)
    print(f"Progress file created: {PROGRESS_PATH}")
    return df


def save_progress(df):
    df.to_csv(PROGRESS_PATH, index=False)


def run_one(args):
    feat1, feat2, bandwidths, n_bandwidths = args
    subdir = f'LME_{feat1}_{feat2}'

    try:
        print(f"  [{subdir}] Starting...", flush=True)

        # Step 1: Grid search
        AP_Pruning(
            feat1=feat1, feat2=feat2,
            input_path=TREE_PORT_PATH,
            input_file_name=PORT_FILE_NAME,
            output_path=GRID_SEARCH_PATH,
            n_train_valid=360, cvN=3, runFullCV=False,
            kmin=K_MIN, kmax=K_MAX,
            RunParallel=False, ParallelN=10, IsTree=True,
            lambda0=LAMBDA0, lambda2=LAMBDA2,
            kernel_cls=GaussianKernel,
            state=_state,
            n_bandwidths=N_BANDWIDTHS,
        )

        # Step 2: Pick best hyperparameters
        res = pick_best_lambda_kernel(
            feat1=feat1, feat2=feat2,
            ap_prune_result_path=GRID_SEARCH_PATH,
            port_n=PORT_N,
            lambda0=LAMBDA0, lambda2=LAMBDA2,
            n_bandwidths=n_bandwidths,
            kernel_name='gaussian',
            full_cv=False, write_table=True,
        )

        i_best, j_best, h_best = res['best_idx']
        print(f"  [{subdir}] Winner: l0={LAMBDA0[i_best]}, "
              f"l2={LAMBDA2[j_best]:.2e}, h={bandwidths[h_best]:.6f}", flush=True)

        # Step 3: Full fit
        kernel_star  = GaussianKernel(h=bandwidths[h_best])
        full_fit_dir = GRID_SEARCH_PATH / 'gaussian' / subdir / 'full_fit'

        result = kernel_full_fit(
            k_target=PORT_N,
            lambda0_star=LAMBDA0[i_best],
            lambda2_star=LAMBDA2[j_best],
            kernel=kernel_star,
            state=_state,
            output_dir=str(full_fit_dir),
            input_path=TREE_PORT_PATH / subdir,
            input_file_name=PORT_FILE_NAME,
            n_train_valid=360,
            kmin=K_MIN, kmax=K_MAX,
            kernel_name='gaussian',
        )

        print(f"  [{subdir}] Done — test_SR={result['test_SR']:.4f}", flush=True)

        return {
            'feat1': feat1, 'feat2': feat2,
            'cross_section': subdir,
            'status': 'done',
            'test_SR': result['test_SR'],
            'valid_SR': res['valid_SR'],
            'months_used': result['months_used'],
            'lambda0': LAMBDA0[i_best],
            'lambda2': LAMBDA2[j_best],
            'h': bandwidths[h_best],
            'error': None,
        }

    except Exception as e:
        print(f"  [{subdir}] FAILED: {e}", flush=True)
        return {
            'feat1': feat1, 'feat2': feat2,
            'cross_section': subdir,
            'status': 'failed',
            'test_SR': None, 'valid_SR': None,
            'months_used': None,
            'lambda0': None, 'lambda2': None, 'h': None,
            'error': 'error',  # see terminal logs for full traceback
        }


_state = None

def init_worker(state):
    global _state
    _state = state


if __name__ == "__main__":

    state_df = pd.read_csv('data/state_variables.csv',
                           index_col='MthCalDt', parse_dates=True)
    state = state_df['svar']
    print(f"State variable loaded: {len(state)} months")

    sigma_s      = state.iloc[:360].std()
    bandwidths   = GaussianKernel.bandwidth_grid(sigma_s, n=N_BANDWIDTHS)
    n_bandwidths = len(bandwidths)

    progress = load_progress()

    pending = progress[progress['status'] != 'done']
    n_done  = len(progress) - len(pending)
    print(f"{n_done}/{len(PAIRS)} already done, {len(pending)} remaining")
    print(f"Running with {N_WORKERS} parallel workers\n")

    if len(pending) == 0:
        print("All combinations already done.")
    else:
        args_list = [
            (row['feat1'], row['feat2'], bandwidths, n_bandwidths)
            for _, row in pending.iterrows()
        ]

        with Pool(processes=N_WORKERS,
                  initializer=init_worker,
                  initargs=(state,)) as pool:

            for result in pool.imap_unordered(run_one, args_list):
                cs = result['cross_section']
                mask = progress['cross_section'] == cs
                if result['error'] is not None:
                    result['error'] = str(result['error'])[:500]
                for col in ['status', 'test_SR', 'valid_SR', 'months_used',
                            'lambda0', 'lambda2', 'h', 'error']:
                    progress.loc[mask, col] = result[col]
                save_progress(progress)
                print(f"  Saved progress for {cs} — status: {result['status']}",
                      flush=True)

    done = progress[progress['status'] == 'done']
    done.to_csv(SUMMARY_PATH, index=False)
    print(f"\nAll done. {len(done)}/{len(PAIRS)} completed.")
    print(f"Summary: {SUMMARY_PATH}")
    if len(done) > 0:
        print(done[['cross_section', 'valid_SR', 'test_SR']].to_string(index=False))

    failed = progress[progress['status'] == 'failed']
    if len(failed) > 0:
        print(f"\n{len(failed)} failed:")
        print(failed[['cross_section', 'error']].to_string(index=False))