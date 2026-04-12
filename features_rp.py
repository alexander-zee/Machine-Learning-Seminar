from pathlib import Path
# Importeer je nieuwe imputatie script
from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import prepare_data, build_state_variables
from part_1_portfolio_creation.tree_portfolio_creation.step1b_impute_data import run_mice_imputation 

from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import create_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step2_RP_tree_portfolios import create_rp_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step2_cluster_portfolios import create_cluster_portfolios
from part_1_portfolio_creation.tree_portfolio_creation.step2_mice_rp_portfolios import create_mice_rp_tree_portfolio
from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import combine_trees
from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_RP_trees import combine_rp_trees
from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_mice_rp import combine_mice_rp_trees
from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import filter_tree_ports

# AP‑Pruning and metric collection modules
from part_2_AP_pruning.AP_Pruning import AP_Pruning
from part_2_AP_pruning.RP_Pruning import RP_Pruning
from part_2_AP_pruning.Mice_RP_Pruning import Mice_RP_Pruning
from part_2_AP_pruning.lasso_kernel_full_fit import kernel_full_fit
from part_3_metrics_collection.pick_best_lambdas import pick_best_lambda, pick_sr_n, get_mu_sigma
from part_3_metrics_collection.mice_pick_best_lambdas import mice_pick_best_lambda, mice_pick_sr_n, mice_get_mu_sigma, mice_pick_best_lambda_kernel
from part_3_metrics_collection.ff5 import evaluate_master_portfolio
from part_3_metrics_collection.mice_ff5 import mice_evaluate_master_portfolio
# Configuration — λ grid (BPZ-style shrinkage). Middle ground: 5×4 = 20 combos (~2× the old 3×3).
from part_2_AP_pruning.kernels.uniform import UniformKernel
from part_2_AP_pruning.kernels.gaussian import GaussianKernel
from part_3_metrics_collection.pick_best_lambdas import pick_best_lambda, pick_sr_n

import pandas as pd

ALL_FEATURES = [
    'LME', 'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]
 
# Values of n_features_per_split to sweep over
N_FEATURES_GRID = [1]
#N_FEATURES_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
 
# Fixed hyperparameters
N_TREES     = 81
TREE_DEPTH  = 4
GLOBAL_SEED = 42
Y_MIN       = 1964
Y_MAX       = 2016
N_BANDWIDTHS = 5

LAMBDA0 = [0.50, 0.55, 0.60]
LAMBDA2 = [10**-7, 10**-7.25, 10**-7.5]
 
K_MIN   = 5
K_MAX   = 50
PORT_N  = 10          # fixed k used for the single-k diagnostics in steps 6 / 6-cont.
 
N_TRAIN_VALID = 360   # months used for train + validation window
 
# Paths
RAW_DATASET_PATH    = Path('data/raw/FINALdataset.csv')
STATE_VARS_PATH     = Path('data/prepared/state_variables.csv')
PORTFOLIO_PATH      = Path('data/results/mice_rp_tree_portfolios')
GRID_PATH           = Path('data/results/grid_search/mice_rp_tree')
FACTOR_PATH         = Path('data/raw')
PORT_FILE           = 'level_all_excess_combined.csv'
 
 
# ── Per-run helpers ───────────────────────────────────────────────────────────
 
def _subdir(n_features: int) -> str:
    """Subdirectory name that encodes both features and n_features_per_split."""
    return f"{'_'.join(ALL_FEATURES)}__nf{n_features}"
 
def _kernel_label(kernel_cls) -> str:
    return kernel_cls.__name__ if kernel_cls is not None else 'UniformKernel'

def _load_state() -> pd.Series:
    """
    Load the state variable series aligned to the portfolio return rows.
    Used only for the Gaussian kernel — pass to Mice_RP_Pruning as state=.
    """
    df = pd.read_csv(STATE_VARS_PATH)
    state_col = [c for c in df.columns if c not in ('MthCalDt', 'date')][0]
    return df[state_col].reset_index(drop=True)


def run_step1a() -> None:
    print("\n--- Step 1a: Prepare panel data ---")
    prepare_data()
    print("\n--- Step 1a cont.: Build state variables ---")
    build_state_variables(
        final_dataset_path = RAW_DATASET_PATH,
        output_path        = STATE_VARS_PATH,
    )
 
 
def run_step1b() -> None:
    print("\n--- Step 1b: MICE imputation ---")
    run_mice_imputation()
 
 
def run_step2(n_features: int) -> None:
    print(f"\n--- Step 2: Build RP trees (nf={n_features}) ---")
    create_mice_rp_tree_portfolio(
        tree_depth           = TREE_DEPTH,
        y_min                = Y_MIN,
        y_max                = Y_MAX,
        n_trees              = N_TREES,
        global_seed          = GLOBAL_SEED,
        all_features         = ALL_FEATURES,
        n_features_per_split = n_features,
        output_path          = PORTFOLIO_PATH,
    )
 
 
def run_step3(n_features: int) -> None:
    print(f"\n--- Step 3: Combine trees (nf={n_features}) ---")
    combine_mice_rp_trees(
        all_features         = ALL_FEATURES,
        n_trees              = N_TREES,
        n_features_per_split = n_features,
        factor_path          = FACTOR_PATH,
        tree_out             = PORTFOLIO_PATH,
    )
 
 
def run_step5(n_features: int, kernel_cls=None, state=None) -> None:
    print(f"\n--- Step 5: RP-Pruning grid search (nf={n_features}, kernel={_kernel_label(kernel_cls)}) ---")
    Mice_RP_Pruning(
        all_features         = ALL_FEATURES,
        n_features_per_split = n_features,
        input_path           = PORTFOLIO_PATH,
        input_file_name      = PORT_FILE,
        output_path          = GRID_PATH,
        n_train_valid        = N_TRAIN_VALID,
        cvN                  = 3,
        runFullCV            = False,
        kmin                 = K_MIN,
        kmax                 = K_MAX,
        RunParallel          = False,
        ParallelN            = 10,
        IsTree               = True,
        lambda0              = LAMBDA0,
        lambda2              = LAMBDA2,
        kernel_cls           = kernel_cls,
        state                = state,
        n_bandwidths         = N_BANDWIDTHS if kernel_cls is not None else None,
    )
 



 
def run_step6(n_features: int, kernel_cls=None, state=None) -> None:
    """
    Pick best lambda and compute mu/sigma.
 
    For the uniform kernel: reads full-fit CSVs directly (h_idx=1).
    For the Gaussian kernel: reads CV-only CSVs across the 3D grid.
    Note — for Gaussian, test_SR is not yet available here; it is computed
    in step 6.5 via kernel_full_fit.
    """
    label = _kernel_label(kernel_cls)
    print(f"\n--- Step 6: Pick best lambda (k={PORT_N}, nf={n_features}, kernel={label}) ---")
 
    if kernel_cls is None:
        # Uniform: 2D grid, full-fit CSVs available
        best_sr = mice_pick_best_lambda(
            all_features         = ALL_FEATURES,
            n_features_per_split = n_features,
            ap_prune_result_path = GRID_PATH,
            port_n               = PORT_N,
            lambda0              = LAMBDA0,
            lambda2              = LAMBDA2,
            portfolio_path       = PORTFOLIO_PATH,
            port_name            = PORT_FILE,
            full_cv              = False,
            write_table          = True,
            kernel_cls           = None,
        )
        print(
            f"  Best SR  k={PORT_N}: "
            f"train={best_sr[0]:.4f}  valid={best_sr[1]:.4f}  test={best_sr[2]:.4f}"
        )
 
        print(f"\n--- Step 6 cont.: Mu / sigma (k={PORT_N}, nf={n_features}, kernel={label}) ---")
        stats = mice_get_mu_sigma(
            all_features         = ALL_FEATURES,
            n_features_per_split = n_features,
            ap_prune_result_path = GRID_PATH,
            portfolio_path       = PORTFOLIO_PATH,
            port_name            = PORT_FILE,
            port_n               = PORT_N,
            n_train_valid        = N_TRAIN_VALID,
            kernel_cls           = None,
        )
        print(
            f"  Train — mu={stats['train']['mu']:.6f}  "
            f"sigma={stats['train']['sigma']:.6f}  SR={stats['train']['SR']:.4f}"
        )
        print(
            f"  Test  — mu={stats['test']['mu']:.6f}  "
            f"sigma={stats['test']['sigma']:.6f}  SR={stats['test']['SR']:.4f}"
        )
        match = abs(best_sr[2] - stats['test']['SR']) < 1e-8
        print(
            f"  Cross-check: saved test_SR={best_sr[2]:.4f}  "
            f"recomputed={stats['test']['SR']:.4f}  match={match}"
        )
 
    else:
        # Gaussian: 3D grid, CV-only — test_SR comes from step 6.5
        res = mice_pick_best_lambda_kernel(
            all_features         = ALL_FEATURES,
            n_features_per_split = n_features,
            ap_prune_result_path = GRID_PATH,
            port_n               = PORT_N,
            lambda0              = LAMBDA0,
            lambda2              = LAMBDA2,
            n_bandwidths         = N_BANDWIDTHS,
            kernel_cls           = kernel_cls,
            portfolio_path       = PORTFOLIO_PATH,
            port_name            = PORT_FILE,
            full_cv              = False,
            write_table          = True,
        )
        i_best, j_best, h_best = res['best_idx']
        print(
            f"  Best valid_SR={res['valid_SR']:.4f}  "
            f"l0_idx={i_best+1}  l2_idx={j_best+1}  h_idx={h_best+1}"
        )
        # Return winning indices so step 6.5 can use them
        return res
    
def run_step6_5(n_features: int, kernel_cls, state: pd.Series,
                best_result: dict) -> None:
    """
    Gaussian full fit for the winning (lambda0, lambda2, h) from step 6.
 
    Only executed when kernel_cls is not None. Runs kernel_full_fit on the
    full train window and evaluates on the test window, producing test_SR
    and saving Selected_Ports / Weights to the full_fit/ subfolder.
 
    Parameters
    ----------
    n_features  : n_features_per_split value for this run
    kernel_cls  : Gaussian kernel class
    state       : monthly state variable series
    best_result : dict returned by mice_pick_best_lambda_kernel (from step 6)
                  must contain 'best_idx' key: (i_best, j_best, h_best)
    """
    label = _kernel_label(kernel_cls)
    print(f"\n--- Step 6.5: Gaussian full fit (k={PORT_N}, nf={n_features}, kernel={label}) ---")
 
    i_best, j_best, h_best = best_result['best_idx']
    lambda0_star = LAMBDA0[i_best]
    lambda2_star = LAMBDA2[j_best]
 
    # Reconstruct the bandwidth grid to get the winning h value
    bandwidths   = kernel_cls.bandwidth_grid_from_state(state, N_TRAIN_VALID, n=N_BANDWIDTHS)
    h_star       = bandwidths[h_best]
    kernel_star  = kernel_cls(h=h_star)
 
    subdir       = _subdir(n_features)
    kernel_name  = kernel_cls.__name__.lower().replace('kernel', '')
    full_fit_dir = GRID_PATH / kernel_name / subdir / 'full_fit'
 
    print(
        f"  Winner: lambda0={lambda0_star}  lambda2={lambda2_star:.2e}  h={h_star:.6f}"
    )
 
    result = kernel_full_fit(
        k_target         = PORT_N,
        lambda0_star     = lambda0_star,
        lambda2_star     = lambda2_star,
        kernel           = kernel_star,
        state            = state,
        output_dir       = str(full_fit_dir),
        input_path       = PORTFOLIO_PATH / subdir,
        input_file_name  = PORT_FILE,
        n_train_valid    = N_TRAIN_VALID,
        kmin             = K_MIN,
        kmax             = K_MAX,
        kernel_name      = kernel_name,
    )
 
    print(
        f"  test_SR={result['test_SR']:.4f}  "
        f"months_used={result.get('months_used', 'n/a')}"
    )

 
def run_step7(n_features: int, kernel_cls=None) -> None:
    print(f"\n--- Step 7: SR sweep k={K_MIN}..{K_MAX} (nf={n_features}, kernel={_kernel_label(kernel_cls)}) ---")
    mice_pick_sr_n(
        all_features         = ALL_FEATURES,
        n_features_per_split = n_features,
        grid_search_path     = GRID_PATH,
        mink                 = K_MIN,
        maxk                 = K_MAX,
        lambda0              = LAMBDA0,
        lambda2              = LAMBDA2,
        port_path            = PORTFOLIO_PATH,
        port_file_name       = PORT_FILE,
        kernel_cls           = kernel_cls,
    )
 
 
def run_step8(n_features: int, kernel_cls=None) -> None:
    print(f"\n--- Step 8: FF5 alpha k={K_MIN}..{K_MAX} (nf={n_features}, kernel={_kernel_label(kernel_cls)}) ---")
    for k in range(K_MIN, K_MAX + 1):
        alpha, pval = mice_evaluate_master_portfolio(
            all_features         = ALL_FEATURES,
            n_features_per_split = n_features,
            k                    = k,
            grid_dir             = GRID_PATH,
            ports_dir            = PORTFOLIO_PATH,
            file_name            = PORT_FILE,
            n_train_valid        = N_TRAIN_VALID,
            kernel_cls           = kernel_cls,
        )
        if alpha is not None:
            print(f"  k={k:2d}  FF5 alpha={alpha:.6f}  p={pval:.4f}")
 
 
# ── Master pipeline ───────────────────────────────────────────────────────────
 
STEPS_ONCE = {
    '1a': run_step1a,
    '1b': run_step1b,
}
 
STEPS = {
    2: run_step2,
    3: run_step3,
    5: run_step5,
    6: run_step6,
    6.5: run_step6_5,
    7: run_step7,
    #8: run_step8,
}
 
 
def run_pipeline(
    n_features_grid: list = N_FEATURES_GRID,
    steps: list           = None,
    kernel_cls            = None,
    state                 = None,
) -> None:
    """
    Run selected pipeline steps for every n_features value in n_features_grid.
 
    Steps 1a and 1b run once (independent of n_features and kernel).
    Steps 2 and 3 run once per n_features value (kernel has no effect).
    Steps 5-8 run once per n_features value and accept kernel_cls / state.
    Step 6.5 runs only when kernel_cls is not None; it is skipped for uniform.
 
    Parameters
    ----------
    n_features_grid : list of int
        Values of n_features_per_split to sweep over.
    steps : list of str/int/float or None
        Which steps to run, e.g. ['1a', '1b', 2, 3, 5].
        None runs all steps: 1a, 1b, 2, 3, 5, 6, 6.5, 7, 8.
        Step 6.5 is silently skipped for the uniform kernel even if listed.
    kernel_cls : kernel class or None
        Passed to steps 5-8. None → uniform kernel.
    state : pd.Series or None
        Monthly state variable series. Required for Gaussian kernel.
        Load via _load_state() or pass directly.
 
    Examples
    --------
    # Uniform kernel (default):
    run_pipeline()
 
    # Gaussian kernel, steps 5 onward:
    from part_2_AP_pruning.kernels.gaussian import GaussianKernel
    run_pipeline(steps=[5, 6, 6.5, 7, 8], kernel_cls=GaussianKernel, state=_load_state())
    """
    all_steps    = ['1a', '1b', 2, 3, 5, 6, 6.5, 7, 8]
    steps_to_run = steps if steps is not None else all_steps
 
    print("=" * 65)
    print("  MICE RP TREE PIPELINE")
    print(f"  Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")
    print(f"  n_features grid : {n_features_grid}")
    print(f"  Steps           : {steps_to_run}")
    print(f"  Kernel          : {_kernel_label(kernel_cls)}")
    print("=" * 65)
 
    # Steps 1a / 1b — run once, no kernel argument
    for step in steps_to_run:
        if step in STEPS_ONCE:
            STEPS_ONCE[step]()
 
    # Steps 2 onward — run once per n_features value
    per_nf_steps = [s for s in steps_to_run if s in STEPS]
    if per_nf_steps:
        for nf in n_features_grid:
            print(f"\n{'=' * 65}")
            print(f"  n_features_per_split = {nf}  |  kernel = {_kernel_label(kernel_cls)}")
            print(f"  subdir: {_subdir(nf)}")
            print(f"{'=' * 65}")
 
            # Step 6 may return best_result for step 6.5 to consume
            best_result = None
 
            for step in per_nf_steps:
                if step in (2, 3):
                    STEPS[step](nf)
 
                elif step == 5:
                    STEPS[step](nf, kernel_cls=kernel_cls, state=state)
 
                elif step == 6:
                    ret = STEPS[step](nf, kernel_cls=kernel_cls, state=state)
                    if kernel_cls is not None:
                        best_result = ret   # carry forward to step 6.5
 
                elif step == 6.5:
                    if kernel_cls is None:
                        print("\n--- Step 6.5: skipped (uniform kernel) ---")
                    else:
                        if best_result is None:
                            raise RuntimeError(
                                "Step 6.5 requires step 6 to have run first "
                                "in the same pipeline call to obtain best_result."
                            )
                        STEPS[step](nf, kernel_cls=kernel_cls,
                                    state=state, best_result=best_result)
 
                else:
                    STEPS[step](nf, kernel_cls=kernel_cls)
 
    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print("=" * 65)
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    # Uniform kernel — default, no extra arguments needed:
    run_pipeline(steps=[6])
    run_pipeline(steps=[6, 6.5], kernel_cls= GaussianKernel, state= _load_state())


    # Gaussian kernel example — uncomment to run:
    # from part_2_AP_pruning.kernels.gaussian import GaussianKernel
    # run_pipeline(
    #     steps      = [5, 6, 6.5, 7, 8],
    #     kernel_cls = GaussianKernel,
    #     state      = _load_state(),
    # )