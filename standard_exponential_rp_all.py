"""
Exponential (time) kernel grid + full fit for **RP tree** portfolios — same **36** triplets as
``standard_gaussian_rp_all.py`` (secondary characteristics **including IdioVol**).

Writes, for each ``LME_feat1_feat2``:

- ``data/results/grid_search/rp_tree/exponential/LME_*/*`` — validation CSVs from ``AP_Pruning``
- ``.../exponential/LME_*/full_fit/full_fit_detail_k10.csv`` — for
  ``export_table51_rp_uniform_vs_gaussian.py`` **Exponential** SR and FF5 alpha

**Prerequisites:** Same as Gaussian batch — RP ``level_all_excess_combined.csv`` per triplet,
``data/state_variables.csv`` (state column used by API; exponential kernel is time-based).

From repo root::

    python standard_exponential_rp_all.py --workers 8

Resume-safe: ``progress_standard_exponential_rp.csv`` under ``rp_tree/exponential/``.

Typical order to fill Table 4 (RP)::

    python standard_gaussian_rp_all.py --workers 8
    python standard_exponential_rp_all.py --workers 8
    python part_3_metrics_collection/export_table51_rp_uniform_vs_gaussian.py \\
        --rows no-idiovol --k 10 --latex-out Figures/tables/tab_rp_three_kernels.tex
"""

from itertools import combinations
from pathlib import Path
from multiprocessing import Pool
import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
    canonical_feat_pair,
)
from part_2_AP_pruning.AP_Pruning import AP_Pruning
from part_2_AP_pruning.kernels.exponential import ExponentialKernel
from part_2_AP_pruning.lasso_kernel_full_fit import kernel_full_fit
from part_3_metrics_collection.pick_best_lambdas import pick_best_lambda_kernel

CHARACTERISTICS = [
    "BEME",
    "r12_2",
    "OP",
    "Investment",
    "ST_Rev",
    "LT_Rev",
    "AC",
    "LTurnover",
    "IdioVol",
]

LAMBDA0 = [0.5, 0.55, 0.6]
LAMBDA2 = [10**-7, 10**-7.25, 10**-7.5]
K_MIN = 5
K_MAX = 50
PORT_N = 10
N_TRAIN_VALID = 360
_DEFAULT_WORKERS = 4

RP_PORT_PATH = Path("data/results/rp_tree_portfolios")
GRID_SEARCH_PATH = Path("data/results/grid_search/rp_tree")
PORT_FILE_NAME = "level_all_excess_combined.csv"

PAIRS = list(combinations(CHARACTERISTICS, 2))

PROGRESS_PATH = GRID_SEARCH_PATH / "exponential" / "progress_standard_exponential_rp.csv"
SUMMARY_PATH = GRID_SEARCH_PATH / "exponential" / "all_cross_sections_summary_standard_exponential_rp.csv"


def load_progress():
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame(
        [
            {
                "feat1": f1,
                "feat2": f2,
                "cross_section": f"LME_{f1}_{f2}",
                "status": "pending",
                "test_SR": None,
                "valid_SR": None,
                "lambda0": None,
                "lambda2": None,
                "lam": None,
                "months_used": None,
                "error": None,
            }
            for f1, f2 in PAIRS
        ]
    )
    if not PROGRESS_PATH.exists():
        template.to_csv(PROGRESS_PATH, index=False)
        print(f"Progress file created: {PROGRESS_PATH}")
        return template

    existing = pd.read_csv(PROGRESS_PATH)
    if (
        len(existing) != len(template)
        or set(existing["cross_section"]) != set(template["cross_section"])
    ):
        old_by_cs = existing.set_index("cross_section", drop=False)
        rows = []
        for _, row in template.iterrows():
            cs = row["cross_section"]
            if cs in old_by_cs.index:
                rows.append(old_by_cs.loc[cs].to_dict())
            else:
                rows.append(row.to_dict())
        merged = pd.DataFrame(rows)
        merged.to_csv(PROGRESS_PATH, index=False)
        print(
            f"Progress file upgraded for {len(template)} triplets: {PROGRESS_PATH}",
            flush=True,
        )
        return merged

    return existing


def save_progress(df: pd.DataFrame) -> None:
    df.to_csv(PROGRESS_PATH, index=False)


def run_one(args):
    feat1, feat2, exp_lambdas, n_bandwidths = args
    subdir = f"LME_{feat1}_{feat2}"

    try:
        print(f"  [{subdir}] Starting (RP Exponential)...", flush=True)

        AP_Pruning(
            feat1=feat1,
            feat2=feat2,
            input_path=RP_PORT_PATH,
            input_file_name=PORT_FILE_NAME,
            output_path=GRID_SEARCH_PATH,
            n_train_valid=N_TRAIN_VALID,
            cvN=3,
            runFullCV=False,
            kmin=K_MIN,
            kmax=K_MAX,
            RunParallel=False,
            ParallelN=10,
            IsTree=True,
            lambda0=LAMBDA0,
            lambda2=LAMBDA2,
            kernel_cls=ExponentialKernel,
            state=_state,
            n_bandwidths=n_bandwidths,
        )

        res = pick_best_lambda_kernel(
            feat1=feat1,
            feat2=feat2,
            ap_prune_result_path=GRID_SEARCH_PATH,
            port_n=PORT_N,
            lambda0=LAMBDA0,
            lambda2=LAMBDA2,
            n_bandwidths=n_bandwidths,
            kernel_name="exponential",
            full_cv=False,
            write_table=True,
        )

        i_best, j_best, il_best = res["best_idx"]
        lam_star = exp_lambdas[il_best]
        print(
            f"  [{subdir}] Winner: l0={LAMBDA0[i_best]}, "
            f"l2={LAMBDA2[j_best]:.2e}, lambda={lam_star:.6f}",
            flush=True,
        )

        kernel_star = ExponentialKernel(lam=lam_star, m=N_TRAIN_VALID)
        full_fit_dir = GRID_SEARCH_PATH / "exponential" / subdir / "full_fit"

        result = kernel_full_fit(
            k_target=PORT_N,
            lambda0_star=LAMBDA0[i_best],
            lambda2_star=LAMBDA2[j_best],
            kernel=kernel_star,
            state=_state,
            output_dir=str(full_fit_dir),
            input_path=RP_PORT_PATH / subdir,
            input_file_name=PORT_FILE_NAME,
            n_train_valid=N_TRAIN_VALID,
            kmin=K_MIN,
            kmax=K_MAX,
            kernel_name="exponential",
        )

        print(f"  [{subdir}] Done — test_SR={result['test_SR']:.4f}", flush=True)

        return {
            "feat1": feat1,
            "feat2": feat2,
            "cross_section": subdir,
            "status": "done",
            "test_SR": result["test_SR"],
            "valid_SR": res["valid_SR"],
            "months_used": result["months_used"],
            "lambda0": LAMBDA0[i_best],
            "lambda2": LAMBDA2[j_best],
            "lam": lam_star,
            "error": None,
        }

    except Exception:
        print(f"  [{subdir}] FAILED", flush=True)
        return {
            "feat1": feat1,
            "feat2": feat2,
            "cross_section": subdir,
            "status": "failed",
            "test_SR": None,
            "valid_SR": None,
            "months_used": None,
            "lambda0": None,
            "lambda2": None,
            "lam": None,
            "error": traceback.format_exc(),
        }


_state = None


def init_worker(state):
    global _state
    _state = state


if __name__ == "__main__":
    pa = argparse.ArgumentParser(
        description="RP Exponential kernel: grid + full_fit for 36 triplets (includes IdioVol)."
    )
    pa.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel worker count (default: RP_N_WORKERS env or %d)." % _DEFAULT_WORKERS,
    )
    pa.add_argument("--feat1", type=str, default=None, help="Run only one triplet (e.g. BEME).")
    pa.add_argument("--feat2", type=str, default=None, help="Run only one triplet (e.g. OP).")
    args = pa.parse_args()
    n_workers = args.workers
    if n_workers is None:
        n_workers = int(os.environ.get("RP_N_WORKERS", str(_DEFAULT_WORKERS)))
    n_workers = max(1, n_workers)

    if (args.feat1 is None) ^ (args.feat2 is None):
        pa.error("Use --feat1 and --feat2 together (or neither).")
    selected_pairs = set(PAIRS)
    if args.feat1 is not None and args.feat2 is not None:
        f1, f2 = canonical_feat_pair(args.feat1, args.feat2)
        selected_pairs = {(f1, f2)}
        print(f"Single-triplet mode: LME_{f1}_{f2}", flush=True)

    state_df = pd.read_csv("data/state_variables.csv", index_col="MthCalDt", parse_dates=True)
    state = state_df["svar"]
    print(f"State series loaded: {len(state)} months (exponential kernel uses time ordering)")

    exp_lambdas = list(ExponentialKernel.bandwidth_grid(m=N_TRAIN_VALID))
    n_bandwidths = len(exp_lambdas)
    print(f"Exponential lambda grid (m={N_TRAIN_VALID}): {exp_lambdas}\n")

    progress = load_progress()
    target_subdirs = {f"LME_{f1}_{f2}" for f1, f2 in selected_pairs}
    pending = progress[
        (progress["status"] != "done")
        & (progress["cross_section"].isin(target_subdirs))
    ]
    n_done = len(selected_pairs) - len(pending)
    print(f"{n_done}/{len(selected_pairs)} already done, {len(pending)} remaining")
    print(f"RP portfolios: {RP_PORT_PATH}")
    print(f"Grid root:     {GRID_SEARCH_PATH} / exponential / …")
    print(f"Running with {n_workers} parallel workers\n")

    if len(pending) == 0:
        print("All combinations already done.")
    else:
        args_list = [
            (row["feat1"], row["feat2"], exp_lambdas, n_bandwidths)
            for _, row in pending.iterrows()
        ]

        with Pool(processes=n_workers, initializer=init_worker, initargs=(state,)) as pool:
            for result in tqdm(
                pool.imap_unordered(run_one, args_list),
                total=len(args_list),
                desc="RP Exponential grid + full_fit",
                unit="triplet",
                file=sys.stderr,
            ):
                cs = result["cross_section"]
                mask = progress["cross_section"] == cs
                if result["error"] is not None:
                    result["error"] = str(result["error"])[:500]
                for col in [
                    "status",
                    "test_SR",
                    "valid_SR",
                    "months_used",
                    "lambda0",
                    "lambda2",
                    "lam",
                    "error",
                ]:
                    progress.loc[mask, col] = result[col]
                save_progress(progress)
                print(f"  Saved progress for {cs} — status: {result['status']}", flush=True)

    done = progress[(progress["status"] == "done") & (progress["cross_section"].isin(target_subdirs))]
    done.to_csv(SUMMARY_PATH, index=False)
    print(f"\nAll done. {len(done)}/{len(selected_pairs)} completed.")
    print(f"Summary: {SUMMARY_PATH}")
    print(
        "Re-run: python part_3_metrics_collection/export_table51_rp_uniform_vs_gaussian.py "
        "--rows all --k 10"
    )

    failed = progress[progress["status"] == "failed"]
    if len(failed) > 0:
        print(f"\n{len(failed)} failed:")
        print(failed[["cross_section", "error"]].to_string(index=False))
