"""
Gaussian kernel on **term spread (TMS)** only: grid + full fit for **RP tree** portfolios
(same **36** triplets as ``standard_gaussian_rp_all.py``).

Writes under ``data/results/grid_search/rp_tree/gaussian-tms/LME_*``.

**Prerequisites:** ``data/results/rp_tree_portfolios/LME_* / level_all_excess_combined.csv``
and ``data/state_variables.csv`` with column ``TMS``.

From repo root::

    # Fast: use more CPU cores (one triplet per worker at a time)
    python standard_gaussian_tms_rp_all.py

    python standard_gaussian_tms_rp_all.py --feat1 BEME --feat2 AC

Resume-safe: progress CSV under ``rp_tree/gaussian-tms/``.
"""

from itertools import combinations
from pathlib import Path
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
from part_2_AP_pruning.kernels.gaussian import GaussianKernel
from part_2_AP_pruning.lasso_kernel_full_fit import kernel_full_fit
from part_3_metrics_collection.pick_best_lambdas import pick_best_lambda_kernel

# Same secondary ordering as ``FEATS_LIST[1:]`` in ``cross_section_triplets.py``
# (includes IdioVol) → C(9,2) = 36 triplets.
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
N_BANDWIDTHS = 5
_DEFAULT_WORKERS = 4

KERNEL_NAME = "gaussian-tms"

RP_PORT_PATH = Path("data/results/rp_tree_portfolios")
GRID_SEARCH_PATH = Path("data/results/grid_search/rp_tree")
PORT_FILE_NAME = "level_all_excess_combined.csv"

PAIRS = list(combinations(CHARACTERISTICS, 2))
PAIR_TO_ROW = {pair: idx for idx, pair in enumerate(PAIRS, start=1)}

PROGRESS_PATH = GRID_SEARCH_PATH / KERNEL_NAME / "progress_standard_gaussian_tms_rp.csv"
SUMMARY_PATH = GRID_SEARCH_PATH / KERNEL_NAME / "all_cross_sections_summary_standard_gaussian_tms_rp.csv"


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
                "h": None,
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
    # If the repo was upgraded from the old 28-triplet batch, grow the CSV in-place
    # instead of silently ignoring new triplets.
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
    feat1, feat2, bandwidths, n_bandwidths = args
    subdir = f"LME_{feat1}_{feat2}"
    row_num = PAIR_TO_ROW.get((feat1, feat2), None)

    try:
        row_prefix = f"ROW {row_num:02d}" if row_num is not None else "ROW ??"
        print(f"  [{row_prefix}] [{subdir}] Starting (RP Gaussian-TMS)...", flush=True)

        AP_Pruning(
            feat1=feat1,
            feat2=feat2,
            input_path=RP_PORT_PATH,
            input_file_name=PORT_FILE_NAME,
            output_path=GRID_SEARCH_PATH,
            n_train_valid=360,
            cvN=3,
            runFullCV=False,
            kmin=K_MIN,
            kmax=K_MAX,
            RunParallel=False,
            ParallelN=10,
            IsTree=True,
            lambda0=LAMBDA0,
            lambda2=LAMBDA2,
            kernel_cls=GaussianKernel,
            state=_state,
            n_bandwidths=N_BANDWIDTHS,
            kernel_name_override=KERNEL_NAME,
        )

        res = pick_best_lambda_kernel(
            feat1=feat1,
            feat2=feat2,
            ap_prune_result_path=GRID_SEARCH_PATH,
            port_n=PORT_N,
            lambda0=LAMBDA0,
            lambda2=LAMBDA2,
            n_bandwidths=n_bandwidths,
            kernel_name=KERNEL_NAME,
            full_cv=False,
            write_table=True,
        )

        i_best, j_best, h_best = res["best_idx"]
        print(
            f"  [{row_prefix}] [{subdir}] Winner: l0={LAMBDA0[i_best]}, "
            f"l2={LAMBDA2[j_best]:.2e}, h={bandwidths[h_best]:.6f}",
            flush=True,
        )

        kernel_star = GaussianKernel(h=bandwidths[h_best])
        full_fit_dir = GRID_SEARCH_PATH / KERNEL_NAME / subdir / "full_fit"

        result = kernel_full_fit(
            k_target=PORT_N,
            lambda0_star=LAMBDA0[i_best],
            lambda2_star=LAMBDA2[j_best],
            kernel=kernel_star,
            state=_state,
            output_dir=str(full_fit_dir),
            input_path=RP_PORT_PATH / subdir,
            input_file_name=PORT_FILE_NAME,
            n_train_valid=360,
            kmin=K_MIN,
            kmax=K_MAX,
            kernel_name=KERNEL_NAME,
        )

        print(f"  [{row_prefix}] [{subdir}] Done — test_SR={result['test_SR']:.4f}", flush=True)

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
            "h": bandwidths[h_best],
            "error": None,
        }

    except Exception:
        row_prefix = f"ROW {row_num:02d}" if row_num is not None else "ROW ??"
        print(f"  [{row_prefix}] [{subdir}] FAILED", flush=True)
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
            "h": None,
            "error": traceback.format_exc(),
        }


_state = None


def init_worker(state):
    global _state
    _state = state


if __name__ == "__main__":
    pa = argparse.ArgumentParser(
        description="RP Gaussian-TMS kernel: grid + full_fit for 36 triplets (TMS state)."
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
    if n_workers != 1:
        print(
            f"Requested --workers={n_workers}, but strict row-by-row mode requires one row at a time. "
            "Using --workers=1.",
            flush=True,
        )
    n_workers = 1

    if (args.feat1 is None) ^ (args.feat2 is None):
        pa.error("Use --feat1 and --feat2 together (or neither).")
    selected_pairs = set(PAIRS)
    if args.feat1 is not None and args.feat2 is not None:
        f1, f2 = canonical_feat_pair(args.feat1, args.feat2)
        selected_pairs = {(f1, f2)}
        print(f"Single-triplet mode: LME_{f1}_{f2}", flush=True)

    state_df = pd.read_csv("data/state_variables.csv", index_col="MthCalDt", parse_dates=True)
    state = state_df["TMS"]
    print(f"State variable TMS: {len(state)} months")

    sigma_s = state.iloc[:360].std()
    bandwidths = GaussianKernel.bandwidth_grid(sigma_s, n=N_BANDWIDTHS)
    n_bandwidths = len(bandwidths)

    progress = load_progress()
    ordered_rows = []
    for feat1, feat2 in PAIRS:
        if (feat1, feat2) not in selected_pairs:
            continue
        cs = f"LME_{feat1}_{feat2}"
        row = progress[progress["cross_section"] == cs]
        if row.empty:
            continue
        if row.iloc[0]["status"] != "done":
            ordered_rows.append((feat1, feat2))
    pending = [p for p in selected_pairs if p in ordered_rows]
    n_done = len(selected_pairs) - len(pending)
    print(f"{n_done}/{len(selected_pairs)} already done, {len(pending)} remaining")
    print(f"RP portfolios: {RP_PORT_PATH}")
    print(f"Grid root:     {GRID_SEARCH_PATH} / {KERNEL_NAME} / …")
    print(f"Running strict sequential mode (row 1 -> row {len(PAIRS)})\n")

    if len(pending) == 0:
        print("All combinations already done.")
    else:
        init_worker(state)
        for feat1, feat2 in tqdm(
            ordered_rows,
            total=len(ordered_rows),
            desc="RP Gaussian-TMS grid + full_fit",
            unit="triplet",
            file=sys.stderr,
        ):
            row_num = PAIR_TO_ROW.get((feat1, feat2), None)
            result = run_one((feat1, feat2, bandwidths, n_bandwidths))
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
                "h",
                "error",
            ]:
                progress.loc[mask, col] = result[col]
            save_progress(progress)
            if row_num is not None:
                print(f"ROW {row_num:02d} FINISHED ({cs}) — status: {result['status']}", flush=True)
            else:
                print(f"ROW FINISHED ({cs}) — status: {result['status']}", flush=True)

    target_subdirs = {f"LME_{f1}_{f2}" for f1, f2 in selected_pairs}
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
