"""
tc_batch.py
-----------
Run transaction cost diagnostics for all kernels × all 36 cross-sections.

For each kernel the runner finds the right inputs, calls the matching entry point in
``transaction_costs.py``, and writes per-cross-section ``transaction_costs_*`` /
``tc_summary_*`` next to the grid (RP: flat ``LME_*`` for all kernels; see module doc
there).

Combined summaries per kernel::

    data/results/grid_search/tree/{kernel}/tc_summary_all_k{k}.csv          (AP)
    data/results/grid_search/rp_tree/{kernel}/tc_summary_all_k{k}.csv       (RP)

Usage
-----
    python -m part_3_metrics_collection.tc_batch_runner              # AP (tree/)
    python -m part_3_metrics_collection.tc_batch_runner --rp        # RP (rp_tree/)
    python -m part_3_metrics_collection.tc_batch_runner --rp --skip-uniform
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import traceback

import pandas as pd

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
    triplet_subdir_name,
)
from part_3_metrics_collection.pick_best_lambdas import pruning_results_base
from part_3_metrics_collection.transaction_costs import (
    compute_net_sharpe,
    compute_net_sharpe_rp,
    compute_net_sharpe_uniform,
    compute_net_sharpe_uniform_rp,
)

# ── Config ─────────────────────────────────────────────────────────────────────

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

PAIRS = list(combinations(CHARACTERISTICS, 2))
PORT_N = 10
TREE_GRID_PATH = Path("data/results/grid_search/tree")
RP_GRID_PATH = Path("data/results/grid_search/rp_tree")
RP_PORT_ROOT = Path("data/results/rp_tree_portfolios")
PANEL_PATH = Path("data/prepared/panel.parquet")
N_TRAIN_VALID = 360

<<<<<<< HEAD
KERNEL_KERNELS = ["gaussian", "exponential", "gaussian-tms"]

# ── Helpers ────────────────────────────────────────────────────────────────────
=======
ALL_KERNELS = ['gaussian', 'exponential', 'gaussian-tms',  'uniform'] 
>>>>>>> 2d18fd6 (bandwith en outlier diagnostics)


def subdir(feat1: str, feat2: str) -> str:
    return f"LME_{feat1}_{feat2}"


# ── Per-kernel runners ─────────────────────────────────────────────────────────


def run_kernel(kernel_name: str, *, grid_base: Path, is_rp: bool) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  Kernel: {kernel_name}")
    print(f"{'='*60}")

    rows = []

    for feat1, feat2 in PAIRS:
        cs = triplet_subdir_name(feat1, feat2) if is_rp else subdir(feat1, feat2)
        detail_path = (
            grid_base / kernel_name / cs / "full_fit" / f"full_fit_detail_k{PORT_N}.csv"
        )

        if not detail_path.exists():
            print(f"  [{cs}] SKIP - detail file not found: {detail_path}")
            rows.append(
                {
                    "feat1": feat1,
                    "feat2": feat2,
                    "cross_section": cs,
                    "gross_SR": None,
                    "net_SR": None,
                    "SR_loss": None,
                    "mean_TC": None,
                    "status": "missing",
                }
            )
            continue

        print(f"\n  [{cs}]")
        try:
            if is_rp:
                result = compute_net_sharpe_rp(
                    detail_path=detail_path,
                    panel_path=PANEL_PATH,
                    cross_section=cs,
                    feat1=feat1,
                    feat2=feat2,
                    rp_portfolios_root=RP_PORT_ROOT,
                    n_train_valid=N_TRAIN_VALID,
                    label=kernel_name,
                )
            else:
                result = compute_net_sharpe(
                    detail_path=detail_path,
                    panel_path=PANEL_PATH,
                    features=["LME", feat1, feat2],
                    n_train_valid=N_TRAIN_VALID,
                    label=kernel_name,
                )
            rows.append(
                {
                    "feat1": feat1,
                    "feat2": feat2,
                    "cross_section": cs,
                    "gross_SR": result["gross_SR"],
                    "net_SR": result["net_SR"],
                    "SR_loss": result["gross_SR"] - result["net_SR"],
                    "mean_TC": float(result["tc_series"].mean()),
                    "status": "done",
                }
            )
        except FileNotFoundError as e:
            print(f"  [{cs}] SKIP - {e}")
            rows.append(
                {
                    "feat1": feat1,
                    "feat2": feat2,
                    "cross_section": cs,
                    "gross_SR": None,
                    "net_SR": None,
                    "SR_loss": None,
                    "mean_TC": None,
                    "status": "missing_npz",
                }
            )
        except ValueError as e:
            if "months were skipped" in str(e):
                print(f"  [{cs}] SKIP - incomplete detail file ({e})")
                rows.append(
                    {
                        "feat1": feat1,
                        "feat2": feat2,
                        "cross_section": cs,
                        "gross_SR": None,
                        "net_SR": None,
                        "SR_loss": None,
                        "mean_TC": None,
                        "status": "skipped_incomplete",
                    }
                )
            else:
                print(f"  [{cs}] ERROR: {e}")
                traceback.print_exc()
                rows.append(
                    {
                        "feat1": feat1,
                        "feat2": feat2,
                        "cross_section": cs,
                        "gross_SR": None,
                        "net_SR": None,
                        "SR_loss": None,
                        "mean_TC": None,
                        "status": f"error: {e}",
                    }
                )
        except Exception as e:
            print(f"  [{cs}] ERROR: {e}")
            traceback.print_exc()
            rows.append(
                {
                    "feat1": feat1,
                    "feat2": feat2,
                    "cross_section": cs,
                    "gross_SR": None,
                    "net_SR": None,
                    "SR_loss": None,
                    "mean_TC": None,
                    "status": f"error: {e}",
                }
            )

    summary = pd.DataFrame(rows)
    out_path = grid_base / kernel_name / f"tc_summary_all_k{PORT_N}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\n  Summary saved -> {out_path}")
    return summary


def run_uniform(*, grid_base: Path, is_rp: bool) -> pd.DataFrame:
    kernel_name = "uniform"
    print(f"\n{'='*60}")
    print(f"  Kernel: {kernel_name}")
    print(f"{'='*60}")

    rows = []

    for feat1, feat2 in PAIRS:
        cs = triplet_subdir_name(feat1, feat2) if is_rp else subdir(feat1, feat2)
        kernel_dir = pruning_results_base(grid_base, feat1, feat2)
        ports_path = kernel_dir / f"Selected_Ports_{PORT_N}.csv"
        weights_path = kernel_dir / f"Selected_Ports_Weights_{PORT_N}.csv"

        if not ports_path.exists() or not weights_path.exists():
            print(f"  [{cs}] SKIP - ports/weights file not found")
            rows.append(
                {
                    "feat1": feat1,
                    "feat2": feat2,
                    "cross_section": cs,
                    "gross_SR": None,
                    "net_SR": None,
                    "SR_loss": None,
                    "mean_TC": None,
                    "status": "missing",
                }
            )
            continue

        print(f"\n  [{cs}]")
        try:
            if is_rp:
                result = compute_net_sharpe_uniform_rp(
                    ports_path=ports_path,
                    weights_path=weights_path,
                    panel_path=PANEL_PATH,
                    cross_section=cs,
                    feat1=feat1,
                    feat2=feat2,
                    rp_portfolios_root=RP_PORT_ROOT,
                    n_train_valid=N_TRAIN_VALID,
                    label=kernel_name,
                )
            else:
                result = compute_net_sharpe_uniform(
                    ports_path=ports_path,
                    weights_path=weights_path,
                    panel_path=PANEL_PATH,
                    features=["LME", feat1, feat2],
                    n_train_valid=N_TRAIN_VALID,
                    label=kernel_name,
                )
            rows.append(
                {
                    "feat1": feat1,
                    "feat2": feat2,
                    "cross_section": cs,
                    "gross_SR": result["gross_SR"],
                    "net_SR": result["net_SR"],
                    "SR_loss": result["gross_SR"] - result["net_SR"],
                    "mean_TC": float(result["tc_series"].mean()),
                    "status": "done",
                }
            )
        except FileNotFoundError as e:
            print(f"  [{cs}] SKIP - {e}")
            rows.append(
                {
                    "feat1": feat1,
                    "feat2": feat2,
                    "cross_section": cs,
                    "gross_SR": None,
                    "net_SR": None,
                    "SR_loss": None,
                    "mean_TC": None,
                    "status": "missing_npz",
                }
            )
        except Exception as e:
            print(f"  [{cs}] ERROR: {e}")
            traceback.print_exc()
            rows.append(
                {
                    "feat1": feat1,
                    "feat2": feat2,
                    "cross_section": cs,
                    "gross_SR": None,
                    "net_SR": None,
                    "SR_loss": None,
                    "mean_TC": None,
                    "status": f"error: {e}",
                }
            )

    summary = pd.DataFrame(rows)
    out_path = grid_base / kernel_name / f"tc_summary_all_k{PORT_N}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\n  Summary saved -> {out_path}")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument(
        "--rp",
        action="store_true",
        help="Use RP grids under data/results/grid_search/rp_tree and rp_tree_portfolios.",
    )
    pa.add_argument(
        "--skip-uniform",
        action="store_true",
        help="Do not recompute uniform TC (Gaussian / Exponential / Gaussian-TMS only).",
    )
    args = pa.parse_args()

    grid_base = RP_GRID_PATH if args.rp else TREE_GRID_PATH
    is_rp = bool(args.rp)

    print(f"\n  Grid root: {grid_base.resolve()}  (RP mode={is_rp}, skip_uniform={args.skip_uniform})")

    all_summaries: dict[str, pd.DataFrame] = {}

    for kernel in KERNEL_KERNELS:
        all_summaries[kernel] = run_kernel(kernel, grid_base=grid_base, is_rp=is_rp)

    if not args.skip_uniform:
        all_summaries["uniform"] = run_uniform(grid_base=grid_base, is_rp=is_rp)

    print(f"\n{'='*60}")
    print("  DONE - median net SR per kernel:")
    for kernel, df in all_summaries.items():
        done = df[df["status"] == "done"]
        if len(done) > 0:
            print(
                f"    {kernel:15s}  median net SR = {done['net_SR'].median():.4f}"
                f"  ({len(done)}/36 cross-sections)"
            )
