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

By default, each cross-section is **skipped** when both ``tc_summary_*`` and
``transaction_costs_*`` already exist and are **not older** than the inputs used
to build them (detail CSV, ports/weights, ``panel.parquet``, and RP ``npz`` when
applicable). Pass ``--force`` to recompute everything.
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
    _rp_kernel_tc_output_dir,
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
<<<<<<< HEAD
KERNEL_KERNELS = ["gaussian", "exponential", "gaussian-tms"]

# ── Helpers ────────────────────────────────────────────────────────────────────
=======
ALL_KERNELS = ['gaussian', 'exponential', 'gaussian-tms',  'uniform'] 
>>>>>>> 2d18fd6 (bandwith en outlier diagnostics)
=======
ALL_KERNELS = ['uniform', 'gaussian-tms'] 
>>>>>>> 7f299d5 (tc/bandwith/outlier diagnostics)


def subdir(feat1: str, feat2: str) -> str:
    return f"LME_{feat1}_{feat2}"


def _detail_k_tag(detail_path: Path) -> str:
    return detail_path.stem.split("_")[-1]


def _kernel_output_paths(
    detail_path: Path,
    cross_section: str,
    kernel_name: str,
    *,
    is_rp: bool,
) -> tuple[Path, Path, Path]:
    """``(out_dir, tc_summary_csv, transaction_costs_csv)`` for kernel TC."""
    k_tag = _detail_k_tag(detail_path)
    suffix = f"_{kernel_name}"
    out_dir = (
        _rp_kernel_tc_output_dir(detail_path, cross_section)
        if is_rp
        else detail_path.parent
    )
    return (
        out_dir,
        out_dir / f"tc_summary_{k_tag}{suffix}.csv",
        out_dir / f"transaction_costs_{k_tag}{suffix}.csv",
    )


def _uniform_output_paths(ports_path: Path, label: str) -> tuple[Path, Path, Path]:
    k_tag = f"k{PORT_N}"
    suffix = f"_{label}"
    out_dir = ports_path.parent
    return (
        out_dir,
        out_dir / f"tc_summary_{k_tag}{suffix}.csv",
        out_dir / f"transaction_costs_{k_tag}{suffix}.csv",
    )


def _should_skip_existing(
    *,
    summary_path: Path,
    monthly_path: Path,
    input_paths: list[Path],
) -> bool:
    """
    True when both outputs exist and no listed input is newer than the oldest output.

    Missing input files => do not skip (caller must have verified prerequisites).
    """
    if not summary_path.is_file() or not monthly_path.is_file():
        return False
    try:
        out_min = min(summary_path.stat().st_mtime, monthly_path.stat().st_mtime)
    except OSError:
        return False
    newest_in = 0.0
    for p in input_paths:
        if not p.is_file():
            return False
        try:
            newest_in = max(newest_in, p.stat().st_mtime)
        except OSError:
            return False
    return newest_in <= out_min


def _metrics_from_existing_summary(summary_path: Path) -> dict[str, float]:
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"empty TC summary: {summary_path}")
    r = df.iloc[0]
    gross = float(r["gross_SR"])
    net = float(r["net_SR"])
    if "SR_loss" in df.columns and pd.notna(r["SR_loss"]):
        sr_loss = float(r["SR_loss"])
    else:
        sr_loss = gross - net
    return {
        "gross_SR": gross,
        "net_SR": net,
        "SR_loss": sr_loss,
        "mean_TC": float(r["mean_TC"]),
    }


# ── Per-kernel runners ─────────────────────────────────────────────────────────


def run_kernel(
    kernel_name: str, *, grid_base: Path, is_rp: bool, incremental: bool
) -> pd.DataFrame:
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

        _, summary_path, monthly_path = _kernel_output_paths(
            detail_path, cs, kernel_name, is_rp=is_rp
        )
        input_paths = [detail_path, PANEL_PATH]
        if is_rp:
            input_paths.append(RP_PORT_ROOT / cs / "projection_matrices.npz")

        if incremental and _should_skip_existing(
            summary_path=summary_path,
            monthly_path=monthly_path,
            input_paths=input_paths,
        ):
            try:
                m = _metrics_from_existing_summary(summary_path)
            except (ValueError, KeyError, OSError) as e:
                print(f"  [{cs}] existing TC unreadable ({e}), recomputing")
            else:
                print(
                    f"  [{cs}] SKIP — reuse existing TC "
                    f"(outputs up-to-date vs inputs; use --force to recompute)"
                )
                rows.append(
                    {
                        "feat1": feat1,
                        "feat2": feat2,
                        "cross_section": cs,
                        **m,
                        "status": "done",
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


def run_uniform(*, grid_base: Path, is_rp: bool, incremental: bool) -> pd.DataFrame:
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

        _, summary_path, monthly_path = _uniform_output_paths(ports_path, kernel_name)
        input_paths = [ports_path, weights_path, PANEL_PATH]
        if is_rp:
            input_paths.append(RP_PORT_ROOT / cs / "projection_matrices.npz")

        if incremental and _should_skip_existing(
            summary_path=summary_path,
            monthly_path=monthly_path,
            input_paths=input_paths,
        ):
            try:
                m = _metrics_from_existing_summary(summary_path)
            except (ValueError, KeyError, OSError) as e:
                print(f"  [{cs}] existing TC unreadable ({e}), recomputing")
            else:
                print(
                    f"  [{cs}] SKIP — reuse existing TC "
                    f"(outputs up-to-date vs inputs; use --force to recompute)"
                )
                rows.append(
                    {
                        "feat1": feat1,
                        "feat2": feat2,
                        "cross_section": cs,
                        **m,
                        "status": "done",
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
    pa.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute every cross-section even when tc_summary and transaction_costs "
            "CSVs already exist and look up-to-date (default skips those for speed)."
        ),
    )
    args = pa.parse_args()

    grid_base = RP_GRID_PATH if args.rp else TREE_GRID_PATH
    is_rp = bool(args.rp)
    incremental = not args.force

    print(
        f"\n  Grid root: {grid_base.resolve()}  "
        f"(RP mode={is_rp}, skip_uniform={args.skip_uniform}, incremental={incremental})"
    )

    all_summaries: dict[str, pd.DataFrame] = {}

    for kernel in KERNEL_KERNELS:
        all_summaries[kernel] = run_kernel(
            kernel, grid_base=grid_base, is_rp=is_rp, incremental=incremental
        )

    if not args.skip_uniform:
        all_summaries["uniform"] = run_uniform(
            grid_base=grid_base, is_rp=is_rp, incremental=incremental
        )

    print(f"\n{'='*60}")
    print("  DONE - median net SR per kernel:")
    for kernel, df in all_summaries.items():
        done = df[df["status"] == "done"]
        if len(done) > 0:
            print(
                f"    {kernel:15s}  median net SR = {done['net_SR'].median():.4f}"
                f"  ({len(done)}/36 cross-sections)"
            )
