#!/usr/bin/env python3
"""
One-click helper for the RP **IdioVol** triplets used in Table-style exports.

What it does (repo root)::

  1) Run ``step1_prepare_data.prepare_data()`` (BEME December t-1 lag is default there).
  2) For each triplet: RP Part 1 + Part 2 (LASSO / pruning grid) via ``run_all_rp_cross_sections.py``.
  3) By default also runs ``pick_best_lambda`` (``--pick-best``) so ``Selected_Ports_*`` exist for
     uniform-style SR/alpha exports.
  4) By default runs kernel full-fit batches: **Gaussian**, **Exponential**, and **Gaussian–TMS**
     (``standard_gaussian_tms_rp_all.py``; needs ``TMS`` in ``data/state_variables.csv``).
     Use ``--kernels exp-tms`` to run **only** Exponential + Gaussian–TMS (skips Gaussian; saves time).
  5) By default writes **SR + FF5 alpha** artefacts:
     - ``export_table51_rp_uniform_vs_gaussian.py`` → Uniform + Gaussian + Exponential (36 rows).
     - ``rp_oos_ff5_multikernel_table.py`` → TMS (and Exponential too if ``--kernels exp-tms``).

Outputs land under ``data/results/diagnostics/table4_rp_ivol_playbook/``.

Typical colleague usage (double-click wrapper on Windows is in ``run_table4_rp_ivol_playbook.ps1``)::

    python run_table4_rp_ivol_playbook.py

Advanced::

    python run_table4_rp_ivol_playbook.py --skip-step1 --kernels both
    python run_table4_rp_ivol_playbook.py --kernels exp-tms
    python run_table4_rp_ivol_playbook.py --include-size-val-ivol
    python run_table4_rp_ivol_playbook.py --no-pick-best --skip-metrics-export

Notes
-----
- Triplet folder names always follow ``canonical_feat_pair`` ordering.
- ``export_table51_rp_uniform_vs_gaussian.py`` assigns row **Id** after sorting by Uniform SR,
  so the numeric Ids are not stable identifiers. This script uses **internal feature names**.
- TMS batch writes under ``grid_search/rp_tree/gaussian-tms/``; metrics CSV merges are left to the
  thesis pipeline if you need a single four-kernel LaTeX table.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent
METRICS_OUT = Path("data/results/diagnostics/table4_rp_ivol_playbook")

EXPORT_TABLE51_RP = REPO / "part_3_metrics_collection/export_table51_rp_uniform_vs_gaussian.py"
MULTIKERNEL_TMS = REPO / "part_3_metrics_collection/rp_oos_ff5_multikernel_table.py"


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
        canonical_feat_pair,
    )

    pa = argparse.ArgumentParser(description="Run RP IdioVol triplets (Table 4 helper).")
    pa.add_argument("--skip-step1", action="store_true", help="Skip step1_prepare_data.")
    pa.add_argument(
        "--kernels",
        choices=("all", "both", "exp-tms", "gaussian", "exponential", "tms", "none"),
        default="all",
        help=(
            "Kernel full-fit batches after RP Part 1+2. "
            "'all'=Gaussian+Exponential+Gaussian-TMS (default); "
            "'both'=Gaussian+Exponential only (legacy); "
            "'exp-tms'=Exponential+Gaussian-TMS only (skip Gaussian; saves time); "
            "'none'=skip kernel batches."
        ),
    )
    pa.add_argument(
        "--no-pick-best",
        action="store_true",
        help="Skip pick_best_lambda after RP_Pruning (default is to run it).",
    )
    pa.add_argument(
        "--include-size-val-ivol",
        action="store_true",
        help="Also run Size–Val–IVol (BEME–IdioVol). Default excludes it (common exclusion in reruns).",
    )
    pa.add_argument(
        "--skip-metrics-export",
        action="store_true",
        help="Skip SR/alpha CSV exports at the end (export_table51 + multikernel TMS).",
    )
    pa.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only.",
    )
    args = pa.parse_args()
    pick_best = not args.no_pick_best

    # Internal (code) names — matches ``cross_section_triplets.FEATS_LIST`` / folder names.
    triplets: list[tuple[str, str]] = [
        ("ST_Rev", "IdioVol"),  # Size–SRev–IVol
        ("LT_Rev", "IdioVol"),  # Size–LRev–IVol
        ("r12_2", "IdioVol"),  # Size–Mom–IVol
        ("LTurnover", "IdioVol"),  # Size–Turn–IVol
        ("Investment", "IdioVol"),  # Size–Inv–IVol
        ("AC", "IdioVol"),  # Size–Acc–IVol
        ("OP", "IdioVol"),  # Size–Prof–IVol
    ]
    if args.include_size_val_ivol:
        triplets.append(("BEME", "IdioVol"))

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    def run_maybe(cmd: list[str]) -> None:
        if args.dry_run:
            print("\n[DRY] " + " ".join(cmd), flush=True)
            return
        _run(cmd, env=env)

    if not args.skip_step1:
        run_maybe([sys.executable, "part_1_portfolio_creation/tree_portfolio_creation/step1_prepare_data.py"])

    py = sys.executable
    k = 10
    kernels = args.kernels

    for f1, f2 in triplets:
        a, b = canonical_feat_pair(f1, f2)
        run_maybe(
            [
                py,
                "run_all_rp_cross_sections.py",
                "--feat1",
                a,
                "--feat2",
                b,
                *(["--pick-best"] if pick_best else []),
            ]
        )

        if kernels in ("all", "both", "gaussian"):
            run_maybe([py, "standard_gaussian_rp_all.py", "--feat1", a, "--feat2", b])
        if kernels in ("all", "both", "exponential", "exp-tms"):
            run_maybe([py, "standard_exponential_rp_all.py", "--feat1", a, "--feat2", b])
        if kernels in ("all", "tms", "exp-tms"):
            run_maybe([py, "standard_gaussian_tms_rp_all.py", "--feat1", a, "--feat2", b])

    if not args.skip_metrics_export:
        out_dir = REPO / METRICS_OUT
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        table51_csv = out_dir / f"table51_rp_uniform_vs_gaussian_k{k}_all36.csv"

        if EXPORT_TABLE51_RP.is_file():
            run_maybe(
                [
                    py,
                    "part_3_metrics_collection/export_table51_rp_uniform_vs_gaussian.py",
                    "--rows",
                    "all",
                    "--k",
                    str(k),
                    "--out",
                    str(table51_csv),
                    "--no-progress",
                ]
            )
        else:
            print(
                f"\nWARNING: {EXPORT_TABLE51_RP} not found — skip Uniform/Gaussian/Exponential "
                "SR+alpha CSV export. Add that script to the repo or copy it from your thesis branch.\n",
                flush=True,
            )

        if MULTIKERNEL_TMS.is_file():
            mk_kernels = (
                "exponential",
                "gaussian-tms",
            ) if kernels == "exp-tms" else ("gaussian-tms",)
            run_maybe(
                [
                    py,
                    "part_3_metrics_collection/rp_oos_ff5_multikernel_table.py",
                    "--kernels",
                    *mk_kernels,
                    "--k",
                    str(k),
                    "--out-dir",
                    str(out_dir),
                    "--no-sleep-guard",
                ]
            )
        else:
            print(
                f"\nWARNING: {MULTIKERNEL_TMS} not found — skip multikernel SR+alpha CSV export.\n",
                flush=True,
            )

    metrics_msg = (
        f"Metrics CSVs (SR + FF5 alpha): {REPO / METRICS_OUT}\n"
        f"  - Uniform + Gaussian + Exponential: {METRICS_OUT / f'table51_rp_uniform_vs_gaussian_k{k}_all36.csv'}\n"
        f"  - TMS (+ Exponential if --kernels exp-tms) multikernel wide/long: "
        f"{METRICS_OUT}/rp_oos_ff5_multikernel_*_k{k}.csv\n"
        if not args.skip_metrics_export
        else "(Metrics export skipped.)\n"
    )
    latex_hint = ""
    if not args.skip_metrics_export and EXPORT_TABLE51_RP.is_file():
        latex_hint = (
            "Optional LaTeX (three kernels) from the table51 CSV:\n"
            f"  python part_3_metrics_collection/export_table51_rp_uniform_vs_gaussian.py "
            f"--latex-only-from-csv {METRICS_OUT / f'table51_rp_uniform_vs_gaussian_k{k}_all36.csv'} "
            f"--latex-out Figures/tables/tab_rp_three_kernels_from_playbook.tex\n"
        )
    print(
        "\nDone.\n"
        f"Kernel batches: kernels={kernels!r}, pick_best={pick_best}.\n"
        + metrics_msg
        + latex_hint,
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
