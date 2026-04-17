#!/usr/bin/env python3
"""
One-click helper for the RP **IdioVol** triplets used in Table-style exports.

What it does (repo root)::

  1) Run ``step1_prepare_data.prepare_data()`` (BEME December t-1 lag is default there).
  2) For each triplet: RP Part 1 + Part 2 (uniform/LASSO grid) via ``run_all_rp_cross_sections.py``.
  3) Optionally run Gaussian + Exponential kernel batches for the same triplets.

Typical colleague usage (double-click wrapper on Windows is in ``run_table4_rp_ivol_playbook.ps1``)::

    python run_table4_rp_ivol_playbook.py

Advanced::

    python run_table4_rp_ivol_playbook.py --skip-step1 --kernels gaussian
    python run_table4_rp_ivol_playbook.py --include-size-val-ivol

Notes
-----
- Triplet folder names always follow ``canonical_feat_pair`` ordering.
- ``export_table51_rp_uniform_vs_gaussian.py`` assigns row **Id** after sorting by Uniform SR,
  so the numeric Ids are not stable identifiers. This script uses **internal feature names**.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent


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
        choices=("both", "gaussian", "exponential", "none"),
        default="both",
        help="Which kernel full-fit batch to run after RP Part 1+2.",
    )
    pa.add_argument(
        "--pick-best",
        action="store_true",
        help="Also run pick_best_lambda after RP_Pruning (usually not needed before kernel batches).",
    )
    pa.add_argument(
        "--include-size-val-ivol",
        action="store_true",
        help="Also run Size–Val–IVol (BEME–IdioVol). Default excludes it (common exclusion in reruns).",
    )
    pa.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only.",
    )
    args = pa.parse_args()

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
                *(["--pick-best"] if args.pick_best else []),
            ]
        )

        if args.kernels in ("both", "gaussian"):
            run_maybe([py, "standard_gaussian_rp_all.py", "--feat1", a, "--feat2", b])
        if args.kernels in ("both", "exponential"):
            run_maybe([py, "standard_exponential_rp_all.py", "--feat1", a, "--feat2", b])

    print(
        "\nDone.\n"
        "Optional table export:\n"
        "  python part_3_metrics_collection/export_table51_rp_uniform_vs_gaussian.py --rows all --k 10\n",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
