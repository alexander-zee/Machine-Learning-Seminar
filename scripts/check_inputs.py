#!/usr/bin/env python3
"""
Preflight: verify required user-supplied files exist before running the research pipeline.

Usage (from repository root):
    python scripts/check_inputs.py

Exit code 0 if all required files are present; 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

REQUIRED: list[tuple[Path, str]] = [
    (
        REPO / "data" / "raw" / "FINALdataset.csv",
        "Main stock panel for step1_prepare_data (see data/raw/README.md).",
    ),
    (
        REPO / "data" / "raw" / "rf_factor.csv",
        "Monthly risk-free rate for combine_trees (percent units).",
    ),
    (
        REPO / "data" / "factor" / "tradable_factors.csv",
        "Tradable factor panel for Table 3–style SDF regressions in Part 3.",
    ),
]


def main() -> int:
    missing: list[tuple[Path, str]] = [(p, note) for p, note in REQUIRED if not p.is_file()]
    if not missing:
        print("check_inputs: all required files are present.")
        print("  You can run:  python run_full_research_pipeline.py")
        return 0

    print("check_inputs: missing required file(s). Add them, then rerun this script.\n")
    for p, note in missing:
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"  - {p}")
        print(f"    {note}\n")
    print("See README.md (Quickstart) and data/*/README.md for exact paths.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
