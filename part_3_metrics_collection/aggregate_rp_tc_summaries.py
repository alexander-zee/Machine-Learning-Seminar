#!/usr/bin/env python3
"""
Collect every per-triplet **TC summary** CSV under the RP grid into **one long CSV**
(and optionally a zip) for easy sharing (e.g. WhatsApp / email).

Looks for files named::

    tc_summary_k{K}_{label}.csv

Examples::

    data/results/grid_search/rp_tree/LME_ST_Rev_LT_Rev/tc_summary_k10_uniform.csv
    data/results/grid_search/rp_tree/gaussian/LME_OP_AC/tc_summary_k10_gaussian.csv

Skips batch roll-ups like ``uniform/tc_summary_all_k10.csv`` (different layout).

From repo root::

    python part_3_metrics_collection/aggregate_rp_tc_summaries.py
    python part_3_metrics_collection/aggregate_rp_tc_summaries.py --k 10 --zip
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _cross_section_from_path(path: Path, grid_root: Path) -> str | None:
    """Return ``LME_*`` folder name; None if path layout is unexpected."""
    try:
        rel = path.relative_to(grid_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    # .../LME_x_y/tc_summary_*.csv
    if parts[0].startswith("LME_") and len(parts) == 2:
        return parts[0]
    # .../gaussian/LME_x_y/tc_summary_*.csv  (or exponential / gaussian-tms)
    if len(parts) == 3 and parts[1].startswith("LME_"):
        return parts[1]
    return None


def _parse_kernel_from_stem(stem: str) -> str | None:
    """``tc_summary_k10_uniform`` -> ``uniform``."""
    prefix = "tc_summary_"
    if not stem.startswith(prefix):
        return None
    rest = stem[len(prefix) :]
    # rest = "k10_uniform" or "k10_gaussian-tms" — split on first "_" after the k\d+ part
    if not rest.startswith("k"):
        return None
    idx = rest.find("_")
    if idx < 0:
        return None
    return rest[idx + 1 :] or None


def collect_summaries(*, grid_root: Path, k: int) -> pd.DataFrame:
    pattern = f"tc_summary_k{k}_*.csv"
    rows: list[dict] = []
    for path in sorted(grid_root.rglob(pattern)):
        if "tc_summary_all" in path.name:
            continue
        cs = _cross_section_from_path(path, grid_root)
        kernel = _parse_kernel_from_stem(path.stem)
        if cs is None or kernel is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Skip (read error) {path}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue
        rec = df.iloc[0].to_dict()
        rec["cross_section"] = cs
        rec["kernel"] = kernel
        rec["source_file"] = str(path.as_posix())
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Stable column order: identifiers first
    front = ["cross_section", "kernel", "source_file"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def main() -> None:
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    pa = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pa.add_argument(
        "--grid-root",
        type=Path,
        default=Path("data/results/grid_search/rp_tree"),
        help="RP grid root (contains LME_* and kernel subfolders).",
    )
    pa.add_argument("--k", type=int, default=10, help="Portfolio count k in filenames.")
    pa.add_argument(
        "--out",
        type=Path,
        default=Path("data/results/diagnostics/rp_tc_bundle/rp_tc_all_summaries_k10.csv"),
        help="Combined long-format CSV path.",
    )
    pa.add_argument(
        "--zip",
        action="store_true",
        help="Also write a .zip next to --out containing the combined CSV.",
    )
    args = pa.parse_args()

    grid_root = (REPO / args.grid_root).resolve() if not args.grid_root.is_absolute() else args.grid_root
    if not grid_root.is_dir():
        raise SystemExit(f"Grid root not found: {grid_root}")

    df = collect_summaries(grid_root=grid_root, k=args.k)
    if df.empty:
        raise SystemExit(f"No tc_summary_k{args.k}_*.csv files found under {grid_root}")

    out_path = (REPO / args.out).resolve() if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path}  ({len(df)} rows; "
        f"{df['cross_section'].nunique()} cross-sections; "
        f"kernels={sorted(df['kernel'].unique())})"
    )

    if args.zip:
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=out_path.name)
        print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
