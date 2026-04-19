#!/usr/bin/env python3
"""
Collect every per-triplet **TC summary** CSV under the RP grid into **one long CSV**
(and optionally a zip) for easy sharing (e.g. WhatsApp / email).

Looks for files named::

    tc_summary_k{K}_{label}.csv

Examples::

    data/results/grid_search/rp_tree/LME_ST_Rev_LT_Rev/tc_summary_k10_uniform.csv
    data/results/grid_search/rp_tree/LME_ST_Rev_LT_Rev/tc_summary_k10_gaussian.csv

(Legacy paths under ``<kernel>/LME_*/full_fit/`` are still picked up if present; flat
``LME_*`` rows win on duplicates.)

Skips batch roll-ups like ``uniform/tc_summary_all_k10.csv`` (different layout).

**Wide monthlies (triplets as columns):** use ``--wide-monthlies`` to merge all
``transaction_costs_k{K}_{label}.csv`` files for one kernel label (default
``uniform``) on ``(yy, mm)`` so each **cross-section** becomes its own column(s)
— handy for Excel / sharing one panel instead of 36 loose files.

From repo root::

    python part_3_metrics_collection/aggregate_rp_tc_summaries.py
    python part_3_metrics_collection/aggregate_rp_tc_summaries.py --k 10 --zip
    python part_3_metrics_collection/aggregate_rp_tc_summaries.py --wide-monthlies --no-summaries --zip
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
    # .../LME_x_y/tc_summary_*.csv  (uniform RP: outputs next to Selected_Ports)
    if parts[0].startswith("LME_") and len(parts) == 2:
        return parts[0]
    # .../<kernel>/LME_x_y/tc_summary_*.csv  (unusual)
    if len(parts) == 3 and parts[1].startswith("LME_"):
        return parts[1]
    # .../<kernel>/LME_x_y/full_fit/tc_summary_*.csv  (Gaussian / Exp / TMS from tc_batch_runner)
    if (
        len(parts) == 4
        and parts[1].startswith("LME_")
        and parts[2] == "full_fit"
    ):
        return parts[1]
    return None


def _parse_kernel_suffix_from_stem(stem: str, prefix: str) -> str | None:
    """``tc_summary_k10_uniform`` / ``transaction_costs_k10_gaussian`` -> kernel suffix."""
    if not stem.startswith(prefix):
        return None
    rest = stem[len(prefix) :]
    if not rest.startswith("k"):
        return None
    idx = rest.find("_")
    if idx < 0:
        return None
    return rest[idx + 1 :] or None


def _parse_kernel_from_stem(stem: str) -> str | None:
    """``tc_summary_k10_uniform`` -> ``uniform``."""
    return _parse_kernel_suffix_from_stem(stem, "tc_summary_")


def collect_summaries(*, grid_root: Path, k: int) -> pd.DataFrame:
    pattern = f"tc_summary_k{k}_*.csv"
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    paths = sorted(
        grid_root.rglob(pattern),
        key=lambda p: (1 if "full_fit" in p.parts else 0, str(p)),
    )
    for path in paths:
        if "tc_summary_all" in path.name:
            continue
        cs = _cross_section_from_path(path, grid_root)
        kernel = _parse_kernel_from_stem(path.stem)
        if cs is None or kernel is None:
            continue
        dedupe_key = (cs, kernel)
        if dedupe_key in seen:
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
        seen.add(dedupe_key)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Stable column order: identifiers first
    front = ["cross_section", "kernel", "source_file"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def collect_transaction_costs_wide(
    *,
    grid_root: Path,
    k: int,
    kernel_label: str,
    metric: str,
) -> pd.DataFrame:
    """
    One row per calendar month; each triplet is one column (``metric``) or three
    columns if ``metric == "all"`` (``*_gross_return``, ``*_tc``, ``*_net_return``).
    """
    pattern = f"transaction_costs_k{k}_{kernel_label}.csv"
    merged: pd.DataFrame | None = None
    keys = ("yy", "mm")
    paths = sorted(
        grid_root.rglob(pattern),
        key=lambda p: (1 if "full_fit" in p.parts else 0, str(p)),
    )
    for path in paths:
        if "diagnostics" in path.parts:
            continue
        cs = _cross_section_from_path(path, grid_root)
        if cs is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Skip (read error) {path}: {e}", file=sys.stderr)
            continue
        if df.empty or not all(c in df.columns for c in keys):
            continue
        if merged is not None:
            if metric == "all":
                if f"{cs}_net_return" in merged.columns:
                    continue
            elif cs in merged.columns:
                continue
        if metric == "all":
            need = ["gross_return", "tc", "net_return"]
            if not all(c in df.columns for c in need):
                continue
            piece = df[list(keys) + need].copy()
            piece = piece.rename(
                columns={
                    "gross_return": f"{cs}_gross_return",
                    "tc": f"{cs}_tc",
                    "net_return": f"{cs}_net_return",
                }
            )
        else:
            if metric not in df.columns:
                continue
            piece = df[list(keys) + [metric]].rename(columns={metric: cs})
        if merged is None:
            merged = piece
        else:
            merged = merged.merge(piece, on=list(keys), how="outer")
    if merged is None or merged.empty:
        return pd.DataFrame()
    # yy, mm first; then triplet columns sorted for stable diffs
    rest = [c for c in merged.columns if c not in keys]
    rest_sorted = sorted(rest)
    return merged[list(keys) + rest_sorted]


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
        help="Also write a .zip next to each written CSV (same basename).",
    )
    pa.add_argument(
        "--no-summaries",
        action="store_true",
        help="Skip the long-format tc_summary merge (only use with --wide-monthlies).",
    )
    pa.add_argument(
        "--wide-monthlies",
        action="store_true",
        help="Merge transaction_costs_k{K}_{label}.csv into one wide table (yy, mm + triplet columns).",
    )
    pa.add_argument(
        "--kernel-label",
        type=str,
        default="uniform",
        help="Kernel suffix in filenames for --wide-monthlies (e.g. uniform, gaussian, gaussian-tms).",
    )
    pa.add_argument(
        "--wide-metric",
        choices=("net_return", "gross_return", "tc", "all"),
        default="net_return",
        help="Which column(s) from each transaction_costs file become triplet columns.",
    )
    pa.add_argument(
        "--wide-out",
        type=Path,
        default=None,
        help="Output path for wide monthlies (default: diagnostics/rp_tc_bundle/...wide...).",
    )
    args = pa.parse_args()

    grid_root = (REPO / args.grid_root).resolve() if not args.grid_root.is_absolute() else args.grid_root
    if not grid_root.is_dir():
        raise SystemExit(f"Grid root not found: {grid_root}")

    if not args.wide_monthlies and args.no_summaries:
        raise SystemExit("Nothing to do: pass --wide-monthlies or omit --no-summaries.")

    bundle_dir = Path("data/results/diagnostics/rp_tc_bundle")

    if args.wide_monthlies:
        wide_df = collect_transaction_costs_wide(
            grid_root=grid_root,
            k=args.k,
            kernel_label=args.kernel_label,
            metric=args.wide_metric,
        )
        if wide_df.empty:
            raise SystemExit(
                f"No transaction_costs_k{args.k}_{args.kernel_label}.csv files found under {grid_root}"
            )
        wide_out = args.wide_out
        if wide_out is None:
            mtag = args.wide_metric if args.wide_metric != "all" else "allmetrics"
            wide_out = bundle_dir / f"rp_tc_monthly_wide_k{args.k}_{args.kernel_label}_{mtag}.csv"
        wide_path = (REPO / wide_out).resolve() if not wide_out.is_absolute() else wide_out
        wide_path.parent.mkdir(parents=True, exist_ok=True)
        wide_df.to_csv(wide_path, index=False)
        ncols = len([c for c in wide_df.columns if c not in ("yy", "mm")])
        print(f"Wrote {wide_path}  ({len(wide_df)} months × {ncols} triplet columns)")
        if args.zip:
            zpath = wide_path.with_suffix(".zip")
            with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(wide_path, arcname=wide_path.name)
            print(f"Wrote {zpath}")

    if not args.no_summaries:
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
