#!/usr/bin/env python3
"""
Out-of-sample **monthly Sharpe ratio** and **FF5 alpha** (with *t*-stat) for **RP trees**,
for the wide thesis table (Uniform vs Gaussian vs Exponential vs Gaussian–TMS).

This script is **self-contained**: it does **not** import or modify
``ff5_batch_regression.py``. It reads the same inputs:

  ``data/results/grid_search/rp_tree/<kernel>/LME_* /full_fit/full_fit_detail_k{k}.csv``
  ``data/raw/F-F_Research_Data_5_Factors_2x3.csv``

Row **Id** order matches ``\\label{tab:rp_uniform_vs_gaussian}`` in the draft (36 triplets;
rows involving **IVol** may stay empty until RP portfolios include that cross-section).

**Windows (optional):** ``SetThreadExecutionState(ES_SYSTEM_REQUIRED)`` while running so
idle sleep is less likely (use Power settings for long grid jobs).

Example::

    cd <repo>
    python part_3_metrics_collection/rp_oos_ff5_multikernel_table.py
    python part_3_metrics_collection/rp_oos_ff5_multikernel_table.py --kernels gaussian exponential

FF5 inputs: by default the script looks for ``data/raw/F-F_Research_Data_5_Factors_2x3.csv``
(Ken French CSV). If that file is absent, factors are loaded via
``ff5.load_ff5_research_panel()`` (network). Override with ``--ff5-csv PATH``.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
    canonical_feat_pair,
    triplet_subdir_name,
)

# ── Same convention as ff5_batch_regression (do not change that file) ─────────
N_TRAIN_VALID = 360
Y_MIN, Y_MAX = 1964, 2016
FF5_CSV = Path("data/raw/F-F_Research_Data_5_Factors_2x3.csv")
RP_GRID = Path("data/results/grid_search/rp_tree")

# LaTeX short labels in the table → internal portfolio codes
_LABEL_TO_CODE: dict[str, str] = {
    "SRev": "ST_Rev",
    "LRev": "LT_Rev",
    "Mom": "r12_2",
    "Prof": "OP",
    "Val": "BEME",
    "Inv": "Investment",
    "Acc": "AC",
    "Turn": "LTurnover",
    "IVol": "IdioVol",
}

# (Id, Char2, Char3) as in ``\\caption{... Random-Projection Trees}`` table body
THESIS_TABLE_ROWS: list[tuple[int, str, str]] = [
    (1, "SRev", "Turn"),
    (2, "SRev", "LRev"),
    (3, "SRev", "IVol"),
    (4, "SRev", "Acc"),
    (5, "LRev", "Turn"),
    (6, "Mom", "LRev"),
    (7, "Acc", "Turn"),
    (8, "LRev", "IVol"),
    (9, "Mom", "Inv"),
    (10, "Mom", "Acc"),
    (11, "Inv", "SRev"),
    (12, "Mom", "Turn"),
    (13, "Prof", "SRev"),
    (14, "LRev", "Acc"),
    (15, "Mom", "SRev"),
    (16, "Prof", "Acc"),
    (17, "Mom", "IVol"),
    (18, "Turn", "IVol"),
    (19, "Inv", "IVol"),
    (20, "Acc", "IVol"),
    (21, "Inv", "Turn"),
    (22, "Inv", "Acc"),
    (23, "Inv", "LRev"),
    (24, "Prof", "IVol"),
    (25, "Mom", "Prof"),
    (26, "Prof", "Turn"),
    (27, "Prof", "LRev"),
    (28, "Prof", "Inv"),
    (29, "Val", "SRev"),
    (30, "Val", "LRev"),
    (31, "Val", "Acc"),
    (32, "Val", "Inv"),
    (33, "Val", "Mom"),
    (34, "Val", "Turn"),
    (35, "Val", "IVol"),
    (36, "Val", "Prof"),
]

DEFAULT_KERNELS = ("uniform", "gaussian", "exponential", "gaussian-tms")


def _generate_dates(y_min: int = Y_MIN, y_max: int = Y_MAX) -> np.ndarray:
    dates: list[int] = []
    for y in range(y_min, y_max + 1):
        for m in range(1, 13):
            dates.append(int(f"{y}{m:02d}"))
    return np.array(dates, dtype=int)


def _load_ff5_from_french_csv(path: Path) -> pd.DataFrame:
    ff5 = pd.read_csv(path, skiprows=4, index_col=0)
    ff5 = ff5[ff5.index.astype(str).str.strip().str.match(r"^\d{6}$")]
    ff5.index = ff5.index.astype(int)
    ff5.index.name = "Date"
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    ff5[factor_cols] = ff5[factor_cols].apply(pd.to_numeric, errors="coerce")
    ff5[factor_cols] = ff5[factor_cols] / 100.0
    return ff5[factor_cols]


def _load_ff5_from_datareader() -> pd.DataFrame:
    """Same monthly FF5 as ``ff5.run_ff5_regression*`` when no local CSV is present."""
    from part_3_metrics_collection.ff5 import load_ff5_research_panel

    ff5 = load_ff5_research_panel()
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    out = ff5[factor_cols].apply(pd.to_numeric, errors="coerce") / 100.0
    return out


def _load_ff5_panel(csv_override: Path | None) -> tuple[pd.DataFrame, str]:
    """
    Return (factor panel in **decimals** per month, description of source).
    """
    if csv_override is not None:
        if not csv_override.is_file():
            raise SystemExit(f"FF5 CSV not found: {csv_override}")
        return _load_ff5_from_french_csv(csv_override), str(csv_override)
    if FF5_CSV.is_file():
        return _load_ff5_from_french_csv(FF5_CSV), str(FF5_CSV)
    print(
        f"Note: {FF5_CSV} not found — using pandas_datareader Fama–French factors "
        f"(same source as part_3_metrics_collection.ff5.load_ff5_research_panel).",
        flush=True,
    )
    return _load_ff5_from_datareader(), "pandas_datareader (F-F_Research_Data_5_Factors_2x3)"


def _load_hyperparams(grid_root: Path, kernel_name: str, subdir: str, k: int) -> dict:
    path = grid_root / kernel_name / subdir / "full_fit" / f"full_fit_summary_k{k}.csv"
    if not path.is_file():
        return {"lambda0": None, "lambda2": None, "h": None}
    row = pd.read_csv(path).iloc[0]
    h = None
    if "h" in row.index and pd.notna(row["h"]):
        h = float(row["h"])
    return {
        "lambda0": float(row["lambda0"]),
        "lambda2": float(row["lambda2"]),
        "h": h,
    }


def _regress_one(
    grid_root: Path,
    kernel_name: str,
    subdir: str,
    feat1: str,
    feat2: str,
    ff5: pd.DataFrame,
    all_dates: np.ndarray,
    k: int,
) -> dict:
    detail_path = grid_root / kernel_name / subdir / "full_fit" / f"full_fit_detail_k{k}.csv"
    base = {
        "cross_section": subdir,
        "char1": "LME",
        "char2": feat1,
        "char3": feat2,
    }
    if not detail_path.is_file():
        return {
            **base,
            "status": "missing",
            "sr": np.nan,
            "alpha_ff5": np.nan,
            "alpha_ff5_tstat": np.nan,
        }

    detail = pd.read_csv(detail_path)
    rets = detail["excess_return"].values
    test_dates = all_dates[N_TRAIN_VALID : N_TRAIN_VALID + len(rets)]
    port_df = pd.DataFrame({"Date": test_dates, "ret": rets})
    merged = pd.merge(port_df, ff5, left_on="Date", right_index=True, how="inner")
    if merged.empty:
        return {
            **base,
            "status": "merge_failed",
            "sr": np.nan,
            "alpha_ff5": np.nan,
            "alpha_ff5_tstat": np.nan,
        }

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    X = sm.add_constant(merged[factor_cols])
    Y = merged["ret"]
    model = sm.OLS(Y, X).fit()
    r = merged["ret"]
    std_r = r.std(ddof=1)
    sr = float(r.mean() / std_r) if std_r > 0 else float("nan")
    hp = _load_hyperparams(grid_root, kernel_name, subdir, k)
    return {
        **base,
        "status": "ok",
        "sr": sr,
        "alpha_ff5": float(model.params["const"]),
        "alpha_ff5_tstat": float(model.tvalues["const"]),
        "lambda0": hp["lambda0"],
        "lambda2": hp["lambda2"],
        "h": hp["h"],
    }


def _subdir_from_labels(lab2: str, lab3: str) -> tuple[str, str, str]:
    """Return (subdir, feat1_canon, feat2_canon) for ``LME_*`` folder."""
    c2 = _LABEL_TO_CODE[lab2]
    c3 = _LABEL_TO_CODE[lab3]
    f1, f2 = canonical_feat_pair(c2, c3)
    return triplet_subdir_name(f1, f2), f1, f2


@contextmanager
def _prevent_windows_idle_sleep():
    if sys.platform != "win32":
        yield
        return
    import ctypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    k32 = ctypes.windll.kernel32
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)


def _fmt_cell(sr: float, alpha: float, tstat: float, status: str) -> str:
    if status != "ok" or not (np.isfinite(sr) and np.isfinite(alpha) and np.isfinite(tstat)):
        return "--- & ---"
    return f"{sr:.2f} & {alpha:.4f} [{tstat:.2f}]"


def main() -> None:
    os.chdir(REPO)
    pa = argparse.ArgumentParser(description="RP OOS SR + FF5 alpha wide table (multi-kernel).")
    pa.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    pa.add_argument("--k", type=int, default=10)
    pa.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/results/diagnostics/rp_oos_ff5_multikernel"),
    )
    pa.add_argument("--no-sleep-guard", action="store_true")
    pa.add_argument("--print-latex", action="store_true", help="Print LaTeX body rows to stdout.")
    pa.add_argument(
        "--only-ids",
        type=str,
        default=None,
        help="Comma-separated table Ids to run (e.g. 31,32,33,34 for the four Size–Value rows).",
    )
    pa.add_argument(
        "--ff5-csv",
        type=Path,
        default=None,
        help=(
            "Ken French F-F_Research_Data_5_Factors_2x3.csv path. "
            f"If omitted, uses {FF5_CSV} when present; otherwise downloads factors via pandas_datareader."
        ),
    )
    args = pa.parse_args()

    table_rows = THESIS_TABLE_ROWS
    out_suffix = ""
    if args.only_ids:
        want = {int(x.strip()) for x in args.only_ids.split(",") if x.strip()}
        table_rows = [r for r in THESIS_TABLE_ROWS if r[0] in want]
        if not table_rows:
            raise SystemExit(f"No rows match --only-ids {args.only_ids!r}")
        out_suffix = "_ids_" + "_".join(str(i) for i in sorted(want))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ff5, ff5_src = _load_ff5_panel(args.ff5_csv)
    print(f"FF5 factors: {ff5_src}", flush=True)
    all_dates = _generate_dates()

    long_rows: list[dict] = []
    sleep_cm = nullcontext() if args.no_sleep_guard else _prevent_windows_idle_sleep()

    with sleep_cm:
        for kn in args.kernels:
            print(f"\nKernel: {kn}", flush=True)
            for row_id, lab2, lab3 in table_rows:
                subdir, f1, f2 = _subdir_from_labels(lab2, lab3)
                out = _regress_one(RP_GRID, kn, subdir, f1, f2, ff5, all_dates, args.k)
                long_rows.append(
                    {
                        "Id": row_id,
                        "Char2_display": lab2,
                        "Char3_display": lab3,
                        "kernel": kn,
                        "cross_section": subdir,
                        "status": out["status"],
                        "sr": out["sr"],
                        "alpha_ff5": out["alpha_ff5"],
                        "alpha_ff5_tstat": out["alpha_ff5_tstat"],
                        "lambda0": out.get("lambda0"),
                        "lambda2": out.get("lambda2"),
                        "h": out.get("h"),
                    }
                )
                if out["status"] == "ok":
                    print(
                        f"  Id {row_id:2d} {subdir:28s}  SR={out['sr']:.3f}  "
                        f"a={out['alpha_ff5']:+.4f} [t={out['alpha_ff5_tstat']:+.2f}]",
                        flush=True,
                    )
                else:
                    print(f"  Id {row_id:2d} {subdir:28s}  [{out['status']}]", flush=True)

    long_df = pd.DataFrame(long_rows)
    long_path = args.out_dir / f"rp_oos_ff5_multikernel_long_k{args.k}{out_suffix}.csv"
    long_df.to_csv(long_path, index=False)
    print(f"\nLong format → {long_path}", flush=True)

    # Wide: one row per Id, kernels as columns
    wide = long_df.pivot_table(
        index=["Id", "Char2_display", "Char3_display"],
        columns="kernel",
        values=["sr", "alpha_ff5", "alpha_ff5_tstat", "status"],
        aggfunc="first",
    )
    # Flatten MultiIndex in **table column order** (each kernel: SR, α, t, status)
    metric_order = ("sr", "alpha_ff5", "alpha_ff5_tstat", "status")
    flat_cols: list[str] = []
    for kn in args.kernels:
        for m in metric_order:
            flat_cols.append(f"{m}_{kn}")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    present = [c for c in flat_cols if c in wide.columns]
    wide = wide[["Id", "Char2_display", "Char3_display", *present]]
    wide_path = args.out_dir / f"rp_oos_ff5_multikernel_wide_k{args.k}{out_suffix}.csv"
    wide.to_csv(wide_path, index=False)
    print(f"Wide format  → {wide_path}", flush=True)

    if args.print_latex:
        # Build lookup (Id, kernel) -> record
        lut: dict[tuple[int, str], dict] = {}
        for r in long_rows:
            lut[(int(r["Id"]), str(r["kernel"]))] = r
        print("% Paste below \\midrule (check bold / empty rows manually)\n")
        for row_id, lab2, lab3 in table_rows:
            cells = []
            for kn in args.kernels:
                rec = lut.get((row_id, kn), {})
                cells.append(
                    _fmt_cell(
                        float(rec.get("sr", np.nan)),
                        float(rec.get("alpha_ff5", np.nan)),
                        float(rec.get("alpha_ff5_tstat", np.nan)),
                        str(rec.get("status", "")),
                    )
                )
            line = (
                f"{row_id} & Size & {lab2} & {lab3} & "
                + " & ".join(cells)
                + " \\\\"
            )
            print(line)


if __name__ == "__main__":
    main()
