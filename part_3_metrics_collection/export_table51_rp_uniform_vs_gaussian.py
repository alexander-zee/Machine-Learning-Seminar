#!/usr/bin/env python3
"""
Table 4.1 / 5.1 style: **RP trees** — OOS monthly Sharpe and FF5 alpha [t], uniform vs Gaussian
(and optional Exponential when ``full_fit`` exists under ``rp_tree/exponential/``).

Same logic as ``export_table51_uniform_vs_gaussian.py``, but defaults point at
``data/results/grid_search/rp_tree`` and ``data/results/rp_tree_portfolios``, and
the combined portfolio file is ``level_all_excess_combined.csv`` (no ``_filtered``
step for RP in this repo).

**Row sets**

- ``all`` — all C(9,2)=36 triplets ``(LME, feat1, feat2)`` (same list as AP trees).
- ``no-idiovol`` — C(8,2)=**28** triplets that do **not** use IdioVol (the IdioVol
  characteristic is excluded from this subset, not “rows left out of the table”).

**LaTeX:** Every row in the chosen subset is written to ``--latex-out`` (28 or 36 data rows).
There is no truncation and no “omitted for brevity” block in the generated fragment.

**Runtime:** With FF factors cached once per run (see ``ff5.load_ff5_research_panel``),
28--36 triplets typically finish in **well under a minute** on a local disk (mostly CSV I/O
and OLS). The first Fama--French download adds **a few seconds** depending on the network.
A tqdm bar prints to stderr by default; use ``--no-progress`` to disable.

From repo root::

    python part_3_metrics_collection/export_table51_rp_uniform_vs_gaussian.py
    python part_3_metrics_collection/export_table51_rp_uniform_vs_gaussian.py --rows no-idiovol --latex-out thesis/tables/tab_rp_uniform_gaussian.tex

    # Full LaTeX from an existing CSV (no FF5 re-run; still all rows):
    python part_3_metrics_collection/export_table51_rp_uniform_vs_gaussian.py \\
        --latex-only-from-csv data/results/grid_search/rp_tree/table51_rp_uniform_vs_gaussian_k10_no_idiovol28.csv \\
        --latex-out Figures/tables/tab_rp_uniform_gaussian.tex
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (
    all_triplet_pairs_excluding_secondary,
)
from part_3_metrics_collection.export_table51_uniform_vs_gaussian import build_table


def _fmt_sr(x: float) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) and np.isfinite(x) else "---"


def _fmt_alpha_bracket_cell(x) -> str:
    """CSV/pandas may use NaN for empty Gaussian cells; NaN is truthy in Python, so avoid ``or``."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    if isinstance(x, str) and not x.strip():
        return "---"
    if pd.isna(x):
        return "---"
    return str(x).strip()


def write_latex_booktabs(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    use_longtable: bool = False,
) -> None:
    """Booktabs table: Id, three chars, Uniform / Gaussian / (optional) Exponential blocks.

    Emits **one LaTeX row per DataFrame row** (full table, no elision).
    """
    has_exp = "Exponential_SR" in df.columns
    if has_exp:
        header_block = [
            r"\multicolumn{4}{c}{} & \multicolumn{2}{c}{Uniform} & \multicolumn{2}{c}{Gaussian} & \multicolumn{2}{c}{Exponential} \\",
            r"\cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}",
            r"Id & Char1 & Char2 & Char3 & SR & $\alpha_{\text{FF5}}$ [t] & SR & $\alpha_{\text{FF5}}$ [t] & SR & $\alpha_{\text{FF5}}$ [t] \\",
        ]
        colspec = r"@{}r lll cc cc cc cc@{}"
        ncols_cont = "10"
    else:
        header_block = [
            r"\multicolumn{4}{c}{} & \multicolumn{2}{c}{Uniform Weights (baseline)} & \multicolumn{2}{c}{Gaussian Kernel} \\",
            r"\cmidrule(lr){5-6} \cmidrule(lr){7-8}",
            r"Id & Char1 & Char2 & Char3 & SR & $\alpha_{\text{FF5}}$ [t] & SR & $\alpha_{\text{FF5}}$ [t] \\",
        ]
        colspec = r"@{}r lll cc cc@{}"
        ncols_cont = "8"

    body_lines: list[str] = []
    for _, row in df.iterrows():
        ua = _fmt_alpha_bracket_cell(row["Uniform_FF5_alpha_bracket_t"])
        ga = _fmt_alpha_bracket_cell(row["Gaussian_FF5_alpha_bracket_t"])
        base = (
            f"{int(row['Id'])} & {row['Char1']} & {row['Char2']} & {row['Char3']} & "
            f"{_fmt_sr(row['Uniform_SR'])} & {ua} & {_fmt_sr(row['Gaussian_SR'])} & {ga}"
        )
        if has_exp:
            ea = _fmt_alpha_bracket_cell(row["Exponential_FF5_alpha_bracket_t"])
            base += f" & {_fmt_sr(row['Exponential_SR'])} & {ea}"
        body_lines.append(base + r" \\")

    if use_longtable:
        lines = [
            r"\begin{longtable}{" + colspec + "}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}\\\\",
            r"\toprule",
            *header_block,
            r"\midrule",
            r"\endfirsthead",
            f"\\caption[]{{{caption} (\\textit{{continued}})}}\\\\",
            r"\toprule",
            *header_block,
            r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{ncols_cont}}}{{r}}{{\textit{{Continued on next page}}}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
            *body_lines,
            r"\end{longtable}",
            "",
        ]
    else:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            *header_block,
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    os.chdir(REPO)
    p = argparse.ArgumentParser(
        description="Export Table 4.1/5.1-style CSV for RP trees (uniform vs Gaussian)."
    )
    p.add_argument("--grid-dir", type=Path, default=Path("data/results/grid_search/rp_tree"))
    p.add_argument("--ports-dir", type=Path, default=Path("data/results/rp_tree_portfolios"))
    p.add_argument(
        "--port-name",
        type=str,
        default="level_all_excess_combined.csv",
        help="Combined excess returns matrix under each LME_* folder",
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-train-valid", type=int, default=360)
    p.add_argument(
        "--uniform-sr",
        choices=("summary", "naive-master"),
        default="summary",
    )
    p.add_argument(
        "--rows",
        choices=("all", "no-idiovol"),
        default="no-idiovol",
        help="all: 36 triplets; no-idiovol: 28 triplets (exclude IdioVol).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default: grid-dir/table51_rp_uniform_vs_gaussian_k{k}_{rows}.csv)",
    )
    p.add_argument(
        "--latex-out",
        type=Path,
        default=None,
        help="Optional booktabs .tex fragment",
    )
    p.add_argument(
        "--latex-caption",
        type=str,
        default="Out-of-Sample Performance and Fama--French 5-Factor Alphas "
        "(Uniform, Gaussian, Exponential), Random-Projection Trees",
    )
    p.add_argument(
        "--no-exponential-cols",
        action="store_true",
        help="Omit Exponential columns from CSV/LaTeX (Gaussian + Uniform only).",
    )
    p.add_argument("--latex-label", type=str, default="tab:rp_uniform_vs_gaussian")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm (default is to show progress on stderr).",
    )
    p.add_argument(
        "--latex-only-from-csv",
        type=Path,
        default=None,
        help="Read an existing table51 CSV and write --latex-out only (all rows; no FF5 rebuild).",
    )
    p.add_argument(
        "--latex-longtable",
        action="store_true",
        help="Emit longtable (needs \\usepackage{longtable}); still lists every cross-section.",
    )
    args = p.parse_args()

    if args.latex_only_from_csv is not None:
        if args.latex_out is None:
            p.error("--latex-only-from-csv requires --latex-out")
        csv_path = args.latex_only_from_csv
        if not csv_path.is_file():
            p.error(f"CSV not found: {csv_path}")
        df_csv = pd.read_csv(csv_path)
        write_latex_booktabs(
            df_csv,
            args.latex_out,
            caption=args.latex_caption,
            label=args.latex_label,
            use_longtable=args.latex_longtable,
        )
        print(f"Wrote {args.latex_out} ({len(df_csv)} rows, full table from {csv_path})")
        return

    if args.rows == "all":
        pairs = None
        tag = "all36"
    else:
        pairs = all_triplet_pairs_excluding_secondary("IdioVol")
        tag = "no_idiovol28"

    out = args.out or (args.grid_dir / f"table51_rp_uniform_vs_gaussian_k{args.k}_{tag}.csv")

    df = build_table(
        grid_dir=args.grid_dir,
        ports_dir=args.ports_dir,
        port_name=args.port_name,
        k=args.k,
        n_train_valid=args.n_train_valid,
        uniform_sr_mode=args.uniform_sr,
        pairs=pairs,
        show_progress=not args.no_progress,
        include_exponential=not args.no_exponential_cols,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(out, index=False)
    except PermissionError:
        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        alt = out.with_name(f"{out.stem}_{stamp}{out.suffix}")
        df.to_csv(alt, index=False)
        print(f"Permission denied on {out}; wrote {alt}", file=sys.stderr)
        out = alt

    sp = args.grid_dir / f"ap_pruned_summary_k{args.k}.csv"
    if args.uniform_sr == "summary":
        print(
            f"Uniform SR: from {sp.name}"
            if sp.is_file()
            else f"Uniform SR: missing {sp}, naive fallback per row where needed"
        )
    else:
        print("Uniform SR: naive mean/std on master test returns")
    print(f"Wrote {out} ({len(df)} rows, rows={args.rows})")

    if args.latex_out is not None:
        write_latex_booktabs(
            df,
            args.latex_out,
            caption=args.latex_caption,
            label=args.latex_label,
            use_longtable=args.latex_longtable,
        )
        print(f"Wrote {args.latex_out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
