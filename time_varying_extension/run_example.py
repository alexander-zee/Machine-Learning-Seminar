"""
Example driver for the time-varying (dual-style) extension.

Run from repository root.

Typical layout after your pipeline:
  - Portfolios: data/results/tree_portfolios/LME_OP_Investment/level_all_excess_combined_filtered.csv
  - Selected k=10: data/results/grid_search/tree/LME_OP_Investment/Selected_Ports_10.csv
  - Panel: data/prepared/panel.parquet

Commands (PowerShell, adjust the cd path):

  cd "C:\\...\\Machine-Learning-Seminar-main"
  python -m time_varying_extension.run_example

  # Same but with Gaussian × exponential time kernel (recency):
  python -m time_varying_extension.run_example --time-decay

  # Custom output folder:
  python -m time_varying_extension.run_example --out time_varying_extension/_my_run

See DUAL_INTERPRETATION_AND_SCOPE.txt for what we take from the Dual Interpretation paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from time_varying_extension.workflow_one_triplet import run_time_varying_one_triplet


def _find_triplet_paths(root: Path) -> tuple[Path, Path]:
    """Return (portfolio_csv, selected_ports_csv) for LME_OP_Investment."""
    layouts = [
        (
            root
            / "data"
            / "results"
            / "tree_portfolios"
            / "LME_OP_Investment"
            / "level_all_excess_combined_filtered.csv",
            root
            / "data"
            / "results"
            / "grid_search"
            / "tree"
            / "LME_OP_Investment"
            / "Selected_Ports_10.csv",
        ),
        (
            root
            / "data"
            / "results"
            / "tree_portfolios"
            / "LME_OP_Investment"
            / "level_all_excess_combined_filtered.csv",
            root
            / "data"
            / "results"
            / "tree_portfolios"
            / "LME_OP_Investment"
            / "Selected_Ports_10.csv",
        ),
        (
            root
            / "paper_data"
            / "tree_portfolio_quantile"
            / "LME_OP_Investment"
            / "level_all_excess_combined_filtered.csv",
            root
            / "paper_data"
            / "tree_portfolio_quantile"
            / "LME_OP_Investment"
            / "Selected_Ports_10.csv",
        ),
    ]
    for port_csv, sel_csv in layouts:
        if port_csv.is_file() and sel_csv.is_file():
            return port_csv, sel_csv
    # Partial match: at least portfolio file
    for port_csv, sel_csv in layouts:
        if port_csv.is_file():
            return port_csv, sel_csv
    return layouts[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Time-varying kernel extension example (one triplet).")
    p.add_argument(
        "--time-decay",
        action="store_true",
        help="Use Gaussian state kernel × exponential time kernel K_E(j).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output directory (default: time_varying_extension/_example_output/...).",
    )
    p.add_argument("--lambda-time", type=float, default=0.95, dest="lambda_time", help="Time decay lambda.")
    p.add_argument("--m-window", type=int, default=120, dest="m_window", help="Time window m (months).")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    portfolio_csv, selected = _find_triplet_paths(root)
    panel = root / "data" / "prepared" / "panel.parquet"

    if args.out:
        out = Path(args.out)
    else:
        sub = "LME_OP_Investment_time_decay" if args.time_decay else "LME_OP_Investment"
        out = root / "time_varying_extension" / "_example_output" / sub

    if not portfolio_csv.is_file():
        raise SystemExit(f"Missing portfolio CSV:\n  {portfolio_csv}\n")
    if not panel.is_file():
        raise SystemExit(
            f"Missing {panel}\nRun: python part_1_portfolio_creation/tree_portfolio_creation/step1_prepare_data.py"
        )
    if not selected.is_file():
        raise SystemExit(
            f"Missing selected ports:\n  {selected}\n"
            "Run pick_best_lambda(..., write_table=True) so Selected_Ports_10.csv exists under grid_search, "
            "or copy that CSV next to your portfolio CSV, or use portfolio_columns_subset in code."
        )

    run_time_varying_one_triplet(
        feat1="OP",
        feat2="Investment",
        portfolio_csv=portfolio_csv,
        panel_parquet=panel,
        output_dir=out,
        selected_ports_csv=selected,
        use_time_decay=args.time_decay,
        time_decay_lambda=args.lambda_time,
        time_window_m=args.m_window,
    )
    print(f"Done. Outputs under:\n  {out.resolve()}")


if __name__ == "__main__":
    main()
