"""
Minimal end-to-end helper for the time-varying extension demos/tests.

This module exists primarily so ``tests/test_research_figures.py`` can import::

    from time_varying_extension.workflow_one_triplet import run_time_varying_one_triplet

The implementation is intentionally lightweight: it reads a toy ``ports.csv`` and
``panel.parquet`` (as produced by the unit test), constructs a compact state
matrix, and delegates plotting/exports to ``research_figures.write_research_figure_bundle``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .research_figures import write_research_figure_bundle


def _load_portfolio_subset(
    portfolio_csv: Path,
    selected_ports_csv: Path,
    columns: Iterable[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load primitive excess-return columns used for TV research figures/tables.

    ``portfolio_csv`` is expected to be a wide CSV of monthly excess returns with a
    header row of column names (strings).

    ``Selected_Ports_k.csv`` in this repo is often **headerless**, where the first row
    encodes the selected primitive column ids/names matching ``portfolio_csv`` columns.
    """
    ports_df = pd.read_csv(portfolio_csv)
    if columns is not None:
        cols = [c for c in columns]
        missing = [c for c in cols if c not in ports_df.columns]
        if missing:
            raise ValueError(f"Missing portfolio columns in {portfolio_csv}: {missing}")
        return ports_df.loc[:, cols], cols

    sel_path = Path(selected_ports_csv)
    raw = pd.read_csv(sel_path, header=None)

    # Detect "first row is header ids" pattern (common for Selected_Ports_*.csv)
    header_guess = raw.iloc[0].astype(str).tolist()
    if all(h in ports_df.columns for h in header_guess):
        cols = header_guess
        sel = raw.iloc[1:].reset_index(drop=True)
    else:
        # Fallback: treat first row as data; infer columns by position against portfolio
        if raw.shape[1] != ports_df.shape[1]:
            raise ValueError(
                f"{sel_path} has {raw.shape[1]} columns but {portfolio_csv} has "
                f"{ports_df.shape[1]}; cannot align without an explicit column list"
            )
        cols = [str(c) for c in ports_df.columns.tolist()]
        sel = raw.reset_index(drop=True)

    out = sel.apply(pd.to_numeric, errors="coerce")
    out.columns = cols
    return out, cols


def run_time_varying_one_triplet(
    *,
    feat1: str,
    feat2: str,
    portfolio_csv: Path,
    panel_parquet: Path,
    output_dir: Path,
    portfolio_columns_subset: list[str],
    y_min: int,
    y_max: int,
    min_train_months: int,
    bandwidth: float,
    research_figures: bool,
    n_train_valid: int,
    time_window_m: int = 20,
    time_decay_lambda: float = 0.9,
    ridge_sigma: float = 1e-3,
) -> None:
    """
    Run a single triplet workflow (toy/test harness).

    Notes
    -----
    - ``y_min`` / ``y_max`` are accepted for API compatibility; this helper does not
      enforce a full CRSP calendar — it uses whatever rows exist in ``panel``.
    - ``research_figures`` must be True for the current tests.
    """
    if not research_figures:
        raise ValueError("research_figures=False is not supported in this minimal workflow")

    ports = pd.read_csv(portfolio_csv)
    panel = pd.read_parquet(panel_parquet)

    missing = [c for c in portfolio_columns_subset if c not in ports.columns]
    if missing:
        raise ValueError(f"Missing portfolio columns in ports.csv: {missing}")

    R = ports[portfolio_columns_subset].to_numpy(dtype=float)
    T, p = R.shape
    if len(panel) != T:
        raise ValueError(f"panel rows {len(panel)} != portfolio rows {T}")

    # Build a simple state matrix: [LME, feat1, feat2] if present, else zeros.
    def _col(name: str) -> np.ndarray:
        if name in panel.columns:
            return pd.to_numeric(panel[name], errors="coerce").to_numpy(dtype=float)
        return np.zeros(T, dtype=float)

    states = np.column_stack([_col("LME"), _col(feat1), _col(feat2)])

    yy = pd.to_numeric(panel["yy"], errors="coerce").to_numpy(dtype=int)
    mm = pd.to_numeric(panel["mm"], errors="coerce").to_numpy(dtype=int)
    dates_yyyymm = (yy * 100 + mm).astype(np.int64)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_research_figure_bundle(
        R,
        states,
        dates_yyyymm,
        list(portfolio_columns_subset),
        bandwidth=float(bandwidth),
        min_train_months=int(min_train_months),
        ridge_sigma=float(ridge_sigma),
        n_train_valid=int(n_train_valid),
        time_window_m=int(time_window_m),
        time_decay_lambda=float(time_decay_lambda),
        output_dir=out,
        feat1=str(feat1),
        feat2=str(feat2),
    )
