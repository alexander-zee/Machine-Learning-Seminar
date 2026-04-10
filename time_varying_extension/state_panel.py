"""
Build monthly state vectors s_t from the stock panel (one triplet's characteristics).

Uses value-weighted cross-sectional averages of quantile-ranked LME, feat1, feat2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def monthly_vw_state(
    panel_path: Path | str,
    feat1: str,
    feat2: str,
    y_min: int = 1964,
    y_max: int = 2016,
    size_col: str = "size",
) -> pd.DataFrame:
    """
    One row per calendar month: yyyymm, s_LME, s_feat1, s_feat2.

    Rows are sorted by date; aligns with row order of portfolio CSVs when
    those follow the same sample window.
    """
    cols = ["date", "yy", "mm", size_col, "LME", feat1, feat2]
    df = pd.read_parquet(panel_path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["yy"] >= y_min) & (df["yy"] <= y_max)].copy()

    def _vw(g: pd.DataFrame, c: str) -> float:
        x = g[c].to_numpy(dtype=float)
        w = g[size_col].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not m.any():
            return float("nan")
        return float(np.average(x[m], weights=w[m]))

    rows = []
    for (yy, mm), g in df.groupby(["yy", "mm"], sort=True):
        rows.append(
            {
                "yy": yy,
                "mm": mm,
                "yyyymm": yy * 100 + mm,
                "s_LME": _vw(g, "LME"),
                "s_feat1": _vw(g, feat1),
                "s_feat2": _vw(g, feat2),
            }
        )
    out = pd.DataFrame(rows).sort_values(["yy", "mm"]).reset_index(drop=True)
    return out


def align_states_to_returns(
    state_df: pd.DataFrame,
    n_rows_returns: int,
) -> np.ndarray:
    """
    If state_df has same length as return matrix rows, use as-is.
    Otherwise trim or raise.
    """
    if len(state_df) != n_rows_returns:
        raise ValueError(
            f"State rows {len(state_df)} != portfolio rows {n_rows_returns}. "
            "Use the same sample window as the tree CSV (e.g. 1964–2016 monthly)."
        )
    return state_df[["s_LME", "s_feat1", "s_feat2"]].to_numpy(dtype=float)
