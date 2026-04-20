from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def monthly_vw_state(
    panel_parquet: Path | str,
    feat1: str,
    feat2: str,
    *,
    y_min: int,
    y_max: int,
) -> pd.DataFrame:
    """
    Build a simple month panel of state variables aligned to the seminar conventions.

    Expected columns in ``panel.parquet`` (toy/test harness):
    - ``yy``, ``mm``
    - ``LME``
    - columns named exactly ``feat1`` / ``feat2`` (e.g. ``OP``, ``Investment``)

    Returns a dataframe sorted by time with a ``yyyymm`` int column.
    """
    df = pd.read_parquet(panel_parquet)
    need = {"yy", "mm", "LME", feat1, feat2}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"panel.parquet missing columns: {missing}")

    out = df.loc[:, ["yy", "mm", "LME", feat1, feat2]].copy()
    out["yyyymm"] = (
        pd.to_numeric(out["yy"], errors="coerce").astype(int) * 100
        + pd.to_numeric(out["mm"], errors="coerce").astype(int)
    )
    out = out[(out["yy"] >= int(y_min)) & (out["yy"] <= int(y_max))]
    out = out.sort_values(["yy", "mm"]).reset_index(drop=True)

    # Ensure numeric state columns
    for c in ["LME", feat1, feat2]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def align_states_to_returns(state_df: pd.DataFrame, T: int) -> np.ndarray:
    """
    Align ``state_df`` rows to a return matrix length ``T``.

    For the toy tests, ``state_df`` already has length ``T``; if not, we slice/pad
    deterministically to avoid hard failures.
    """
    sdf = state_df.reset_index(drop=True)
    cols = [c for c in ["LME", sdf.columns[3], sdf.columns[4]] if c in sdf.columns]
    # columns 3/4 correspond to feat1/feat2 in monthly_vw_state(); keep robust:
    if len(cols) < 3:
        # fall back to last two non-metadata columns
        meta = {"yy", "mm", "yyyymm"}
        rest = [c for c in sdf.columns if c not in meta]
        cols = ["LME", rest[0], rest[1]] if len(rest) >= 2 else rest

    X = sdf.loc[:, cols].to_numpy(dtype=float)
    if X.shape[0] == T:
        return X
    if X.shape[0] > T:
        return X[:T, :]
    pad = np.full((T - X.shape[0], X.shape[1]), np.nan, dtype=float)
    return np.vstack([X, pad])
