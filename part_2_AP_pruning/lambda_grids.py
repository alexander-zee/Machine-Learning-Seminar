"""
Lambda grids for AP pruning (mean shrinkage lambda0, LASSO ridge lambda2).

- fast: original 3x3 coarse grid (quick runs).
- paper: BPZ-style ranges with moderate steps (~9 x 10 points) — smoother heatmaps, still heavy for trees.
- paper_full: matches commented grid in 0_code/main.R (seq(0,0.9,0.05) x 0.1^seq(5,8,0.25)) — very slow for large p.

Set environment variable AP_PRUNE_LAMBDA_GRID to: fast | paper | paper_full
(default if unset: fast, for backward compatibility).
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

GridMode = Literal["fast", "paper", "paper_full"]


def ap_lambda_grid_mode() -> GridMode:
    v = os.environ.get("AP_PRUNE_LAMBDA_GRID", "").strip().lower()
    if v in ("paper", "paper_dense"):
        return "paper"
    if v in ("paper_full", "full", "bpz"):
        return "paper_full"
    return "fast"


def get_lambda_grids(mode: GridMode | None = None) -> tuple[list[float], list[float]]:
    if mode is None:
        mode = ap_lambda_grid_mode()

    if mode == "fast":
        lambda0 = [0.0, 0.1, 0.2]
        lambda2 = [0.01, 0.05, 0.1]
        return lambda0, lambda2

    if mode == "paper":
        # Figure 10–style ranges: lambda0 to ~0.8, lambda2 on log scale ~1e-8 … 1e-5
        lambda0 = np.linspace(0.0, 0.8, 9).tolist()
        lambda2 = np.logspace(-8, -5, 10).tolist()
        return lambda0, lambda2

    # paper_full — replication script comments in 0_code/main.R
    lambda0 = np.arange(0.0, 0.9 + 1e-9, 0.05).tolist()
    powers = np.linspace(5.0, 8.0, 13)
    lambda2 = np.power(0.1, powers).tolist()
    return lambda0, lambda2
