"""Smoke tests for causal optimal-quantile tree splits (extension 1)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import (  # noqa: E402
    OPT_MIN_LEAF,
    assign_nodes_month,
    assign_nodes_month_opt,
    compute_one_tree,
)


def _panel_months(n_stock: int = 80, n_month: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for mi in range(n_month):
        for i in range(n_stock):
            rows.append(
                {
                    "permno": i,
                    "yy": 2000,
                    "mm": mi + 1,
                    "ret": float(rng.normal(0, 0.05)),
                    "size": 1.0,
                    "LME": float(i + rng.normal(0, 0.1)),
                    "OP": float((i + mi) % 20),
                    "Investment": float(i % 15),
                }
            )
    return pd.DataFrame(rows)


def test_opt_matches_ntile_first_month():
    df = _panel_months()
    m0 = df[df["mm"] == 1].copy()
    feat_list = ["LME", "OP", "LME", "Investment"]
    ntile = assign_nodes_month(m0, feat_list, 4, 2)
    opt = assign_nodes_month_opt(m0, feat_list, 4, 2, {}, 0)
    pd.testing.assert_frame_equal(
        ntile[[f"split_{k}" for k in range(1, 5)] + [f"port{i}" for i in range(5)]].reset_index(
            drop=True
        ),
        opt[[f"split_{k}" for k in range(1, 5)] + [f"port{i}" for i in range(5)]].reset_index(
            drop=True
        ),
    )


def test_opt_runs_sequential_months():
    df = _panel_months(n_stock=max(OPT_MIN_LEAF * 2 + 5, 80), n_month=6, seed=2)
    feat_list = ["LME", "LME", "LME", "LME"]
    prior: dict[int, pd.DataFrame] = {}
    for mi in range(6):
        sub = df[df["mm"] == mi + 1].copy()
        out = assign_nodes_month_opt(sub, feat_list, 4, 2, prior, mi)
        prior[mi] = out
        assert out["split_1"].isin([1, 2]).all()
        assert len(out) == len(sub)


def test_compute_one_tree_opt_mode_runs():
    df = _panel_months(n_stock=100, n_month=4, seed=3)
    feats = ["LME", "OP", "Investment"]
    feat_list = ["LME", "OP", "LME", "Investment"]
    ret_nt, _, _ = compute_one_tree(
        df, feat_list, feats, 4, 2, 2000, 2000, split_mode="ntile"
    )
    ret_oq, _, _ = compute_one_tree(
        df, feat_list, feats, 4, 2, 2000, 2000, split_mode="opt_quantile", min_leaf=10
    )
    assert ret_nt.shape == ret_oq.shape == (12, 31)
    assert np.isfinite(ret_nt).any()
    assert np.isfinite(ret_oq).any()
