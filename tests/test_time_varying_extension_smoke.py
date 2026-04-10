"""Smoke tests for plug-in time_varying_extension (no baseline pipeline imports)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from time_varying_extension.conditional_mv import mean_variance_weights, weighted_mean_cov
from time_varying_extension.kernel import (
    combined_analogue_kernel_weights,
    effective_sample_size,
    gaussian_kernel_weights,
)
from time_varying_extension.workflow_one_triplet import run_time_varying_one_triplet


def test_kernel_and_mv():
    rng = np.random.default_rng(42)
    states = rng.normal(size=(80, 3))
    t = 50
    w = gaussian_kernel_weights(states, t, bandwidth=1.0, min_train=10)
    assert w.shape == (t,)
    assert abs(w.sum() - 1.0) < 1e-6
    R = rng.normal(size=(t, 4))
    mu, Sig = weighted_mean_cov(R, w, ridge=1e-4)
    w_mv = mean_variance_weights(mu, Sig, ridge=1e-3)
    assert w_mv.shape == (4,)
    assert abs(np.sum(np.abs(w_mv)) - 1.0) < 1e-5
    assert effective_sample_size(w) >= 1.0


def test_combined_gaussian_and_time_decay():
    rng = np.random.default_rng(1)
    states = rng.normal(size=(40, 3))
    t = 25
    w = combined_analogue_kernel_weights(
        states,
        t,
        bandwidth=1.0,
        min_train=5,
        use_time_decay=True,
        time_window_m=20,
        time_decay_lambda=0.9,
    )
    assert w.shape == (t,)
    assert abs(np.nansum(w) - 1.0) < 1e-5
    assert np.nanmax(w) <= 1.0 + 1e-9


def test_workflow_synthetic_end_to_end():
    rng = np.random.default_rng(0)
    T, p = 72, 4
    R = pd.DataFrame(rng.normal(0.01, 0.05, size=(T, p)), columns=[f"c{i}" for i in range(p)])
    dates = []
    yy, mm = 2000, 1
    for i in range(T):
        dates.append(
            {
                "date": pd.Timestamp(f"{yy}-{mm:02d}-28"),
                "yy": yy,
                "mm": mm,
                "LME": 0.3 + 0.001 * i,
                "OP": 0.4 + 0.002 * np.sin(i / 5.0),
                "Investment": 0.5 + 0.0015 * np.cos(i / 7.0),
                "size": 1.0,
            }
        )
        mm += 1
        if mm > 12:
            mm = 1
            yy += 1
    panel = pd.DataFrame(dates)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        port_path = td / "ports.csv"
        pan_path = td / "panel.parquet"
        R.to_csv(port_path, index=False)
        panel.to_parquet(pan_path, index=False)
        out = td / "out"
        run_time_varying_one_triplet(
            feat1="OP",
            feat2="Investment",
            portfolio_csv=port_path,
            panel_parquet=pan_path,
            output_dir=out,
            portfolio_columns_subset=[f"c{i}" for i in range(p)],
            y_min=2000,
            y_max=2100,
            min_train_months=15,
            bandwidth=0.5,
        )
        assert (out / "tv_mv_weights.csv").is_file()
        assert (out / "interpretation_tv.txt").is_file()
        assert (out / "tv_kernel_params.json").is_file()

        out2 = td / "out_decay"
        run_time_varying_one_triplet(
            feat1="OP",
            feat2="Investment",
            portfolio_csv=port_path,
            panel_parquet=pan_path,
            output_dir=out2,
            portfolio_columns_subset=[f"c{i}" for i in range(p)],
            y_min=2000,
            y_max=2100,
            min_train_months=15,
            bandwidth=0.5,
            use_time_decay=True,
            time_window_m=30,
            time_decay_lambda=0.92,
        )
        assert (out2 / "tv_kernel_params.json").is_file()
        txt = (out2 / "interpretation_tv.txt").read_text(encoding="utf-8")
        assert "lambda^j" in txt or "lambda=" in txt
