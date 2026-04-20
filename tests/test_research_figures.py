"""Research figure bundle: OOS Sharpe, kernel heatmaps (no full pipeline)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from time_varying_extension.research_figures import compute_research_bundle, write_research_figure_bundle
from time_varying_extension.workflow_one_triplet import run_time_varying_one_triplet


def test_compute_research_bundle_runs():
    rng = np.random.default_rng(2)
    T, p = 90, 4
    R = rng.normal(0.008, 0.04, size=(T, p))
    states = rng.normal(size=(T, 3))
    b = compute_research_bundle(
        R,
        states,
        bandwidth=0.8,
        min_train_months=18,
        ridge_sigma=1e-3,
        n_train_valid=50,
        time_window_m=24,
        time_decay_lambda=0.92,
    )
    assert b["W_gaussian"].shape == (T, p)
    assert b["H_gaussian"].shape == (T, T)
    assert b["n_oos_months"] >= 1


@patch(
    "part_3_metrics_collection.tv_extension_summary_table._safe_capm_ff5",
    return_value=(
        (0.01, 1.0, 0.5, 0.5, 100),
        {
            "alpha": 0.01,
            "p_alpha": 0.5,
            "r2": 0.2,
            "nobs": 100,
            "beta_Mkt-RF": 1.1,
        },
    ),
)
def test_write_research_figure_bundle_creates_files(_mock_ff):
    rng = np.random.default_rng(3)
    T, p = 64, 3
    R = rng.normal(0.01, 0.05, size=(T, p))
    states = rng.normal(size=(T, 3))
    yy, mm = 2005, 1
    dates = []
    for _ in range(T):
        dates.append(yy * 100 + mm)
        mm += 1
        if mm > 12:
            mm = 1
            yy += 1
    dates = np.array(dates, dtype=np.int64)
    cols = [f"c{i}" for i in range(p)]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "res"
        write_research_figure_bundle(
            R,
            states,
            dates,
            cols,
            bandwidth=0.6,
            min_train_months=15,
            ridge_sigma=1e-3,
            n_train_valid=36,
            time_window_m=20,
            time_decay_lambda=0.9,
            output_dir=out,
            feat1="OP",
            feat2="Investment",
        )
        assert (out / "tv_oos_sharpe_summary.json").is_file()
        assert (out / "tv_oos_strategy_returns.csv").is_file()
        assert (out / "tv_oos_strategy_factor_table.csv").is_file()
        fac = pd.read_csv(out / "tv_oos_strategy_factor_table.csv")
        assert len(fac) == 3
        assert set(fac["strategy"].tolist()) == {
            "static_tangency_train_valid",
            "tv_kernel_gaussian_state",
            "tv_kernel_gaussian_x_time_decay",
        }
        assert (out / "figures" / "fig_oos_sharpe_comparison.png").is_file()


def test_workflow_with_research_fig_flag():
    rng = np.random.default_rng(0)
    T, p = 72, 4
    R = pd.DataFrame(rng.normal(0.01, 0.05, size=(T, p)), columns=[f"c{i}" for i in range(p)])
    dates_rows = []
    yy, mm = 2000, 1
    for i in range(T):
        dates_rows.append(
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
    panel = pd.DataFrame(dates_rows)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        R.to_csv(td / "ports.csv", index=False)
        panel.to_parquet(td / "panel.parquet", index=False)
        out = td / "out_rf"
        with patch(
            "part_3_metrics_collection.tv_extension_summary_table._safe_capm_ff5",
            return_value=(
                (0.0, 1.0, 1.0, 1.0, 50),
                {
                    "alpha": 0.0,
                    "p_alpha": 1.0,
                    "r2": 0.0,
                    "nobs": 50,
                    "beta_Mkt-RF": 1.0,
                },
            ),
        ):
            run_time_varying_one_triplet(
                feat1="OP",
                feat2="Investment",
                portfolio_csv=td / "ports.csv",
                panel_parquet=td / "panel.parquet",
                output_dir=out,
                portfolio_columns_subset=[f"c{i}" for i in range(p)],
                y_min=2000,
                y_max=2100,
                min_train_months=15,
                bandwidth=0.5,
                research_figures=True,
                n_train_valid=40,
            )
        assert (out / "figures" / "fig_oos_sharpe_comparison.png").is_file()
        assert (out / "tv_oos_strategy_factor_table.csv").is_file()
