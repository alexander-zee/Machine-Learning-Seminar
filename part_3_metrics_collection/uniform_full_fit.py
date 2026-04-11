"""
uniform_full_fit.py — Reconstruct excess return time series for the uniform SDF.

Reads the Selected_Ports and Selected_Ports_Weights files written by
pick_best_lambda, slices to the test window, and saves two CSVs matching
the format of lasso_kernel_full_fit.py so that ff5_batch_regression.py
and ledoit_wolf_sr_test.py work unchanged with KERNEL_NAME = "uniform".

Output (in uniform/LME_f1_f2/full_fit/)
----------------------------------------
full_fit_summary_k{k}.csv
    Columns: k, test_SR, mean_ret, std_ret, lambda0, lambda2, h, kernel

full_fit_detail_k{k}.csv
    Columns: excess_return
    One row per test month (276 rows).
    Note: weights are constant in the uniform case so we only store
    excess_return, not the per-month weight columns.

Usage
-----
Run for a single cross-section:
    python -m part_3_metrics_collection.uniform_full_fit

Or call uniform_full_fit() directly from standard_uniform_all.py.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GRID_SEARCH_PATH = Path("data/results/grid_search/tree")
N_TRAIN_VALID    = 360
K                = 10

LAMBDA0 = [0.5, 0.55, 0.6]
LAMBDA2 = [10**-7, 10**-7.25, 10**-7.5]

CHARACTERISTICS = [
    "BEME", "r12_2", "OP", "Investment",
    "ST_Rev", "LT_Rev", "AC", "LTurnover",
    "IdioVol",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def uniform_full_fit(
    feat1: str,
    feat2: str,
    k: int = K,
    grid_search_path: Path = GRID_SEARCH_PATH,
    n_train_valid: int = N_TRAIN_VALID,
    lambda0: list = LAMBDA0,
    lambda2: list = LAMBDA2,
) -> dict:
    """
    Reconstruct the test-period excess return time series for the uniform SDF.

    Parameters
    ----------
    feat1, feat2     : characteristic names (e.g. 'OP', 'Investment')
    k                : portfolio count (must match what pick_best_lambda wrote)
    grid_search_path : root grid search directory
    n_train_valid    : months in train+validation window (default 360)
    lambda0, lambda2 : hyperparameter grids (needed to read best values)

    Returns
    -------
    dict with keys: test_SR, mean_ret, std_ret, months_used
    """
    subdir  = f"LME_{feat1}_{feat2}"
    base    = grid_search_path / "uniform" / subdir
    out_dir = base / "full_fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load selected ports and weights ──────────────────────────────────────
    ports_path   = base / f"Selected_Ports_{k}.csv"
    weights_path = base / f"Selected_Ports_Weights_{k}.csv"

    if not ports_path.exists():
        raise FileNotFoundError(
            f"Selected_Ports_{k}.csv not found at {base}\n"
            "Run pick_best_lambda first."
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Selected_Ports_Weights_{k}.csv not found at {base}\n"
            "Run pick_best_lambda first."
        )

    ports_df = pd.read_csv(ports_path)
    weights  = pd.read_csv(weights_path).values.flatten()

    # ── Slice to test window ─────────────────────────────────────────────────
    if len(ports_df) <= n_train_valid:
        raise ValueError(
            f"Selected_Ports has only {len(ports_df)} rows — "
            f"expected more than n_train_valid={n_train_valid}."
        )

    ports_test     = ports_df.iloc[n_train_valid:]          # 276 rows
    excess_returns = ports_test.values @ weights             # (276,)

    # ── Sharpe ratio ─────────────────────────────────────────────────────────
    mean_ret = float(excess_returns.mean())
    std_ret  = float(excess_returns.std(ddof=1))
    test_SR  = float(mean_ret / std_ret) if std_ret > 0 else np.nan

    # ── Read best lambda0/lambda2 from valid_SR grid ──────────────────────────
    valid_sr_path = base / f"valid_SR_{k}.csv"
    if valid_sr_path.exists():
        valid_sr_df  = pd.read_csv(valid_sr_path, header=0)
        best_flat    = valid_sr_df.values.argmax()
        i_best, j_best = np.unravel_index(best_flat, valid_sr_df.shape)
        lambda0_star = lambda0[i_best]
        lambda2_star = lambda2[j_best]
    else:
        # Fallback if grid file missing
        lambda0_star = None
        lambda2_star = None

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary_path = out_dir / f"full_fit_summary_k{k}.csv"
    pd.DataFrame([{
        "k":        k,
        "test_SR":  test_SR,
        "mean_ret": mean_ret,
        "std_ret":  std_ret,
        "lambda0":  lambda0_star,
        "lambda2":  lambda2_star,
        "h":        None,           # no bandwidth for uniform kernel
        "kernel":   "uniform",
    }]).to_csv(summary_path, index=False)
    print(f"  Summary saved → {summary_path}  (test_SR={test_SR:.4f})", flush=True)

    # ── Save detail CSV ───────────────────────────────────────────────────────
    # Uniform: weights are constant across months, so we only store excess_return
    detail_path = out_dir / f"full_fit_detail_k{k}.csv"
    pd.DataFrame({"excess_return": excess_returns}).to_csv(detail_path, index=False)
    print(f"  Detail saved  → {detail_path}  ({len(excess_returns)} rows)", flush=True)

    return {
        "test_SR":     test_SR,
        "mean_ret":    mean_ret,
        "std_ret":     std_ret,
        "months_used": len(excess_returns),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner — all cross-sections
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    characteristics: list[str] = CHARACTERISTICS,
    k: int = K,
    grid_search_path: Path = GRID_SEARCH_PATH,
    n_train_valid: int = N_TRAIN_VALID,
    lambda0: list = LAMBDA0,
    lambda2: list = LAMBDA2,
) -> pd.DataFrame:
    from itertools import combinations

    pairs   = list(combinations(characteristics, 2))
    records = []

    print(f"\nUniform full fit — k={k}, {len(pairs)} cross-sections\n", flush=True)

    for feat1, feat2 in pairs:
        subdir = f"LME_{feat1}_{feat2}"
        try:
            result = uniform_full_fit(
                feat1, feat2, k=k,
                grid_search_path=grid_search_path,
                n_train_valid=n_train_valid,
                lambda0=lambda0, lambda2=lambda2,
            )
            print(
                f"  {subdir:<35}  SR={result['test_SR']:+.4f}  "
                f"n={result['months_used']}",
                flush=True,
            )
            records.append({"cross_section": subdir, "status": "ok", **result})
        except FileNotFoundError as e:
            print(f"  {subdir:<35}  [MISSING — run pick_best_lambda first]", flush=True)
            records.append({"cross_section": subdir, "status": "missing"})
        except Exception as e:
            print(f"  {subdir:<35}  [ERROR: {e}]", flush=True)
            records.append({"cross_section": subdir, "status": "error", "error": str(e)})

    df = pd.DataFrame(records)
    ok = df[df["status"] == "ok"]
    if len(ok) > 0:
        print(f"\n  Completed: {len(ok)}/{len(pairs)}")
        print(f"  Mean SR:   {ok['test_SR'].mean():.4f}")
    return df


if __name__ == "__main__":
    run_batch()