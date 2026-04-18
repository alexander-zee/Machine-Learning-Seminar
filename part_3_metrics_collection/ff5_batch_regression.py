"""
ff5_batch_regression.py — FF5 alpha for every triplet's kernel full-fit (k=10).

Produces a summary table modelled on Table B.1 of Bryzgalova et al. (2025):

    Id | Char1 | Char2 | Char3 | SR | αFF5 [t-stat] | λ0 | λ2 [| h]

Rows are ordered by SR ascending (matching the paper's convention).
Char1 is always "LME" (Size); Char2/Char3 are the two secondary characteristics.

Usage
-----
    python ff5_batch_regression.py

Adjust CONFIG below.  Outputs:
    <OUTPUT_PATH>/ff5_results_<kernel>_k{K}.csv   — full numeric results
    <OUTPUT_PATH>/ff5_table_<kernel>_k{K}.csv      — paper-style formatted table
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.simplefilter(action="ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
KERNEL_NAME   = "gaussian"   # subfolder under GRID_SEARCH_PATH; "uniform" also works
K             = 10
N_TRAIN_VALID = 360
Y_MIN, Y_MAX  = 1964, 2016

GRID_SEARCH_PATH = Path("data/results/grid_search/tree")
OUTPUT_PATH      = Path("data/results/diagnostics")

# Add "IdioVol" here when ready — just append to the list; folder names are automatic
CHARACTERISTICS = [
    "BEME", "r12_2", "OP", "Investment",
    "ST_Rev", "LT_Rev", "AC", "LTurnover",
    "IdioVol",
]

# Human-readable short names matching the paper's abbreviations
CHAR_LABELS: dict[str, str] = {
    "LME":        "Size",
    "BEME":       "Val",
    "r12_2":      "Mom",
    "OP":         "Prof",
    "Investment": "Inv",
    "ST_Rev":     "SRev",
    "LT_Rev":     "LRev",
    "AC":         "Acc",
    "LTurnover":  "Turn",
    "IdioVol":    "IVol",
}

# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_dates(y_min: int = Y_MIN, y_max: int = Y_MAX) -> np.ndarray:
    dates = []
    for y in range(y_min, y_max + 1):
        for m in range(1, 13):
            dates.append(int(f"{y}{m:02d}"))
    return np.array(dates)


# ─────────────────────────────────────────────────────────────────────────────
# FF5 factors — loaded from local CSV (downloaded once from French library)
# ─────────────────────────────────────────────────────────────────────────────

FF5_CSV = Path("data/raw/F-F_Research_Data_5_Factors_2x3.csv")

_ff5_cache: pd.DataFrame | None = None


def _load_ff5() -> pd.DataFrame:
    global _ff5_cache
    if _ff5_cache is not None:
        return _ff5_cache
    print(f"Loading FF5 factors from {FF5_CSV}...", flush=True)
    # French CSV has 4 descriptive header lines before the column row
    ff5 = pd.read_csv(FF5_CSV, skiprows=4, index_col=0)
    # Drop annual summary rows at the bottom (index is no longer 6-digit YYYYMM)
    ff5 = ff5[ff5.index.astype(str).str.strip().str.match(r'^\d{6}$')]
    ff5.index = ff5.index.astype(int)
    ff5.index.name = "Date"
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    ff5[factor_cols] = ff5[factor_cols].apply(pd.to_numeric, errors="coerce")
    ff5[factor_cols] = ff5[factor_cols] / 100.0
    print(f"  FF5 loaded: {len(ff5)} months "
          f"({ff5.index[0]} - {ff5.index[-1]})", flush=True)
    _ff5_cache = ff5
    return ff5

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter lookup from the full_fit summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def _load_hyperparams(kernel_name: str, subdir: str, k: int) -> dict:
    """
    Read lambda0, lambda2 (and h for non-uniform kernels) from
    full_fit_summary_k{k}.csv produced by kernel_full_fit.
    Returns a dict with None values if the file is missing.
    """
    path = (
        GRID_SEARCH_PATH / kernel_name / subdir / "full_fit"
        / f"full_fit_summary_k{k}.csv"
    )
    if not path.exists():
        return {"lambda0": None, "lambda2": None, "h": None}
    row = pd.read_csv(path).iloc[0]
    return {
        "lambda0": float(row["lambda0"]),
        "lambda2": float(row["lambda2"]),
        "h": float(row["h"]) if "h" in row and pd.notna(row["h"]) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core regression for one cross-section
# ─────────────────────────────────────────────────────────────────────────────

def _regress_one(
    subdir: str,
    feat1: str,
    feat2: str,
    ff5: pd.DataFrame,
    all_dates: np.ndarray,
    kernel_name: str,
    k: int,
) -> dict:
    detail_path = (
        GRID_SEARCH_PATH / kernel_name / subdir / "full_fit"
        / f"full_fit_detail_k{k}.csv"
    )

    base = {
        "cross_section": subdir,
        "char1": "LME",
        "char2": feat1,
        "char3": feat2,
    }

    if not detail_path.exists():
        return {
            **base, "status": "missing",
            "sr": None, "mean_ret": None, "std_ret": None,
            "alpha_ff5": None, "alpha_ff5_tstat": None, "alpha_ff5_pval": None,
            "r2": None,
            "beta_MktRF": None, "beta_SMB": None,
            "beta_HML": None, "beta_RMW": None, "beta_CMA": None,
            "lambda0": None, "lambda2": None, "h": None, "n_obs": None,
        }

    detail = pd.read_csv(detail_path)
    rets   = detail["excess_return"].values   # already excess returns

    # Align dates: detail CSV starts at N_TRAIN_VALID; may be shorter than the
    # full test window if LARS skipped some months — align by position
    test_dates = all_dates[N_TRAIN_VALID : N_TRAIN_VALID + len(rets)]

    port_df = pd.DataFrame({"Date": test_dates, "ret": rets})
    merged  = pd.merge(port_df, ff5, left_on="Date", right_index=True, how="inner")

    if merged.empty:
        return {
            **base, "status": "merge_failed",
            "sr": None, "mean_ret": None, "std_ret": None,
            "alpha_ff5": None, "alpha_ff5_tstat": None, "alpha_ff5_pval": None,
            "r2": None,
            "beta_MktRF": None, "beta_SMB": None,
            "beta_HML": None, "beta_RMW": None, "beta_CMA": None,
            "lambda0": None, "lambda2": None, "h": None, "n_obs": 0,
        }

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    X     = sm.add_constant(merged[factor_cols])
    Y     = merged["ret"]
    model = sm.OLS(Y, X).fit()

    r     = merged["ret"]
    std_r = r.std(ddof=1)
    sr    = float(r.mean() / std_r) if std_r > 0 else np.nan

    hp = _load_hyperparams(kernel_name, subdir, k)

    return {
        **base,
        "status":          "ok",
        "n_obs":           int(model.nobs),
        "sr":              sr,
        "mean_ret":        float(r.mean()),
        "std_ret":         float(std_r),
        "alpha_ff5":       float(model.params["const"]),
        "alpha_ff5_tstat": float(model.tvalues["const"]),
        "alpha_ff5_pval":  float(model.pvalues["const"]),
        "r2":              float(model.rsquared),
        "beta_MktRF":      float(model.params["Mkt-RF"]),
        "beta_SMB":        float(model.params["SMB"]),
        "beta_HML":        float(model.params["HML"]),
        "beta_RMW":        float(model.params["RMW"]),
        "beta_CMA":        float(model.params["CMA"]),
        "lambda0":         hp["lambda0"],
        "lambda2":         hp["lambda2"],
        "h":               hp["h"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build the paper-style display table
# ─────────────────────────────────────────────────────────────────────────────

def _build_display_table(results_df: pd.DataFrame, kernel_name: str) -> pd.DataFrame:
    """
    Paper-style table (Table B.1 format):
        Id | Char1 | Char2 | Char3 | SR | αFF5 [t-stat] | λ0 | λ2 [| h]

    Rows sorted by SR ascending (lowest SR first, as in the paper).
    Id is assigned after sorting so Id=1 is the lowest-SR cross-section.
    Row order within the numeric CSV is untouched.
    """
    ok = results_df[results_df["status"] == "ok"].copy()
    ok = ok.sort_values("sr", ascending=True).reset_index(drop=True)
    ok.index = ok.index + 1   # 1-based Id

    rows = []
    for idx, row in ok.iterrows():
        def fmt(val, fmt_str):
            return format(val, fmt_str) if pd.notna(val) else "—"

        alpha_str = (
            f"{fmt(row['alpha_ff5'], '.2f')} [{fmt(row['alpha_ff5_tstat'], '.2f')}]"
            if pd.notna(row["alpha_ff5"]) else "—"
        )

        entry = {
            "Id":       idx,
            "Char1":    CHAR_LABELS.get(row["char1"], row["char1"]),
            "Char2":    CHAR_LABELS.get(row["char2"], row["char2"]),
            "Char3":    CHAR_LABELS.get(row["char3"], row["char3"]),
            "SR":       fmt(row["sr"], ".2f"),
            "αFF5 [t]": alpha_str,
            "λ0":       fmt(row["lambda0"], ".2f"),
            "λ2":       fmt(row["lambda2"], ".2e"),
        }

        # h column only for non-uniform kernels
        if kernel_name != "uniform":
            entry["h"] = fmt(row.get("h"), ".4f") if pd.notna(row.get("h")) else "—"

        rows.append(entry)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    kernel_name: str = KERNEL_NAME,
    k: int = K,
    characteristics: list[str] = CHARACTERISTICS,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:

    output_path.mkdir(parents=True, exist_ok=True)
    all_dates = _generate_dates()
    ff5       = _load_ff5()

    pairs   = list(combinations(characteristics, 2))
    records = []

    print(
        f"\nFF5 regressions — kernel={kernel_name}, k={k}, "
        f"{len(pairs)} cross-sections\n",
        flush=True,
    )

    for feat1, feat2 in pairs:
        subdir = f"LME_{feat1}_{feat2}"
        row    = _regress_one(subdir, feat1, feat2, ff5, all_dates, kernel_name, k)
        records.append(row)

        if row["status"] == "ok":
            h_part = (
                f"  h={row['h']:.4f}"
                if kernel_name != "uniform" and row["h"] is not None
                else ""
            )
            print(
                f"  {subdir:<35}  SR={row['sr']:+.3f}  "
                f"αFF5={row['alpha_ff5']:+.4f} [t={row['alpha_ff5_tstat']:+.2f}]  "
                f"λ0={row['lambda0']:.2f}  λ2={row['lambda2']:.2e}"
                + h_part,
                flush=True,
            )
        else:
            print(f"  {subdir:<35}  [{row['status']}]", flush=True)

    results_df = pd.DataFrame(records)

    # Full numeric CSV — natural iteration order (not sorted)
    numeric_csv = output_path / f"ff5_results_{kernel_name}_k{k}.csv"
    results_df.to_csv(numeric_csv, index=False)
    print(f"\nNumeric results → {numeric_csv}", flush=True)

    # Paper-style display table — sorted by SR ascending
    display_df  = _build_display_table(results_df, kernel_name)
    display_csv = output_path / f"ff5_table_{kernel_name}_k{k}.csv"
    display_df.to_csv(display_csv, index=False)
    print(f"Display table   → {display_csv}", flush=True)

    # Print to console
    print(f"\n{'─'*80}")
    print(display_df.to_string(index=False))
    print(f"{'─'*80}\n")

    # Summary stats
    ok = results_df[results_df["status"] == "ok"]
    if len(ok) > 0:
        sig05 = ok[ok["alpha_ff5_pval"] < 0.05]
        print(f"  Completed:             {len(ok)}/{len(pairs)}")
        print(f"  Missing / failed:      {len(pairs) - len(ok)}")
        print(f"  Mean SR (monthly):     {ok['sr'].mean():.3f}")
        print(f"  Mean αFF5:             {ok['alpha_ff5'].mean():.4f}")
        print(f"  Sig. alpha (p<.05):    {len(sig05)}/{len(ok)}")

    return results_df


if __name__ == "__main__":
    run_batch()