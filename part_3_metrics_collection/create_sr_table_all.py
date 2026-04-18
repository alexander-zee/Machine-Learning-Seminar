"""
create_sr_table_all.py — Assemble a master FF5 comparison table across all kernels.

Reads the ff5_results_<kernel>_k{K}.csv files produced by ff5_batch_regression.py
and merges them into one wide CSV, ordered by the uniform kernel's Sharpe ratio
(ascending, matching the paper's convention).

Column structure
----------------
Id | Char1 | Char2 | Char3
  | SR_uniform  | alpha_uniform  | tstat_uniform
  | SR_gaussian | alpha_gaussian | tstat_gaussian
  | SR_gaussian_tms | alpha_gaussian_tms | tstat_gaussian_tms
  | SR_exponential | alpha_exponential | tstat_exponential

The numeric columns are kept as floats so the CSV can be passed directly to
a LaTeX formatter later.  A separate "display" CSV with formatted strings is
also written.

Usage
-----
    python create_sr_table_all.py          # from project root
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

K = 10

INPUT_PATH  = Path("data/results/diagnostics")
OUTPUT_PATH = Path("data/results/diagnostics")

# Order matters — uniform is the sort anchor
KERNELS = [
    ("uniform",      "Uniform"),
    ("gaussian",     "Gaussian (svar)"),
    ("gaussian-tms", "Gaussian (TMS)"),
    ("exponential",  "Exponential"),
]

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
# Load one kernel's results
# ─────────────────────────────────────────────────────────────────────────────

def _load_kernel(kernel_name: str, k: int) -> pd.DataFrame:
    path = INPUT_PATH / f"ff5_results_{kernel_name}_k{k}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    return df[["cross_section", "sr", "alpha_ff5", "alpha_ff5_tstat"]]


# ─────────────────────────────────────────────────────────────────────────────
# Build master table
# ─────────────────────────────────────────────────────────────────────────────

def build_master_table(k: int = K) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    numeric_df  : raw float values, ready for programmatic use / LaTeX formatting
    display_df  : human-readable strings matching paper style
    """

    # ── 1. Load uniform as the base (determines sort order) ──────────────────
    uniform_df = _load_kernel("uniform", k)
    uniform_df = uniform_df.sort_values("sr", ascending=True).reset_index(drop=True)
    uniform_df.index = uniform_df.index + 1  # 1-based Id

    # ── 2. Build the wide numeric dataframe ──────────────────────────────────
    # Start from the sorted uniform cross-section list
    base = uniform_df[["cross_section"]].copy()

    rows_numeric = []
    for _, base_row in base.iterrows():
        cs = base_row["cross_section"]
        # Parse char labels from cross_section name: LME_feat1_feat2
        parts = cs.split("_", 1)  # ['LME', 'feat1_feat2'] won't work for multi-word
        # Better: split on _ but cross_section is always LME_{feat1}_{feat2}
        _, feat1, feat2 = cs.split("_", 2)

        row = {
            "cross_section": cs,
            "Char1": CHAR_LABELS.get("LME", "Size"),
            "Char2": CHAR_LABELS.get(feat1, feat1),
            "Char3": CHAR_LABELS.get(feat2, feat2),
        }

        for kernel_name, _ in KERNELS:
            try:
                kdf = _load_kernel(kernel_name, k)
                match = kdf[kdf["cross_section"] == cs]
                if len(match) == 0:
                    row[f"sr_{kernel_name}"]    = None
                    row[f"alpha_{kernel_name}"] = None
                    row[f"tstat_{kernel_name}"] = None
                else:
                    r = match.iloc[0]
                    row[f"sr_{kernel_name}"]    = float(r["sr"])
                    row[f"alpha_{kernel_name}"] = float(r["alpha_ff5"])
                    row[f"tstat_{kernel_name}"] = float(r["alpha_ff5_tstat"])
            except FileNotFoundError:
                row[f"sr_{kernel_name}"]    = None
                row[f"alpha_{kernel_name}"] = None
                row[f"tstat_{kernel_name}"] = None

        rows_numeric.append(row)

    numeric_df = pd.DataFrame(rows_numeric)
    numeric_df.insert(0, "Id", range(1, len(numeric_df) + 1))

    # ── 3. Build display dataframe with formatted strings ─────────────────────
    def fmt(val, fmt_str):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "—"
        return format(val, fmt_str)

    display_rows = []
    for _, row in numeric_df.iterrows():
        drow = {
            "Id":    int(row["Id"]),
            "Char1": row["Char1"],
            "Char2": row["Char2"],
            "Char3": row["Char3"],
        }
        for kernel_name, kernel_label in KERNELS:
            sr    = row.get(f"sr_{kernel_name}")
            alpha = row.get(f"alpha_{kernel_name}")
            tstat = row.get(f"tstat_{kernel_name}")
            drow[f"SR ({kernel_label})"]         = fmt(sr, ".2f")
            drow[f"αFF5 [t] ({kernel_label})"]   = (
                f"{fmt(alpha, '.2f')} [{fmt(tstat, '.2f')}]"
                if alpha is not None and not pd.isna(alpha) else "—"
            )
        display_rows.append(drow)

    display_df = pd.DataFrame(display_rows)

    return numeric_df, display_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Building master SR table for k={K}...\n")
    numeric_df, display_df = build_master_table(k=K)

    # Save numeric CSV
    numeric_out = OUTPUT_PATH / f"sr_table_all_k{K}.csv"
    numeric_df.to_csv(numeric_out, index=False)
    print(f"Numeric table → {numeric_out}")

    # Save display CSV
    display_out = OUTPUT_PATH / f"sr_table_all_display_k{K}.csv"
    display_df.to_csv(display_out, index=False)
    print(f"Display table → {display_out}")

    # Print to console
    print(f"\n{'─'*100}")
    print(display_df.to_string(index=False))
    print(f"{'─'*100}")

    # Summary
    print(f"\n{len(numeric_df)} cross-sections, sorted by uniform SR ascending.")