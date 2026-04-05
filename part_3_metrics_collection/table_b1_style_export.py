"""
Table B.1–style summary CSV (seminar analog): one row per AP-tree cross-section.

Columns align with the paper layout where the codebase supports them:

- **SR** → monthly **test** Sharpe at λ* (out-of-sample; same scale as ``pick_best_lambda``).
- **α FF3 / FF5 / FF11** → intercepts from ``regression_table_sdf`` (SDF ~ tradable factors).
- **λ0, λ2** → grid values at the chosen (λ0, λ2) cell.

The paper’s **α XSF** column is **not** replicated here: we do not have that factor specification
in ``paper_style_outputs.regression_table_sdf`` (only FF3, FF5, FF11). Those columns are left
empty (NaN); see ``Table_B1_style_seminar_NOTES.txt``.

Prerequisites
-------------
1. ``prepare_data`` with ``svar`` → ``IdioVol`` (36 triplets).
2. Part 1 + Part 2 for every triplet (e.g. ``python run_all_tree_cross_sections.py``).
3. Then::

       python -m part_3_metrics_collection.table_b1_style_export

Output: ``data/results/tables_seminar/Table_B1_style_seminar_cross_section.csv``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from part_3_metrics_collection.pick_best_lambda import (  # noqa: E402
    AP_PRUNE_DEFAULT,
    TREE_PORT_ROOT,
    discover_lme_tree_ap_subdirs,
    pick_best_lambda,
)
from part_3_metrics_collection.paper_style_outputs import (  # noqa: E402
    TRADABLE_FACTORS,
    build_sdf_series,
    regression_table_sdf,
)
from part_3_metrics_collection.table1_style_bpz_export import _paper_triplet_labels  # noqa: E402

TABLES_DIR = REPO_ROOT / "data" / "results" / "tables_seminar"
TREE_CSV = "level_all_excess_combined_filtered.csv"


def _short_paper_labels(c1: str, c2: str, c3: str) -> tuple[str, str, str]:
    """Table B.1 uses short names (Val, Mom, Prof, Inv, …)."""
    m = {
        "Size": "Size",
        "Value": "Val",
        "Mom": "Mom",
        "OP": "Prof",
        "Inv": "Inv",
        "SRev": "SRev",
        "LRev": "LRev",
        "Acc": "Acc",
        "Turnover": "Turn",
        "IVol": "IVol",
    }
    return m.get(c1, c1), m.get(c2, c2), m.get(c3, c3)


def build_table_b1_dataframe(
    ap_root: Path | None = None,
    port_n: int = 10,
    write_selected_ports: bool = True,
    sort_by_sr_ascending: bool = True,
) -> pd.DataFrame:
    """Rows = discovered ``LME_*`` folders; sorted by monthly test SR like Table B.1 (low to high)."""
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    rows: list[dict] = []
    for sub in discover_lme_tree_ap_subdirs(ap):
        try:
            c1, c2, c3 = _paper_triplet_labels(sub)
        except KeyError as e:
            print(f"skip {sub}: {e}")
            continue
        s1, s2, s3 = _short_paper_labels(c1, c2, c3)
        tree_csv = TREE_PORT_ROOT / sub / TREE_CSV
        r = pick_best_lambda(
            ap,
            sub,
            port_n,
            tree_csv,
            returns_index_col=None,
            write_tables=write_selected_ports,
        )
        la0 = np.asarray(r["lambda0"], dtype=float)
        la2 = np.asarray(r["lambda2"], dtype=float)
        i0, j0 = r["best_i_lambda0"] - 1, r["best_j_lambda2"] - 1
        lam0_star = float(la0[i0])
        lam2_star = float(la2[j0])

        a3 = a5 = a11 = np.nan
        t3 = t5 = t11 = np.nan
        pn = int(r["port_n"])
        ports_p = ap / sub / f"Selected_Ports_{pn}.csv"
        w_p = ap / sub / f"Selected_Ports_Weights_{pn}.csv"
        if ports_p.is_file() and w_p.is_file():
            sdf = build_sdf_series(ports_p, w_p, TRADABLE_FACTORS)
            reg = regression_table_sdf(sdf, TRADABLE_FACTORS)
            r3 = reg.loc[reg["spec"] == "FF3"].iloc[0]
            r5 = reg.loc[reg["spec"] == "FF5"].iloc[0]
            r11 = reg.loc[reg["spec"] == "FF11"].iloc[0]
            a3, t3 = float(r3["alpha"]), float(r3["t_stat"])
            a5, t5 = float(r5["alpha"]), float(r5["t_stat"])
            a11, t11 = float(r11["alpha"]), float(r11["t_stat"])

        rows.append(
            {
                "char_1": s1,
                "char_2": s2,
                "char_3": s3,
                "SR_monthly_test": float(r["test_SR"]),
                "alpha_FF3": a3,
                "t_FF3": t3,
                "alpha_FF5": a5,
                "t_FF5": t5,
                "alpha_XSF": np.nan,
                "t_XSF": np.nan,
                "alpha_FF11": a11,
                "t_FF11": t11,
                "lambda_0": lam0_star,
                "lambda_2": lam2_star,
                "subdir": sub,
                "port_n": pn,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("SR_monthly_test", ascending=sort_by_sr_ascending).reset_index(drop=True)
    df.insert(0, "row_order", range(1, len(df) + 1))
    return df


def write_table_b1_export(
    ap_root: Path | None = None,
    port_n: int = 10,
    out_csv: Path | None = None,
    out_notes: Path | None = None,
) -> tuple[Path, Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = build_table_b1_dataframe(ap_root=ap_root, port_n=port_n)
    csv_path = out_csv or TABLES_DIR / "Table_B1_style_seminar_cross_section.csv"
    notes_path = out_notes or TABLES_DIR / "Table_B1_style_seminar_NOTES.txt"
    df.to_csv(csv_path, index=False)
    notes_path.write_text(
        """Table B.1 style (seminar) — layout similar to BPZ Table B.1; numbers differ by sample/code.

SR
--
SR_monthly_test = monthly Sharpe on the **test** window at λ* (max validation Sharpe), from Part 2.

Alphas
------
alpha_FF3, alpha_FF5, alpha_FF11: intercepts from OLS sdf ~ 1 + tradable factors (same as
``regression_table_sdf`` / Table 3 style). t_* are OLS t-ratios on the intercept.

alpha_XSF / t_XSF
-------------------
Not computed in this repository. The paper’s “XSF” specification is separate from FF3/5/11;
fill from their appendix if you need an exact match.

λ0, λ2
------
Values on the Part 2 λ grid at the selected cell (same as ``lambda_grid_meta.json``).

Paper “Id” column (1–36)
-------------------------
Not reproduced: those IDs are fixed labels in the paper’s cross-section list. This export uses
row_order after sorting by SR (ascending, like the example table). Join on (char_1,char_2,char_3)
or ``subdir`` if you need to align with their numbering.

Row count
---------
Up to 36 rows when all triplets are built (``run_all_tree_cross_sections.py``) and ``IdioVol``
is in the panel.
""",
        encoding="utf-8",
    )
    return csv_path, notes_path


def _main() -> None:
    csv_p, notes_p = write_table_b1_export()
    print(f"Wrote:\n  {csv_p}\n  {notes_p}")
    df = pd.read_csv(csv_p)
    if len(df):
        print(df.head(12).to_string(index=False))
        print(f"\n... ({len(df)} rows total)")
    else:
        print("No rows — build Part 1+2 outputs under data/results/ap_pruning/LME_* first.")


if __name__ == "__main__":
    _main()
