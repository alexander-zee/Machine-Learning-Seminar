"""
Table 1 *analog* for the seminar (CSV), not a byte-for-byte BPZ replication.

The paper's Table 1 mixes: gross vs net Sharpe, HAC-style p-values on SR, CAPM-style
β / SE on the managed portfolio, λ grid at λ*, and turnover. This repo's LASSO path
exports **one** monthly Sharpe definition per split (train / valid / test) — not
separate gross/net columns — and we do **not** yet compute turnover or a market-β
column from a dedicated one-factor regression like the PDF.

This module writes the **closest automated summary** from existing outputs:

- Paper-style characteristic labels (Size + two names)
- Monthly SRs at λ* (train / valid / test)
- λ₀*, λ₂* on the Part 2 grid
- FF5 **tradable** intercept α and SE from ``regression_table_sdf`` (same as Table 3 style)

Columns that **cannot** be filled without extra methodology are left empty (NaN) or
omitted; see the companion ``.txt`` notes.

Row count: **36** cross-sections when ``IdioVol`` is in the panel (from raw ``svar``),
i.e. C(9,2) among non-LME sorts — aligned with Table B.1 enumeration.

Run (from repo root)::

    python -m part_3_metrics_collection.table1_style_bpz_export
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (  # noqa: E402
    all_triplet_pairs,
    triplet_subdir_name,
)
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

TABLES_DIR = REPO_ROOT / "data" / "results" / "tables_seminar"
TREE_CSV = "level_all_excess_combined_filtered.csv"

# Paper / thesis labels (BPZ Table 1 style names); seminar column names on the right
PAPER_CHAR_LABEL: dict[str, str] = {
    "LME": "Size",
    "BEME": "Value",
    "r12_2": "Mom",
    "OP": "OP",
    "Investment": "Inv",
    "ST_Rev": "SRev",
    "LT_Rev": "LRev",
    "AC": "Acc",
    "LTurnover": "Turnover",
    "IdioVol": "IVol",
}

# Reverse map: canonical ap_pruning folder name -> (feat1, feat2)
_SUBDIR_TO_PAIR: dict[str, tuple[str, str]] = {
    triplet_subdir_name(a, b): (a, b) for a, b in all_triplet_pairs()
}


def _paper_triplet_labels(sub_dir: str) -> tuple[str, str, str]:
    p = _SUBDIR_TO_PAIR.get(sub_dir)
    if p is None:
        raise KeyError(
            f"Unknown triplet folder {sub_dir!r}. "
            "Expected one of the 28 folders from all_triplet_pairs()."
        )
    a, b = p
    return (
        PAPER_CHAR_LABEL["LME"],
        PAPER_CHAR_LABEL.get(a, a),
        PAPER_CHAR_LABEL.get(b, b),
    )


def build_table1_style_dataframe(
    ap_root: Path | None = None,
    port_n: int = 10,
    write_selected_ports: bool = True,
) -> pd.DataFrame:
    """
    One row per discovered ``LME_*`` tree cross-section with Part 1+2 complete.
    """
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    rows: list[dict] = []
    for idx, sub in enumerate(discover_lme_tree_ap_subdirs(ap), start=1):
        try:
            c1, c2, c3 = _paper_triplet_labels(sub)
        except KeyError as e:
            print(f"skip {sub}: {e}")
            continue
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

        alpha_ff5 = se_ff5 = np.nan
        pn = int(r["port_n"])
        ports_p = ap / sub / f"Selected_Ports_{pn}.csv"
        w_p = ap / sub / f"Selected_Ports_Weights_{pn}.csv"
        if ports_p.is_file() and w_p.is_file():
            sdf = build_sdf_series(ports_p, w_p, TRADABLE_FACTORS)
            reg = regression_table_sdf(sdf, TRADABLE_FACTORS)
            row5 = reg.loc[reg["spec"] == "FF5"].iloc[0]
            alpha_ff5 = float(row5["alpha"])
            se_ff5 = float(row5["se"])

        rows.append(
            {
                "id": idx,
                "char_1": c1,
                "char_2": c2,
                "char_3": c3,
                "monthly_SR_train": r["train_SR"],
                "monthly_SR_valid": r["valid_SR"],
                "monthly_SR_test": r["test_SR"],
                "lambda_0_star": lam0_star,
                "lambda_2_star": lam2_star,
                "alpha_FF5_tradable": alpha_ff5,
                "se_alpha_FF5_tradable": se_ff5,
                "subdir": sub,
                "port_n": pn,
            }
        )

    return pd.DataFrame(rows)


def write_table1_style_export(
    ap_root: Path | None = None,
    port_n: int = 10,
    out_csv: Path | None = None,
    out_notes: Path | None = None,
) -> tuple[Path, Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = build_table1_style_dataframe(ap_root=ap_root, port_n=port_n)
    csv_path = out_csv or TABLES_DIR / "Table1_style_seminar_cross_section.csv"
    notes_path = out_notes or TABLES_DIR / "Table1_style_seminar_NOTES.txt"
    df.to_csv(csv_path, index=False)
    notes_path.write_text(
        """Table 1 style (seminar) — NOT a full Bryzgalova et al. Table 1 replication

Included
--------
- id, char_1..char_3: paper-style labels (Size = LME; Value = BEME; Mom = r12_2; …)
- monthly_SR_* : single monthly Sharpe definition from the LASSO grid (not separate gross/net)
- lambda_0_star, lambda_2_star : grid values at chosen (i,j) for max validation Sharpe
- alpha_FF5_tradable, se_alpha_FF5_tradable : SDF ~ FF5 tradable factors (see Table 3 notes)

NOT included (would need extra code / data / definitions)
-------------------------------------------------------
- Gross vs net SR (e.g. transaction costs) — seminar uses one SR column per split
- p-value on Sharpe
- CAPM (or other) market β and SE β as in the paper’s “Excess returns” block
- Portfolio turnover
- Row count: 36 once ``prepare_data`` includes ``svar`` → ``IdioVol`` and all triplets are run

For a block of text like the paper’s pasted table, filter/rename columns in Excel or
build a LaTeX table from this CSV after you decide which seminar columns map to which
paper columns.
""",
        encoding="utf-8",
    )
    return csv_path, notes_path


def _main() -> None:
    csv_p, notes_p = write_table1_style_export()
    print(f"Wrote:\n  {csv_p}\n  {notes_p}")
    df = pd.read_csv(csv_p)
    print(df_to_plain_preview(df))


def df_to_plain_preview(df: pd.DataFrame, max_rows: int = 40) -> str:
    """Space-separated preview (not identical to BPZ formatting)."""
    cols = [
        "id",
        "char_1",
        "char_2",
        "char_3",
        "monthly_SR_train",
        "monthly_SR_valid",
        "monthly_SR_test",
        "alpha_FF5_tradable",
        "se_alpha_FF5_tradable",
        "lambda_0_star",
        "lambda_2_star",
        "subdir",
    ]
    use = [c for c in cols if c in df.columns]
    return df[use].head(max_rows).to_string(index=False)


if __name__ == "__main__":
    _main()
