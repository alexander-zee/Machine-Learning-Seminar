"""
Thesis-style **36-row** tables and **Figure 7–style** cross-section SR plot.

- Table: fixed IDs 1–36 and characteristic order from ``cross_section_paper_order``
  (matches standard BPZ enumeration with nine non-LME sorts).
- Columns: SR (monthly **out-of-sample** Sharpe at λ*, i.e. the pipeline’s ``test_SR`` after
  train/validation), FF5 **tradable** α / SE / *p* (SDF regression), λ₀*, λ₂*.

- Figure 7 style: x-axis = cross-section IDs **sorted by ascending AP-tree SR**; optional Ward
  **clustering** reference (horizontal line: same OOS SR for all cross-sections, one MVE on
  cluster returns); optional benchmark CSV (triple-sort / XSF).

- **Four-way export** (``--all-approaches``): tables ``AP``, ``RP``, **time-varying** (``TV_*``),
  and **clustering** (Ward **one-row summary** by default; optional 36-row alignment). Combined figure:
  ``--figure7-multi`` (AP + RP + TV lines + Ward horizontal).

Requires **Part 2** outputs: ``LME_*_*`` (AP), ``RP_LME_*_*`` (RP), ``TV_LME_*_*`` (TV — same
tree return CSVs as AP until you add a TV pruning step), ``Ward_clusters_10`` (clustering).

Run from repo root::

    python -m part_3_metrics_collection.thesis_style_tables_figures --help
    python -m part_3_metrics_collection.thesis_style_tables_figures --all-approaches --write-tex
    python -m part_3_metrics_collection.thesis_style_tables_figures --figure7-multi --figure7-benchmark bench.csv
    python -m part_3_metrics_collection.thesis_style_tables_figures --figure-alpha-t --alpha-factor CAPM

``--figure-alpha-t`` plots **t-statistics on the SDF intercept** from ``sdf ~ 1 + Mkt-RF`` (CAPM) or
``sdf ~ 1 + tradable FF5`` (same factor CSV as the thesis FF5 α columns), x-order by AP-tree SR.

Extension 1 (opt-quantile AP trees) after Part~2 under ``ap_pruning_optquantile``::

    python -m part_3_metrics_collection.thesis_style_tables_figures --approach AP --figure7 \\
        --ap-root data/results/ap_pruning_optquantile \\
        --tree-port-root data/results/tree_portfolios_optquantile \\
        --export-suffix _optquant

**Triple sort / XSF:** not computed in Python here; pass ``--figure7-benchmark path.csv`` with
columns ``paper_id``, ``SR_TS32``, ``SR_TS64`` (and optionally ``SR_XSF``) to overlay lines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from part_1_portfolio_creation.tree_portfolio_creation.cross_section_paper_order import (  # noqa: E402
    THESIS_CHAR_LABELS,
    ap_subdir_for_pair,
    paper_table36_feature_pairs,
    rp_ap_subdir_for_pair,
    tv_ap_subdir_for_pair,
)
from part_1_portfolio_creation.tree_portfolio_creation.cross_section_triplets import (  # noqa: E402
    triplet_subdir_name,
)
from part_3_metrics_collection.pick_best_lambda import (  # noqa: E402
    AP_PRUNE_DEFAULT,
    CLUSTER_RETURNS,
    RP_TREE_PORT_ROOT,
    TREE_PORT_ROOT,
    pick_best_lambda,
)
from part_3_metrics_collection.paper_style_outputs import (  # noqa: E402
    TRADABLE_FACTORS,
    build_sdf_series,
    regression_table_sdf,
)

Approach = Literal["AP", "RP", "TV"]

FIG_DIR = REPO_ROOT / "data" / "results" / "figures_seminar"
TAB_DIR = REPO_ROOT / "data" / "results" / "tables_seminar"
TREE_PORT_FILE = "level_all_excess_combined_filtered.csv"
RP_PORT_FILE = "level_all_excess_combined.csv"


def _portfolio_csv(
    approach: Approach,
    fa: str,
    fb: str,
    *,
    tree_port_root: Path | None = None,
) -> Path:
    # Must match Part 1 folder names (FEATS_LIST order), not raw paper (fa, fb) order.
    sub = triplet_subdir_name(fa, fb)
    if approach == "RP":
        return RP_TREE_PORT_ROOT / sub / RP_PORT_FILE
    root = Path(tree_port_root) if tree_port_root is not None else TREE_PORT_ROOT
    return root / sub / TREE_PORT_FILE


def _ap_subdir(approach: Approach, fa: str, fb: str) -> str:
    if approach == "AP":
        return ap_subdir_for_pair(fa, fb)
    if approach == "RP":
        return rp_ap_subdir_for_pair(fa, fb)
    return tv_ap_subdir_for_pair(fa, fb)


def _ff5_pct_from_pick(ap: Path, sub_ap: str, r: dict) -> tuple[float, float, float]:
    """Tradable FF5 α/SE/p (α, SE as percent points) from Selected_Ports* written by pick."""
    a_pct = se_pct = p_ff = np.nan
    pn = int(r["port_n"])
    ports_p = ap / sub_ap / f"Selected_Ports_{pn}.csv"
    w_p = ap / sub_ap / f"Selected_Ports_Weights_{pn}.csv"
    if ports_p.is_file() and w_p.is_file():
        sdf = build_sdf_series(ports_p, w_p, TRADABLE_FACTORS)
        reg = regression_table_sdf(sdf, TRADABLE_FACTORS)
        row5 = reg.loc[reg["spec"] == "FF5"].iloc[0]
        a_pct = float(row5["alpha"]) * 100.0
        se_pct = float(row5["se"]) * 100.0
        p_ff = float(row5["p_value"])
    return a_pct, se_pct, p_ff


def _factor_tstat_from_pick(ap: Path, sub_ap: str, r: dict, factor_spec: str) -> float:
    """
    t-statistic on the intercept from ``sdf ~ 1 + factors`` (see ``regression_table_sdf``).

    ``factor_spec``: ``CAPM`` (Mkt-RF only) or ``FF5`` (tradable five-factor block).
    """
    if factor_spec not in ("CAPM", "FF5"):
        raise ValueError("factor_spec must be CAPM or FF5")
    pn = int(r["port_n"])
    ports_p = ap / sub_ap / f"Selected_Ports_{pn}.csv"
    w_p = ap / sub_ap / f"Selected_Ports_Weights_{pn}.csv"
    if not ports_p.is_file() or not w_p.is_file():
        return np.nan
    try:
        sdf = build_sdf_series(ports_p, w_p, TRADABLE_FACTORS)
        reg = regression_table_sdf(sdf, TRADABLE_FACTORS)
        row = reg.loc[reg["spec"] == factor_spec]
        if row.empty:
            return np.nan
        return float(row.iloc[0]["t_stat"])
    except Exception:
        return np.nan


def build_thesis_factor_tstat_table(
    approach: Approach,
    port_n: int = 10,
    ap_root: Path | None = None,
    write_pick_tables: bool = True,
    tree_port_root: Path | None = None,
    factor_spec: str = "CAPM",
) -> pd.DataFrame:
    """
    One row per paper ID 1–36: t-stat on alpha for the chosen SDF at λ* (needs Selected_Ports* from pick).
    """
    if factor_spec not in ("CAPM", "FF5"):
        raise ValueError("factor_spec must be CAPM or FF5")
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    rows: list[dict] = []
    for paper_id, (fa, fb) in enumerate(paper_table36_feature_pairs(), start=1):
        c1 = THESIS_CHAR_LABELS["LME"]
        c2 = THESIS_CHAR_LABELS[fa]
        c3 = THESIS_CHAR_LABELS[fb]
        sub_ap = _ap_subdir(approach, fa, fb)
        pcsv = _portfolio_csv(approach, fa, fb, tree_port_root=tree_port_root)
        base = {
            "ID": paper_id,
            "char_1": c1,
            "char_2": c2,
            "char_3": c3,
            "ap_subdir": sub_ap,
        }
        if not pcsv.is_file() or not (ap / sub_ap).is_dir():
            rows.append({**base, "t_stat": np.nan})
            continue
        try:
            r = pick_best_lambda(
                ap,
                sub_ap,
                port_n,
                pcsv,
                returns_index_col=None,
                write_tables=write_pick_tables,
            )
        except Exception as e:
            print(f"ID {paper_id} {sub_ap}: pick_best failed: {e}")
            rows.append({**base, "t_stat": np.nan})
            continue
        t_val = _factor_tstat_from_pick(ap, sub_ap, r, factor_spec)
        rows.append({**base, "t_stat": t_val})

    return pd.DataFrame(rows)


def build_thesis_summary_table(
    approach: Approach,
    port_n: int = 10,
    ap_root: Path | None = None,
    write_pick_tables: bool = False,
    tree_port_root: Path | None = None,
    include_factor_tstats: bool = False,
) -> pd.DataFrame:
    """
    One row per paper ID 1–36; missing Part~2 → NaNs for numeric fields.

    If ``include_factor_tstats`` is True, adds ``t_alpha_CAPM`` and ``t_alpha_FF5`` (t-stats on the
    intercept in ``sdf ~ 1 + Mkt-RF`` and ``sdf ~ 1 + tradable FF5``), same regression window as
    ``paper_style_outputs.regression_table_sdf``.
    """
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    rows: list[dict] = []
    for paper_id, (fa, fb) in enumerate(paper_table36_feature_pairs(), start=1):
        c1 = THESIS_CHAR_LABELS["LME"]
        c2 = THESIS_CHAR_LABELS[fa]
        c3 = THESIS_CHAR_LABELS[fb]
        sub_ap = _ap_subdir(approach, fa, fb)
        pcsv = _portfolio_csv(approach, fa, fb, tree_port_root=tree_port_root)
        base = {
            "ID": paper_id,
            "char_1": c1,
            "char_2": c2,
            "char_3": c3,
            "ap_subdir": sub_ap,
        }
        if not pcsv.is_file() or not (ap / sub_ap).is_dir():
            row_nan = {
                **base,
                "SR": np.nan,
                "alpha_FF5_pct": np.nan,
                "SE_FF5_pct": np.nan,
                "p_FF5": np.nan,
                "lambda_0": np.nan,
                "lambda_2": np.nan,
            }
            if include_factor_tstats:
                row_nan["t_alpha_CAPM"] = np.nan
                row_nan["t_alpha_FF5"] = np.nan
            rows.append(row_nan)
            continue
        try:
            r = pick_best_lambda(
                ap,
                sub_ap,
                port_n,
                pcsv,
                returns_index_col=None,
                write_tables=write_pick_tables,
            )
        except Exception as e:
            print(f"ID {paper_id} {sub_ap}: pick_best failed: {e}")
            row_nan = {
                **base,
                "SR": np.nan,
                "alpha_FF5_pct": np.nan,
                "SE_FF5_pct": np.nan,
                "p_FF5": np.nan,
                "lambda_0": np.nan,
                "lambda_2": np.nan,
            }
            if include_factor_tstats:
                row_nan["t_alpha_CAPM"] = np.nan
                row_nan["t_alpha_FF5"] = np.nan
            rows.append(row_nan)
            continue

        la0 = np.asarray(r["lambda0"], dtype=float)
        la2 = np.asarray(r["lambda2"], dtype=float)
        i0, j0 = r["best_i_lambda0"] - 1, r["best_j_lambda2"] - 1
        lam0_star = float(la0[i0])
        lam2_star = float(la2[j0])

        a_pct, se_pct, p_ff = _ff5_pct_from_pick(ap, sub_ap, r)

        row_ok: dict = {
            **base,
            "SR": float(r["test_SR"]),
            "alpha_FF5_pct": a_pct,
            "SE_FF5_pct": se_pct,
            "p_FF5": p_ff,
            "lambda_0": lam0_star,
            "lambda_2": lam2_star,
        }
        if include_factor_tstats:
            # Needs Selected_Ports* — use write_pick_tables=True at least once before this path
            row_ok["t_alpha_CAPM"] = _factor_tstat_from_pick(ap, sub_ap, r, "CAPM")
            row_ok["t_alpha_FF5"] = _factor_tstat_from_pick(ap, sub_ap, r, "FF5")
        rows.append(row_ok)

    return pd.DataFrame(rows)


def _ward_clustering_metrics(
    port_n: int,
    ap: Path,
    *,
    write_pick_tables: bool,
) -> tuple[float, float, float, float, float, float]:
    """(SR, alpha_pct, SE_pct, p_FF5, lambda_0, lambda_2) from one Ward pick; NaNs on failure."""
    sr = a_pct = se_pct = p_ff = lam0 = lam2 = np.nan
    try:
        if not CLUSTER_RETURNS.is_file():
            raise FileNotFoundError(f"Missing {CLUSTER_RETURNS}")
        r = pick_best_lambda(
            ap,
            WARD_SUB,
            port_n,
            CLUSTER_RETURNS,
            returns_index_col=0,
            write_tables=write_pick_tables,
        )
        sr = float(r["test_SR"])
        la0 = np.asarray(r["lambda0"], dtype=float)
        la2 = np.asarray(r["lambda2"], dtype=float)
        i0, j0 = r["best_i_lambda0"] - 1, r["best_j_lambda2"] - 1
        lam0, lam2 = float(la0[i0]), float(la2[j0])
        a_pct, se_pct, p_ff = _ff5_pct_from_pick(ap, WARD_SUB, r)
    except Exception as e:
        print(f"Clustering / Ward pick failed: {e}")
    return sr, a_pct, se_pct, p_ff, lam0, lam2


def build_thesis_clustering_summary(
    port_n: int = 10,
    ap_root: Path | None = None,
    *,
    write_pick_tables: bool = False,
) -> pd.DataFrame:
    """
    **Thesis-friendly:** one row for the single global Ward MVE (same columns as other tables).

    Use this in the thesis instead of repeating identical numbers on 36 rows.
    """
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    sr, a_pct, se_pct, p_ff, lam0, lam2 = _ward_clustering_metrics(
        port_n, ap, write_pick_tables=write_pick_tables
    )
    return pd.DataFrame(
        [
            {
                "ID": "---",
                "char_1": "Ward (global)",
                "char_2": "10 portfolios",
                "char_3": "n/a",
                "ap_subdir": WARD_SUB,
                "SR": sr,
                "alpha_FF5_pct": a_pct,
                "SE_FF5_pct": se_pct,
                "p_FF5": p_ff,
                "lambda_0": lam0,
                "lambda_2": lam2,
            }
        ]
    )


def build_thesis_clustering_table36(
    port_n: int = 10,
    ap_root: Path | None = None,
    *,
    write_pick_tables: bool = False,
) -> pd.DataFrame:
    r"""
    36 rows: same ID / characteristic labels as AP-trees; SR, α, λ* repeat the single Ward MVE.

    For thesis text, prefer ``build_thesis_clustering_summary`` (one row). This table is only for
    alignment with cross-section IDs or debugging.
    """
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    sr, a_pct, se_pct, p_ff, lam0, lam2 = _ward_clustering_metrics(
        port_n, ap, write_pick_tables=write_pick_tables
    )

    rows: list[dict] = []
    for paper_id, (fa, fb) in enumerate(paper_table36_feature_pairs(), start=1):
        rows.append(
            {
                "ID": paper_id,
                "char_1": THESIS_CHAR_LABELS["LME"],
                "char_2": THESIS_CHAR_LABELS[fa],
                "char_3": THESIS_CHAR_LABELS[fb],
                "ap_subdir": WARD_SUB,
                "SR": sr,
                "alpha_FF5_pct": a_pct,
                "SE_FF5_pct": se_pct,
                "p_FF5": p_ff,
                "lambda_0": lam0,
                "lambda_2": lam2,
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_latex_longtable(
    df: pd.DataFrame,
    caption: str,
    label: str,
    path: Path,
) -> Path:
    """Minimal ``longtable`` body: numeric cells formatted for LaTeX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        r"\begin{longtable}{c c c c c c c c c c}",
        rf"\caption{{{caption}}} \label{{{label}}} \\",
        r"\toprule",
        r" & \multicolumn{3}{c}{Characteristics} & \multicolumn{1}{c}{Sharpe Ratio} & \multicolumn{3}{c}{Alpha (FF5 tradable)} & \multicolumn{2}{c}{Shrinkage} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-5} \cmidrule(lr){6-8} \cmidrule(lr){9-10}",
        r"ID & 1 & 2 & 3 & SR & $\alpha$ (\%) & SE & $p$-value & $\lambda_0$ & $\lambda_2$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"ID & 1 & 2 & 3 & SR & $\alpha$ (\%) & SE & $p$-value & $\lambda_0$ & $\lambda_2$ \\",
        r"\midrule",
        r"\endhead",
    ]

    def fmt_num(x: float, nd: int = 4) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "---"
        return f"{x:.{nd}f}"

    def fmt_id_cell(v) -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "---"
        if isinstance(v, str):
            return v.replace("&", r"\&")
        try:
            return str(int(v))
        except (ValueError, TypeError):
            return str(v).replace("&", r"\&")

    for _, row in df.iterrows():
        p = row.get("p_FF5", np.nan)
        pstr = f"{float(p):.6f}" if np.isfinite(p) else "---"
        line = (
            f"{fmt_id_cell(row['ID'])} & {row['char_1']} & {row['char_2']} & {row['char_3']} & "
            f"{fmt_num(row.get('SR', np.nan), 4)} & "
            f"{fmt_num(row.get('alpha_FF5_pct', np.nan), 4)} & "
            f"{fmt_num(row.get('SE_FF5_pct', np.nan), 4)} & "
            f"{pstr} & "
            f"{fmt_num(row.get('lambda_0', np.nan), 4)} & "
            f"{fmt_num(row.get('lambda_2', np.nan), 4)} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


WARD_SUB = "Ward_clusters_10"


def _ward_oos_monthly_sr(ap_root: Path, port_n: int) -> float | None:
    """Same OOS monthly Sharpe as thesis SR column: ``test_SR`` at validation-chosen λ* on Ward outputs."""
    ward = ap_root / WARD_SUB
    if not ward.is_dir() or not CLUSTER_RETURNS.is_file():
        return None
    try:
        r = pick_best_lambda(
            ap_root,
            WARD_SUB,
            port_n,
            CLUSTER_RETURNS,
            returns_index_col=0,
            write_tables=False,
        )
        return float(r["test_SR"])
    except Exception:
        return None


def _ward_oos_monthly_sr_with_fallback(ap_primary: Path | None, port_n: int) -> float | None:
    """Try ``ap_primary`` for ``Ward_clusters_10``; if missing, try baseline ``ap_pruning`` (opt-quantile folders often have no Ward)."""
    primary = Path(ap_primary) if ap_primary is not None else AP_PRUNE_DEFAULT
    w = _ward_oos_monthly_sr(primary, port_n)
    if w is not None:
        return w
    try:
        if primary.resolve() == AP_PRUNE_DEFAULT.resolve():
            return None
    except OSError:
        pass
    return _ward_oos_monthly_sr(AP_PRUNE_DEFAULT, port_n)


def _order_ids_figure7(
    df_ap: pd.DataFrame,
    df_rp: pd.DataFrame | None = None,
    df_tv: pd.DataFrame | None = None,
) -> list[int]:
    """
    All 36 paper IDs on the x-axis, ordered by ascending **AP-tree** SR (NaNs last), then RP, then TV
    if AP has no finite SR at all (same fallback order as ``_sort_ids_by_first_available_sr``).
    """
    all_ids = list(range(1, 37))
    for df in (df_ap, df_rp, df_tv):
        if df is None:
            continue
        s = df.set_index("ID")["SR"].reindex(all_ids)
        if s.notna().any():
            return [int(i) for i in s.sort_values(ascending=True, na_position="last").index]
    raise ValueError(
        "No finite SR values in any of the provided tables; need at least AP, RP, or TV results."
    )


def plot_figure7_multi(
    df_ap: pd.DataFrame,
    out_path: Path,
    *,
    df_rp: pd.DataFrame | None = None,
    df_tv: pd.DataFrame | None = None,
    ap_root: Path | None = None,
    port_n: int = 10,
    include_clustering: bool = True,
    benchmark_csv: Path | None = None,
    title: str | None = None,
) -> Path:
    """
    Four-way comparison: AP-, RP-, and TV-tree SR by cross-section (sorted by **AP** SR, NaNs last),
    plus Ward clustering as a horizontal line.
    """
    ids = _order_ids_figure7(df_ap, df_rp, df_tv)
    x = np.arange(len(ids))
    n = len(ids)
    use_ids = pd.DataFrame({"ID": ids})

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Align each approach to the same x-order (paper IDs)
    def aligned_sr(df: pd.DataFrame | None) -> np.ndarray | None:
        if df is None:
            return None
        m = df.set_index("ID").reindex(ids)
        return m["SR"].astype(float).values

    y_ap = aligned_sr(df_ap)
    if y_ap is None:
        y_ap = np.full(n, np.nan)
    y_rp = aligned_sr(df_rp)
    y_tv = aligned_sr(df_tv)

    ymax = 0.0
    for arr in (y_ap, y_rp, y_tv):
        if arr is not None and np.isfinite(arr).any():
            ymax = max(ymax, float(np.nanmax(arr)))

    if np.isfinite(y_ap).any():
        ax.plot(
            x,
            y_ap,
            color="#d62728",
            marker="o",
            markersize=4,
            linewidth=1.2,
            linestyle="-",
            label="AP-Tree",
            zorder=4,
        )
        ymax = max(ymax, float(np.nanmax(y_ap)))
    if y_rp is not None and np.isfinite(y_rp).any():
        ax.plot(
            x,
            y_rp,
            color="#1f77b4",
            marker="s",
            markersize=4,
            linewidth=1.2,
            linestyle=":",
            label="RP-Tree",
            zorder=3,
        )
        ymax = max(ymax, float(np.nanmax(y_rp)))
    if y_tv is not None and np.isfinite(y_tv).any():
        ax.plot(
            x,
            y_tv,
            color="#2ca02c",
            marker="^",
            markersize=4,
            linewidth=1.2,
            linestyle=":",
            label="Time-varying (TV)",
            zorder=3,
        )
        ymax = max(ymax, float(np.nanmax(y_tv)))

    ap_trees = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    w_sr: float | None = None
    if include_clustering:
        w_sr = _ward_oos_monthly_sr_with_fallback(ap_trees, port_n)
        if w_sr is None:
            print(
                "Figure 7 (multi): clustering line skipped — need Ward Part~2 outputs and "
                "cluster_returns.csv."
            )
        else:
            ax.plot(
                x,
                np.full(n, w_sr),
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.4,
                label="Clustering (Ward)",
                zorder=2,
            )
            ymax = max(ymax, w_sr)

    if benchmark_csv is not None and Path(benchmark_csv).is_file():
        b = pd.read_csv(benchmark_csv)
        if "paper_id" in b.columns and "SR_TS32" in b.columns:
            m = use_ids.merge(b, left_on="ID", right_on="paper_id", how="left")
            if m["SR_TS32"].notna().any():
                ax.plot(
                    x,
                    m["SR_TS32"].values,
                    color="#8c564b",
                    linestyle=":",
                    marker="D",
                    markersize=3,
                    label="Triple sort (32)",
                )
                ts = float(np.nanmax(m["SR_TS32"].values))
                if np.isfinite(ts):
                    ymax = max(ymax, ts)
        if "SR_TS64" in b.columns:
            m64 = use_ids.merge(b, left_on="ID", right_on="paper_id", how="left")
            if m64["SR_TS64"].notna().any():
                ax.plot(
                    x,
                    m64["SR_TS64"].values,
                    color="#9467bd",
                    linestyle=":",
                    marker="P",
                    markersize=3,
                    label="Triple sort (64)",
                )
                ts64 = float(np.nanmax(m64["SR_TS64"].values))
                if np.isfinite(ts64):
                    ymax = max(ymax, ts64)
        if "SR_XSF" in b.columns:
            mx = use_ids.merge(b, left_on="ID", right_on="paper_id", how="left")
            if mx["SR_XSF"].notna().any():
                ax.plot(
                    x,
                    mx["SR_XSF"].values,
                    color="#7f7f7f",
                    linestyle="--",
                    marker="+",
                    markersize=5,
                    label="XSF",
                )
                tx = float(np.nanmax(mx["SR_XSF"].values))
                if np.isfinite(tx):
                    ymax = max(ymax, tx)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ids], rotation=90, fontsize=7)
    ax.set_xlabel("Cross-sections")
    ax.set_ylabel("Monthly Sharpe Ratio (SR)")
    ax.axhline(0.0, color="k", linewidth=0.6, zorder=1)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, color="0.85", alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0.0, top=max(0.05, ymax * 1.08))

    if title is None:
        title = "AP-, RP-, time-varying trees vs. clustering: monthly Sharpe ratios"
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(
        title="Basis portfolios:",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _ward_factor_tstat_with_fallback(
    ap_primary: Path | None, port_n: int, factor_spec: str
) -> float | None:
    """Ward clustering: one t-stat (same at every x); try primary ap_root then baseline."""
    primary = Path(ap_primary) if ap_primary is not None else AP_PRUNE_DEFAULT
    for ap_try in (primary, AP_PRUNE_DEFAULT):
        ward = ap_try / WARD_SUB
        if not ward.is_dir() or not CLUSTER_RETURNS.is_file():
            continue
        try:
            r = pick_best_lambda(
                ap_try,
                WARD_SUB,
                port_n,
                CLUSTER_RETURNS,
                returns_index_col=0,
                write_tables=True,
            )
            t = _factor_tstat_from_pick(ap_try, WARD_SUB, r, factor_spec)
            if np.isfinite(t):
                return float(t)
        except Exception:
            continue
    return None


def plot_figure_alpha_t_multi(
    df_ap: pd.DataFrame,
    out_path: Path,
    *,
    df_rp: pd.DataFrame | None = None,
    df_tv: pd.DataFrame | None = None,
    ap_root: Path | None = None,
    port_n: int = 10,
    include_clustering: bool = True,
    factor_spec: str = "CAPM",
    t_column: str = "t_alpha_CAPM",
    title: str | None = None,
) -> Path:
    """
    Like ``plot_figure7_multi`` but y = **t-statistic on alpha** from ``sdf ~ 1 + factors``
    (CAPM = Mkt-RF only; FF5 = tradable five-factor block). X-order still by ascending AP **SR**
    (from SR table — pass the same ``df_ap`` used for Figure 7 with ``SR`` column).

    ``t_column`` must exist on each dataframe (e.g. ``t_alpha_CAPM`` from
    ``build_thesis_summary_table(..., include_factor_tstats=True)``).
    """
    if "SR" not in df_ap.columns:
        raise ValueError("df_ap must include SR column for x-axis ordering (same as Figure 7).")
    ids = _order_ids_figure7(df_ap, df_rp, df_tv)
    x = np.arange(len(ids))
    n = len(ids)

    def aligned_t(df: pd.DataFrame | None) -> np.ndarray | None:
        if df is None or t_column not in df.columns:
            return None
        m = df.set_index("ID").reindex(ids)
        return m[t_column].astype(float).values

    y_ap = aligned_t(df_ap)
    if y_ap is None:
        y_ap = np.full(n, np.nan)
    y_rp = aligned_t(df_rp)
    y_tv = aligned_t(df_tv)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    spec_lab = "CAPM (Mkt-RF)" if factor_spec == "CAPM" else "tradable FF5"
    if np.isfinite(y_ap).any():
        ax.plot(
            x,
            y_ap,
            color="#d62728",
            marker="o",
            markersize=4,
            linewidth=1.2,
            linestyle="-",
            label="AP-Tree",
            zorder=4,
        )
    if y_rp is not None and np.isfinite(y_rp).any():
        ax.plot(
            x,
            y_rp,
            color="#1f77b4",
            marker="s",
            markersize=4,
            linewidth=1.2,
            linestyle=":",
            label="RP-Tree",
            zorder=3,
        )
    if y_tv is not None and np.isfinite(y_tv).any():
        ax.plot(
            x,
            y_tv,
            color="#2ca02c",
            marker="^",
            markersize=4,
            linewidth=1.2,
            linestyle=":",
            label="Time-varying (TV)",
            zorder=3,
        )

    ap_trees = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    if include_clustering:
        w_t = _ward_factor_tstat_with_fallback(ap_trees, port_n, factor_spec)
        if w_t is None:
            print(
                "Alpha-t figure: Ward t-stat line skipped — need Ward Part~2 and cluster_returns.csv."
            )
        else:
            ax.plot(
                x,
                np.full(n, w_t),
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.4,
                label="Clustering (Ward)",
                zorder=2,
            )

    ax.axhline(0.0, color="k", linewidth=0.6, zorder=1)
    ax.axhline(1.96, color="0.55", linewidth=0.8, linestyle="--", zorder=0)
    ax.axhline(-1.96, color="0.55", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ids], rotation=90, fontsize=7)
    ax.set_xlabel("Cross-sections")
    ax.set_ylabel("t-statistic (alpha)")
    ax.grid(axis="y", linestyle="-", linewidth=0.8, color="0.85", alpha=0.9)
    ax.set_axisbelow(True)

    ymax = 3.0
    ymin = -3.0
    for arr in (y_ap, y_rp, y_tv):
        if arr is not None and np.isfinite(arr).any():
            ymax = max(ymax, float(np.nanmax(arr)))
            ymin = min(ymin, float(np.nanmin(arr)))
    pad = max(0.5, (ymax - ymin) * 0.08)
    ax.set_ylim(ymin - pad, ymax + pad)

    if title is None:
        title = (
            f"SDF alpha t-statistics vs. {spec_lab} — AP-, RP-, TV-trees vs. Ward "
            "(x sorted by AP-tree SR)"
        )
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(
        title="Basis portfolios:",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_figure7_style(
    df: pd.DataFrame,
    out_path: Path,
    benchmark_csv: Path | None = None,
    *,
    ap_root: Path | None = None,
    port_n: int = 10,
    include_clustering: bool = True,
    primary_label: str = "AP-Tree",
    title: str | None = None,
) -> Path:
    """
    Sort rows by ``SR`` ascending; x-tick labels = ``ID`` in that order.

    Styling aligned with BPZ-style Figure 7: paper-like axis labels (no “test SR” wording);
    Ward clustering is a **horizontal** reference (one MVE on cluster returns, same SR read
    at each x position), matching ``paper_style_bpz_figures`` Fig.~3.
    """
    use = df.dropna(subset=["SR"]).copy()
    if use.empty:
        raise ValueError("No finite SR values to plot.")
    use = use.sort_values("SR", ascending=True).reset_index(drop=True)
    x = np.arange(len(use))
    n = len(use)

    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.plot(
        x,
        use["SR"].values,
        color="#d62728",
        marker="o",
        markersize=4,
        linewidth=1.2,
        linestyle="-",
        label=primary_label,
        zorder=3,
    )

    ap_trees = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    w_sr: float | None = None
    if include_clustering:
        w_sr = _ward_oos_monthly_sr_with_fallback(ap_trees, port_n)
        if w_sr is None:
            print(
                "Figure 7: clustering line skipped — need ap_pruning/Ward_clusters_10, "
                "portfolios/clusters/cluster_returns.csv, and a successful Ward pick "
                "(run Part 2 with clusters enabled once)."
            )
        if w_sr is not None:
            ax.plot(
                x,
                np.full(n, w_sr),
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.4,
                label="Clustering (Ward)",
                zorder=2,
            )

    ymax = float(np.nanmax(use["SR"].values))
    if w_sr is not None:
        ymax = max(ymax, w_sr)

    if benchmark_csv is not None and Path(benchmark_csv).is_file():
        b = pd.read_csv(benchmark_csv)
        if "paper_id" in b.columns and "SR_TS32" in b.columns:
            m = use[["ID"]].merge(b, left_on="ID", right_on="paper_id", how="left")
            if m["SR_TS32"].notna().any():
                ax.plot(
                    x,
                    m["SR_TS32"].values,
                    color="#2ca02c",
                    linestyle=":",
                    marker="^",
                    markersize=4,
                    label="Triple sort (32)",
                )
                ts = float(np.nanmax(m["SR_TS32"].values))
                if np.isfinite(ts):
                    ymax = max(ymax, ts)
        if "SR_TS64" in b.columns:
            m64 = use[["ID"]].merge(b, left_on="ID", right_on="paper_id", how="left")
            if m64["SR_TS64"].notna().any():
                ax.plot(
                    x,
                    m64["SR_TS64"].values,
                    color="#1f77b4",
                    linestyle=":",
                    marker="s",
                    markersize=4,
                    label="Triple sort (64)",
                )
                ts64 = float(np.nanmax(m64["SR_TS64"].values))
                if np.isfinite(ts64):
                    ymax = max(ymax, ts64)
        if "SR_XSF" in b.columns:
            mx = use[["ID"]].merge(b, left_on="ID", right_on="paper_id", how="left")
            if mx["SR_XSF"].notna().any():
                ax.plot(
                    x,
                    mx["SR_XSF"].values,
                    color="#9467bd",
                    linestyle="--",
                    marker="+",
                    markersize=5,
                    label="XSF",
                )
                tx = float(np.nanmax(mx["SR_XSF"].values))
                if np.isfinite(tx):
                    ymax = max(ymax, tx)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(i)) for i in use["ID"].values], rotation=90, fontsize=7)
    ax.set_xlabel("Cross-sections")
    ax.set_ylabel("Monthly Sharpe Ratio (SR)")
    ax.axhline(0.0, color="k", linewidth=0.6, zorder=1)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, color="0.85", alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0.0, top=max(0.05, ymax * 1.08))

    if title is None:
        title = f"{primary_label}s vs. clustering: monthly Sharpe ratios by cross-section"
    ax.set_title(title, fontsize=11, pad=8)

    ax.legend(
        title="Basis portfolios:",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _main_cli() -> None:
    p = argparse.ArgumentParser(description="Thesis tables / Figure 7 style exports.")
    p.add_argument("--approach", choices=["AP", "RP", "TV"], default="AP")
    p.add_argument("--port-n", type=int, default=10)
    p.add_argument("--write-tex", action="store_true")
    p.add_argument("--figure7", action="store_true")
    p.add_argument(
        "--figure7-multi",
        action="store_true",
        help="Combined figure: AP + RP + TV (if data exist) + Ward clustering horizontal line",
    )
    p.add_argument(
        "--all-approaches",
        action="store_true",
        help="Write AP/RP/TV 36-row tables + Ward clustering (default: one-row summary; use --clustering-36-row for 36-row alignment).",
    )
    p.add_argument(
        "--figure7-benchmark",
        type=Path,
        default=None,
        help="CSV columns: paper_id, SR_TS32, SR_TS64; optional SR_XSF",
    )
    p.add_argument(
        "--caption",
        default="Summary statistics (seminar pipeline; FF5 = tradable factors)",
    )
    p.add_argument(
        "--no-figure7-clustering",
        action="store_true",
        help="Do not draw Ward clustering reference line on --figure7 / --figure7-multi",
    )
    p.add_argument(
        "--clustering-only",
        action="store_true",
        help="Only export Ward clustering table (needs Ward Part 2 + cluster_returns.csv).",
    )
    p.add_argument(
        "--clustering-36-row",
        action="store_true",
        help="Repeat Ward metrics on 36 rows (ID 1–36); default is a one-row thesis summary.",
    )
    p.add_argument(
        "--ap-root",
        type=Path,
        default=None,
        help="Part 2 output root (default: data/results/ap_pruning). Use ap_pruning_optquantile for opt-quantile trees.",
    )
    p.add_argument(
        "--tree-port-root",
        type=Path,
        default=None,
        help="AP/TV filtered tree CSV root (default: data/results/tree_portfolios). Use tree_portfolios_optquantile for extension 1.",
    )
    p.add_argument(
        "--clustering-ap-root",
        type=Path,
        default=None,
        help="Folder containing Ward_clusters_10 (default: --ap-root if Ward exists there, else baseline ap_pruning).",
    )
    p.add_argument(
        "--export-suffix",
        default="",
        help="Append to exported .csv/.tex/.png basenames (e.g. _optquant) to avoid overwriting baseline tables.",
    )
    p.add_argument(
        "--figure-alpha-t",
        action="store_true",
        help="Plot t-stats on SDF alpha (CAPM or FF5) for AP/RP/TV + Ward horizontal; writes extended CSVs.",
    )
    p.add_argument(
        "--alpha-factor",
        choices=["CAPM", "FF5"],
        default="CAPM",
        help="Benchmark for alpha regression (with --figure-alpha-t). CAPM = Mkt-RF only; FF5 = tradable block.",
    )
    args = p.parse_args()

    cli_ap = args.ap_root if args.ap_root is not None else AP_PRUNE_DEFAULT
    cli_tree = args.tree_port_root if args.tree_port_root is not None else TREE_PORT_ROOT
    ward_ap = (
        args.clustering_ap_root
        if args.clustering_ap_root is not None
        else (cli_ap if (cli_ap / WARD_SUB).is_dir() else AP_PRUNE_DEFAULT)
    )
    sfx = args.export_suffix or ""

    if args.clustering_only:
        if args.all_approaches:
            p.error("Use either --clustering-only or --all-approaches, not both.")
        if args.clustering_36_row:
            df_cl = build_thesis_clustering_table36(
                port_n=args.port_n,
                ap_root=ward_ap,
                write_pick_tables=True,
            )
            base, tex_label = "Thesis_Table36_Clustering_Ward", "tab:thesis36_clustering_ward"
            notes_txt = (
                "FF5 α/SE/p: SDF ~ tradable FF5 factors (see paper_style_outputs NOTES).\n"
                "SR: monthly out-of-sample Sharpe at λ* from validation max Sharpe.\n"
                "36 rows: identical Ward metrics repeated for ID alignment with AP/TV/RP tables only.\n"
                "For the thesis, prefer the default one-row summary (omit --clustering-36-row).\n"
            )
        else:
            df_cl = build_thesis_clustering_summary(
                port_n=args.port_n,
                ap_root=ward_ap,
                write_pick_tables=True,
            )
            base, tex_label = "Thesis_Clustering_Ward_summary", "tab:thesis_clustering_ward_summary"
            notes_txt = (
                "Single global Ward MVE on cluster_returns.csv (not cross-section-specific).\n"
                "FF5 α/SE/p: tradable factors; SR: OOS monthly Sharpe at λ*.\n"
            )
        base_out = f"{base}{sfx}"
        csv_c = TAB_DIR / f"{base_out}.csv"
        write_csv(df_cl, csv_c)
        print(f"Wrote {csv_c}")
        (TAB_DIR / f"{base_out}_NOTES.txt").write_text(notes_txt, encoding="utf-8")
        print(f"Wrote {TAB_DIR / f'{base_out}_NOTES.txt'}")
        if args.write_tex:
            tex_c = TAB_DIR / f"{base_out}.tex"
            write_latex_longtable(
                df_cl,
                "Summary statistics of portfolios formed from Clustering approach",
                tex_label,
                tex_c,
            )
            print(f"Wrote {tex_c}")
        print("Done (--clustering-only).")
        return

    common_notes = (
        "FF5 α/SE/p: SDF ~ tradable FF5 factors (see paper_style_outputs NOTES).\n"
        "SR: monthly out-of-sample Sharpe (hold-out / test window) at λ* from validation max Sharpe.\n"
        "Rows with ---: missing Part 2 or failed pick for that cross-section.\n"
    )

    dfs_multi: dict[str, pd.DataFrame] = {}
    df_one: pd.DataFrame | None = None
    tag_one: str | None = None

    if args.all_approaches:
        bundles: list[tuple[Approach, str, str]] = [
            (
                "AP",
                "AP_trees",
                "Summary statistics of portfolios formed from AP-Trees",
            ),
            (
                "RP",
                "RP_trees",
                "Summary statistics of portfolios formed from Random Projection Approach",
            ),
            (
                "TV",
                "TV_trees",
                "Summary statistics of portfolios formed from Time-Varying Approach",
            ),
        ]
        for appr, tag, caption in bundles:
            df_b = build_thesis_summary_table(
                appr,
                port_n=args.port_n,
                write_pick_tables=True,
                ap_root=cli_ap,
                tree_port_root=cli_tree,
            )
            dfs_multi[appr] = df_b
            tag_fn = f"Thesis_Table36_{tag}{sfx}"
            csv_p = TAB_DIR / f"{tag_fn}.csv"
            write_csv(df_b, csv_p)
            print(f"Wrote {csv_p}")
            notes_p = TAB_DIR / f"{tag_fn}_NOTES.txt"
            extra = ""
            if appr == "TV":
                extra = (
                    "TV: ap_pruning/TV_LME_*_* from TV_Pruning (recency kernel on train moments; "
                    "run_all_tv_cross_sections.py). Env TV_KERNEL_HALFLIFE_MONTHS (default 72).\n"
                )
            notes_p.write_text(common_notes + extra, encoding="utf-8")
            print(f"Wrote {notes_p}")
            if args.write_tex:
                tex = TAB_DIR / f"{tag_fn}.tex"
                write_latex_longtable(df_b, caption, f"tab:thesis36_{tag.lower()}", tex)
                print(f"Wrote {tex}")

        if args.clustering_36_row:
            df_cl = build_thesis_clustering_table36(
                port_n=args.port_n,
                ap_root=ward_ap,
                write_pick_tables=True,
            )
            cbase = "Thesis_Table36_Clustering_Ward"
            cl_notes = (
                common_notes
                + "Clustering (36-row): identical Ward metrics on every row — alignment layout only.\n"
                "Default export is one-row summary; omit --clustering-36-row next time for thesis text.\n"
            )
            tex_label_cl = "tab:thesis36_clustering_ward"
        else:
            df_cl = build_thesis_clustering_summary(
                port_n=args.port_n,
                ap_root=ward_ap,
                write_pick_tables=True,
            )
            cbase = "Thesis_Clustering_Ward_summary"
            cl_notes = (
                common_notes
                + "Clustering: single global Ward MVE (one-row summary).\n"
            )
            tex_label_cl = "tab:thesis_clustering_ward_summary"
        dfs_multi["CL"] = df_cl
        cbase_fn = f"{cbase}{sfx}"
        csv_c = TAB_DIR / f"{cbase_fn}.csv"
        write_csv(df_cl, csv_c)
        print(f"Wrote {csv_c}")
        (TAB_DIR / f"{cbase_fn}_NOTES.txt").write_text(cl_notes, encoding="utf-8")
        print(f"Wrote {TAB_DIR / f'{cbase_fn}_NOTES.txt'}")
        if args.write_tex:
            tex_c = TAB_DIR / f"{cbase_fn}.tex"
            write_latex_longtable(
                df_cl,
                "Summary statistics of portfolios formed from Clustering approach",
                tex_label_cl,
                tex_c,
            )
            print(f"Wrote {tex_c}")

    if not args.all_approaches:
        df_one = build_thesis_summary_table(
            args.approach,
            port_n=args.port_n,
            write_pick_tables=True,
            ap_root=cli_ap,
            tree_port_root=cli_tree,
        )
        tag_map = {"AP": "AP_trees", "RP": "RP_trees", "TV": "TV_trees"}
        tag_one = tag_map[args.approach]
        tag_one_fn = f"Thesis_Table36_{tag_one}{sfx}"
        csv_p = TAB_DIR / f"{tag_one_fn}.csv"
        write_csv(df_one, csv_p)
        print(f"Wrote {csv_p}")

        notes = TAB_DIR / f"{tag_one_fn}_NOTES.txt"
        tv_note = (
            "TV: run python run_all_tv_cross_sections.py; halflife TV_KERNEL_HALFLIFE_MONTHS (default 72).\n"
            if args.approach == "TV"
            else ""
        )
        notes.write_text(
            common_notes
            + tv_note
            + "Figure 7: Ward clustering is one horizontal reference (same OOS SR at every cross-section).\n",
            encoding="utf-8",
        )
        print(f"Wrote {notes}")

        if args.write_tex:
            tex = TAB_DIR / f"{tag_one_fn}.tex"
            write_latex_longtable(df_one, args.caption, f"tab:thesis36_{tag_one.lower()}", tex)
            print(f"Wrote {tex}")

    if args.figure7 and not args.figure7_multi:
        if args.all_approaches:
            df_plot = dfs_multi["AP"]
            tag_plot = "AP_trees"
            primary = "AP-Tree"
        else:
            df_plot = df_one
            tag_plot = tag_one
            primary = {"AP": "AP-Tree", "RP": "RP-Tree", "TV": "Time-varying (TV)"}[args.approach]
        assert df_plot is not None and tag_plot is not None
        figp = FIG_DIR / f"Thesis_Fig7_style_SR_{tag_plot}{sfx}.png"
        plot_figure7_style(
            df_plot,
            figp,
            benchmark_csv=args.figure7_benchmark,
            ap_root=cli_ap,
            port_n=args.port_n,
            include_clustering=not args.no_figure7_clustering,
            primary_label=primary,
        )
        print(f"Wrote {figp}")

    if args.figure7_multi:
        if args.all_approaches:
            df_ap = dfs_multi["AP"]
            df_rp = dfs_multi["RP"]
            df_tv = dfs_multi["TV"]
        else:
            wp = False
            print("Building AP/RP/TV tables for multi-figure (write_pick_tables=False for speed).")
            df_ap = build_thesis_summary_table(
                "AP",
                port_n=args.port_n,
                write_pick_tables=wp,
                ap_root=cli_ap,
                tree_port_root=cli_tree,
            )
            df_rp = build_thesis_summary_table(
                "RP",
                port_n=args.port_n,
                write_pick_tables=wp,
                ap_root=cli_ap,
                tree_port_root=cli_tree,
            )
            df_tv = build_thesis_summary_table(
                "TV",
                port_n=args.port_n,
                write_pick_tables=wp,
                ap_root=cli_ap,
                tree_port_root=cli_tree,
            )
        figp = FIG_DIR / f"Thesis_Fig7_multi_SR_AP_RP_TV_Ward{sfx}.png"
        plot_figure7_multi(
            df_ap,
            figp,
            df_rp=df_rp,
            df_tv=df_tv,
            ap_root=cli_ap,
            port_n=args.port_n,
            include_clustering=not args.no_figure7_clustering,
            benchmark_csv=args.figure7_benchmark,
        )
        print(f"Wrote {figp}")

    if args.figure_alpha_t:
        fc = args.alpha_factor
        tcol = "t_alpha_CAPM" if fc == "CAPM" else "t_alpha_FF5"
        print(
            "Building AP/RP/TV with alpha t-stats (pick_best + SDF ~ factors; may take several minutes)..."
        )
        df_ap_at = build_thesis_summary_table(
            "AP",
            port_n=args.port_n,
            write_pick_tables=True,
            ap_root=cli_ap,
            tree_port_root=cli_tree,
            include_factor_tstats=True,
        )
        df_rp_at = build_thesis_summary_table(
            "RP",
            port_n=args.port_n,
            write_pick_tables=True,
            ap_root=cli_ap,
            tree_port_root=cli_tree,
            include_factor_tstats=True,
        )
        df_tv_at = build_thesis_summary_table(
            "TV",
            port_n=args.port_n,
            write_pick_tables=True,
            ap_root=cli_ap,
            tree_port_root=cli_tree,
            include_factor_tstats=True,
        )
        for tag, df in [
            ("AP_trees", df_ap_at),
            ("RP_trees", df_rp_at),
            ("TV_trees", df_tv_at),
        ]:
            outp = TAB_DIR / f"Thesis_Table36_{tag}_with_alpha_t_{fc}{sfx}.csv"
            write_csv(df, outp)
            print(f"Wrote {outp}")
        figp_at = FIG_DIR / f"Thesis_Fig_alpha_t_{fc}_multi_AP_RP_TV_Ward{sfx}.png"
        plot_figure_alpha_t_multi(
            df_ap_at,
            figp_at,
            df_rp=df_rp_at,
            df_tv=df_tv_at,
            ap_root=cli_ap,
            port_n=args.port_n,
            include_clustering=not args.no_figure7_clustering,
            factor_spec=fc,
            t_column=tcol,
        )
        print(f"Wrote {figp_at}")


if __name__ == "__main__":
    _main_cli()
