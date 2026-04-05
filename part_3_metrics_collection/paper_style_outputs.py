"""
Paper-style figures and tables adapted for the seminar (Ward vs AP-trees, optional more models).

Uses outputs from Part 2 + pick_best_lambda:
  - data/results/ap_pruning/<subdir>/results_full_l0_*_l2_*.csv
  - train_SR_<n>.csv, valid_SR_<n>.csv, test_SR_<n>.csv (from pick_best_lambda)
  - Selected_Ports_<n>.csv, Selected_Ports_Weights_<n>.csv

Factor regressions: **same column definitions as** ``0_code/3_Metrics_Collection/SDF_TimeSeries_Regressions.R``
on ``paper_data/factor/tradable_factors.csv``. In BPZ replication code, labels ``FF3`` / ``FF5`` / ``FF11``
refer to **tradable characteristic-mimicking factors** (Mkt-RF + LME/BEME/OP/Investment/...), **not**
the classic Kenneth French library factors (SMB, HML, RMW, CMA). See ``write_metrics_notes``.

Output filenames include paper figure/table references (e.g. Fig10a–d, Fig7-style, Table3-style) for
side-by-side comparison with the PDF; values will still differ from the paper if sample or code paths differ.

Run from repo root:
  python -m part_3_metrics_collection.paper_style_outputs
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from part_3_metrics_collection.pick_best_lambda import (
        AP_PRUNE_DEFAULT,
        load_lambda_grid_meta,
        run_default_picks,
        run_full_paper_picks,
        print_ap_comparison,
    )
except ImportError:
    from pick_best_lambda import (  # type: ignore
        AP_PRUNE_DEFAULT,
        load_lambda_grid_meta,
        run_default_picks,
        run_full_paper_picks,
        print_ap_comparison,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "data" / "results" / "figures_seminar"
TABLES_DIR = REPO_ROOT / "data" / "results" / "tables_seminar"


def _resolve_tradable_factors_path() -> Path:
    """
    Prefer user-managed input under data/, with paper_data/ as backward-compatible fallback.
    """
    candidates = [
        REPO_ROOT / "data" / "factor" / "tradable_factors.csv",
        REPO_ROOT / "paper_data" / "factor" / "tradable_factors.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


TRADABLE_FACTORS = _resolve_tradable_factors_path()

# Matches AP_Pruning.py defaults
LAMBDA0_DEFAULT = np.array([0.0, 0.1, 0.2], dtype=float)
LAMBDA2_DEFAULT = np.array([0.01, 0.05, 0.1], dtype=float)

# R FF_regression uses rows 361:636 (inclusive) — 276 months aligned with paper code
R_REGRESSION_START_ROW = 360  # 0-based iloc start
R_REGRESSION_END_ROW = 636  # exclusive iloc end -> rows 360..635

# Paper figure mapping (Bryzgalova–Pelger–Zhu style; your plots are **analogues**, same layout)
# Fig.10: (a) validation SR vs N, (b) validation λ heatmap, (c) test SR vs N, (d) test λ heatmap
# Fig.7:  monthly SR across cross-sections (multiple triplets); we use a **bar** for one triplet only
# Fig.6:  SR + α + R² panels — we only automate SR lines via plot_cross_section_lines + template CSV
# Table 3 style: one row per model with SDF SR + factor regression α (tradable specs)


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_sr_matrices(
    ap_root: Path, sub_dir: str, port_n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = ap_root / sub_dir
    train = np.loadtxt(sub / f"train_SR_{port_n}.csv", delimiter=",")
    valid = np.loadtxt(sub / f"valid_SR_{port_n}.csv", delimiter=",")
    test = np.loadtxt(sub / f"test_SR_{port_n}.csv", delimiter=",")
    return train, valid, test


def _lambda_axis_ticklabels(x: np.ndarray) -> list[str]:
    out: list[str] = []
    for v in np.asarray(x, dtype=float).ravel():
        if v == 0.0:
            out.append("0")
        elif 0 < abs(v) < 0.001:
            out.append(f"{v:.1e}")
        else:
            out.append(f"{v:g}")
    return out


def plot_lambda_heatmaps(
    ap_root: Path,
    sub_dir: str,
    port_n: int,
    lambda0: np.ndarray | None = None,
    lambda2: np.ndarray | None = None,
    best_ij: tuple[int, int] | None = None,
    out_prefix: str | None = None,
    annotate_cells: bool | None = None,
) -> tuple[Path, Path]:
    """
    Figure 10 (b) and (d) analogues: validation and test Sharpe over (lambda0, lambda2).
    best_ij: (i,j) 1-based indices marking lambda* from validation (red dot).
    """
    lambda0 = LAMBDA0_DEFAULT if lambda0 is None else np.asarray(lambda0, dtype=float)
    lambda2 = LAMBDA2_DEFAULT if lambda2 is None else np.asarray(lambda2, dtype=float)
    n_cells = int(len(lambda0) * len(lambda2))
    if annotate_cells is None:
        annotate_cells = n_cells <= 49
    train, valid, test = load_sr_matrices(ap_root, sub_dir, port_n)
    _ensure_dirs()
    slug = sub_dir.replace(" ", "_")
    paths = []

    for name, Z, title_suffix, fig_panel in (
        ("valid", valid, "Validation", "b"),
        ("test", test, "Testing", "d"),
    ):
        # train_sr[i,j] is (lambda0_i, lambda2_j); imshow rows=y -> transpose so x=lambda0, y=lambda2
        Zt = np.asarray(Z).T
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        # Nearest-neighbor: one solid tile per (λ0, λ2) grid point (paper-style blocks, not smoothed blends).
        im = ax.imshow(
            Zt, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest"
        )
        ax.set_xticks(range(len(lambda0)))
        ax.set_xticklabels(_lambda_axis_ticklabels(lambda0), rotation=45, ha="right")
        ax.set_yticks(range(len(lambda2)))
        ax.set_yticklabels(_lambda_axis_ticklabels(lambda2))
        ax.set_xlabel(r"$\lambda_0$ (mean shrinkage)")
        ax.set_ylabel(r"$\lambda_2$ (variance shrinkage)")
        ax.set_title(
            f"Figure 10 ({fig_panel}): {sub_dir}\n{title_suffix} SR, {port_n} portfolios "
            f"(shrinkage grid; red dot = lambda* from validation)"
        )
        plt.colorbar(im, ax=ax, label="Monthly Sharpe ratio (SR)")
        if annotate_cells:
            zmin, zmax = float(np.nanmin(Zt)), float(np.nanmax(Zt))
            mid = 0.5 * (zmin + zmax)
            fs = max(5, min(8, int(90 // max(Zt.shape))))
            for yi in range(Zt.shape[0]):
                for xj in range(Zt.shape[1]):
                    val = Zt[yi, xj]
                    txt = ax.text(
                        xj,
                        yi,
                        f"{val:.4f}",
                        ha="center",
                        va="center",
                        fontsize=fs,
                        color="white" if val < mid else "black",
                    )
                    txt.set_path_effects([pe.withStroke(linewidth=2, foreground="0.2")])
        if best_ij is not None:
            bi, bj = best_ij[0] - 1, best_ij[1] - 1
            ax.scatter([bi], [bj], c="red", s=140, zorder=5, marker="o", edgecolors="k", linewidths=1.5)
        fig.tight_layout()
        fname = f"Fig10{fig_panel}_{slug}_port{port_n}_lambda_heatmap_{name}.png"
        p = FIGURES_DIR / (out_prefix + "_" + fname if out_prefix else fname)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths[0], paths[1]


def plot_sr_vs_n_portfolios(
    ap_root: Path,
    sub_dir: str,
    i_lambda0: int,
    j_lambda2: int,
    sample: str = "test",
    out_name: str | None = None,
    file_suffix: str = "",
) -> Path:
    """
    Figure 10 (a) [validation] and (c) [test] analogues: SR vs N at fixed (lambda0, lambda2).
    (Fig. 14 in the paper is a related but different comparison across basis types.)
    sample: 'train' | 'valid' | 'test' — valid uses CV file fold 3.
    """
    sub = ap_root / sub_dir
    fig_panel = "a" if sample == "valid" else "c" if sample == "test" else "x"
    if sample in ("train", "test"):
        path = sub / f"results_full_l0_{i_lambda0}_l2_{j_lambda2}.csv"
        col = "train_SR" if sample == "train" else "test_SR"
    else:
        path = sub / f"results_cv_3_l0_{i_lambda0}_l2_{j_lambda2}.csv"
        col = "valid_SR"
    df = pd.read_csv(path)
    if "portsN" not in df.columns or col not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {df.columns.tolist()}")
    d = df.sort_values("portsN")
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(d["portsN"], d[col], marker="o", linewidth=1.5, color="#2E6F9E")
    ax.set_xlabel("Number of portfolios")
    if sample == "valid":
        ax.set_ylabel("Validation SR")
    elif sample == "test":
        ax.set_ylabel("Testing SR")
    else:
        ax.set_ylabel("Train SR")
    slug = sub_dir.replace(" ", "_")
    ax.set_title(
        f"Figure 10 ({fig_panel}): {sub_dir}\nSR vs number of portfolios "
        f"(l0 index={i_lambda0}, l2 index={j_lambda2}; optimal shrinkage from validation)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    extra = f"_{file_suffix}" if file_suffix else ""
    name = (
        out_name
        if out_name
        else f"Fig10{fig_panel}_{slug}_portN_curve_{sample}_l0{i_lambda0}_l2{j_lambda2}{extra}.png"
    )
    p = FIGURES_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _model_bar_label(r: dict) -> str:
    sub = str(r.get("subdir", "?"))
    pn = r.get("port_n")
    return f"{sub}\nN={pn}" if pn is not None else sub


def plot_seminar_model_comparison(
    rows: list[dict],
    out_name: str = "Fig07_style_seminar_single_cross_section_TestSR_bar.png",
    sort_by_sr: bool = True,
) -> Path:
    """
    **Figure 7 style (reduced):** paper Fig. 7 plots SR across 36 cross-sections; with one triplet
    we use a bar chart of **test** SR (analogue: one point per model on the y-axis of Fig. 7).
    """
    _ensure_dirs()
    data = [(_model_bar_label(r), float(r["test_SR"])) for r in rows]
    if sort_by_sr:
        data.sort(key=lambda x: -x[1])
    labels, vals = zip(*data) if data else ([], [])
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 1.5), 5.0))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=["#c44e52", "#4c72b0", "#55a868", "#8172b3", "#8c8c8c"][: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Monthly Sharpe ratio (SR)")
    ax.set_title(
        "Figure 7 style (seminar): out-of-sample test SR by model\n"
        "(paper: lines over 36 cross-sections; here: one characteristic triplet, lambda* from validation)"
    )
    ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    p = FIGURES_DIR / out_name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_cross_section_lines(
    sr_wide: pd.DataFrame,
    sort_by: str,
    out_name: str = "Fig06_Fig07_style_cross_section_monthly_SR.png",
    ylabel: str = "Monthly Sharpe ratio (SR)",
) -> Path:
    """
    **Figure 6 / 7 style (panel a analogue):** index = cross-section ids (e.g. triplet codes),
    columns = model names; x-axis sorted by ``sort_by`` like the paper's AP-Tree ordering.
    """
    _ensure_dirs()
    order = sr_wide[sort_by].sort_values().index
    df = sr_wide.loc[order]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(df))
    markers = ["o", "s", "^", "D", "P", "X"]
    for i, col in enumerate(df.columns):
        ax.plot(
            x,
            df[col].values,
            marker=markers[i % len(markers)],
            label=col,
            linewidth=1.2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in df.index], rotation=90, fontsize=7)
    ax.set_xlabel("Cross-sections (sorted by ascending SR of: " + sort_by + ")")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(
        "Figure 6 / 7 style (panel a): monthly out-of-sample SR across cross-sections\n"
        "(fill template CSV; paper uses 36 triplets — add rows to replicate full figure shape)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / out_name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _parse_yyyymm_series(s: pd.Series) -> pd.Series:
    """Normalize index/date column to int yyyymm for merging."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.year * 100 + s.dt.month
    out = []
    for v in s:
        if pd.isna(v):
            out.append(np.nan)
            continue
        t = str(int(v)) if isinstance(v, (int, float)) and not pd.isna(v) else str(v).strip()
        t = t.replace("-", "")[:6]
        out.append(int(t[:6]))
    return pd.Series(out, index=s.index)


def _index_looks_like_calendar(idx: pd.Index) -> bool:
    if len(idx) == 0:
        return False
    v = idx[0]
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return True
    s = str(v).strip()
    if len(s) >= 6 and s[:4].isdigit():
        return True
    try:
        x = int(float(v))
        return x >= 180001 and x <= 210012
    except (TypeError, ValueError):
        return False


def build_sdf_series(
    selected_ports_path: Path,
    weights_path: Path,
    tradable_path: Path = TRADABLE_FACTORS,
) -> pd.DataFrame:
    """Weighted SDF return plus yyyymm. If row index is not a calendar, align to factor dates by row order."""
    ports = pd.read_csv(selected_ports_path, index_col=0)
    w = pd.read_csv(weights_path)
    if "weight" not in w.columns:
        raise ValueError(f"Expected 'weight' column in {weights_path}")
    weights = w["weight"].to_numpy(dtype=float)
    cols = ports.columns.tolist()
    if len(cols) != len(weights):
        raise ValueError(f"Column count {len(cols)} != weights {len(weights)}")
    r = ports.to_numpy(dtype=float) @ weights
    T = len(r)
    if _index_looks_like_calendar(ports.index):
        yyyymm = _parse_yyyymm_series(ports.index.to_series()).values
    else:
        fac = pd.read_csv(tradable_path).sort_values("Date")
        if len(fac) < T:
            raise ValueError(
                f"Factor file has {len(fac)} rows but Selected_Ports has {T}; "
                "cannot align by row order."
            )
        yyyymm = _parse_yyyymm_series(fac["Date"].iloc[:T]).values
    return pd.DataFrame({"yyyymm": yyyymm, "sdf": r})


# Same factor sets as SDF_TimeSeries_Regressions.R (tradable_factors.csv columns).
_FF3 = ["Mkt-RF", "LME", "BEME"]
_FF5 = ["Mkt-RF", "LME", "BEME", "OP", "Investment"]
_FF11 = [
    "Mkt-RF",
    "LME",
    "BEME",
    "OP",
    "Investment",
    "r12_2",
    "ST_REV",
    "LT_REV",
    "AC",
    "IdioVol",
    "Lturnover",
]


def _factor_block(m: pd.DataFrame, spec: str) -> tuple[pd.DataFrame, list[str]]:
    names = {"FF3": _FF3, "FF5": _FF5, "FF11": _FF11}[spec]
    missing = [c for c in names if c not in m.columns]
    if missing:
        raise KeyError(f"Merged data missing factor columns {missing}")
    return m[names].astype(float), names


def ols_intercept_t(y: np.ndarray, X: np.ndarray) -> tuple[float, float, float, float]:
    """Return alpha, se_alpha, t_stat, p_value (two-sided) for intercept in y ~ 1 + X."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X1 = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = n - X1.shape[1]
    if dof <= 0:
        return float(beta[0]), np.nan, np.nan, np.nan
    mse = float(np.sum(resid**2) / dof)
    try:
        cov = mse * np.linalg.inv(X1.T @ X1)
    except np.linalg.LinAlgError:
        return float(beta[0]), np.nan, np.nan, np.nan
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t0 = float(beta[0] / se[0]) if se[0] > 0 else np.nan
    p0 = float(2 * stats.t.sf(abs(t0), dof)) if np.isfinite(t0) else np.nan
    return float(beta[0]), float(se[0]), t0, p0


def regression_table_sdf(
    sdf_df: pd.DataFrame,
    tradable_path: Path = TRADABLE_FACTORS,
    row_start: int = R_REGRESSION_START_ROW,
    row_end: int = R_REGRESSION_END_ROW,
) -> pd.DataFrame:
    """
    Time-series OLS: sdf ~ 1 + tradable factors (same as R ``lm`` with intercept column in X).

    Labels FF3 / FF5 / FF11 refer to **columns in tradable_factors.csv** per
    ``SDF_TimeSeries_Regressions.R``, not to Kenneth French SMB/HML/RMW/CMA files.
    """
    fac = pd.read_csv(tradable_path)
    fac["yyyymm"] = _parse_yyyymm_series(fac["Date"])
    m = sdf_df.merge(fac, on="yyyymm", how="inner")
    m = m.iloc[row_start:row_end]
    y = m["sdf"].to_numpy(dtype=float)
    rows = []
    for spec in ("FF3", "FF5", "FF11"):
        Xblock, names = _factor_block(m, spec)
        X = Xblock.to_numpy(dtype=float)
        a, se, t, p = ols_intercept_t(y, X)
        rows.append(
            {
                "spec": spec,
                "n_obs": len(y),
                "alpha": a,
                "se": se,
                "t_stat": t,
                "p_value": p,
                "factors": "+".join(names),
            }
        )
    return pd.DataFrame(rows)


def table_model_summary(
    ap_root: Path,
    sub_dir: str,
    port_n: int,
    pick_row: dict,
    tradable_path: Path = TRADABLE_FACTORS,
) -> pd.DataFrame:
    """One row: Table 3 style — monthly SRs + intercept alpha (tradable factor specs)."""
    sub = ap_root / sub_dir
    sdf = build_sdf_series(
        sub / f"Selected_Ports_{port_n}.csv",
        sub / f"Selected_Ports_Weights_{port_n}.csv",
    )
    reg = regression_table_sdf(sdf, tradable_path)
    flat: dict = {
        "model": sub_dir,
        "port_n": int(port_n),
        "train_SR": pick_row["train_SR"],
        "valid_SR": pick_row["valid_SR"],
        "test_SR": pick_row["test_SR"],
    }
    for _, r in reg.iterrows():
        spec = r["spec"]
        flat[f"alpha_{spec}_tradable"] = r["alpha"]
        flat[f"t_{spec}_tradable"] = r["t_stat"]
    return pd.DataFrame([flat])


def write_table3_style_notes(suffix: str, csv_path: Path) -> Path:
    """Plain-text companion so side-by-side comparison with the PDF does not confuse factor names."""
    safe = suffix.replace(" ", "_").replace("/", "-")
    notes = TABLES_DIR / f"Table3_style_NOTES_{safe}.txt"
    text = f"""Table 3 style (seminar) — companion to: {csv_path.name}
================================================================================
Paper reference: Table 3 (and related) in Bryzgalova, Pelger, Zhu — compare LAYOUT only;
your numeric values will differ if sample, Python LARS, MICE, or Ward steps differ.

SDF monthly returns: weighted sum of Selected_Ports with Selected_Ports_Weights (lambda* from validation).

Columns alpha_FF3_tradable, alpha_FF5_tradable, alpha_FF11_tradable:
  Intercept from OLS:  SDF_t = alpha + beta' f_t + eps_t
  on the merged monthly panel, rows 361-636 (R indexing) = iloc[360:636] in Python,
  same window as SDF_TimeSeries_Regressions.R FF_regression().

IMPORTANT — these are NOT classic Fama-French factors from Kenneth French's website.
They are the paper's TRADABLE long-short factors in paper_data/factor/tradable_factors.csv,
exactly as wired in 0_code/3_Metrics_Collection/SDF_TimeSeries_Regressions.R:

  FF3_tradable:  Mkt-RF, LME, BEME
  FF5_tradable:  Mkt-RF, LME, BEME, OP, Investment
  FF11_tradable: Mkt-RF, LME, BEME, OP, Investment, r12_2, ST_REV, LT_REV, AC, IdioVol, Lturnover

t_*_tradable: heteroskedasticity-agnostic OLS t-ratio on alpha (matches R lm summary on intercept).

Standard errors: OLS with homoskedastic covariance (same as base R lm); the paper may use
HAC in some tables — check their appendix; upgrade here with statsmodels if you need Newey-West.
"""
    notes.write_text(text, encoding="utf-8")
    return notes


def run_seminar_outputs(
    port_n: int = 10,
    ap_root: Path | None = None,
    skip_pick: bool = False,
    picked_rows: list[dict] | None = None,
) -> dict:
    """
    Generate heatmaps, SR-vs-N, comparison bar, and regression summary CSV.
    If skip_pick and picked_rows is set, uses those dicts (must include best_i_lambda0, etc.).
    """
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    _ensure_dirs()

    if picked_rows is None and not skip_pick:
        picked_rows = run_default_picks(port_n=port_n, ap_root=ap)

    if picked_rows is None:
        picked_rows = []

    out: dict = {"figures": [], "tables": []}

    for row in picked_rows:
        sub = row["subdir"]
        pn = int(row.get("port_n", port_n))
        bij = (row["best_i_lambda0"], row["best_j_lambda2"])
        l0, l2 = row.get("lambda0"), row.get("lambda2")
        if (
            isinstance(l0, list)
            and isinstance(l2, list)
            and len(l0) > 0
            and len(l2) > 0
        ):
            la0, la2 = np.asarray(l0, dtype=float), np.asarray(l2, dtype=float)
        else:
            la0, la2 = None, None
        if la0 is None or la2 is None:
            meta = load_lambda_grid_meta(ap / sub)
            if meta is not None:
                la0, la2 = meta
        out["figures"].extend(
            plot_lambda_heatmaps(ap, sub, pn, lambda0=la0, lambda2=la2, best_ij=bij)
        )
        sfx = f"pickN{pn}"
        out["figures"].append(
            plot_sr_vs_n_portfolios(
                ap,
                sub,
                row["best_i_lambda0"],
                row["best_j_lambda2"],
                sample="valid",
                file_suffix=sfx,
            )
        )
        out["figures"].append(
            plot_sr_vs_n_portfolios(
                ap,
                sub,
                row["best_i_lambda0"],
                row["best_j_lambda2"],
                sample="test",
                file_suffix=sfx,
            )
        )

    if len(picked_rows) >= 1:
        out["figures"].append(plot_seminar_model_comparison(picked_rows))

    summary_parts = []
    for row in picked_rows:
        pn = int(row.get("port_n", port_n))
        try:
            summary_parts.append(table_model_summary(ap, row["subdir"], pn, row))
        except FileNotFoundError as e:
            print(f"Skip table for {row['subdir']} N={pn}: {e}")
        except Exception as e:
            print(f"Skip table for {row['subdir']} N={pn}: {e}")
    if summary_parts:
        full = pd.concat(summary_parts, ignore_index=True)
        if len(picked_rows) > 1:
            tag = "full_paper_bundle"
        else:
            tag = f"port{int(picked_rows[0].get('port_n', port_n))}"
        tpath = TABLES_DIR / f"Table3_style_SDF_and_tradable_FF_alpha_{tag}.csv"
        full.to_csv(tpath, index=False)
        out["tables"].append(tpath)
        npath = write_table3_style_notes(tag, tpath)
        out["tables"].append(npath)

    try:
        from part_3_metrics_collection.paper_style_bpz_figures import (
            generate_bpz_style_figures,
        )

        extra = generate_bpz_style_figures(ap, port_n_pick=port_n)
        out["figures"].extend(extra.get("figures", []))
        out["tables"].extend(extra.get("tables", []))
        if extra.get("errors"):
            for msg in extra["errors"]:
                print(f"BPZ-style figures note: {msg}")
    except Exception as e:
        print(f"BPZ-style auto figures skipped: {e}")

    return out


def run_complete_paper_outputs(ap_root: Path | None = None) -> dict:
    """
    Full pick set (Ward N=10, AP-trees N=10 and N=40) + all Fig.10-style panels per pick +
    Fig.7-style bar + Table 3 CSV. Call after Part 2 has finished (run_pick_best can be off).

    Set environment variable ``FULL_PAPER_ALL_TRIPLETS=1`` to generate picks and figures for
    every computed ``LME_*`` cross-section (large output); default is OP × Investment only.
    """
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    all_tr = os.environ.get("FULL_PAPER_ALL_TRIPLETS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    print("--- Full paper picks (Ward + trees N=10,40) ---")
    if all_tr:
        print("  FULL_PAPER_ALL_TRIPLETS=1 — including every LME_* tree cross-section")
    rows = run_full_paper_picks(ap_root=ap, all_tree_triplets=all_tr)
    print_ap_comparison(rows)
    return run_seminar_outputs(skip_pick=True, picked_rows=rows, port_n=10)


def demo_cross_section_template(out_csv: Path | None = None) -> Path:
    """
    Template for Figure 6 / 7 (panel a): index = cross-section label (paper uses numeric IDs).
    """
    _ensure_dirs()
    p = out_csv or TABLES_DIR / "Fig06_Fig07_style_cross_section_SR_input_template.csv"
    pd.DataFrame(
        {
            "cross_section_index": [30],
            "AP_tree_test_SR": [0.3344],
            "Ward_clusters_10_test_SR": [0.1491],
        }
    ).to_csv(p, index=False)
    print(f"Wrote template: {p} - add rows (one per triplet), then plot_cross_section_lines(df.set_index('cross_section_index').dropna(axis=1), sort_by=...)")
    return p


if __name__ == "__main__":
    _ensure_dirs()
    print("Running pick_best + seminar figures/tables...")
    res = run_seminar_outputs(port_n=10)
    print("Figures:", *res["figures"], sep="\n  ")
    print("Tables:", *res["tables"], sep="\n  ")
    demo_cross_section_template()
