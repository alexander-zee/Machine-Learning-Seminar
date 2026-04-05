"""
Generate BPZ/SSRN-style figures *from pipeline outputs* (no manual CSV filling).

Writes PNGs to ``data/results/figures_seminar/``:

- ``BPZ_Fig3_cross_section_monthly_SR.png`` — AP-Trees (one point per available triplet)
  vs clustering (Ward) as a horizontal reference at the same test-window SR scale.
- ``BPZ_Fig4_heatmap_SR_AP_vs_Clustering.png`` — two heatmaps: rows = models (Ward + each
  tree triplet), columns = number of managed portfolios N (subset that exists in LASSO path).
- ``BPZ_Fig6_weights_and_FF5_alpha.png`` — bar chart of weights + FF5 alpha with OLS SE
  (tradable factors), for the first tree triplet found (configurable).

Also writes ``tables_seminar/BPZ_style_auto_summary.csv`` with the numbers used.

Call ``generate_bpz_style_figures()`` after Part 2 + pick_best, or it is invoked from
``run_seminar_outputs`` / ``run_complete_paper_outputs``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part_3_metrics_collection.pick_best_lambda import (
    AP_PRUNE_DEFAULT,
    CLUSTER_RETURNS,
    TREE_PORT_ROOT,
    pick_best_lambda,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "data" / "results" / "figures_seminar"
TAB_DIR = REPO_ROOT / "data" / "results" / "tables_seminar"

WARD_SUB = "Ward_clusters_10"
TREE_CSV_NAME = "level_all_excess_combined_filtered.csv"


def _discover_tree_subdirs(ap_root: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(ap_root.iterdir()):
        if not p.is_dir() or not p.name.startswith("LME_"):
            continue
        if (TREE_PORT_ROOT / p.name / TREE_CSV_NAME).is_file():
            out.append(p.name)
    return out


def _test_sr_at_fixed_lambda(
    ap_root: Path, sub_dir: str, i_star: int, j_star: int, port_n: int
) -> float | None:
    """test_SR at one LASSO path row with portsN == port_n (same λ cell for all N)."""
    p = ap_root / sub_dir / f"results_full_l0_{i_star}_l2_{j_star}.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    row = df.loc[df["portsN"] == port_n]
    if row.empty:
        return None
    return float(row.iloc[0]["test_SR"])


def generate_bpz_style_figures(
    ap_root: Path | None = None,
    port_n_pick: int = 10,
    n_assets_columns: tuple[int, ...] = (5, 10, 20, 40),
    fig6_subdir: str | None = None,
    fig6_port_n: int | None = None,
) -> dict:
    """
    Build figures from existing ``data/results/ap_pruning/*/`` CSVs.
    Uses ``pick_best_lambda(..., write_tables=False)`` to avoid overwriting exports.
    """
    ap = Path(ap_root) if ap_root is not None else AP_PRUNE_DEFAULT
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    out: dict = {"figures": [], "tables": [], "errors": []}

    tree_subs = _discover_tree_subdirs(ap)
    ward_ok = (ap / WARD_SUB).is_dir() and CLUSTER_RETURNS.is_file()

    # --- Picks (no file writes) ---
    tree_rows: list[tuple[str, dict]] = []
    for sub in tree_subs:
        try:
            r = pick_best_lambda(
                ap,
                sub,
                port_n_pick,
                TREE_PORT_ROOT / sub / TREE_CSV_NAME,
                returns_index_col=None,
                write_tables=False,
            )
            tree_rows.append((sub, r))
        except Exception as e:
            out["errors"].append(f"{sub}: pick_best {e}")

    ward_row: dict | None = None
    if ward_ok:
        try:
            ward_row = pick_best_lambda(
                ap,
                WARD_SUB,
                port_n_pick,
                CLUSTER_RETURNS,
                returns_index_col=0,
                write_tables=False,
            )
        except Exception as e:
            out["errors"].append(f"Ward: {e}")

    # --- Fig 3: cross-section index = triplet folder; AP line + Ward reference ---
    if tree_rows:
        fig, ax = plt.subplots(figsize=(11.0, 4.5))
        x = np.arange(len(tree_rows))
        labels = [t[0].replace("LME_", "").replace("_", "--") for t in tree_rows]
        y_ap = [float(t[1]["test_SR"]) for t in tree_rows]
        ax.plot(x, y_ap, color="#1f77b4", linewidth=1.4, marker="o", label="AP-Trees (test SR)")
        if ward_row is not None:
            y_w = float(ward_row["test_SR"])
            ax.axhline(y_w, color="#d62728", linestyle="--", linewidth=1.4, label="Clustering / Ward (test SR)")
        ax.axhline(0.0, color="k", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Monthly Sharpe ratio (SR)")
        ax.set_xlabel("Cross-section (one AP-tree triplet per folder under ap_pruning)")
        ax.set_title(
            "BPZ-style Fig. 3 analogue: out-of-sample test SR by triplet vs Ward clustering\n"
            "(add more triplets by running Part 2 for other characteristic pairs)"
        )
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        p3 = FIG_DIR / "BPZ_Fig3_cross_section_monthly_SR.png"
        fig.savefig(p3, dpi=200, bbox_inches="tight")
        plt.close(fig)
        out["figures"].append(p3)

    # --- Fig 4: left = AP triplets × N; right = Ward × N (same λ* cell per model) ---
    ap_rows: list[list[float | None]] = []
    labels_ap: list[str] = []
    for name, r in tree_rows:
        i_s, j_s = r["best_i_lambda0"], r["best_j_lambda2"]
        ap_rows.append(
            [_test_sr_at_fixed_lambda(ap, name, i_s, j_s, n) for n in n_assets_columns]
        )
        labels_ap.append(name.replace("LME_", "").replace("_", "-")[:40])

    ward_vec: list[float | None] | None = None
    if ward_row is not None:
        i_s, j_s = ward_row["best_i_lambda0"], ward_row["best_j_lambda2"]
        ward_vec = [
            _test_sr_at_fixed_lambda(ap, WARD_SUB, i_s, j_s, n) for n in n_assets_columns
        ]

    if ap_rows or ward_vec:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.5, max(3.5, 0.4 * max(len(labels_ap), 1) + 2.0)),
        )
        all_vals: list[float] = []
        Z_ap = np.asarray(
            [[np.nan if v is None else float(v) for v in row] for row in ap_rows],
            dtype=float,
        ) if ap_rows else np.zeros((0, len(n_assets_columns)))
        if Z_ap.size and np.isfinite(Z_ap).any():
            all_vals.extend(Z_ap[np.isfinite(Z_ap)].ravel().tolist())
        wrow = None
        if ward_vec is not None:
            wrow = np.asarray(
                [[np.nan if v is None else float(v) for v in ward_vec]], dtype=float
            )
            if np.isfinite(wrow).any():
                all_vals.extend(wrow[np.isfinite(wrow)].ravel().tolist())

        vmin, vmax = (-0.5, 0.5)
        if all_vals:
            vmin = float(min(vmin, min(all_vals)))
            vmax = float(max(vmax, max(all_vals)))
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax

        if Z_ap.size > 0:
            im0 = axes[0].imshow(
                Z_ap,
                aspect="auto",
                origin="lower",
                cmap="RdYlGn",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[0].set_title("AP-Trees — test SR at $\\lambda^*$ (same grid cell for all $N$)")
            axes[0].set_yticks(range(len(labels_ap)))
            axes[0].set_yticklabels(labels_ap, fontsize=7)
            axes[0].set_xticks(range(len(n_assets_columns)))
            axes[0].set_xticklabels([str(n) for n in n_assets_columns])
            axes[0].set_xlabel("Number of managed portfolios")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="SR")
        else:
            axes[0].text(0.5, 0.5, "No AP-tree triplets in ap_pruning", ha="center", va="center")

        if wrow is not None:
            im1 = axes[1].imshow(
                wrow,
                aspect="auto",
                origin="lower",
                cmap="RdYlGn",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[1].set_title("Ward clustering — test SR at $\\lambda^*$")
            axes[1].set_yticks([0])
            axes[1].set_yticklabels(["Ward"])
            axes[1].set_xticks(range(len(n_assets_columns)))
            axes[1].set_xticklabels([str(n) for n in n_assets_columns])
            axes[1].set_xlabel("Number of managed portfolios")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="SR")
        else:
            axes[1].text(0.5, 0.5, "No Ward output", ha="center", va="center")

        fig.suptitle(
            "BPZ-style Fig.~4 analogue: test SR vs $N$ at fixed $\\lambda^*$ (AP vs Ward)",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        p4 = FIG_DIR / "BPZ_Fig4_heatmap_SR_AP_vs_Clustering.png"
        fig.savefig(p4, dpi=200, bbox_inches="tight")
        plt.close(fig)
        out["figures"].append(p4)

    # --- Fig 6: weights + alpha for one tree triplet ---
    sub6 = fig6_subdir or (tree_rows[0][0] if tree_rows else None)
    pn6 = fig6_port_n if fig6_port_n is not None else port_n_pick
    if sub6 and (ap / sub6).is_dir():
        try:
            from part_3_metrics_collection.paper_style_outputs import (
                TRADABLE_FACTORS,
                build_sdf_series,
                regression_table_sdf,
            )

            ports_guess = ap / sub6 / f"Selected_Ports_{pn6}.csv"
            wg_guess = ap / sub6 / f"Selected_Ports_Weights_{pn6}.csv"
            write_w = not (ports_guess.is_file() and wg_guess.is_file())
            r6 = pick_best_lambda(
                ap,
                sub6,
                pn6,
                TREE_PORT_ROOT / sub6 / TREE_CSV_NAME,
                returns_index_col=None,
                write_tables=write_w,
            )
            w = np.asarray(r6["weights"], dtype=float)
            names = [str(x) for x in r6["selected_columns"]]

            ports_path = ap / sub6 / f"Selected_Ports_{r6['port_n']}.csv"
            wpath = ap / sub6 / f"Selected_Ports_Weights_{r6['port_n']}.csv"
            if ports_path.is_file() and wpath.is_file():
                sdf = build_sdf_series(ports_path, wpath, TRADABLE_FACTORS)
                reg = regression_table_sdf(sdf, TRADABLE_FACTORS)
                row_ff5 = reg.loc[reg["spec"] == "FF5"].iloc[0]
                alpha = float(row_ff5["alpha"])
                se_a = float(row_ff5["se"])

                fig, ax1 = plt.subplots(figsize=(10.0, 4.2))
                xb = np.arange(len(w))
                ax1.bar(xb, w, color="#c0c0c0", width=0.75, label="Weight")
                ax1.set_ylabel("Weight")
                ax1.set_xticks(xb)
                ax1.set_xticklabels(names, rotation=90, fontsize=6)
                ax2 = ax1.twinx()
                ax2.axhline(alpha, color="#d62728", linewidth=2.0, label=f"FF5 $\\alpha$ = {alpha:.4f}")
                ax2.fill_between(
                    [-0.5, len(w) - 0.5],
                    alpha - 1.96 * se_a,
                    alpha + 1.96 * se_a,
                    color="#d62728",
                    alpha=0.2,
                )
                ax2.set_ylabel("Alpha (FF5 tradable)")
                ax1.set_title(
                    f"BPZ-style Fig. 6 analogue: weights + FF5 alpha — {sub6}, N={r6['port_n']}"
                )
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
                fig.tight_layout()
                p6 = FIG_DIR / "BPZ_Fig6_weights_and_FF5_alpha.png"
                fig.savefig(p6, dpi=200, bbox_inches="tight")
                plt.close(fig)
                out["figures"].append(p6)
        except Exception as e:
            out["errors"].append(f"Fig6: {e}")

    # --- Summary CSV ---
    rows_out = []
    if ward_row:
        rows_out.append(
            {
                "model": WARD_SUB,
                "test_SR": ward_row["test_SR"],
                "valid_SR": ward_row["valid_SR"],
                "port_n": ward_row["port_n"],
            }
        )
    for sub, r in tree_rows:
        rows_out.append(
            {
                "model": sub,
                "test_SR": r["test_SR"],
                "valid_SR": r["valid_SR"],
                "port_n": r["port_n"],
            }
        )
    if rows_out:
        csv_path = TAB_DIR / "BPZ_style_auto_summary.csv"
        pd.DataFrame(rows_out).to_csv(csv_path, index=False)
        out["tables"].append(csv_path)

    return out


def _main() -> None:
    r = generate_bpz_style_figures()
    print("Figures:", *r.get("figures", []), sep="\n  ")
    print("Tables:", *r.get("tables", []), sep="\n  ")
    if r.get("errors"):
        print("Warnings:", *r["errors"], sep="\n  ")


if __name__ == "__main__":
    _main()
