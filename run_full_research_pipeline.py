#!/usr/bin/env python3
"""
Complete research pipeline for paper-style outputs (slow, no shortcuts).

Runs, in order:
  1. prepare_data          -> data/prepared/panel_benchmark.parquet
  2. Sync panel.parquet     -> required by step2_tree_portfolios (copy from benchmark)
  3. impute_characteristics (MICE) -> panel_clustering_mice.parquet
  4. create_tree_portfolio  -> 81 tree CSVs under data/results/tree_portfolios/LME_OP_Investment/
  5. create_cluster_portfolios -> cluster_returns.csv (Ward)
  6. combine_trees + filter_tree_ports -> filtered excess returns for AP-trees
  7. Part 2 AP pruning      -> Ward + LME_OP_Investment LASSO grids (tree step is very slow)
  8. Full Part 3 picks      -> Ward N=10, trees N=10 and N=40 (lambda* each)
  9. paper_style_outputs    -> Fig.10a-d per pick, Fig.7-style bar, Table 3 CSV + NOTES

From repository root:
  python run_full_research_pipeline.py

Rough runtime (order of magnitude; depends on CPU/RAM):
  - prepare_data:        minutes (large CSV)
  - MICE:                ~3 min (636 months)
  - Ward clusters:       ~7–10 min
  - 81 benchmark trees:  often 30 min – several hours
  - Part 2 Ward AP:      ~1–5 min
  - Part 2 tree AP:      often many hours (~1500 cols × λ grid; full pipeline uses AP_PRUNE_LAMBDA_GRID=paper ≈ 9×10 grid)
  - Part 3 + figures:    ~1–2 min

If panel_benchmark.parquet already exists and you do not have FINALdataset.csv in-repo:
  set SKIP_PREPARE_DATA=1  (skips prepare_data only; you still need MICE inputs etc.)

NOT included (add manually if your paper requires them):
  - Loop over all 36 characteristic triplets (Fig. 6/7 across cross-sections)
  - AP-trees without interaction nodes (Fig. 8 ablation)
  - Rolling / kernel moments (Fig. 13, colleagues' extension)
  - ML baselines (Fig. 9)
  - Newey-West / HAC standard errors (optional upgrade to regression helper)
  - Cross-sectional R² panels (Fig. 6b/c) — needs extra code on top of factor regressions
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _chdir_repo() -> None:
    if Path.cwd().resolve() != REPO:
        os.chdir(REPO)
        print(f"Working directory set to: {REPO}\n")


def sync_panel_for_trees() -> None:
    """step2_tree_portfolios reads data/prepared/panel.parquet; prepare_data writes panel_benchmark."""
    bench = REPO / "data" / "prepared" / "panel_benchmark.parquet"
    panel = REPO / "data" / "prepared" / "panel.parquet"
    if not bench.is_file():
        print("WARNING: panel_benchmark.parquet not found — prepare_data may have failed.")
        return
    if not panel.is_file() or bench.stat().st_mtime > panel.stat().st_mtime:
        shutil.copy2(bench, panel)
        print("Synced data/prepared/panel.parquet <- panel_benchmark.parquet (for tree build).")


def main() -> None:
    _chdir_repo()
    print("=" * 72)
    print("FULL RESEARCH PIPELINE — expect long runtime (especially Part 2 AP-trees).")
    print("=" * 72)

    from part_1_portfolio_creation.tree_portfolio_creation.step1_prepare_data import (
        OUTPUT_PATH as PANEL_BENCHMARK_PATH,
        RAW_PATH,
        prepare_data,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step1b_impute_data import (
        impute_characteristics,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import (
        create_tree_portfolio,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step2_cluster_portfolios import (
        create_cluster_portfolios,
    )
    from part_1_portfolio_creation.tree_portfolio_creation.step3_combine_trees import combine_trees
    from part_1_portfolio_creation.tree_portfolio_creation.step4_filter_portfolios import (
        filter_tree_ports,
    )

    skip_prep = os.environ.get("SKIP_PREPARE_DATA", "").lower() in ("1", "true", "yes")
    print("\n--- Step 1: prepare_data (benchmark panel) ---")
    if skip_prep and PANEL_BENCHMARK_PATH.is_file():
        print(f"SKIP_PREPARE_DATA=1 and found {PANEL_BENCHMARK_PATH.name} — skipping prepare_data.")
    elif not RAW_PATH.is_file():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(
            "\nMissing raw CSV. Created folder (if needed):\n"
            f"  {RAW_PATH.parent}\n"
            "Put your file here as:\n"
            f"  {RAW_PATH.name}\n"
            f"full path: {RAW_PATH}\n\n"
            "Or change RAW_PATH in part_1_portfolio_creation/tree_portfolio_creation/step1_prepare_data.py.\n\n"
            "If you already ran prepare_data before and have:\n"
            f"  {PANEL_BENCHMARK_PATH}\n"
            "run with:\n"
            "  $env:SKIP_PREPARE_DATA='1'; python run_full_research_pipeline.py\n"
        )
        sys.exit(1)
    else:
        prepare_data()
    sync_panel_for_trees()

    print("\n--- Step 1b: MICE imputation ---")
    impute_characteristics()

    print("\n--- Step 2a: Benchmark AP-trees (81 trees, OP x Investment) ---")
    create_tree_portfolio(
        feat1="OP",
        feat2="Investment",
        output_path=Path("data/results/tree_portfolios"),
    )

    print("\n--- Step 2b: Ward cluster portfolios ---")
    create_cluster_portfolios()

    print("\n--- Step 2c: Combine + filter trees ---")
    combine_trees(feat1="OP", feat2="Investment")
    filter_tree_ports(feat1="OP", feat2="Investment")

    print("\n--- Part 2: AP LASSO grids (clusters + trees) ---")
    # Dense λ grid (paper-style heatmaps); override with AP_PRUNE_LAMBDA_GRID=fast or paper_full
    os.environ.setdefault("AP_PRUNE_LAMBDA_GRID", "paper")
    import importlib.util

    p2 = REPO / "part_2_AP_pruning" / "run_part2.py"
    spec = importlib.util.spec_from_file_location("seminar_run_part2", p2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_part2(run_trees=True, run_clusters=True, run_pick_best=False)

    print("\n--- Part 3 + figures/tables (full paper bundle) ---")
    from part_3_metrics_collection.paper_style_outputs import run_complete_paper_outputs

    run_complete_paper_outputs()
    from part_3_metrics_collection.paper_style_outputs import demo_cross_section_template

    demo_cross_section_template()

    print("\n" + "=" * 72)
    print("DONE.")
    print("  Figures: data/results/figures_seminar/")
    print("  Tables:  data/results/tables_seminar/")
    print("See module docstring above for what is still manual for a thesis-length paper.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
