# Required factor input

Place this file here before generating Table 3 style outputs:

- `tradable_factors.csv`

Used by `part_3_metrics_collection/paper_style_outputs.py` for SDF factor regressions.

`run_full_research_pipeline.py` exits early if this file is missing (same check as `python scripts/check_inputs.py`). For Part 3 figures only, you could still run heatmaps if you had AP outputs, but the full pipeline expects this file.
