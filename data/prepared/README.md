# Prepared panels (generated)

After a successful `prepare_data` step, this folder will contain:

- `panel_benchmark.parquet` — benchmark panel for tree portfolios
- `panel.parquet` — copy/sync used by tree scripts (created by the full pipeline)
- `panel_clustering_mice.parquet` — MICE-imputed panel for Ward clustering

**Do not commit** these files (they are gitignored). They are recreated when you run the pipeline.
