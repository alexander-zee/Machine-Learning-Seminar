"""
Tree portfolios (step 2).

``split_mode="ntile"`` — R-style equal-count splits (default, matches seminar baseline).

``split_mode="opt_quantile"`` — extension: at each node, choose the quantile cut in
``OPT_QUANTILE_GRID`` that maximizes the sum of |VW(ret)_low − VW(ret)_high| over **prior**
calendar months only, with ``OPT_MIN_LEAF`` minimum names per child; month 0 uses ntile;
fallback to ntile when scores tie at zero or *n* is too small. Outputs go to
``OUTPUT_PATH_OPT_QUANT`` unless ``output_path`` is set.

**Faster runs (still causal / forgivable approximations):**

- **Parallel 81 trees** — for ``ntile`` and ``opt_quantile``, uses ``min(cpu_count-1, 16)`` processes by default
  when ``SEMINAR_TREE_BUILD_WORKERS`` is unset. Set ``SEMINAR_TREE_BUILD_WORKERS=1`` for sequential; or set an
  explicit integer cap. Same numerics as sequential; document worker count for thesis wall-time comparisons.

- **Rolling prior window / stride** — ``SEMINAR_OPTQUANT_PRIOR_MAX_MONTHS`` (e.g. 120),
  ``SEMINAR_OPTQUANT_PRIOR_STRIDE`` (e.g. 2). See ``_opt_prior_score_limits()``.

- **Opt only at top splits** — ``SEMINAR_OPTQUANT_MAX_SPLIT_DEPTH=2`` uses optimal quantiles
  only for ``split_1``–``split_2``; deeper levels use ``ntile`` (large speedup; say so in text).

- **Narrower** ``OPT_QUANTILE_GRID`` (e.g. three points) in this file.

**Example “thesis speed” preset (often ~10–30× vs naive full-history, all-depth opt, serial):**

``SEMINAR_TREE_BUILD_WORKERS`` unset (auto parallel), ``SEMINAR_OPTQUANT_PRIOR_MAX_MONTHS=120``,
``SEMINAR_OPTQUANT_PRIOR_STRIDE=2``, ``SEMINAR_OPTQUANT_MAX_SPLIT_DEPTH=2``.
"""

import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from typing import Literal


# ── Config ────────────────────────────────────────────────────────────────────
PANEL_PATH  = Path('data/prepared/panel.parquet')
OUTPUT_PATH = Path('data/results/tree_portfolios')
OUTPUT_PATH_OPT_QUANT = Path('data/results/tree_portfolios_optquantile')
TREE_DEPTH  = 4
Q_NUM       = 2        # binary median splits
Y_MIN       = 1964
Y_MAX       = 2016

# Opt-quantile extension (causal scoring on past months only)
OPT_QUANTILE_GRID = np.array([0.40, 0.45, 0.50, 0.55, 0.60], dtype=float)
OPT_MIN_LEAF = 30


def _opt_prior_score_limits() -> tuple[int | None, int]:
    """
    Speed knobs (still causal: only s < current month).

    SEMINAR_OPTQUANT_PRIOR_MAX_MONTHS — if set to a positive int, score each split using
    only the last N prior calendar months (rolling window). Default: use full history.

    SEMINAR_OPTQUANT_PRIOR_STRIDE — use every k-th prior month in that window (default 1).
    Stride 2 ≈ halves prior-month work; document as a computational approximation.

    Example (often ~5–15× faster on late sample dates, depending on N and stride)::

        set SEMINAR_OPTQUANT_PRIOR_MAX_MONTHS=240
        set SEMINAR_OPTQUANT_PRIOR_STRIDE=2
    """
    raw_m = os.environ.get("SEMINAR_OPTQUANT_PRIOR_MAX_MONTHS", "").strip()
    max_months: int | None = None
    if raw_m.isdigit():
        v = int(raw_m)
        if v > 0:
            max_months = v
    raw_s = os.environ.get("SEMINAR_OPTQUANT_PRIOR_STRIDE", "1").strip()
    stride = int(raw_s) if raw_s.isdigit() else 1
    stride = max(1, stride)
    return max_months, stride


def _tree_build_print_year(y: int) -> None:
    """Skip in ProcessPool workers so parallel runs do not interleave Year lines."""
    if y % 5 != 0:
        return
    try:
        if mp.parent_process() is not None:
            return
    except Exception:
        pass
    print(f"    Year {y}")


def _opt_max_split_depth() -> int | None:
    """
    If set, only ``split_1`` … ``split_k`` use optimal quantiles; deeper splits use ``ntile``.

    SEMINAR_OPTQUANT_MAX_SPLIT_DEPTH=k (e.g. 2): much cheaper; document as “extension at top
    of tree only.” Unset = all depths use opt (slowest, original spec).
    """
    raw = os.environ.get("SEMINAR_OPTQUANT_MAX_SPLIT_DEPTH", "").strip()
    if raw.isdigit():
        v = int(raw)
        if v >= 1:
            return v
    return None


# Set by ProcessPool initializer (one copy of panel per worker process).
_MP_PANEL: pd.DataFrame | None = None


def _mp_init_worker(panel: pd.DataFrame) -> None:
    global _MP_PANEL
    _MP_PANEL = panel


def _mp_run_single_tree(task: tuple) -> tuple[int, tuple, tuple]:
    """Worker: one of 81 trees. Returns (k, combo, (ret_table, feat_min_tables, feat_max_tables))."""
    global _MP_PANEL
    if _MP_PANEL is None:
        raise RuntimeError("multiprocessing tree worker: panel not initialized")
    (
        k,
        combo,
        feats,
        tree_depth,
        q_num,
        y_min,
        y_max,
        split_mode,
        min_leaf,
        qg_list,
    ) = task
    qg = np.asarray(qg_list, dtype=float) if qg_list is not None else None
    feat_list = [feats[i] for i in combo]
    out = compute_one_tree(
        _MP_PANEL,
        feat_list,
        feats,
        tree_depth,
        q_num,
        y_min,
        y_max,
        split_mode=split_mode,
        quantile_grid=qg,
        min_leaf=min_leaf,
    )
    return k, combo, out


SplitMode = Literal["ntile", "opt_quantile"]

# Node names for depth=4, matches R's cnames exactly
# Encoding: first digit always 1 (root), subsequent digits 1=low 2=high
CNAMES = [
    '1',
    '11', '12',
    '111', '112', '121', '122',
    '1111', '1112', '1121', '1122', '1211', '1212', '1221', '1222',
    '11111', '11112', '11121', '11122', '11211', '11212', '11221', '11222',
    '12111', '12112', '12121', '12122', '12211', '12212', '12221', '12222',
]
# Sanity check: 1 + 2 + 4 + 8 + 16 = 31 nodes
assert len(CNAMES) == 31


# ── Data loading ──────────────────────────────────────────────────────────────

def load_triplet(feat1: str, feat2: str) -> pd.DataFrame:
    """
    Load panel and filter to complete cases for this triplet.
    Equivalent to R's intersect operation per (year, month).

    Parameters
    ----------
    feat1, feat2 : characteristic column names (already quantile-ranked in parquet)

    Returns
    -------
    DataFrame with columns: permno, date, yy, mm, ret, size, LME, feat1, feat2
    NaN rows for any of the three characteristics are dropped.
    """
    df = pd.read_parquet(
        PANEL_PATH,
        columns=['permno', 'date', 'yy', 'mm', 'ret', 'size', 'LME', feat1, feat2]
    )
    before = len(df)
    df = df.dropna(subset=['LME', feat1, feat2, 'ret', 'size'])
    after  = len(df)
    print(f"  Triplet (LME, {feat1}, {feat2}): {after:,} rows kept ({before-after:,} dropped)")
    return df.reset_index(drop=True)


# ── Node assignment for one month ─────────────────────────────────────────────

def assign_nodes_month(df_month: pd.DataFrame,
                        feat_list: list,
                        tree_depth: int,
                        q_num: int) -> pd.DataFrame:
    """Assign splits and compute port indices for one month."""
    df = df_month.copy()
    # Initialize split and port columns
    for k in range(1, tree_depth + 1):
        df[f'split_{k}'] = 0
    for i in range(0, tree_depth + 1):
        df[f'port{i}'] = 1

    if len(df) == 0:
        return df

    # Depth‑1 split
    df['split_1'] = _ntile(df[feat_list[0]], q_num)

    # Recursively assign deeper splits, bin by bin
    for bin_val in range(1, q_num + 1):
        mask = df['split_1'] == bin_val
        sub = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        sub = _recurse_subset(sub, feat_list, depth=1,
                              tree_depth=tree_depth, q_num=q_num)
        # Copy back split columns for depths > 1
        for k in range(2, tree_depth + 1):
            df.loc[mask, f'split_{k}'] = sub[f'split_{k}'].values

    # Compute port indices
    for i in range(1, tree_depth + 1):
        port = np.ones(len(df), dtype=int)
        for k in range(1, i + 1):
            split_vals = df[f'split_{k}'].to_numpy(dtype=int)
            port += (split_vals - 1) * (q_num ** (i - k))
        df[f'port{i}'] = port

    return df

def _recurse_subset(df: pd.DataFrame,
                    feat_list: list,
                    depth: int,
                    tree_depth: int,
                    q_num: int) -> pd.DataFrame:
    """Recursively assign splits for a subset; returns updated subset."""
    if depth >= tree_depth:
        return df
    next_depth = depth + 1
    # Assign split at next_depth
    df[f'split_{next_depth}'] = _ntile(df[feat_list[next_depth - 1]], q_num)
    # Recurse into each bin
    for bin_val in range(1, q_num + 1):
        mask = df[f'split_{next_depth}'] == bin_val
        sub = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        sub = _recurse_subset(sub, feat_list, next_depth,
                              tree_depth, q_num)
        # Copy back deeper splits
        for k in range(next_depth + 1, tree_depth + 1):
            df.loc[mask, f'split_{k}'] = sub[f'split_{k}'].values
    return df

def _ntile(series: pd.Series, q: int) -> pd.Series:
    """
    R's ntile: splits into q roughly equal groups,
    even when values are tied.
    """
    n = len(series)
    # Stable sort order (preserves original order for ties)
    order = np.argsort(series.to_numpy(), kind='stable')
    
    # Calculate bucket sizes
    size = n // q
    remainder = n % q
    sizes = [size + 1] * remainder + [size] * (q - remainder)
    
    # Assign bucket numbers to sorted positions
    bucket = np.empty(n, dtype=int)
    pos = 0
    for i, sz in enumerate(sizes):
        bucket[order[pos:pos+sz]] = i + 1
        pos += sz
    
    return pd.Series(bucket, index=series.index)


def _k_left_quantile(n: int, q: float, min_leaf: int) -> int | None:
    if n < 2 * min_leaf:
        return None
    k = int(np.floor(float(q) * n))
    k = max(min_leaf, min(n - min_leaf, k))
    if k < min_leaf or n - k < min_leaf:
        return None
    return k


def _labels_from_quantile_k(series: pd.Series, k: int) -> pd.Series:
    n = len(series)
    order = np.argsort(series.to_numpy(), kind="stable")
    lab = np.ones(n, dtype=int)
    lab[order[k:]] = 2
    lab[order[:k]] = 1
    return pd.Series(lab, index=series.index)


def _vw_abs_spread_split(sub: pd.DataFrame, feat_col: str, q: float, min_leaf: int) -> float:
    n = len(sub)
    k = _k_left_quantile(n, q, min_leaf)
    if k is None:
        return 0.0
    order = np.argsort(sub[feat_col].to_numpy(), kind="stable")
    w = sub["size"].to_numpy(dtype=float)
    r = sub["ret"].to_numpy(dtype=float)
    il, ir = order[:k], order[k:]
    wl, wr = w[il].sum(), w[ir].sum()
    if wl <= 0.0 or wr <= 0.0:
        return 0.0
    rl = (w[il] * r[il]).sum() / wl
    rr = (w[ir] * r[ir]).sum() / wr
    return float(abs(rl - rr))


def _mask_path_splits(df: pd.DataFrame, path_bits: tuple[int, ...]) -> np.ndarray:
    if not path_bits:
        return np.ones(len(df), dtype=bool)
    m = np.ones(len(df), dtype=bool)
    for i, b in enumerate(path_bits):
        m &= df[f"split_{i + 1}"].to_numpy() == b
    return m


def _accumulate_score_prior(
    prior_by_ti: dict[int, pd.DataFrame],
    t_idx: int,
    path_bits: tuple[int, ...],
    feat_col: str,
    q_frac: float,
    min_leaf: int,
) -> float:
    total = 0.0
    max_m, stride = _opt_prior_score_limits()
    start_s = 0 if max_m is None else max(0, t_idx - max_m)
    for s in range(start_s, t_idx, stride):
        p = prior_by_ti.get(s)
        if p is None or len(p) == 0:
            continue
        sub = p.loc[_mask_path_splits(p, path_bits)]
        if len(sub) < 2 * min_leaf:
            continue
        total += _vw_abs_spread_split(sub, feat_col, q_frac, min_leaf)
    return total


def _choose_opt_split_labels(
    series: pd.Series,
    prior_by_ti: dict[int, pd.DataFrame],
    t_idx: int,
    path_bits: tuple[int, ...],
    feat_col: str,
    quantile_grid: np.ndarray,
    min_leaf: int,
    q_num: int,
) -> pd.Series:
    n = len(series)
    if n < 2 * min_leaf:
        return _ntile(series, q_num)
    best_sc = -1.0
    best_q = 0.5
    for qf in quantile_grid:
        sc = _accumulate_score_prior(
            prior_by_ti, t_idx, path_bits, feat_col, float(qf), min_leaf
        )
        if sc > best_sc or (
            np.isclose(sc, best_sc)
            and sc >= 0.0
            and abs(float(qf) - 0.5) < abs(best_q - 0.5)
        ):
            best_sc = sc
            best_q = float(qf)
    if best_sc <= 0.0:
        return _ntile(series, q_num)
    k = _k_left_quantile(n, best_q, min_leaf)
    if k is None:
        return _ntile(series, q_num)
    return _labels_from_quantile_k(series, k)


def _opt_recurse_subset(
    df: pd.DataFrame,
    feat_list: list,
    depth: int,
    tree_depth: int,
    q_num: int,
    prior_by_ti: dict[int, pd.DataFrame],
    t_idx: int,
    path_bits: tuple[int, ...],
    quantile_grid: np.ndarray,
    min_leaf: int,
) -> pd.DataFrame:
    if depth >= tree_depth:
        return df
    next_depth = depth + 1
    feat_col = feat_list[next_depth - 1]
    md = _opt_max_split_depth()
    use_opt = t_idx > 0 and (md is None or next_depth <= md)
    if use_opt:
        df[f"split_{next_depth}"] = _choose_opt_split_labels(
            df[feat_col],
            prior_by_ti,
            t_idx,
            path_bits,
            feat_col,
            quantile_grid,
            min_leaf,
            q_num,
        ).to_numpy()
    else:
        df[f"split_{next_depth}"] = _ntile(df[feat_col], q_num).to_numpy()
    for bin_val in range(1, q_num + 1):
        mask = df[f"split_{next_depth}"] == bin_val
        sub = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        sub = _opt_recurse_subset(
            sub,
            feat_list,
            next_depth,
            tree_depth,
            q_num,
            prior_by_ti,
            t_idx,
            path_bits + (bin_val,),
            quantile_grid,
            min_leaf,
        )
        for k in range(next_depth + 1, tree_depth + 1):
            df.loc[mask, f"split_{k}"] = sub[f"split_{k}"].values
    return df


def assign_nodes_month_opt(
    df_month: pd.DataFrame,
    feat_list: list,
    tree_depth: int,
    q_num: int,
    prior_by_ti: dict[int, pd.DataFrame],
    t_idx: int,
    quantile_grid: np.ndarray | None = None,
    min_leaf: int = OPT_MIN_LEAF,
) -> pd.DataFrame:
    qg = OPT_QUANTILE_GRID if quantile_grid is None else np.asarray(quantile_grid, dtype=float)
    df = df_month.copy()
    for k in range(1, tree_depth + 1):
        df[f"split_{k}"] = 0
    for i in range(0, tree_depth + 1):
        df[f"port{i}"] = 1
    if len(df) == 0:
        return df
    if t_idx == 0:
        return assign_nodes_month(df_month, feat_list, tree_depth, q_num)
    md = _opt_max_split_depth()
    root_opt = t_idx > 0 and (md is None or md >= 1)
    if root_opt:
        df["split_1"] = _choose_opt_split_labels(
            df[feat_list[0]],
            prior_by_ti,
            t_idx,
            (),
            feat_list[0],
            qg,
            min_leaf,
            q_num,
        ).to_numpy()
    else:
        df["split_1"] = _ntile(df[feat_list[0]], q_num).to_numpy()
    for bin_val in range(1, q_num + 1):
        mask = df["split_1"] == bin_val
        sub = df.loc[mask].copy()
        if len(sub) == 0:
            continue
        sub = _opt_recurse_subset(
            sub,
            feat_list,
            1,
            tree_depth,
            q_num,
            prior_by_ti,
            t_idx,
            (bin_val,),
            qg,
            min_leaf,
        )
        for k in range(2, tree_depth + 1):
            df.loc[mask, f"split_{k}"] = sub[f"split_{k}"].values
    for i in range(1, tree_depth + 1):
        port = np.ones(len(df), dtype=int)
        for k in range(1, i + 1):
            split_vals = df[f"split_{k}"].to_numpy(dtype=int)
            port += (split_vals - 1) * (q_num ** (i - k))
        df[f"port{i}"] = port
    return df


# ── Single tree: compute all 31 node portfolios ───────────────────────────────

def compute_one_tree(
    panel: pd.DataFrame,
    feat_list: list,
    feats: list,
    tree_depth: int,
    q_num: int,
    y_min: int,
    y_max: int,
    *,
    split_mode: SplitMode = "ntile",
    quantile_grid: np.ndarray | None = None,
    min_leaf: int = OPT_MIN_LEAF,
):
    """
    Compute value-weighted returns and characteristic min/max for all 31 nodes
    of one tree defined by feat_list.

    Matches R's tree_portfolio function.

    Parameters
    ----------
    panel     : full triplet panel (all years, already filtered for NaNs)
    feat_list : list of length tree_depth, which characteristic to split on
                at each depth level e.g. ['LME', 'OP', 'LME', 'Investment']
    feats     : the three base characteristics ['LME', feat1, feat2]
                used to track min/max quantile ranges per node
    split_mode: ``ntile`` (baseline) or ``opt_quantile`` (causal extension).

    Returns
    -------
    ret_table        : np.ndarray (T, 31)  value-weighted excess returns
    feat_min_tables  : list of 3 np.ndarray (T, 31)  min quantile per node
    feat_max_tables  : list of 3 np.ndarray (T, 31)  max quantile per node
    """
    n_months = (y_max - y_min + 1) * 12
    n_nodes  = q_num ** (tree_depth + 1) - 1    # 31 for depth=4
    n_feats  = len(feats)

    ret_table        = np.zeros((n_months, n_nodes))
    feat_min_tables  = [np.zeros((n_months, n_nodes)) for _ in range(n_feats)]
    feat_max_tables  = [np.zeros((n_months, n_nodes)) for _ in range(n_feats)]

    prior_by_ti: dict[int, pd.DataFrame] = {}
    qg = OPT_QUANTILE_GRID if quantile_grid is None else np.asarray(quantile_grid, dtype=float)

    for y in range(y_min, y_max + 1):
        _tree_build_print_year(y)

        df_year = panel[panel['yy'] == y]
        if len(df_year) == 0:
            continue

        # Assign node memberships month by month (fresh splits each month)
        month_frames = []
        for m in range(1, 13):
            t_idx = 12 * (y - y_min) + (m - 1)
            df_month = df_year[df_year['mm'] == m].copy()
            if len(df_month) == 0:
                continue
            if split_mode == "opt_quantile":
                df_month = assign_nodes_month_opt(
                    df_month,
                    feat_list,
                    tree_depth,
                    q_num,
                    prior_by_ti,
                    t_idx,
                    quantile_grid=qg,
                    min_leaf=min_leaf,
                )
                prior_by_ti[t_idx] = df_month
            else:
                df_month = assign_nodes_month(df_month, feat_list, tree_depth, q_num)
            month_frames.append(df_month)

        if len(month_frames) == 0:
            continue

        df_assigned = pd.concat(month_frames, ignore_index=True)

        # ── For each depth level, month, and node: compute VW return + ranges ─
        for i in range(0, tree_depth + 1):     # depth 0 (root) to tree_depth
            n_nodes_at_depth = q_num ** i       # 1, 2, 4, 8, 16

            for m in range(1, 13):
                t_idx = 12 * (y - y_min) + (m - 1)   # 0-indexed time row

                for k in range(1, n_nodes_at_depth + 1):   # 1-indexed node
                    # Column index matches R's: 2^i - 1 + k  (converted to 0-indexed)
                    col_idx = (2 ** i - 1) + (k - 1)

                    mask   = (df_assigned['mm'] == m) & (df_assigned[f'port{i}'] == k)
                    subset = df_assigned[mask]

                    if len(subset) == 0:
                        continue

                    total_size = subset['size'].sum()
                    if total_size == 0:
                        continue

                    # Value-weighted return: sum(ret * size) / sum(size)
                    vw_ret = (subset['ret'] * subset['size']).sum() / total_size
                    ret_table[t_idx, col_idx] = vw_ret

                    # Min/max quantile of each characteristic within this node
                    # These are the quantile-ranked values (0-1) from parquet
                    for f_idx, feat in enumerate(feats):
                        feat_min_tables[f_idx][t_idx, col_idx] = subset[feat].min()
                        feat_max_tables[f_idx][t_idx, col_idx] = subset[feat].max()

    return ret_table, feat_min_tables, feat_max_tables


# ── All 81 trees for one triplet ──────────────────────────────────────────────

def create_tree_portfolio(
    feat1: str,
    feat2: str,
    tree_depth: int = TREE_DEPTH,
    q_num: int = Q_NUM,
    y_min: int = Y_MIN,
    y_max: int = Y_MAX,
    output_path: Path | None = None,
    *,
    split_mode: SplitMode = "ntile",
    quantile_grid: np.ndarray | None = None,
    min_leaf: int = OPT_MIN_LEAF,
) -> None:
    """
    Build all q_num^tree_depth = 81 trees for the triplet (LME, feat1, feat2).
    Saves results to output_path/LME_feat1_feat2/

    Matches R's create_tree_portfolio function.
    For each of the 81 trees, saves:
        {file_id}ret.csv              : (T, 31) value-weighted returns
        {file_id}{feat}_min.csv       : (T, 31) min quantile per node
        {file_id}{feat}_max.csv       : (T, 31) max quantile per node
    """
    feats = ["LME", feat1, feat2]
    sub_dir = "_".join(feats)
    if output_path is None:
        output_path = OUTPUT_PATH_OPT_QUANT if split_mode == "opt_quantile" else OUTPUT_PATH
    out_dir = output_path / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding trees for triplet: {feats}  (split_mode={split_mode})")

    # Load triplet once — all workers share this filtered panel
    panel = load_triplet(feat1, feat2)

    # All 3^4 = 81 combinations of which feature to split on at each depth
    # Matches R's expand.grid(rep(list(1:n_feats), tree_depth))
    # Each combo is a tuple of 0-based indices into feats
    all_combos = list(product(range(len(feats)), repeat=tree_depth))
    print(f"  Processing {len(all_combos)} trees ({len(feats)}^{tree_depth})...")

    qg_list = (
        None
        if quantile_grid is None
        else np.asarray(quantile_grid, dtype=float).tolist()
    )
    w_raw = os.environ.get("SEMINAR_TREE_BUILD_WORKERS", "").strip()
    if w_raw.isdigit():
        n_workers = int(w_raw)
    else:
        cpu = os.cpu_count() or 1
        n_workers = min(max(1, cpu - 1), len(all_combos), 16)
        if n_workers >= 2:
            print(
                f"  parallel tree build ({split_mode}), {n_workers} processes "
                f"(set SEMINAR_TREE_BUILD_WORKERS=1 for sequential)"
            )
    if n_workers > 1:
        n_workers = min(n_workers, len(all_combos))

    def _save_one_tree(combo: tuple, ret_table, feat_min_tables, feat_max_tables) -> None:
        file_id = "".join(str(i + 1) for i in combo)
        pd.DataFrame(ret_table, columns=CNAMES).to_csv(
            out_dir / f"{file_id}ret.csv", index=False
        )
        for f_idx, feat in enumerate(feats):
            pd.DataFrame(feat_min_tables[f_idx], columns=CNAMES).to_csv(
                out_dir / f"{file_id}{feat}_min.csv", index=False
            )
            pd.DataFrame(feat_max_tables[f_idx], columns=CNAMES).to_csv(
                out_dir / f"{file_id}{feat}_max.csv", index=False
            )

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        tasks = [
            (
                k,
                combo,
                feats,
                tree_depth,
                q_num,
                y_min,
                y_max,
                split_mode,
                min_leaf,
                qg_list,
            )
            for k, combo in enumerate(all_combos)
        ]
        print(f"  Parallel: {n_workers} processes (SEMINAR_TREE_BUILD_WORKERS)")
        try:
            from concurrent.futures import as_completed

            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_mp_init_worker,
                initargs=(panel,),
            ) as ex:
                futures = [ex.submit(_mp_run_single_tree, t) for t in tasks]
                results = []
                n_tot = len(tasks)
                for k_done, fut in enumerate(as_completed(futures), start=1):
                    results.append(fut.result())
                    print(
                        f"  [trees] {k_done}/{n_tot} finished "
                        f"({100.0 * k_done / n_tot:.1f}%)",
                        flush=True,
                    )
        except Exception as e:
            print(f"  Parallel build failed ({e}); falling back to sequential.")
            results = []
            for k, combo in enumerate(all_combos):
                feat_list = [feats[i] for i in combo]
                out = compute_one_tree(
                    panel,
                    feat_list,
                    feats,
                    tree_depth,
                    q_num,
                    y_min,
                    y_max,
                    split_mode=split_mode,
                    quantile_grid=quantile_grid,
                    min_leaf=min_leaf,
                )
                results.append((k, combo, out))
        results.sort(key=lambda x: x[0])
        if n_workers > 1:
            print(f"  Writing {len(all_combos)} trees to CSV...", flush=True)
        for k, combo, (ret_table, feat_min_tables, feat_max_tables) in results:
            file_id = "".join(str(i + 1) for i in combo)
            if n_workers <= 1:
                print(f"  Tree {k + 1:02d}/{len(all_combos)}: id={file_id}  (saved)")
            _save_one_tree(combo, ret_table, feat_min_tables, feat_max_tables)
    else:
        for k, combo in enumerate(all_combos):
            feat_list = [feats[i] for i in combo]
            file_id = "".join(str(i + 1) for i in combo)
            print(f"  Tree {k+1:02d}/{len(all_combos)}: id={file_id}  splits={feat_list}")
            ret_table, feat_min_tables, feat_max_tables = compute_one_tree(
                panel,
                feat_list,
                feats,
                tree_depth,
                q_num,
                y_min,
                y_max,
                split_mode=split_mode,
                quantile_grid=quantile_grid,
                min_leaf=min_leaf,
            )
            _save_one_tree(combo, ret_table, feat_min_tables, feat_max_tables)

    print(f"  Done. Results saved to {out_dir}")