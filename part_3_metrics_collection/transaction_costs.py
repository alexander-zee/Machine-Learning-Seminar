"""
transaction_costs.py
--------------------
Diagnostic module: compute gross vs net Sharpe ratio after accounting for
proportional transaction costs (Bemelmans et al.).

Two entry points
----------------
compute_net_sharpe
    Kernel: per-month SDF weights stored in full_fit_detail CSV.

compute_net_sharpe_uniform
    Uniform (static): fixed SDF weights from Selected_Ports_Weights_{k}.csv,
    portfolio returns from Selected_Ports_{k}.csv.

Both functions call the shared _run_tc_loop and _save_results helpers.

$ python -m part_3_metrics_collection.transaction_costs2
"""

import numpy as np
import pandas as pd
from pathlib import Path

from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import (
    assign_nodes_month,
    TREE_DEPTH,
    Q_NUM,
)
from part_1_portfolio_creation.tree_portfolio_creation.step2_RP_tree_portfolios import (
    assign_nodes_month_rp,
)

Y_MIN         = 1964
Y_MAX         = 2016
N_TRAIN_VALID = 360


# ── Column name decoder ────────────────────────────────────────────────────────

def decode_column(col_name: str, features: list) -> tuple:
    """
    Parse a portfolio column name into (feat_list, depth, node_num).

    Column format:  <tree_id>.<node_path>
      tree_id   : one digit per depth level (1-3), selects from features list
      node_path : first digit always '1' (root), rest are split directions
                  1 = low half, 2 = high half at that depth

    Returns
    -------
    feat_list : list[str]  which feature to split on at each depth
    depth     : int        number of splits (0 = market portfolio)
    node_num  : int        1-indexed port{depth} value to filter on
    """
    tree_id, node_path = col_name.split('.')
    feat_list  = [features[int(d) - 1] for d in tree_id]
    depth      = len(node_path) - 1
    directions = [int(d) for d in node_path[1:]]
    node_num   = 1
    for k, d in enumerate(directions):
        node_num += (d - 1) * (Q_NUM ** (depth - 1 - k))
    return feat_list, depth, node_num


def load_rp_projection_dict(npz_path: Path) -> dict[str, np.ndarray]:
    """
    Load ``projection_matrices.npz`` written by ``step2_RP_tree_portfolios``.

    Returns a mapping ``tree_id`` (zero-padded, e.g. ``'04'``) → array of shape
    ``(tree_depth, n_feats)``.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"RP projection file not found: {npz_path}")
    data = np.load(npz_path)
    return {str(k): np.asarray(data[k]) for k in data.files}


def decode_rp_column(
    col_name: str,
    feat_cols: list[str],
    proj_by_tree_id: dict[str, np.ndarray],
) -> tuple[list[str], int, int, np.ndarray]:
    """
    Parse an RP combined column ``{tree_id}.{node_path}`` (e.g. ``04.11212``).

    ``tree_id`` must match a key in ``proj_by_tree_id`` (same zero-padding as
    step2/step3). ``node_path`` uses the usual CNAMES encoding; ``port{depth}``
    indexing matches ``assign_nodes_month_rp``.

    Returns
    -------
    feat_list, depth, node_num, proj_matrix
        Tuple suitable as one entry in ``col_info`` for ``_run_tc_loop`` (RP row).
    """
    parts = col_name.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad RP column name: {col_name!r}")
    tree_id, node_path = parts[0], parts[1]
    if tree_id not in proj_by_tree_id:
        keys = sorted(proj_by_tree_id.keys())
        sample = keys[: min(8, len(keys))]
        raise KeyError(
            f"Unknown RP tree_id {tree_id!r} in column {col_name!r}. "
            f"Sample keys: {sample}"
        )
    proj = proj_by_tree_id[tree_id]
    depth = len(node_path) - 1
    if depth < 1:
        raise ValueError(f"Bad RP node_path in column: {col_name!r}")
    directions = [int(d) for d in node_path[1:]]
    node_num = 1
    for k_, d in enumerate(directions):
        node_num += (d - 1) * (Q_NUM ** (depth - 1 - k_))
    feat_list = list(feat_cols)
    if proj.shape[1] != len(feat_list):
        raise ValueError(
            f"Projection width {proj.shape[1]} != len(feat_cols)={len(feat_list)} "
            f"for column {col_name!r}"
        )
    return feat_list, depth, node_num, proj


# ── Stock value-weights for one portfolio in one month ─────────────────────────

def get_stock_value_weights(month_panel: pd.DataFrame,
                            feat_list: list,
                            depth: int,
                            node_num: int) -> pd.Series:
    """
    Return value-weights (size_i / sum(size)) for stocks in a given node.
    Series is indexed by permno. Empty if node has no stocks.
    """
    df     = assign_nodes_month(month_panel, feat_list, TREE_DEPTH, Q_NUM)
    subset = df[df[f'port{depth}'] == node_num]
    if subset.empty:
        return pd.Series(dtype=float)
    total_size = subset['size'].sum()
    if total_size == 0:
        return pd.Series(dtype=float)
    return subset.set_index('permno')['size'] / total_size


def get_stock_value_weights_rp(
    month_panel: pd.DataFrame,
    feat_cols: list[str],
    proj_matrix: np.ndarray,
    depth: int,
    node_num: int,
) -> pd.Series:
    """
    Value-weights for stocks in one RP-tree node for one month (true RP partition).
    """
    df = assign_nodes_month_rp(
        month_panel, feat_cols, proj_matrix, TREE_DEPTH, Q_NUM
    )
    subset = df[df[f"port{depth}"] == node_num]
    if subset.empty:
        return pd.Series(dtype=float)
    total_size = subset["size"].sum()
    if total_size == 0:
        return pd.Series(dtype=float)
    return subset.set_index("permno")["size"] / total_size


# ── Calendar month index ───────────────────────────────────────────────────────

def _build_month_index() -> pd.DataFrame:
    months = [(y, m) for y in range(Y_MIN, Y_MAX + 1) for m in range(1, 13)]
    return pd.DataFrame(months, columns=['yy', 'mm'])


# ── Panel loader ───────────────────────────────────────────────────────────────

def _load_panel(panel_path: Path, features: list, test_months: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_parquet(
        panel_path,
        columns=['permno', 'yy', 'mm', 'ret', 'size'] + features
    )
    panel = panel.dropna(subset=features + ['ret', 'size'])
    panel = panel.merge(
        test_months[['yy', 'mm']].drop_duplicates(),
        on=['yy', 'mm'],
        how='inner',
    )
    return panel


# ── Shared TC computation loop ─────────────────────────────────────────────────

def _run_tc_loop(
    gross_ret: np.ndarray,
    sdf_w_mat: np.ndarray,
    port_cols: list,
    col_info:  dict,
    panel:     pd.DataFrame,
    test_yy:   list,
    test_mm:   list,
) -> np.ndarray:
    """
    Core month-by-month TC loop.

    Parameters
    ----------
    gross_ret : (T_test,)    SDF excess return each month
    sdf_w_mat : (T_test, J)  SDF weights per month (constant rows for uniform)
    port_cols : column names matching sdf_w_mat columns
    col_info  : {col: (feat_list, depth, node_num)} for AP/median trees, or
                {col: (feat_list, depth, node_num, proj_matrix)} for RP columns
                (same TC formula; stock sets from ``assign_nodes_month_rp``).
    panel     : test-period stock panel
    test_yy/mm: plain Python int lists, length T_test

    Returns
    -------
    tc_series : (T_test,) transaction cost each month
    """
    T_test    = len(gross_ret)
    tc_series = np.zeros(T_test)

    # VW_prev stores RAW value-weights from last month, without SDF weights applied.
    # Structure: {permno: {col: vw}}
    # This lets us apply the CURRENT month's SDF weights to last month's stock
    # memberships when computing the drift — correct for time-varying SDF weights.
    VW_prev: dict = {}

    for t in range(T_test):
        yy = test_yy[t]
        mm = test_mm[t]

        month_panel = panel[(panel['yy'] == yy) & (panel['mm'] == mm)].copy()

        if month_panel.empty:
            print(f"  Warning: empty panel for {yy}-{mm:02d}, TC set to 0")
            VW_prev = {}
            continue

        # Rank-normalised market cap → cost parameter (Bemelmans et al. eq. 2)
        N_t = len(month_panel)
        month_panel['me'] = month_panel['size'].rank(method='average') / N_t
        month_panel['c']  = 0.006 - 0.0025 * month_panel['me']

        r_mve = float(gross_ret[t])

        # ── Build VW_current and W_current ────────────────────────────────────
        # VW_current: {permno: {col: vw}}  raw value-weights, stored for next month
        # W_current:  {permno: float}      combined SDF weight = sum_j(w_j * vw_ij)
        VW_current: dict = {}
        W_current:  dict = {}

        for col_idx, col in enumerate(port_cols):
            w_j = float(sdf_w_mat[t, col_idx])
            if w_j == 0.0:
                continue
            meta = col_info[col]
            if len(meta) == 4:
                feat_list, depth, node_num, proj_m = meta
                stock_vw = get_stock_value_weights_rp(
                    month_panel, feat_list, proj_m, depth, node_num
                )
            else:
                feat_list, depth, node_num = meta
                stock_vw = get_stock_value_weights(
                    month_panel, feat_list, depth, node_num
                )

            for permno, vw in stock_vw.items():
                vw_f = float(vw)
                # Store raw vw per portfolio (for next month's drift)
                if permno not in VW_current:
                    VW_current[permno] = {}
                VW_current[permno][col] = vw_f
                # Accumulate combined weight
                W_current[permno] = W_current.get(permno, 0.0) + w_j * vw_f

        # ── Compute W_prev using CURRENT month's SDF weights ──────────────────

        W_prev: dict = {}
        for permno, col_vw in VW_prev.items():
            total = 0.0
            for col_idx, col in enumerate(port_cols):
                vw = col_vw.get(col, 0.0)
                if vw == 0.0:
                    continue
                total += float(sdf_w_mat[t, col_idx]) * vw
            if total != 0.0:
                W_prev[permno] = total

        # ── TC_t = sum_i |W_it - W_drifted_it| * c_it ────────────────────────
        cost_lookup = month_panel.set_index('permno')['c'].to_dict()
        ret_lookup  = month_panel.set_index('permno')['ret'].to_dict()
        tc_t = 0.0

        for permno in set(W_current.keys()) | set(W_prev.keys()):
            W_it      = W_current.get(permno, 0.0)
            W_prev_it = W_prev.get(permno, 0.0)
            R_it      = ret_lookup.get(permno, 0.0)
            W_drifted = W_prev_it * (1.0 + R_it) / (1.0 + r_mve)
            tc_t     += abs(W_it - W_drifted) * cost_lookup.get(permno, 0.0)

        tc_series[t] = tc_t
        VW_prev = VW_current   # carry forward raw weights for next month

        if t == 0 or (t + 1) % 20 == 0:
            print(f"  t={t+1:3d}/{T_test}  [{yy}-{mm:02d}]  "
                  f"gross={r_mve:+.5f}  TC={tc_t:.6f}  "
                  f"net={r_mve - tc_t:+.5f}")

    return tc_series


# ── Shared save + report ───────────────────────────────────────────────────────

def _save_results(
    gross_ret:   np.ndarray,
    tc_series:   np.ndarray,
    test_months: pd.DataFrame,
    out_dir:     Path,
    k_tag:       str,
    label:       str = '',
) -> dict:
    net_ret  = gross_ret - tc_series
    gross_SR = float(gross_ret.mean() / gross_ret.std(ddof=1))
    net_SR   = float(net_ret.mean()   / net_ret.std(ddof=1))

    print(f"\n{'─'*45}  {label}")
    print(f"  Gross SR      : {gross_SR:.4f}")
    print(f"  Net SR        : {net_SR:.4f}")
    print(f"  Mean TC/month : {tc_series.mean():.6f}")
    print(f"  SR loss       : {gross_SR - net_SR:.4f}")
    print(f"{'─'*45}")

    suffix = f"_{label}" if label else ""
    pd.DataFrame({
        'yy':           test_months['yy'].values,
        'mm':           test_months['mm'].values,
        'gross_return': gross_ret,
        'tc':           tc_series,
        'net_return':   net_ret,
    }).to_csv(out_dir / f'transaction_costs_{k_tag}{suffix}.csv', index=False)

    pd.DataFrame([{
        'k':        k_tag,
        'label':    label,
        'gross_SR': gross_SR,
        'net_SR':   net_SR,
        'SR_loss':  gross_SR - net_SR,
        'mean_TC':  float(tc_series.mean()),
        'total_TC': float(tc_series.sum()),
    }]).to_csv(out_dir / f'tc_summary_{k_tag}{suffix}.csv', index=False)

    print(f"  Saved → {out_dir / f'transaction_costs_{k_tag}{suffix}.csv'}")
    print(f"  Saved → {out_dir / f'tc_summary_{k_tag}{suffix}.csv'}")

    return {
        'gross_SR':      gross_SR,
        'net_SR':        net_SR,
        'gross_returns': gross_ret,
        'net_returns':   net_ret,
        'tc_series':     tc_series,
    }


# ── Entry point 1: kernel ──────────────────────────────────────────────────────

def compute_net_sharpe(
    detail_path:   Path,
    panel_path:    Path,
    features:      list,
    n_train_valid: int = N_TRAIN_VALID,
    label:         str = '',
) -> dict:
    """
    Compute net SR for a kernel SDF.

    detail_path : full_fit_detail_k{k}.csv
                  T_test rows × (excess_return + one column per portfolio)
    label       : string tag written into output filenames, e.g. 'gaussian'
    """
    detail    = pd.read_csv(detail_path)
    port_cols = [c for c in detail.columns if c != 'excess_return']
    gross_ret = detail['excess_return'].to_numpy(dtype=float)
    sdf_w_mat = detail[port_cols].to_numpy(dtype=float)
    T_test    = len(detail)

    all_months      = _build_month_index()
    T_test_expected = len(all_months) - n_train_valid   # 636 - 360 = 276

    if T_test != T_test_expected:
        raise ValueError(
            f"Cannot compute TC for {detail_path.name}: "
            f"expected {T_test_expected} rows but found {T_test}. "
            f"One or more months were skipped during LARS. "
            f"Re-run kernel_full_fit with month_idx tracking before computing TC."
        )

    test_months = all_months.iloc[n_train_valid:].reset_index(drop=True)

    panel    = _load_panel(panel_path, features, test_months)
    col_info = {col: decode_column(col, features) for col in port_cols}
    test_yy  = test_months['yy'].to_numpy(dtype=int).tolist()
    test_mm  = test_months['mm'].to_numpy(dtype=int).tolist()

    tc_series = _run_tc_loop(gross_ret, sdf_w_mat, port_cols, col_info,
                             panel, test_yy, test_mm)

    k_tag = detail_path.stem.split('_')[-1]
    return _save_results(gross_ret, tc_series, test_months,
                         detail_path.parent, k_tag, label)


# ── Entry point 2: uniform (static weights) ───────────────────────────────────

def compute_net_sharpe_uniform(
    ports_path:    Path,
    weights_path:  Path,
    panel_path:    Path,
    features:      list,
    n_train_valid: int = N_TRAIN_VALID,
    label:         str = 'uniform',
) -> dict:
    """
    Compute net SR for the uniform (static) SDF.

    ports_path    : Selected_Ports_{k}.csv
                    636 rows × k columns — full-period portfolio returns
    weights_path  : Selected_Ports_Weights_{k}.csv
                    k rows, single column — fixed SDF weights
    """
    ports_full = pd.read_csv(ports_path, header=0)
    port_cols  = ports_full.columns.tolist()
    ports_test = ports_full.iloc[n_train_valid:].reset_index(drop=True)
    T_test     = len(ports_test)

    sdf_w_fixed = pd.read_csv(weights_path, header=0).to_numpy(dtype=float).flatten()
    if len(sdf_w_fixed) != len(port_cols):
        raise ValueError(
            f"weights file has {len(sdf_w_fixed)} entries but ports file has "
            f"{len(port_cols)} columns."
        )

    gross_ret = ports_test.to_numpy(dtype=float) @ sdf_w_fixed   # (T_test,)
    sdf_w_mat = np.tile(sdf_w_fixed, (T_test, 1))                 # (T_test, k)

    all_months  = _build_month_index()
    test_months = all_months.iloc[n_train_valid:n_train_valid + T_test].reset_index(drop=True)

    panel    = _load_panel(panel_path, features, test_months)
    col_info = {col: decode_column(col, features) for col in port_cols}
    test_yy  = test_months['yy'].to_numpy(dtype=int).tolist()
    test_mm  = test_months['mm'].to_numpy(dtype=int).tolist()

    tc_series = _run_tc_loop(gross_ret, sdf_w_mat, port_cols, col_info,
                             panel, test_yy, test_mm)

    k_tag   = f"k{len(port_cols)}"
    out_dir = ports_path.parent
    return _save_results(gross_ret, tc_series, test_months, out_dir, k_tag, label)


# ── Quick run ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    BASE  = Path("data/results/grid_search/tree")
    PANEL = Path("data/prepared/panel.parquet")
    FEATS = ['LME', 'OP', 'Investment']

    compute_net_sharpe(
        detail_path = BASE / "gaussian/LME_OP_Investment/full_fit/full_fit_detail_k10.csv",
        panel_path  = PANEL,
        features    = FEATS,
        label       = 'gaussian',
    )

    compute_net_sharpe_uniform(
        ports_path   = BASE / "uniform/LME_OP_Investment/Selected_Ports_10.csv",
        weights_path = BASE / "uniform/LME_OP_Investment/Selected_Ports_Weights_10.csv",
        panel_path   = PANEL,
        features     = FEATS,
    )