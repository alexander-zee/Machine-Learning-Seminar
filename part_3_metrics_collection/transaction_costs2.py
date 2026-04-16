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

Run with: python -m part_3_metrics_collection.transaction_costs2
"""

import numpy as np
import pandas as pd
from pathlib import Path

from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import assign_nodes_month, TREE_DEPTH, Q_NUM

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
    # Merge is faster than row-wise apply for filtering to test months
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
    Core month-by-month TC loop shared by both entry points.

    Parameters
    ----------
    gross_ret : (T_test,)    SDF excess return each month
    sdf_w_mat : (T_test, J)  SDF weights per month (constant rows for uniform)
    port_cols : column names matching sdf_w_mat columns
    col_info  : {col: (feat_list, depth, node_num)} pre-decoded
    panel     : test-period stock panel
    test_yy/mm: plain Python int lists, length T_test

    Returns
    -------
    tc_series : (T_test,) transaction cost each month
    """
    T_test    = len(gross_ret)
    tc_series = np.zeros(T_test)
    W_prev: dict = {}

    for t in range(T_test):
        yy = test_yy[t]
        mm = test_mm[t]

        month_panel = panel[(panel['yy'] == yy) & (panel['mm'] == mm)].copy()

        if month_panel.empty:
            print(f"  Warning: empty panel for {yy}-{mm:02d}, TC set to 0")
            W_prev = {}
            continue

        # Rank-normalised market cap → cost parameter (Bemelmans et al. eq. 2)
        N_t = len(month_panel)
        month_panel['me'] = month_panel['size'].rank(method='average') / N_t
        month_panel['c']  = 0.006 - 0.0025 * month_panel['me']

        r_mve = float(gross_ret[t])

        # Build W_current: total spanning-portfolio weight per stock
        W_current: dict = {}
        for col_idx, col in enumerate(port_cols):
            w_j = float(sdf_w_mat[t, col_idx])
            if w_j == 0.0:
                continue
            feat_list, depth, node_num = col_info[col]
            stock_vw = get_stock_value_weights(month_panel, feat_list, depth, node_num)
            for permno, vw in stock_vw.items():
                W_current[permno] = W_current.get(permno, 0.0) + w_j * float(vw)

        # TC_t = sum_i |W_it - W_drifted_{i,t}| * c_it
        cost_lookup = month_panel.set_index('permno')['c'].to_dict()
        ret_lookup  = month_panel.set_index('permno')['ret'].to_dict()
        tc_t = 0.0
        #Loop over all of the stocks in the current and previous portfolios
        for permno in set(W_current.keys()) | set(W_prev.keys()):
            W_it      = W_current.get(permno, 0.0)
            W_prev_it = W_prev.get(permno, 0.0)
            R_it      = ret_lookup.get(permno, 0.0)
            W_drifted = W_prev_it * (1.0 + R_it) / (1.0 + r_mve)
            tc_t     += abs(W_it - W_drifted) * cost_lookup.get(permno, 0.0)

        tc_series[t] = tc_t
        W_prev = W_current

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


# ── Entry point 1: kernel / rolling ───────────────────────────────────────────

def compute_net_sharpe(
    detail_path:   Path,
    panel_path:    Path,
    features:      list,
    n_train_valid: int = N_TRAIN_VALID,
    label:         str = '',
) -> dict:
    """
    Compute net SR for a kernel or rolling-window SDF.

    detail_path : full_fit_detail_k{k}.csv
                  T_test rows × (excess_return + one column per portfolio)
    label       : string tag written into output filenames, e.g. 'gaussian'
    """
    detail    = pd.read_csv(detail_path)
    port_cols = [c for c in detail.columns if c != 'excess_return']
    gross_ret = detail['excess_return'].to_numpy(dtype=float)
    sdf_w_mat = detail[port_cols].to_numpy(dtype=float)
    T_test    = len(detail)

    all_months  = _build_month_index()
    test_months = all_months.iloc[n_train_valid:n_train_valid + T_test].reset_index(drop=True)

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