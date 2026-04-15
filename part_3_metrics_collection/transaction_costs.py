"""
transaction_costs.py
--------------------
Diagnostic module: compute gross vs net Sharpe ratio for a kernel-weighted
SDF after accounting for proportional transaction costs (Bemelmans et al.).

Inputs
------
- full_fit_detail_k{k}.csv  : per-month SDF weights + excess returns (test period)
- panel.parquet              : stock-level panel with ret, size, characteristics

Usage
-----
    from transaction_costs import compute_net_sharpe

    results = compute_net_sharpe(
        detail_path = Path("data/results/grid_search/tree/gaussian/LME_OP_Investment/full_fit/full_fit_detail_k10.csv"),
        panel_path  = Path("data/prepared/panel.parquet"),
        features    = ['LME', 'OP', 'Investment'],
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path

from part_1_portfolio_creation.tree_portfolio_creation.step2_tree_portfolios import assign_nodes_month, TREE_DEPTH, Q_NUM

# Full panel date range — must match step1/step2 config
Y_MIN         = 1964
Y_MAX         = 2016
N_TRAIN_VALID = 360   # months in train + validation window


# ── Column name decoder ────────────────────────────────────────────────────────

def decode_column(col_name: str, features: list) -> tuple:
    """
    Parse a portfolio column name into the information needed to reconstruct
    its stock membership.

    Parameters
    ----------
    col_name : str  e.g. '1221.11212'
    features : list e.g. ['LME', 'OP', 'Investment']

    Returns
    -------
    feat_list  : list[str]  which feature to split on at each depth level
                 e.g. ['LME', 'OP', 'OP', 'LME']
    depth      : int        number of splits defining this node (0 = market)
    node_num   : int        1-indexed port{depth} value to filter on

    Notes
    -----
    Column format:  <tree_id>.<node_path>
      tree_id   : digits 1-3, one per depth level, selecting from features
      node_path : first digit always '1' (root marker), remaining digits are
                  split directions (1=low half, 2=high half) at each depth
    """
    tree_id, node_path = col_name.split('.')

    # Which feature to split on at each depth
    feat_list = [features[int(d) - 1] for d in tree_id]

    # Depth = number of splits = node_path length minus the root marker digit
    depth = len(node_path) - 1

    # Node number: mirrors step2's port computation
    #   port = 1 + sum_k (split_k - 1) * 2^(depth - k)
    # where split_k ∈ {1, 2} is the direction at depth k
    directions = [int(d) for d in node_path[1:]]   # skip root '1'
    node_num = 1
    for k, d in enumerate(directions):
        node_num += (d - 1) * (Q_NUM ** (depth - 1 - k))

    return feat_list, depth, node_num


# ── Stock membership for one portfolio in one month ────────────────────────────

def get_stock_value_weights(month_panel: pd.DataFrame,
                            feat_list: list,
                            depth: int,
                            node_num: int) -> pd.Series:
    """
    Run assign_nodes_month and return value-weights for stocks in a given node.

    Parameters
    ----------
    month_panel : DataFrame for a single month, must contain permno, size,
                  and all columns in feat_list
    feat_list   : which feature to split at each depth (length = TREE_DEPTH)
    depth       : which port{depth} column to filter on
    node_num    : the port{depth} value identifying this node

    Returns
    -------
    pd.Series indexed by permno, values = size_i / sum(size) within node.
    Empty series if node has no stocks.
    """
    df = assign_nodes_month(month_panel, feat_list, TREE_DEPTH, Q_NUM)

    subset = df[df[f'port{depth}'] == node_num]
    if subset.empty:
        return pd.Series(dtype=float)

    total_size = subset['size'].sum()
    if total_size == 0:
        return pd.Series(dtype=float)

    return subset.set_index('permno')['size'] / total_size


# ── Build calendar month index for the panel ───────────────────────────────────

def _build_month_index() -> pd.DataFrame:
    """
    Return a DataFrame with columns [yy, mm] for every month from Y_MIN to Y_MAX,
    in chronological order. Row i corresponds to time index i in the portfolio CSVs.
    """
    months = []
    for y in range(Y_MIN, Y_MAX + 1):
        for m in range(1, 13):
            months.append((y, m))
    return pd.DataFrame(months, columns=['yy', 'mm'])

def compute_net_sharpe(
    detail_path: Path,
    panel_path: Path,
    features: list,
    n_train_valid: int = N_TRAIN_VALID,
) -> dict:
    """
    Compute gross and net Sharpe ratios for a kernel-weighted SDF, accounting
    for proportional transaction costs at the individual stock level.
 
    Parameters
    ----------
    detail_path   : path to full_fit_detail_k{k}.csv
    panel_path    : path to panel.parquet
    features      : the three characteristics for this cross-section,
                    in the same order as used in step2, e.g. ['LME','OP','Investment']
    n_train_valid : number of months in the train+validation window (default 360)
 
    Returns
    -------
    dict with keys:
        gross_SR, net_SR,
        gross_returns, net_returns, tc_series   (all numpy arrays, length T_test)
    """
    # ── Load detail file ──────────────────────────────────────────────────────
    detail    = pd.read_csv(detail_path)
    port_cols = [c for c in detail.columns if c != 'excess_return']
    gross_ret = detail['excess_return'].to_numpy(dtype=float)   # (T_test,)
    sdf_w_mat = detail[port_cols].values             # (T_test, N_cols)
    T_test    = len(detail)
 
    # ── Align detail rows to calendar months ──────────────────────────────────
    # The portfolio CSVs have one row per month from Y_MIN to Y_MAX.
    # The test period starts at row n_train_valid (0-indexed).
    all_months  = _build_month_index()
    test_months = all_months.iloc[n_train_valid:].reset_index(drop=True)
 
    if T_test > len(test_months):
        raise ValueError(
            f"detail file has {T_test} rows but only {len(test_months)} "
            "test months exist. Check n_train_valid."
        )
    if T_test < len(test_months):
        # Some months were skipped by LARS — not handled here
        raise ValueError(
            f"detail file has {T_test} rows but expected {len(test_months)}. "
            "Some months were skipped during LARS. Cannot align without "
            "month indices in the detail file. Re-run kernel_full_fit with "
            "month tracking, or verify k_target convergence."
        )
 
    # ── Load panel: only columns we need, only test-period months ─────────────
    panel = pd.read_parquet(
        panel_path,
        columns=['permno', 'yy', 'mm', 'ret', 'size'] + features
    )
    # Drop rows missing any characteristic used for splits
    panel = panel.dropna(subset=features + ['ret', 'size'])
    # Keep only test months
    test_ym = set(zip(test_months['yy'], test_months['mm']))
    panel   = panel[panel.apply(lambda r: (r['yy'], r['mm']) in test_ym, axis=1)]
 
    # ── Pre-decode all column names (done once) ────────────────────────────────
    col_info = {col: decode_column(col, features) for col in port_cols}
 
    # Pre-extract as plain Python lists to avoid pandas scalar type ambiguity
    test_yy = test_months['yy'].to_numpy(dtype=int).tolist()
    test_mm = test_months['mm'].to_numpy(dtype=int).tolist()
 
    # Main loop: one iteration per test month
    tc_series = np.zeros(T_test)
    W_prev    = {}
 
    for t in range(T_test):
        yy = test_yy[t]
        mm = test_mm[t]
 
        month_panel = panel[(panel['yy'] == yy) & (panel['mm'] == mm)].copy()
 
        if month_panel.empty:
            print(f"  Warning: empty panel for {yy}-{mm:02d}, skipping TC")
            W_prev = {}
            continue
 
        # Rank-normalised market cap: me_it = rank / N  (Bemelmans et al. eq.2 fn4)
        N_t = len(month_panel)
        month_panel['me'] = month_panel['size'].rank(method='average') / N_t
        month_panel['c']  = 0.006 - 0.0025 * month_panel['me']
 
        # Scalar SDF portfolio return this month (for drift denominator)
        r_mve = float(gross_ret[t])
 
        # ── Build W_current: stock-level combined position ─────────────────
        W_current = {}
 
        for col_idx, col in enumerate(port_cols):
            w_j = float(sdf_w_mat[t, col_idx])
            if w_j == 0.0:
                continue
 
            feat_list, depth, node_num = col_info[col]
            stock_vw = get_stock_value_weights(month_panel, feat_list, depth, node_num)
 
            for permno, vw in stock_vw.items():
                W_current[permno] = W_current.get(permno, 0.0) + w_j * float(vw)
 
        # ── Compute TC_t ──────────────────────────────────────────────────
        cost_lookup = month_panel.set_index('permno')['c'].to_dict()
        ret_lookup  = month_panel.set_index('permno')['ret'].to_dict()
 
        all_permnos = set(W_current.keys()) | set(W_prev.keys())
        tc_t = 0.0
 
        for permno in all_permnos:
            W_it       = W_current.get(permno, 0.0)
            W_prev_it  = W_prev.get(permno, 0.0)
            R_it       = ret_lookup.get(permno, 0.0)
 
            # Drift previous weight to start of period t before rebalancing
            W_drifted = W_prev_it * (1.0 + R_it) / (1.0 + r_mve)
 
            c_it  = cost_lookup.get(permno, 0.0)
            tc_t += abs(W_it - W_drifted) * c_it
 
        tc_series[t] = tc_t
        W_prev = W_current   # only this dict is kept; all else is garbage collected
 
        if t == 0 or (t + 1) % 20 == 0:
            print(f"  t={t+1:3d}/{T_test}  [{yy}-{mm:02d}]  "
                  f"gross={r_mve:+.5f}  TC={tc_t:.6f}  "
                  f"net={r_mve - tc_t:+.5f}")
 
    # ── Sharpe ratios ──────────────────────────────────────────────────────────
    net_ret  = gross_ret - tc_series
 
    gross_SR = float(gross_ret.mean() / gross_ret.std(ddof=1))
    net_SR   = float(net_ret.mean()   / net_ret.std(ddof=1))
 
    print(f"\n{'─'*45}")
    print(f"  Gross SR : {gross_SR:.4f}")
    print(f"  Net SR   : {net_SR:.4f}")
    print(f"  Mean TC/month : {tc_series.mean():.6f}")
    print(f"  SR loss  : {gross_SR - net_SR:.4f}")
    print(f"{'─'*45}")
 
    # ── Save outputs next to the detail file ───────────────────────────────────
    out_dir = detail_path.parent
    k_tag   = detail_path.stem.split('_')[-1]   # e.g. 'k10'
 
    # Per-month series
    pd.DataFrame({
        'yy':          test_months['yy'].values,
        'mm':          test_months['mm'].values,
        'gross_return': gross_ret,
        'tc':           tc_series,
        'net_return':   net_ret,
    }).to_csv(out_dir / f'transaction_costs_{k_tag}.csv', index=False)
 
    # One-row summary
    pd.DataFrame([{
        'k':         k_tag,
        'gross_SR':  gross_SR,
        'net_SR':    net_SR,
        'SR_loss':   gross_SR - net_SR,
        'mean_TC':   float(tc_series.mean()),
        'total_TC':  float(tc_series.sum()),
    }]).to_csv(out_dir / f'tc_summary_{k_tag}.csv', index=False)
 
    print(f"  Saved → {out_dir / f'transaction_costs_{k_tag}.csv'}")
    print(f"  Saved → {out_dir / f'tc_summary_{k_tag}.csv'}")
 
    return {
        'gross_SR':      gross_SR,
        'net_SR':        net_SR,
        'gross_returns': gross_ret,
        'net_returns':   net_ret,
        'tc_series':     tc_series,
    }
 
 
 
 
# ── Quick run ──────────────────────────────────────────────────────────────────
 
if __name__ == '__main__':
    results = compute_net_sharpe(
        detail_path = Path("data/results/grid_search/tree/gaussian/LME_OP_Investment/full_fit/full_fit_detail_k10.csv"),
        panel_path  = Path("data/prepared/panel.parquet"),
        features    = ['LME', 'OP', 'Investment'],
    )