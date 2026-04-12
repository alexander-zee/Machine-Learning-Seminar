"""
ledoit_wolf_sr_test.py — HAC inference for SR differences (Ledoit & Wolf 2008, Section 3.1).

Implements exactly the HAC test from Section 3.1 of:
    Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis testing
    with the Sharpe ratio. Journal of Empirical Finance, 15:850-859.

The test operates on the 4-dimensional vector (equation numbering from paper):

    y_t = (r_ti - mu_i,  r_tn - mu_n,  r²_ti - gamma_i,  r²_tn - gamma_n)

The long-run covariance matrix Psi of y_t is estimated via HAC kernel estimation
(Section 3.1), with the T/(T-4) small-sample degrees-of-freedom correction.

The standard error for Delta_hat = SR_i - SR_n is obtained via the delta method
(Eq. 4-5 of the paper):

    s(Delta_hat) = sqrt( grad_f(v_hat)' @ Psi_hat @ grad_f(v_hat) / T )

where f(a,b,c,d) = a/sqrt(c-a²) - b/sqrt(d-b²)  maps the four moments
(mu_i, mu_n, gamma_i, gamma_n) to the SR difference, and

    grad_f(a,b,c,d) = (  c/(c-a²)^1.5,
                         -d/(d-b²)^1.5,
                         -0.5*a/(c-a²)^1.5,
                          0.5*b/(d-b²)^1.5  )

The two-sided p-value is:
    p = 2 * Phi(-|Delta_hat| / s(Delta_hat))

We use the Quadratic Spectral (QS) kernel with Andrews (1991) automatic bandwidth,
as recommended by the paper for the time series case.

Three pairwise comparisons are run:
    uniform vs exponential
    uniform vs gaussian
    exponential vs gaussian

Output CSVs: data/results/diagnostics/lw_test_{A}_vs_{B}_k{K}.csv
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
K             = 10
N_TRAIN_VALID = 360

GRID_SEARCH_PATH = Path("data/results/grid_search/tree")
OUTPUT_PATH      = Path("data/results/diagnostics")

CHARACTERISTICS = [
    "BEME", "r12_2", "OP", "Investment",
    "ST_Rev", "LT_Rev", "AC", "LTurnover", "IdioVol",
]

PAIRS = list(combinations(CHARACTERISTICS, 2))

CHAR_LABELS: dict[str, str] = {
    "LME":        "Size",
    "BEME":       "Val",
    "r12_2":      "Mom",
    "OP":         "Prof",
    "Investment": "Inv",
    "ST_Rev":     "SRev",
    "LT_Rev":     "LRev",
    "AC":         "Acc",
    "LTurnover":  "Turn",
    "IdioVol":    "IVol",
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_excess_returns(kernel_name: str, subdir: str, k: int) -> pd.Series | None:
    path = (
        GRID_SEARCH_PATH / kernel_name / subdir / "full_fit"
        / f"full_fit_detail_k{k}.csv"
    )
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "excess_return" not in df.columns:
            warnings.warn(f"No excess_return column in {path}")
            return None
        return df["excess_return"].reset_index(drop=True)
    except Exception as e:
        warnings.warn(f"Failed to load {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build y_t vectors (Section 3.1)
#
# y_t = (r_ti - mu_i,  r_tn - mu_n,  r²_ti - gamma_i,  r²_tn - gamma_n)
#
# mu_i = E[r_ti],  gamma_i = E[r²_ti]  (uncentered second moment)
# Note: sigma²_i = gamma_i - mu_i²
# ─────────────────────────────────────────────────────────────────────────────

def _build_y(r_i: np.ndarray, r_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    Y     : (T, 4) matrix of y_t vectors
    v_hat : (4,) = (mu_i, mu_n, gamma_i, gamma_n)
    """
    mu_i    = r_i.mean()
    mu_n    = r_n.mean()
    gamma_i = (r_i ** 2).mean()
    gamma_n = (r_n ** 2).mean()

    v_hat = np.array([mu_i, mu_n, gamma_i, gamma_n])

    Y = np.column_stack([
        r_i    - mu_i,
        r_n    - mu_n,
        r_i**2 - gamma_i,
        r_n**2 - gamma_n,
    ])  # (T, 4)

    return Y, v_hat


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — HAC estimator for Psi (Section 3.1)
#
# Psi_hat = T/(T-4) * sum_{j=-(T-1)}^{T-1} k(j/S_T) * C_hat_T(j)
#
# C_hat_T(j) = (1/T) * sum_{t=j+1}^{T} y_hat_t @ y_hat_{t-j}'   for j >= 0
#
# We use the QS kernel with Andrews (1991) automatic bandwidth.
# ─────────────────────────────────────────────────────────────────────────────

def _qs_kernel(x: float) -> float:
    """
    Quadratic Spectral kernel k(x).
    k(0) = 1,  k(x) -> 0 as x -> ±inf.
    QS has characteristic exponent q=2, guaranteeing PSD estimates.
    """
    if abs(x) < 1e-10:
        return 1.0
    z = 6 * np.pi * x / 5
    return (25 / (12 * np.pi**2 * x**2)) * (np.sin(z) / z - np.cos(z))


def _andrews_bandwidth(Y: np.ndarray) -> float:
    """
    Andrews (1991) automatic bandwidth for the QS kernel.

    Implements exactly Eq. 6.4 of Andrews (1991) for q=2 (QS kernel):

        alpha_hat(2) = [ sum_a  4*rho_a^2 * sigma_a^4 / (1-rho_a)^8 ]
                       / [ sum_a  sigma_a^4 / (1-rho_a)^4 ]

        S_T = 1.3221 * (alpha_hat(2) * T)^(1/5)

    where a = 1,...,p indexes the p=4 columns of Y, rho_a is the OLS
    AR(1) coefficient for column a, and sigma_a^2 is the innovation
    variance (mean squared AR(1) residual) for column a.

    Equal weights w_a = 1/p are used, which cancel in the ratio,
    leaving the expression above.

    The constant 1.3221 is the kernel-specific constant for the QS kernel
    derived analytically by Andrews (1991), Table I.
    """
    T = Y.shape[0]

    numerator   = 0.0
    denominator = 0.0

    for col in range(Y.shape[1]):
        y = Y[:, col]

        if y.std() < 1e-12:
            continue

        # OLS AR(1) — no intercept needed since Y is already mean-zero
        y_lag = y[:-1]
        y_cur = y[1:]
        denom_ols = np.dot(y_lag, y_lag)
        if denom_ols < 1e-12:
            continue

        rho = float(np.dot(y_lag, y_cur) / denom_ols)
        rho = np.clip(rho, -0.999, 0.999)

        # Innovation variance: sigma_a^2 = (1/T) * sum(residuals^2)
        resid  = y_cur - rho * y_lag
        sigma2 = float(np.dot(resid, resid) / T)
        sigma4 = sigma2 ** 2

        # Accumulate numerator and denominator of Andrews (1991) Eq. 6.4
        numerator   += 4.0 * rho**2 * sigma4 / (1.0 - rho)**8
        denominator += sigma4 / (1.0 - rho)**4

    if denominator < 1e-12:
        alpha2 = 1.0
    else:
        alpha2 = numerator / denominator

    alpha2 = max(alpha2, 1e-6)   # numerical guard

    # Andrews (1991) Table I, QS kernel: S_T = 1.3221 * (alpha(2) * T)^(1/5)
    return float(1.3221 * (alpha2 * T) ** 0.2)


def _hac_psi(Y: np.ndarray) -> np.ndarray:
    """
    HAC estimator for the (4x4) long-run covariance matrix Psi.

    Exactly implements Section 3.1 of Ledoit & Wolf (2008):

        Psi_hat = T/(T-4) * sum_{j=-(T-1)}^{T-1} k(j/S_T) * C_hat_T(j)

    Exploits symmetry C_hat(-j) = C_hat(j)' to sum j=0,...,T-1 only.
    The T/(T-4) factor is the small-sample DOF correction for estimating
    the 4-vector v_hat = (mu_i, mu_n, gamma_i, gamma_n).
    """
    T = Y.shape[0]
    p = Y.shape[1]   # 4

    S_T  = _andrews_bandwidth(Y)
    Psi  = np.zeros((p, p))

    for j in range(T):
        weight = _qs_kernel(j / S_T)
        if abs(weight) < 1e-14:
            continue

        # C_hat_T(j) = (1/T) * Y[j:]' @ Y[:T-j]   shape (p, p)
        # Paper: C_hat_T(j) = (1/T) sum_{t=j+1}^T y_hat_t @ y_hat_{t-j}'
        # In 0-indexed Python: sum over t=j,...,T-1 of Y[t] @ Y[t-j]'
        #                    = Y[j:].T @ Y[:T-j]  / T
        C_j = (Y[j:].T @ Y[:T - j]) / T

        if j == 0:
            Psi += weight * C_j
        else:
            # j > 0: add C_hat(j) and C_hat(-j) = C_hat(j)'
            Psi += weight * (C_j + C_j.T)

    # Small-sample DOF correction: T/(T-4)
    Psi *= T / (T - 4)

    return Psi


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Delta method gradient (paper, just below Eq. 4)
#
# f(a,b,c,d) = a/sqrt(c-a²) - b/sqrt(d-b²)
#
# grad_f = ( c/(c-a²)^1.5,      d/d(mu_i)
#            -d/(d-b²)^1.5,     d/d(mu_n)
#            -0.5a/(c-a²)^1.5,  d/d(gamma_i)
#             0.5b/(d-b²)^1.5 ) d/d(gamma_n)
# ─────────────────────────────────────────────────────────────────────────────

def _delta_grad(v_hat: np.ndarray) -> np.ndarray | None:
    """
    Gradient of f(v) = SR_i - SR_n w.r.t. v = (mu_i, mu_n, gamma_i, gamma_n).
    Returns None if either variance is non-positive (degenerate).
    """
    a, b, c, d = v_hat

    var_i = c - a**2    # sigma²_i
    var_n = d - b**2    # sigma²_n

    if var_i <= 0 or var_n <= 0:
        return None

    return np.array([
         c / var_i**1.5,           # d f / d mu_i
        -d / var_n**1.5,           # d f / d mu_n
        -0.5 * a / var_i**1.5,    # d f / d gamma_i
         0.5 * b / var_n**1.5,    # d f / d gamma_n
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Assemble the test (Eq. 5 + two-sided p-value)
# ─────────────────────────────────────────────────────────────────────────────

def _lw_hac_test(r_i: np.ndarray, r_n: np.ndarray) -> dict:
    """
    Full Section 3.1 HAC test from Ledoit & Wolf (2008).

    Parameters
    ----------
    r_i : (T,) excess returns for strategy i  (kernel A)
    r_n : (T,) excess returns for strategy n  (kernel B)

    Returns
    -------
    dict with SR_i, SR_n, Delta_hat, se, t_stat, p_value, n_obs
    """
    T = len(r_i)
    assert len(r_n) == T

    # Moment estimates and y_t matrix
    Y, v_hat = _build_y(r_i, r_n)
    a, b, c, d = v_hat

    var_i = c - a**2
    var_n = d - b**2

    if var_i <= 0 or var_n <= 0:
        return {
            "SR_i": np.nan, "SR_n": np.nan, "Delta_hat": np.nan,
            "se": np.nan, "t_stat": np.nan, "p_value": np.nan, "n_obs": T,
        }

    SR_i      = a / np.sqrt(var_i)
    SR_n      = b / np.sqrt(var_n)
    Delta_hat = SR_i - SR_n    # = f(v_hat)

    # HAC long-run covariance matrix (4x4)
    Psi_hat = _hac_psi(Y)

    # Delta method gradient
    grad = _delta_grad(v_hat)
    if grad is None:
        return {
            "SR_i": SR_i, "SR_n": SR_n, "Delta_hat": Delta_hat,
            "se": np.nan, "t_stat": np.nan, "p_value": np.nan, "n_obs": T,
        }

    # Standard error: s(Delta_hat) = sqrt( grad' Psi_hat grad / T )  [Eq. 5]
    var_Delta = float(grad @ Psi_hat @ grad) / T
    if var_Delta <= 0:
        return {
            "SR_i": SR_i, "SR_n": SR_n, "Delta_hat": Delta_hat,
            "se": np.nan, "t_stat": np.nan, "p_value": np.nan, "n_obs": T,
        }

    se      = np.sqrt(var_Delta)
    t_stat  = Delta_hat / se

    # Two-sided p-value: p = 2 * Phi(-|Delta_hat / se|)
    p_value = float(2 * stats.norm.cdf(-abs(t_stat)))

    return {
        "SR_i":      float(SR_i),
        "SR_n":      float(SR_n),
        "Delta_hat": float(Delta_hat),
        "se":        float(se),
        "t_stat":    float(t_stat),
        "p_value":   float(p_value),
        "n_obs":     T,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_lw_comparison(
    kernel_a: str,
    kernel_b: str,
    k: int = K,
    characteristics: list[str] = CHARACTERISTICS,
    output_path: Path = OUTPUT_PATH,
    save: bool = True,
    expected_test_length: int = 276,
) -> pd.DataFrame:
    """
    Run LW HAC SR tests for all cross-sections comparing kernel_a vs kernel_b.
    kernel_a = strategy i,  kernel_b = strategy n,  Delta = SR_i - SR_n.

    Alignment policy
    ----------------
    The full_fit_detail CSVs do not store dates — rows are positional.
    Positional alignment is only valid when both series are complete
    (length == expected_test_length). Any shorter series means LARS
    skipped at least one month; without dates we cannot know which
    month was skipped and cannot safely align the two series.

        Both == expected_test_length  -> proceed
        Lengths differ                -> SKIP (definitely misaligned)
        Both == same length < expected-> SKIP (alignment unverifiable)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    pairs   = list(combinations(characteristics, 2))
    records = []

    label_a = kernel_a.capitalize()
    label_b = kernel_b.capitalize()

    print(
        f"\nLedoit-Wolf HAC SR test (Section 3.1): {label_a} vs {label_b}  (k={k})\n"
        f"{'─'*75}",
        flush=True,
    )

    for feat1, feat2 in pairs:
        subdir   = f"LME_{feat1}_{feat2}"
        cs_label = (
            f"{CHAR_LABELS.get('LME','LME')}x"
            f"{CHAR_LABELS.get(feat1, feat1)}x"
            f"{CHAR_LABELS.get(feat2, feat2)}"
        )

        base = {
            "cross_section": subdir,
            "char1": "LME",
            "char2": feat1,
            "char3": feat2,
            "cs_label": cs_label,
        }

        r_a_s = _load_excess_returns(kernel_a, subdir, k)
        r_b_s = _load_excess_returns(kernel_b, subdir, k)

        if r_a_s is None or r_b_s is None:
            missing = (
                ([kernel_a] if r_a_s is None else []) +
                ([kernel_b] if r_b_s is None else [])
            )
            print(f"  {subdir:<35}  [SKIP -- missing: {', '.join(missing)}]", flush=True)
            records.append({
                **base,
                f"SR_{label_a}": np.nan, f"SR_{label_b}": np.nan,
                "Delta_hat": np.nan, "se": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "sig_10": False, "sig_05": False, "sig_01": False,
                "n_obs": None, "status": f"missing_{'+'.join(missing)}",
            })
            continue

        len_a, len_b = len(r_a_s), len(r_b_s)

        if len_a != len_b:
            print(
                f"  {subdir:<35}  [SKIP -- length mismatch: "
                f"{kernel_a}={len_a}, {kernel_b}={len_b}. "
                f"Cannot align without dates.]",
                flush=True,
            )
            records.append({
                **base,
                f"SR_{label_a}": np.nan, f"SR_{label_b}": np.nan,
                "Delta_hat": np.nan, "se": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "sig_10": False, "sig_05": False, "sig_01": False,
                "n_obs": None,
                "status": f"length_mismatch_{kernel_a}={len_a}_{kernel_b}={len_b}",
            })
            continue

        if len_a != expected_test_length:
            print(
                f"  {subdir:<35}  [SKIP -- both series length {len_a}, "
                f"expected {expected_test_length}. LARS skipped "
                f"{expected_test_length - len_a} month(s); "
                f"positional alignment unverifiable without dates.]",
                flush=True,
            )
            records.append({
                **base,
                f"SR_{label_a}": np.nan, f"SR_{label_b}": np.nan,
                "Delta_hat": np.nan, "se": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "sig_10": False, "sig_05": False, "sig_01": False,
                "n_obs": len_a,
                "status": f"alignment_unverifiable_length={len_a}",
            })
            continue

        # Both series are complete -- positional alignment is valid
        res = _lw_hac_test(np.asarray(r_a_s), np.asarray(r_b_s))

        sig_10 = bool(res["p_value"] < 0.10) if not np.isnan(res["p_value"]) else False
        sig_05 = bool(res["p_value"] < 0.05) if not np.isnan(res["p_value"]) else False
        sig_01 = bool(res["p_value"] < 0.01) if not np.isnan(res["p_value"]) else False
        stars  = "***" if sig_01 else "**" if sig_05 else "*" if sig_10 else ""

        print(
            f"  {subdir:<35}  "
            f"SR_{label_a}={res['SR_i']:+.3f}  SR_{label_b}={res['SR_n']:+.3f}  "
            f"Δ={res['Delta_hat']:+.3f}  se={res['se']:.3f}  "
            f"t={res['t_stat']:+.2f}  p={res['p_value']:.3f}  {stars}",
            flush=True,
        )

        records.append({
            **base,
            f"SR_{label_a}":  res["SR_i"],
            f"SR_{label_b}":  res["SR_n"],
            "Delta_hat":      res["Delta_hat"],
            "se":             res["se"],
            "t_stat":         res["t_stat"],
            "p_value":        res["p_value"],
            "sig_10":         sig_10,
            "sig_05":         sig_05,
            "sig_01":         sig_01,
            "n_obs":          res["n_obs"],
            "status":         "ok",
        })

    df = pd.DataFrame(records)

    ok = df[df["status"] == "ok"]
    print(f"\n{'─'*75}")
    print(f"  Cross-sections tested:            {len(ok)}/{len(pairs)}")
    if len(ok) > 0:
        print(f"  {label_a} beats {label_b}:             {int((ok['Delta_hat'] > 0).sum())}/{len(ok)}")
        print(f"  {label_b} beats {label_a}:             {int((ok['Delta_hat'] < 0).sum())}/{len(ok)}")
        print(f"  Mean ΔSR ({label_a}−{label_b}):       {ok['Delta_hat'].mean():+.4f}")
        print(f"  Significant at 10%:               {int(ok['sig_10'].sum())}/{len(ok)}")
        print(f"  Significant at  5%:               {int(ok['sig_05'].sum())}/{len(ok)}")
        print(f"  Significant at  1%:               {int(ok['sig_01'].sum())}/{len(ok)}")

    if save:
        out_csv = output_path / f"lw_test_{kernel_a}_vs_{kernel_b}_k{k}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n  Saved → {out_csv}", flush=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    COMPARISONS = [
        ("uniform",     "exponential"),
        ("uniform",     "gaussian"),
        ("exponential", "gaussian"),
    ]

    for kernel_a, kernel_b in COMPARISONS:
        run_lw_comparison(
            kernel_a=kernel_a,
            kernel_b=kernel_b,
            k=K,
            output_path=OUTPUT_PATH,
            save=True,
        )

    print("\n" + "═" * 75)
    print("All comparisons complete.")
    print("═" * 75)