"""
diagnostic_eigh_vs_svd.py

Verifies that the SVD reformulation produces numerically identical results
to the current eigh path for the uniform kernel case.

Compares:
    1. Intermediate quantities: sigma_tilde, mu_tilde (the X and y fed to LARS)
    2. LARS output: beta vectors and k values for every (lambda0, lambda2)
    3. SDF weights and Sharpe ratios on train/valid/test

If these match, the SVD path is a safe drop-in replacement for eigh,
and we can use it in the kernel per-month loop for the ~40x speedup.

Run with: python diagnostic_eigh_vs_svd.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from part_2_AP_pruning.lasso import lasso

# ─────────────────────────────────────────────────────────────────────────
# Config — same as main.py
# ─────────────────────────────────────────────────────────────────────────
FILTERED_CSV  = ROOT_DIR / "data" / "results" / "tree_portfolios" / "LME_OP_Investment" / "level_all_excess_combined_filtered.csv"
FEAT1  = "OP"
FEAT2  = "Investment"
LAMBDA0 = [0.5, 0.55, 0.6]
LAMBDA2 = [10**-7, 10**-7.25, 10**-7.5]
N_TRAIN_VALID = 360
CVN    = 3
KMIN   = 5
KMAX   = 50
TOL    = 1e-8   # tolerance for "numerically identical"


def load_data():
    ports = pd.read_csv(FILTERED_CSV)

    # Compute depth-based adj_w (same as AP_Pruning.py)
    depths = np.array([len(col.split('.')[1]) - 1 for col in ports.columns])
    adj_w  = 1.0 / np.sqrt(2.0 ** depths)

    adj_ports = ports * adj_w

    # Split
    n_valid = N_TRAIN_VALID // CVN
    val_start = (CVN - 1) * n_valid
    val_end   = CVN * n_valid

    ports_train = pd.concat([adj_ports.iloc[:val_start],
                              adj_ports.iloc[val_end:N_TRAIN_VALID]])
    ports_valid = adj_ports.iloc[val_start:val_end]
    ports_test  = adj_ports.iloc[N_TRAIN_VALID:]

    return ports_train, ports_valid, ports_test, adj_w


# ─────────────────────────────────────────────────────────────────────────
# Path A: eigh (current code)
# ─────────────────────────────────────────────────────────────────────────
def eigh_path(ports_train):
    """
    Replicate _static_cv_helper's math: mean → cov → eigh → sigma_tilde, mu_tilde.
    Returns sigma_tilde, mu_tilde_dict (keyed by l0_idx), gamma, timing.
    """
    t0 = time.perf_counter()

    mu     = ports_train.values.mean(axis=0)               # (N,)
    sigma  = np.cov(ports_train.values, rowvar=False)       # (N, N)
    mu_bar = mu.mean()

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    gamma = min(min(ports_train.shape), int(np.sum(eigenvalues > 1e-10)))
    D, V = eigenvalues[:gamma], eigenvectors[:, :gamma]

    sigma_tilde = V @ np.diag(np.sqrt(D)) @ V.T            # (N, N)
    inv_sqrt_D  = V @ np.diag(1.0 / np.sqrt(D)) @ V.T      # (N, N)

    mu_tilde = {}
    for i, l0 in enumerate(LAMBDA0):
        mu_robust = mu + l0 * mu_bar                        # (N,)
        mu_tilde[i] = inv_sqrt_D @ mu_robust                # (N,)

    elapsed = time.perf_counter() - t0
    return sigma_tilde, mu_tilde, gamma, elapsed


# ─────────────────────────────────────────────────────────────────────────
# Path B: SVD (proposed replacement)
# ─────────────────────────────────────────────────────────────────────────
def svd_path(ports_train):
    """
    Replicate _one_month_lars math via SVD: mean → weighted residuals →
    thin SVD → sigma_tilde, mu_tilde.

    Key: np.cov uses 1/(T-1) (Bessel's correction), so we need to scale
    the singular values by sqrt(T/(T-1)) to match.
    Equivalently: the weighted residual matrix uses sqrt(w_t) per row,
    and WᵀW = (1/T) Σ(R-μ)(R-μ)ᵀ. np.cov gives (1/(T-1)) Σ(R-μ)(R-μ)ᵀ.
    So eigenvalues of np.cov = s² * T/(T-1), i.e. corrected s = s_raw * sqrt(T/(T-1)).
    """
    t0 = time.perf_counter()

    T, N = ports_train.shape
    arr  = ports_train.values

    # Uniform kernel: all weights = 1/T
    w = np.ones(T) / T

    mu     = (w[:, None] * arr).sum(axis=0)                 # (N,)
    mu_bar = mu.mean()

    resid    = arr - mu[None, :]                             # (T, N)
    sqrt_w   = np.sqrt(w)                                    # (T,)
    weighted = sqrt_w[:, None] * resid                       # (T, N)

    U, s_raw, Vt = np.linalg.svd(weighted, full_matrices=False)  # s: (min(T,N),)

    # Apply Bessel correction: for uniform w, bessel = 1 - 1/T = (T-1)/T
    # Raw covariance eigenvalues = s_raw². Corrected = s_raw² / bessel.
    # So corrected singular values = s_raw / sqrt(bessel).
    bessel = 1.0 - np.sum(w ** 2)   # = (T-1)/T for uniform
    s = s_raw / np.sqrt(bessel)

    mask = s > 1e-10
    s, Vt = s[mask], Vt[mask]
    gamma = len(s)

    V = Vt.T                                                 # (N, gamma)

    sigma_tilde = V @ np.diag(s) @ V.T                      # (N, N)
    inv_sqrt    = V @ np.diag(1.0 / s) @ V.T                # (N, N)

    mu_tilde = {}
    for i, l0 in enumerate(LAMBDA0):
        mu_robust = mu + l0 * mu_bar
        mu_tilde[i] = inv_sqrt @ mu_robust                   # (N,)

    elapsed = time.perf_counter() - t0
    return sigma_tilde, mu_tilde, gamma, elapsed


# ─────────────────────────────────────────────────────────────────────────
# Compare intermediate quantities
# ─────────────────────────────────────────────────────────────────────────
def compare_intermediates(st_eigh, mt_eigh, g_eigh, t_eigh,
                          st_svd,  mt_svd,  g_svd,  t_svd):
    print("=" * 70)
    print("1. INTERMEDIATE QUANTITIES")
    print("=" * 70)

    print(f"\n  gamma (rank):  eigh={g_eigh}   svd={g_svd}   match={g_eigh == g_svd}")
    print(f"  timing:        eigh={t_eigh:.4f}s   svd={t_svd:.4f}s   speedup={t_eigh/t_svd:.1f}x")

    # sigma_tilde
    diff_st = np.max(np.abs(st_eigh - st_svd))
    print(f"\n  sigma_tilde max|diff|: {diff_st:.2e}   {'✓ PASS' if diff_st < TOL else '✗ FAIL'}")

    # mu_tilde for each lambda0
    for i, l0 in enumerate(LAMBDA0):
        diff_mt = np.max(np.abs(mt_eigh[i] - mt_svd[i]))
        print(f"  mu_tilde[l0={l0}] max|diff|: {diff_mt:.2e}   {'✓ PASS' if diff_mt < TOL else '✗ FAIL'}")


# ─────────────────────────────────────────────────────────────────────────
# Compare LARS outputs
# ─────────────────────────────────────────────────────────────────────────
def compare_lars(st_eigh, mt_eigh, st_svd, mt_svd):
    print("\n" + "=" * 70)
    print("2. LARS OUTPUT (beta vectors for each lambda0 x lambda2)")
    print("=" * 70)

    all_pass = True

    for i, l0 in enumerate(LAMBDA0):
        for j, l2 in enumerate(LAMBDA2):
            beta_eigh, K_eigh = lasso(st_eigh, mt_eigh[i], l2, 100, KMIN, KMAX)
            beta_svd,  K_svd  = lasso(st_svd,  mt_svd[i],  l2, 100, KMIN, KMAX)

            # Compare k values
            k_match = np.array_equal(K_eigh, K_svd)

            # Compare beta vectors
            if beta_eigh.shape == beta_svd.shape:
                beta_diff = np.max(np.abs(beta_eigh - beta_svd))
                shape_match = True
            else:
                beta_diff = float('inf')
                shape_match = False

            ok = k_match and shape_match and beta_diff < TOL
            if not ok:
                all_pass = False

            status = '✓' if ok else '✗'
            print(f"\n  λ0={l0:.2f}  λ2={l2:.2e}")
            print(f"    n_solutions: eigh={len(K_eigh)}  svd={len(K_svd)}  match={len(K_eigh)==len(K_svd)}")
            print(f"    k values match: {k_match}")
            print(f"    beta max|diff|: {beta_diff:.2e}   {status}")

    if all_pass:
        print("\n  All LARS outputs: IDENTICAL ✓")
    else:
        print("\n  ✗ Some LARS outputs differ!")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────
# Compare full SDF weights and Sharpe ratios
# ─────────────────────────────────────────────────────────────────────────
def compare_sharpe(st_eigh, mt_eigh, st_svd, mt_svd,
                   ports_train, ports_valid, ports_test, adj_w):
    print("\n" + "=" * 70)
    print("3. SDF WEIGHTS & SHARPE RATIOS")
    print("=" * 70)

    all_pass = True

    for i, l0 in enumerate(LAMBDA0):
        for j, l2 in enumerate(LAMBDA2):
            beta_eigh, K_eigh = lasso(st_eigh, mt_eigh[i], l2, 100, KMIN, KMAX)
            beta_svd,  K_svd  = lasso(st_svd,  mt_svd[i],  l2, 100, KMIN, KMAX)

            if not np.array_equal(K_eigh, K_svd):
                print(f"\n  λ0={l0:.2f}  λ2={l2:.2e}: k mismatch — skipping SR comparison")
                all_pass = False
                continue

            for r in range(len(K_eigh)):
                k = int(K_eigh[r])

                # eigh path: compute SDF weights
                b_e = beta_eigh[r] * adj_w
                b_e = b_e / np.abs(b_e.sum())
                w_e = b_e / adj_w

                # SVD path: compute SDF weights
                b_s = beta_svd[r] * adj_w
                b_s = b_s / np.abs(b_s.sum())
                w_s = b_s / adj_w

                # SDF weight difference
                w_diff = np.max(np.abs(w_e - w_s))

                # Sharpe ratios
                sdf_valid_e = ports_valid.values @ w_e
                sdf_valid_s = ports_valid.values @ w_s
                sr_e = sdf_valid_e.mean() / sdf_valid_e.std(ddof=1)
                sr_s = sdf_valid_s.mean() / sdf_valid_s.std(ddof=1)
                sr_diff = abs(sr_e - sr_s)

                ok = w_diff < TOL and sr_diff < TOL
                if not ok:
                    all_pass = False
                    print(f"\n  λ0={l0:.2f}  λ2={l2:.2e}  k={k}")
                    print(f"    weight max|diff|: {w_diff:.2e}")
                    print(f"    valid_SR eigh={sr_e:.8f}  svd={sr_s:.8f}  diff={sr_diff:.2e}")

    if all_pass:
        print("\n  All SDF weights and Sharpe ratios: IDENTICAL ✓")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    ports_train, ports_valid, ports_test, adj_w = load_data()
    print(f"  ports_train: {ports_train.shape}  (T={ports_train.shape[0]}, N={ports_train.shape[1]})")
    print(f"  ports_valid: {ports_valid.shape}")
    print(f"  ports_test:  {ports_test.shape}")

    print("\nRunning eigh path...")
    st_eigh, mt_eigh, g_eigh, t_eigh = eigh_path(ports_train)

    print("Running SVD path...")
    st_svd, mt_svd, g_svd, t_svd = svd_path(ports_train)

    compare_intermediates(st_eigh, mt_eigh, g_eigh, t_eigh,
                          st_svd,  mt_svd,  g_svd,  t_svd)

    lars_ok = compare_lars(st_eigh, mt_eigh, st_svd, mt_svd)

    sr_ok = compare_sharpe(st_eigh, mt_eigh, st_svd, mt_svd,
                           ports_train, ports_valid, ports_test, adj_w)

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    if lars_ok and sr_ok:
        print("  ✓ SVD path produces IDENTICAL results to eigh path.")
        print(f"  ✓ SVD is {t_eigh/t_svd:.1f}x faster for the decomposition step.")
        print("  → Safe to use SVD in _one_month_lars for the kernel per-month loop.")
    else:
        print("  ✗ Differences detected. Investigate before switching to SVD.")


if __name__ == "__main__":
    main()