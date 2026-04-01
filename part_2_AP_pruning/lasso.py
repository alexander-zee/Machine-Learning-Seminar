"""
lasso.py — LARS path for LASSO with Ridge-augmented system.

The Ridge penalty (lambda2) is incorporated by augmenting X and y before
running LARS, which traces the full L1 path in one pass. We return only
solutions where the number of nonzero coefficients k is in [kmin, kmax].
"""

import numpy as np
from sklearn.linear_model import lars_path


def lasso(X: np.ndarray, y: np.ndarray, lambda2: float, steps: int = 70, kmin: int = 5, kmax: int = 50):
    """
    Run LARS on the Ridge-augmented system and return sparse solutions.
    Input  : X (n,p) design matrix (sigma_tilde), y (n,) response (mu_tilde[:,i]),
             lambda2 scalar Ridge penalty, kmin/kmax portfolio count bounds.
    Output : (beta_filtered, K_filtered) — beta (n_solutions, p) weight matrix
             and K (n_solutions,) vector of corresponding nonzero counts.
    """
    p = X.shape[1]
    yy = np.concatenate([y, np.zeros(p)])
    XX = np.vstack([X, np.diag(np.full(p, np.sqrt(lambda2)))])

    _, _, coefs = lars_path(XX, yy, method='lasso')
    beta = coefs.T  # (n_steps, p)

    K = np.sum(beta != 0, axis=1)
    mask = (K >= kmin) & (K <= kmax)
    return beta[mask], K[mask]