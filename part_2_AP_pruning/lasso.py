import numpy as np
from sklearn.linear_model import lars_path

# sklearn can leave tiny numerical crumbs instead of exact zeros; R LARS often has exact zeros.
# Use a tolerance so sparsity counts match R-style "active" sets more closely.
_COEF_ACTIVE_TOL = 1e-10


def lasso(X, y, lambda2, steps=100, kmin=5, kmax=50):
    # Get dimensions of X
    # R: n = nrow(X); p = ncol(X)
    n, p = X.shape
    
    # Pad y with zeros
    # R: yy = c(y, rep(0, p))
    yy = np.concatenate([y, np.zeros(p)])
    
    # Append the diagonal lambda2 penalty matrix to X
    # R: XX = rbind(X, diag(sqrt(lambda2), p, p))
    penalty_matrix = np.diag(np.full(p, np.sqrt(lambda2)))
    XX = np.vstack([X, penalty_matrix])
    
    # Run the LARS algorithm
    # R: lars(XX, yy, type="lasso", normalize = FALSE, intercept = FALSE)
    # `steps` is a floor; effective max_iter is max(steps, p+1, min(n+p, 5000)) so large p is covered.
    alphas, active, coefs = lars_path(
        XX, yy, method="lasso", max_iter=int(max(steps, p + 1, min(n + p, 5000)))
    )
    
    # scikit-learn returns coefficients as (features, steps). 
    # We transpose it to (steps, features) to match R's output format perfectly.
    beta = coefs.T
    
    # Count active coefficients per path step (tolerance for float noise vs R exact zeros).
    # R: K = apply(beta, 1, function(x){return(sum(x != 0))})
    K = np.sum(np.abs(beta) > _COEF_ACTIVE_TOL, axis=1)
    
    # Filter for sparsity between kmin and kmax
    # R: subset = K >= kmin & K <= kmax
    subset_mask = (K >= kmin) & (K <= kmax)
    
    # Return the filtered betas and K counts
    # R: return(list(beta[subset,], K[subset]))
    return beta[subset_mask, :], K[subset_mask]
