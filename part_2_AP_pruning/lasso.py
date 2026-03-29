import numpy as np
from sklearn.linear_model import lars_path

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
    # sklearn default max_iter=500 can truncate when p is large (~1500 tree nodes); need ≥ p steps.
    alphas, active, coefs = lars_path(
        XX, yy, method="lasso", max_iter=int(max(steps, p + 1, min(n + p, 5000)))
    )
    
    # scikit-learn returns coefficients as (features, steps). 
    # We transpose it to (steps, features) to match R's output format perfectly.
    beta = coefs.T
    
    # Count non-zero elements in each row
    # R: K = apply(beta, 1, function(x){return(sum(x != 0))})
    K = np.sum(beta != 0, axis=1)
    
    # Filter for sparsity between kmin and kmax
    # R: subset = K >= kmin & K <= kmax
    subset_mask = (K >= kmin) & (K <= kmax)
    
    # Return the filtered betas and K counts
    # R: return(list(beta[subset,], K[subset]))
    return beta[subset_mask, :], K[subset_mask]
