import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def lasso_valid_full(ports, lambda0, lambda2, main_dir, sub_dir, adj_w, 
                     n_train_valid=360, cvN=3, runFullCV=False, kmin=5, kmax=50, 
                     RunParallel=False, ParallelN=10, weights_dict_df=None):
    
    os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)
    
    # Extract weights or default to equal weighting
    if weights_dict_df is not None:
        # Assumes weights_dict_df is sorted identically to ports
        all_weights = weights_dict_df['Kernel_Weight'].values
    else:
        all_weights = np.ones(len(ports))
        
    ports_test = ports.iloc[n_train_valid:]
    
    n_valid = int(n_train_valid / cvN)
    loop_range = range(1, cvN + 1) if runFullCV else range(cvN, cvN + 1)
    
    for i in loop_range:
        val_start = (i - 1) * n_valid
        val_end = i * n_valid
        
        ports_valid = ports.iloc[val_start:val_end]
        ports_train = pd.concat([ports.iloc[:val_start], ports.iloc[val_end:n_train_valid]])
        
        # Slice weights to match the training returns exactly
        weights_train = np.concatenate([all_weights[:val_start], all_weights[val_end:n_train_valid]])
        
        # Normalize weights for this specific CV fold
        if np.sum(weights_train) > 0:
            weights_train = weights_train / np.sum(weights_train)
        
        cv_name = f'cv_{i}'
        lasso_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2, 
                        main_dir, sub_dir, adj_w, len(ports), cv_name, 
                        train_weights=weights_train, kmin=kmin, kmax=kmax, 
                        RunParallel=RunParallel, ParallelN=ParallelN)
    
    # Final fit on the whole train+valid period
    ports_train_full = ports.iloc[:n_train_valid]
    weights_full = all_weights[:n_train_valid]
    if np.sum(weights_full) > 0:
        weights_full = weights_full / np.sum(weights_full)

    lasso_cv_helper(ports_train_full, None, ports_test, lambda0, lambda2, 
                    main_dir, sub_dir, adj_w, len(ports), 'full', 
                    train_weights=weights_full, kmin=kmin, kmax=kmax, 
                    RunParallel=RunParallel, ParallelN=ParallelN)

def lasso_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2, main_dir, sub_dir, adj_w, n_total, cv_name, train_weights, kmin=5, kmax=50, RunParallel=False, ParallelN=10):
    # Calculate Kernel-Weighted Mean and Covariance
    mu = np.average(ports_train.values, axis=0, weights=train_weights)
    sigma = np.cov(ports_train.values, rowvar=False, aweights=train_weights, ddof=1)
    
    mu_bar = np.mean(mu)
    gamma = min(ports_train.shape)
    
    # Eigen-decomposition for the regression transformation
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    idx = np.argsort(eigenvalues)[::-1]
    D_all, V_all = eigenvalues[idx], eigenvectors[:, idx]
    
    gamma = min(gamma, np.sum(D_all > 1e-10))
    D, V = D_all[:gamma], V_all[:, :gamma]
    
    sigma_tilde = V @ np.diag(np.sqrt(D)) @ V.T
    mu_matrix = mu.reshape(-1, 1) + (np.array(lambda0).reshape(1, -1) * mu_bar)
    mu_tilde = V @ np.diag(1.0 / np.sqrt(D)) @ V.T @ mu_matrix
    
    if RunParallel:
        Parallel(n_jobs=ParallelN)(
            delayed(_process_lambda0)(
                i, lambda0[i], lambda0, lambda2, sigma_tilde, mu_tilde, 
                ports_train, ports_valid, ports_test, adj_w, kmin, kmax, 
                main_dir, sub_dir, cv_name
            ) for i in range(len(lambda0))
        )
    else:
        for i in range(len(lambda0)):
            _process_lambda0(i, lambda0[i], lambda0, lambda2, sigma_tilde, mu_tilde, 
                             ports_train, ports_valid, ports_test, adj_w, kmin, kmax, 
                             main_dir, sub_dir, cv_name)

def _process_lambda0(i, l0_val, lambda0, lambda2, sigma_tilde, mu_tilde, ports_train, ports_valid, ports_test, adj_w, kmin, kmax, main_dir, sub_dir, cv_name):
    # Note: Ensure 'lasso' is imported or defined in your environment
    from lasso import lasso 
    for j, l2_val in enumerate(lambda2):
        lasso_results = lasso(sigma_tilde, mu_tilde[:, i], l2_val, 100, kmin, kmax)
        
        beta_subset, K_subset = lasso_results
        n_res = beta_subset.shape[0]
        if n_res == 0: continue
            
        train_SR, test_SR = np.zeros(n_res), np.zeros(n_res)
        valid_SR = np.zeros(n_res) if ports_valid is not None else None
        betas = np.zeros((n_res, ports_train.shape[1]))
        
        for r in range(n_res):
            b = (beta_subset[r, :] * adj_w)
            b = b / (np.sum(np.abs(b)) + 1e-10) # Normalize weights
            
            # SR calculation
            def get_sr(p_data, w_vec):
                sdf = np.dot(p_data.values, (w_vec / adj_w))
                return np.mean(sdf) / (np.std(sdf, ddof=1) + 1e-8)

            train_SR[r] = get_sr(ports_train, b)
            test_SR[r] = get_sr(ports_test, b)
            if ports_valid is not None:
                valid_SR[r] = get_sr(ports_valid, b)
            betas[r, :] = b
            
        res_df = pd.DataFrame({'train_SR': train_SR, 'test_SR': test_SR, 'portsN': K_subset})
        if ports_valid is not None: res_df.insert(1, 'valid_SR', valid_SR)
        
        final_res = pd.concat([res_df, pd.DataFrame(betas, columns=ports_train.columns)], axis=1)
        final_res.to_csv(os.path.join(main_dir, sub_dir, f'results_{cv_name}_l0_{i+1}_l2_{j+1}.csv'), index=False)