import os
import pandas as pd
from lasso_valid_par_full import lasso_valid_full

def AP_Pruning(feats_list, feat1, feat2, input_path, input_file_name, output_path, 
               n_train_valid=360, cvN=3, runFullCV=False, kmin=5, kmax=50, 
               RunParallel=False, ParallelN=10, IsTree=True, 
               lambda0=[0, 0.1, 0.2], lambda2=[0.01, 0.05, 0.1], 
               weights_dict_df=None):
    
    feats_chosen = ['LME', feats_list[feat1], feats_list[feat2]]
    sub_dir = "_".join(feats_chosen)
    
    # Load ports
    ports = pd.read_csv(os.path.join(input_path, sub_dir, input_file_name))
    
    # --- FIX: Remove the date column before math begins ---
    if 'yyyymm' in ports.columns:
        ports = ports.drop(columns=['yyyymm'])
    # ------------------------------------------------------
    
    if IsTree:
        # Parse the tree sorting weights from column names
        adj_w = [float(col.split('_')[-1]) for col in ports.columns]
    else:
        adj_w = [1.0] * ports.shape[1]
    
    lasso_valid_full(ports, lambda0, lambda2, output_path, sub_dir, adj_w, 
                     n_train_valid, cvN, runFullCV, kmin, kmax, 
                     RunParallel, ParallelN, weights_dict_df)
