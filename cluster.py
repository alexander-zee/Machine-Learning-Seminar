import pandas as pd
import numpy as np
import sklearn
import scipy
import glob
from pathlib import Path
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor


def data_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_interp = ["MthRet","LME","BEME","OP","Investment","Accrual",
                      "r12_2","ST_Rev","LT_Rev","Lturnover"]
    
    # Ensure datetime index
    df = df.sort_values(["PERMNO", "date"]).set_index("date")
    
    # Time-based interpolation per stock
    df[cols_to_interp] = (
        df.groupby("PERMNO")[cols_to_interp]
          .transform(lambda g: g.interpolate(method="time", limit_area="inside"))
    )
    
    # Forward and backward fill remaining NaNs
    df[cols_to_interp] = df.groupby("PERMNO")[cols_to_interp].transform(lambda g: g.ffill().bfill())
    
    # Remove any remaining inf values
    df[cols_to_interp] = df[cols_to_interp].replace([np.inf, -np.inf], np.nan)

    return df


def multi_colinearity(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with NaN or inf
    X = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    vif_data = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    
    print(vif_data)


#Not needed
def vif_weighting():

    return 0

def found_vif_weights():

    return 0

def baseline_clustering():

    return 0

def sharpe_clustering():

    return 0


def main_clustering():
    
    df = pd.read_csv("paper_data/datadatadatafinaldata.csv")
    
    # Convert to datetime
    df["date"] = pd.to_datetime(df["MthCalDt"])
    
    # Interpolate and fill missing values
    df_interp = data_interpolation(df)
    
    # Reset index to have PERMNO and date as columns
    df_interp = df_interp.reset_index()
    df_interp["date"] = pd.to_datetime(df_interp["date"]).dt.floor("D")
    
    # Set MultiIndex
    df_interp = df_interp.set_index(["PERMNO", "date"])
    
    # Columns for VIF
    cols_for_vif = ["MthRet","LME","BEME","OP","Investment","Accrual",
                    "r12_2","ST_Rev","LT_Rev","Lturnover"]
    
    df_vif = df_interp[cols_for_vif]
    multi_colinearity(df_vif)

main_clustering()