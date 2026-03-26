import pandas as pd
import numpy as np
import sklearn
import scipy
import glob
from pathlib import Path
import os


def data_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_interp = ["ret", "LME", "OP", "Investment", "size"]
     # Ensure sorted properly (date is index now)
    df = df.sort_values(["permno", df.index.name])

    # Interpolate per permno
    df[cols_to_interp] = (
        df.groupby("permno")[cols_to_interp]
          .apply(lambda g: g.interpolate(method="time", limit_area="inside"))
    )

    return df


def multi_colinearity():
    return 0



def vif_weighting():

    return 0

def found_vif_weights():

    return 0

def baseline_clustering():

    return 0

def sharpe_clustering():

    return 0


def main_clustering():
    
    folder = Path("paper_data") / "data_chunk_files_quantile/LME_OP_Investment"

    files = list(folder.glob("*.csv"))

    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    # Prepare data
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["permno", "date"]).set_index("date")

    df_interp = data_interpolation(df)

    factors = pd.read_csv("paper_data/factor/tradable_factors.csv")


main_clustering()