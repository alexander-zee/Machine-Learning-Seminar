import os
import stat

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --- PADEN (Aangepast naar jouw nieuwe MICE file) ---
SCRIPT_DIR = Path(__file__).parent.absolute()
# We gaan vanuit tree_portfolio_creation naar de data map
INPUT_PATH = SCRIPT_DIR.parent.parent / 'data' / 'prepared' / 'panel_clustering_mice.parquet'
OUTPUT_DIR = SCRIPT_DIR.parent.parent / 'data' / 'portfolios' / 'clusters'
N_CLUSTERS = 10
CHARACTERISTICS = [
    'LME', 'BEME', 'r12_2', 'OP', 'Investment',
    'ST_Rev', 'LT_Rev', 'AC', 'LTurnover', 'IdioVol',
]


def _write_csv_replace(df: pd.DataFrame, path: Path) -> None:
    """Write CSV via a sibling temp file, then replace. Avoids partial reads of target.

    On Windows, overwriting fails with PermissionError if the target is open (e.g. Excel)
    or read-only; we chmod the existing file when possible and leave a .tmp file if replace fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.write_tmp{path.suffix}"
    df.to_csv(tmp, encoding="utf-8")
    if path.exists() and os.name == "nt":
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass
    try:
        os.replace(tmp, path)
    except PermissionError as e:
        raise PermissionError(
            f"Kan {path} niet overschrijven (bestand open in Excel/andere app, of map alleen-lezen). "
            f"Nieuwe matrix staat in: {tmp}\n"
            f"Sluit alles dat {path.name} gebruikt, verwijder of hernoem het oude bestand, "
            f"hernoem daarna {tmp.name} naar {path.name}, of run opnieuw."
        ) from e


def create_cluster_portfolios():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} niet gevonden. Run eerst je MICE script!")
        return

    print(f"Loading MICE-imputed panel data: {INPUT_PATH.name}...")
    df = pd.read_parquet(INPUT_PATH)
    
    # Zorg dat de datum goed staat
    df['date'] = pd.to_datetime(df['date'])
    dates = sorted(df['date'].unique())
    all_monthly_returns = []

    print(f"Clustering over {len(dates)} months met {N_CLUSTERS} clusters...")
    
    for date in tqdm(dates):
        month_data = df[df['date'] == date].copy()
        month_data = month_data.dropna(subset=['ret'] + CHARACTERISTICS)
        if len(month_data) < N_CLUSTERS:
            continue

        cluster_model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
        month_data['cluster'] = cluster_model.fit_predict(month_data[CHARACTERISTICS])

        # Value weights: LME in de panel is quantiel [0,1]; ruwe log-ME staat in 'size' (step1_prepare_data)
        if 'size' in month_data.columns and month_data['size'].notna().any():
            s = month_data['size'].astype(float)
            s = s.fillna(s.median())
            month_data['weights'] = np.exp(np.clip(s, -50.0, 50.0))
        else:
            month_data['weights'] = 1.0

        cl_ids = sorted(month_data['cluster'].unique())
        vw = []
        for c in cl_ids:
            sub = month_data[month_data['cluster'] == c]
            w = sub['weights'].to_numpy(dtype=float)
            if np.all(w <= 0) or not np.isfinite(w).all():
                w = np.ones(len(sub))
            vw.append(np.average(sub['ret'].to_numpy(dtype=float), weights=w))
        monthly_rets = pd.Series(vw, index=cl_ids, name=date)
        all_monthly_returns.append(monthly_rets)

    # Matrix opslaan
    print("\nMatrix assembleren...")
    cluster_matrix = pd.concat(all_monthly_returns, axis=1).T
    cluster_matrix.columns = [f'cluster_{i+1}' for i in range(N_CLUSTERS)]
    
    out_csv = OUTPUT_DIR / "cluster_returns.csv"
    _write_csv_replace(cluster_matrix, out_csv)
    print(f"Returns opgeslagen in: {out_csv}")

    # --- VISUALISATIE: Dendrogram van de laatste maand ---
    print("\nGenerating dendrogram for the most recent month...")
    last_month_date = dates[-1]
    last_month_data = df[df['date'] == last_month_date].copy()
    
    sample_size = min(200, len(last_month_data))
    sample = last_month_data.sample(sample_size, random_state=42)
    
    Z = linkage(sample[CHARACTERISTICS], method='ward')
    
    plt.figure(figsize=(12, 7))
    plt.title(f'Ward Clustering Dendrogram (MICE Data) - {last_month_date.strftime("%Y-%m")}')
    plt.xlabel('Aandelen (Sample)')
    plt.ylabel('Afstand (Ward Linkage)')
    
    dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90., leaf_font_size=10.)
    
    plt.axhline(y=Z[-N_CLUSTERS+1, 2], color='r', linestyle='--', label=f'Cut-off voor {N_CLUSTERS} Clusters')
    plt.legend()
    
    plot_path = OUTPUT_DIR / 'cluster_dendrogram.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Dendrogram opgeslagen als: {plot_path.name}")
    plt.close()

if __name__ == "__main__":
    create_cluster_portfolios()
