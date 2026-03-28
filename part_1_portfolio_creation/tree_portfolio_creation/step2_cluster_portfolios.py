import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Paden
INPUT_PATH = Path('../../data/prepared/panel.parquet')
OUTPUT_DIR = Path('../../data/portfolios/clusters/')
N_CLUSTERS = 10
CHARACTERISTICS = ['LME', 'BEME', 'r12_2', 'OP', 'Investment', 'ST_Rev', 'LT_Rev', 'AC', 'LTurnover']

def create_cluster_portfolios():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} niet gevonden.")
        return

    print("Loading panel data...")
    df = pd.read_parquet(INPUT_PATH)
    dates = sorted(df['date'].unique())
    all_monthly_returns = []

    print(f"Clustering over {len(dates)} months...")
    for date in tqdm(dates):
        month_data = df[df['date'] == date].copy()
        if len(month_data) < N_CLUSTERS: continue
            
        cluster_model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
        month_data['cluster'] = cluster_model.fit_predict(month_data[CHARACTERISTICS])
        
        # Value-Weighted Returns
        def vw_avg(group):
            return np.average(group['ret'], weights=group['size'])
            
        monthly_rets = month_data.groupby('cluster').apply(vw_avg)
        monthly_rets.name = date
        all_monthly_returns.append(monthly_rets)

    # Matrix opslaan
    cluster_matrix = pd.concat(all_monthly_returns, axis=1).T
    cluster_matrix.columns = [f'cluster_{i+1}' for i in range(N_CLUSTERS)]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster_matrix.to_csv(OUTPUT_DIR / 'cluster_returns.csv')

    # --- VISUALISATIE: Dendrogram van de laatste maand ---
    print("\nGenerating dendrogram for the most recent month...")
    last_month_date = dates[-1]
    last_month_data = df[df['date'] == last_month_date].copy()
    
    # We nemen een sample van 150 aandelen voor de leesbaarheid van de grafiek
    sample_size = min(150, len(last_month_data))
    sample = last_month_data.sample(sample_size, random_state=42)
    
    # Bereken de linkage matrix voor het dendrogram
    Z = linkage(sample[CHARACTERISTICS], method='ward')
    
    plt.figure(figsize=(12, 7))
    plt.title(f'Ward Clustering Dendrogram - {last_month_date.strftime("%Y-%m")}')
    plt.xlabel('Aandelen (Sample)')
    plt.ylabel('Afstand (Ward Linkage)')
    
    dendrogram(Z, truncate_mode='lastp', p=N_CLUSTERS, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
    
    plt.axhline(y=Z[-(N_CLUSTERS-1), 2], color='r', linestyle='--', label=f'{N_CLUSTERS} Clusters')
    plt.legend()
    
    # Opslaan als PNG
    plot_path = OUTPUT_DIR / 'cluster_dendrogram.png'
    plt.savefig(plot_path)
    print(f"Dendrogram opgeslagen als: {plot_path}")
    plt.show()

if __name__ == "__main__":
    create_cluster_portfolios()
