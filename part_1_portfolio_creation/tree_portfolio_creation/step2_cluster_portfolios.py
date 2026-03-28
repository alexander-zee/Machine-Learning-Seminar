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
CHARACTERISTICS = ['LME', 'BEME', 'r12_2', 'OP', 'Investment', 'ST_Rev', 'LT_Rev', 'AC', 'LTurnover']

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
        if len(month_data) < N_CLUSTERS: 
            continue
            
        # Ward Clustering op de geïmputeerde kenmerken
        cluster_model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
        month_data['cluster'] = cluster_model.fit_predict(month_data[CHARACTERISTICS])
        
        # Value-Weighted Returns berekenen
        # LET OP: We gebruiken LME (Log Market Equity) of de originele size voor weging.
        # Meestal wordt de ruwe Market Equity gebruikt, maar LME is hier onze proxy.
        month_data['weights'] = np.exp(month_data['LME']) # Terug naar normale schaal voor weging
        
        def vw_avg(group):
            return np.average(group['ret'], weights=group['weights'])
            
        monthly_rets = month_data.groupby('cluster', group_keys=False).apply(vw_avg)
        monthly_rets.name = date
        all_monthly_returns.append(monthly_rets)

    # Matrix opslaan
    print("\nMatrix assembleren...")
    cluster_matrix = pd.concat(all_monthly_returns, axis=1).T
    cluster_matrix.columns = [f'cluster_{i+1}' for i in range(N_CLUSTERS)]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster_matrix.to_csv(OUTPUT_DIR / 'cluster_returns.csv')
    print(f"Returns opgeslagen in: {OUTPUT_DIR / 'cluster_returns.csv'}")

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
    plt.savefig(plot_path)
    print(f"Dendrogram opgeslagen als: {plot_path.name}")
    plt.show()

if __name__ == "__main__":
    create_cluster_portfolios()
