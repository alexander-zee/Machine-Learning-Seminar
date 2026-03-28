import pandas as pd
import numpy as np
from pathlib import Path

# Paden
INPUT_PATH = Path('data/prepared/panel_benchmark.parquet')
OUTPUT_PATH = Path('data/prepared/panel_clustering.parquet')
CHARACTERISTICS = ['LME', 'BEME', 'r12_2', 'OP', 'Investment', 'ST_Rev', 'LT_Rev', 'AC', 'LTurnover']

def impute_characteristics():
    print("Inladen van benchmark data voor imputatie...")
    df = pd.read_parquet(INPUT_PATH)
    
    # Sorteer op permno en datum voor correcte forward fill
    df = df.sort_values(['permno', 'date'])
    
    print("Stap 1: Forward filling per aandeel (max 6 maanden)...")
    # We vullen gaten met de laatste bekende waarde, maar niet oneindig lang (bijv. max 6 mnd)
    df[CHARACTERISTICS] = df.groupby('permno')[CHARACTERISTICS].ffill(limit=6)
    
    print("Stap 2: Backward filling per aandeel (max 2 maanden)...")
    # Soms missen de eerste maanden, die vullen we kort terug
    df[CHARACTERISTICS] = df.groupby('permno')[CHARACTERISTICS].bfill(limit=2)
    
    print("Stap 3: Resterende gaten vullen met 0.5 (neutrale rank)...")
    # Voor aandelen die echt geen data hebben voor een specifiek kenmerk
    df[CHARACTERISTICS] = df[CHARACTERISTICS].fillna(0.5)
    
    # Opslaan
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Geimputeerde data opgeslagen in: {OUTPUT_PATH}")

if __name__ == "__main__":
    impute_characteristics()
