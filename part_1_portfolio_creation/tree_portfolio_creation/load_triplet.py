import pandas as pd

def load_triplet(feat1, feat2):
    
    """
    Load panel and filter to complete cases for this specific triplet.
    Matches original R intersect behavior per triplet.
    """
    df = pd.read_parquet(
        'data/prepared/panel.parquet',
        columns=['permno', 'date', 'yy', 'mm', 
                 'ret', 'size', 'LME', feat1, feat2]
    )
    # This is the equivalent of their intersect operation
    # Only keep stocks with all three characteristics available
    before = len(df)
    df = df.dropna(subset=['LME', feat1, feat2])
    after  = len(df)
    print(f"Triplet (LME, {feat1}, {feat2}): "
          f"kept {after}/{before} rows after filtering")
    return df