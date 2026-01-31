import pandas as pd
try:
    df = pd.read_parquet('data/processed/train_v3.parquet')
    print("Columns:", list(df.columns))
    print("Shape:", df.shape)
    print("First 3 rows:")
    print(df.head(3).T)
except Exception as e:
    print(e)
