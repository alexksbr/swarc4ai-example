# explore_data.py
import pandas as pd

# Load January data from parquet file
df = pd.read_parquet('data/yellow_tripdata_2025-01.parquet')
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nData types:")
print(df.dtypes)