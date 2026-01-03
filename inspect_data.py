import pandas as pd
import os

file_path = "data/raw/complaints.csv/Complaints.csv"

# Read first few rows to get columns and a sense of data
try:
    df_chunk = pd.read_csv(file_path, nrows=5)
    print("Columns:", df_chunk.columns.tolist())
    
    # Read unique products (batching to avoid memory issues if I read whole col)
    # Actually for unique products I might need to read the whole 'Product' column.
    # checking unique products on a chunk to guess, but eventually I might need to map them.
    # Let's read the 'Product' column entirey, it should fit in memory (strings).
    
    df_products = pd.read_csv(file_path, usecols=['Product'])
    print("\nUnique Products found:")
    print(df_products['Product'].value_counts())

except Exception as e:
    print(f"Error: {e}")
