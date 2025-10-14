# app/cleaning.py
import pandas as pd

def clean_sales_df(df: pd.DataFrame) -> pd.DataFrame:
    # Simple cleaning steps explained for beginners:
    # 1. Strip column names, lower-case them
    df.columns = [c.strip() for c in df.columns]
    # 2. Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # 3. Fill missing numeric 'quantity' with 0 (or median depending on use-case)
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['quantity'].fillna(0, inplace=True)
        df['quantity'] = df['quantity'].astype(int)
    # 4. Convert price to numeric
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    # 5. Drop rows with no order_id
    df = df.dropna(subset=['order_id'])
    return df
