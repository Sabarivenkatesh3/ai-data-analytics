# app/cleaning.py
import pandas as pd

def clean_sales_df(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Strip column names
    df.columns = [c.strip() for c in df.columns]

    # 2. Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 3. Convert numeric columns
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        # Fill missing with median if you prefer; here we use median
        median_qty = int(df['quantity'].median(skipna=True)) if not df['quantity'].dropna().empty else 0
        df['quantity'].fillna(median_qty, inplace=True)
        df['quantity'] = df['quantity'].astype(int)

    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        median_price = float(df['price'].median(skipna=True)) if not df['price'].dropna().empty else 0.0
        df['price'].fillna(median_price, inplace=True)
        df['price'] = df['price'].astype(float)

    # 4. Compute total_amount column
    if 'quantity' in df.columns and 'price' in df.columns:
        df['total_amount'] = df['quantity'] * df['price']

    # 5. Drop rows with no order_id
    if 'order_id' in df.columns:
        df = df.dropna(subset=['order_id'])

    return df
