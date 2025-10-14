# streamlit_app.py (robust version)
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "sqlite:///./data/cleaned_data.db"

@st.cache_data
def load_data():
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM sales"), conn)
    return df

def safe_to_datetime(series):
    try:
        return pd.to_datetime(series, errors='coerce')
    except Exception:
        return pd.Series([pd.NaT] * len(series))

def main():
    st.set_page_config(page_title="AI Data Analytics - Dashboard", layout="wide")
    st.title("AI Data Analytics — Dashboard (Phase 2)")

    df = load_data()

    if df.empty:
        st.info("No data found in DB. Upload CSV via FastAPI /docs endpoint and come back.")
        return

    # Ensure expected columns exist and have safe dtypes
    # Normalize column names to avoid casing/whitespace issues
    df.columns = [c.strip() for c in df.columns]

    # Convert date column safely if present
    if 'date' in df.columns:
        df['date'] = safe_to_datetime(df['date'])
    else:
        # add a date column of NaT to avoid key errors later
        df['date'] = pd.Series([pd.NaT] * len(df))

    # Ensure numeric columns exist
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    else:
        df['quantity'] = 0

    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0).astype(float)
    else:
        df['price'] = 0.0

    # total_amount: prefer existing column else compute
    if 'total_amount' in df.columns:
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce').fillna(df['quantity'] * df['price'])
    else:
        df['total_amount'] = df['quantity'] * df['price']

    # Top KPIs
    total_revenue = float(df['total_amount'].sum())
    total_orders = int(df['order_id'].nunique()) if 'order_id' in df.columns else len(df)
    avg_order_value = total_revenue / total_orders if total_orders else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Revenue", f"₹{total_revenue:,.2f}")
    k2.metric("Total Orders", f"{total_orders}")
    k3.metric("Avg Order Value", f"₹{avg_order_value:,.2f}")

    st.markdown("---")

    # Sales over Time (only if we have at least one valid date)
    if df['date'].notna().any():
        st.subheader("Sales over Time")
        # Group by date (day) and compute revenue per day
        # Use pd.Grouper to be robust
        daily = df.set_index('date').groupby(pd.Grouper(freq='D'))['total_amount'].sum().reset_index()
        daily.columns = ['date', 'revenue']
        # Drop rows where date is NaT
        daily = daily.dropna(subset=['date'])
        if not daily.empty:
            # set index to date for plotting
            daily = daily.sort_values('date')
            daily = daily.set_index('date')
            st.line_chart(daily['revenue'])
        else:
            st.info("No valid daily data to plot.")
    else:
        st.info("No valid 'date' column found — upload a CSV with a date column to see time-series charts.")

    st.subheader("Top Regions")
    if 'region' in df.columns:
        region_df = df.groupby('region').agg({'total_amount':'sum'}).reset_index().sort_values('total_amount', ascending=False)
        if not region_df.empty:
            st.bar_chart(region_df.set_index('region')['total_amount'])
        else:
            st.info("No region data to show.")
    else:
        st.info("No 'region' column found in data.")

    st.subheader("Raw Data")
    st.dataframe(df)

    # Sidebar filters
    st.sidebar.header("Filters")
    products = ["All"] + sorted(df['product'].dropna().unique().tolist()) if 'product' in df.columns else ["All"]
    regions = ["All"] + sorted(df['region'].dropna().unique().tolist()) if 'region' in df.columns else ["All"]
    sel_product = st.sidebar.selectbox("Product", products)
    sel_region = st.sidebar.selectbox("Region", regions)

    filtered = df.copy()
    if sel_product != "All":
        filtered = filtered[filtered['product'] == sel_product]
    if sel_region != "All":
        filtered = filtered[filtered['region'] == sel_region]

    st.sidebar.markdown(f"Filtered rows: {len(filtered)}")
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.experimental_rerun()

    # Download button
    csv_bytes = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered CSV", csv_bytes, file_name="filtered.csv", mime="text/csv")

if __name__ == "__main__":
    main()
