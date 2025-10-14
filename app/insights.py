# app/insights.py
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "sqlite:///./data/cleaned_data.db"

def load_df():
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM sales"), conn)
    # normalize
    df.columns = [c.strip() for c in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0).astype(float)
    if 'total_amount' in df.columns:
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce').fillna(df['quantity'] * df['price'])
    else:
        df['total_amount'] = df['quantity'] * df['price']
    return df

# app/insights.py (only the summarize_overall shown here; keep other helpers)
def summarize_overall(df: pd.DataFrame, days: int = 7, mode: str = "recent") -> dict:
    """
    Summarize revenue/orders.
    - days: window size
    - mode: "recent" (default) -> attempts to use last `days` relative to today,
             if that yields empty but df has dates, anchor window to data max date,
             if mode="all" -> use entire dataset.
    """
    # Normalize dates if present
    has_dates = 'date' in df.columns and df['date'].notna().any()

    if mode == "all" or not has_dates:
        # Use entire dataset
        recent = df.copy()
        prev = df.iloc[0:0]
        period_start = None
        period_end = None
    else:
        # Try using actual "today" window first
        today = pd.Timestamp.now().normalize()
        period_end = today
        period_start = today - pd.Timedelta(days=days-1)
        recent = df[(df['date'] >= period_start) & (df['date'] <= period_end)]
        prev = df[(df['date'] >= (period_start - pd.Timedelta(days=days))) & (df['date'] < period_start)]

        # If recent is empty but df has dates, anchor window to data's latest date
        if recent.empty and has_dates:
            data_max = df['date'].max().normalize()
            period_end = data_max
            period_start = data_max - pd.Timedelta(days=days-1)
            recent = df[(df['date'] >= period_start) & (df['date'] <= period_end)]
            prev = df[(df['date'] >= (period_start - pd.Timedelta(days=days))) & (df['date'] < period_start)]

    # Compute KPIs (same as before)
    recent_revenue = float(recent['total_amount'].sum()) if not recent.empty else 0.0
    prev_revenue = float(prev['total_amount'].sum()) if not prev.empty else 0.0
    revenue_change_pct = ((recent_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue else None

    total_orders = int(recent['order_id'].nunique()) if 'order_id' in recent.columns and not recent.empty else int(len(recent))
    avg_order = (recent['total_amount'].sum() / total_orders) if total_orders else 0.0

    top_product = None
    top_product_share = None
    if 'product' in recent.columns and not recent.empty:
        p = recent.groupby('product')['total_amount'].sum().sort_values(ascending=False)
        if not p.empty:
            top_product = p.index[0]
            top_product_share = float(p.iloc[0]) / recent_revenue * 100 if recent_revenue else 0.0

    top_region = None
    if 'region' in recent.columns and not recent.empty:
        r = recent.groupby('region')['total_amount'].sum().sort_values(ascending=False)
        if not r.empty:
            top_region = r.index[0]

    summary = {
        "mode": mode,
        "period_days": days,
        "period_start": str(period_start) if period_start is not None else None,
        "period_end": str(period_end) if period_end is not None else None,
        "recent_revenue": recent_revenue,
        "prev_revenue": prev_revenue,
        "revenue_change_pct": revenue_change_pct,
        "total_orders": total_orders,
        "avg_order_value": avg_order,
        "top_product": top_product,
        "top_product_share_pct": top_product_share,
        "top_region": top_region,
        "rows_in_recent_window": len(recent)
    }
    return summary


def format_text_summary(s: dict) -> str:
    # Human-readable text from the summary dict
    lines = []
    lines.append(f"Summary for last {s['period_days']} day(s):")
    lines.append(f"- Revenue: ₹{s['recent_revenue']:.2f}")
    if s['revenue_change_pct'] is None:
        lines.append(f"- No previous period to compare.")
    else:
        sign = "+" if s['revenue_change_pct'] >= 0 else ""
        lines.append(f"- Change vs previous period: {sign}{s['revenue_change_pct']:.2f}%")
    lines.append(f"- Orders: {s['total_orders']}")
    lines.append(f"- Avg order value: ₹{s['avg_order_value']:.2f}")
    if s.get('top_product'):
        lines.append(f"- Top product: {s['top_product']} ({s['top_product_share_pct']:.1f}% of revenue)")
    if s.get('top_region'):
        lines.append(f"- Top region: {s['top_region']}")
    return "\n".join(lines)
