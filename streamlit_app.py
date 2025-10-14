# streamlit_app.py (replace existing)
import streamlit as st
import pandas as pd
from app.db import engine, list_tables, read_table
from app.insights import summarize_table

def main():
    st.set_page_config(page_title="AI Data Analytics - Dynamic Dashboard", layout="wide")
    st.title("AI Data Analytics — Dynamic Dashboard")

    tables = list_tables()
    if not tables:
        st.info("No tables found in DB. Upload a CSV via FastAPI /upload-csv first.")
        return

    table = st.selectbox("Choose table", tables)
    limit = st.sidebar.number_input("Rows to preview", min_value=10, max_value=5000, value=200, step=10)
    df = read_table(table, limit=limit)

    st.subheader(f"Table: {table} — {len(df)} rows (preview {limit})")
    st.dataframe(df)

    # Show generic insights
    st.subheader("Auto-generated summary")
    summary = summarize_table(df)
    st.write("**Text summary:**")
    st.write(summary.get("text_summary", ""))

    st.write("**Numeric columns**")
    for c, stats in summary["numeric"].items():
        st.write(f"- {c}: sum={stats['sum']:.2f}, mean={stats['mean']:.2f}, min={stats['min']}, max={stats['max']}, nulls={stats['nulls']}")

    st.write("**Categorical columns (top values)**")
    for c, info in summary["categorical"].items():
        st.write(f"- {c}: nulls={info['nulls']}, top: {info['top_values']}")

    st.write("**Date columns**")
    for c, info in summary["dates"].items():
        st.write(f"- {c}: min={info['min']}, max={info['max']}, nulls={info['nulls']}")

if __name__ == "__main__":
    main()
