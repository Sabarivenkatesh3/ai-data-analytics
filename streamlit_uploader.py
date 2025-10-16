# streamlit_uploader.py
"""
Web-based file uploader - Upload ANY dataset from your browser!
Drag and drop CSV/Excel files directly in the browser.
"""

import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from app.cleaning import detect_and_clean
from app.db import create_table_from_df, list_tables, read_table, sanitize_table_name
from app.insights import summarize_table
import os

# Page config
st.set_page_config(
    page_title="AI Data Analytics - Universal Upload",
    page_icon="📊",
    layout="wide"
)

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/cleaned", exist_ok=True)

def main():
    st.title("🚀 AI Data Analytics - Upload ANY Dataset")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Navigation")
        page = st.radio(
            "Choose Action:",
            ["Upload New Dataset", "View Existing Data", "System Info"]
        )
        st.markdown("---")
        st.markdown("### 🎯 Features")
        st.markdown("""
        - ✅ Upload ANY CSV/Excel
        - 🧹 Auto-clean data
        - 📊 Generate insights
        - 💾 Download cleaned files
        - 🔍 View all tables
        """)
    
    # ===================================================================
    # PAGE 1: UPLOAD NEW DATASET
    # ===================================================================
    if page == "Upload New Dataset":
        st.header("📤 Upload Your Dataset")
        st.markdown("""
        Upload **any CSV or Excel file** from your computer. The system will:
        1. ✨ Automatically detect data types
        2. 🧹 Clean the data intelligently
        3. 💾 Save to database
        4. 📊 Generate insights
        5. 📥 Create downloadable cleaned file
        """)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Drag and drop or click to browse"
        )
        
        # Optional table name
        custom_table_name = st.text_input(
            "Custom table name (optional)",
            help="Leave empty to use filename"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
            handle_outliers = st.checkbox("Handle outliers", value=True)
            outlier_threshold = st.slider(
                "Outlier sensitivity (IQR multiplier)",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.5,
                help="Lower = more aggressive outlier removal"
            )
        
        # Process button
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: **{uploaded_file.name}**")
            
            if st.button("🚀 Process Dataset", type="primary"):
                with st.spinner("🔄 Processing your dataset..."):
                    try:
                        # Read file
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                            file_type = "CSV"
                        else:
                            df = pd.read_excel(uploaded_file)
                            file_type = "Excel"
                        
                        # Display original data info
                        st.subheader("📥 Original Data")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Rows", df.shape[0])
                        col2.metric("Columns", df.shape[1])
                        col3.metric("Missing Values", df.isna().sum().sum())
                        
                        with st.expander("👀 Preview Original Data (first 10 rows)"):
                            st.dataframe(df.head(10))
                        
                        # Save raw file
                        raw_path = os.path.join("data/raw", uploaded_file.name)
                        if file_type == "CSV":
                            df.to_csv(raw_path, index=False)
                        else:
                            df.to_excel(raw_path, index=False)
                        
                        # Clean the data
                        st.subheader("🧹 Cleaning Data...")
                        table_name = custom_table_name or uploaded_file.name.rsplit('.', 1)[0]
                        safe_name = sanitize_table_name(table_name)
                        
                        cleaned_df, report = detect_and_clean(
                            df,
                            file_basename=safe_name,
                            remove_duplicates=remove_duplicates,
                            handle_outliers=handle_outliers,
                            outlier_threshold=outlier_threshold
                        )
                        
                        # Display cleaned data info
                        st.subheader("✅ Cleaned Data")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Rows", cleaned_df.shape[0], delta=cleaned_df.shape[0] - df.shape[0])
                        col2.metric("Columns", cleaned_df.shape[1])
                        col3.metric("Missing Values", cleaned_df.isna().sum().sum())
                        col4.metric("Quality Score", f"{((1 - cleaned_df.isna().sum().sum() / cleaned_df.size) * 100):.1f}%")
                        
                        # Cleaning report
                        with st.expander("📋 Detailed Cleaning Report"):
                            st.write("**Operations Performed:**")
                            for op in report['operations']:
                                st.write(f"- {op}")
                            
                            if report['issues_found']:
                                st.warning("**Issues Encountered:**")
                                for issue in report['issues_found']:
                                    st.write(f"- {issue}")
                            
                            st.json(report['summary'])
                        
                        # Preview cleaned data
                        with st.expander("👀 Preview Cleaned Data (first 20 rows)"):
                            st.dataframe(cleaned_df.head(20))
                        
                        # Save to database
                        db_result = create_table_from_df(cleaned_df, table_name=safe_name, if_exists="replace")
                        
                        if db_result.get("status") == "ok":
                            st.success(f"💾 Saved to database as table: **{safe_name}**")
                        
                        # Save cleaned CSV
                        cleaned_path = os.path.join("data/cleaned", f"{safe_name}_cleaned.csv")
                        cleaned_df.to_csv(cleaned_path, index=False)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Cleaned CSV",
                            data=cleaned_df.to_csv(index=False).encode('utf-8'),
                            file_name=f"{safe_name}_cleaned.csv",
                            mime="text/csv"
                        )
                        
                        # Generate insights
                        st.subheader("📊 Auto-Generated Insights")
                        summary = summarize_table(cleaned_df)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Numeric Columns", len(summary['numeric']))
                        col2.metric("Categorical Columns", len(summary['categorical']))
                        col3.metric("Date Columns", len(summary['dates']))
                        
                        st.info(summary['text_summary'])
                        
                        st.success(f"""
                        🎉 **Success!** Your dataset has been processed.
                        
                        - ✅ Cleaned and saved to database
                        - ✅ Exported to: `{cleaned_path}`
                        - ✅ Ready to view in dashboard
                        
                        **Next Steps:**
                        1. Go to "View Existing Data" to analyze
                        2. Or upload another dataset!
                        """)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
                        st.exception(e)
    
    # ===================================================================
    # PAGE 2: VIEW EXISTING DATA
    # ===================================================================
    elif page == "View Existing Data":
        st.header("📊 View & Analyze Your Data")
        
        tables = list_tables()
        
        if not tables:
            st.info("📭 No datasets uploaded yet. Go to 'Upload New Dataset' to get started!")
            return
        
        # Table selector
        selected_table = st.selectbox(
            "Select a dataset:",
            tables,
            help="Choose from your uploaded datasets"
        )
        
        # Preview limit
        limit = st.slider("Rows to display", min_value=10, max_value=5000, value=100, step=10)
        
        try:
            df = read_table(selected_table, limit=limit)
            
            # Basic stats
            st.subheader(f"📈 Dataset: {selected_table}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", len(df))
            col2.metric("Columns", len(df.columns))
            col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            col4.metric("Missing Values", df.isna().sum().sum())
            
            # Data preview
            st.subheader("📄 Data Preview")
            st.dataframe(df, use_container_width=True)
            
            # Column info
            with st.expander("📋 Column Details"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Null': df.isna().sum(),
                    'Unique': df.nunique()
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            summary = summarize_table(df)
            
            # Numeric columns
            if summary['numeric']:
                st.write("**Numeric Columns:**")
                numeric_df = pd.DataFrame(summary['numeric']).T
                st.dataframe(numeric_df, use_container_width=True)
            
            # Categorical columns
            if summary['categorical']:
                st.write("**Categorical Columns (Top Values):**")
                for col, info in summary['categorical'].items():
                    with st.expander(f"🔤 {col}"):
                        st.write(f"Missing: {info['nulls']}")
                        st.bar_chart(info['top_values'])
            
            # Date columns
            if summary['dates']:
                st.write("**Date Columns:**")
                date_df = pd.DataFrame(summary['dates']).T
                st.dataframe(date_df, use_container_width=True)
            
            # Download
            st.download_button(
                label="📥 Download This Data",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"{selected_table}_export.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error loading table: {str(e)}")
    
    # ===================================================================
    # PAGE 3: SYSTEM INFO
    # ===================================================================
    else:
        st.header("ℹ️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Database")
            tables = list_tables()
            st.metric("Total Tables", len(tables))
            if tables:
                st.write("**Available Tables:**")
                for t in tables:
                    st.write(f"- {t}")
        
        with col2:
            st.subheader("📁 Files")
            raw_files = [f for f in os.listdir("data/raw") if f.endswith(('.csv', '.xlsx', '.xls'))]
            cleaned_files = [f for f in os.listdir("data/cleaned") if f.endswith('.csv')]
            
            st.metric("Raw Files", len(raw_files))
            st.metric("Cleaned Files", len(cleaned_files))
        
        st.subheader("🔧 Configuration")
        st.code(f"""
Database Path: app/data.db
Raw Files: data/raw/
Cleaned Files: data/cleaned/
        """)
        
        st.subheader("📚 Supported Features")
        features = {
            "File Types": "CSV, Excel (.xlsx, .xls)",
            "Auto-Detection": "Dates, Numbers, Currency, Percentages, Text, Booleans",
            "Cleaning": "Duplicates, Outliers, Missing Values, Whitespace, Type Conversion",
            "Export": "Cleaned CSV, Database Tables",
            "Analysis": "Summary Statistics, Insights, Visualizations"
        }
        for key, value in features.items():
            st.write(f"**{key}:** {value}")

if __name__ == "__main__":
    main()