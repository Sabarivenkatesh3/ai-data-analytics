# app/cleaning.py
"""
Universal Data Cleaning Module
Handles ANY dataset with intelligent detection and cleaning
"""

import pandas as pd
import numpy as np
import re
from dateutil.parser import parse as date_parse
from typing import Tuple, Dict, Any

# -----------------------------------------------------------
# 🔧 UTILITY FUNCTIONS
# -----------------------------------------------------------

def _normalize_colname(c: str) -> str:
    """Convert column names to snake_case and remove special characters"""
    c = str(c).strip().lower()
    # Replace spaces and hyphens with underscores
    c = c.replace(" ", "_").replace("-", "_")
    # Remove special characters except underscores
    c = re.sub(r"[^a-z0-9_]", "", c)
    # Remove consecutive underscores
    c = re.sub(r"_+", "_", c)
    # Remove leading/trailing underscores
    c = c.strip("_")
    return c or "unnamed_col"


def _is_likely_date_series(s: pd.Series) -> bool:
    """
    Intelligent date detection using multiple heuristics
    """
    if s.empty or s.isna().all():
        return False
    
    name = str(s.name).lower()
    
    # Check if column name contains date-related keywords
    date_keywords = ["date", "time", "timestamp", "day", "month", "year", "dob", "created", "updated", "birth"]
    if any(k in name for k in date_keywords):
        return True
    
    # Sample non-null values
    sample = s.dropna().astype(str).head(30)
    if sample.empty:
        return False
    
    # Try parsing a sample
    parsed_count = 0
    for val in sample:
        try:
            parsed = date_parse(val, fuzzy=False)
            # Check if it looks like a real date (not just numbers)
            if 1900 <= parsed.year <= 2100:
                parsed_count += 1
        except:
            pass
    
    # If 60%+ are parseable dates, treat as date column
    return (parsed_count / len(sample)) >= 0.6


def _detect_currency(s: pd.Series) -> bool:
    """Detect if column contains currency values (e.g., $1,234.56)"""
    sample = s.dropna().astype(str).head(20)
    if sample.empty:
        return False
    
    currency_pattern = r'[$€£¥₹]\s*[\d,.]+'
    matches = sample.str.contains(currency_pattern, regex=True, na=False).sum()
    return matches / len(sample) >= 0.5


def _detect_percentage(s: pd.Series) -> bool:
    """Detect if column contains percentages (e.g., 45.5%)"""
    sample = s.dropna().astype(str).head(20)
    if sample.empty:
        return False
    
    pct_pattern = r'^\s*\d+\.?\d*\s*%\s*$'
    matches = sample.str.contains(pct_pattern, regex=True, na=False).sum()
    return matches / len(sample) >= 0.5


def _clean_currency(s: pd.Series) -> pd.Series:
    """Remove currency symbols and convert to float"""
    return s.astype(str).str.replace(r'[$€£¥₹,]', '', regex=True).astype(float)


def _clean_percentage(s: pd.Series) -> pd.Series:
    """Remove % symbol and convert to decimal (45% -> 0.45)"""
    return s.astype(str).str.replace('%', '', regex=True).astype(float) / 100


def _remove_outliers_iqr(s: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Remove outliers using IQR method
    Values outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are set to NaN
    """
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Replace outliers with NaN (to be filled later)
    return s.where((s >= lower_bound) & (s <= upper_bound), np.nan)


# -----------------------------------------------------------
# 🧹 MAIN CLEANING FUNCTION
# -----------------------------------------------------------

def detect_and_clean(
    df: pd.DataFrame,
    file_basename: str = "uploaded",
    remove_duplicates: bool = True,
    handle_outliers: bool = True,
    outlier_threshold: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Universal data cleaning pipeline that adapts to ANY dataset.
    
    Steps:
    1. Normalize column names
    2. Remove completely empty rows/columns
    3. Trim whitespace in text columns
    4. Detect and convert date columns
    5. Detect and convert currency/percentage columns
    6. Detect and convert numeric columns
    7. Handle outliers (optional)
    8. Remove duplicates (optional)
    9. Fill missing values intelligently
    10. Add metadata (row_id if missing)
    
    Returns:
        (cleaned_df, cleaning_report)
    """
    
    report = {
        "original_shape": df.shape,
        "operations": [],
        "columns_cleaned": {},
        "issues_found": []
    }
    
    df = df.copy()
    
    # -----------------------------------------------------------
    # 1. Normalize column names
    # -----------------------------------------------------------
    original_cols = df.columns.tolist()
    df.columns = [_normalize_colname(c) for c in df.columns]
    
    # Handle duplicate column names
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    
    report["operations"].append(f"✅ Normalized {len(df.columns)} column names")
    
    # -----------------------------------------------------------
    # 2. Remove completely empty rows and columns
    # -----------------------------------------------------------
    initial_rows = len(df)
    df = df.dropna(how='all', axis=0)  # Remove rows where ALL values are NaN
    df = df.dropna(how='all', axis=1)  # Remove columns where ALL values are NaN
    
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        report["operations"].append(f"🗑️ Removed {rows_removed} completely empty rows")
    
    # -----------------------------------------------------------
    # 3. Trim whitespace in string columns
    # -----------------------------------------------------------
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        try:
            df[col] = df[col].astype(str).str.strip()
            # Replace string representations of null
            df.loc[df[col].isin(["nan", "None", "NaN", "NULL", "null", ""]), col] = np.nan
        except:
            pass
    
    report["operations"].append(f"✨ Trimmed whitespace in {len(text_cols)} text columns")
    
    # -----------------------------------------------------------
    # 4. Detect and convert date columns
    # -----------------------------------------------------------
    date_cols = []
    for col in df.columns:
        if _is_likely_date_series(df[col]):
            try:
                for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M']:
                    try:
                        df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                        if df[col].notna().any():
                            break
                    except:
                        continue
                else:
                    # If no format works, let pandas infer (without deprecated param)
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

                date_cols.append(col)
                report["columns_cleaned"][col] = "date"
            except Exception as e:
                report["issues_found"].append(f"⚠️ Could not parse dates in '{col}': {str(e)}")
    
    if date_cols:
        report["operations"].append(f"📅 Detected and converted {len(date_cols)} date columns: {date_cols}")
    
    # -----------------------------------------------------------
    # 5. Detect and convert currency columns
    # -----------------------------------------------------------
    currency_cols = []
    for col in df.columns:
        if col in date_cols:
            continue
        if _detect_currency(df[col]):
            try:
                df[col] = _clean_currency(df[col])
                currency_cols.append(col)
                report["columns_cleaned"][col] = "currency"
            except Exception as e:
                report["issues_found"].append(f"⚠️ Could not clean currency in '{col}': {str(e)}")
    
    if currency_cols:
        report["operations"].append(f"💰 Detected and cleaned {len(currency_cols)} currency columns: {currency_cols}")
    
    # -----------------------------------------------------------
    # 6. Detect and convert percentage columns
    # -----------------------------------------------------------
    pct_cols = []
    for col in df.columns:
        if col in date_cols + currency_cols:
            continue
        if _detect_percentage(df[col]):
            try:
                df[col] = _clean_percentage(df[col])
                pct_cols.append(col)
                report["columns_cleaned"][col] = "percentage"
            except Exception as e:
                report["issues_found"].append(f"⚠️ Could not clean percentage in '{col}': {str(e)}")
    
    if pct_cols:
        report["operations"].append(f"📊 Detected and cleaned {len(pct_cols)} percentage columns: {pct_cols}")
    
    # -----------------------------------------------------------
    # 7. Detect numeric columns
    # -----------------------------------------------------------
    numeric_cols = []
    for col in df.columns:
        if col in date_cols + currency_cols + pct_cols:
            continue
        
        # Try converting to numeric
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_null = coerced.notna().sum()
        total = len(coerced)
        
        if total == 0:
            continue
        
        # If 60%+ are convertible, treat as numeric
        if non_null / total >= 0.6:
            df[col] = coerced
            numeric_cols.append(col)
            report["columns_cleaned"][col] = "numeric"
    
    if numeric_cols:
        report["operations"].append(f"🔢 Detected {len(numeric_cols)} numeric columns: {numeric_cols}")
    
    # -----------------------------------------------------------
    # 8. Handle outliers in numeric columns (optional)
    # -----------------------------------------------------------
    if handle_outliers and numeric_cols:
        outliers_removed = 0
        for col in numeric_cols:
            before = df[col].isna().sum()
            df[col] = _remove_outliers_iqr(df[col], multiplier=outlier_threshold)
            after = df[col].isna().sum()
            outliers_removed += (after - before)
        
        if outliers_removed > 0:
            report["operations"].append(f"🎯 Removed {outliers_removed} outliers using IQR method")
    
    # -----------------------------------------------------------
    # 9. Remove duplicates (optional)
    # -----------------------------------------------------------
    if remove_duplicates:
        before_dedup = len(df)
        df = df.drop_duplicates()
        duplicates_removed = before_dedup - len(df)
        
        if duplicates_removed > 0:
            report["operations"].append(f"🔄 Removed {duplicates_removed} duplicate rows")
    
    # -----------------------------------------------------------
    # 10. Fill missing values intelligently
    # -----------------------------------------------------------
    all_numeric = numeric_cols + currency_cols + pct_cols
    
    # For numeric columns: fill with median
    for col in all_numeric:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            median_val = df[col].median(skipna=True)
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
            report["operations"].append(f"🔧 Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})")
    
    # For categorical/text columns: fill with mode or 'Unknown'
    cat_cols = [c for c in df.columns if c not in all_numeric + date_cols]
    for col in cat_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            mode_val = None
            try:
                mode_series = df[col].mode()
                if not mode_series.empty:
                    mode_val = mode_series.iloc[0]
            except:
                pass
            
            fill_val = mode_val if mode_val and str(mode_val).strip() != "" else "Unknown"
            df[col] = df[col].fillna(fill_val)
            report["operations"].append(f"🔧 Filled {missing_count} missing values in '{col}' with '{fill_val}'")
    
    # For date columns: leave NaT as is (don't fill arbitrarily)
    for col in date_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            report["operations"].append(f"📅 '{col}' has {missing_count} missing dates (left as NaT)")
    
    # -----------------------------------------------------------
    # 11. Add row_id if not present
    # -----------------------------------------------------------
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(1, len(df) + 1))
        report["operations"].append("🆔 Added 'row_id' column")
    
    # -----------------------------------------------------------
    # 12. Convert boolean-like strings to actual booleans
    # -----------------------------------------------------------
    for col in cat_cols:
        unique_vals = df[col].dropna().astype(str).str.lower().unique()
        bool_vals = {"true", "false", "yes", "no", "y", "n", "1", "0"}
        
        if len(unique_vals) <= 10 and any(v in bool_vals for v in unique_vals):
            try:
                bool_map = {
                    "true": True, "false": False,
                    "yes": True, "no": False,
                    "y": True, "n": False,
                    "1": True, "0": False
                }
                df[col] = df[col].astype(str).str.lower().map(bool_map)
                report["operations"].append(f"✔️ Converted '{col}' to boolean")
            except:
                pass
    
    # -----------------------------------------------------------
    # Final report summary
    # -----------------------------------------------------------
    report["final_shape"] = df.shape
    report["summary"] = {
        "rows_cleaned": df.shape[0],
        "columns_cleaned": df.shape[1],
        "numeric_columns": len(all_numeric),
        "categorical_columns": len(cat_cols),
        "date_columns": len(date_cols),
        "total_operations": len(report["operations"])
    }
    
    print("\n" + "="*60)
    print("🧹 CLEANING REPORT")
    print("="*60)
    for op in report["operations"]:
        print(op)
    
    if report["issues_found"]:
        print("\n⚠️ Issues encountered:")
        for issue in report["issues_found"]:
            print(issue)
    
    print("="*60 + "\n")
    
    return df, report


# -----------------------------------------------------------
# 📊 GET CLEANING REPORT (for API responses)
# -----------------------------------------------------------

def get_cleaning_report(df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a detailed comparison report between original and cleaned data
    """
    return {
        "original": {
            "rows": df_original.shape[0],
            "columns": df_original.shape[1],
            "missing_values": int(df_original.isna().sum().sum())
        },
        "cleaned": {
            "rows": df_cleaned.shape[0],
            "columns": df_cleaned.shape[1],
            "missing_values": int(df_cleaned.isna().sum().sum())
        },
        "improvements": {
            "rows_removed": df_original.shape[0] - df_cleaned.shape[0],
            "missing_values_filled": int(df_original.isna().sum().sum() - df_cleaned.isna().sum().sum())
        }
    }