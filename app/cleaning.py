# app/cleaning.py
"""
Universal Data Cleaning Module
ENHANCED FOR AMAZON PRODUCT DATA
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
    c = c.replace(" ", "_").replace("-", "_").replace("/", "_")
    c = re.sub(r"[^a-z0-9_]", "", c)
    c = re.sub(r"_+", "_", c)
    c = c.strip("_")
    return c or "unnamed_col"


def _clean_rating(value):
    """Extract numeric rating from text like '4.6 out of 5 stars'"""
    if pd.isna(value):
        return np.nan
    text = str(value)
    match = re.search(r'(\d+\.?\d*)\s*out of', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Try to extract any decimal number
    match = re.search(r'(\d+\.?\d*)', text)
    return float(match.group(1)) if match else np.nan


def _clean_review_count(value):
    """Clean review counts like '2,457' or '35882'"""
    if pd.isna(value):
        return np.nan
    text = str(value).replace(',', '')
    match = re.search(r'(\d+)', text)
    return int(match.group(1)) if match else np.nan


def _clean_bought_last_month(value):
    """Parse '6K+ bought in past month' to number"""
    if pd.isna(value):
        return np.nan
    
    text = str(value).upper()
    
    # Extract number with K/M notation
    match = re.search(r'(\d+\.?\d*)\s*([KM])\+?', text)
    if match:
        num = float(match.group(1))
        multiplier = 1000 if match.group(2) == 'K' else 1000000
        return int(num * multiplier)
    
    # Extract plain number
    match = re.search(r'(\d+)', text)
    return int(match.group(1)) if match else np.nan


def _clean_price(value):
    """Clean price strings like '$159.00' or '89.68'"""
    if pd.isna(value) or str(value).strip() == '':
        return np.nan
    
    text = str(value)
    
    # Skip non-price text
    if 'basic variant' in text.lower() or 'no discount' in text.lower():
        return np.nan
    
    # Remove currency symbols and clean
    text = re.sub(r'[$€£¥₹,]', '', text)
    match = re.search(r'(\d+\.?\d*)', text)
    
    if match:
        try:
            return float(match.group(1))
        except:
            return np.nan
    return np.nan


def _clean_variant_price(value):
    """Extract price from 'basic variant price: $162.24' or similar"""
    if pd.isna(value):
        return np.nan
    
    text = str(value)
    
    # Skip unwanted patterns
    if 'nan' in text.lower() or 'no discount' in text.lower():
        return np.nan
    
    # Look for price after colon
    match = re.search(r':\s*\$?(\d+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1))
        except:
            return np.nan
    
    # Try to find any price
    return _clean_price(value)


def _clean_badge_column(value):
    """Clean badge columns - convert 'No Badge' to empty/NaN"""
    if pd.isna(value):
        return np.nan
    
    text = str(value).strip()
    if text.lower() in ['no badge', 'nan', 'none', '']:
        return np.nan
    
    return text


def _clean_boolean_text(value):
    """Convert text like 'Sponsored', 'Organic' to boolean or category"""
    if pd.isna(value):
        return 'Unknown'
    
    text = str(value).strip()
    if text.lower() in ['nan', 'none', '']:
        return 'Unknown'
    
    return text


def _clean_coupon_text(value):
    """Clean coupon text"""
    if pd.isna(value):
        return 'No Coupon'
    
    text = str(value).strip()
    if 'no coupon' in text.lower():
        return 'No Coupon'
    
    # Extract coupon percentage
    match = re.search(r'(\d+)%', text)
    if match:
        return f"Save {match.group(1)}%"
    
    return text if text else 'No Coupon'


def _is_likely_date_series(s: pd.Series) -> bool:
    """Intelligent date detection"""
    if s.empty or s.isna().all():
        return False
    
    name = str(s.name).lower()
    date_keywords = ["date", "time", "timestamp", "collected", "created", "updated"]
    if any(k in name for k in date_keywords):
        return True
    
    sample = s.dropna().astype(str).head(30)
    if sample.empty:
        return False
    
    parsed_count = 0
    for val in sample:
        try:
            parsed = date_parse(val, fuzzy=False)
            if 1900 <= parsed.year <= 2100:
                parsed_count += 1
        except:
            pass
    
    return (parsed_count / len(sample)) >= 0.6


# -----------------------------------------------------------
# 🧹 MAIN CLEANING FUNCTION
# -----------------------------------------------------------

def detect_and_clean(
    df: pd.DataFrame,
    file_basename: str = "uploaded",
    remove_duplicates: bool = True,
    handle_outliers: bool = False,
    outlier_threshold: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Universal data cleaning pipeline with special handling for Amazon data
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
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)
    
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
            df.loc[df[col].isin(["nan", "None", "NaN", "NULL", "null", ""]), col] = np.nan
        except:
            pass
    
    report["operations"].append(f"✨ Trimmed whitespace in {len(text_cols)} text columns")
    
    # -----------------------------------------------------------
    # 4. AMAZON-SPECIFIC CLEANING
    # -----------------------------------------------------------
    
    # Clean rating column
    rating_cols = [c for c in df.columns if 'rating' in c]
    for col in rating_cols:
        try:
            df[col] = df[col].apply(_clean_rating)
            report["operations"].append(f"⭐ Cleaned rating column: {col}")
            report["columns_cleaned"][col] = "rating"
        except Exception as e:
            report["issues_found"].append(f"⚠️ Could not clean rating '{col}': {str(e)}")
    
    # Clean review count
    review_cols = [c for c in df.columns if 'review' in c]
    for col in review_cols:
        try:
            df[col] = df[col].apply(_clean_review_count)
            report["operations"].append(f"📊 Cleaned review count: {col}")
            report["columns_cleaned"][col] = "review_count"
        except Exception as e:
            report["issues_found"].append(f"⚠️ Could not clean reviews '{col}': {str(e)}")
    
    # Clean bought_in_last_month
    bought_cols = [c for c in df.columns if 'bought' in c]
    for col in bought_cols:
        try:
            df[col] = df[col].apply(_clean_bought_last_month)
            report["operations"].append(f"🛒 Cleaned bought count: {col}")
            report["columns_cleaned"][col] = "bought_count"
        except Exception as e:
            report["issues_found"].append(f"⚠️ Could not clean bought '{col}': {str(e)}")
    
    # Clean price columns
    price_cols = [c for c in df.columns if 'price' in c]
    for col in price_cols:
        try:
            if 'variant' in col:
                df[col] = df[col].apply(_clean_variant_price)
            else:
                df[col] = df[col].apply(_clean_price)
            report["operations"].append(f"💰 Cleaned price column: {col}")
            report["columns_cleaned"][col] = "price"
        except Exception as e:
            report["issues_found"].append(f"⚠️ Could not clean price '{col}': {str(e)}")
    
    # Clean badge columns
    badge_cols = [c for c in df.columns if 'badge' in c or 'seller' in c]
    for col in badge_cols:
        try:
            df[col] = df[col].apply(_clean_badge_column)
            report["operations"].append(f"🏷️ Cleaned badge column: {col}")
        except Exception as e:
            pass
    
    # Clean sponsored/organic columns
    sponsor_cols = [c for c in df.columns if 'sponsored' in c]
    for col in sponsor_cols:
        try:
            df[col] = df[col].apply(_clean_boolean_text)
            report["operations"].append(f"📢 Cleaned sponsored column: {col}")
        except Exception as e:
            pass
    
    # Clean coupon columns
    coupon_cols = [c for c in df.columns if 'coupon' in c]
    for col in coupon_cols:
        try:
            df[col] = df[col].apply(_clean_coupon_text)
            report["operations"].append(f"🎟️ Cleaned coupon column: {col}")
        except Exception as e:
            pass
    
    # -----------------------------------------------------------
    # 5. Detect and convert date columns
    # -----------------------------------------------------------
    date_cols = []
    for col in df.columns:
        if col in report["columns_cleaned"]:
            continue
            
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
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

                date_cols.append(col)
                report["columns_cleaned"][col] = "date"
            except Exception as e:
                report["issues_found"].append(f"⚠️ Could not parse dates in '{col}': {str(e)}")
    
    if date_cols:
        report["operations"].append(f"📅 Detected and converted {len(date_cols)} date columns: {date_cols}")
    
    # -----------------------------------------------------------
    # 6. Remove duplicates (optional)
    # -----------------------------------------------------------
    if remove_duplicates:
        before_dedup = len(df)
        df = df.drop_duplicates()
        duplicates_removed = before_dedup - len(df)
        
        if duplicates_removed > 0:
            report["operations"].append(f"🔄 Removed {duplicates_removed} duplicate rows")
    
    # -----------------------------------------------------------
    # 7. Add row_id if not present
    # -----------------------------------------------------------
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(1, len(df) + 1))
        report["operations"].append("🆔 Added 'row_id' column")
    
    # -----------------------------------------------------------
    # Final report summary
    # -----------------------------------------------------------
    report["final_shape"] = df.shape
    report["summary"] = {
        "rows_cleaned": df.shape[0],
        "columns_cleaned": df.shape[1],
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


def get_cleaning_report(df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> Dict[str, Any]:
    """Generate a detailed comparison report"""
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