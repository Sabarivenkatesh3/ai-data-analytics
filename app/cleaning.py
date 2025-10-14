# app/cleaning.py
import pandas as pd
import numpy as np
from dateutil.parser import parse as date_parse

def _normalize_colname(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")

def _is_likely_date_series(s: pd.Series) -> bool:
    # Heuristic: name contains date/time words OR many parseable values
    name = s.name.lower() if isinstance(s.name, str) else ""
    if any(k in name for k in ["date", "time", "timestamp", "day", "dob"]):
        return True
    # Try to parse a sample of non-null values
    sample = s.dropna().astype(str).head(20)
    if sample.empty:
        return False
    parsed = 0
    for v in sample:
        try:
            _ = date_parse(v, fuzzy=False)
            parsed += 1
        except Exception:
            pass
    return parsed / len(sample) >= 0.6

def detect_and_clean(df: pd.DataFrame, file_basename: str = "uploaded") -> pd.DataFrame:
    """
    Generic cleaning pipeline:
    - normalize column names
    - detect date columns and convert
    - convert numeric-like columns
    - fill missing values (median for numeric, mode/'Unknown' for categorical)
    - add a row_id column if missing
    """
    # 1. Normalize column names
    df = df.copy()
    df.columns = [_normalize_colname(c) for c in df.columns]

    # 2. Trim whitespace in string columns
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            df[c] = df[c].astype(str).str.strip()
            # replace 'nan' strings made by cast to str
            df.loc[df[c].isin(["nan", "None", "NaN", "nan.0"]), c] = None
        except Exception:
            pass

    # 3. Detect date columns
    date_cols = []
    for c in df.columns:
        if _is_likely_date_series(df[c]):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                date_cols.append(c)
            except Exception:
                pass

    # 4. Detect numeric columns
    # Use pandas to_numeric with coercion, but not for date columns
    numeric_cols = []
    for c in df.columns:
        if c in date_cols:
            continue
        # Try converting to numeric, if >60% convertable mark as numeric
        coerced = pd.to_numeric(df[c], errors="coerce")
        non_null = coerced.notna().sum()
        total = len(coerced)
        if total == 0:
            continue
        if non_null / total >= 0.6:
            df[c] = coerced
            numeric_cols.append(c)

    # 5. For numeric cols: fill missing with median
    for c in numeric_cols:
        try:
            median = df[c].median(skipna=True)
            if pd.isna(median):
                median = 0
            df[c] = df[c].fillna(median)
        except Exception:
            df[c] = df[c].fillna(0)

    # 6. For non-numeric, non-date columns treat as categorical/text
    cat_cols = [c for c in df.columns if c not in numeric_cols + date_cols]
    for c in cat_cols:
        # fill with 'Unknown' if many missing
        if df[c].isna().sum() > 0:
            mode = None
            try:
                mode = df[c].mode().iloc[0]
            except Exception:
                mode = None
            fill_val = mode if mode and str(mode).strip() != "" else "Unknown"
            df[c] = df[c].fillna(fill_val)

    # 7. Add a row_id if not present
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(1, len(df) + 1))

    # 8. Final type enforcement (optional)
    # Convert boolean-like strings to bool
    for c in cat_cols:
        if df[c].dropna().astype(str).str.lower().isin(["true","false","yes","no","y","n"]).any():
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False, "yes": True, "no": False, "y": True, "n": False})

    return df
