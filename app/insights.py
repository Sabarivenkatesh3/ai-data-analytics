# app/insights.py
import os
import pandas as pd
import numpy as np
from typing import Dict, Any

DATA_DIR = "data"  # Folder where cleaned datasets are saved

# -----------------------------------------------------------
# ✅ Load DataFrame dynamically
# -----------------------------------------------------------
def load_df(latest: bool = True):
    """
    Load the most recently cleaned CSV file from the /data folder.
    This makes it dynamic — works with any new dataset.
    """
    if not os.path.exists(DATA_DIR):
        return None

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not files:
        return None

    # Get most recent file
    if latest:
        files.sort(key=lambda f: os.path.getmtime(os.path.join(DATA_DIR, f)), reverse=True)
        latest_file = files[0]
    else:
        latest_file = files[0]

    file_path = os.path.join(DATA_DIR, latest_file)
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return None


# -----------------------------------------------------------
# ✅ Summarize any table dynamically
# -----------------------------------------------------------
def summarize_overall(df: pd.DataFrame, days: int = 7, mode: str = "recent") -> Dict[str, Any]:
    """
    General dynamic summary (works for any dataset type).
    Uses summarize_table internally.
    """
    base_summary = summarize_table(df)
    base_summary["mode"] = mode
    base_summary["days_used"] = days
    return base_summary


# -----------------------------------------------------------
# ✅ Convert summary dict into readable text
# -----------------------------------------------------------
def format_text_summary(summary: Dict[str, Any]) -> str:
    """
    Create a human-readable text summary for UI or API response.
    """
    if not summary or "text_summary" not in summary:
        return "No data available or failed to generate summary."

    return (
        f"🔍 Data Summary:\n\n"
        f"{summary['text_summary']}\n\n"
        f"📊 Rows: {summary.get('rows', 0)}\n"
        f"🔢 Numeric columns: {len(summary.get('numeric', {}))}\n"
        f"🔠 Categorical columns: {len(summary.get('categorical', {}))}\n"
        f"📅 Date columns: {len(summary.get('dates', {}))}\n"
    )


# -----------------------------------------------------------
# ✅ Core summary logic (already present)
# -----------------------------------------------------------
def summarize_table(df: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
    """
    Return a generic summary:
    - row_count
    - numeric column stats
    - categorical top values and counts
    - date min/max if present
    """
    res = {}
    res["rows"] = int(len(df))

    # identify types using pandas dtypes
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
    object_cols = [c for c in df.columns if c not in numeric_cols + datetime_cols]

    # Numeric stats
    num_stats = {}
    for c in numeric_cols:
        s = df[c]
        num_stats[c] = {
            "sum": float(s.sum()),
            "mean": float(s.mean()) if not s.empty else None,
            "min": float(s.min()) if not s.empty else None,
            "max": float(s.max()) if not s.empty else None,
            "nulls": int(s.isna().sum())
        }
    res["numeric"] = num_stats

    # Categorical top values
    cat_stats = {}
    for c in object_cols:
        s = df[c].astype(str)
        top = s.value_counts(dropna=True).head(top_n).to_dict()
        cat_stats[c] = {"top_values": top, "nulls": int(df[c].isna().sum())}
    res["categorical"] = cat_stats

    # Dates
    date_stats = {}
    for c in datetime_cols:
        s = df[c]
        date_stats[c] = {
            "min": str(s.min()) if not s.dropna().empty else None,
            "max": str(s.max()) if not s.dropna().empty else None,
            "nulls": int(s.isna().sum())
        }
    res["dates"] = date_stats

    # Make a simple text summary
    text_lines = []
    text_lines.append(f"Table has {res['rows']} rows.")
    if numeric_cols:
        sums = {c: num_stats[c]["sum"] for c in numeric_cols}
        top_num = sorted(sums.items(), key=lambda x: -abs(x[1]))[0]
        text_lines.append(f"Top numeric column by absolute sum: {top_num[0]} = {top_num[1]:.2f}")
    if object_cols:
        sample_col = object_cols[0]
        tv = cat_stats[sample_col]["top_values"]
        if tv:
            first_val, cnt = list(tv.items())[0]
            text_lines.append(f"Sample categorical: '{sample_col}' top value = '{first_val}' ({cnt} occurrences)")
    if datetime_cols:
        c = datetime_cols[0]
        date_min = date_stats[c]["min"]
        date_max = date_stats[c]["max"]
        text_lines.append(f"Date column '{c}' ranges from {date_min} to {date_max}.")

    res["text_summary"] = "\n".join(text_lines)
    return res
