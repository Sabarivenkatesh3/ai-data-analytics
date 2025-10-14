# app/insights.py
import pandas as pd
import numpy as np
from typing import Dict, Any

def summarize_table(df: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
    """
    Return a generic summary:
    - row_count
    - numeric column stats (sum, mean, min, max, nulls)
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
        # show top numeric by sum
        sums = {c: num_stats[c]["sum"] for c in numeric_cols}
        top_num = sorted(sums.items(), key=lambda x: -abs(x[1]))[0]
        text_lines.append(f"Top numeric column by absolute sum: {top_num[0]} = {top_num[1]:.2f}")
    if object_cols:
        # show most frequent categorical across all cat cols
        # pick the column with largest unique values? instead show a sample top
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
