# app/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import io
import pandas as pd

from .db import init_db, create_table_from_df, list_tables, sanitize_table_name
from .cleaning import detect_and_clean
from .insights import load_df, summarize_overall, format_text_summary

# -----------------------------------------------------------
# ✅ Initialize the FastAPI app
# -----------------------------------------------------------
app = FastAPI(title="AI Data Analytics - Phase 1", version="0.2.0")

# Allow Streamlit or other UIs to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the database is ready
init_db()


# -----------------------------------------------------------
# ✅ Root endpoint - health check
# -----------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "msg": "AI Data Analytics - Dynamic Phase running. POST /upload-csv to send file."
    }


# -----------------------------------------------------------
# ✅ Upload CSV and auto-clean dynamically
# -----------------------------------------------------------
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), table_name: str | None = Form(None)):
    """Upload a CSV and store it dynamically in the DB after cleaning."""
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        try:
            df = pd.read_csv(io.BytesIO(contents), engine="python")
        except Exception as e2:
            return {"status": "error", "detail": f"Failed to read CSV: {e}; {e2}"}

    # Sanitize and name the table
    original_name = table_name or (file.filename or "uploaded").rsplit(".", 1)[0]
    safe_name = sanitize_table_name(original_name)

    # Dynamic cleaning (auto-detects numeric, text, date cols)
    cleaned_df = detect_and_clean(df, file_basename=safe_name)

    # Save to DB
    result = create_table_from_df(cleaned_df, table_name=safe_name, if_exists="replace")

    if result.get("status") != "ok":
        return result

    # Return sample preview
    sample = cleaned_df.head(5).to_dict(orient="records")
    return {
        "status": "ok",
        "table": result.get("table"),
        "rows_saved": result.get("rows"),
        "columns": cleaned_df.columns.tolist(),
        "sample": sample,
    }


# -----------------------------------------------------------
# ✅ Insights endpoint
# -----------------------------------------------------------
@app.get("/insights")
def get_insights(days: int = 7, mode: str = "recent"):
    """
    Summarize recent trends from the last N days (if date column found).
    Works dynamically for any dataset.
    """
    df = load_df()
    if df is None or df.empty:
        return {"summary": {}, "text": "No data available for analysis."}

    summary = summarize_overall(df, days=days, mode=mode)
    text = format_text_summary(summary)
    return {"summary": summary, "text": text}
