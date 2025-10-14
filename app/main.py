# app/main.py
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import io
from .cleaning import clean_sales_df
from .db import init_db, engine, sales_table
from sqlalchemy import insert
from .insights import load_df, summarize_overall, format_text_summary
from .db import init_db, ensure_total_amount_column, engine, sales_table
app = FastAPI(title="AI Data Analytics - Phase1")

# Ensure DB exists
init_db()
ensure_total_amount_column()

@app.get("/")
def read_root():
    return {"msg": "AI Data Analytics - Phase 1 running. POST /upload-csv to send file."}

@app.get("/insights")
def get_insights(days: int = 7, mode: str = "recent"):
    df = load_df()
    summary = summarize_overall(df, days=days, mode=mode)
    text = format_text_summary(summary)
    return {"summary": summary, "text": text}



@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return {"status": "error", "detail": f"Failed to read CSV: {e}"}

    # Clean the dataframe
    cleaned = clean_sales_df(df)

    # Save to sqlite table
    rows = cleaned.to_dict(orient="records")
    with engine.connect() as conn:
        for r in rows:
            stmt = insert(sales_table).values(
                order_id = int(r.get("order_id")),
                date = str(r.get("date")),
                region = r.get("region"),
                product = r.get("product"),
                quantity = int(r.get("quantity") or 0),
                price = float(r.get("price") or 0.0),
                total_amount = float(r.get("total_amount") or 0.0)
            )
            conn.execute(stmt)
        conn.commit()

    return {"status": "success", "rows_saved": len(rows)}
