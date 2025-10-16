# app/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import io
import os
import pandas as pd
from pathlib import Path

from .db import init_db, create_table_from_df, list_tables, sanitize_table_name, read_table
from .cleaning import detect_and_clean, get_cleaning_report
from .insights import load_df, summarize_overall, format_text_summary

# -----------------------------------------------------------
# ✅ Configuration
# -----------------------------------------------------------
DATA_DIR = "data"
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# -----------------------------------------------------------
# ✅ Initialize the FastAPI app
# -----------------------------------------------------------
app = FastAPI(
    title="AI Data Analytics - Universal Data Processor",
    version="1.0.0",
    description="Upload any CSV/Excel file, get it cleaned automatically with AI-powered insights"
)

# Allow CORS for frontend access
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
        "status": "running",
        "message": "AI Data Analytics API - Universal Data Processor",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload-file - Upload CSV/Excel files",
            "list": "GET /tables - List all processed tables",
            "download": "GET /download/{table_name} - Download cleaned file",
            "insights": "GET /insights/{table_name} - Get data insights",
            "raw_file": "GET /raw/{filename} - Download original raw file"
        }
    }


# -----------------------------------------------------------
# ✅ Upload CSV/Excel and auto-clean dynamically
# -----------------------------------------------------------
@app.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    table_name: str | None = Form(None)
):
    """
    Upload a CSV or Excel file. The system will:
    1. Save the raw file
    2. Detect data types intelligently
    3. Clean the data automatically
    4. Save to database
    5. Generate a cleaning report
    6. Export cleaned CSV
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Read file contents
    contents = await file.read()
    filename = file.filename
    
    # Save raw file for reference
    raw_path = os.path.join(RAW_DIR, filename)
    with open(raw_path, "wb") as f:
        f.write(contents)
    
    # -----------------------------------------------------------
    # ✅ Step 1: Read file based on extension
    # -----------------------------------------------------------
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
            file_type = "Excel"
        elif filename.endswith('.csv'):
            # Try different encodings
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
                except:
                    df = pd.read_csv(io.BytesIO(contents), encoding='iso-8859-1')
            file_type = "CSV"
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV or Excel (.xlsx, .xls) files only."
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # -----------------------------------------------------------
    # ✅ Step 2: Sanitize table name
    # -----------------------------------------------------------
    original_name = table_name or filename.rsplit(".", 1)[0]
    safe_name = sanitize_table_name(original_name)
    
    # -----------------------------------------------------------
    # ✅ Step 3: Intelligent cleaning (detects all data types)
    # -----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"🔍 Processing: {filename}")
    print(f"📊 Original shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"{'='*60}\n")
    
    cleaned_df, cleaning_report = detect_and_clean(df, file_basename=safe_name)
    
    print(f"\n{'='*60}")
    print(f"✅ Cleaned shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
    print(f"{'='*60}\n")
    
    # -----------------------------------------------------------
    # ✅ Step 4: Save cleaned data to database
    # -----------------------------------------------------------
    db_result = create_table_from_df(cleaned_df, table_name=safe_name, if_exists="replace")
    
    if db_result.get("status") != "ok":
        raise HTTPException(status_code=500, detail=f"Database error: {db_result.get('detail')}")
    
    # -----------------------------------------------------------
    # ✅ Step 5: Export cleaned CSV
    # -----------------------------------------------------------
    cleaned_path = os.path.join(CLEANED_DIR, f"{safe_name}_cleaned.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    
    # -----------------------------------------------------------
    # ✅ Step 6: Generate preview
    # -----------------------------------------------------------
    sample_preview = cleaned_df.head(10).to_dict(orient="records")
    
    # -----------------------------------------------------------
    # ✅ Step 7: Return comprehensive response
    # -----------------------------------------------------------
    return {
        "status": "success",
        "message": f"File processed successfully: {filename}",
        "file_info": {
            "original_filename": filename,
            "file_type": file_type,
            "table_name": safe_name,
            "raw_file_path": raw_path,
            "cleaned_file_path": cleaned_path
        },
        "data_info": {
            "original_rows": df.shape[0],
            "original_columns": df.shape[1],
            "cleaned_rows": cleaned_df.shape[0],
            "cleaned_columns": cleaned_df.shape[1],
            "columns": cleaned_df.columns.tolist(),
            "rows_removed": df.shape[0] - cleaned_df.shape[0]
        },
        "cleaning_report": cleaning_report,
        "preview": sample_preview,
        "download_links": {
            "cleaned_csv": f"/download/{safe_name}",
            "raw_file": f"/raw/{filename}"
        }
    }


# -----------------------------------------------------------
# ✅ List all processed tables
# -----------------------------------------------------------
@app.get("/tables")
def get_tables():
    """List all tables in the database with metadata"""
    tables = list_tables()
    
    table_info = []
    for table in tables:
        try:
            df = read_table(table, limit=1)
            table_info.append({
                "table_name": table,
                "columns": df.columns.tolist(),
                "download_link": f"/download/{table}"
            })
        except:
            table_info.append({
                "table_name": table,
                "error": "Could not read table metadata"
            })
    
    return {
        "total_tables": len(tables),
        "tables": table_info
    }


# -----------------------------------------------------------
# ✅ Download cleaned CSV
# -----------------------------------------------------------
@app.get("/download/{table_name}")
def download_cleaned(table_name: str):
    """Download the cleaned CSV file"""
    safe_name = sanitize_table_name(table_name)
    file_path = os.path.join(CLEANED_DIR, f"{safe_name}_cleaned.csv")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Cleaned file not found: {safe_name}")
    
    return FileResponse(
        file_path,
        filename=f"{safe_name}_cleaned.csv",
        media_type="text/csv"
    )


# -----------------------------------------------------------
# ✅ Download original raw file
# -----------------------------------------------------------
@app.get("/raw/{filename}")
def download_raw(filename: str):
    """Download the original uploaded raw file"""
    file_path = os.path.join(RAW_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Raw file not found: {filename}")
    
    return FileResponse(file_path, filename=filename)


# -----------------------------------------------------------
# ✅ Get insights for a specific table
# -----------------------------------------------------------
@app.get("/insights/{table_name}")
def get_table_insights(table_name: str, days: int = 7, mode: str = "recent"):
    """
    Generate comprehensive insights for a specific table.
    Works dynamically for any dataset structure.
    """
    safe_name = sanitize_table_name(table_name)
    
    try:
        df = read_table(safe_name, limit=10000)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Table not found: {str(e)}")
    
    if df.empty:
        return {"summary": {}, "text": "No data available in this table."}
    
    summary = summarize_overall(df, days=days, mode=mode)
    text = format_text_summary(summary)
    
    return {
        "table_name": safe_name,
        "summary": summary,
        "text_summary": text,
        "row_count": len(df),
        "column_count": len(df.columns)
    }


# -----------------------------------------------------------
# ✅ General insights endpoint (legacy support)
# -----------------------------------------------------------
@app.get("/insights")
def get_insights(days: int = 7, mode: str = "recent"):
    """
    Get insights from the most recently uploaded dataset.
    (Legacy endpoint - use /insights/{table_name} for specific tables)
    """
    df = load_df()
    if df is None or df.empty:
        return {"summary": {}, "text": "No data available for analysis."}
    
    summary = summarize_overall(df, days=days, mode=mode)
    text = format_text_summary(summary)
    return {"summary": summary, "text": text}


# -----------------------------------------------------------
# ✅ Health check with detailed system info
# -----------------------------------------------------------
@app.get("/health")
def health_check():
    """Detailed system health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "tables_count": len(list_tables()),
        "raw_files": len([f for f in os.listdir(RAW_DIR) if f.endswith(('.csv', '.xlsx', '.xls'))]),
        "cleaned_files": len([f for f in os.listdir(CLEANED_DIR) if f.endswith('.csv')]),
        "directories": {
            "data": DATA_DIR,
            "raw": RAW_DIR,
            "cleaned": CLEANED_DIR
        }
    }