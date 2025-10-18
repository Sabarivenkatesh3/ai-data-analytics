# app/main_ai.py
"""
AI-Enhanced Data Cleaning API
Intelligently cleans any dataset using Gemini AI
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import io
import os
import pandas as pd
from pathlib import Path
from typing import Optional

from .db import init_db, create_table_from_df, list_tables, sanitize_table_name, read_table
from .ai_cleaning_enhanced import IntelligentDataCleaner, clean_with_ai
from .insights import load_df, summarize_overall, format_text_summary

# Configuration
DATA_DIR = "data"
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
RAW_DIR = os.path.join(DATA_DIR, "raw")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

# Create directories
for directory in [DATA_DIR, CLEANED_DIR, RAW_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="AI Data Cleaning Platform",
    version="2.0.0",
    description="Upload any dataset - AI analyzes and cleans it intelligently"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Initialize AI cleaner (will be created per request to support different API keys)
def get_ai_cleaner(api_key: Optional[str] = None):
    """Get AI cleaner instance with optional custom API key"""
    return IntelligentDataCleaner(api_key=api_key)


@app.get("/")
def read_root():
    return {
        "status": "running",
        "message": "🤖 AI Data Cleaning Platform v2.0",
        "features": [
            "AI-powered intelligent data analysis",
            "Custom cleaning strategy generation",
            "Automatic pattern detection",
            "Smart missing value handling",
            "Real-time quality scoring"
        ],
        "endpoints": {
            "upload_ai": "POST /upload-ai - AI-powered cleaning",
            "upload_standard": "POST /upload-file - Standard cleaning",
            "analyze": "POST /analyze-only - AI analysis without cleaning",
            "list": "GET /tables - List all tables",
            "download": "GET /download/{table_name}",
            "insights": "GET /insights/{table_name}",
            "report": "GET /report/{table_name}"
        }
    }


@app.post("/upload-ai")
async def upload_with_ai(
    file: UploadFile = File(...),
    table_name: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None)
):
    """
    🤖 AI-POWERED UPLOAD
    
    Upload any CSV/Excel file. AI will:
    1. Deeply analyze the data structure
    2. Identify all issues and patterns
    3. Generate custom cleaning strategy
    4. Execute intelligent cleaning
    5. Validate and report results
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Read file
    contents = await file.read()
    filename = file.filename
    
    # Save raw file
    raw_path = os.path.join(RAW_DIR, filename)
    with open(raw_path, "wb") as f:
        f.write(contents)
    
    # Parse file based on extension
    try:
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
            file_type = "Excel"
        elif filename.endswith('.csv'):
            # Try multiple encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                    break
                except:
                    continue
            file_type = "CSV"
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use CSV or Excel.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Generate safe table name
    original_name = table_name or filename.rsplit(".", 1)[0]
    safe_name = sanitize_table_name(original_name)
    
    print(f"\n{'='*80}")
    print(f"🤖 AI CLEANING REQUEST: {filename}")
    print(f"📊 Original: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"{'='*80}")
    
    # AI-powered cleaning
    try:
        cleaner = get_ai_cleaner(api_key)
        cleaned_df, report = cleaner.analyze_and_clean(df, dataset_name=safe_name)
    except ValueError as e:
        # API key issue
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ AI cleaning failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI cleaning failed: {str(e)}")
    
    # Save to database
    db_result = create_table_from_df(cleaned_df, table_name=safe_name, if_exists="replace")
    
    if db_result.get("status") != "ok":
        raise HTTPException(status_code=500, detail=f"Database error: {db_result.get('detail')}")
    
    # Export cleaned CSV
    cleaned_path = os.path.join(CLEANED_DIR, f"{safe_name}_ai_cleaned.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    
    # Save report
    report_path = os.path.join(REPORTS_DIR, f"{safe_name}_report.json")
    import json
    with open(report_path, 'w') as f:
        # Make report JSON serializable
        serializable_report = json.loads(json.dumps(report, default=str))
        json.dump(serializable_report, f, indent=2)
    
    # Generate preview
    preview = cleaned_df.head(10).to_dict(orient="records")
    
    # Prepare response
    return {
        "status": "success",
        "message": f"✅ AI cleaning complete: {filename}",
        "file_info": {
            "original_filename": filename,
            "file_type": file_type,
            "table_name": safe_name,
            "raw_path": raw_path,
            "cleaned_path": cleaned_path,
            "report_path": report_path
        },
        "data_transformation": {
            "original_shape": report["original_shape"],
            "cleaned_shape": report["cleaned_shape"],
            "rows_changed": report["original_shape"][0] - report["cleaned_shape"][0],
            "columns_changed": report["original_shape"][1] - report["cleaned_shape"][1]
        },
        "ai_analysis": {
            "overall_assessment": report["ai_analysis"].get("overall_assessment", "N/A"),
            "operations_count": len(report["ai_analysis"].get("recommended_operations", [])),
            "quality_score_before": report["profile"]["data_quality_score"],
            "quality_score_after": report["validation"]["metrics"]["cleaned_quality"],
            "improvement": report["validation"]["metrics"]["improvement"]
        },
        "cleaning_operations": [
            {
                "column": op["column"],
                "action": op["action"],
                "priority": op["priority"]
            }
            for op in report["ai_analysis"].get("recommended_operations", [])[:10]  # First 10
        ],
        "preview": preview,
        "summary": report["summary"],
        "download_links": {
            "cleaned_csv": f"/download/{safe_name}",
            "raw_file": f"/raw/{filename}",
            "report": f"/report/{safe_name}"
        }
    }


@app.post("/analyze-only")
async def analyze_dataset(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None)
):
    """
    🔍 ANALYZE ONLY (No Cleaning)
    
    Get AI analysis and recommendations without actually cleaning the data.
    Useful for understanding what needs to be done before committing to changes.
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    contents = await file.read()
    filename = file.filename
    
    # Parse file
    try:
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
        elif filename.endswith('.csv'):
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                    break
                except:
                    continue
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Analyze with AI
    try:
        cleaner = get_ai_cleaner(api_key)
        
        print(f"🔍 Analyzing: {filename}")
        profile = cleaner._deep_profile_dataset(df)
        ai_analysis = cleaner._get_ai_analysis(profile, df)
        
        return {
            "status": "analysis_complete",
            "filename": filename,
            "dataset_info": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "quality_score": profile["data_quality_score"],
                "issues_found": len(profile["detected_issues"])
            },
            "ai_assessment": ai_analysis.get("overall_assessment", "N/A"),
            "recommended_operations": ai_analysis.get("recommended_operations", []),
            "priority_columns": ai_analysis.get("priority_order", []),
            "estimated_improvement": ai_analysis.get("estimated_quality_improvement", "N/A"),
            "column_details": [
                {
                    "name": col,
                    "current_quality": data["quality"]["completeness"],
                    "issues": [i["message"] for i in data["issues"]],
                    "patterns": data["patterns"]
                }
                for col, data in profile["columns"].items()
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/report/{table_name}")
def get_cleaning_report(table_name: str):
    """Get detailed cleaning report for a table"""
    safe_name = sanitize_table_name(table_name)
    report_path = os.path.join(REPORTS_DIR, f"{safe_name}_report.json")
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(report_path, filename=f"{safe_name}_report.json")


@app.post("/upload-file")
async def upload_standard(
    file: UploadFile = File(...),
    table_name: Optional[str] = Form(None)
):
    """
    📊 STANDARD UPLOAD (No AI)
    
    Use rule-based cleaning without AI.
    Faster but less intelligent than AI-powered cleaning.
    """
    # Import standard cleaning
    from .cleaning import detect_and_clean
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    contents = await file.read()
    filename = file.filename
    
    # Save raw
    raw_path = os.path.join(RAW_DIR, filename)
    with open(raw_path, "wb") as f:
        f.write(contents)
    
    # Parse file
    try:
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
        elif filename.endswith('.csv'):
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                    break
                except:
                    continue
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read: {str(e)}")
    
    # Clean with standard rules
    original_name = table_name or filename.rsplit(".", 1)[0]
    safe_name = sanitize_table_name(original_name)
    
    cleaned_df, report = detect_and_clean(df, file_basename=safe_name)
    
    # Save to database
    db_result = create_table_from_df(cleaned_df, table_name=safe_name, if_exists="replace")
    
    if db_result.get("status") != "ok":
        raise HTTPException(status_code=500, detail=f"Database error: {db_result.get('detail')}")
    
    # Export CSV
    cleaned_path = os.path.join(CLEANED_DIR, f"{safe_name}_cleaned.csv")
    cleaned_df.to_csv(cleaned_path, index=False)
    
    preview = cleaned_df.head(10).to_dict(orient="records")
    
    return {
        "status": "success",
        "message": f"Standard cleaning complete: {filename}",
        "file_info": {
            "original_filename": filename,
            "table_name": safe_name,
            "cleaned_path": cleaned_path
        },
        "data_info": {
            "original_shape": df.shape,
            "cleaned_shape": cleaned_df.shape
        },
        "cleaning_report": report,
        "preview": preview
    }


@app.get("/tables")
def get_tables():
    """List all processed tables"""
    tables = list_tables()
    
    table_info = []
    for table in tables:
        try:
            df = read_table(table, limit=1)
            
            # Check if AI report exists
            report_exists = os.path.exists(
                os.path.join(REPORTS_DIR, f"{table}_report.json")
            )
            
            table_info.append({
                "table_name": table,
                "columns": df.columns.tolist(),
                "has_ai_report": report_exists,
                "download_link": f"/download/{table}",
                "report_link": f"/report/{table}" if report_exists else None
            })
        except:
            table_info.append({
                "table_name": table,
                "error": "Could not read metadata"
            })
    
    return {
        "total_tables": len(tables),
        "tables": table_info
    }


@app.get("/download/{table_name}")
def download_cleaned(table_name: str):
    """Download cleaned CSV"""
    safe_name = sanitize_table_name(table_name)
    
    # Try AI cleaned version first
    ai_path = os.path.join(CLEANED_DIR, f"{safe_name}_ai_cleaned.csv")
    standard_path = os.path.join(CLEANED_DIR, f"{safe_name}_cleaned.csv")
    
    file_path = ai_path if os.path.exists(ai_path) else standard_path
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=f"{safe_name}_cleaned.csv")


@app.get("/raw/{filename}")
def download_raw(filename: str):
    """Download original raw file"""
    file_path = os.path.join(RAW_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Raw file not found")
    
    return FileResponse(file_path, filename=filename)


@app.get("/insights/{table_name}")
def get_table_insights(table_name: str, days: int = 7, mode: str = "recent"):
    """Get comprehensive insights for a table"""
    safe_name = sanitize_table_name(table_name)
    
    try:
        df = read_table(safe_name, limit=10000)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Table not found: {str(e)}")
    
    if df.empty:
        return {"summary": {}, "text": "No data available"}
    
    summary = summarize_overall(df, days=days, mode=mode)
    text = format_text_summary(summary)
    
    return {
        "table_name": safe_name,
        "summary": summary,
        "text_summary": text,
        "row_count": len(df),
        "column_count": len(df.columns)
    }


@app.get("/health")
def health_check():
    """System health check"""
    tables = list_tables()
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(('.csv', '.xlsx', '.xls'))]
    cleaned_files = [f for f in os.listdir(CLEANED_DIR) if f.endswith('.csv')]
    reports = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.json')]
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "ai_enabled": True,
        "database": "connected",
        "statistics": {
            "tables": len(tables),
            "raw_files": len(raw_files),
            "cleaned_files": len(cleaned_files),
            "ai_reports": len(reports)
        },
        "directories": {
            "data": DATA_DIR,
            "raw": RAW_DIR,
            "cleaned": CLEANED_DIR,
            "reports": REPORTS_DIR
        }
    }


@app.get("/compare/{table_name}")
def compare_versions(table_name: str):
    """
    Compare AI-cleaned vs standard-cleaned versions
    (if both exist)
    """
    safe_name = sanitize_table_name(table_name)
    
    ai_path = os.path.join(CLEANED_DIR, f"{safe_name}_ai_cleaned.csv")
    standard_path = os.path.join(CLEANED_DIR, f"{safe_name}_cleaned.csv")
    
    comparison = {"table_name": safe_name}
    
    if os.path.exists(ai_path):
        ai_df = pd.read_csv(ai_path)
        comparison["ai_version"] = {
            "exists": True,
            "shape": ai_df.shape,
            "quality_score": round((1 - ai_df.isna().sum().sum() / ai_df.size) * 100, 2)
        }
    else:
        comparison["ai_version"] = {"exists": False}
    
    if os.path.exists(standard_path):
        std_df = pd.read_csv(standard_path)
        comparison["standard_version"] = {
            "exists": True,
            "shape": std_df.shape,
            "quality_score": round((1 - std_df.isna().sum().sum() / std_df.size) * 100, 2)
        }
    else:
        comparison["standard_version"] = {"exists": False}
    
    return comparison