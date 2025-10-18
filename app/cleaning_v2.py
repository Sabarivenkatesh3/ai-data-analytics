# app/cleaning_v2.py
"""
AI-Enhanced Data Cleaning Pipeline
Combines rule-based cleaning with AI analysis
"""

import pandas as pd
from typing import Tuple, Dict, Any
from .cleaning import detect_and_clean as basic_clean
from .ai_cleaning_enhanced import AIDataAnalyzer, apply_ai_cleaning


def ai_enhanced_clean(
    df: pd.DataFrame,
    file_basename: str = "uploaded",
    remove_duplicates: bool = True,
    handle_outliers: bool = True,
    outlier_threshold: float = 1.5,
    use_ai_analysis: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enhanced cleaning pipeline that uses AI to analyze data first,
    then applies both AI-recommended and standard cleaning.
    
    Flow:
    1. AI analyzes raw data → identifies patterns & issues
    2. AI recommends custom cleaning strategies
    3. Apply custom parsers for special formats
    4. Apply standard cleaning (dates, currency, missing values, etc.)
    5. Generate comprehensive report
    
    Args:
        df: Raw DataFrame
        file_basename: Name for the cleaned file
        remove_duplicates: Whether to remove duplicate rows
        handle_outliers: Whether to handle outliers
        outlier_threshold: IQR multiplier for outlier detection
        use_ai_analysis: Whether to use AI analysis (set False if no OpenAI key)
    
    Returns:
        (cleaned_df, comprehensive_report)
    """
    
    print("\n" + "=" * 70)
    print("🤖 AI-ENHANCED DATA CLEANING PIPELINE")
    print("=" * 70 + "\n")
    
    # -----------------------------------------------------------
    # STEP 1: AI Analysis (if enabled)
    # -----------------------------------------------------------
    ai_report = None
    if use_ai_analysis:
        print("🔍 Step 1: AI is analyzing your dataset...")
        print("-" * 70)
        
        analyzer = AIDataAnalyzer()
        ai_report = analyzer.analyze_dataset(df)
        
        print(ai_report["summary"])
        print("\n")
        
        # -----------------------------------------------------------
        # STEP 2: Apply AI-recommended custom cleaning
        # -----------------------------------------------------------
        print("🧠 Step 2: Applying AI-recommended custom parsers...")
        print("-" * 70)
        
        df = apply_ai_cleaning(df, ai_report)
        print("\n")
    
    # -----------------------------------------------------------
    # STEP 3: Apply standard cleaning pipeline
    # -----------------------------------------------------------
    print("🧹 Step 3: Applying standard cleaning operations...")
    print("-" * 70)
    
    cleaned_df, standard_report = basic_clean(
        df,
        file_basename=file_basename,
        remove_duplicates=remove_duplicates,
        handle_outliers=handle_outliers,
        outlier_threshold=outlier_threshold
    )
    
    # -----------------------------------------------------------
    # STEP 4: Generate comprehensive report
    # -----------------------------------------------------------
    comprehensive_report = {
        "ai_analysis": ai_report if ai_report else {"message": "AI analysis disabled"},
        "standard_cleaning": standard_report,
        "final_summary": _generate_final_summary(df, cleaned_df, ai_report, standard_report)
    }
    
    print("\n" + "=" * 70)
    print("✅ AI-ENHANCED CLEANING COMPLETE!")
    print("=" * 70 + "\n")
    
    return cleaned_df, comprehensive_report


def _generate_final_summary(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    ai_report: Dict[str, Any],
    standard_report: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a final summary combining AI and standard cleaning results
    """
    return {
        "transformation_summary": {
            "original_shape": df_original.shape,
            "final_shape": df_cleaned.shape,
            "rows_removed": df_original.shape[0] - df_cleaned.shape[0],
            "columns_added": df_cleaned.shape[1] - df_original.shape[1]
        },
        "ai_contributions": {
            "custom_parsers_applied": len(ai_report.get("cleaning_recommendations", {}).get("custom_parsers_needed", {})) if ai_report else 0,
            "issues_identified": len(ai_report.get("dataset_profile", {}).get("overall_issues", [])) if ai_report else 0
        },
        "standard_contributions": {
            "operations_performed": standard_report.get("summary", {}).get("total_operations", 0),
            "data_types_detected": {
                "numeric": standard_report.get("summary", {}).get("numeric_columns", 0),
                "categorical": standard_report.get("summary", {}).get("categorical_columns", 0),
                "dates": standard_report.get("summary", {}).get("date_columns", 0)
            }
        },
        "quality_improvement": {
            "missing_values": {
                "before": int(df_original.isna().sum().sum()),
                "after": int(df_cleaned.isna().sum().sum()),
                "improvement": int(df_original.isna().sum().sum() - df_cleaned.isna().sum().sum())
            },
            "data_quality_score": _calculate_quality_score(df_cleaned)
        }
    }


def _calculate_quality_score(df: pd.DataFrame) -> float:
    """
    Calculate overall data quality score (0-100)
    """
    if df.empty:
        return 0.0
    
    # Factors:
    # 1. Completeness (% non-null)
    completeness = (1 - df.isna().sum().sum() / df.size) * 100
    
    # 2. Consistency (% proper data types)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).shape[1]
    date_cols = df.select_dtypes(include=['datetime64']).shape[1]
    total_cols = df.shape[1]
    consistency = ((numeric_cols + date_cols) / total_cols) * 100 if total_cols > 0 else 0
    
    # 3. Uniqueness (no duplicates)
    uniqueness = (1 - df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 100
    
    # Weighted average
    quality_score = (completeness * 0.5) + (consistency * 0.3) + (uniqueness * 0.2)
    
    return round(quality_score, 2)