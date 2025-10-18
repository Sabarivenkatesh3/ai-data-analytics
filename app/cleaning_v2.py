# app/cleaning_v2.py
"""
AI-Enhanced Data Cleaning Pipeline
Uses Gemini AI for intelligent cleaning
"""
import os  
import pandas as pd
from typing import Tuple, Dict, Any
from .cleaning import detect_and_clean as basic_clean

# Try to import AI cleaner
try:
    from .ai_cleaning_enhanced import IntelligentDataCleaner
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ AI modules not available - using basic cleaning only")


def ai_enhanced_clean(
    df: pd.DataFrame,
    file_basename: str = "uploaded",
    remove_duplicates: bool = True,
    handle_outliers: bool = True,
    outlier_threshold: float = 1.5,
    use_ai_analysis: bool = True,
    api_key: str = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enhanced cleaning pipeline with optional AI analysis.
    
    If use_ai_analysis=True and AI available:
        Uses Gemini AI for intelligent analysis and custom cleaning
    Else:
        Falls back to standard rule-based cleaning
    
    Args:
        df: Raw DataFrame
        file_basename: Name for the cleaned file
        remove_duplicates: Whether to remove duplicate rows
        handle_outliers: Whether to handle outliers
        outlier_threshold: IQR multiplier for outlier detection
        use_ai_analysis: Whether to use AI (requires API key)
        api_key: Optional Gemini API key (uses env var if not provided)
    
    Returns:
        (cleaned_df, comprehensive_report)
    """
    
    # Check if AI is requested and available
    if use_ai_analysis and AI_AVAILABLE and api_key or os.getenv("GEMINI_API_KEY"):
        try:
            print("\n" + "=" * 70)
            print("🤖 USING AI-POWERED CLEANING (GEMINI)")
            print("=" * 70 + "\n")
            
            # Use AI cleaner
            cleaner = IntelligentDataCleaner(api_key=api_key)
            cleaned_df, ai_report = cleaner.analyze_and_clean(df, dataset_name=file_basename)
            
            return cleaned_df, {
                "method": "ai_powered",
                "ai_analysis": ai_report.get("ai_analysis", {}),
                "profile": ai_report.get("profile", {}),
                "validation": ai_report.get("validation", {}),
                "summary": ai_report.get("summary", ""),
                "cleaning_strategy": ai_report.get("cleaning_strategy", {})
            }
            
        except Exception as e:
            print(f"⚠️ AI cleaning failed: {e}")
            print("📊 Falling back to standard cleaning...")
    
    # Fall back to standard cleaning
    print("\n" + "=" * 70)
    print("🧹 USING STANDARD CLEANING (RULE-BASED)")
    print("=" * 70 + "\n")
    
    cleaned_df, standard_report = basic_clean(
        df,
        file_basename=file_basename,
        remove_duplicates=remove_duplicates,
        handle_outliers=handle_outliers,
        outlier_threshold=outlier_threshold
    )
    
    return cleaned_df, {
        "method": "standard",
        "standard_cleaning": standard_report,
        "operations": standard_report.get("operations", []),
        "summary": standard_report.get("summary", {})
    }


