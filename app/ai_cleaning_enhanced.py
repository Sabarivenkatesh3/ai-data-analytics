# app/ai_cleaning_enhanced.py
"""
Enhanced AI-Powered Data Cleaning System
Uses Gemini AI to intelligently analyze and clean any dataset
WITH AUTOMATIC MODEL FALLBACK
"""

import pandas as pd
import json
import re
import os
import google.generativeai as genai
from typing import Dict, Any, List, Tuple, Callable
import numpy as np
from datetime import datetime
import warnings


class IntelligentDataCleaner:
    """
    AI-powered data cleaner that analyzes datasets and generates
    custom cleaning strategies on the fly
    """

    # List of models to try in order (from best to fallback)
    AVAILABLE_MODELS = [
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-2.0-flash-exp',
        'gemini-2.0-flash',
        'gemini-flash-latest',
        'gemini-pro-latest'
    ]

    def __init__(self, api_key: str = None):
        """Initialize with Google Gemini API"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "❌ No API key found! Set GEMINI_API_KEY environment variable.\n"
                "Get your free key at: https://makersuite.google.com/app/apikey"
            )

        genai.configure(api_key=self.api_key)
        
        # Try to initialize with the best available model
        self.model = None
        self.current_model_name = None
        self._initialize_model()

    def _initialize_model(self):
        """Try to initialize with the first available model"""
        print("🔍 Detecting available Gemini models...")
        
        for model_name in self.AVAILABLE_MODELS:
            try:
                self.model = genai.GenerativeModel(model_name)
                # Test if model works with a simple prompt
                test_response = self.model.generate_content("Say 'OK'")
                if test_response and test_response.text:
                    self.current_model_name = model_name
                    print(f"✅ Connected to Google Gemini AI: {model_name}")
                    return
            except Exception as e:
                print(f"⚠️  Model {model_name} not available: {str(e)[:50]}...")
                continue
        
        # If no model works, raise error
        raise ValueError(
            "❌ Could not connect to any Gemini model. "
            "Please check your API key and internet connection."
        )

    def _try_ai_call_with_fallback(self, prompt: str) -> str:
        """Try AI call with automatic model fallback"""
        last_error = None
        
        for model_name in self.AVAILABLE_MODELS:
            try:
                # Try current model first, then fallbacks
                if model_name == self.current_model_name and self.model:
                    response = self.model.generate_content(prompt)
                else:
                    temp_model = genai.GenerativeModel(model_name)
                    response = temp_model.generate_content(prompt)
                
                if response and response.text:
                    # If we switched models, update current
                    if model_name != self.current_model_name:
                        self.current_model_name = model_name
                        self.model = genai.GenerativeModel(model_name)
                        print(f"🔄 Switched to model: {model_name}")
                    return response.text.strip()
                    
            except Exception as e:
                last_error = e
                print(f"⚠️  Model {model_name} failed: {str(e)[:50]}...")
                continue
        
        # If all models failed, raise the last error
        raise Exception(f"All models failed. Last error: {last_error}")

    def analyze_and_clean(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main entry point: AI analyzes the dataset and performs intelligent cleaning

        Returns:
            (cleaned_dataframe, comprehensive_report)
        """
        print(f"\n{'='*80}")
        print(f"🤖 AI-POWERED CLEANING: {dataset_name}")
        print(f"{'='*80}\n")

        # Step 1: Deep Dataset Profiling
        print("📊 Step 1: Profiling dataset structure...")
        profile = self._deep_profile_dataset(df)

        # Step 2: AI Analysis
        print("🧠 Step 2: AI analyzing data patterns and issues...")
        ai_analysis = self._get_ai_analysis(profile, df)

        # Step 3: Generate Custom Cleaning Strategy
        print("⚙️  Step 3: Generating custom cleaning strategy...")
        cleaning_strategy = self._generate_cleaning_strategy(ai_analysis, df)

        # Step 4: Execute Cleaning
        print("🧹 Step 4: Executing AI-recommended cleaning operations...")
        cleaned_df = self._execute_cleaning(df.copy(), cleaning_strategy)

        # Step 5: Validate Results
        print("✅ Step 5: Validating cleaned data quality...")
        validation = self._validate_cleaning(df, cleaned_df)

        # Generate comprehensive report
        report = {
            "original_shape": df.shape,
            "cleaned_shape": cleaned_df.shape,
            "profile": profile,
            "ai_analysis": ai_analysis,
            "cleaning_strategy": cleaning_strategy,
            "validation": validation,
            "model_used": self.current_model_name,
            "summary": self._generate_summary(df, cleaned_df, ai_analysis, validation)
        }

        print(f"\n{'='*80}")
        print("🎉 AI CLEANING COMPLETE!")
        print(f"{'='*80}\n")

        return cleaned_df, report

    def _deep_profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a comprehensive dataset profile for AI analysis"""
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": {},
            "data_quality_score": 0,
            "detected_issues": [],
            "sample_data": {}
        }

        for col in df.columns:
            col_data = df[col]

            # Get diverse samples (not just first values)
            sample_values = []
            if not col_data.empty:
                # Sample from different parts of the dataset
                indices = np.linspace(0, len(col_data) - 1, min(15, max(1, len(col_data))), dtype=int)
                sample_values = [str(col_data.iloc[i])[:100] for i in indices if pd.notna(col_data.iloc[i])]

            # Detect data patterns
            patterns = self._detect_advanced_patterns(col_data)

            # Statistical analysis
            stats = self._column_statistics(col_data)

            # Data quality metrics
            quality = self._assess_column_quality(col_data)

            profile["columns"][col] = {
                "dtype": str(col_data.dtype),
                "samples": sample_values,
                "patterns": patterns,
                "statistics": stats,
                "quality": quality,
                "issues": self._identify_column_issues(col, col_data, patterns, quality)
            }

            # Collect all issues
            profile["detected_issues"].extend(profile["columns"][col]["issues"])

        # Overall data quality score
        profile["data_quality_score"] = self._calculate_overall_quality(profile)

        return profile

    def _detect_advanced_patterns(self, series: pd.Series) -> List[str]:
        """Detect sophisticated data patterns using multiple techniques"""
        patterns: List[str] = []
        non_null = series.dropna().astype(str)
        sample = non_null.head(100)

        if sample.empty:
            return ["empty_or_all_null"]

        # Pattern catalog with regex
        pattern_checks = {
            "k_plus_notation": r'\d+K\+?',
            "m_plus_notation": r'\d+M\+?',
            "rating_text": r'\d+\.?\d*\s*(?:out of|\/)',
            "star_rating": r'\d+\.?\d*\s*stars?',
            "currency_usd": r'\$\s*\d+',
            "currency_euro": r'€\s*\d+',
            "currency_pound": r'£\s*\d+',
            "percentage": r'\d+\.?\d*\s*%',
            "phone_number": r'\+?\d{1,3}[-.\s]?(?:\d{1,4})?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "url": r'https?://[^\s]+',
            "date_iso": r'\d{4}-\d{2}-\d{2}',
            "date_us": r'\d{1,2}/\d{1,2}/\d{2,4}',
            "date_text": r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
            "time_24h": r'\d{2}:\d{2}(?::\d{2})?',
            "comma_separated_numbers": r'\d{1,3}(?:,\d{3})+',
            "parentheses_negative": r'\(\d+\.?\d*\)',
            "range_notation": r'\d+\.?\d*\s*-\s*\d+\.?\d*',
            "boolean_text": r'\b(?:true|false|yes|no|y|n)\b',
            "missing_indicators": r'\b(?:na|n/a|null|none|nan|missing|unknown)\b'
        }

        # FIXED: Suppress regex match group warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            for pattern_name, regex in pattern_checks.items():
                try:
                    matches = sample.str.contains(regex, regex=True, case=False, na=False)
                    if matches.any():
                        patterns.append(pattern_name)
                except Exception:
                    continue

        # Check if mostly numeric (even if stored as string)
        try:
            numeric_series = pd.to_numeric(sample, errors='coerce')
            if len(numeric_series) > 0:
                numeric_ratio = numeric_series.notna().sum() / len(numeric_series)
                if numeric_ratio > 0.7:
                    patterns.append("numeric_as_string")
        except Exception:
            pass

        # Check for cardinality / categorical signals
        series_len = len(series)
        unique_count = series.nunique(dropna=True) if series_len > 0 else 0
        unique_ratio = unique_count / series_len if series_len > 0 else 0.0

        if series_len > 20 and unique_ratio < 0.05:
            patterns.append("low_cardinality_categorical")
        elif unique_ratio < 0.3 and series_len > 0:
            patterns.append("categorical")

        if series_len > 0 and unique_ratio > 0.95:
            patterns.append("high_cardinality_id")

        return patterns if patterns else ["standard"]

    def _column_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive column statistics"""
        total_count = len(series)
        null_count = int(series.isna().sum()) if total_count > 0 else 0
        stats = {
            "total_count": total_count,
            "null_count": null_count,
            "null_percentage": float((null_count / total_count * 100) if total_count > 0 else 0.0),
            "unique_count": int(series.nunique()) if total_count > 0 else 0,
            "unique_percentage": float((series.nunique() / total_count * 100) if total_count > 0 else 0.0)
        }

        # Try numeric statistics
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.notna().sum() > 0:
                stats["numeric_stats"] = {
                    "min": float(numeric_series.min()),
                    "max": float(numeric_series.max()),
                    "mean": float(numeric_series.mean()),
                    "median": float(numeric_series.median()),
                    "std": float(numeric_series.std())
                }
            else:
                stats["numeric_stats"] = None
        except Exception:
            stats["numeric_stats"] = None

        # Most common values
        if series.notna().sum() > 0:
            top_values = series.value_counts().head(5).to_dict()
            stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}
        else:
            stats["top_values"] = {}

        return stats

    def _assess_column_quality(self, series: pd.Series) -> Dict[str, Any]:
        """Assess data quality metrics for a column"""
        total = len(series)
        completeness = (1 - series.isna().sum() / total) * 100 if total > 0 else 0.0
        quality = {
            "completeness": completeness,
            "consistency": 0,
            "validity": 0
        }

        # Consistency: Check if data format is consistent
        non_null = series.dropna().astype(str)
        if len(non_null) > 0:
            lengths = non_null.str.len()
            length_std = (lengths.std() / lengths.mean()) if lengths.mean() > 0 else 0
            quality["consistency"] = max(0, 100 - (length_std * 10))

        # Validity
        quality["validity"] = 100 - (series.isna().sum() / total * 100) if total > 0 else 0

        return quality

    def _identify_column_issues(
        self, col_name: str, series: pd.Series, patterns: List[str], quality: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Identify specific issues in a column"""
        issues: List[Dict[str, str]] = []

        # Missing values
        if quality["completeness"] < 50:
            issues.append({
                "type": "high_missing_rate",
                "severity": "high",
                "message": f"{quality['completeness']:.1f}% complete - too many missing values"
            })
        elif quality["completeness"] < 90:
            issues.append({
                "type": "moderate_missing_rate",
                "severity": "medium",
                "message": f"{quality['completeness']:.1f}% complete"
            })

        # Format issues
        if any(p in patterns for p in ["k_plus_notation", "rating_text", "comma_separated_numbers"]):
            issues.append({
                "type": "needs_parsing",
                "severity": "high",
                "message": "Contains formatted numbers requiring parsing"
            })

        # Mixed types
        if "numeric_as_string" in patterns:
            issues.append({
                "type": "wrong_dtype",
                "severity": "medium",
                "message": "Numeric data stored as text"
            })

        # Inconsistent formatting
        if quality["consistency"] < 50:
            issues.append({
                "type": "inconsistent_format",
                "severity": "medium",
                "message": "Inconsistent data formatting detected"
            })

        return issues

    def _calculate_overall_quality(self, profile: Dict[str, Any]) -> float:
        """Calculate overall dataset quality score (0-100)"""
        if not profile["columns"]:
            return 0.0

        quality_scores = []
        for col_data in profile["columns"].values():
            col_quality = col_data["quality"]
            avg_quality = (col_quality.get("completeness", 0) + col_quality.get("consistency", 0) + col_quality.get("validity", 0)) / 3
            quality_scores.append(avg_quality)

        return round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0

    def _get_ai_analysis(self, profile: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Use Gemini AI to analyze the dataset and provide intelligent recommendations"""

        # Create a focused prompt for AI
        prompt = f"""You are an expert data scientist analyzing a dataset for cleaning. 

DATASET OVERVIEW:
- Rows: {profile['shape']['rows']}
- Columns: {profile['shape']['columns']}
- Overall Quality Score: {profile['data_quality_score']}/100

COLUMN ANALYSIS:
{self._format_columns_for_ai(profile)}

YOUR TASK:
Analyze this dataset and provide a JSON response with cleaning recommendations.

For each problematic column, specify:
1. What's wrong with it
2. Exact transformation needed
3. Priority (high/medium/low)

Respond with ONLY valid JSON in this format:
{{
  "overall_assessment": "Brief summary of main issues",
  "recommended_operations": [
    {{
      "column": "column_name",
      "issue": "description",
      "action": "parse_format|fill_missing|convert_type",
      "details": {{
        "method": "specific method to use"
      }},
      "priority": "high|medium|low"
    }}
  ],
  "priority_order": ["high_priority_column1"],
  "estimated_quality_improvement": "X%"
}}"""

        try:
            # Call Gemini API with automatic fallback
            response_text = self._try_ai_call_with_fallback(prompt)

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                ai_response = json.loads(json_match.group())
                print(f"✅ AI Analysis Complete: {len(ai_response.get('recommended_operations', []))} operations recommended")
                print(f"🤖 Model used: {self.current_model_name}")
                return ai_response
            else:
                print("⚠️  AI response not in expected format, using fallback analysis")
                return self._fallback_analysis(profile)

        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")
            print("📊 Using fallback pattern-based analysis...")
            return self._fallback_analysis(profile)

    def _format_columns_for_ai(self, profile: Dict[str, Any]) -> str:
        """Format column information for AI prompt"""
        lines = []
        for col_name, col_data in profile["columns"].items():
            lines.append(f"\nColumn: {col_name}")
            lines.append(f"  Type: {col_data['dtype']}")
            lines.append(f"  Patterns: {', '.join(col_data.get('patterns', []))}")
            lines.append(f"  Quality: {col_data['quality'].get('completeness', 0):.1f}% complete")
            samples = col_data.get('samples', [])[:5]
            lines.append(f"  Samples: {samples}")
            if col_data.get('issues'):
                lines.append(f"  Issues: {[i['message'] for i in col_data['issues']]}")
        return '\n'.join(lines)

    def _fallback_analysis(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent fallback when AI is unavailable"""
        operations = []

        for col_name, col_data in profile["columns"].items():
            patterns = col_data.get("patterns", [])
            issues = col_data.get("issues", [])

            # Generate operations based on patterns
            if "k_plus_notation" in patterns or "m_plus_notation" in patterns:
                operations.append({
                    "column": col_name,
                    "issue": "Contains K+/M+ notation",
                    "action": "parse_format",
                    "details": {"method": "parse_magnitude_notation"},
                    "priority": "high"
                })

            if "rating_text" in patterns or "star_rating" in patterns:
                operations.append({
                    "column": col_name,
                    "issue": "Contains rating text",
                    "action": "parse_format",
                    "details": {"method": "extract_numeric_rating"},
                    "priority": "high"
                })

            if "comma_separated_numbers" in patterns:
                operations.append({
                    "column": col_name,
                    "issue": "Numbers with comma separators",
                    "action": "parse_format",
                    "details": {"method": "remove_comma_separators"},
                    "priority": "medium"
                })

            if any("currency" in p for p in patterns):
                operations.append({
                    "column": col_name,
                    "issue": "Contains currency symbols",
                    "action": "parse_format",
                    "details": {"method": "remove_currency_symbols"},
                    "priority": "medium"
                })

            # Handle missing values
            for issue in issues:
                if "missing" in issue.get("type", ""):
                    operations.append({
                        "column": col_name,
                        "issue": issue["message"],
                        "action": "fill_missing",
                        "details": {"method": "intelligent_fill"},
                        "priority": issue.get("severity", "medium")
                    })

        return {
            "overall_assessment": f"Found {len(operations)} operations needed",
            "recommended_operations": operations,
            "priority_order": [op["column"] for op in operations if op.get("priority") == "high"]
        }

    def _generate_cleaning_strategy(self, ai_analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, List[Callable]]:
        """Convert AI recommendations into executable cleaning functions"""
        strategy: Dict[str, List[Callable]] = {}

        for operation in ai_analysis.get("recommended_operations", []):
            col = operation.get("column")
            action = operation.get("action")
            details = operation.get("details", {}) or {}

            if not col or col not in df.columns:
                continue

            if col not in strategy:
                strategy[col] = []

            # Map actions to cleaning functions
            if action == "parse_format":
                method = details.get("method")
                if method == "parse_magnitude_notation":
                    strategy[col].append(self._parse_magnitude)
                elif method == "extract_numeric_rating":
                    strategy[col].append(self._extract_rating)
                elif method == "remove_comma_separators":
                    strategy[col].append(self._remove_commas)
                elif method == "remove_currency_symbols":
                    strategy[col].append(self._remove_currency)

            elif action == "fill_missing":
                strategy[col].append(lambda v, s=df[col]: self._intelligent_fill(v, s))

            elif action == "convert_type":
                target_type = details.get("target_type", "numeric")
                strategy[col].append(lambda v, t=target_type: self._convert_type(v, t))

            elif action == "remove_outliers":
                strategy[col].append(self._remove_outliers)

        return strategy

    def _execute_cleaning(self, df: pd.DataFrame, strategy: Dict[str, List[Callable]]) -> pd.DataFrame:
        """Execute the cleaning strategy on the dataframe"""
        for col, functions in strategy.items():
            if col not in df.columns:
                continue

            print(f"  🔧 Cleaning column: {col} ({len(functions)} operations)")

            for func in functions:
                try:
                    try:
                        result = df[col].apply(func)
                        if isinstance(result, pd.Series):
                            df[col] = result
                        else:
                            df[col] = result
                    except Exception:
                        try:
                            new_series = func(df[col])
                            if isinstance(new_series, pd.Series):
                                df[col] = new_series
                        except Exception as e:
                            print(f"    ⚠️  Operation failed: {e}")
                except Exception as e:
                    print(f"    ⚠️  Operation failed: {e}")

        return df

    # ============ CLEANING FUNCTIONS ============

    def _parse_magnitude(self, value):
        """Parse K+, M+ notation"""
        if pd.isna(value):
            return np.nan

        text = str(value).upper()
        match = re.search(r'(\d+\.?\d*)\s*([KM])\+?', text)

        if match:
            num = float(match.group(1))
            multiplier = 1000 if match.group(2) == 'K' else 1000000
            return num * multiplier

        match = re.search(r'(\d+\.?\d*)', text)
        return float(match.group(1)) if match else np.nan

    def _extract_rating(self, value):
        """Extract numeric rating from text"""
        if pd.isna(value):
            return np.nan

        text = str(value)
        match = re.search(r'(\d+\.?\d*)\s*(?:out of|/|stars?)', text, flags=re.IGNORECASE)
        return float(match.group(1)) if match else np.nan

    def _remove_commas(self, value):
        """Remove comma separators from numbers"""
        if pd.isna(value):
            return np.nan

        text = str(value).replace(',', '')
        try:
            return float(text)
        except Exception:
            return np.nan

    def _remove_currency(self, value):
        """Remove currency symbols"""
        if pd.isna(value):
            return np.nan

        text = re.sub(r'[$€£¥₹,]', '', str(value))
        try:
            return float(text.strip())
        except Exception:
            return np.nan

    def _intelligent_fill(self, value, series):
        """Intelligently fill missing values based on column type"""
        if pd.notna(value):
            return value

        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.notna().sum() > 0:
            return numeric_series.median()

        mode_series = series.mode()
        if len(mode_series) > 0:
            return mode_series.iloc[0]

        return "Unknown"

    def _convert_type(self, value, target_type):
        """Convert value to target type"""
        if pd.isna(value):
            return np.nan

        try:
            if target_type == "numeric":
                return float(value)
            elif target_type == "integer":
                return int(float(value))
            elif target_type == "string":
                return str(value)
            else:
                return value
        except Exception:
            return np.nan

    def _remove_outliers(self, series):
        """Remove outliers using IQR method"""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            Q1 = numeric_series.quantile(0.25)
            Q3 = numeric_series.quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            return series.where((numeric_series >= lower) & (numeric_series <= upper), np.nan)
        except Exception:
            return series

    def _validate_cleaning(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the cleaning results"""
        validation = {
            "data_quality_improved": False,
            "metrics": {}
        }

        original_nulls = int(original_df.isna().sum().sum())
        cleaned_nulls = int(cleaned_df.isna().sum().sum())

        original_quality = (1 - original_nulls / original_df.size) * 100 if original_df.size > 0 else 0
        cleaned_quality = (1 - cleaned_nulls / cleaned_df.size) * 100 if cleaned_df.size > 0 else 0

        validation["metrics"] = {
            "original_quality": round(original_quality, 2),
            "cleaned_quality": round(cleaned_quality, 2),
            "improvement": round(cleaned_quality - original_quality, 2),
            "nulls_removed": int(original_nulls - cleaned_nulls)
        }

        validation["data_quality_improved"] = cleaned_quality > original_quality

        return validation

    def _generate_summary(
        self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame,
        ai_analysis: Dict[str, Any], validation: Dict[str, Any]
    ) -> str:
        """Generate human-readable summary"""
        lines = [
            "\n" + "=" * 80,
            "🎯 AI CLEANING SUMMARY",
            "=" * 80,
            "",
            f"📊 Dataset: {original_df.shape[0]} rows × {original_df.shape[1]} columns",
            f"🤖 AI Model: {self.current_model_name}",
            f"🤖 AI Assessment: {ai_analysis.get('overall_assessment', 'N/A')}",
            f"🔧 Operations Performed: {len(ai_analysis.get('recommended_operations', []))}",
            "",
            f"📈 Quality Improvement:",
            f"   Before: {validation['metrics'].get('original_quality', 'N/A')}%",
            f"   After:  {validation['metrics'].get('cleaned_quality', 'N/A')}%",
            f"   Gain:   +{validation['metrics'].get('improvement', 'N/A')}%",
            "",
            f"✅ Result: {'SUCCESS' if validation.get('data_quality_improved') else 'PARTIAL'}",
            "=" * 80
        ]

        return "\n".join(lines)


def clean_with_ai(df: pd.DataFrame, dataset_name: str = "dataset", api_key: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    One-line function to clean any dataset with AI
    """
    cleaner = IntelligentDataCleaner(api_key=api_key)
    return cleaner.analyze_and_clean(df, dataset_name)