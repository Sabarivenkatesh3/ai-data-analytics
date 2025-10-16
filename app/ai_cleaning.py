# app/ai_cleaning.py
"""
AI-Powered Data Cleaning Module
Uses GPT-4 to analyze raw data and generate custom cleaning strategies
"""

import pandas as pd
import json
import re
from typing import Dict, Any, List, Tuple
import os

# Placeholder for OpenAI integration (will be implemented in Phase 3)
try:
    from .ai_free import FreeLLMClient
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class AIDataAnalyzer:
    """
    Analyzes raw datasets and generates intelligent cleaning recommendations
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key and OPENAI_AVAILABLE:
            openai.api_key = self.api_key
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main entry point: Analyze a raw dataset and provide cleaning recommendations
        
        Returns:
            {
                "dataset_profile": {...},
                "cleaning_recommendations": {...},
                "custom_parsers_needed": [...],
                "priority_issues": [...]
            }
        """
        
        profile = self._profile_dataset(df)
        recommendations = self._generate_recommendations(df, profile)
        
        return {
            "dataset_profile": profile,
            "cleaning_recommendations": recommendations,
            "summary": self._generate_summary(profile, recommendations)
        }
    
    def _profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a detailed profile of the dataset
        """
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": {},
            "overall_issues": []
        }
        
        for col in df.columns:
            col_profile = self._profile_column(df[col], col)
            profile["columns"][col] = col_profile
            
            # Collect issues
            if col_profile["issues"]:
                profile["overall_issues"].extend(col_profile["issues"])
        
        return profile
    
    def _profile_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """
        Profile a single column
        """
        # Get sample values
        sample_values = series.dropna().head(10).tolist()
        
        # Detect patterns
        patterns = self._detect_patterns(series)
        
        # Identify issues
        issues = self._identify_issues(series, col_name, patterns)
        
        return {
            "name": col_name,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique()),
            "sample_values": [str(v)[:100] for v in sample_values],  # Truncate long values
            "detected_patterns": patterns,
            "issues": issues,
            "suggested_cleaning": self._suggest_cleaning(col_name, patterns, issues)
        }
    
    def _detect_patterns(self, series: pd.Series) -> List[str]:
        """
        Detect common patterns in the data
        """
        patterns = []
        sample = series.dropna().astype(str).head(50)
        
        if sample.empty:
            return ["empty_column"]
        
        # Pattern 1: K+ notation (6K+, 10K+)
        if sample.str.contains(r'\d+K\+', regex=True, case=False).any():
            patterns.append("k_plus_notation")
        
        # Pattern 2: Rating format (4.5 out of 5 stars)
        if sample.str.contains(r'\d+\.?\d*\s*out of\s*\d+', regex=True, case=False).any():
            patterns.append("rating_with_text")
        
        # Pattern 3: Currency with symbols ($, €, £)
        if sample.str.contains(r'[$€£¥₹]\s*\d', regex=True).any():
            patterns.append("currency_with_symbol")
        
        # Pattern 4: Percentage
        if sample.str.contains(r'\d+\.?\d*\s*%', regex=True).any():
            patterns.append("percentage")
        
        # Pattern 5: Date formats
        date_patterns = [
            r'\d{2}[-/]\d{2}[-/]\d{4}',  # DD-MM-YYYY or DD/MM/YYYY
            r'\d{4}[-/]\d{2}[-/]\d{2}',  # YYYY-MM-DD
            r'\w{3}\s+\d{1,2},?\s+\d{4}' # Mon DD, YYYY
        ]
        for pattern in date_patterns:
            if sample.str.contains(pattern, regex=True).any():
                patterns.append("date_format")
                break
        
        # Pattern 6: Mixed text and numbers
        if sample.str.contains(r'[a-zA-Z]+.*\d+|\d+.*[a-zA-Z]+', regex=True).any():
            if "k_plus_notation" not in patterns and "rating_with_text" not in patterns:
                patterns.append("mixed_text_numbers")
        
        # Pattern 7: URLs
        if sample.str.contains(r'https?://', regex=True).any():
            patterns.append("url")
        
        # Pattern 8: Multiple values separated by delimiter
        if sample.str.contains(r'[,;|]', regex=True).sum() > len(sample) * 0.3:
            patterns.append("delimited_values")
        
        # Pattern 9: Parentheses notation (negative numbers)
        if sample.str.contains(r'\(\d+\.?\d*\)', regex=True).any():
            patterns.append("parentheses_numbers")
        
        return patterns if patterns else ["standard"]
    
    def _identify_issues(self, series: pd.Series, col_name: str, patterns: List[str]) -> List[Dict[str, str]]:
        """
        Identify data quality issues
        """
        issues = []
        
        # Issue 1: High missing value rate
        missing_rate = series.isna().sum() / len(series)
        if missing_rate > 0.5:
            issues.append({
                "type": "high_missing_rate",
                "severity": "high",
                "description": f"Column '{col_name}' has {missing_rate*100:.1f}% missing values"
            })
        elif missing_rate > 0.1:
            issues.append({
                "type": "moderate_missing_rate",
                "severity": "medium",
                "description": f"Column '{col_name}' has {missing_rate*100:.1f}% missing values"
            })
        
        # Issue 2: Mixed data types
        if "mixed_text_numbers" in patterns:
            issues.append({
                "type": "mixed_data_types",
                "severity": "high",
                "description": f"Column '{col_name}' contains mixed text and numbers - cannot analyze numerically"
            })
        
        # Issue 3: Non-standard formats requiring parsing
        if any(p in patterns for p in ["k_plus_notation", "rating_with_text", "parentheses_numbers"]):
            issues.append({
                "type": "needs_custom_parser",
                "severity": "medium",
                "description": f"Column '{col_name}' has non-standard format requiring custom parsing"
            })
        
        # Issue 4: Column name issues
        if " " in col_name or "/" in col_name or "-" in col_name:
            issues.append({
                "type": "column_name_format",
                "severity": "low",
                "description": f"Column name '{col_name}' contains special characters"
            })
        
        return issues
    
    def _suggest_cleaning(self, col_name: str, patterns: List[str], issues: List[Dict]) -> List[str]:
        """
        Generate cleaning suggestions based on patterns and issues
        """
        suggestions = []
        
        if "k_plus_notation" in patterns:
            suggestions.append("Parse 'K+' notation: extract number, multiply 'K' by 1000, remove text")
        
        if "rating_with_text" in patterns:
            suggestions.append("Extract numeric rating value from text (e.g., '4.6 out of 5 stars' → 4.6)")
        
        if "currency_with_symbol" in patterns:
            suggestions.append("Remove currency symbols and convert to float")
        
        if "percentage" in patterns:
            suggestions.append("Remove % symbol and convert to decimal (45% → 0.45)")
        
        if "date_format" in patterns:
            suggestions.append("Convert to standardized datetime format")
        
        if "mixed_text_numbers" in patterns:
            suggestions.append("Decide if this should be numeric (extract numbers) or categorical (keep as text)")
        
        if "url" in patterns:
            suggestions.append("Keep as text or extract domain name if needed for analysis")
        
        if "delimited_values" in patterns:
            suggestions.append("Consider splitting into multiple columns or keeping as list")
        
        if "parentheses_numbers" in patterns:
            suggestions.append("Convert parentheses notation to negative numbers")
        
        # Missing value suggestions
        for issue in issues:
            if "missing" in issue["type"]:
                if "k_plus_notation" in patterns or "rating_with_text" in patterns:
                    suggestions.append("Fill missing values with 0 or median after parsing")
                else:
                    suggestions.append("Fill missing values with appropriate method (median/mode/Unknown)")
        
        return suggestions if suggestions else ["No special cleaning needed - standard processing"]
    
    def _generate_recommendations(self, df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate high-level cleaning recommendations
        """
        recommendations = {
            "priority_high": [],
            "priority_medium": [],
            "priority_low": [],
            "custom_parsers_needed": {}
        }
        
        for col_name, col_profile in profile["columns"].items():
            # Collect issues by priority
            for issue in col_profile["issues"]:
                priority = issue["severity"]
                rec = {
                    "column": col_name,
                    "issue": issue["description"],
                    "suggestions": col_profile["suggested_cleaning"]
                }
                
                if priority == "high":
                    recommendations["priority_high"].append(rec)
                elif priority == "medium":
                    recommendations["priority_medium"].append(rec)
                else:
                    recommendations["priority_low"].append(rec)
            
            # Identify columns needing custom parsers
            patterns = col_profile["detected_patterns"]
            if any(p in patterns for p in ["k_plus_notation", "rating_with_text", "parentheses_numbers"]):
                recommendations["custom_parsers_needed"][col_name] = {
                    "patterns": patterns,
                    "parser_function": self._generate_parser_name(patterns)
                }
        
        return recommendations
    
    def _generate_parser_name(self, patterns: List[str]) -> str:
        """Generate a descriptive parser function name"""
        if "k_plus_notation" in patterns:
            return "parse_k_notation"
        elif "rating_with_text" in patterns:
            return "extract_rating"
        elif "parentheses_numbers" in patterns:
            return "parse_parentheses_negative"
        else:
            return "custom_parser"
    
    def _generate_summary(self, profile: Dict[str, Any], recommendations: Dict[str, Any]) -> str:
        """
        Generate human-readable summary
        """
        lines = []
        lines.append(f"📊 Dataset Analysis Summary")
        lines.append(f"=" * 60)
        lines.append(f"Total Rows: {profile['shape']['rows']}")
        lines.append(f"Total Columns: {profile['shape']['columns']}")
        lines.append(f"")
        
        # Priority issues
        high_priority = len(recommendations["priority_high"])
        medium_priority = len(recommendations["priority_medium"])
        low_priority = len(recommendations["priority_low"])
        
        lines.append(f"🚨 Issues Found:")
        lines.append(f"   High Priority: {high_priority}")
        lines.append(f"   Medium Priority: {medium_priority}")
        lines.append(f"   Low Priority: {low_priority}")
        lines.append(f"")
        
        # Custom parsers needed
        if recommendations["custom_parsers_needed"]:
            lines.append(f"🔧 Custom Parsers Needed:")
            for col, info in recommendations["custom_parsers_needed"].items():
                lines.append(f"   - {col}: {', '.join(info['patterns'])}")
            lines.append(f"")
        
        # Top recommendations
        if recommendations["priority_high"]:
            lines.append(f"⚡ Top Priority Actions:")
            for rec in recommendations["priority_high"][:3]:
                lines.append(f"   - {rec['column']}: {rec['issue']}")
        
        return "\n".join(lines)
    
    def generate_ai_prompt(self, profile: Dict[str, Any]) -> str:
        """
        Generate prompt for GPT-4 to create custom cleaning code
        """
        prompt = f"""You are an expert data engineer. Analyze this dataset profile and generate Python cleaning code.

Dataset Profile:
{json.dumps(profile, indent=2)}

Task: Generate custom cleaning functions for problematic columns.

Requirements:
1. Create a function for each column that needs custom parsing
2. Handle edge cases (None, empty strings, invalid formats)
3. Include error handling
4. Add docstrings explaining the cleaning logic
5. Return clean, production-ready Python code

Format your response as executable Python code."""

        return prompt


# -----------------------------------------------------------
# 🎯 SMART CLEANING FUNCTIONS (AI-Inspired)
# -----------------------------------------------------------

def parse_k_notation(value: str) -> int:
    """
    Parse values like '6K+', '10K+', '300+' bought in past month
    
    Examples:
        '6K+ bought in past month' → 6000
        '300+ bought in past month' → 300
        '10K+' → 10000
    """
    if pd.isna(value) or value == "":
        return 0
    
    try:
        # Extract number and K suffix
        match = re.search(r'(\d+\.?\d*)(K?)\+?', str(value), re.IGNORECASE)
        if match:
            num = float(match.group(1))
            multiplier = 1000 if match.group(2).upper() == 'K' else 1
            return int(num * multiplier)
    except:
        pass
    
    return 0


def extract_rating(value: str) -> float:
    """
    Extract numeric rating from text like '4.6 out of 5 stars'
    
    Examples:
        '4.6 out of 5 stars' → 4.6
        '3.5 out of 5' → 3.5
    """
    if pd.isna(value) or value == "":
        return 0.0
    
    try:
        match = re.search(r'(\d+\.?\d*)\s*out of', str(value), re.IGNORECASE)
        if match:
            return float(match.group(1))
    except:
        pass
    
    return 0.0


def extract_price_from_variant(value: str) -> float:
    """
    Extract price from 'basic variant price: $162.24' format
    
    Examples:
        'basic variant price: $162.24' → 162.24
        'basic variant price: 2.4GHz' → 0.0 (not a price)
    """
    if pd.isna(value) or value == "":
        return 0.0
    
    try:
        # Look for currency symbol followed by number
        match = re.search(r'[$€£¥₹]\s*(\d+\.?\d*)', str(value))
        if match:
            return float(match.group(1))
    except:
        pass
    
    return 0.0


def apply_ai_cleaning(df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply AI-recommended cleaning to the dataset
    """
    df_clean = df.copy()
    
    # Apply custom parsers based on AI analysis
    parsers = analysis["cleaning_recommendations"].get("custom_parsers_needed", {})
    
    for col_name, parser_info in parsers.items():
        patterns = parser_info.get("patterns", [])
        
        if "k_plus_notation" in patterns:
            df_clean[col_name] = df_clean[col_name].apply(parse_k_notation)
            print(f"✅ Applied K+ notation parser to '{col_name}'")
        
        elif "rating_with_text" in patterns:
            df_clean[col_name] = df_clean[col_name].apply(extract_rating)
            print(f"✅ Applied rating extractor to '{col_name}'")
    
    # Handle price_on_variant specially if it exists
    if "price_on_variant" in df_clean.columns:
        df_clean["extracted_variant_price"] = df_clean["price_on_variant"].apply(extract_price_from_variant)
        print(f"✅ Extracted prices from 'price_on_variant' to new column")
    
    return df_clean