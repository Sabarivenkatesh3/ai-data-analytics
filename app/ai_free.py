# app/ai_free.py
"""
Free LLM Integration Module
Supports: Google Gemini, Groq, Hugging Face, Ollama
No paid API needed - Perfect for students and testing!
"""

import os
import json
import requests
from typing import Dict, Any, Optional

class FreeLLMClient:
    """
    Universal client for free LLMs
    Auto-detects which service to use based on available API keys
    """
    
    def __init__(self):
        """Initialize with available free LLM services"""
        self.provider = None
        self.api_key = None
        
        # Try to detect available service
        self._detect_provider()
    
    def _detect_provider(self):
        """Auto-detect which free LLM service is configured"""
        
        # Check for Google Gemini (RECOMMENDED)
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            self.provider = "gemini"
            self.api_key = gemini_key
            print("✅ Using Google Gemini (FREE)")
            return
        
        # Check for Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self.provider = "groq"
            self.api_key = groq_key
            print("✅ Using Groq (FREE & FAST)")
            return
        
        # Check for Hugging Face
        hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_key:
            self.provider = "huggingface"
            self.api_key = hf_key
            print("✅ Using Hugging Face (FREE)")
            return
        
        # Check if Ollama is running locally
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.provider = "ollama"
                print("✅ Using Ollama (LOCAL & FREE)")
                return
        except:
            pass
        
        # No provider found - will use pattern-based cleaning
        print("⚠️ No AI service configured - using smart pattern detection")
        self.provider = "none"
    
    def analyze_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze dataset and generate cleaning recommendations
        
        Args:
            dataset_info: Dictionary with dataset structure and sample data
        
        Returns:
            Cleaning recommendations from AI
        """
        if self.provider == "none":
            return self._fallback_analysis(dataset_info)
        
        prompt = self._create_analysis_prompt(dataset_info)
        
        try:
            if self.provider == "gemini":
                return self._call_gemini(prompt)
            elif self.provider == "groq":
                return self._call_groq(prompt)
            elif self.provider == "huggingface":
                return self._call_huggingface(prompt)
            elif self.provider == "ollama":
                return self._call_ollama(prompt)
        except Exception as e:
            print(f"⚠️ AI call failed: {e}")
            print("📊 Falling back to pattern-based analysis...")
            return self._fallback_analysis(dataset_info)
    
    def _create_analysis_prompt(self, dataset_info: Dict[str, Any]) -> str:
        """Create prompt for AI analysis"""
        prompt = f"""You are a data cleaning expert. Analyze this dataset and provide cleaning recommendations.

Dataset Information:
- Rows: {dataset_info.get('rows', 0)}
- Columns: {dataset_info.get('columns', [])}

Sample Data:
{json.dumps(dataset_info.get('sample_data', {}), indent=2)}

Column Details:
{json.dumps(dataset_info.get('column_info', {}), indent=2)}

Task: Analyze each column and identify:
1. Data quality issues
2. Format problems (e.g., "6K+" notation, "4.6 out of 5 stars", currency symbols)
3. Missing value patterns
4. Recommended cleaning operations

Respond ONLY with valid JSON in this exact format:
{{
  "columns": {{
    "column_name": {{
      "issue": "description of the issue",
      "pattern": "detected pattern (e.g., k_plus_notation, rating_with_text)",
      "recommendation": "what to do to fix it",
      "priority": "high/medium/low"
    }}
  }},
  "summary": "brief summary of main issues"
}}

IMPORTANT: Return ONLY the JSON object, no other text."""

        return prompt
    
    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """Call Google Gemini API (FREE)"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2048
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            # Extract JSON from response
            try:
                # Remove markdown code blocks if present
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                
                return json.loads(text.strip())
            except json.JSONDecodeError:
                print("⚠️ Could not parse AI response as JSON")
                return {"error": "Invalid JSON response"}
        else:
            raise Exception(f"Gemini API error: {response.status_code}")
    
    def _call_groq(self, prompt: str) -> Dict[str, Any]:
        """Call Groq API (FREE & FAST)"""
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-70b-versatile",  # Fast and smart
            "messages": [
                {"role": "system", "content": "You are a data cleaning expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2048
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            # Extract JSON
            try:
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                
                return json.loads(text.strip())
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response"}
        else:
            raise Exception(f"Groq API error: {response.status_code}")
    
    def _call_huggingface(self, prompt: str) -> Dict[str, Any]:
        """Call Hugging Face Inference API (FREE)"""
        url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            text = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
            
            try:
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                
                return json.loads(text.strip())
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response"}
        else:
            raise Exception(f"Hugging Face API error: {response.status_code}")
    
    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call local Ollama (100% FREE, runs on your computer)"""
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": "llama3",  # Or mistral, codellama, etc.
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }
        
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            text = result["response"]
            
            try:
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                
                return json.loads(text.strip())
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response"}
        else:
            raise Exception(f"Ollama error: {response.status_code}")
    
    def _fallback_analysis(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback to smart pattern-based analysis when no AI is available
        Still very intelligent - uses regex patterns and heuristics
        """
        print("📊 Using smart pattern detection (no AI needed)")
        
        columns_analysis = {}
        
        for col_name, col_data in dataset_info.get('column_info', {}).items():
            sample_values = col_data.get('sample_values', [])
            
            # Analyze patterns in sample values
            analysis = {
                "issue": None,
                "pattern": "standard",
                "recommendation": "Standard cleaning",
                "priority": "low"
            }
            
            # Check for K+ notation
            if any('K+' in str(v).upper() for v in sample_values):
                analysis = {
                    "issue": "Contains 'K+' notation - not numeric",
                    "pattern": "k_plus_notation",
                    "recommendation": "Extract number, multiply K by 1000",
                    "priority": "high"
                }
            
            # Check for rating format
            elif any('out of' in str(v).lower() for v in sample_values):
                analysis = {
                    "issue": "Rating text format",
                    "pattern": "rating_with_text",
                    "recommendation": "Extract numeric rating value",
                    "priority": "high"
                }
            
            # Check for currency
            elif any(any(c in str(v) for c in ['$', '€', '£', '¥', '₹']) for v in sample_values):
                analysis = {
                    "issue": "Contains currency symbols",
                    "pattern": "currency_with_symbol",
                    "recommendation": "Remove currency symbols, convert to float",
                    "priority": "medium"
                }
            
            # Check for percentage
            elif any('%' in str(v) for v in sample_values):
                analysis = {
                    "issue": "Percentage format",
                    "pattern": "percentage",
                    "recommendation": "Remove % and convert to decimal",
                    "priority": "medium"
                }
            
            columns_analysis[col_name] = analysis
        
        return {
            "columns": columns_analysis,
            "summary": f"Detected {len([c for c in columns_analysis.values() if c['priority'] in ['high', 'medium']])} columns needing special handling"
        }


# -----------------------------------------------------------
# 🎓 STUDENT-FRIENDLY SETUP GUIDE
# -----------------------------------------------------------

def setup_free_ai():
    """
    Interactive setup guide for students
    Helps choose and configure a free AI service
    """
    print("\n" + "=" * 70)
    print("🎓 FREE AI SETUP FOR STUDENTS")
    print("=" * 70)
    print("\nLet's set up a free AI service for your project!")
    print("\nAvailable FREE options:")
    print("1. Google Gemini (RECOMMENDED) - Unlimited, fast, no credit card")
    print("2. Groq - Super fast, unlimited")
    print("3. Hugging Face - Many models to choose from")
    print("4. Ollama - Run locally, 100% offline")
    print("\nEnter your choice (1-4): ", end="")
    
    # This is a helper - actual implementation would be interactive
    print("\n\n📝 QUICK SETUP INSTRUCTIONS:")
    print("\n1. FOR GOOGLE GEMINI (EASIEST):")
    print("   - Go to: https://makersuite.google.com/app/apikey")
    print("   - Click 'Create API Key'")
    print("   - Copy the key")
    print("   - Add to your project:")
    print("     export GEMINI_API_KEY='your-key-here'")
    print("\n2. FOR GROQ (FASTEST):")
    print("   - Go to: https://console.groq.com/")
    print("   - Sign up")
    print("   - Create API key")
    print("   - Add: export GROQ_API_KEY='your-key-here'")
    print("\n3. FOR OLLAMA (LOCAL):")
    print("   - Install: curl -fsSL https://ollama.com/install.sh | sh")
    print("   - Run: ollama pull llama3")
    print("   - Done! No API key needed")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    setup_free_ai()