import json
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime

class MatchCriteria(BaseModel):
    type: str  # exact, fuzzy, amount, date, composite
    source_columns: List[str]
    target_columns: List[str]
    confidence: float
    reasoning: str
    options: Dict[str, Any] = Field(default_factory=dict)

class RiskFactor(BaseModel):
    type: str  # amount, date, pattern, business_rule
    description: str
    severity: str  # high, medium, low
    threshold: Optional[float] = None
    affected_items: List[Dict[str, Any]] = Field(default_factory=list)

class CategorizationRule(BaseModel):
    category: str
    conditions: List[Dict[str, Any]]
    description: str
    priority: int

class BusinessRule(BaseModel):
    name: str
    condition: str
    action: str
    severity: str
    threshold: Optional[float] = None

class HistoricalPattern(BaseModel):
    pattern_type: str  # reference, description, amount, date
    pattern: str
    confidence: float
    examples: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)

class Rule(BaseModel):
    name: str
    type: str
    source_columns: List[str]
    target_columns: List[str]
    options: Dict[str, Any] = Field(default_factory=dict)

class LLMOutput(BaseModel):
    clarification_needed: str
    reconciliation_type: str
    confidence: float
    reasoning: str
    keys: List[str]
    keys_reasoning: str
    rules: List[Rule] = Field(default_factory=list)
    match_criteria: List[MatchCriteria] = Field(default_factory=list)
    matched_items: List[Dict[str, Any]] = Field(default_factory=list)
    unmatched_items: List[Dict[str, Any]] = Field(default_factory=list)
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    categorization_rules: List[CategorizationRule] = Field(default_factory=list)
    business_rules: List[BusinessRule] = Field(default_factory=list)
    historical_patterns: List[HistoricalPattern] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)

class LLMAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the LLM analyzer.
        
        Args:
            api_key (str): OpenAI API key
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
    
    def analyze_files(self, df1: pd.DataFrame, df2: pd.DataFrame, clarification_context: str = "") -> Dict[str, Any]:
        """Analyze two dataframes and generate matching rules, with optional clarification context."""
        prompt = self._build_analysis_prompt(df1, df2, clarification_context)
        response = self._call_llm(prompt)
        return self._parse_rules(response)
    
    def _build_analysis_prompt(self, df1: pd.DataFrame, df2: pd.DataFrame, clarification_context: str = "") -> str:
        """Build the prompt for LLM analysis.
        
        Args:
            df1 (pd.DataFrame): First dataframe
            df2 (pd.DataFrame): Second dataframe
            clarification_context (str): Optional clarification context
            
        Returns:
            str: Formatted prompt for LLM
        """
        base_prompt = f"""
You are a senior accountant performing a reconciliation. Follow these steps, and ask for clarification if you are unsure at any point:

1. **Identify the reconciliation type** (e.g., bank-to-GL, AP-to-vendor, AR-to-customer, etc.).
   - If you are not sure, ask the user for clarification before proceeding.

2. **Identify the most likely key(s) for matching** (e.g., date, amount, customer ID, PO ID, invoice number, etc.).
   - There may be more than one key. Use the highest-confidence set of keys first.
   - If you are unsure, ask the user for clarification.
   - List the keys and provide reasoning for their selection.

3. **High-confidence matching:**
   - Use the identified keys to match items between the two datasets.
   - Show matched items, the keys used, and the confidence/reasoning for each match.

4. **Handle unmatched items:**
   - For items that remain unmatched, check for common reconciliation issues:
     - Timing differences (e.g., date off by a few days)
     - Split payments (one-to-many or many-to-one matches)
     - Reference mismatches (e.g., typo in PO ID)
   - If you are unsure, ask the user for clarification.

5. **List completely unmatched items:**
   - Clearly list items that could not be matched by any rule or heuristic.

6. **Historical pattern check:**
   - If historical data is available, check for recurring issues (e.g., recurring timing differences, regular split payments, etc.).
   - Note any patterns and suggest possible resolutions.

---

File 1 Schema:
{df1.dtypes.to_string()}

File 1 Sample (first 5 rows):
{df1.head().to_string()}

File 2 Schema:
{df2.dtypes.to_string()}

File 2 Sample (first 5 rows):
{df2.head().to_string()}

---
"""
        if clarification_context:
            base_prompt += f"\nAdditional user clarification/context so far:\n{clarification_context}\n---\n"
        base_prompt += """
Based on the above, analyze and return a JSON object with the following structure:
{
    "clarification_needed": "string (if you need clarification, state the question here, otherwise empty)",
    "reconciliation_type": "string (bank_to_gl, invoice_to_payment, etc.)",
    "confidence": float (0-1),
    "reasoning": "string (explanation of reconciliation type)",
    "keys": ["string (list of keys used for matching)"],
    "keys_reasoning": "string (reasoning for key selection)",
    "rules": [
        {
            "name": "string",
            "type": "string (exact, fuzzy, amount, date, composite)",
            "source_columns": ["string"],
            "target_columns": ["string"],
            "options": {}
        }
    ],
    "match_criteria": [
        {
            "type": "string",
            "source_columns": ["string"],
            "target_columns": ["string"],
            "confidence": float,
            "reasoning": "string",
            "options": {}
        }
    ],
    "matched_items": [
        {
            "source_index": int,
            "target_index": int,
            "keys_used": ["string"],
            "confidence": float,
            "reasoning": "string"
        }
    ],
    "unmatched_items": [
        {
            "index": int,
            "side": "source|target",
            "reason": "string (timing difference, split payment, reference mismatch, unknown, etc.)",
            "clarification_needed": "string (if you need clarification, state the question here, otherwise empty)"
        }
    ],
    "risk_factors": [
        {
            "type": "string",
            "description": "string",
            "severity": "string (high, medium, low)",
            "threshold": float,
            "affected_items": [
                {"index": int, "side": "source|target", "reason": "string"}
            ]
        }
    ],
    "categorization_rules": [
        {
            "category": "string",
            "conditions": [
                {"column": "string", "operator": "string", "value": "string or number"}
            ],
            "description": "string",
            "priority": int
        }
    ],
    "business_rules": [
        {
            "name": "string",
            "condition": "string",
            "action": "string",
            "severity": "string",
            "threshold": float
        }
    ],
    "historical_patterns": [
        {
            "pattern_type": "string",
            "pattern": "string",
            "confidence": float,
            "examples": ["string"]
        }
    ],
    "summary": {
        "total_transactions": int,
        "matched_transactions": int,
        "unmatched_transactions": int,
        "high_risk_items": int,
        "data_quality_score": float
    }
}

IMPORTANT:
1. Follow the stepwise process above. If you need clarification, set the 'clarification_needed' field and stop.
2. Return ONLY the JSON object, no other text
3. Ensure all JSON is properly formatted
4. All required fields must be present
5. Use appropriate data types for each field
6. Keep descriptions concise but informative
7. For historical_patterns.examples, use an array of strings, not objects
8. For any field that is a list of objects (e.g., affected_items, conditions), do NOT use plain numbers or strings. Always use a list of dictionaries with the required keys as shown in the example above.
"""
        return base_prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the analysis prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: LLM response
        """
        response = self.client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a data reconciliation expert with deep knowledge of accounting, banking, and financial systems. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _parse_rules(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a rules dictionary and validate with Pydantic."""
        try:
            # Clean the response string
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Parse and validate
            data = json.loads(response)
            validated = LLMOutput.model_validate(data)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            print("\n[LLM RAW OUTPUT]\n", response)
            raise ValueError(f"Failed to parse and validate LLM response: {e}") 