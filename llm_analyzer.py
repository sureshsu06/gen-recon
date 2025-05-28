import json
from typing import Dict, Any, List, Optional
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field, ValidationError

class Rule(BaseModel):
    name: str
    type: str
    source_columns: List[str]
    target_columns: List[str]
    options: Dict[str, Any] = Field(default_factory=dict)

class LLMOutput(BaseModel):
    reconciliation_type: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    rules: List[Rule] = Field(default_factory=list)

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
    
    def analyze_files(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """Analyze two dataframes and generate matching rules.
        
        Args:
            df1 (pd.DataFrame): First dataframe
            df2 (pd.DataFrame): Second dataframe
            
        Returns:
            dict: Analysis results including reconciliation type and rules
        """
        prompt = self._build_analysis_prompt(df1, df2)
        response = self._call_llm(prompt)
        return self._parse_rules(response)
    
    def _build_analysis_prompt(self, df1: pd.DataFrame, df2: pd.DataFrame) -> str:
        """Build the prompt for LLM analysis.
        
        Args:
            df1 (pd.DataFrame): First dataframe
            df2 (pd.DataFrame): Second dataframe
            
        Returns:
            str: Formatted prompt for LLM
        """
        return f"""
        I need to reconcile these two datasets. Analyze them and tell me:
        1. What columns should be used for matching
        2. What type of matching (exact, fuzzy, amount with tolerance)
        3. Any data transformations needed
        
        File 1 Schema:
        {df1.dtypes.to_string()}
        
        File 1 Sample (first 5 rows):
        {df1.head().to_string()}
        
        File 2 Schema:
        {df2.dtypes.to_string()}
        
        File 2 Sample (first 5 rows):
        {df2.head().to_string()}
        
        Based on the data patterns, first identify what type of reconciliation this is:
        - bank_to_gl: Bank statements matching to general ledger
        - invoice_to_payment: Customer invoices matching to payments received
        - purchase_order_to_invoice: PO matching to vendor invoices
        - inventory_physical_to_system: Physical counts to system records
        - payroll_to_bank: Payroll register to bank disbursements
        - vendor_statement: Vendor statements to AP records
        - custom: Other reconciliation type (describe it)
        
        Return ONLY a single JSON object with the following top-level keys: reconciliation_type, confidence, reasoning, rules.
        The rules field MUST be a list of rule objects, each with:
        - name (string)
        - type ("exact" | "fuzzy" | "amount")
        - source_columns (list of strings)
        - target_columns (list of strings)
        - options (object, can be empty)
        Do NOT return a dictionary for rules. Only return a list of rule objects.
        """
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the analysis prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: LLM response
        """
        response = self.client.chat.completions.create(
            model="gpt-4.1-2025-04-14",  # Using the specified model
            messages=[
                {"role": "system", "content": "You are a data reconciliation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low temperature for more deterministic output
        )
        return response.choices[0].message.content
    
    def _parse_rules(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a rules dictionary and validate with Pydantic."""
        try:
            data = json.loads(response)
            validated = LLMOutput.model_validate(data)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            print("\n[LLM RAW OUTPUT]\n", response)
            raise ValueError(f"Failed to parse and validate LLM response: {e}") 