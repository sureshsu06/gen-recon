import os
import pandas as pd
from dotenv import load_dotenv
from llm_analyzer import LLMAnalyzer
from rule_executor import RuleExecutor
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class MVPReconciler:
    def __init__(self, api_key=None):
        """Initialize the reconciliation engine.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to get from environment.
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided and not found in environment")
        
        self.analyzer = LLMAnalyzer(api_key)
        self.executor = RuleExecutor()

    def reconcile(self, file1_path: str, file2_path: str) -> dict:
        """Reconcile two CSV files with interactive clarification loop."""
        # Load CSV files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        clarification_context = ""
        while True:
            # Get LLM analysis and rules, passing clarification context if any
            analysis = self.analyzer.analyze_files(df1, df2, clarification_context)
            if analysis.get('clarification_needed'):
                print(f"\nLLM needs clarification: {analysis['clarification_needed']}")
                user_input = input("Your clarification: ")
                clarification_context += f"\nUser clarification: {user_input}"
            else:
                break

        # Execute rules
        results = self.executor.execute(df1, df2, analysis['rules'])
        
        # Generate detailed reports
        detailed_results = self._generate_detailed_reports(df1, df2, results, analysis)
        
        return detailed_results

    def _generate_detailed_reports(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                 results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed reconciliation reports.
        
        Args:
            df1 (pd.DataFrame): First dataframe
            df2 (pd.DataFrame): Second dataframe
            results (dict): Basic reconciliation results
            analysis (dict): LLM analysis results
            
        Returns:
            dict: Detailed reconciliation reports
        """
        detailed_results = {
            'basic_results': results,
            'analysis': analysis,
            'detailed_match_report': self._generate_match_report(df1, df2, results, analysis),
            'categorized_unmatched': self._categorize_unmatched_items(df1, df2, results, analysis),
            'exception_analysis': self._analyze_exceptions(results, analysis),
            'reconciliation_summary': self._generate_reconciliation_summary(df1, df2, results, analysis),
            'actionable_items': self._generate_actionable_items(results, analysis)
        }
        
        return detailed_results

    def _generate_match_report(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                             results: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed match report for each matched/pending item."""
        match_report = []
        
        for match in results.get('matched', []):
            source_item = df1.iloc[match['source_index']]
            target_item = df2.iloc[match['target_index']]
            
            match_report.append({
                'source': {
                    'date': source_item.get('date', ''),
                    'description': source_item.get('description', ''),
                    'reference': source_item.get('reference', ''),
                    'amount': source_item.get('amount', 0)
                },
                'target': {
                    'date': target_item.get('date', ''),
                    'description': target_item.get('description', ''),
                    'reference': target_item.get('reference', ''),
                    'amount': target_item.get('amount', 0)
                },
                'match_confidence': match.get('confidence', 0),
                'match_criteria': match.get('criteria', ''),
                'days_difference': self._calculate_days_difference(
                    source_item.get('date', ''), 
                    target_item.get('date', '')
                )
            })
        
        return match_report

    def _categorize_unmatched_items(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                  results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize unmatched items into meaningful groups."""
        categories = {
            'bank_only': {
                'bank_fees': [],
                'interest_earned': [],
                'other': []
            },
            'gl_only': {
                'outstanding_checks': [],
                'deposits_in_transit': [],
                'future_dated_entries': [],
                'other': []
            }
        }
        
        # Apply categorization rules from analysis
        for rule in analysis.get('categorization_rules', []):
            self._apply_categorization_rule(rule, df1, df2, results, categories)
        
        return categories

    def _analyze_exceptions(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze and identify high-risk items requiring review."""
        exceptions = []
        
        # Apply risk factors from analysis
        for risk in analysis.get('risk_factors', []):
            self._apply_risk_factor(risk, results, exceptions)
        
        return exceptions

    def _generate_reconciliation_summary(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                       results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reconciliation summary with adjusted balances."""
        # Find the best rule for balance calculation from LLM output
        bank_rule = self._find_balance_rule(analysis, df1)
        gl_rule = self._find_balance_rule(analysis, df2)

        summary = {
            'bank_statement_balance': self._calculate_balance(df1, bank_rule),
            'outstanding_checks': self._calculate_outstanding_checks(results),
            'deposits_in_transit': self._calculate_deposits_in_transit(results),
            'other_adjustments': self._calculate_other_adjustments(results),
            'adjusted_bank_balance': 0,  # Calculated below
            'gl_cash_balance': self._calculate_balance(df2, gl_rule),
            'unrecorded_fees': self._calculate_unrecorded_fees(results),
            'unrecorded_interest': self._calculate_unrecorded_interest(results),
            'adjusted_gl_balance': 0,  # Calculated below
            'difference': 0  # Calculated below
        }
        
        # Calculate adjusted balances
        summary['adjusted_bank_balance'] = (
            summary['bank_statement_balance'] -
            summary['outstanding_checks'] +
            summary['deposits_in_transit'] +
            summary['other_adjustments']
        )
        
        summary['adjusted_gl_balance'] = (
            summary['gl_cash_balance'] -
            summary['unrecorded_fees'] +
            summary['unrecorded_interest']
        )
        
        summary['difference'] = (
            summary['adjusted_bank_balance'] -
            summary['adjusted_gl_balance']
        )
        
        return summary

    def _find_balance_rule(self, analysis: Dict[str, Any], df: pd.DataFrame) -> dict:
        """Find the best rule for balance calculation from LLM output."""
        # Look for a composite or amount rule that uses Debit/Credit or amount columns
        for rule in analysis.get('rules', []):
            cols = set(rule.get('source_columns', []) + rule.get('target_columns', []))
            if any(col.lower() in [c.lower() for c in df.columns] for col in cols):
                if rule.get('type') in ['composite', 'amount']:
                    return rule
        return None

    def _calculate_balance(self, df: pd.DataFrame, rule: dict = None) -> float:
        """Calculate total balance from a dataframe using LLM rule if available."""
        if rule and 'options' in rule and 'balance_formula' in rule['options']:
            formula = rule['options']['balance_formula']
            local_vars = {col: df[col].sum() for col in df.columns if col in formula}
            try:
                return eval(formula, {}, local_vars)
            except Exception:
                pass
        # Try Debit/Credit
        if 'Debit' in df.columns and 'Credit' in df.columns:
            return df['Credit'].sum() - df['Debit'].sum()
        elif 'amount' in df.columns:
            return df['amount'].sum()
        else:
            return 0.0

    def _calculate_outstanding_checks(self, results: Dict[str, Any]) -> float:
        """Calculate total outstanding checks."""
        return sum(item.get('amount', 0) for item in results.get('unmatched_source', [])
                  if item.get('type') == 'check')

    def _calculate_deposits_in_transit(self, results: Dict[str, Any]) -> float:
        """Calculate total deposits in transit."""
        return sum(item.get('amount', 0) for item in results.get('unmatched_source', [])
                  if item.get('type') == 'deposit')

    def _calculate_other_adjustments(self, results: Dict[str, Any]) -> float:
        """Calculate total other adjustments."""
        return sum(item.get('amount', 0) for item in results.get('unmatched_source', [])
                  if item.get('type') not in ['check', 'deposit'])

    def _calculate_unrecorded_fees(self, results: Dict[str, Any]) -> float:
        """Calculate total unrecorded fees."""
        return sum(item.get('amount', 0) for item in results.get('unmatched_target', [])
                  if item.get('type') == 'fee')

    def _calculate_unrecorded_interest(self, results: Dict[str, Any]) -> float:
        """Calculate total unrecorded interest."""
        return sum(item.get('amount', 0) for item in results.get('unmatched_target', [])
                  if item.get('type') == 'interest')

    def _apply_categorization_rule(self, rule: Dict[str, Any], df1: pd.DataFrame, 
                                 df2: pd.DataFrame, results: Dict[str, Any],
                                 categories: Dict[str, Any]) -> None:
        """Apply a categorization rule to unmatched items."""
        # Implementation depends on specific rule structure
        pass

    def _apply_risk_factor(self, risk: Dict[str, Any], results: Dict[str, Any],
                          exceptions: List[Dict[str, Any]]) -> None:
        """Apply a risk factor to identify exceptions."""
        # Implementation depends on specific risk factor structure
        pass

    def _apply_business_rule(self, rule: Dict[str, Any], results: Dict[str, Any],
                            actionable_items: List[Dict[str, Any]]) -> None:
        """Apply a business rule to generate actionable items."""
        # Implementation depends on specific business rule structure
        pass

    def _calculate_days_difference(self, date1: str, date2: str) -> int:
        """Calculate days difference between two dates."""
        try:
            d1 = datetime.strptime(date1, '%Y-%m-%d')
            d2 = datetime.strptime(date2, '%Y-%m-%d')
            return abs((d2 - d1).days)
        except (ValueError, TypeError):
            return 0

    def _generate_actionable_items(self, results: Dict[str, Any], 
                                 analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate list of actionable items requiring attention."""
        actionable_items = []
        
        # Apply business rules from analysis
        for rule in analysis.get('business_rules', []):
            self._apply_business_rule(rule, results, actionable_items)
        
        return actionable_items

def main():
    """Example usage of the reconciliation engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reconcile two CSV files')
    parser.add_argument('file1', help='Path to first CSV file')
    parser.add_argument('file2', help='Path to second CSV file')
    parser.add_argument('--output', help='Path to output JSON file', default='reconciliation_results.json')
    args = parser.parse_args()
    
    reconciler = MVPReconciler()
    results = reconciler.reconcile(args.file1, args.file2)
    
    # Save results to JSON file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Print summary
    print("\nReconciliation Results:")
    print(f"Results saved to {args.output}")
    
    # Print key metrics
    summary = results['reconciliation_summary']
    print(f"\nBank Statement Balance: ${summary['bank_statement_balance']:,.2f}")
    print(f"Less: Outstanding Checks: (${summary['outstanding_checks']:,.2f})")
    print(f"Add: Deposits in Transit: ${summary['deposits_in_transit']:,.2f}")
    print(f"Add/Less: Other Adjustments: ${summary['other_adjustments']:,.2f}")
    print(f"Adjusted Bank Balance: ${summary['adjusted_bank_balance']:,.2f}")
    print(f"\nGL Cash Balance: ${summary['gl_cash_balance']:,.2f}")
    print(f"Less: Unrecorded Fees: (${summary['unrecorded_fees']:,.2f})")
    print(f"Add: Unrecorded Interest: ${summary['unrecorded_interest']:,.2f}")
    print(f"Adjusted GL Balance: ${summary['adjusted_gl_balance']:,.2f}")
    print(f"\nDifference: ${summary['difference']:,.2f}")

if __name__ == '__main__':
    main() 