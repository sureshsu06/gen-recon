from typing import Dict, List, Any
import pandas as pd
from fuzzywuzzy import fuzz
from datetime import datetime
import re
from difflib import SequenceMatcher

class RuleExecutor:
    def __init__(self):
        """Initialize the rule executor."""
        pass

    def execute(self, df1: pd.DataFrame, df2: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute reconciliation rules on two dataframes.
        
        Args:
            df1 (pd.DataFrame): First dataframe
            df2 (pd.DataFrame): Second dataframe
            rules (List[Dict[str, Any]]): List of rules to execute
            
        Returns:
            dict: Reconciliation results
        """
        results = {
            'matched': [],
            'unmatched_source': [],
            'unmatched_target': [],
            'pending_review': []
        }
        
        # Create copies of dataframes to track matched items
        df1_matched = pd.Series(False, index=df1.index)
        df2_matched = pd.Series(False, index=df2.index)
        
        # Execute each rule
        for rule in rules:
            self._execute_rule(rule, df1, df2, df1_matched, df2_matched, results)
        
        # Add unmatched items
        self._add_unmatched_items(df1, df2, df1_matched, df2_matched, results)
        
        return results

    def _execute_rule(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                     df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute a single reconciliation rule.
        
        Args:
            rule (Dict[str, Any]): Rule to execute
            df1 (pd.DataFrame): First dataframe
            df2 (pd.DataFrame): Second dataframe
            df1_matched (pd.Series): Boolean series tracking matched items in df1
            df2_matched (pd.Series): Boolean series tracking matched items in df2
            results (Dict[str, Any]): Results dictionary to update
        """
        rule_type = rule.get('type', '')
        
        if rule_type == 'exact':
            self._execute_exact_match(rule, df1, df2, df1_matched, df2_matched, results)
        elif rule_type == 'fuzzy':
            self._execute_fuzzy_match(rule, df1, df2, df1_matched, df2_matched, results)
        elif rule_type == 'amount':
            self._execute_amount_match(rule, df1, df2, df1_matched, df2_matched, results)
        elif rule_type == 'date':
            self._execute_date_match(rule, df1, df2, df1_matched, df2_matched, results)
        elif rule_type == 'composite':
            self._execute_composite_match(rule, df1, df2, df1_matched, df2_matched, results)

    def _execute_exact_match(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                           df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute exact matching rule."""
        source_cols = rule.get('source_columns', [])
        target_cols = rule.get('target_columns', [])
        
        for idx1 in df1.index:
            if df1_matched[idx1]:
                continue
                
            for idx2 in df2.index:
                if df2_matched[idx2]:
                    continue
                
                match = True
                for src_col, tgt_col in zip(source_cols, target_cols):
                    if df1.loc[idx1, src_col] != df2.loc[idx2, tgt_col]:
                        match = False
                        break
                
                if match:
                    results['matched'].append({
                        'source_index': idx1,
                        'target_index': idx2,
                        'confidence': 1.0,
                        'criteria': f"Exact match on {', '.join(source_cols)}",
                        'rule_name': rule.get('name', '')
                    })
                    df1_matched[idx1] = True
                    df2_matched[idx2] = True

    def _execute_fuzzy_match(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                           df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute fuzzy matching rule."""
        source_cols = rule.get('source_columns', [])
        target_cols = rule.get('target_columns', [])
        threshold = rule.get('options', {}).get('threshold', 0.8)
        
        for idx1 in df1.index:
            if df1_matched[idx1]:
                continue
                
            for idx2 in df2.index:
                if df2_matched[idx2]:
                    continue
                
                match = True
                confidence = 1.0
                
                for src_col, tgt_col in zip(source_cols, target_cols):
                    val1 = str(df1.loc[idx1, src_col])
                    val2 = str(df2.loc[idx2, tgt_col])
                    similarity = SequenceMatcher(None, val1, val2).ratio()
                    
                    if similarity < threshold:
                        match = False
                        break
                    
                    confidence *= similarity
                
                if match:
                    results['matched'].append({
                        'source_index': idx1,
                        'target_index': idx2,
                        'confidence': confidence,
                        'criteria': f"Fuzzy match on {', '.join(source_cols)}",
                        'rule_name': rule.get('name', '')
                    })
                    df1_matched[idx1] = True
                    df2_matched[idx2] = True

    def _execute_amount_match(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                            df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute amount matching rule."""
        source_col = rule.get('source_columns', [''])[0]
        target_col = rule.get('target_columns', [''])[0]
        tolerance = rule.get('options', {}).get('tolerance', 0.01)
        
        for idx1 in df1.index:
            if df1_matched[idx1]:
                continue
                
            for idx2 in df2.index:
                if df2_matched[idx2]:
                    continue
                
                amount1 = float(df1.loc[idx1, source_col])
                amount2 = float(df2.loc[idx2, target_col])
                
                if abs(amount1 - amount2) <= tolerance:
                    results['matched'].append({
                        'source_index': idx1,
                        'target_index': idx2,
                        'confidence': 1.0 - (abs(amount1 - amount2) / max(abs(amount1), abs(amount2))),
                        'criteria': f"Amount match within {tolerance} tolerance",
                        'rule_name': rule.get('name', '')
                    })
                    df1_matched[idx1] = True
                    df2_matched[idx2] = True

    def _execute_date_match(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                          df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute date matching rule."""
        source_col = rule.get('source_columns', [''])[0]
        target_col = rule.get('target_columns', [''])[0]
        max_days = rule.get('options', {}).get('max_days', 7)
        
        for idx1 in df1.index:
            if df1_matched[idx1]:
                continue
                
            for idx2 in df2.index:
                if df2_matched[idx2]:
                    continue
                
                try:
                    date1 = datetime.strptime(str(df1.loc[idx1, source_col]), '%Y-%m-%d')
                    date2 = datetime.strptime(str(df2.loc[idx2, target_col]), '%Y-%m-%d')
                    days_diff = abs((date2 - date1).days)
                    
                    if days_diff <= max_days:
                        results['matched'].append({
                            'source_index': idx1,
                            'target_index': idx2,
                            'confidence': 1.0 - (days_diff / max_days),
                            'criteria': f"Date match within {max_days} days",
                            'rule_name': rule.get('name', '')
                        })
                        df1_matched[idx1] = True
                        df2_matched[idx2] = True
                except (ValueError, TypeError):
                    continue

    def _execute_composite_match(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                               df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute composite matching rule."""
        source_cols = rule.get('source_columns', [])
        target_cols = rule.get('target_columns', [])
        weights = rule.get('options', {}).get('weights', {})
        threshold = rule.get('options', {}).get('threshold', 0.8)
        
        for idx1 in df1.index:
            if df1_matched[idx1]:
                continue
                
            for idx2 in df2.index:
                if df2_matched[idx2]:
                    continue
                
                confidence = 0.0
                total_weight = 0.0
                
                for src_col, tgt_col in zip(source_cols, target_cols):
                    weight = weights.get(src_col, 1.0)
                    total_weight += weight
                    
                    val1 = str(df1.loc[idx1, src_col])
                    val2 = str(df2.loc[idx2, tgt_col])
                    
                    if val1 == val2:
                        confidence += weight
                    else:
                        similarity = SequenceMatcher(None, val1, val2).ratio()
                        confidence += weight * similarity
                
                if total_weight > 0:
                    confidence /= total_weight
                
                if confidence >= threshold:
                    results['matched'].append({
                        'source_index': idx1,
                        'target_index': idx2,
                        'confidence': confidence,
                        'criteria': f"Composite match using {', '.join(source_cols)}",
                        'rule_name': rule.get('name', '')
                    })
                    df1_matched[idx1] = True
                    df2_matched[idx2] = True

    def _add_unmatched_items(self, df1: pd.DataFrame, df2: pd.DataFrame,
                            df1_matched: pd.Series, df2_matched: pd.Series,
                            results: Dict[str, Any]) -> None:
        """Add unmatched items to results."""
        # Add unmatched items from source
        for idx in df1.index:
            if not df1_matched[idx]:
                item = df1.loc[idx].to_dict()
                item['index'] = idx
                item['type'] = self._infer_transaction_type(item)
                results['unmatched_source'].append(item)
        
        # Add unmatched items from target
        for idx in df2.index:
            if not df2_matched[idx]:
                item = df2.loc[idx].to_dict()
                item['index'] = idx
                item['type'] = self._infer_transaction_type(item)
                results['unmatched_target'].append(item)

    def _infer_transaction_type(self, item: Dict[str, Any]) -> str:
        """Infer transaction type from item data."""
        description = str(item.get('description', '')).lower()
        amount = float(item.get('amount', 0))
        
        # Check for bank fees
        if any(term in description for term in ['fee', 'charge', 'service']):
            return 'fee'
        
        # Check for interest
        if 'interest' in description:
            return 'interest'
        
        # Check for checks
        if any(term in description for term in ['check', 'chk']):
            return 'check'
        
        # Check for deposits
        if any(term in description for term in ['deposit', 'dep']):
            return 'deposit'
        
        # Default to other
        return 'other' 