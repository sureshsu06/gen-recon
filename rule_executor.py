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
        """Execute reconciliation rules on two dataframes: strict one-to-one matching first, then a flexible second pass."""
        results = {
            'matched': [],
            'flexible_matched': [],
            'unmatched_source': [],
            'unmatched_target': [],
            'pending_review': []
        }

        # Track matched rows by integer position
        df1_matched = pd.Series(False, index=range(len(df1)))
        df2_matched = pd.Series(False, index=range(len(df2)))

        # --- Stage 1: Strict One-to-One Matching ---
        for rule in rules:
            if rule.get('options', {}).get('allow_one_to_many', False):
                continue  # Skip one-to-many rules in this pass
            self._execute_one_to_one_rule(rule, df1, df2, df1_matched, df2_matched, results)

        # --- Stage 2: Flexible Matching on Unmatched Items ---
        # Only run if there are unmatched items and there are rules with 'flexible' option
        unmatched_df1 = df1[~df1_matched.values].reset_index(drop=True)
        unmatched_df2 = df2[~df2_matched.values].reset_index(drop=True)
        if not unmatched_df1.empty and not unmatched_df2.empty:
            for rule in rules:
                if not rule.get('options', {}).get('flexible', False):
                    continue  # Only process flexible rules in this pass
                self._execute_flexible_rule(rule, unmatched_df1, unmatched_df2, results)

        # Add unmatched items
        self._add_unmatched_items(df1, df2, df1_matched, df2_matched, results)

        # Debug: Check for overlap between matched and unmatched
        matched_src_indices = set(m['source_index'] for m in results['matched'])
        unmatched_src_indices = set(item['index'] for item in results['unmatched_source'])
        overlap = matched_src_indices & unmatched_src_indices
        if overlap:
            print(f"[DEBUG] Overlap in matched and unmatched_source indices: {overlap}")

        matched_tgt_indices = set(m['target_index'] for m in results['matched'])
        unmatched_tgt_indices = set(item['index'] for item in results['unmatched_target'])
        overlap_tgt = matched_tgt_indices & unmatched_tgt_indices
        if overlap_tgt:
            print(f"[DEBUG] Overlap in matched and unmatched_target indices: {overlap_tgt}")

        return results

    def _execute_one_to_one_rule(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                                df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Apply a rule for one-to-one matching only."""
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

    def _execute_one_to_many_rule(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                                  df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Apply a rule for one-to-many or many-to-one matching. (Stub: implement your logic here)"""
        # Example: For each unmatched source, try to find a group of unmatched targets that sum to the source amount (or vice versa)
        # This is a stub for demonstration; you can expand this logic as needed for your use case.
        pass

    def _add_match(self, results, df1, df2, idx1, idx2, confidence, criteria, rule_name):
        source_item = df1.iloc[idx1].to_dict()
        target_item = df2.iloc[idx2].to_dict()
        results['matched'].append({
            'source_index': idx1,
            'target_index': idx2,
            'source_transaction': source_item,
            'target_transaction': target_item,
            'confidence': confidence,
            'criteria': criteria,
            'rule_name': rule_name
        })

    def _execute_exact_match(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                           df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute exact matching rule."""
        source_cols = rule.get('source_columns', [])
        target_cols = rule.get('target_columns', [])
        for idx1 in range(len(df1)):
            if df1_matched[idx1]:
                continue
            for idx2 in range(len(df2)):
                if df2_matched[idx2]:
                    continue
                match = True
                for src_col, tgt_col in zip(source_cols, target_cols):
                    if df1.iloc[idx1][src_col] != df2.iloc[idx2][tgt_col]:
                        match = False
                        break
                if match:
                    self._add_match(results, df1, df2, idx1, idx2, 1.0, f"Exact match on {', '.join(source_cols)}", rule.get('name', ''))
                    df1_matched[idx1] = True
                    df2_matched[idx2] = True
                    break

    def _execute_fuzzy_match(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                           df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute fuzzy matching rule."""
        source_cols = rule.get('source_columns', [])
        target_cols = rule.get('target_columns', [])
        threshold = rule.get('options', {}).get('threshold', 0.8)
        for idx1 in range(len(df1)):
            if df1_matched[idx1]:
                continue
            for idx2 in range(len(df2)):
                if df2_matched[idx2]:
                    continue
                match = True
                confidence = 1.0
                for src_col, tgt_col in zip(source_cols, target_cols):
                    val1 = str(df1.iloc[idx1][src_col])
                    val2 = str(df2.iloc[idx2][tgt_col])
                    similarity = SequenceMatcher(None, val1, val2).ratio()
                    if similarity < threshold:
                        match = False
                        break
                    confidence *= similarity
                if match:
                    self._add_match(results, df1, df2, idx1, idx2, confidence, f"Fuzzy match on {', '.join(source_cols)}", rule.get('name', ''))
                    df1_matched[idx1] = True
                    df2_matched[idx2] = True
                    break

    def _execute_amount_match(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                            df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute amount matching rule."""
        source_col = rule.get('source_columns', [''])[0]
        target_col = rule.get('target_columns', [''])[0]
        tolerance = rule.get('options', {}).get('tolerance', 0.01)
        for idx1 in range(len(df1)):
            if df1_matched[idx1]:
                continue
            for idx2 in range(len(df2)):
                if df2_matched[idx2]:
                    continue
                amount1 = float(df1.iloc[idx1][source_col])
                amount2 = float(df2.iloc[idx2][target_col])
                if abs(amount1 - amount2) <= tolerance:
                    confidence = 1.0 - (abs(amount1 - amount2) / max(abs(amount1), abs(amount2)))
                    self._add_match(results, df1, df2, idx1, idx2, confidence, f"Amount match within {tolerance} tolerance", rule.get('name', ''))
                    df1_matched[idx1] = True
                    df2_matched[idx2] = True
                    break

    def _execute_date_match(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                          df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute date matching rule."""
        source_col = rule.get('source_columns', [''])[0]
        target_col = rule.get('target_columns', [''])[0]
        max_days = rule.get('options', {}).get('max_days', 7)
        for idx1 in range(len(df1)):
            if df1_matched[idx1]:
                continue
            for idx2 in range(len(df2)):
                if df2_matched[idx2]:
                    continue
                try:
                    date1 = datetime.strptime(str(df1.iloc[idx1][source_col]), '%Y-%m-%d')
                    date2 = datetime.strptime(str(df2.iloc[idx2][target_col]), '%Y-%m-%d')
                    days_diff = abs((date2 - date1).days)
                    if days_diff <= max_days:
                        confidence = 1.0 - (days_diff / max_days)
                        self._add_match(results, df1, df2, idx1, idx2, confidence, f"Date match within {max_days} days", rule.get('name', ''))
                        df1_matched[idx1] = True
                        df2_matched[idx2] = True
                        break
                except (ValueError, TypeError):
                    continue

    def _execute_composite_match(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame,
                               df1_matched: pd.Series, df2_matched: pd.Series, results: Dict[str, Any]) -> None:
        """Execute composite matching rule."""
        source_cols = rule.get('source_columns', [])
        target_cols = rule.get('target_columns', [])
        weights = rule.get('options', {}).get('weights', {})
        threshold = rule.get('options', {}).get('threshold', 0.8)
        for idx1 in range(len(df1)):
            if df1_matched[idx1]:
                continue
            for idx2 in range(len(df2)):
                if df2_matched[idx2]:
                    continue
                confidence = 0.0
                total_weight = 0.0
                for src_col, tgt_col in zip(source_cols, target_cols):
                    weight = weights.get(src_col, 1.0)
                    total_weight += weight
                    val1 = str(df1.iloc[idx1][src_col])
                    val2 = str(df2.iloc[idx2][tgt_col])
                    if val1 == val2:
                        confidence += weight
                    else:
                        similarity = SequenceMatcher(None, val1, val2).ratio()
                        confidence += weight * similarity
                if total_weight > 0:
                    confidence /= total_weight
                if confidence >= threshold:
                    self._add_match(results, df1, df2, idx1, idx2, confidence, f"Composite match using {', '.join(source_cols)}", rule.get('name', ''))
                    df1_matched[idx1] = True
                    df2_matched[idx2] = True
                    break

    def _add_unmatched_items(self, df1: pd.DataFrame, df2: pd.DataFrame,
                            df1_matched: pd.Series, df2_matched: pd.Series,
                            results: Dict[str, Any]) -> None:
        """Add unmatched items to results using integer position."""
        # Add unmatched items from source
        for idx in range(len(df1)):
            if not df1_matched[idx]:
                item = df1.iloc[idx].to_dict()
                item['index'] = idx
                item['type'] = self._infer_transaction_type(item)
                results['unmatched_source'].append(item)
        
        # Add unmatched items from target
        for idx in range(len(df2)):
            if not df2_matched[idx]:
                item = df2.iloc[idx].to_dict()
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

    def _execute_flexible_rule(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Apply a flexible matching rule for the second pass. Only accept matches with strong logical reasoning."""
        source_cols = rule.get('source_columns', [])
        target_cols = rule.get('target_columns', [])
        threshold = rule.get('options', {}).get('threshold', 0.7)  # Lower threshold for flexible pass
        for idx1 in range(len(df1)):
            for idx2 in range(len(df2)):
                # Fuzzy/partial match logic: require at least 2 strong similarities and a reasonable explanation
                match_score = 0
                reasons = []
                for src_col, tgt_col in zip(source_cols, target_cols):
                    val1 = str(df1.iloc[idx1][src_col])
                    val2 = str(df2.iloc[idx2][tgt_col])
                    if src_col.lower() in ['amount', 'debit', 'credit']:
                        try:
                            num1 = float(val1) if val1 not in ['nan', 'NaN', 'None', ''] else 0.0
                            num2 = float(val2) if val2 not in ['nan', 'NaN', 'None', ''] else 0.0
                            if abs(num1 - num2) < 1.0:  # Allow up to $1 difference
                                match_score += 1
                                reasons.append(f"Amount nearly matches: {num1} vs {num2}")
                        except Exception:
                            continue
                    elif src_col.lower() in ['reference', 'description']:
                        # Fuzzy match: ignore spaces/dashes/case
                        norm1 = val1.replace(' ', '').replace('-', '').lower()
                        norm2 = val2.replace(' ', '').replace('-', '').lower()
                        if norm1 == norm2:
                            match_score += 1
                            reasons.append(f"{src_col} matches after normalization: {val1} vs {val2}")
                        elif norm1 in norm2 or norm2 in norm1:
                            match_score += 0.5
                            reasons.append(f"{src_col} is a partial match: {val1} vs {val2}")
                    elif src_col.lower() == 'date':
                        try:
                            d1 = pd.to_datetime(val1)
                            d2 = pd.to_datetime(val2)
                            if abs((d1 - d2).days) <= 3:
                                match_score += 1
                                reasons.append(f"Date within 3 days: {val1} vs {val2}")
                        except Exception:
                            continue
                if match_score >= 2:  # Require at least 2 strong similarities
                    results['flexible_matched'].append({
                        'source_transaction': df1.iloc[idx1].to_dict(),
                        'target_transaction': df2.iloc[idx2].to_dict(),
                        'match_score': match_score,
                        'reasons': reasons,
                        'rule_name': rule.get('name', '')
                    }) 