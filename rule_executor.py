from typing import Dict, List, Any
import pandas as pd
from fuzzywuzzy import fuzz
from datetime import datetime
import re
from difflib import SequenceMatcher
import itertools

class RuleExecutor:
    def __init__(self):
        """Initialize the rule executor."""
        pass

    def execute(self, df1: pd.DataFrame, df2: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute reconciliation rules on two dataframes: strict one-to-one matching first, then a flexible second pass."""
        print("\n[DEBUG] Starting rule execution")
        print(f"[DEBUG] Total source rows: {len(df1)}")
        print(f"[DEBUG] Total target rows: {len(df2)}")
        
        results = {
            'matched': [],
            'flexible_matched': [],
            'one_to_many_matched': [],  # New category for one-to-many matches
            'unmatched_source': [],
            'unmatched_target': [],
            'pending_review': []
        }

        # Track matched rows by integer position
        df1_matched = pd.Series(False, index=range(len(df1)))
        df2_matched = pd.Series(False, index=range(len(df2)))

        # Add a flexible rule if none exists
        has_flexible_rule = any(rule.get('options', {}).get('flexible', False) for rule in rules)
        print(f"[DEBUG] Has flexible rule: {has_flexible_rule}")
        if not has_flexible_rule:
            print("[DEBUG] Adding default flexible rule")
            flexible_rule = {
                'name': 'Flexible Match',
                'type': 'composite',
                'source_columns': ['Date', 'Description', 'Reference', 'Debit', 'Credit'],
                'target_columns': ['Date', 'Description', 'Reference', 'Debit', 'Credit'],
                'options': {
                    'flexible': True,
                    'threshold': 0.7,
                    'allow_minor_formatting_differences': True,
                    'ignore_case': True,
                    'strip_spaces_and_dashes': True
                }
            }
            rules.append(flexible_rule)

        # --- Stage 1: Strict One-to-One Matching ---
        print("\n[DEBUG] Starting strict matching pass")
        for rule in rules:
            if rule.get('options', {}).get('allow_one_to_many', False):
                continue  # Skip one-to-many rules in this pass
            if rule.get('options', {}).get('flexible', False):
                continue  # Skip flexible rules in strict pass
            self._execute_one_to_one_rule(rule, df1, df2, df1_matched, df2_matched, results)

        # --- Stage 2: Flexible Matching on Unmatched Items ---
        unmatched_df1 = df1[~df1_matched.values].reset_index()
        unmatched_df2 = df2[~df2_matched.values].reset_index()
        print(f"\n[DEBUG] After strict matching:")
        print(f"[DEBUG] Unmatched source rows: {len(unmatched_df1)}")
        print(f"[DEBUG] Unmatched target rows: {len(unmatched_df2)}")
        
        if not unmatched_df1.empty and not unmatched_df2.empty:
            print("\n[DEBUG] Starting flexible matching pass")
            for rule in rules:
                if not rule.get('options', {}).get('flexible', False):
                    continue  # Only process flexible rules in this pass
                print(f"[DEBUG] Executing flexible rule: {rule.get('name', 'unnamed')}")
                self._execute_flexible_rule(rule, unmatched_df1, unmatched_df2, results, df1_matched, df2_matched)
        else:
            print("[DEBUG] Skipping flexible matching - no unmatched items or empty dataframes")

        # Print unmatched after flexible matching
        unmatched_df1 = df1[~df1_matched.values].reset_index()
        unmatched_df2 = df2[~df2_matched.values].reset_index()
        print(f"\n[DEBUG] After flexible matching:")
        print(f"[DEBUG] Unmatched source rows: {len(unmatched_df1)}")
        print(f"[DEBUG] Unmatched target rows: {len(unmatched_df2)}")

        # --- Stage 3: One-to-Many Matching for Fees ---
        print("\n[DEBUG] Starting one-to-many matching pass")
        if not unmatched_df1.empty and not unmatched_df2.empty:
            self._execute_one_to_many_fee_matching(unmatched_df1, unmatched_df2, results, df1_matched, df2_matched)
        else:
            print("[DEBUG] Skipping one-to-many matching - no unmatched items or empty dataframes")

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

        print(f"\n[DEBUG] Final results:")
        print(f"[DEBUG] Matched pairs: {len(results['matched'])}")
        print(f"[DEBUG] Flexible matched pairs: {len(results['flexible_matched'])}")
        print(f"[DEBUG] One-to-many matched groups: {len(results['one_to_many_matched'])}")
        print(f"[DEBUG] Unmatched source items: {len(results['unmatched_source'])}")
        print(f"[DEBUG] Unmatched target items: {len(results['unmatched_target'])}")
        
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

    def _amount_fields_match(self, src_item, tgt_item, src_debit_col='Debit', src_credit_col='Credit', tgt_debit_col='Debit', tgt_credit_col='Credit', tolerance=1.0):
        """Return True if any of the following pairs are nearly equal (within tolerance):
        - src_credit vs tgt_debit
        - src_debit vs tgt_credit
        - src_debit vs tgt_debit
        - src_credit vs tgt_credit
        """
        pairs = []
        # Get values
        src_debit = src_item.get(src_debit_col)
        src_credit = src_item.get(src_credit_col)
        tgt_debit = tgt_item.get(tgt_debit_col)
        tgt_credit = tgt_item.get(tgt_credit_col)
        # Try all combinations
        try:
            if not pd.isna(src_credit) and not pd.isna(tgt_debit):
                if abs(float(src_credit) - float(tgt_debit)) < tolerance:
                    return True, f"Bank Credit matches GL Debit: {src_credit} vs {tgt_debit}"
            if not pd.isna(src_debit) and not pd.isna(tgt_credit):
                if abs(float(src_debit) - float(tgt_credit)) < tolerance:
                    return True, f"Bank Debit matches GL Credit: {src_debit} vs {tgt_credit}"
            if not pd.isna(src_debit) and not pd.isna(tgt_debit):
                if abs(float(src_debit) - float(tgt_debit)) < tolerance:
                    return True, f"Debit matches Debit: {src_debit} vs {tgt_debit}"
            if not pd.isna(src_credit) and not pd.isna(tgt_credit):
                if abs(float(src_credit) - float(tgt_credit)) < tolerance:
                    return True, f"Credit matches Credit: {src_credit} vs {tgt_credit}"
        except Exception as e:
            pass
        return False, None

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
                    if src_col.lower() in ['debit', 'credit'] and tgt_col.lower() in ['debit', 'credit']:
                        match, _ = self._amount_fields_match(df1.iloc[idx1], df2.iloc[idx2], tolerance=0.01)
                        if not match:
                            match = False
                            break
                    else:
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
        for idx1 in range(len(df1)):
            if df1_matched[idx1]:
                continue
            for idx2 in range(len(df2)):
                if df2_matched[idx2]:
                    continue
                match, _ = self._amount_fields_match(df1.iloc[idx1], df2.iloc[idx2], tolerance=rule.get('options', {}).get('tolerance', 0.01))
                if match:
                    confidence = 1.0
                    self._add_match(results, df1, df2, idx1, idx2, confidence, f"Amount match within tolerance", rule.get('name', ''))
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
                    # Special handling for amount fields
                    if src_col.lower() in ['debit', 'credit'] and tgt_col.lower() in ['debit', 'credit']:
                        match, _ = self._amount_fields_match(df1.iloc[idx1], df2.iloc[idx2])
                        if match:
                            confidence += weight
                        else:
                            confidence += 0  # No partial for amount fields
                    else:
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

    def _llm_text_confidence(self, desc1, ref1, desc2, ref2, analyzer=None):
        """Call the LLM to get a confidence score and explanation for description/reference similarity."""
        if analyzer is None:
            # If no analyzer provided, fallback to basic fuzzy logic
            from difflib import SequenceMatcher
            desc_score = SequenceMatcher(None, str(desc1), str(desc2)).ratio()
            ref_score = SequenceMatcher(None, str(ref1), str(ref2)).ratio()
            avg_score = (desc_score + ref_score) / 2
            return avg_score, f"Fuzzy match fallback: desc_score={desc_score:.2f}, ref_score={ref_score:.2f}"
        prompt = f"""
Given these two transaction descriptions and references, how likely are they to refer to the same transaction? Return a confidence score (0 to 1) and a brief explanation.\n\nDescription 1: "{desc1}"
Reference 1: "{ref1}"
Description 2: "{desc2}"
Reference 2: "{ref2}"
"""
        llm_result = analyzer.score_text_similarity(desc1, ref1, desc2, ref2, prompt=prompt)
        # Expecting llm_result = {'confidence': float, 'explanation': str}
        return llm_result.get('confidence', 0.0), llm_result.get('explanation', '')

    def _execute_flexible_rule(self, rule: Dict[str, Any], df1: pd.DataFrame, df2: pd.DataFrame, results: Dict[str, Any], df1_matched=None, df2_matched=None, analyzer=None) -> None:
        print(f"\n[DEBUG] Starting flexible rule execution for rule: {rule.get('name', 'unnamed')}")
        print(f"[DEBUG] Source columns: {rule.get('source_columns', [])}")
        print(f"[DEBUG] Target columns: {rule.get('target_columns', [])}")
        print(f"[DEBUG] Number of unmatched source items: {len(df1)}")
        print(f"[DEBUG] Number of unmatched target items: {len(df2)}")

        source_cols = rule.get('source_columns', [])
        target_cols = rule.get('target_columns', [])
        threshold = rule.get('options', {}).get('threshold', 0.7)
        
        # Track which items are matched during this flexible pass
        flexible_matched_source = set()
        flexible_matched_target = set()
        
        for idx1 in range(len(df1)):
            # Skip if source item was already matched in this pass
            if idx1 in flexible_matched_source:
                continue
                
            source_item = df1.iloc[idx1]
            print(f"\n[DEBUG] Checking source item {idx1}:")
            print(f"[DEBUG] Source: Date={source_item.get('Date')}, Desc={source_item.get('Description')}, Ref={source_item.get('Reference')}, Debit={source_item.get('Debit')}, Credit={source_item.get('Credit')}")
            
            for idx2 in range(len(df2)):
                # Skip if target item was already matched in this pass
                if idx2 in flexible_matched_target:
                    continue
                    
                target_item = df2.iloc[idx2]
                print(f"[DEBUG] Against target item {idx2}:")
                print(f"[DEBUG] Target: Date={target_item.get('Date')}, Desc={target_item.get('Description')}, Ref={target_item.get('Reference')}, Debit={target_item.get('Debit')}, Credit={target_item.get('Credit')}")
                
                match_score = 0
                reasons = []
                
                # Amount fields: use cross-field logic
                amount_match, amount_reason = self._amount_fields_match(source_item, target_item)
                if amount_match:
                    match_score += 1
                    reasons.append(amount_reason)
                    print(f"[DEBUG] Amount match found: {amount_reason}")
                else:
                    print("[DEBUG] No amount match")
                    continue  # Require amount match for candidate
                
                # Date field: require within 3 days
                date_match = False
                for src_col, tgt_col in zip(source_cols, target_cols):
                    if src_col.lower() == 'date' and tgt_col.lower() == 'date':
                        val1 = str(source_item[src_col])
                        val2 = str(target_item[tgt_col])
                        try:
                            d1 = pd.to_datetime(val1)
                            d2 = pd.to_datetime(val2)
                            days_diff = abs((d1 - d2).days)
                            if days_diff <= 3:
                                match_score += 1
                                reasons.append(f"Date within 3 days: {val1} vs {val2}")
                                date_match = True
                                print(f"[DEBUG] Date match found: {days_diff} days difference")
                            else:
                                print(f"[DEBUG] Date mismatch: {days_diff} days difference")
                        except Exception as e:
                            print(f"[DEBUG] Date parsing error: {e}")
                            continue
                
                if not date_match:
                    print("[DEBUG] Skipping - no date match")
                    continue  # Require date match for candidate
                
                # LLM-based confidence for Description/Reference
                desc1 = source_item.get('Description', '')
                ref1 = source_item.get('Reference', '')
                desc2 = target_item.get('Description', '')
                ref2 = target_item.get('Reference', '')
                llm_conf, llm_expl = self._llm_text_confidence(desc1, ref1, desc2, ref2, analyzer=analyzer)
                match_score += llm_conf  # Add LLM confidence (0-1)
                reasons.append(f"LLM text confidence: {llm_conf:.2f} - {llm_expl}")
                print(f"[DEBUG] LLM confidence: {llm_conf:.2f} - {llm_expl}")
                
                # More lenient threshold: require amount+date+some text confidence
                if match_score >= 2.0:  # Lowered from 2.5
                    print(f"[DEBUG] Match found! Score: {match_score:.2f}")
                    results['flexible_matched'].append({
                        'source_transaction': source_item.to_dict(),
                        'target_transaction': target_item.to_dict(),
                        'match_score': match_score,
                        'reasons': reasons,
                        'rule_name': rule.get('name', '')
                    })
                    if df1_matched is not None and df2_matched is not None:
                        orig_idx1 = source_item['index'] if 'index' in source_item else source_item.name
                        orig_idx2 = target_item['index'] if 'index' in target_item else target_item.name
                        df1_matched[orig_idx1] = True
                        df2_matched[orig_idx2] = True
                    # Mark these items as matched in this flexible pass
                    flexible_matched_source.add(idx1)
                    flexible_matched_target.add(idx2)
                    break  # Stop checking this source item once we find a match
                else:
                    print(f"[DEBUG] No match - score {match_score:.2f} below threshold 2.0")

    def _execute_one_to_many_fee_matching(self, df1: pd.DataFrame, df2: pd.DataFrame, results: Dict[str, Any], 
                                        df1_matched: pd.Series, df2_matched: pd.Series) -> None:
        """Match multiple bank fees to a single GL entry using subset sum (combinatorial) approach, handling sign convention."""
        print("[DEBUG] Starting one-to-many fee matching (combinatorial, sign-aware)")
        
        # Get all unmatched bank fees (source items)
        bank_fees = []
        for idx, row in df1.iterrows():
            if self._is_bank_fee(row) and not df1_matched[row['index'] if 'index' in row else idx]:
                bank_fees.append((row['index'] if 'index' in row else idx, row))
        
        # Get all unmatched GL fee/charge entries (target items)
        gl_entries = []
        for idx, row in df2.iterrows():
            if self._is_bank_fee(row) and not df2_matched[row['index'] if 'index' in row else idx]:
                gl_entries.append((row['index'] if 'index' in row else idx, row))
        
        print(f"[DEBUG] Found {len(bank_fees)} unmatched bank fees")
        print(f"[DEBUG] Found {len(gl_entries)} unmatched GL fee entries")
        
        # Try to match groups of fees to GL entries
        for gl_idx, gl_row in gl_entries:
            gl_amount = self._get_amount(gl_row)
            if pd.isna(gl_amount):
                continue
            print(f"\n[DEBUG] Checking GL entry: {gl_row.get('Description')} - Amount: {gl_amount}")
            fee_indices = [idx for idx, _ in bank_fees]
            fee_rows = [row for _, row in bank_fees]
            fee_amounts = [self._get_amount(row) for row in fee_rows]
            n = len(fee_indices)
            found = False
            # Try all combinations of 2 to n fees
            for r in range(2, n+1):
                for combo_indices in itertools.combinations(range(n), r):
                    combo_sum = sum(fee_amounts[i] for i in combo_indices)
                    # Try both sum and -sum for sign convention
                    if abs(combo_sum - gl_amount) < 0.01 or abs(-combo_sum - gl_amount) < 0.01:
                        matching_fees = [(fee_indices[i], fee_rows[i]) for i in combo_indices]
                        print(f"[DEBUG] Found matching fee group! Sum: {combo_sum} (or {-combo_sum})")
                        results['one_to_many_matched'].append({
                            'target_transaction': gl_row.to_dict(),
                            'source_transactions': [f[1].to_dict() for f in matching_fees],
                            'match_type': 'fee_group',
                            'total_amount': gl_amount,
                            'fee_details': [{'amount': self._get_amount(f[1]), 'description': f[1].get('Description')} 
                                          for f in matching_fees]
                        })
                        # Mark all matched items
                        df2_matched[gl_idx] = True
                        for fee_idx, _ in matching_fees:
                            df1_matched[fee_idx] = True
                        found = True
                        break
                if found:
                    break

    def _is_bank_fee(self, row: pd.Series) -> bool:
        """Check if a row represents a bank fee."""
        desc = str(row.get('Description', '')).lower()
        return any(term in desc for term in ['fee', 'charge', 'service'])

    def _get_amount(self, row: pd.Series) -> float:
        """Get the amount from a row, handling both Debit and Credit fields."""
        debit = row.get('Debit')
        credit = row.get('Credit')
        if not pd.isna(debit):
            return float(debit)
        if not pd.isna(credit):
            return float(credit)
        return None 