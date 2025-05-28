import os
import pandas as pd
from dotenv import load_dotenv
from llm_analyzer import LLMAnalyzer
from rule_executor import RuleExecutor

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
        """Reconcile two CSV files.
        
        Args:
            file1_path (str): Path to first CSV file
            file2_path (str): Path to second CSV file
            
        Returns:
            dict: Reconciliation results including matches and unmatched items
        """
        # Load CSV files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        # Get LLM analysis and rules
        rules = self.analyzer.analyze_files(df1, df2)
        
        # Execute rules
        results = self.executor.execute(df1, df2, rules)
        # Merge metadata from rules into results
        for key in ['reconciliation_type', 'confidence', 'reasoning', 'rules']:
            if key in rules:
                results[key] = rules[key]
        return results

def main():
    """Example usage of the reconciliation engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reconcile two CSV files')
    parser.add_argument('file1', help='Path to first CSV file')
    parser.add_argument('file2', help='Path to second CSV file')
    args = parser.parse_args()
    
    reconciler = MVPReconciler()
    results = reconciler.reconcile(args.file1, args.file2)
    
    # Print results
    print("\nReconciliation Results:")
    if 'reconciliation_type' in results:
        print(f"Reconciliation type: {results['reconciliation_type']}")
        if 'confidence' in results:
            print(f"Confidence: {results['confidence']}")
        if 'reasoning' in results:
            print(f"Reasoning: {results['reasoning']}")
    print(f"Matched items: {len(results['matched'])}")
    if results.get("pending_review"):
        print(f"Pending review: {len(results['pending_review'])}")
        for item in results["pending_review"]:
            print(f"- {item.get('explanation', 'Needs review')}")
    print(f"Unmatched in source: {len(results['unmatched_source'])}")
    print(f"Unmatched in target: {len(results['unmatched_target'])}")

if __name__ == '__main__':
    main() 