import os
import pytest
from main import MVPReconciler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture
def reconciler():
    """Create a reconciler instance for testing."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        pytest.skip("OpenAI API key not found in environment")
    return MVPReconciler(api_key)

def test_bank_to_gl_reconciliation(reconciler):
    """Test bank statement to general ledger reconciliation."""
    results = reconciler.reconcile(
        'test_cases/bank_stmt.csv',
        'test_cases/general_ledger.csv'
    )
    
    # Check that we got results
    assert results is not None
    assert 'matched' in results
    assert 'unmatched_source' in results
    assert 'unmatched_target' in results
    
    # Check that all items were matched
    assert len(results['matched']) == 5
    assert len(results['unmatched_source']) == 0
    assert len(results['unmatched_target']) == 0
    
    # Verify specific matches
    matches = results['matched']
    
    # Check first match (PAYMENT RECEIVED)
    first_match = next(m for m in matches if m['source_data']['description'] == 'PAYMENT RECEIVED')
    assert first_match['source_data']['amount'] == 1000.00
    assert first_match['target_data']['credit'] == 1000.00
    
    # Check second match (WIRE TRANSFER)
    second_match = next(m for m in matches if 'WIRE' in m['source_data']['description'])
    assert second_match['source_data']['amount'] == 2500.00
    assert second_match['target_data']['credit'] == 2500.00

def test_reconciliation_type_inference(reconciler):
    """Test that the LLM correctly infers the reconciliation type."""
    results = reconciler.reconcile(
        'test_cases/bank_stmt.csv',
        'test_cases/general_ledger.csv'
    )
    
    # The LLM should have identified this as a bank_to_gl reconciliation
    assert results['reconciliation_type'] == 'bank_to_gl'
    assert results['confidence'] >= 0.8  # High confidence in type inference

def test_fuzzy_matching(reconciler):
    """Test fuzzy matching on descriptions."""
    results = reconciler.reconcile(
        'test_cases/bank_stmt.csv',
        'test_cases/general_ledger.csv'
    )
    
    # Check that fuzzy matching was used for descriptions
    matches = results['matched']
    for match in matches:
        source_desc = match['source_data']['description']
        target_ref = match['target_data']['reference']
        
        # The descriptions should be similar but not necessarily identical
        assert any(word in target_ref for word in source_desc.split()) or \
               any(word in source_desc for word in target_ref.split())

def test_amount_matching(reconciler):
    """Test amount matching with tolerance."""
    results = reconciler.reconcile(
        'test_cases/bank_stmt.csv',
        'test_cases/general_ledger.csv'
    )
    
    # Check that amounts match exactly
    matches = results['matched']
    for match in matches:
        source_amount = match['source_data']['amount']
        target_debit = match['target_data']['debit']
        target_credit = match['target_data']['credit']
        
        # Amount should match either debit or credit
        assert abs(source_amount) == abs(target_debit) or \
               abs(source_amount) == abs(target_credit)

def test_vendor_statement_to_ap_records(reconciler):
    """Test vendor statement to AP records reconciliation with partial, fuzzy, and unmatched records."""
    results = reconciler.reconcile(
        'test_cases/vendor_statement.csv',
        'test_cases/ap_records.csv'
    )
    # Check that we got results
    assert results is not None
    assert 'matched' in results
    assert 'unmatched_source' in results
    assert 'unmatched_target' in results
    assert 'pending_review' in results

    # Check reconciliation type
    assert results.get('reconciliation_type') in ['vendor_statement', 'custom']

    # There should be 4 strong matches (INV-1001, INV-1002, INV-1003, INV-1004)
    assert len(results['matched']) == 4

    # There should be 1 pending review (INV-9999)
    assert len(results['pending_review']) == 1
    pending = results['pending_review'][0]
    assert 'INV-9999' in str(pending['source_data'].get('invoice_ref', '')) or 'INV-9999' in str(pending['source_data'])

    # There should be 1 unmatched in target (INV-8888)
    assert len(results['unmatched_target']) == 1

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 