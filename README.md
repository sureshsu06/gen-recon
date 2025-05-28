# Gen-Recon: LLM-Powered Reconciliation Engine

An intelligent reconciliation engine that uses LLMs to automatically understand and match data between different sources.

## Features

- Automatic reconciliation type inference
- Smart matching rule generation
- Support for exact and fuzzy matching
- Deterministic rule execution
- Test suite for validation

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'  # On Windows: set OPENAI_API_KEY=your-api-key-here
```

## Project Structure

```
gen-recon/
│
├── main.py                # Entry point, orchestrates the flow
├── llm_analyzer.py        # LLM Analyzer class
├── rule_executor.py       # Rule Executor class
├── test_cases/            # Folder for test CSVs
│   ├── bank_stmt.csv
│   ├── general_ledger.csv
│   ├── invoices.csv
│   └── payments.csv
├── tests.py               # Automated tests
├── requirements.txt       # Dependencies
└── README.md
```

## Usage

```python
from main import MVPReconciler

reconciler = MVPReconciler(api_key="your-api-key")
results = reconciler.reconcile("file1.csv", "file2.csv")
```

## Testing

Run the test suite:
```bash
python tests.py
``` 