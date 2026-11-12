# Customer retention assistant

This project uses customer churn records and bank reference documents to answer retention questions. The planned application uses LangChain for retrieval and response generation, FAISS for document search, and Gradio for the chat interface.

## Current work

The data layer is implemented first. `CustomerDB` looks up records by `CustomerId`; it does not produce a new churn probability. The `Exited` field is a historical label, and the optional sentiment column is loaded from `Churn_With_Sentiment.csv`.

The source checker verifies the two customer tables and the six reference documents before they are used in the application. In particular, it checks that the enriched table has the same customer IDs and fields as the original table. Numeric formatting differences such as `0` and `0.0` are treated as equivalent.

## Local setup

Use Python 3.11 or newer. Codespaces creates a Python 3.12 virtual environment and runs the foundation tests when the container is built. Put the supplied files in the following folders:

```text
data/raw/
  Churn_Modeling.csv
  Churn_With_Sentiment.csv

docs/reference/
  complaints_handling.txt
  faq_accounts.md
  fee_waiver_policy.txt
  product_summary.txt
  retention_policy.txt
  rewards_and_benefits.txt
```

Run the checks from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_sources.py
```

The unit tests use small synthetic records and run without the supplied CSVs. The second command checks the actual input files and reports document hashes and customer counts without printing customer profiles. The input files and any generated indexes are excluded from Git.

## Data interpretation

`CustomerDB.get_context(customer_id)` returns the profile and optional sentiment label, or `None` for an unknown ID. Financial values are kept as strings with their source precision and no assumed currency. The churn label must not be described as a predicted probability or evidence of a customer's current status. The supplied files do not contain all the payment-history and approval details required to authorize a fee waiver.

Document ingestion, embeddings, the RAG workflow, Gradio, and the final answer evaluation will be added in later milestones. These are not part of the current data-checking code.
