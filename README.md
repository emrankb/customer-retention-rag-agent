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

## Policy document search

The retrieval layer reads the six files in `docs/reference/` with LangChain
`TextLoader`. `RecursiveCharacterTextSplitter` creates overlapping passages and
keeps the original file and line numbers. The embedding model is
`sentence-transformers/all-MiniLM-L6-v2`, running locally on CPU; FAISS stores
its vectors. Customer profiles are looked up directly through `CustomerDB`, not
added to the policy index.

Install the retrieval packages and build the index after placing the documents
in `docs/reference/`:

```bash
.venv/bin/python -m pip install -r requirements-retrieval.txt
.venv/bin/python scripts/build_policy_index.py
.venv/bin/python scripts/search_policies.py --query "How can I waive my annual fee?"
```

The first build downloads the embedding model if it is not already cached.
The generated index lives under ignored `index/policies/`. Rebuilding requires
removing the old index directory first; the build command does not overwrite it.
The loader checks the source files and saved vector metadata before searching.
See `docs/retrieval.md` for indexing details and how the tests are organized.

Tests run without uploading documents or customer data. The offline test suite
uses fixed vectors, while a separate integration suite exercises LangChain and
FAISS when the optional packages are installed:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Chat interface

The application combines customer lookup and policy search with a Gradio
interface. The model adapter uses LangChain and a configured OpenAI model.
An API key is required to use it, and submitting a question sends a limited
customer profile and relevant document excerpts to that provider. The service
can incur charges. Do not use private customer data without authorization.

Install the application packages alongside the retrieval packages:

```bash
.venv/bin/python -m pip install -r requirements-app.txt
```

Place the original CSV files and reference documents in the folders above,
then build the policy index if it does not already exist. Set `OPENAI_API_KEY`
using the environment's secret management mechanism. Optionally set
`RETENTION_MODEL` to choose a model and `RETENTION_PORT` for the local port.
Start the interface from the repository root:

```bash
.venv/bin/python -m scripts.run_app
```

The application binds to `127.0.0.1` and does not create a public share URL.
It is intended for local development, not public deployment. Account approvals,
fee waivers, and retention offers still require an authorized review.

GitHub Actions runs the offline tests without a model key or real customer
records, and separately checks the Gradio installation and FAISS integration.
These checks do not measure answer quality against the supplied dataset.
Response evaluation and a review of the final assignment requirements remain
open tasks.

## Offline answer checks

Run `python -m scripts.evaluate_answers` to exercise six synthetic
question-answer cases without using a model API key. The report is written to
`reports/answer_contracts.json` and is excluded from Git.

These checks cover missing customers, missing policy results, source markers,
and human-review flags. They do not measure the accuracy of generated answers
against the original policy documents. See `docs/evaluation.md` for details
and the remaining manual evaluation work.

## Citation checks

The assistant rejects model responses containing source markers that do not
match the retrieved passages, even if another citation is valid. Long or
invalid model responses also produce a fallback message. These checks do not
prove that the answer's claims are supported by its cited text. See
`docs/citation_validation.md`.

## Application integration test

GitHub Actions also runs the customer lookup, local FAISS index, answer
generator, and interface callback together. The test creates temporary
customer and policy files and uses fixed keyword embeddings and a fake
model reply. It checks a cited response, an unknown customer, a damaged
index, and Gradio app construction.

This checks whether the components work together without a hosted model.
It does not measure the quality of responses from an actual LLM.

## Reviewing document retrieval

To measure retrieval against the original policy files, create a reviewed
JSONL question set with expected source filenames and run:

```bash
python -m scripts.evaluate_retrieval --cases path/to/reviewed_questions.jsonl
```

The command reports source hit rate, source recall, and reciprocal rank at k
using the saved FAISS index. It does not call the chat model. The reviewed
question set and policy documents are not included in the repository; CI
checks the scoring formulas with synthetic examples instead. See
`docs/retrieval_evaluation.md` for the required format and metric limits.
