# Customer retention assistant

A local application for answering customer questions using records from the supplied churn CSVs and six bank policy documents. It uses LangChain, sentence-transformer embeddings, FAISS, and Gradio. The text model is configured separately and is not required for the offline tests.

## Project files

Put the supplied files in these folders. They are not tracked in Git.

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

The sentiment table is optional. If supplied, its customer records must match the original table except for the sentiment column. `Exited` is a historical dataset label, not a prediction or a current customer status.

## Setup

Use Python 3.12 from the repository root.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-retrieval.txt -r requirements-app.txt
.venv/bin/python -m scripts.check_sources
```

Build the policy index once after adding the reference documents:

```bash
.venv/bin/python scripts/build_policy_index.py
```

The embedding model may be downloaded the first time it is used. Index files are generated under `index/policies/` and are not committed. If a document changes, remove the old index directory, rebuild it, and restart the application.

To run the chat interface, set `OPENAI_API_KEY` through your environment's secret settings, then run:

```bash
.venv/bin/python -m scripts.run_app
```

The Gradio server listens on `127.0.0.1:7860` by default. Set `RETENTION_PORT` to change the port or `RETENTION_MODEL` to select the configured chat model. It does not create a public share link.

**Data and cost:** Sending a question to the chat model transmits a limited customer profile and retrieved document excerpts to the configured provider and may incur charges. Use only data you are authorized to send. No model requests are made by the automated tests.

## What the code does

- `CustomerDB` reads and validates records and looks up customers by ID.
- The source checker validates both CSVs and the six policy files.
- The retrieval module splits policy documents into passages and stores embeddings in FAISS, retaining file and line references.
- Startup validates the source files and index, then reuses the loaded index for questions.
- A LangChain Runnable chain looks up the customer, retrieves policy passages, and calls the configured model. It checks citation markers and returns a fallback when the response has missing or invalid source references.
- Gradio maintains chat history for the active customer and displays the response and its sources. Changing customer ID clears the chat. Waivers, refunds, eligibility decisions, and retention offers are not approved by the app.

A correct citation marker does **not** prove that the answer is supported by its source.

## Tests and evaluation

Run the offline and installed-package tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.evaluate_answers
```

GitHub Actions runs the tests on pull requests. The full CI job installs both dependency sets and tests customer lookup, the FAISS index, answer generation with a fake model, and the interface using temporary data.

The offline answer evaluation uses six synthetic cases. It checks application behavior, not language-model answer accuracy.

For retrieval evaluation against reviewed policy questions, run:

```bash
.venv/bin/python -m scripts.evaluate_retrieval --cases path/to/reviewed_questions.jsonl
```

This reports source hit rate, mean source recall, and mean reciprocal rank. A reviewer must supply expected documents from the actual policy materials. Do not treat test fixtures as real-data evaluation results.

## Limitations

The project does not train a new churn prediction model. The historical `Exited` label must not be treated as a predicted churn probability. Policy search and citation-marker validation do not guarantee factual answers or authorize account decisions. End-to-end testing with the supplied private documents and an approved live model, along with a check against the assignment rubric, remains to be completed.

Additional details are in `docs/retrieval.md`, `docs/evaluation.md`, `docs/retrieval_evaluation.md`, `docs/startup_checks.md`, and `docs/index_reuse.md`.

## Assignment coverage

The Runnable chain and customer-specific Gradio chat history are implemented.
The remaining mandatory evaluation work is assessing grounding, personalization,
tone, and coherence using reviewed answers from the supplied policy documents.
See `docs/assignment_checklist.md` for the six-task checklist and limitations.

## Reviewing model responses

Task 6 also requires reviewing grounding, personalization, tone, and
coherence. Save actual model responses, their source lines, and a reviewer
score and explanation for each dimension in an ignored local JSONL file.
Then summarize the completed reviews with:

```bash
.venv/bin/python -m scripts.summarize_response_reviews --reviews reports/reviews.jsonl
```

The scorer rejects missing ratings. It does not make model calls or grade
answers automatically. See `docs/response_review.md` for the 0–2 rubric.
The three assignment sample questions have not yet been scored against live
model output, so Task 6 is not marked complete.

## Capturing assignment answers

The three questions from the assignment are in
`evaluation/sample_questions.jsonl`. To capture actual responses and the
passages retrieved for them, use the local evaluation command after loading
the approved customer and policy files:

```bash
.venv/bin/python -m scripts.capture_response_reviews --allow-model-requests
```

This command requires an API key and deliberately asks for an explicit flag:
it sends limited customer context and policy excerpts to the configured
model and may incur API charges. It writes ignored files under `reports/`.
Ratings remain empty until the responses are reviewed. The review-summary
command refuses to report incomplete ratings. See
`docs/capture_evaluation.md`.

CI verifies this capture logic using fixed model responses, not a live LLM.
No grounding, personalization, tone, or coherence score is claimed until
real responses are recorded and reviewed.

## Retention segment ambiguity

The supplied retention policy can match the same customer to different
segments through its tenure and credit-score rules. When that policy is
retrieved, the assistant passes the matching segment signals and an overlap
warning to the model, without sending the raw credit score. An overlapping
match is not resolved automatically and does not authorize an offer.

Retention-offer questions are marked for review. The original CSVs do not
contain the payment history, waiver approvals, and CRM records needed to
confirm eligibility. See `docs/retention_segments.md`.
