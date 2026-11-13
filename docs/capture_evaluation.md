# Capturing answers for Task 6

The assignment asks us to assess grounding accuracy, personalization,
tone, and coherence. The previous unit tests check code behavior; they do
not show how the configured language model performs on the policy documents.

The sample questions are in `evaluation/sample_questions.jsonl`. They are
the three examples from the assignment and use the customer ID given there.
No customer profiles or source documents are committed.

## Running a real model evaluation

Place the supplied CSVs and six policy files in their local folders, install
both requirement files, and build the FAISS index. Review the data-sharing
rules and model charges first. The following command **does make external API
requests** and may transmit limited customer context and policy passages:

```bash
python -m scripts.capture_response_reviews --allow-model-requests
```

The command refuses to call the model without the explicit flag and an API
key. It saves `reports/response_reviews_pending.jsonl` and
`reports/retrieved_evidence.jsonl`. Both are ignored by Git. Existing output
files are not overwritten. The evidence file keeps the retrieved text for
checking citations against the original reference documents.

Each captured answer is paired with its customer question, model name, and
cited source lines. All four review ratings are initially `null`, with empty
notes. Reviewers must read the responses and supporting documents and enter
0, 1, or 2 with a specific reason for each dimension. The summarizer refuses
unreviewed cases:

```bash
python -m scripts.summarize_response_reviews --reviews reports/response_reviews_pending.jsonl
```

That command succeeds only after the ratings and notes have been completed.
The scores and reasons must reflect observed responses, not the expected
behavior of test doubles.

GitHub Actions tests parsing, capture logic, and the empty-score guard using
a fake model. It does not call ChatOpenAI or claim a real-model quality score.

The evaluation is not complete until reviewed model responses exist and the
four required criteria have been scored.

## Model calls versus application guards

The capture records whether the configured model was actually called.
The private evidence JSONL contains `answer_origin`, `model_calls`, and
`configured_model` for each case. A response produced by a deterministic
safety guard or fallback is labeled `no_model_call` in the review file,
even if a language model was configured in the application.

After the grounded-response-guards milestone, the three assignment questions
may all be handled by application policy guards with zero model calls. Those
answers can be reviewed for grounding, personalization, tone, and coherence
as **application responses**, but must never be presented as new LLM quality
scores. To measure LLM quality independently, a distinct reviewed test set
that reaches the model is necessary. This metadata does not classify which
specific guard or fallback ran; only whether generation was invoked.
