# Retrieval evaluation

The offline answer checks use fixed replies. This evaluation instead measures
whether FAISS retrieves the policy documents identified by a reviewer.

Prepare a JSONL file with one question per line. Each record needs `id`,
`question`, and `relevant_sources`. Use filenames from the six documents in
`docs/reference/`. The reviewer should read the documents and record every
source that supports the question's answer; do not infer labels from what the
retriever returns.

Example format (illustrative, **not** a real gold-standard judgment):

```json
{"id":"example-1","question":"Example question","relevant_sources":["faq_accounts.md"]}
```

Run locally with the original policy documents and an existing index:

```bash
python -m scripts.evaluate_retrieval --cases path/to/reviewed_questions.jsonl
```

The command writes `reports/retrieval_evaluation.json`. Reports are excluded
from Git. You can specify `--k`, `--model-name`, `--docs-dir`, and
`--index-dir`.

The report includes source hit rate (fraction of questions with at least one
relevant source among the first k chunks), mean source recall (fraction of
labeled sources found per question), and mean reciprocal rank (based on the
first relevant distinct source). Repeated chunks from one file are counted as
one source. **These are document-level metrics, not passage-level accuracy or
answer correctness.** In particular, a retrieved document can be relevant
while its displayed passage does not answer the question.

GitHub Actions tests the scoring formulas using synthetic retrieval results.
It does not contain the supplied reference documents or reviewed gold labels,
so it cannot report real-data retrieval scores. The evaluation makes no
model API calls.
