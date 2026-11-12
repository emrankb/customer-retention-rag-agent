# Reviewing generated responses

Task 6 asks for grounding accuracy, personalization, tone, and coherence.
This module records a review of **actual observed model responses**, rather
than inferring quality from successful unit tests.

For each response, use a score of 0, 1, or 2 and write a short supporting
note for each dimension:

- **Grounding:** 0 = unsupported or contradictory; 1 = partly supported or
  missing important evidence; 2 = supported by the relevant policy passages
  without invented approvals or conditions
- **Personalization:** 0 = incorrect customer facts; 1 = limited use of
  relevant available context; 2 = appropriately uses correct customer context
  and acknowledges missing information where necessary
- **Tone:** 0 = dismissive or inappropriate; 1 = generally polite but misses
  the customer's concern; 2 = clear, respectful, and empathetic
- **Coherence:** 0 = confusing or contradictory; 1 = understandable with
  significant gaps; 2 = direct, logically ordered, and responsive

Use the three questions listed in the assignment first:

1. How can I waive my annual fee?
2. What should we offer to retain customer 15634602?
3. Explain the difference between Gold and Platinum cards.

Additional cases should include unavailable customer IDs and questions that
require information absent from the provided CSVs. Do not infer payment history,
account eligibility, or approvals from the historical `Exited` label.

## File format

Save completed reviews in a local JSONL file outside version control, such
as `reports/reviews.jsonl`. One record per line, with fields `id`,
`question`, `response`, `sources`, `model`, `ratings`, and `notes`.

`sources` is a list of objects containing `source`, `line_start`, and
`line_end`. `ratings` and `notes` each have keys `grounding`,
`personalization`, `tone`, and `coherence`. Every score needs a supporting
note. Store actual model responses and exact cited lines; do not fill the
scores before seeing the response. A review should check the original
reference passages directly.

Run:

```bash
python -m scripts.summarize_response_reviews --reviews reports/reviews.jsonl
```

The report gives means and score counts for all four dimensions. Missing,
invalid, or unreviewed scores cause an error instead of being counted as
success. The output contains only case IDs and summary statistics; answer
text stays in the private input file.

GitHub Actions tests validation and score calculations with fabricated
examples. Those checks are not real model-evaluation scores. No API requests
are made by the summarizer. A live evaluation with an approved model is
still necessary for a complete Task 6 result.
