# Answer evaluation

Run the offline answer checks with:

```bash
python -m scripts.evaluate_answers
```

This reads `tests/fixtures/answer_cases.jsonl` and writes a JSON report to
`reports/answer_contracts.json`. The report folder is ignored by Git.
Each case uses a made-up customer, made-up policy passage, and a fixed model
reply. No language model or real customer CSV is used.

The runner checks the branch the assistant takes (answer, missing customer,
missing evidence, or rejected citation), whether the answer needs human review,
whether retrieval and generation were called when expected, and whether the
returned citations point to the supplied passage metadata. It reports the
number of cases passed, failed, and a pass fraction. Failing expectations
produce a nonzero exit code.

**These are contract tests, not a policy accuracy benchmark.** The synthetic
cases cannot establish that a generated sentence is supported by a document,
that retrieved passages are relevant, or that a language model consistently
follows instructions. A model can cite a real passage while making an
unsupported claim. The current assistant checks citation markers, not textual
entailment.

To evaluate real answers, prepare an independently reviewed question set from
the supplied policy documents, note the expected sources and when the assistant
should decline to answer, and inspect actual model responses. Record sample
selection, model configuration, passage retrieval, and each observed failure.
Do not report synthetic contract pass rates as model accuracy. Calling a hosted
model requires separate authorization because it can incur charges and transmit
customer context.
