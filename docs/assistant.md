# Grounded answer orchestration

This milestone combines the validated `CustomerDB` customer lookup with the policy
retriever. `answer_question()` fetches the customer by ID, searches approved
documents using FAISS, builds a bounded prompt, calls an explicitly supplied
text generator, and returns an `Answer` with source filenames and line ranges.

The text generator is **not configured or invoked automatically**. An application
layer must supply a callable, such as an authenticated LangChain model adapter.
This separation keeps unit tests offline and allows the deployment owner to
choose a model provider and data-sharing policy. Do not send customer data to an
external provider without appropriate authorization.

A missing customer, missing evidence, empty output, or absent/invalid source
citation produces a safe fallback rather than an uncited policy claim. The
answer's `needs_review` flag marks fee-waiver and eligibility questions for
human review. It is not an eligibility engine or authorization to act.

Only a minimal customer-data subset is placed into the model prompt. Historical
`Exited` is explicitly labeled as historical, not a churn probability.
Source snippets are marked untrusted to limit prompt-injection risk.
The application does not independently prove that cited sentences truly entail
the generated claims. Human evaluation and adversarial testing are still
needed before deployment.

## Testing

```bash
python -m unittest discover -s tests -v
```

The tests exercise orchestration through injected deterministic search and
generation functions, requiring no API keys, customer CSVs, or model download.

## Next milestones

Connect a LangChain chat model with an explicit deployment configuration,
build the Gradio chat interface, then evaluate factuality, citation correctness,
unsupported-claim rates, privacy, and prompt-injection handling using the
supplied materials.
