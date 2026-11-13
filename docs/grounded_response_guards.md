# Response grounding regression fixes

The first observed Task 6 model responses showed two claim-level problems:
a segment-specific waiver was suggested despite conflicting customer
signals, and a Gold/Platinum comparison cited passages that did not
contain both products' terms. A citation marker being syntactically valid
was not enough.

## What changed

- When the retrieved context contains the retention policy and customer
  tenure and credit-score segments conflict, requests about retention,
  offers, and fee waivers receive a deterministic, review-needed response.
  The answer says that the available indicators conflict and requests
  authorized verification; it does **not** pick a tier or a percentage
  discount. Where applicable, it cites only retrieved fee-waiver eligibility
  or retention-offer passages. It never sends the numeric score to a model.
- For explicit Gold-versus-Platinum card comparisons, the assistant extracts
  lines headed `Annual fee`, `Limit`, `Reward rate`, `Rewards`, or
  `Eligibility` from the two respective product-summary sections and cites
  each section. Loyalty-tier passages are deliberately not used as
  evidence for credit-card benefits.
- If either card's product section is absent from retrieved passages, the
  answer reports insufficient evidence instead of inventing a comparison.
- Other questions keep the LangChain Runnable and injected model flow.

## What this does not prove

These are narrowly scoped, deterministic safeguards informed by the three
observed model outputs, **not** a general semantic citation verifier. All
other model-generated claims can still be unsupported despite well-formed
citation markers. The policies and numeric thresholds must be reviewed if
the source documents change, and a fresh real-model response evaluation
is required before claiming that all Task 6 quality issues have been fixed.

The regression tests use fictional policy snippets. No supplied customer
records, policy documents, API keys, index files, or actual generated
private answers are committed.
