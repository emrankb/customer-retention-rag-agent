# Retention segment handling

The supplied `retention_policy.txt` defines high-, moderate-, and low-value
segments using **or** conditions for tenure and credit score (lines 9–12).
Those conditions can match more than one segment for the same customer.

When a retrieval result includes `retention_policy.txt`, the assistant
calculates which segment each of the two fields would match. It passes those
labels and any overlap warning to the model, **not the exact credit score**.
The model must not present an ambiguous classification as a confirmed tier or
promise an offer. Retention and offer questions are also marked for
authorized review.

The thresholds reflect the currently supplied retention policy:

- Tenure: above three years, one through three years, or below one year
- Credit score: above 750, 650 through 750, or below 650

These are signals for discussing the policy, not account-eligibility checks.
No payment history, CRM approval, account offer history, or verified hardship
is available from the supplied churn CSVs. The fee waiver policy has separate
conditions, so no annual waiver can be approved from these signals.

The thresholds are implemented locally and must be rechecked if the
retention policy changes. Saved-index validation alone cannot detect a
changed interpretation of the policy. Unit tests cover boundary values,
missing scores, conflicting matches, and the review flag.

A real model evaluation still needs the returned answers checked against the
source passages. Prompt instructions and unit tests are not proof that a
model will always follow these limits.
