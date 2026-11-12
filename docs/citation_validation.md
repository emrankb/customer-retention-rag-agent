# Citation validation

The assistant numbers retrieved passages as `[S1]`, `[S2]`, and so on.
A generated response is returned only if it contains at least one source
marker and every marker refers to a passage supplied in that request.

Markers such as `[S0]`, `[S01]`, `[S99]`, or `[Sx]` cause the
assistant to return a fallback, even when the response also contains a valid
marker. Empty, non-text, overly long, or NUL-containing model responses are
also rejected. The output limit is 8,000 characters.

The checks prevent invalid source references from being displayed. They do
not establish whether the model's statements are supported by the cited
passages. Checking factual support still requires separate evaluation using
the reference documents and reviewed expected answers.
