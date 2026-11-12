# Startup checks

Before Gradio starts, the application checks the original customer table and
the optional sentiment table, verifies that their shared records match, checks
all six reference documents, and opens the saved FAISS index.

The index loader checks the saved hashes and model name against the current
documents. If a source changes after indexing, startup fails rather than
serving an out-of-date policy index. To fix that condition, rebuild the index
from the supplied documents. Do not edit the original files to make the check
pass.

The check happens before creating the hosted chat-model adapter, so source
validation does not require an API request. Loading local embeddings can
download the embedding model if it is not already cached.

The check_startup() return value includes only customer counts, whether the
sentiment table is present, document count, vector count, and the embedding
model name. It does not return customer records or policy text.

Tests create temporary CSVs and reference documents and build a FAISS index
with deterministic local vectors. GitHub Actions runs these tests with the
retrieval packages installed. It cannot verify the privately supplied files
because those files are deliberately not committed.

## Loaded index lifecycle

The application calls prepare_startup() once and passes its verified search
callable to the chat interface. This avoids opening the same FAISS index again
after preflight. check_startup() remains available for commands that only need
the validation summary.

The index is a snapshot. Stop the app, rebuild the index, and restart after
changing any policy source. The app does not monitor files for changes.
