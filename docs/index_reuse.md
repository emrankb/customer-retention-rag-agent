# Reusing the policy index in the chat app

The local chat app opens and verifies the FAISS index when it starts. It
then keeps the loaded search object in memory and uses it for later
questions. The CLI search command still opens and verifies the index for
each one-off invocation.

This avoids reloading the embedding model and reading index files for each
message. It does not change the order or scores returned by the search
algorithm. There are no performance measurements in this change.

The source files and index must not be changed while the app is running.
If a policy changes, stop the app, rebuild the index, and restart. Startup
validation checks the file hashes; the cached search object is not a live
file watcher.

Integration tests submit multiple questions with local deterministic
embeddings and verify that the loaded index is reused. They use synthetic
records, not the supplied customer dataset.
