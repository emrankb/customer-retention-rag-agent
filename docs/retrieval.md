# Policy document retrieval

The searchable material comes from the six supplied files in `docs/reference/`. Customer profiles are not embedded; `CustomerDB` continues to look up customers by ID.

`load_documents()` uses LangChain `TextLoader` with UTF-8 decoding. A `RecursiveCharacterTextSplitter` makes overlapping chunks, and each chunk keeps its source filename and original line range. The default settings are 480 characters with 80 characters of overlap. The source text is left unchanged.

The embedding model is `sentence-transformers/all-MiniLM-L6-v2` through `HuggingFaceEmbeddings`. It runs on CPU and normalizes vectors. FAISS searches those vectors, returning the nearest passages and their source lines. The returned distance is not a probability or a confidence score.

## Build and search

Install the retrieval dependencies in the existing Python environment:

```bash
.venv/bin/python -m pip install -r requirements-retrieval.txt
.venv/bin/python scripts/build_policy_index.py
.venv/bin/python scripts/search_policies.py --query "How do I request an annual fee waiver?"
```

The first build downloads the sentence-transformer model if it is not already cached. Index files are saved in `index/policies/` and ignored by Git. The build refuses to replace an existing directory. To rebuild after a document changes, remove the old `index/policies/` directory manually and run the build command again.

The saved index contains native `vectors.faiss`, `chunks.json`, and `manifest.json`. There are no pickle-backed metadata files. When opening an index, the loader checks its stored file hashes, source document hashes, vector count, and embedding model name. It also checks that source line ranges are valid. Only load indexes built in a trusted workspace; file hashes are integrity checks, not proof of a file's origin.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The retrieval contract suite uses synthetic documents and a deterministic fake embedding implementation so it does not download a model or make API requests. The retrieval integration suite additionally exercises the actual LangChain loader, splitter, and FAISS routines when the packages are installed. These checks do not measure semantic retrieval quality over every possible customer question.

The main RAG workflow and Gradio interface will use the retrieved passages in later steps. Retrieval alone does not authorize a waiver or decide whether an account qualifies for an offer.
