"""Validate local customer sources and policy index before opening the app."""
from __future__ import annotations

from pathlib import Path

from .retrieval import MODEL_NAME, open_index
from .source_checks import inspect_sources


def check_startup(
    data_dir: str | Path,
    docs_dir: str | Path,
    index_dir: str | Path,
    *,
    embeddings=None,
    model_name: str = MODEL_NAME,
) -> dict:
    """Check both customer tables, policy documents, and the saved FAISS index.

    When the sentiment table is absent, the original customer table is used.
    Passing a test embedding implementation avoids downloading a model in CI.
    """
    report = inspect_sources(data_dir, docs_dir)
    store = open_index(
        docs_dir, index_dir, embeddings=embeddings, model_name=model_name,
    )
    if store.index.ntotal <= 0:
        raise ValueError("Saved policy index has no vectors.")
    return {
        "customer_records": report["customer_records"],
        "sentiment_available": report["sentiment_available"],
        "policy_documents": len(report["documents"]),
        "indexed_passages": int(store.index.ntotal),
        "embedding_model": model_name,
    }
