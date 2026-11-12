"""Validate local sources and prepare the verified index for the chat app."""
from __future__ import annotations

from pathlib import Path

from .retrieval import MODEL_NAME, open_index, search_loaded_index
from .source_checks import inspect_sources


def prepare_startup(
    data_dir: str | Path,
    docs_dir: str | Path,
    index_dir: str | Path,
    *,
    embeddings=None,
    model_name: str = MODEL_NAME,
):
    """Check sources once and return a summary plus a loaded search callable."""
    report = inspect_sources(data_dir, docs_dir)
    store = open_index(
        docs_dir, index_dir, embeddings=embeddings, model_name=model_name,
    )
    if store.index.ntotal <= 0:
        raise ValueError("Saved policy index has no vectors.")
    summary = {
        "customer_records": report["customer_records"],
        "sentiment_available": report["sentiment_available"],
        "policy_documents": len(report["documents"]),
        "indexed_passages": int(store.index.ntotal),
        "embedding_model": model_name,
    }
    return summary, search_loaded_index(store)


def check_startup(
    data_dir: str | Path,
    docs_dir: str | Path,
    index_dir: str | Path,
    *,
    embeddings=None,
    model_name: str = MODEL_NAME,
) -> dict:
    """Return the source and index summary without exposing the loaded index."""
    summary, _search = prepare_startup(
        data_dir, docs_dir, index_dir,
        embeddings=embeddings, model_name=model_name,
    )
    return summary
