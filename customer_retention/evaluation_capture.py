"""Capture actual RAG responses and citations for later human review.

This module does not score answers or make external model calls on its own.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .rag_chain import make_rag_chain

DIMENSIONS = ("grounding", "personalization", "tone", "coherence")
_FIELDS = {"id", "customer_id", "question"}


def load_questions(path: str | Path) -> list[dict]:
    questions = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {number}.") from exc
            if (not isinstance(row, dict) or set(row) != _FIELDS
                    or not isinstance(row["id"], str) or not row["id"].strip()
                    or len(row["id"]) > 100 or row["id"] in seen
                    or not isinstance(row["question"], str)
                    or not row["question"].strip() or len(row["question"]) > 1000
                    or "\x00" in row["question"]
                    or not isinstance(row["customer_id"], str)
                    or not row["customer_id"].isascii()
                    or not row["customer_id"].isdecimal()):
                raise ValueError(f"Invalid evaluation question on line {number}.")
            questions.append(row)
            seen.add(row["id"])
    if not questions:
        raise ValueError("No evaluation questions found.")
    return questions


def capture_responses(
    cases: list[dict],
    *,
    customer_db,
    docs_dir,
    index_dir,
    generate: Callable[[str], str],
    search: Callable,
    model_name: str,
) -> tuple[list[dict], list[dict]]:
    """Capture source references and response text from actual chain executions.

    The first result is compatible with the response-review format but has
    unfilled human ratings. The second records retrieved passages separately.
    """
    if not cases:
        raise ValueError("No cases provided.")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Model name is required.")
    captured = []
    evidence = []
    for case in cases:
        passages = []

        def recording_search(question, docs_dir, index_dir, *, k):
            hits = search(question, docs_dir, index_dir, k=k)
            passages.extend(hits)
            return hits

        chain = make_rag_chain(
            customer_db=customer_db, docs_dir=docs_dir,
            index_dir=index_dir, generate=generate, search=recording_search,
        )
        result = chain.invoke({
            "question": case["question"],
            "customer_id": case["customer_id"],
            "history": [],
        })
        sources = [
            {
                "source": c["source"],
                "line_start": c["line_start"],
                "line_end": c["line_end"],
            }
            for c in result.citations
        ]
        captured.append({
            "id": case["id"],
            "question": case["question"],
            "response": result.text,
            "sources": sources,
            "model": model_name,
            "ratings": {dimension: None for dimension in DIMENSIONS},
            "notes": {dimension: "" for dimension in DIMENSIONS},
        })
        evidence.append({
            "id": case["id"],
            "retrieved_passages": [
                {
                    "source": hit["source"],
                    "line_start": hit["line_start"],
                    "line_end": hit["line_end"],
                    "text": hit["text"],
                    "chunk_id": hit["chunk_id"],
                }
                for hit in passages
            ],
            "customer_found": result.customer_found,
            "needs_review": result.needs_review,
        })
    return captured, evidence
