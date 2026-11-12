"""Evaluate ranked document retrieval against reviewed source labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .retrieval import MODEL_NAME, search_policies
from .source_checks import REFERENCE_FILES


def load_judgments(path: str | Path) -> list[dict]:
    """Read questions with independently reviewed relevant source filenames."""
    cases = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}.") from exc
            if not isinstance(item, dict) or set(item) != {
                "id", "question", "relevant_sources",
            }:
                raise ValueError(f"Invalid evaluation fields at line {line_number}.")
            case_id, question, sources = (
                item["id"], item["question"], item["relevant_sources"],
            )
            if (not isinstance(case_id, str) or not case_id.strip()
                    or len(case_id) > 100 or case_id in seen
                    or not isinstance(question, str) or not question.strip()
                    or len(question) > 1000 or "\x00" in question
                    or not isinstance(sources, list) or not sources
                    or len(sources) != len(set(sources))
                    or any(not isinstance(name, str) or name not in REFERENCE_FILES
                           for name in sources)):
                raise ValueError(f"Invalid evaluation case at line {line_number}.")
            seen.add(case_id)
            cases.append(item)
    if not cases:
        raise ValueError("Evaluation file contains no cases.")
    return cases


def evaluate_retrieval(
    cases: list[dict],
    docs_dir: str | Path,
    index_dir: str | Path,
    *,
    k: int = 4,
    model_name: str = MODEL_NAME,
    search: Callable = search_policies,
) -> dict:
    """Measure source-level hit rate, recall, and reciprocal rank at k.

    Multiple chunks from one file count as one retrieved source. The first
    occurrence determines its rank. The scores are based on reviewer labels,
    not on the truthfulness of any generated answer.
    """
    if type(k) is not int or not 1 <= k <= 10:
        raise ValueError("k must be an integer between 1 and 10.")
    if not cases:
        raise ValueError("No evaluation cases supplied.")
    outcomes = []
    for case in cases:
        hits = search(
            case["question"], docs_dir, index_dir, k=k, model_name=model_name,
        )
        names = []
        for hit in hits:
            name = hit["source"]
            if name not in REFERENCE_FILES:
                raise ValueError("Search returned an unknown source.")
            if name not in names:
                names.append(name)
        expected = set(case["relevant_sources"])
        matched = expected.intersection(names)
        ranks = [i for i, name in enumerate(names, 1) if name in expected]
        rank = min(ranks) if ranks else None
        outcomes.append({
            "id": case["id"],
            "retrieved_sources": names,
            "expected_sources": sorted(expected),
            "hit": rank is not None,
            "recall": len(matched) / len(expected),
            "reciprocal_rank": (1 / rank) if rank else 0.0,
        })
    total = len(outcomes)
    return {
        "evaluation_type": "reviewed_source_retrieval",
        "model_name": model_name,
        "k": k,
        "total": total,
        "hit_rate": sum(item["hit"] for item in outcomes) / total,
        "mean_source_recall": sum(item["recall"] for item in outcomes) / total,
        "mean_reciprocal_rank": sum(
            item["reciprocal_rank"] for item in outcomes
        ) / total,
        "cases": outcomes,
    }
