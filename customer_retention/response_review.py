"""Validate and summarize independently reviewed model responses.

Reviewers grade actual observed answers; this module never assigns scores.
"""
from __future__ import annotations

import json
from pathlib import Path

DIMENSIONS = ("grounding", "personalization", "tone", "coherence")
_ALLOWED = {"id", "question", "response", "sources", "model", "ratings", "notes"}
_SOURCE_KEYS = {"source", "line_start", "line_end"}


def load_reviews(path: str | Path) -> list[dict]:
    seen = set()
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}.") from exc
            if not isinstance(row, dict) or set(row) != _ALLOWED:
                raise ValueError(f"Incorrect fields at line {line_number}.")
            if (not isinstance(row["id"], str) or not row["id"].strip()
                    or row["id"] in seen
                    or not isinstance(row["question"], str) or not row["question"].strip()
                    or not isinstance(row["response"], str) or not row["response"].strip()
                    or not isinstance(row["model"], str) or not row["model"].strip()
                    or not isinstance(row["sources"], list)
                    or not isinstance(row["ratings"], dict)
                    or set(row["ratings"]) != set(DIMENSIONS)
                    or not isinstance(row["notes"], dict)
                    or set(row["notes"]) != set(DIMENSIONS)):
                raise ValueError(f"Invalid review record at line {line_number}.")
            for dimension in DIMENSIONS:
                value = row["ratings"][dimension]
                note = row["notes"][dimension]
                if (type(value) is not int or value not in (0, 1, 2)
                        or not isinstance(note, str) or not note.strip()):
                    raise ValueError(
                        f"Missing score or supporting note for {dimension} "
                        f"at line {line_number}."
                    )
            for source in row["sources"]:
                if (not isinstance(source, dict) or set(source) != _SOURCE_KEYS
                        or not isinstance(source["source"], str) or not source["source"]
                        or type(source["line_start"]) is not int
                        or type(source["line_end"]) is not int
                        or not 1 <= source["line_start"] <= source["line_end"]):
                    raise ValueError(f"Invalid source reference at line {line_number}.")
            seen.add(row["id"])
            records.append(row)
    if not records:
        raise ValueError("No completed reviews found.")
    return records


def summarize_reviews(records: list[dict]) -> dict:
    if not records:
        raise ValueError("No completed reviews found.")
    for record in records:
        if any(type(record["ratings"][d]) is not int
               or record["ratings"][d] not in (0, 1, 2) for d in DIMENSIONS):
            raise ValueError("Review must have four completed 0–2 scores.")
    totals = {d: sum(row["ratings"][d] for row in records) for d in DIMENSIONS}
    return {
        "evaluation_type": "human_reviewed_model_responses",
        "reviewed_responses": len(records),
        "score_scale": "0=poor, 1=partial, 2=meets criterion",
        "mean_scores": {d: totals[d] / len(records) for d in DIMENSIONS},
        "score_counts": {
            d: {str(score): sum(row["ratings"][d] == score for row in records)
                for score in (0, 1, 2)}
            for d in DIMENSIONS
        },
        "case_ids": [row["id"] for row in records],
    }
