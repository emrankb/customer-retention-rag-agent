"""Offline checks for the assistant's observable answer behavior.

This is a contract evaluation, not a measure of semantic answer correctness.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .assistant import answer_question

_VALID_BEHAVIORS = {"answer", "missing_customer", "missing_evidence", "uncited"}
_REQUIRED_KEYS = {"id", "question", "customer_id", "customer_found", "passages",
                  "model_reply", "expected_behavior", "expected_review"}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load independent test cases, rejecting duplicates or incomplete records."""
    cases = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {number}.") from exc
            if not isinstance(case, dict) or set(case) != _REQUIRED_KEYS:
                raise ValueError(f"Invalid evaluation fields on line {number}.")
            if (not isinstance(case["id"], str) or not case["id"].strip()
                    or case["id"] in seen or not isinstance(case["question"], str)
                    or not case["question"].strip()
                    or not isinstance(case["customer_id"], str)
                    or not case["customer_id"].isascii()
                    or not case["customer_id"].isdecimal()
                    or not isinstance(case["customer_found"], bool)
                    or not isinstance(case["passages"], list)
                    or not isinstance(case["model_reply"], str)
                    or case["expected_behavior"] not in _VALID_BEHAVIORS
                    or not isinstance(case["expected_review"], bool)):
                raise ValueError(f"Invalid evaluation case on line {number}.")
            for p in case["passages"]:
                if (not isinstance(p, dict)
                        or set(p) != {"text", "source", "line_start", "line_end", "chunk_id"}
                        or not isinstance(p["text"], str)
                        or not p["text"].strip()
                        or not isinstance(p["source"], str) or not p["source"]
                        or not isinstance(p["chunk_id"], str) or not p["chunk_id"]
                        or type(p["line_start"]) is not int
                        or type(p["line_end"]) is not int
                        or not 1 <= p["line_start"] <= p["line_end"]):
                    raise ValueError(f"Invalid passage on line {number}.")
            seen.add(case["id"])
            cases.append(case)
    if not cases:
        raise ValueError("No evaluation cases found.")
    return cases


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Run specified cases with fake retrieval and model outputs.

    Checks whether the assistant takes the expected branch and returns correct
    citation markers. Does not determine if policy claims are actually true.
    """
    results = []
    for case in cases:
        class TestDB:
            def get_context(self, customer_id):
                if not case["customer_found"]:
                    return None
                return {
                    "customer_id": customer_id, "tenure_years": 2,
                    "num_products": 1, "has_credit_card": False,
                    "is_active_member": True, "historical_exited": False,
                    "sentiment_label": "Neutral",
                }

        calls = {"search": 0, "model": 0}

        def fake_search(question, docs_dir, index_dir, *, k):
            calls["search"] += 1
            return case["passages"]

        def fake_model(prompt):
            calls["model"] += 1
            return case["model_reply"]

        answer = answer_question(
            case["question"], case["customer_id"], customer_db=TestDB(),
            docs_dir="unused", index_dir="unused",
            generate=fake_model, search=fake_search,
        )
        behavior = ("missing_customer" if not answer.customer_found
                    else "missing_evidence" if not case["passages"]
                    else "uncited" if not answer.citations else "answer")
        expected_calls = (
            (0, 0) if case["expected_behavior"] == "missing_customer" else
            (1, 0) if case["expected_behavior"] == "missing_evidence" else
            (1, 1)
        )
        expected_markers = {
            f"[S{i}]" for i in range(1, len(case["passages"]) + 1)
        }
        citations_valid = all(
            c["marker"] in expected_markers and any(
                c["source"] == p["source"]
                and c["line_start"] == p["line_start"]
                and c["line_end"] == p["line_end"]
                for p in case["passages"]
            ) for c in answer.citations
        )
        passed = (
            behavior == case["expected_behavior"]
            and answer.needs_review == case["expected_review"]
            and (calls["search"], calls["model"]) == expected_calls
            and citations_valid
        )
        results.append({
            "id": case["id"], "passed": passed,
            "expected_behavior": case["expected_behavior"],
            "observed_behavior": behavior,
            "expected_review": case["expected_review"],
            "observed_review": answer.needs_review,
            "citations_valid": citations_valid,
            "search_calls": calls["search"], "model_calls": calls["model"],
        })
    passed_count = sum(result["passed"] for result in results)
    return {
        "evaluation_type": "offline_contract",
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": passed_count / len(results),
        "cases": results,
    }
