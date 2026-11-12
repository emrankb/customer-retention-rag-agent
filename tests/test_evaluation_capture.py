"""Test response capture without customer files or paid model calls."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from customer_retention.evaluation_capture import load_questions, capture_responses


class EvaluationCaptureTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"id": "a", "customer_id": "42",
                       "question": "Can I request a waiver?"}]

    def _load(self, lines):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "questions.jsonl"
            target.write_text(lines, encoding="utf-8")
            return load_questions(target)

    def test_load_questions(self):
        lines = "".join(json.dumps(row) + "\n" for row in self.cases)
        self.assertEqual(self._load(lines), self.cases)

    def test_rejects_duplicate_ids(self):
        line = json.dumps(self.cases[0]) + "\n"
        with self.assertRaises(ValueError):
            self._load(line + line)

    def test_rejects_missing_or_bad_customer_id(self):
        for identifier in (None, "abc", "", "-5"):
            with self.subTest(identifier=identifier):
                row = {**self.cases[0], "customer_id": identifier}
                with self.assertRaises(ValueError):
                    self._load(json.dumps(row) + "\n")

    def test_rejects_missing_questions(self):
        with self.assertRaises(ValueError):
            self._load("")
        with self.assertRaises(ValueError):
            self._load("{broken")

    def test_capture_marks_scores_unreviewed(self):
        try:
            from langchain_core.runnables import RunnableLambda  # noqa: F401
        except ImportError:
            self.skipTest("Install LangChain Core for capture integration")
        db = Mock()
        db.get_context.return_value = {
            "customer_id": "42", "tenure_years": 3,
            "num_products": 1, "has_credit_card": True,
            "is_active_member": True, "historical_exited": False,
            "sentiment_label": "Neutral",
        }
        search = Mock(return_value=[{
            "source": "fee_waiver_policy.txt", "line_start": 1,
            "line_end": 2, "chunk_id": "fee_waiver_policy.txt:1",
            "text": "Requests need review.",
        }])
        model = Mock(return_value="Requests need review [S1].")
        reviews, evidence = capture_responses(
            self.cases, customer_db=db, docs_dir="docs", index_dir="index",
            generate=model, search=search, model_name="offline-test",
        )
        self.assertEqual(reviews[0]["response"], "Requests need review [S1].\n\n"
                         "Any fee waiver, refund, account eligibility, or retention "
                         "offer requires verification and authorized bank review.")
        self.assertEqual(reviews[0]["sources"][0]["source"], "fee_waiver_policy.txt")
        self.assertTrue(all(x is None for x in reviews[0]["ratings"].values()))
        self.assertEqual(evidence[0]["retrieved_passages"][0]["text"], "Requests need review.")
        self.assertEqual(search.call_count, 1)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(db.get_context.call_count, 1)

    def test_summary_rejects_unreviewed_capture(self):
        from customer_retention.response_review import summarize_reviews
        with self.assertRaises(ValueError):
            summarize_reviews([{
                "ratings": dict.fromkeys(
                    ("grounding", "personalization", "tone", "coherence")
                )
            }])


if __name__ == "__main__":
    unittest.main()
