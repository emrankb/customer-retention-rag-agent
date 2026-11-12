"""Test source-level retrieval scoring and judgment validation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from customer_retention.retrieval_evaluation import (
    evaluate_retrieval, load_judgments,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {"id": "a", "question": "fee waiver",
             "relevant_sources": ["fee_waiver_policy.txt"]},
            {"id": "b", "question": "rewards",
             "relevant_sources": ["rewards_and_benefits.txt", "product_summary.txt"]},
            {"id": "c", "question": "complaints",
             "relevant_sources": ["complaints_handling.txt"]},
        ]

    def test_metrics_and_duplicate_chunks(self):
        answers = {
            "fee waiver": ["fee_waiver_policy.txt", "fee_waiver_policy.txt",
                           "product_summary.txt"],
            "rewards": ["faq_accounts.md", "rewards_and_benefits.txt"],
            "complaints": ["faq_accounts.md", "product_summary.txt"],
        }

        def search(question, docs_dir, index_dir, *, k, model_name):
            return [{"source": name} for name in answers[question][:k]]

        report = evaluate_retrieval(
            self.cases, "docs", "index", k=3,
            model_name="test-embedding", search=search,
        )
        self.assertEqual(report["total"], 3)
        self.assertAlmostEqual(report["hit_rate"], 2 / 3)
        self.assertAlmostEqual(report["mean_source_recall"], (1 + .5 + 0) / 3)
        self.assertAlmostEqual(report["mean_reciprocal_rank"], (1 + .5 + 0) / 3)
        self.assertEqual(report["cases"][0]["retrieved_sources"],
                         ["fee_waiver_policy.txt", "product_summary.txt"])
        self.assertEqual(report["cases"][2]["reciprocal_rank"], 0)

    def test_all_misses_have_zero_scores(self):
        report = evaluate_retrieval(
            self.cases[:1], "docs", "index",
            search=lambda *args, **kwargs: [],
        )
        self.assertEqual(report["hit_rate"], 0)
        self.assertEqual(report["mean_source_recall"], 0)
        self.assertEqual(report["mean_reciprocal_rank"], 0)

    def test_invalid_k(self):
        for k in (0, 11, True, 2.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                evaluate_retrieval(self.cases, "d", "i", k=k)

    def test_unknown_returned_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown source"):
            evaluate_retrieval(
                self.cases[:1], "docs", "index",
                search=lambda *args, **kwargs: [{"source": "private.csv"}],
            )

    def test_read_and_validate_cases(self):
        self.assertEqual(self.read_cases(self.cases), self.cases)

    def test_bad_judgments_rejected(self):
        variants = [
            self.cases + [self.cases[0]],
            [{"id": "a", "question": "", "relevant_sources": ["faq_accounts.md"]}],
            [{"id": "a", "question": "x", "relevant_sources": []}],
            [{"id": "a", "question": "x", "relevant_sources": ["made_up.txt"]}],
            [{"id": "a", "question": "x", "relevant_sources": ["faq_accounts.md",
                                                                  "faq_accounts.md"]}],
            [{"id": "a", "question": "x", "relevant_sources": [123]}],
            [{"id": "a", "question": "x", "relevant_sources": ["faq_accounts.md"],
              "extra": "unexpected"}],
        ]
        for variant in variants:
            with self.subTest(case=variant), self.assertRaises(ValueError):
                self.read_cases(variant)

    def test_empty_and_malformed_json_rejected(self):
        for data in ("", "{broken}\n"):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as temp:
                p = Path(temp) / "cases.jsonl"
                p.write_text(data)
                with self.assertRaises(ValueError):
                    load_judgments(p)

    @staticmethod
    def read_cases(cases):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8",
            )
            return load_judgments(path)


if __name__ == "__main__":
    unittest.main()
