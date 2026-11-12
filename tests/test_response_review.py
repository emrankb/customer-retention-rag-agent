"""Review summary tests. These are scoring-contract tests, not model results."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from customer_retention.response_review import (
    DIMENSIONS, load_reviews, summarize_reviews,
)


def example(id="one", scores=None):
    values = scores or dict.fromkeys(DIMENSIONS, 2)
    return {
        "id": id,
        "question": "What is the card policy?",
        "response": "It depends on the card type [S1].",
        "sources": [{"source": "product_summary.txt",
                     "line_start": 3, "line_end": 5}],
        "model": "test-response-only",
        "ratings": values,
        "notes": {name: "Checked against the supplied example." for name in DIMENSIONS},
    }


class ResponseReviewTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "reviews.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return load_reviews(path)

    def test_summary_math(self):
        one = example(scores={"grounding": 2, "personalization": 1,
                              "tone": 0, "coherence": 2})
        two = example("two", {"grounding": 0, "personalization": 2,
                              "tone": 2, "coherence": 0})
        summary = summarize_reviews(self.load([one, two]))
        self.assertEqual(summary["reviewed_responses"], 2)
        self.assertEqual(summary["mean_scores"]["grounding"], 1)
        self.assertEqual(summary["mean_scores"]["personalization"], 1.5)
        self.assertEqual(summary["score_counts"]["tone"]["0"], 1)
        self.assertNotIn("response", str(summary))

    def test_duplicate_case_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid review record"):
            self.load([example(), example()])

    def test_missing_reviewed_score_is_rejected(self):
        case = example()
        case["ratings"]["grounding"] = None
        with self.assertRaisesRegex(ValueError, "Missing score"):
            self.load([case])

    def test_boolean_score_rejected(self):
        case = example()
        case["ratings"]["tone"] = True
        with self.assertRaises(ValueError):
            self.load([case])

    def test_score_without_reason_rejected(self):
        case = example()
        case["notes"]["coherence"] = ""
        with self.assertRaisesRegex(ValueError, "supporting note"):
            self.load([case])

    def test_invalid_source_line_rejected(self):
        case = example()
        case["sources"][0]["line_start"] = 0
        with self.assertRaisesRegex(ValueError, "Invalid source"):
            self.load([case])

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            self.load([])

    def test_invalid_score_range_rejected(self):
        case = example(scores=dict.fromkeys(DIMENSIONS, 3))
        with self.assertRaises(ValueError):
            summarize_reviews([case])


if __name__ == "__main__":
    unittest.main()
