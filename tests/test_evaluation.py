"""Evaluation loader, scoring, and failure tests."""
import json
import tempfile
import unittest
from pathlib import Path

from customer_retention.evaluation import evaluate_cases, load_cases

FIXTURES = Path(__file__).parent / "fixtures/answer_cases.jsonl"


class EvaluationTests(unittest.TestCase):
    def test_fixture_cases_pass(self):
        results = evaluate_cases(load_cases(FIXTURES))
        self.assertEqual(results["total"], 6)
        self.assertEqual(results["passed"], 6)
        self.assertEqual(results["failed"], 0)
        self.assertEqual(results["evaluation_type"], "offline_contract")

    def test_wrong_expectation_fails(self):
        cases = load_cases(FIXTURES)
        cases[0]["expected_behavior"] = "uncited"
        result = evaluate_cases(cases)
        self.assertEqual(result["failed"], 1)
        self.assertFalse(result["cases"][0]["passed"])

    def test_duplicate_identifier_rejected(self):
        lines = FIXTURES.read_text().splitlines()
        self.assert_invalid("\n".join([lines[0], lines[0]]) + "\n")

    def test_invalid_passage_rejected(self):
        case = json.loads(FIXTURES.read_text().splitlines()[0])
        case["passages"][0]["line_start"] = -1
        self.assert_invalid(json.dumps(case) + "\n")

    def test_bad_json_rejected(self):
        self.assert_invalid('{"id": ')

    def test_empty_case_file_rejected(self):
        self.assert_invalid(" \n")

    def test_missing_field_rejected(self):
        case = json.loads(FIXTURES.read_text().splitlines()[0])
        del case["question"]
        self.assert_invalid(json.dumps(case) + "\n")

    def assert_invalid(self, content):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.jsonl"
            path.write_text(content)
            with self.assertRaises(ValueError):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()
