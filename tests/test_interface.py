"""Offline tests for the UI callback and model adapter."""
import os
import unittest
from unittest.mock import Mock, patch

from customer_retention.assistant import Answer
from customer_retention.interface import make_handler
from customer_retention.model_adapter import create_generator


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.db = Mock()
        self.answer = Mock(return_value=Answer(
            "Policy requires review [S1].", True,
            ({"marker": "[S1]", "source": "retention_policy.txt",
              "line_start": 3, "line_end": 4, "chunk_id": "retention_policy.txt:1"},),
            True,
        ))
        self.generate = Mock()
        self.handle = make_handler(
            customer_db=self.db, docs_dir="docs", index_dir="index",
            generate=self.generate, answer=self.answer,
        )

    def test_normal_response_has_citation_details(self):
        response = self.handle(" 42 ", "Can I request a review?")
        self.assertIn("retention_policy.txt (lines 3-4)", response)
        self.assertIn("Policy requires review", response)
        self.assertEqual(self.answer.call_args.args[:2], ("Can I request a review?", "42"))

    def test_missing_customer_id(self):
        self.assertEqual(self.handle(" ", "Question"), "Enter a customer ID.")
        self.answer.assert_not_called()

    def test_missing_question(self):
        self.assertEqual(self.handle("42", " "), "Enter a question.")
        self.answer.assert_not_called()

    def test_invalid_input_is_shown_without_traceback(self):
        self.answer.side_effect = ValueError("Customer ID must contain only digits.")
        self.assertIn("Invalid input", self.handle("abc", "Question"))

    def test_internal_error_is_hidden(self):
        self.answer.side_effect = RuntimeError("secret token")
        response = self.handle("42", "Question")
        self.assertNotIn("secret token", response)
        self.assertIn("could not be completed", response)

    def test_uncited_result_has_no_source_section(self):
        self.answer.return_value = Answer("No policy found.", True, (), True)
        self.assertEqual(self.handle("42", "Question"), "No policy found.")

    def test_requires_explicit_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                create_generator()


if __name__ == "__main__":
    unittest.main()
