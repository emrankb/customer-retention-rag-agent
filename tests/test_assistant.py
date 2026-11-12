"""Fast offline tests for grounded assistant orchestration."""
import unittest
from unittest.mock import Mock

from customer_retention.assistant import answer_question, make_prompt


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.db = Mock()
        self.db.get_context.return_value = {
            "customer_id": "42",
            "surname": "DoNotSend",
            "credit_score": 600,
            "tenure_years": 3,
            "num_products": 2,
            "has_credit_card": True,
            "is_active_member": True,
            "historical_exited": False,
            "sentiment_label": "Neutral",
        }
        self.hits = [{
            "text": "A waiver request must undergo review.",
            "source": "fee_waiver_policy.txt",
            "line_start": 2,
            "line_end": 3,
            "chunk_id": "fee_waiver_policy.txt:1",
        }]
        self.search = Mock(return_value=self.hits)
        self.generate = Mock(return_value="Requests require review [S1].")

    def ask(self, **kwargs):
        return answer_question(
            "Can I get a fee waiver?", "42", customer_db=self.db,
            docs_dir="unused", index_dir="unused",
            search=self.search, generate=self.generate, **kwargs
        )

    def test_citations_and_review(self):
        result = self.ask()
        self.assertTrue(result.customer_found)
        self.assertTrue(result.needs_review)
        self.assertEqual(result.citations[0]["source"], "fee_waiver_policy.txt")
        self.assertIn("authorized bank review", result.text)
        self.assertEqual(self.search.call_args.kwargs["k"], 4)

    def test_missing_customer_does_not_call_search_or_model(self):
        self.db.get_context.return_value = None
        result = self.ask()
        self.assertFalse(result.customer_found)
        self.search.assert_not_called()
        self.generate.assert_not_called()

    def test_missing_evidence_does_not_call_model(self):
        self.search.return_value = []
        result = self.ask()
        self.assertTrue(result.needs_review)
        self.generate.assert_not_called()

    def test_uncited_answer_is_not_returned(self):
        self.generate.return_value = "Approved."
        result = self.ask()
        self.assertNotIn("Approved", result.text)
        self.assertFalse(result.citations)

    def test_unknown_citation_is_not_returned(self):
        self.generate.return_value = "Policy says yes [S2]."
        self.assertFalse(self.ask().citations)

    def test_valid_multiple_citations(self):
        self.search.return_value = self.hits + [{
            "text": "Requests must be submitted through customer service.",
            "source": "complaints_handling.txt",
            "line_start": 4, "line_end": 5,
            "chunk_id": "complaints_handling.txt:1",
        }]
        self.generate.return_value = "Two steps are described [S1] [S2]."
        result = self.ask()
        self.assertEqual([c["marker"] for c in result.citations], ["[S1]", "[S2]"])

    def test_valid_citation_plus_unknown_marker_rejected(self):
        self.generate.return_value = "Review required [S1], guaranteed [S99]."
        result = self.ask()
        self.assertFalse(result.citations)
        self.assertNotIn("guaranteed", result.text)

    def test_noncanonical_or_malformed_marker_rejected(self):
        for marker in ("[S0]", "[S01]", "[Sx]", "[S-1]", "[S 1]"):
            with self.subTest(marker=marker):
                self.generate.return_value = f"Review required [S1] {marker}."
                self.assertFalse(self.ask().citations)

    def test_excessive_or_nul_response_rejected(self):
        for response in ("Policy [S1]." * 1000, "Policy [S1].\x00"):
            with self.subTest(response_length=len(response)):
                self.generate.return_value = response
                self.assertFalse(self.ask().citations)

    def test_no_private_demographics_in_prompt(self):
        prompt = make_prompt("What is the policy?", self.db.get_context("42"), self.hits)
        self.assertNotIn("DoNotSend", prompt)
        self.assertNotIn("600", prompt)
        self.assertIn("historical_exited", prompt)

    def test_conflicting_retention_rules_are_flagged_without_raw_credit_score(self):
        profile = self.db.get_context("42")
        profile["tenure_years"] = 2
        profile["credit_score"] = 619
        passages = [{
            "source": "retention_policy.txt", "text": "Policy segments depend on tenure or score.",
            "line_start": 9, "line_end": 12,
        }]
        prompt = make_prompt("What can we offer?", profile, passages)
        self.assertIn("retention_tenure_rule: moderate-value", prompt)
        self.assertIn("retention_credit_score_rule: low-value", prompt)
        self.assertIn("overlapping conditions", prompt)
        self.assertNotIn("619", prompt)
        self.assertNotIn("credit_score: 619", prompt)

    def test_retention_segment_boundaries(self):
        passages = [{"source": "retention_policy.txt", "text": "Policy", "line_start": 1, "line_end": 1}]
        cases = [
            (0, 649, "low-value", "low-value"),
            (1, 650, "moderate-value", "moderate-value"),
            (3, 750, "moderate-value", "moderate-value"),
            (4, 751, "high-value", "high-value"),
            (4, 649, "high-value", "low-value"),
        ]
        for tenure, score, by_tenure, by_score in cases:
            with self.subTest(tenure=tenure, score=score):
                profile = {**self.db.get_context("42"),
                           "tenure_years": tenure, "credit_score": score}
                prompt = make_prompt("Retention options?", profile, passages)
                self.assertIn(f"retention_tenure_rule: {by_tenure}", prompt)
                self.assertIn(f"retention_credit_score_rule: {by_score}", prompt)

    def test_retention_band_not_sent_without_retention_evidence(self):
        prompt = make_prompt("Account details?", self.db.get_context("42"), self.hits)
        self.assertNotIn("retention_credit_score_rule:", prompt)

    def test_retention_rules_missing_credit_score_are_not_assumed(self):
        profile = {**self.db.get_context("42")}
        profile.pop("credit_score")
        passages = [{"source": "retention_policy.txt", "text": "Policy", "line_start": 1, "line_end": 1}]
        prompt = make_prompt("Retention options?", profile, passages)
        self.assertIn("retention_segment_status: unknown", prompt)

    def test_retention_offer_question_requires_review(self):
        self.search.return_value = [{
            "source": "retention_policy.txt",
            "line_start": 14, "line_end": 20,
            "chunk_id": "retention_policy.txt:1",
            "text": "A retention offer needs approval.",
        }]
        answer = answer_question(
            "What should we offer to retain this customer?", "42",
            customer_db=self.db, docs_dir="unused", index_dir="unused",
            generate=self.generate, search=self.search,
        )
        self.assertTrue(answer.needs_review)
        self.assertIn("authorized bank review", answer.text)

    def test_bad_question_and_k_rejected(self):
        with self.assertRaises(ValueError):
            self.ask(k=0)
        with self.assertRaises(ValueError):
            answer_question(" ", "42", customer_db=self.db, docs_dir="x",
                            index_dir="y", generate=self.generate, search=self.search)


if __name__ == "__main__":
    unittest.main()
