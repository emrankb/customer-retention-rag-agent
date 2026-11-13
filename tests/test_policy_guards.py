"""Regression checks based on failures observed in the private Task 6 review.

Fixtures are fictional excerpts; no original banking documents or customer
profiles are committed. No paid model calls are made.
"""
import unittest
from unittest.mock import Mock

from customer_retention.assistant import answer_question


def hit(source, text, start=1, end=25, index=1):
    return {
        "source": source, "text": text,
        "line_start": start, "line_end": end,
        "chunk_id": f"{source}:{index}",
    }


class GroundedResponseGuardsTests(unittest.TestCase):
    def setUp(self):
        self.db = Mock()
        self.db.get_context.return_value = {
            "customer_id": "42", "tenure_years": 2, "credit_score": 619,
            "num_products": 1, "is_active_member": True,
            "historical_exited": False, "has_credit_card": True,
            "sentiment_label": "Negative",
        }
        self.generate = Mock(return_value="Unsupported 50% offer [S1].")
        self.hits = []
        self.search = Mock(side_effect=lambda *_args, **_kwargs: self.hits)

    def ask(self, question):
        return answer_question(
            question, "42", customer_db=self.db,
            docs_dir="unused", index_dir="unused",
            generate=self.generate, search=self.search,
        )

    def test_ambiguous_waiver_does_not_choose_moderate_offer(self):
        self.hits = [
            hit("fee_waiver_policy.txt",
                "Fee Waiver Policy\n1. Eligibility\nNo late payment in the last 3 cycles.\n"
                "Credit score >700 or verified hardship.", 1, 6),
            hit("retention_policy.txt",
                "3. Retention Offers\nHigh-value: rewards.\n"
                "Moderate-value: 50% annual fee waiver.\nLow-value: counselling.", 14, 18),
        ]
        result = self.ask("How can I waive my annual fee?")
        self.assertTrue(result.needs_review)
        self.assertIn("different retention segments", result.text)
        self.assertNotIn("50% annual fee waiver", result.text)
        self.assertNotIn("619", result.text)
        self.assertIn("[S1]", result.text)
        self.assertIn("[S2]", result.text)
        self.generate.assert_not_called()

    def test_ambiguous_retention_does_not_select_offer(self):
        self.hits = [
            hit("retention_policy.txt",
                "3. Retention Offers\nModerate-value: 50% annual fee waiver.\n"
                "Low-value: free add-on card.", 14, 17),
        ]
        result = self.ask("What should we offer to retain this customer?")
        self.assertTrue(result.needs_review)
        self.assertIn("cannot determine one segment", result.text)
        self.assertNotIn("50% annual fee waiver", result.text)
        self.assertIn("[S1]", result.text)
        self.generate.assert_not_called()

    def test_different_customer_same_tier_does_not_trigger_guard(self):
        self.db.get_context.return_value["credit_score"] = 700
        self.hits = [hit("retention_policy.txt", "3. Retention Offers\nModerate-value: a waiver.")]
        result = self.ask("What retention offers exist?")
        self.generate.assert_called_once()
        self.assertTrue(result.citations)

    def test_missing_segment_source_does_not_trigger_guard(self):
        self.hits = [hit("fee_waiver_policy.txt", "1. Eligibility\nAnnual fee waiver rules.")]
        self.ask("Can I waive a fee?")
        self.generate.assert_called_once()

    def test_card_comparison_cites_gold_and_platinum_sections(self):
        self.hits = [
            hit("product_summary.txt",
                "3. Platinum Credit Card\n - Annual fee: ₹2,000\n"
                " - Limit: ₹2,00,000–₹5,00,000\n"
                " - Rewards: 5X on travel.\n"
                " - Eligibility: income above ₹8L per annum, credit score >750.\n"
                "4. Personal Loan\n", 17, 28),
            hit("rewards_and_benefits.txt",
                "6. Loyalty Tier System\n"
                " - Platinum: spending above ₹3 lakh/year, airport lounge."),
            hit("product_summary.txt",
                "2. Gold Credit Card\n - Annual fee: ₹1,000\n"
                " - Limit: ₹50,000–₹2,00,000\n"
                " - Reward rate: 2X on groceries, fuel, and dining.\n"
                "3. Platinum Credit Card\n", 10, 17, 2),
        ]
        result = self.ask("Explain the difference between Gold and Platinum cards.")
        self.assertIn("Gold Credit Card", result.text)
        self.assertIn("Platinum Credit Card", result.text)
        self.assertIn("₹1,000", result.text)
        self.assertIn("₹2,000", result.text)
        self.assertIn("income above ₹8L", result.text)
        self.assertNotIn("spending above ₹3 lakh", result.text)
        self.assertIn("[S3]", result.text)
        self.assertIn("[S1]", result.text)
        self.assertNotIn("[S2]", result.text)
        self.assertEqual(
            {c["source"] for c in result.citations}, {"product_summary.txt"}
        )
        self.assertEqual(len(result.citations), 2)
        self.generate.assert_not_called()

    def test_comparison_missing_gold_fails_closed(self):
        self.hits = [
            hit("product_summary.txt",
                "3. Platinum Credit Card\n - Annual fee: ₹2,000", 17, 20),
        ]
        result = self.ask("Gold vs Platinum card comparison?")
        self.assertIn("cannot verify a complete", result.text)
        self.assertEqual(result.citations, ())
        self.assertTrue(result.needs_review)
        self.generate.assert_not_called()

    def test_other_question_still_uses_model(self):
        self.hits = [hit("faq_accounts.md", "A support representative can help.")]
        self.ask("What happens if I miss a payment?")
        self.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
