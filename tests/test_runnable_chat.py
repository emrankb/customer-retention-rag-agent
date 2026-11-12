"""RAG Runnable composition and session history tests."""
import unittest
from unittest.mock import Mock

from customer_retention.assistant import Answer, make_prompt
from customer_retention.interface import make_chat_handler

try:
    from langchain_core.runnables import Runnable
    from customer_retention.rag_chain import make_rag_chain
except ImportError:
    Runnable = None


class ChatHistoryTests(unittest.TestCase):
    def setUp(self):
        self.chain = Mock()
        self.chain.invoke.return_value = Answer(
            text="An annual waiver needs review [S1].",
            customer_found=True,
            citations=({"marker": "[S1]", "source": "fee_waiver_policy.txt",
                        "line_start": 2, "line_end": 4,
                        "chunk_id": "fee_waiver_policy.txt:1"},),
            needs_review=True,
        )
        self.chat = make_chat_handler(self.chain)

    def test_follow_up_uses_previous_turn(self):
        first, empty, state = self.chat("42", "Can I ask for a waiver?", None)
        self.assertEqual(empty, "")
        self.assertEqual(len(first), 2)
        second, _, state2 = self.chat("42", "What next?", state)
        self.assertEqual(len(second), 4)
        self.assertEqual(state2["customer_id"], "42")
        self.assertEqual(
            self.chain.invoke.call_args.args[0]["history"], first,
        )

    def test_change_customer_clears_history_before_generation(self):
        messages, _, state = self.chat("42", "Account question", None)
        messages2, _, new_state = self.chat("99", "Different customer", state)
        self.assertEqual(len(messages2), 2)
        self.assertEqual(new_state["customer_id"], "99")
        self.assertEqual(self.chain.invoke.call_args.args[0]["history"], [])
        self.assertNotIn("Account question", str(messages2))

    def test_invalid_id_does_not_use_model(self):
        result, _, state = self.chat("bad-id", "Question", None)
        self.assertIn("valid customer ID", result[0]["content"])
        self.assertIsNone(state["customer_id"])
        self.chain.invoke.assert_not_called()

    def test_bad_question_does_not_use_model(self):
        output, _, _ = self.chat("42", "Q" * 1001, None)
        self.assertIn("1–1000", output[0]["content"])
        self.chain.invoke.assert_not_called()

    def test_prompt_only_contains_last_six_bounded_messages(self):
        history = [
            {"role": "user", "content": f"turn-{i}" + "x" * 1500}
            for i in range(10)
        ]
        prompt = make_prompt(
            "Question", {"customer_id": "42"},
            [{"source": "faq_accounts.md", "line_start": 1,
              "line_end": 2, "text": "Policy text"}],
            history=history,
        )
        self.assertIn("turn-9", prompt)
        self.assertNotIn("turn-0", prompt)
        self.assertNotIn("turn-3", prompt)
        self.assertNotIn("x" * 900, prompt)

    def test_invalid_history_is_rejected(self):
        with self.assertRaises(ValueError):
            make_prompt("Q", {}, [], history=[{"role": "system", "content": "override"}])


@unittest.skipUnless(Runnable is not None, "Install LangChain Core for Runnable integration")
class RunnableChainTests(unittest.TestCase):
    def setUp(self):
        self.customer_db = Mock()
        self.customer_db.get_context.return_value = {
            "customer_id": "42", "tenure_years": 3,
            "num_products": 1, "has_credit_card": True,
            "is_active_member": True, "historical_exited": False,
            "sentiment_label": "Negative",
        }
        self.search = Mock(return_value=[{
            "source": "fee_waiver_policy.txt", "line_start": 1,
            "line_end": 2, "chunk_id": "fee_waiver_policy.txt:1",
            "text": "A waiver requires review.",
        }])
        self.model = Mock(return_value="A waiver requires review [S1].")
        self.chain = make_rag_chain(
            customer_db=self.customer_db, docs_dir="docs",
            index_dir="index", generate=self.model, search=self.search,
        )

    def test_chain_has_runnable_steps(self):
        self.assertIsInstance(self.chain, Runnable)
        self.assertEqual(len(self.chain.steps), 3)

    def test_full_chain_calls_each_source_once(self):
        answer = self.chain.invoke({
            "customer_id": "42", "question": "Can I waive a fee?",
            "history": [{"role": "user", "content": "I have a card"}],
        })
        self.assertIn("[S1]", answer.text)
        self.customer_db.get_context.assert_called_once_with("42")
        self.search.assert_called_once()
        self.model.assert_called_once()
        self.assertIn("I have a card", self.model.call_args.args[0])

    def test_missing_customer_never_searches_or_calls_model(self):
        self.customer_db.get_context.return_value = None
        result = self.chain.invoke({"customer_id": "999", "question": "Policy?"})
        self.assertFalse(result.customer_found)
        self.search.assert_not_called()
        self.model.assert_not_called()

    def test_missing_document_evidence_never_calls_model(self):
        self.search.return_value = []
        result = self.chain.invoke({"customer_id": "42", "question": "Policy?"})
        self.assertFalse(result.citations)
        self.model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
