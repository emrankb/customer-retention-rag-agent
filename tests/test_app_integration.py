"""Exercise the complete local answer path with synthetic inputs."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from customer_retention.assistant import answer_question
from customer_retention.customer_db import CUSTOMER_COLUMNS, CustomerDB
from customer_retention.interface import create_app, make_handler
from customer_retention.retrieval import build_index, search_policies
from customer_retention.source_checks import REFERENCE_FILES

try:
    import faiss  # noqa: F401
    import gradio  # noqa: F401
    from langchain_core.embeddings import Embeddings
except ImportError:
    PACKAGES_AVAILABLE = False
    Embeddings = object
else:
    PACKAGES_AVAILABLE = True


class KeywordEmbeddings(Embeddings):
    """Local keyword vectors avoid downloads and API requests."""

    def _vector(self, text):
        text = text.lower()
        groups = (
            ("fee", "waiver", "annual"),
            ("complaint", "fraud", "report"),
            ("reward", "travel", "points"),
            ("account", "balance", "payment"),
        )
        values = [sum(text.count(word) for word in group) for group in groups]
        size = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / size for value in values]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


@unittest.skipUnless(PACKAGES_AVAILABLE, "Install both requirements files for app integration")
class AppIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.docs = root / "reference"
        self.docs.mkdir()
        self.index = root / "index" / "policies"
        texts = {
            "complaints_handling.txt": "Fraud complaint reporting requires contacting support.\n",
            "faq_accounts.md": "Account balance and payment information is available on statements.\n",
            "fee_waiver_policy.txt": "Annual fee waiver requests require staff approval.\n",
            "product_summary.txt": "The bank offers several cards and account products.\n",
            "retention_policy.txt": "Retention offers require individual review.\n",
            "rewards_and_benefits.txt": "Travel rewards and bonus points depend on card type.\n",
        }
        self.assertEqual(set(texts), set(REFERENCE_FILES))
        for name, body in texts.items():
            (self.docs / name).write_text(body, encoding="utf-8")

        csv_path = root / "customers.csv"
        fields = {
            "CustomerId": "42", "Surname": "Sample",
            "CreditScore": "700", "Geography": "France",
            "Gender": "Female", "Age": "40", "Tenure": "3",
            "Balance": "100.50", "NumOfProducts": "2", "HasCrCard": "1",
            "IsActiveMember": "1", "EstimatedSalary": "2000.00",
            "Exited": "0",
        }
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CUSTOMER_COLUMNS)
            writer.writeheader()
            writer.writerow(fields)
        self.db = CustomerDB(csv_path)
        self.embeddings = KeywordEmbeddings()
        build_index(
            self.docs, self.index, embeddings=self.embeddings,
            model_name="ci-keyword-vectors",
        )
        self.model = Mock(return_value="Waiver requests need staff approval [S1].")

        def search(question, docs_dir, index_dir, *, k):
            return search_policies(
                question, docs_dir, index_dir, k=k,
                embeddings=self.embeddings, model_name="ci-keyword-vectors",
            )

        def run_answer(question, customer_id, **kwargs):
            return answer_question(question, customer_id, search=search, **kwargs)

        self.handler = make_handler(
            customer_db=self.db, docs_dir=self.docs, index_dir=self.index,
            generate=self.model, answer=run_answer,
        )

    def test_customer_lookup_to_faiss_to_answer_and_ui(self):
        question = "How do I request an annual fee waiver?"
        result = self.handler("42", question)
        self.assertIn("Waiver requests need staff approval [S1].", result)
        self.assertIn("fee_waiver_policy.txt (lines 1-1)", result)
        self.assertIn("authorized bank review", result)
        self.model.assert_called_once()
        prompt = self.model.call_args.args[0]
        self.assertIn("Annual fee waiver requests require staff approval.", prompt)
        self.assertNotIn("Sample", prompt)
        self.assertNotIn("2000.00", prompt)

    def test_unknown_customer_does_not_call_model(self):
        result = self.handler("999", "How do I request an annual fee waiver?")
        self.assertIn("No customer record was found", result)
        self.model.assert_not_called()

    def test_tampered_index_does_not_call_model(self):
        (self.index / "chunks.json").write_text("[]", encoding="utf-8")
        result = self.handler("42", "How do I request an annual fee waiver?")
        self.assertIn("could not be completed", result)
        self.assertNotIn("[]", result)
        self.model.assert_not_called()

    def test_gradio_object_builds_with_real_customer_db(self):
        app = create_app(
            customer_db=self.db, docs_dir=self.docs,
            index_dir=self.index, generate=self.model,
        )
        self.assertIsNotNone(app)
        app.close()
        self.model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
