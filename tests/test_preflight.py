"""Preflight checks use temporary data and a locally built FAISS index."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile
import unittest

from customer_retention.customer_db import CUSTOMER_COLUMNS
from customer_retention.preflight import check_startup
from customer_retention.source_checks import REFERENCE_FILES

try:
    import faiss  # noqa: F401
    from langchain_core.embeddings import Embeddings
    from customer_retention.retrieval import build_index
except ImportError:
    Embeddings = object
    AVAILABLE = False
else:
    AVAILABLE = True


class LocalEmbeddings(Embeddings):
    def _vector(self, text):
        words = ("fee", "account", "complaint", "reward")
        values = [text.lower().count(word) for word in words]
        length = math.sqrt(sum(value ** 2 for value in values)) or 1.0
        return [value / length for value in values]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


@unittest.skipUnless(AVAILABLE, "Install retrieval packages for FAISS checks")
class PreflightTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.data = root / "data"
        self.data.mkdir()
        self.docs = root / "docs"
        self.docs.mkdir()
        self.index = root / "index"
        self.base = self.data / "Churn_Modeling.csv"
        self.sentiment = self.data / "Churn_With_Sentiment.csv"
        self.record = {
            "CustomerId": "42", "Surname": "Example", "CreditScore": "700",
            "Geography": "France", "Gender": "Female", "Age": "40",
            "Tenure": "3", "Balance": "100.50", "NumOfProducts": "2",
            "HasCrCard": "1", "IsActiveMember": "1",
            "EstimatedSalary": "2000.00", "Exited": "0",
        }
        self.write_csv(self.base, [self.record])
        self.write_csv(self.sentiment, [{**self.record, "Sentiment_Label": "Neutral"}],
                       sentiment=True)
        for name in REFERENCE_FILES:
            (self.docs / name).write_text(
                f"{name}: account fees, rewards, and support.\n", encoding="utf-8",
            )
        self.embeddings = LocalEmbeddings()
        build_index(
            self.docs, self.index, embeddings=self.embeddings,
            model_name="preflight-test-encoder",
        )

    def write_csv(self, path, records, sentiment=False):
        columns = list(CUSTOMER_COLUMNS)
        if sentiment:
            columns.append("Sentiment_Label")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)

    def check(self):
        return check_startup(
            self.data, self.docs, self.index,
            embeddings=self.embeddings, model_name="preflight-test-encoder",
        )

    def test_complete_sources_and_index(self):
        result = self.check()
        self.assertEqual(result["customer_records"], 1)
        self.assertEqual(result["policy_documents"], 6)
        self.assertTrue(result["sentiment_available"])
        self.assertGreaterEqual(result["indexed_passages"], 6)
        self.assertNotIn("Example", repr(result))
        self.assertNotIn("2000.00", repr(result))

    def test_optional_sentiment_file(self):
        self.sentiment.unlink()
        self.assertFalse(self.check()["sentiment_available"])

    def test_mismatched_customer_tables_rejected(self):
        self.write_csv(
            self.sentiment,
            [{**self.record, "CreditScore": "701", "Sentiment_Label": "Neutral"}],
            sentiment=True,
        )
        with self.assertRaisesRegex(ValueError, "Non-sentiment fields differ"):
            self.check()

    def test_missing_base_table_rejected(self):
        self.base.unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_modified_document_rejected(self):
        (self.docs / REFERENCE_FILES[0]).write_text("changed policy\n")
        with self.assertRaisesRegex(ValueError, "Policy documents changed"):
            self.check()

    def test_damaged_index_rejected(self):
        (self.index / "chunks.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "index file changed"):
            self.check()

    def test_embedding_model_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "Embedding model differs"):
            check_startup(
                self.data, self.docs, self.index,
                embeddings=self.embeddings, model_name="wrong-model",
            )


if __name__ == "__main__":
    unittest.main()
