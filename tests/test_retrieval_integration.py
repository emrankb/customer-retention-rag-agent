"""Exercise LangChain's loaders and the native FAISS index with offline vectors."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from customer_retention.retrieval import (
    build_index, load_documents, open_index, search_policies, split_documents,
)
from customer_retention.source_checks import REFERENCE_FILES

try:
    import faiss  # noqa: F401
    from langchain_community.document_loaders import TextLoader  # noqa: F401
    from langchain_community.vectorstores import FAISS  # noqa: F401
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
except ImportError:
    Embeddings = object
    PACKAGES_AVAILABLE = False
else:
    PACKAGES_AVAILABLE = True


class LocalEmbeddings(Embeddings):
    """Fixed keyword vectors; only the index integration is under test."""

    categories = (
        ("annual", "fee", "waiver", "late", "billing"),
        ("gold", "platinum", "card", "reward", "travel"),
        ("complaint", "fraud", "report", "escalation", "support"),
        ("interest", "account", "emi", "payment", "balance"),
    )

    def vector(self, text):
        lowered = text.lower()
        values = [sum(lowered.count(term) for term in category) for category in self.categories]
        length = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / length for value in values]

    def embed_documents(self, texts):
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        return self.vector(text)


@unittest.skipUnless(PACKAGES_AVAILABLE, "Install requirements-retrieval.txt for LangChain/FAISS integration")
class RetrievalIntegrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.docs = root / "reference"
        self.docs.mkdir()
        self.index = root / "index" / "policies"
        examples = {
            "complaints_handling.txt": "Complaint reporting and escalation procedure. Contact support for fraud complaints.\n",
            "faq_accounts.md": "Account balances, billing statements, missed payments, and card limits.\n",
            "fee_waiver_policy.txt": "Annual fee waiver eligibility review. Fees may be waived only after approval.\n",
            "product_summary.txt": "Gold and Platinum cards have different annual fees and benefits.\n",
            "retention_policy.txt": "Retention support may include interest reductions or rewards, subject to approval.\n",
            "rewards_and_benefits.txt": "Gold and Platinum travel rewards and bonus loyalty points.\n",
        }
        self.assertEqual(set(examples), set(REFERENCE_FILES))
        for name, text in examples.items():
            (self.docs / name).write_text(text, encoding="utf-8")
        self.embeddings = LocalEmbeddings()

    def test_actual_loaders_and_splitter_preserve_source_lines(self):
        docs = load_documents(self.docs)
        self.assertEqual(len(docs), 6)
        pieces = split_documents(docs, chunk_size=80, overlap=12)
        self.assertGreaterEqual(len(pieces), 6)
        for doc in pieces:
            self.assertIn(doc.metadata["source"], REFERENCE_FILES)
            self.assertGreaterEqual(doc.metadata["line_start"], 1)
            self.assertLessEqual(doc.metadata["line_start"], doc.metadata["line_end"])
            source_lines = (self.docs / doc.metadata["source"]).read_text().splitlines(keepends=True)
            self.assertIn(doc.page_content, "".join(source_lines[doc.metadata["line_start"] - 1:doc.metadata["line_end"]]))

    def test_real_faiss_build_reload_and_search(self):
        manifest = build_index(self.docs, self.index, embeddings=self.embeddings,
                               model_name="test-vocabulary", chunk_size=80, overlap=12)
        self.assertEqual(len(manifest["sources"]), 6)
        self.assertGreaterEqual(manifest["chunk_count"], 6)
        store = open_index(self.docs, self.index, embeddings=self.embeddings,
                           model_name="test-vocabulary")
        self.assertEqual(store.index.ntotal, manifest["chunk_count"])
        results = search_policies("annual fee waiver", self.docs, self.index,
                                  embeddings=self.embeddings, model_name="test-vocabulary", k=4)
        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "fee_waiver_policy.txt")
        self.assertTrue(all(row["line_start"] >= 1 for row in results))
        self.assertFalse((self.index / "index.pkl").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
