"""Offline tests for policy source handling and saved index boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from customer_retention.retrieval import (
    build_index, load_documents, open_index, search_policies, split_documents,
)
from customer_retention.source_checks import REFERENCE_FILES


class MockDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class MockTextLoader:
    def __init__(self, path, encoding):
        self.path = Path(path)
        self.encoding = encoding

    def load(self):
        return [MockDocument(self.path.read_text(encoding=self.encoding), {"source": self.path.name})]


class MockSplitter:
    def __init__(self, chunk_size, chunk_overlap, add_start_index):
        self.chunk_size, self.overlap = chunk_size, chunk_overlap
        if not add_start_index:
            raise AssertionError("Missing source offsets")

    def split_documents(self, documents):
        result = []
        for doc in documents:
            text = doc.page_content
            for start in range(0, len(text), self.chunk_size - self.overlap):
                chunk = text[start:start + self.chunk_size]
                result.append(MockDocument(chunk, {**doc.metadata, "start_index": start}))
                if start + self.chunk_size >= len(text):
                    break
        return result


class MockVectorIndex:
    def __init__(self, documents, embeddings, ids):
        self.documents = documents
        self.embeddings = embeddings
        self.ids = ids
        self.ntotal = len(documents)
        self.d = 3


class MockStore:
    @classmethod
    def from_documents(cls, documents, embedding, ids):
        store = cls.__new__(cls)
        store.index = MockVectorIndex(documents, embedding, ids)
        return store

    def __init__(self, embedding_function, index, docstore, index_to_docstore_id):
        self.index = index
        self.docstore = docstore
        self.map = index_to_docstore_id

    def similarity_search_with_score(self, query, k):
        words = set(query.lower().split())
        rows = []
        for i in range(self.index.ntotal):
            doc = self.docstore.data[self.map[i]]
            distance = 20.0 - len(words.intersection(doc.page_content.lower().split()))
            rows.append((doc, distance))
        return sorted(rows, key=lambda x: x[1])[:k]


class MockInMemoryDocstore:
    def __init__(self, data):
        self.data = data


class MockFaiss(types.ModuleType):
    @staticmethod
    def write_index(index, path):
        Path(path).write_text(json.dumps({"count": index.ntotal, "dimension": index.d}), encoding="utf-8")

    @staticmethod
    def read_index(path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return types.SimpleNamespace(ntotal=data["count"], d=data["dimension"])


@contextmanager
def mock_retrieval_packages():
    modules = {}
    for name in (
        "langchain_community", "langchain_community.document_loaders",
        "langchain_community.vectorstores", "langchain_community.docstore",
        "langchain_community.docstore.in_memory", "langchain_text_splitters",
        "langchain_core", "langchain_core.documents",
    ):
        modules[name] = types.ModuleType(name)
    modules["langchain_community.document_loaders"].TextLoader = MockTextLoader
    modules["langchain_community.vectorstores"].FAISS = MockStore
    modules["langchain_community.docstore.in_memory"].InMemoryDocstore = MockInMemoryDocstore
    modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MockSplitter
    modules["langchain_core.documents"].Document = MockDocument
    modules["faiss"] = MockFaiss("faiss")
    with patch.dict(sys.modules, modules):
        yield


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        self.index = self.root / "index" / "policies"
        for name in REFERENCE_FILES:
            (self.docs / name).write_text(
                f"{name}\nAnnual fee waiver terms apply after review.\n"
                "Line three includes more example wording.\n" * 7,
                encoding="utf-8",
            )
        self.packages = mock_retrieval_packages()
        self.packages.__enter__()
        self.addCleanup(self.packages.__exit__, None, None, None)

    def make_index(self):
        return build_index(self.docs, self.index, embeddings=object(), model_name="offline-test", chunk_size=100, overlap=20)

    def test_loads_only_the_six_known_documents(self):
        (self.docs / "extra.txt").write_text("Ignore me", encoding="utf-8")
        docs = load_documents(self.docs)
        self.assertEqual([x.metadata["source"] for x in docs], list(REFERENCE_FILES))
        self.assertEqual(len(docs), 6)
        self.assertEqual(len(docs[0].metadata["source_sha256"]), 64)

    def test_chunk_offsets_keep_original_line_numbers(self):
        chunks = split_documents(load_documents(self.docs), chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 6)
        for chunk in chunks:
            meta = chunk.metadata
            source = (self.docs / meta["source"]).read_text(encoding="utf-8")
            lines = source.splitlines(keepends=True)
            cited = "".join(lines[meta["line_start"] - 1:meta["line_end"]])
            self.assertIn(chunk.page_content, cited)
            self.assertEqual(meta["chunk_id"].split(":")[0], meta["source"])

    def test_bad_chunk_settings_rejected(self):
        docs = load_documents(self.docs)
        for size, overlap in ((50, 1), (100, 100), (100, -1), (4001, 0)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                split_documents(docs, chunk_size=size, overlap=overlap)

    def test_saved_index_and_search_return_citations(self):
        manifest = self.make_index()
        self.assertEqual(len(manifest["sources"]), 6)
        self.assertEqual(set(p.name for p in self.index.iterdir()), {"vectors.faiss", "chunks.json", "manifest.json"})
        self.assertEqual(manifest["model_name"], "offline-test")
        hits = search_policies("annual fee waiver", self.docs, self.index, k=3, embeddings=object(), model_name="offline-test")
        self.assertEqual(len(hits), 3)
        for hit in hits:
            self.assertIn(hit["source"], REFERENCE_FILES)
            self.assertGreaterEqual(hit["line_start"], 1)
            self.assertGreaterEqual(hit["line_end"], hit["line_start"])
            self.assertIn("chunk_id", hit)
            self.assertIn("distance", hit)

    def test_does_not_overwrite_existing_index(self):
        self.make_index()
        existing_hash = sha256((self.index / "vectors.faiss").read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            self.make_index()
        self.assertEqual(sha256((self.index / "vectors.faiss").read_bytes()).hexdigest(), existing_hash)

    def test_changed_policy_requires_rebuild(self):
        self.make_index()
        (self.docs / REFERENCE_FILES[0]).write_text("Modified content", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "documents changed"):
            open_index(self.docs, self.index, embeddings=object(), model_name="offline-test")

    def test_tampered_json_is_rejected(self):
        self.make_index()
        with (self.index / "chunks.json").open("a", encoding="utf-8") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "index file changed"):
            open_index(self.docs, self.index, embeddings=object(), model_name="offline-test")

    def test_rehashed_wrong_citation_is_rejected(self):
        self.make_index()
        chunks_path = self.index / "chunks.json"
        rows = json.loads(chunks_path.read_text(encoding="utf-8"))
        rows[0]["metadata"]["line_start"] = 999
        rows[0]["metadata"]["line_end"] = 999
        chunks_path.write_text(json.dumps(rows), encoding="utf-8")
        manifest_path = self.index / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["chunks_sha256"] = sha256(chunks_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "citation does not match"):
            open_index(self.docs, self.index, embeddings=object(), model_name="offline-test")

    def test_tampered_vectors_are_rejected(self):
        self.make_index()
        (self.index / "vectors.faiss").write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "index file changed"):
            open_index(self.docs, self.index, embeddings=object(), model_name="offline-test")

    def test_wrong_model_is_rejected(self):
        self.make_index()
        with self.assertRaisesRegex(ValueError, "Embedding model differs"):
            open_index(self.docs, self.index, embeddings=object())

    def test_unexpected_vector_count_is_rejected(self):
        self.make_index()
        vectors = self.index / "vectors.faiss"
        vectors.write_text(json.dumps({"count": 500, "dimension": 3}), encoding="utf-8")
        manifest_path = self.index / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["vectors_sha256"] = sha256(vectors.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "count or dimension"):
            open_index(self.docs, self.index, embeddings=object(), model_name="offline-test")

    def test_invalid_search_inputs_are_rejected(self):
        for query in ("", "  ", "a\x00b", "x" * 1001, 10):
            with self.subTest(query=repr(query)[:16]), self.assertRaises(ValueError):
                search_policies(query, self.docs, self.index)
        for k in (0, -1, 11, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                search_policies("fee waiver", self.docs, self.index, k=k)

    def test_missing_document_rejected_before_embedding(self):
        (self.docs / REFERENCE_FILES[2]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            self.make_index()


if __name__ == "__main__":
    unittest.main(verbosity=2)
