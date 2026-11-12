"""Load policy documents, build a local FAISS index, and return cited passages."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile

from .source_checks import REFERENCE_FILES, inspect_documents

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 480
CHUNK_OVERLAP = 80
INDEX_VERSION = 1


def _sha256(contents: bytes) -> str:
    return sha256(contents).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def make_embeddings(model_name: str = MODEL_NAME):
    """Load the local Hugging Face encoder when retrieval is requested."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError("Install requirements-retrieval.txt before building or searching.") from exc
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(docs_dir: str | Path):
    """Read the six approved sources through LangChain's TextLoader."""
    from langchain_community.document_loaders import TextLoader

    root = Path(docs_dir)
    checks = {item["file"]: item["sha256"] for item in inspect_documents(root)}
    documents = []
    for name in REFERENCE_FILES:
        loaded = TextLoader(str(root / name), encoding="utf-8-sig").load()
        if len(loaded) != 1 or not loaded[0].page_content.strip():
            raise ValueError(f"Unexpected document content: {name}.")
        doc = loaded[0]
        doc.metadata = {"source": name, "source_sha256": checks[name]}
        documents.append(doc)
    return documents


def split_documents(documents, *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split documents while retaining their original source line numbers."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not 80 <= chunk_size <= 4000 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid document chunk size or overlap.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    pieces = splitter.split_documents(documents)
    if not pieces:
        raise ValueError("Document splitting produced no chunks.")

    original = {doc.metadata["source"]: doc.page_content for doc in documents}
    counts = {name: 0 for name in original}
    for doc in pieces:
        name = doc.metadata["source"]
        source = original[name]
        start = doc.metadata.get("start_index")
        if (not isinstance(start, int) or start < 0
                or source[start:start + len(doc.page_content)] != doc.page_content):
            raise ValueError(f"Chunk offset does not match {name}.")
        counts[name] += 1
        doc.metadata = {
            "source": name,
            "source_sha256": doc.metadata["source_sha256"],
            "chunk_id": f"{name}:{counts[name]}",
            "line_start": source.count("\n", 0, start) + 1,
            "line_end": source.count("\n", 0, start + len(doc.page_content) - 1) + 1,
        }
    return pieces


def _verify_index_files(folder: Path, manifest: dict, source_texts: dict[str, str]) -> list[dict]:
    if manifest.get("version") != INDEX_VERSION:
        raise ValueError("Unsupported policy index version.")
    for file, field in (("vectors.faiss", "vectors_sha256"), ("chunks.json", "chunks_sha256")):
        path = folder / file
        if _sha256(path.read_bytes()) != manifest.get(field):
            raise ValueError(f"Saved policy index file changed: {file}.")
    chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or len(chunks) != manifest.get("chunk_count") or not chunks:
        raise ValueError("Saved policy index has an unexpected chunk count.")
    seen = set()
    for row in chunks:
        if not isinstance(row, dict) or set(row) != {"text", "metadata"}:
            raise ValueError("Malformed saved policy chunk.")
        meta = row["metadata"]
        if (not isinstance(row["text"], str) or not row["text"].strip()
                or not isinstance(meta, dict)
                or set(meta) != {"source", "source_sha256", "chunk_id", "line_start", "line_end"}
                or meta["source"] not in REFERENCE_FILES
                or not isinstance(meta["chunk_id"], str)
                or meta["chunk_id"] in seen
                or not isinstance(meta["line_start"], int)
                or not isinstance(meta["line_end"], int)
                or not 1 <= meta["line_start"] <= meta["line_end"]):
            raise ValueError("Malformed saved policy metadata.")
        if meta["source_sha256"] != manifest["sources"][meta["source"]]:
            raise ValueError("Saved policy chunk has the wrong source checksum.")
        if not meta["chunk_id"].startswith(meta["source"] + ":"):
            raise ValueError("Saved policy chunk has the wrong source identifier.")
        source_lines = source_texts[meta["source"]].splitlines(keepends=True)
        if (meta["line_end"] > len(source_lines)
                or row["text"] not in "".join(source_lines[
                    meta["line_start"] - 1:meta["line_end"]
                ])):
            raise ValueError("Saved policy citation does not match the source lines.")
        seen.add(meta["chunk_id"])
    return chunks


def build_index(
    docs_dir: str | Path, index_dir: str | Path, *,
    embeddings=None, model_name: str = MODEL_NAME,
    chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
) -> dict:
    """Create a new local index without replacing an existing one."""
    import faiss
    from langchain_community.vectorstores import FAISS

    target = Path(index_dir)
    if target.exists():
        raise FileExistsError(f"Index directory already exists: {target}.")
    docs = load_documents(docs_dir)
    chunks = split_documents(docs, chunk_size=chunk_size, overlap=overlap)
    embeddings = embeddings if embeddings is not None else make_embeddings(model_name)
    ids = [doc.metadata["chunk_id"] for doc in chunks]
    store = FAISS.from_documents(chunks, embedding=embeddings, ids=ids)

    records = [{"text": doc.page_content, "metadata": doc.metadata} for doc in chunks]
    source_hashes = {doc.metadata["source"]: doc.metadata["source_sha256"] for doc in docs}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".policy-index-", dir=target.parent) as tmp:
        temp = Path(tmp)
        faiss.write_index(store.index, str(temp / "vectors.faiss"))
        (temp / "chunks.json").write_bytes(_json_bytes(records))
        manifest = {
            "version": INDEX_VERSION,
            "model_name": model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "chunk_count": len(records),
            "dimension": store.index.d,
            "sources": source_hashes,
            "vectors_sha256": _sha256((temp / "vectors.faiss").read_bytes()),
            "chunks_sha256": _sha256((temp / "chunks.json").read_bytes()),
        }
        (temp / "manifest.json").write_bytes(_json_bytes(manifest))
        temp.rename(target)
    return manifest


def open_index(
    docs_dir: str | Path, index_dir: str | Path, *,
    embeddings=None, model_name: str = MODEL_NAME,
):
    """Read only trusted locally generated vectors and JSON, never pickle."""
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    folder = Path(index_dir)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_name") != model_name:
        raise ValueError("Embedding model differs from the saved policy index.")
    expected = {row["file"]: row["sha256"] for row in inspect_documents(docs_dir)}
    if manifest.get("sources") != expected:
        raise ValueError("Policy documents changed; rebuild the index.")
    source_texts = {
        name: (Path(docs_dir) / name).read_text(encoding="utf-8-sig")
        for name in REFERENCE_FILES
    }
    chunks = _verify_index_files(folder, manifest, source_texts)
    index = faiss.read_index(str(folder / "vectors.faiss"))
    if index.ntotal != len(chunks) or index.d != manifest.get("dimension"):
        raise ValueError("FAISS vector count or dimension differs from the manifest.")
    documents = [Document(page_content=row["text"], metadata=row["metadata"]) for row in chunks]
    identifiers = [doc.metadata["chunk_id"] for doc in documents]
    embedding_function = embeddings if embeddings is not None else make_embeddings(model_name)
    return FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore(dict(zip(identifiers, documents))),
        index_to_docstore_id=dict(enumerate(identifiers)),
    )


def _search_loaded_store(question: str, store, *, k: int) -> list[dict]:
    if (not isinstance(question, str) or not question.strip()
            or len(question) > 1000 or "\x00" in question):
        raise ValueError("Search question must contain 1–1000 characters.")
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 10:
        raise ValueError("Number of results must be between 1 and 10.")
    hits = []
    for doc, distance in store.similarity_search_with_score(question.strip(), k=k):
        score = float(distance)
        if not math.isfinite(score):
            raise ValueError("FAISS returned a non-finite distance.")
        hits.append({
            "text": doc.page_content,
            "source": doc.metadata["source"],
            "line_start": doc.metadata["line_start"],
            "line_end": doc.metadata["line_end"],
            "chunk_id": doc.metadata["chunk_id"],
            "distance": score,
        })
    return hits


def make_policy_search(
    docs_dir: str | Path, index_dir: str | Path, *,
    embeddings=None, model_name: str = MODEL_NAME,
):
    """Verify and load the local index once for a running application."""
    store = open_index(
        docs_dir, index_dir, embeddings=embeddings, model_name=model_name,
    )

    def search(question: str, _docs_dir, _index_dir, *, k: int = 4) -> list[dict]:
        return _search_loaded_store(question, store, k=k)

    return search


def search_policies(
    question: str, docs_dir: str | Path, index_dir: str | Path, *,
    k: int = 4, embeddings=None, model_name: str = MODEL_NAME,
) -> list[dict]:
    """One-off policy search that reloads and verifies the saved index."""
    # Validate before loading a potentially large embedding model.
    if (not isinstance(question, str) or not question.strip()
            or len(question) > 1000 or "\x00" in question):
        raise ValueError("Search question must contain 1–1000 characters.")
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 10:
        raise ValueError("Number of results must be between 1 and 10.")
    store = open_index(docs_dir, index_dir, embeddings=embeddings, model_name=model_name)
    return _search_loaded_store(question, store, k=k)
