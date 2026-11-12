"""Check the customer tables and the six policy documents before indexing."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from .customer_db import load_customers

REFERENCE_FILES = (
    "complaints_handling.txt",
    "faq_accounts.md",
    "fee_waiver_policy.txt",
    "product_summary.txt",
    "retention_policy.txt",
    "rewards_and_benefits.txt",
)


def inspect_documents(directory: str | Path) -> list[dict[str, object]]:
    root = Path(directory)
    results = []
    for name in REFERENCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"Missing reference document: {name}.")
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ValueError(f"Reference document is not valid UTF-8: {name}.") from exc
        if not text.strip() or "\x00" in text:
            raise ValueError(f"Reference document is blank or contains NUL bytes: {name}.")
        results.append({
            "file": name,
            "characters": len(text),
            "sha256": sha256(raw).hexdigest(),
        })
    return results


def inspect_sources(data_dir: str | Path, docs_dir: str | Path) -> dict:
    data_path = Path(data_dir)
    base = load_customers(data_path / "Churn_Modeling.csv")
    sentiment_path = data_path / "Churn_With_Sentiment.csv"
    enriched = load_customers(sentiment_path) if sentiment_path.is_file() else None
    if enriched is not None:
        if set(base) != set(enriched):
            raise ValueError("Customer IDs differ between the original and sentiment tables.")
        for identifier, customer in base.items():
            if customer != replace(enriched[identifier], sentiment_label=None):
                raise ValueError("Non-sentiment fields differ between the customer tables.")

    summary = {
        "customer_records": len(base),
        "historical_exited": sum(customer.exited for customer in base.values()),
        "sentiment_available": enriched is not None,
        "documents": inspect_documents(docs_dir),
    }
    if enriched is not None:
        counts = Counter(customer.sentiment_label for customer in enriched.values())
        summary["sentiment_counts"] = dict(sorted(counts.items()))
    return summary
