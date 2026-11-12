"""Start the local Gradio chat interface.

Install requirements-retrieval.txt and requirements-app.txt first.
No model API requests are made until a user submits a question.
"""
from __future__ import annotations

import os
from pathlib import Path

from customer_retention.customer_db import CustomerDB
from customer_retention.interface import create_app
from customer_retention.model_adapter import create_generator
from customer_retention.preflight import check_startup
from customer_retention.retrieval import make_policy_search

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_path = ROOT / "data/raw/Churn_With_Sentiment.csv"
    if not data_path.is_file():
        data_path = ROOT / "data/raw/Churn_Modeling.csv"
    if not data_path.is_file():
        raise FileNotFoundError("Put a supplied customer CSV in data/raw/.")

    docs_dir = ROOT / "docs/reference"
    index_dir = ROOT / "index/policies"
    if not index_dir.is_dir():
        raise FileNotFoundError("Build the policy index before starting the application.")

    check_startup(ROOT / "data/raw", docs_dir, index_dir)
    customer_db = CustomerDB(data_path)
    search = make_policy_search(docs_dir, index_dir)
    generator = create_generator()
    app = create_app(
        customer_db=customer_db, docs_dir=docs_dir,
        index_dir=index_dir, generate=generator, search=search,
    )
    # Never expose the development server via a public share link.
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("RETENTION_PORT", "7860")),
        share=False,
        show_error=False,
    )


if __name__ == "__main__":
    main()
