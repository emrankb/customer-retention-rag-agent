"""Run specified questions through the configured model for human review."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from customer_retention.customer_db import CustomerDB
from customer_retention.evaluation_capture import capture_responses, load_questions
from customer_retention.model_adapter import create_generator
from customer_retention.preflight import prepare_startup

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "evaluation/sample_questions.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/response_reviews_pending.jsonl")
    parser.add_argument("--evidence", type=Path,
                        default=ROOT / "reports/retrieved_evidence.jsonl")
    parser.add_argument("--allow-model-requests", action="store_true",
                        help="Confirm external LLM requests and possible charges")
    args = parser.parse_args()

    if not args.allow_model_requests:
        parser.exit(2, "No model calls made. Pass --allow-model-requests only "
                    "after authorizing data sharing and possible API charges.\n")
    if args.output.resolve() == args.evidence.resolve():
        parser.exit(2, "Review and evidence paths must be different.\n")
    if args.output.exists() or args.evidence.exists():
        parser.exit(2, "Output already exists; choose new paths.\n")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.exit(2, "OPENAI_API_KEY is required. No model calls made.\n")

    cases = load_questions(args.questions)
    data_dir = ROOT / "data/raw"
    csv_path = data_dir / "Churn_With_Sentiment.csv"
    if not csv_path.is_file():
        csv_path = data_dir / "Churn_Modeling.csv"
    customer_db = CustomerDB(csv_path)
    docs_dir = ROOT / "docs/reference"
    index_dir = ROOT / "index/policies"
    _summary, search = prepare_startup(data_dir, docs_dir, index_dir)
    model_name = os.environ.get("RETENTION_MODEL", "gpt-4o-mini")
    generate = create_generator(model_name=model_name)
    reviews, evidence = capture_responses(
        cases, customer_db=customer_db, docs_dir=docs_dir,
        index_dir=index_dir, generate=generate, search=search,
        model_name=model_name,
    )
    _write_jsonl(args.output, reviews)
    _write_jsonl(args.evidence, evidence)
    print(f"Captured {len(reviews)} actual responses. Human ratings are not filled.")
    print(f"Reviews: {args.output}")
    print(f"Evidence: {args.evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
