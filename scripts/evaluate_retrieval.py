"""Score a saved policy index against reviewed question-to-source judgments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from customer_retention.retrieval import MODEL_NAME
from customer_retention.retrieval_evaluation import (
    evaluate_retrieval, load_judgments,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True,
                        help="JSONL file of reviewed questions and expected source files")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs/reference")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "index/policies")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/retrieval_evaluation.json")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    try:
        cases = load_judgments(args.cases)
        result = evaluate_retrieval(
            cases, args.docs_dir, args.index_dir,
            k=args.k, model_name=args.model_name,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Retrieval evaluation failed: {exc}\n")
    print(f"Evaluated {result['total']} reviewed questions at k={result['k']}")
    print(f"Source hit rate: {result['hit_rate']:.3f}")
    print(f"Mean source recall: {result['mean_source_recall']:.3f}")
    print(f"Mean reciprocal rank: {result['mean_reciprocal_rank']:.3f}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
