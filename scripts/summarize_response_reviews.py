"""Summarize completed model-response reviews without publishing answer text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from customer_retention.response_review import load_reviews, summarize_reviews

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/response_review_summary.json")
    args = parser.parse_args()
    try:
        summary = summarize_reviews(load_reviews(args.reviews))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Response review failed: {exc}\n")
    print(f"Reviewed responses: {summary['reviewed_responses']}")
    for name, score in summary["mean_scores"].items():
        print(f"{name}: {score:.2f}/2")
    print(f"Summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
