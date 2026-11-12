"""Run offline answer contracts and save a report under the ignored reports folder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from customer_retention.evaluation import evaluate_cases, load_cases

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline answer checks")
    parser.add_argument("--cases", type=Path, default=ROOT / "tests/fixtures/answer_cases.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/answer_contracts.json")
    args = parser.parse_args()
    result = evaluate_cases(load_cases(args.cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Offline answer checks: {result['passed']}/{result['total']} passed")
    print(f"Report: {args.output}")
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
