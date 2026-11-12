#!/usr/bin/env python3
"""Search the saved bank-policy index and show source references."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from customer_retention.retrieval import search_policies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs/reference")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "index/policies")
    args = parser.parse_args()
    try:
        hits = search_policies(args.query, args.docs_dir, args.index_dir, k=args.k)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Policy search failed: {exc}\n")
    print(json.dumps(hits, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
