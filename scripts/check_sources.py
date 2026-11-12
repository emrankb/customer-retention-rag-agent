#!/usr/bin/env python3
"""Check the supplied customer tables and reference documents."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from customer_retention.source_checks import inspect_sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs/reference")
    args = parser.parse_args()
    try:
        report = inspect_sources(args.data_dir, args.docs_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Source check failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
