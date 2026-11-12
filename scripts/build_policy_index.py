#!/usr/bin/env python3
"""Build the FAISS index for the six bank reference documents."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from customer_retention.retrieval import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs/reference")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "index/policies")
    args = parser.parse_args()
    try:
        manifest = build_index(args.docs_dir, args.index_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Index build failed: {exc}\n")
    print(json.dumps({"model": manifest["model_name"], "chunks": manifest["chunk_count"],
                      "documents": len(manifest["sources"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
