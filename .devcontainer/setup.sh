#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
.venv/bin/python -m unittest discover -s tests -q
