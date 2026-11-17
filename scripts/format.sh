#!/bin/bash
# Auto-format the codebase: fix lint issues (incl. import order), then apply black.
set -e
cd "$(dirname "$0")/.."

echo "Running ruff --fix..."
uv run ruff check --fix backend

echo "Running black..."
uv run black backend
