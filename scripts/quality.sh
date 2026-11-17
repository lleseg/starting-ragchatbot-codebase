#!/bin/bash
# Run all code quality checks without modifying files. Exits non-zero on failure.
set -e
cd "$(dirname "$0")/.."

echo "Checking formatting (black)..."
uv run black --check --diff backend

echo "Linting (ruff)..."
uv run ruff check backend

echo "All quality checks passed."
