#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

coverage erase || true
coverage run -m pytest "$@"
coverage html
coverage xml

echo
echo "Coverage artifacts generated:"
echo " - HTML: $REPO_ROOT/htmlcov/index.html"
echo " - XML : $REPO_ROOT/coverage.xml"
