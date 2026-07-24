#!/usr/bin/env bash
# Publisher: Carlos Fundora · GitHub: @carlosfundora · MIT
# CI step: Run the python test suite
set -euo pipefail
cd "$(dirname "$0")/.."

# Detect python binary
PYTHON_BIN="${VIRTUAL_ENV:-.venv}/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "[ci] Running pytest integration tests..."
"$PYTHON_BIN" -m pytest tests/
