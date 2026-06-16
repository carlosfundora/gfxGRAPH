#!/usr/bin/env bash
# Publisher: Carlos Fundora · GitHub: @carlosfundora · MIT
# CI step: Run the CTest native test suite
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[ci] Running CTest native test suite..."
ctest --test-dir build --output-on-failure
