#!/usr/bin/env bash
# Publisher: Carlos Fundora · GitHub: @carlosfundora · MIT
# CI step: Validate sdist and wheel compilation using uv build
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[ci] Validating package builds..."
if command -v uv >/dev/null 2>&1; then
    uv build >/dev/null
    echo "[ci] Build ok: $(ls -t dist/*.whl | head -1)"
else
    echo "[ci] Skip: uv not installed on PATH"
fi
