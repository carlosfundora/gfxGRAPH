#!/usr/bin/env bash
# Publisher: Carlos Fundora · GitHub: @carlosfundora · MIT
# CI step: Verify version synchronization between __init__.py and pyproject.toml
set -euo pipefail
cd "$(dirname "$0")/.."

# Detect python binary
PYTHON_BIN="${VIRTUAL_ENV:-.venv}/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "[ci] Checking version synchronization..."
"$PYTHON_BIN" - <<'PY'
import tomllib, re, pathlib
pj = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"]
init = pathlib.Path("python/gfxgraph/__init__.py").read_text()
got = re.search(r'__version__\s*=\s*"([^"]+)"', init).group(1)
assert got == pj, f"version drift: __version__={got!r} != pyproject={pj!r}"
print(f"[ci] Version-sync check passed: {pj}")
PY
