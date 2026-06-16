#!/usr/bin/env bash
# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora · MIT
# Local CI for gfxGRAPH — GitHub Actions is billing-locked, so THIS is the release/merge gate.
# Runs on the local box (which has gfx1030 + ROCm, unlike the dead ubuntu-latest workflows).
#   ./ci.sh           # full gate: tests + build + version-sync
#   uv-only; no network publish here (publish is a separate manual `uv publish --token`).
set -euo pipefail
cd "$(dirname "$0")"
: "${VIRTUAL_ENV:=/home/local/ai/.venv}"
export VIRTUAL_ENV UV_LINK_MODE=copy
echo "[ci] gfxGRAPH local CI (venv=$VIRTUAL_ENV)"

# 1) version-sync guard — pyproject [project].version must equal gfxgraph.__version__.
#    Catches the 1.0.0 regression where the wheel shipped __version__="0.4.0".
python - <<'PY'
import tomllib, re, pathlib
pj = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"]
init = pathlib.Path("python/gfxgraph/__init__.py").read_text()
got = re.search(r'__version__\s*=\s*"([^"]+)"', init).group(1)
assert got == pj, f"version drift: __version__={got!r} != pyproject={pj!r}"
print(f"[ci] version-sync ok: {pj}")
PY

# 2) test suite (ephemeral pytest overlay over the active venv).
uv run --no-project --with pytest python -m pytest tests/ -q

# 3) build sdist+wheel — catches packaging errors before a manual publish.
uv build >/dev/null
echo "[ci] build ok: $(ls -t dist/*.whl | head -1)"

echo "[ci] PASS"
