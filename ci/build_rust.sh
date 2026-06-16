#!/usr/bin/env bash
# Publisher: Carlos Fundora · GitHub: @carlosfundora · MIT
# CI step: Compile and register PyO3 Rust bindings via Maturin
set -euo pipefail
cd "$(dirname "$0")/.."

# Detect Maturin path
if [ -x "/home/local/.local/bin/maturin" ]; then
    MATURIN_BIN="/home/local/.local/bin/maturin"
else
    MATURIN_BIN="maturin"
fi

# Detect VENV
VENV_DIR="${VIRTUAL_ENV:-$PWD/.venv}"

echo "[ci] Rebuilding Rust bindings via system maturin..."
cd rust/rs_gfxgraph
VIRTUAL_ENV="$VENV_DIR" "$MATURIN_BIN" develop --release
