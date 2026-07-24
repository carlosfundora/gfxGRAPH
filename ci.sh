#!/usr/bin/env bash
# Publisher: Carlos Fundora · GitHub: @carlosfundora · MIT
# Local CI runner for gfxGRAPH (rust-hip-cpp branch)
set -euo pipefail

cd "$(dirname "$0")"
PROJECT_DIR="$PWD"

# Detect and export branch-local virtual environment
VENV_DIR="$PROJECT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    export VIRTUAL_ENV="$VENV_DIR"
    echo "[ci] Using branch-local virtual environment: $VENV_DIR"
else
    echo "[ci] Warning: local .venv not found. Running with system environment"
fi

# Ensure all step scripts are executable
chmod +x ci/*.sh

STAGES=(
    "ci/version_sync.sh"
    "ci/build_rust.sh"
    "ci/build_cpp.sh"
    "ci/test_cpp.sh"
    "ci/test_python.sh"
    "ci/build_package.sh"
)

echo "=== Starting gfxGRAPH Local CI Pipeline ==="
for stage in "${STAGES[@]}"; do
    echo "--------------------------------------------------"
    echo "Running stage: $stage"
    if ! ./"$stage"; then
        echo "⛔ Stage failed: $stage" >&2
        exit 1
    fi
done

echo "--------------------------------------------------"
echo "✓ ALL CI STAGES PASSED SUCCESSFULLY"
