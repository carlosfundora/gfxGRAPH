#!/usr/bin/env bash
set -euo pipefail

# Verify rocm-smi is available in the environment

echo "Verifying ROCm SMI availability..."

if command -v rocm-smi >/dev/null 2>&1; then
    echo "rocm-smi is installed and available."
    rocm-smi --showmeminfo vram
    exit 0
else
    echo "Error: rocm-smi is not found in the path. Please ensure ROCm is installed."
    exit 1
fi
