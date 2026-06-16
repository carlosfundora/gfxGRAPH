#!/usr/bin/env bash
# Publisher: Carlos Fundora · GitHub: @carlosfundora · MIT
# CI step: Build native C++ compat bridge target using CMake
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[ci] Configuring and building C++ native bridge..."
cmake -S . -B build --preset release -DBUILD_CUDA_COMPAT=ON
cmake --build build -j$(nproc)
