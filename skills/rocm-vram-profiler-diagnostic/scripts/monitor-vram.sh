#!/usr/bin/env bash
set -euo pipefail

# Monitor ROCm VRAM usage to detect potential out-of-memory states

echo "Monitoring ROCm VRAM% output and Memory Temp fields."
echo "Press Ctrl+C to stop."
echo "Note: If VRAM exceeds 98%, the system driver will usually aggressively kill the python interpreter."

watch -d -n 1 rocm-smi
