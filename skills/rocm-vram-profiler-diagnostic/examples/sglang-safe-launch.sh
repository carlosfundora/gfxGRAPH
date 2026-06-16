#!/usr/bin/env bash
set -euo pipefail

# Example: Safe SGLang launch for RX 6700 XT (gfx1030 / RDNA2)
# Mitigation parameters for context-limiting and safe RAM ceilings.
# Aiter does NOT work on RDNA2!

echo "Starting SGLang server with strict VRAM mitigations for RDNA2..."

python3 -m sglang.launch_server \
  --model meta-llama/Llama-3-8B-Instruct \
  --dtype float16 \
  --attention-backend triton \
  --mem-fraction-static 0.8 \
  --context-length 4096 \
  --host 0.0.0.0 \
  --port 30000
