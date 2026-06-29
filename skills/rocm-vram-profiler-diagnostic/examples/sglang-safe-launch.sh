#!/usr/bin/env bash
set -euo pipefail

# Example: Safe SGLang launch for RX 6700 XT (gfx1030 / RDNA2)
# Mitigation parameters for context-limiting and safe RAM ceilings.
# Native AITER attention WORKS on this patched gfx1030 (enable via SGLANG_USE_AITER=1); the old
# "Aiter does NOT work on RDNA2" note was wrong. Leaving the backend unforced lets sglang prefer
# aiter. (For capture, keep cuda-graph on the triton path until the aiter+hgb_decode_pool lane is wired.)

echo "Starting SGLang server with strict VRAM mitigations for RDNA2..."

SGLANG_USE_AITER=1 python3 -m sglang.launch_server \
  --model meta-llama/Llama-3-8B-Instruct \
  --dtype float16 \
  --mem-fraction-static 0.8 \
  --context-length 4096 \
  --host 0.0.0.0 \
  --port 30000
