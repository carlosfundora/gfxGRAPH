#!/usr/bin/env bash
set -euo pipefail

# Example: Safe vLLM launch for RX 6700 XT (gfx1030 / RDNA2)
# Mitigation parameters for context-limiting and safe RAM ceilings.

echo "Starting vLLM server with strict VRAM mitigations for RDNA2..."

python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-8B-Instruct \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000
