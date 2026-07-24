#!/bin/bash
# Example script to start the ATOM OpenAI-compatible server

MODEL="deepseek-ai/DeepSeek-R1"
TP_SIZE=8
KV_CACHE_DTYPE="fp8"

# Start the server with Tensor Parallelism and Multi-Token Prediction (MTP) speculative decoding
python3 -m atom.entrypoints.openai_server \
  --model $MODEL \
  --kv_cache_dtype $KV_CACHE_DTYPE \
  -tp $TP_SIZE \
  --method mtp \
  --num-speculative-tokens 3
