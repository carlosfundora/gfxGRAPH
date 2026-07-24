---
name: rocm-atom-inference
description: Guide and resources for deployment and development using the ROCm/AMD ATOM (AiTer Optimized Model) inference backend.
---

# ROCm ATOM Inference Backend

ATOM (AiTer Optimized Model) is a lightweight vLLM-like implementation, focusing on integration and optimization based on AITER kernels. It provides an OpenAI-Compatible API drop-in server (`/v1/chat/completions` and `/v1/completions`) that is optimized for AMD's ROCm platform.

## When to use this skill
- You need to deploy a Large Language Model (e.g., LLaMa, Qwen, DeepSeek, Mixtral) using an AMD GPU environment.
- You need to benchmark or test inference throughput and latency on ROCm.
- You want to start an OpenAI-compatible API server using the ATOM backend.
- You need to manage KV cache transfers, speculative decoding (MTP), or multi-GPU parallelism (TP/DP/EP) on AMD hardware.

## Features
- **ROCm Optimized**: Built on AMD's ROCm platform with AITER kernels (ASM, CK, Triton).
- **Piecewise `torch.compile`**: 4 compilation levels with CUDA graph capture for low-latency decode.
- **Multi-GPU Parallelism**: Tensor parallelism (TP), data parallelism (DP), and expert parallelism (EP) with MORI all-to-all.
- **Speculative Decoding**: Multi-Token Prediction (MTP) with EAGLE proposer.
- **Prefix Caching**: xxhash64-based KV cache block sharing across sequences.

## Installation
The recommended way to install and use ATOM is via the nightly Docker image. See `scripts/install_atom_container.sh` for an example. Alternatively, you can build from the base ROCm image by pulling `rocm/pytorch` and installing `amd-aiter` via pip.

## Basic Usage

### Simple Inference Example
```bash
# Requires ninja and huggingface_hub installed
python -m atom.examples.simple_inference --model meta-llama/Meta-Llama-3-8B --kv_cache_dtype fp8
```
*Note: First-time execution may take approximately 10 minutes for model compilation.*

### Starting the Server
Start an OpenAI-compatible server using the entrypoint. See `examples/example_openai_server.sh` for multi-GPU and speculative decoding examples.

```bash
python -m atom.entrypoints.openai_server --model Qwen/Qwen3-0.6B --kv_cache_dtype fp8
```

## Performance & Profiling
To run an online throughput benchmark against a running server:
```bash
python -m atom.benchmarks.benchmark_serving \
  --model=deepseek-ai/DeepSeek-R1 --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=1024 --random-output-len=1024 \
  --random-range-ratio=0.8 \
  --num-prompts=1280 --max-concurrency=128 \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

To collect a profile trace, launch the server with `--torch-profiler-dir ./trace --mark-trace`.
