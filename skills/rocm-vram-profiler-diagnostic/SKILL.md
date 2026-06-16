---
name: rocm-vram-profiler-diagnostic
description: Core runbook for diagnosing memory ceiling limits on AMD's ROCm toolkit natively. Details mitigation parameters and context-limiting configurations explicitly tailored for the Radeon RX 6700 XT.
---

# ROCm System & VRAM Diagnostics

Use this skill when the local LLM processes (vLLM, SGLang, PyTorch code) crash with out-of-memory errors on the AMD hardware, or when assessing parameters prior to model loading.

## Execution Resources

This skill bundle includes practical execution resources alongside this file:
- `scripts/monitor-vram.sh`: A wrapper to continuously monitor VRAM limits.
- `examples/vllm-safe-launch.sh`: Reference implementation for safe vLLM memory boundaries.
- `examples/sglang-safe-launch.sh`: Reference implementation for safe SGLang memory boundaries.
- `tests/verify-rocm-smi.sh`: A validation mechanism to ensure ROCm environment availability.

## 1. Hardware Constriction Assessment

The target device operates with **12 GB of GDDR6 VRAM**.
- OS Baseline (X11/Wayland + Chrome): ~1.2 GB.
- Container Overhead: ~0.5 GB.
- **Max Theoretical Free Bandwidth:** 10.3 GB.

This forces a harsh boundary. A common Q5 quantized 8-billion parameter model takes exactly 5.5 to 6.3 GB just to sit idle. This leaves ~4.0 GB for the KV Cache.

## 2. Debugging Native Commands

If the user reports system freezing or agents outputting empty blocks, immediately run:
```bash
watch -d -n 1 rocm-smi
```
*Tip: You can use the provided script at `scripts/monitor-vram.sh` to run this natively.*

Watch the specific VRAM% output and Memory Temp fields. If VRAM exceeds 98%, the system driver will usually aggressively kill the python interpreter to prevent graphics server crashes.

## 3. Context Length and KV Cache Slicing

If generation length causes failure, you must limit the context windows. Memory for text inference works exponentially on context scales.

**In vLLM Launch Strings:**
1. Decrease the max-model-len: `--max-model-len 4096` instead of `8192`.
2. Apply strict GPU utilization scaling: `--gpu-memory-utilization 0.85` forces the engine to cap RAM limits, returning clean `503 Service Unavailable` instead of halting the root shell.

**In SGLang Launch Strings:**
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3-8B-Instruct \
  --dtype float16 \
  --attention-backend triton \ # Aiter does NOT work on RDNA2!
  --mem-fraction-static 0.8 \
  --context-length 4096 \
  --host 0.0.0.0 \
  --port 30000
```

## 4. DType Fatalities

The RX 6700 XT (gfx1030 / RDNA2) lacks the matrix instructions necessary for native `bfloat16`. 
If a Python repository forces `bfloat16` under the hood without your knowledge, PyTorch will fallback to incredibly slow emulation modes or instantly crash producing an `Illegal instruction (core dumped)` error.

Always inject the `--dtype float16` constraint whenever parsing any Transformers-based execution wrapper.

## Verification

Before asserting that ROCm constraints are causing the issue or assuming environment capabilities, run `tests/verify-rocm-smi.sh` to check tool availability and current VRAM usage boundaries.
