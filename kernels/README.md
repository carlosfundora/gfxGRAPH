# Kernel Reference Archive

Copies of all custom/modified GPU kernels from ENCOM/THOTH build work.
These serve as reusable references for future HIP/ROCm/CUDA porting.

## Contents

### [`deepspeed-hip/`](deepspeed-hip/) — DeepSpeed CUDA→HIP Ports
Three core inference kernels (layer_norm, linear, rms_norm) ported from CUDA to HIP
for ROCm/RDNA2 compatibility. Includes tiled GEMM with parallel dequantization.
- **19 files** | `.hip`, `.cpp`, `.cuh`, `.h`

### [`sglang-prism-q1/`](sglang-prism-q1/) — PRISM Q1_0 GPU Kernels
1-bit quantization (Q1_0) CUDA kernels for sglang's GGUF inference path.
Symmetric binary quantization: each bit → `+d` or `-d`. Two block sizes (32, 128).
Uses `dp4a` INT8 dot product for fast accumulation.
- **6 source files + 1 patch** | `.cu`, `.cuh`, `.h`
- Source: `sglang-1-bit-turbo` fork (uncommitted on `main`)

### [`llama-cpp-tq3-kvcache/`](llama-cpp-tq3-kvcache/) — TQ3_0 KV Cache Quantization
TurboQuant 3-bit (3.5 bpw) KV cache compression kernels for llama.cpp.
Per-block Walsh-Hadamard Transform rotation with 4-centroid MSE codebook.
CUDA and CPU implementations.
- **12 source files + 1 patch** | `.cu`, `.cuh`, `.c`, `.h`
- Source: `llama.cpp-1-bit-turbo` fork, commit `a432f38e5`

### [`vllm-rdna2/`](vllm-rdna2/) — RDNA2 Advanced Indexing Fallback
CPU fallback for HIP runtime crash on RDNA2 during speculative decode
tensor indexing. Catches HIP `AcceleratorError` and falls back to
CPU-side `index_select`.
- **1 source file + 1 patch** | `.py`
- Source: `vllm` fork, branch `rdna2-index-fallback`

## Not included

- **SpecForge** — No custom CUDA/HIP kernels (pure Python/ML framework)
- **llama.cpp `review/rocm-hardening`** — Only test harness files, no kernel changes
- **WIP llama.cpp builds** — Compiled `.so` artifacts only, no unique kernel sources

## How to use

Each directory contains full source files plus a `.patch` file showing just our
modifications vs upstream. The patches are the fastest way to see what changed:

```bash
# View what we added to sglang for PRISM Q1_0:
cat sglang-prism-q1/PRISM_Q1_0.patch

# Apply TQ3_0 changes to a fresh llama.cpp checkout:
cd /path/to/llama.cpp && git apply /mnt/ai/build/kernels/llama-cpp-tq3-kvcache/TQ3_0_KV_CACHE.patch
```
