# RDNA2 (gfx1030) Optimized Kernels

This directory contains Wave32-tuned HIP/Triton kernels specifically optimized for AMD RDNA2 (gfx1030) consumer GPUs. These kernels are designed to replace failing or unoptimized upstream kernels in frameworks like SGLang and vLLM.

## Architecture Highlights
- **Wave32 Tuning**: All kernels assume 32-thread wavefronts, the native size for RDNA2.
- **No Matrix Cores**: RDNA2 lacks CDNA-style Matrix Cores. Computations rely on vector ALUs and `dp4a` intrinsics where applicable.
- **Bandwidth Optimization**: Kernels are structured around 2-4 warps (64-128 threads) with vec8 memory access patterns to saturate RDNA2's memory bandwidth.

## Compilation Backends
The module attempts to compile kernels using the fastest available backend:
1. **AITER JIT (`@compile_ops`)**: Preferred, fastest execution path.
2. **Torch C++ Extension**: Inline compilation fallback if AITER is unavailable.
3. **Pure Triton**: Universal fallback if native C++ compilation fails.

## Included Kernels
- **RMSNorm**: Standard and fused-add variants.
- **FP8 Dequantization**: Software-based int8 → fp16 with scale.
- **RoPE**: Positional encoding for NeoX and GPT-J styles.
- **Fused QKNorm + RoPE**: For models like Qwen3, Gemma 4, and DeepSeek-V3.
- **Activations**: SiLU, GELU.

## Usage
These kernels are automatically dispatched when `HSA_OVERRIDE_GFX_VERSION` indicates an RDNA2 target (e.g., `10.3.0`).
