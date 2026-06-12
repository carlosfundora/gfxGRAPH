# Changelog

All notable changes to the RDNA2 (gfx1030) kernels will be documented in this file.

## [Unreleased]

### Added
- Initial set of RDNA2 (gfx1030) optimized HIP kernels for SGLang compatibility.
- Wave32-tuned RMSNorm (with fused add).
- FP8 software dequantization kernel (`int8` → `fp16`).
- RoPE positional encoding (NeoX and GPT-J styles).
- Fused QKNorm + RoPE for newer architectures.
- SiLU and GELU activation kernels.
- Triple-backend compilation system (AITER JIT -> Torch C++ Extension -> Triton).
- `BENCHMARK_RESULTS_GFX1030_2026-06-11` documenting kernel performance against eager and upstream.
