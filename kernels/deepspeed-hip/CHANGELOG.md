# Changelog

All notable changes to the DeepSpeed HIP ported kernels will be documented in this file.

## [Unreleased]

### Added
- Initial port of DeepSpeed inference kernels from CUDA to HIP.
- `hip_layer_norm/`: Fused layer normalization kernel targeting ROCm/RDNA2.
- `hip_linear/`: Tiled GEMM implementation with parallel dequantization for INT4/INT8 workloads.
- `hip_rms_norm/`: Root-mean-square normalization kernel optimized for LLaMA-family models.

### Changed
- Hardened layer norm and RMSNorm launch scheduling on RDNA2 so the subblock path is only used for supported 1/2/4/8/16-thread groups; larger small-row float cases now route to the full-block schedule instead of silently skipping launch.
- Kept RMSNorm reciprocal-square-root scaling in float until the final cast to reduce half/bfloat16 precision loss.
