# Changelog

All notable changes to the DeepSpeed HIP ported kernels will be documented in this file.

## [Unreleased]

### Added
- Initial port of DeepSpeed inference kernels from CUDA to HIP.
- `hip_layer_norm/`: Fused layer normalization kernel targeting ROCm/RDNA2.
- `hip_linear/`: Tiled GEMM implementation with parallel dequantization for INT4/INT8 workloads.
- `hip_rms_norm/`: Root-mean-square normalization kernel optimized for LLaMA-family models.
