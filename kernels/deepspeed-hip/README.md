# DeepSpeed HIP Kernel Ports

Source: `DeepSpeed` fork — CUDA→HIP ports for ROCm/RDNA2 compatibility

## What these kernels do

Three core DeepSpeed inference kernels ported from CUDA to HIP:
- **Layer Norm** — fused layer normalization for transformer inference
- **Linear** — quantized INT4/INT8 GEMM with parallel dequantization
- **RMS Norm** — root-mean-square normalization (used by LLaMA-family models)

## Files

### `hip_layer_norm/`
- `layer_norm_hip.hip` — HIP kernel implementation
- `layer_norm.cpp` — PyTorch C++ binding
- `layer_norm.h` — Header

### `hip_linear/`
- `linear_kernels_hip.hip` — HIP GEMM kernel with tiled matmul
- `linear_kernels.cpp` — PyTorch C++ binding
- `include/` — Shared memory utils, PTX→HIP intrinsic wrappers, configs

### `hip_rms_norm/`
- `rms_norm_hip.hip` — HIP kernel implementation
- `rms_norm.cpp` — PyTorch C++ binding
- `rms_norm.h` — Header
