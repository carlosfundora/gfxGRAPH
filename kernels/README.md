# Kernel Reference Archive

Copies of custom/modified GPU kernels from ENCOM/THOTH build work.
These serve as reusable references for future HIP/ROCm/CUDA porting.

> [!NOTE]
> This directory is for reference material only and is not packaged or shipped in the `gfxGRAPH` PyPI wheel.

## Contents

### [`deepspeed-hip/`](deepspeed-hip/) — DeepSpeed CUDA→HIP Ports
Three core inference kernels (layer_norm, linear, rms_norm) ported from CUDA to HIP
for ROCm/RDNA2 compatibility. Includes tiled GEMM with parallel dequantization.
- **19 files** | `.hip`, `.cpp`, `.cuh`, `.h`

### [`rdna2/`](rdna2/) — Triton RDNA2 Optimizations
Triton kernels explicitly optimized for the AMD Radeon RX 6700 XT (gfx1030) architecture. Features optimized block sizes, manual unrolling, and avoidance of `bfloat16` instructions not supported on RDNA2.

## How to use

These kernels are provided as-is for reference and manual inclusion into inference pipelines. They do not automatically load into PyTorch via `gfxGRAPH`.
