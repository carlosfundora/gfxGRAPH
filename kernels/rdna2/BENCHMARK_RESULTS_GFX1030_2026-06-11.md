# RDNA2 Custom Kernel Benchmarks — Real GPU, 2026-06-11

**Hardware:** AMD Radeon RX 6700 XT — native arch **gfx1031** (Navi 22), single GPU.
**Stack:** torch `2.13.0a0+git53bbebe`, ROCm/HIP `7.2.26015`, triton `3.7.0`, `torch.cuda.is_available()` = True.
**Method:** `torch.cuda.Event`, 10 warmup + 100 timed iterations, median (p90 also captured). fp16 for LLM ops, fp32 for Snake. speedup = eager_median / kernel_median. **Correctness gates timing** — a fast-but-wrong kernel is REJECT.

Per-run data: `BENCHMARK_RESULTS_GFX1030_2026-06-11.json`. The placeholder `KERNEL_BENCHMARK_RESULTS.json` (0.0ms stubs) was left untouched and is superseded by this file.

## Verdicts

| Kernel | Subject | Correct | speedups (3 shapes) | Verdict |
|---|---|---|---|---|
| **rmsnorm** | `rdna2/rmsnorm.py` (HIP C++) | ✅ | 6.7× / 5.9× / 19.7× | **ADOPT** |
| **fused_qknorm_rope** | `rdna2/fused_qknorm_rope.py` (HIP C++) | ✅ | 41.0× / 10.9× / 17.5× | **ADOPT** |
| **snake_activation** | gfxatom `model_ops/rdna2/snake_activation.py` (Triton) | ✅ | 2.8× / 2.5× / 3.2× | **ADOPT** |
| **rope** | `rdna2/rope.py` (HIP C++) | ❌ | (16.7/7.5/8.9×, irrelevant) | **REJECT — incorrect** |
| **fp8_dequant** | `rdna2/fp8_dequant.py` (HIP C++) | ❌ | (54/42/42×, irrelevant) | **REJECT — incorrect** |

### Why the two rejects
- **rope**: correct only at a single decode token (position 0 = identity rotation); at every realistic multi-token prefill shape it diverges from the eager reference by abs err ~7–8. Genuine kernel bug; speed is moot.
- **fp8_dequant**: disagrees with the module's *own* PyTorch E4M3 reference by abs err 24.0 at all three shapes. At least one of {kernel, fallback} is buggy; correctness cannot be certified.

## Environment caveats (must inform any GPU work on this box)
1. **`HSA_OVERRIDE_GFX_VERSION=10.3.0` (the box-default in `.bashrc`) segfaults torch 2.13 / ROCm 7.2 GPU compute** — deterministic core-dump on `torch.randn` alloc, reproduced 8+ times. Benchmarks ran with the override **unset** (native gfx1031). Any torch-on-GPU path must unset it or use a different torch build.
2. **The GPU is gfx1031, not gfx1030.** The four HIP kernels hardcode `--offload-arch=gfx1030` and aiter's `chip_info` rejects gfx1031. Ran with a `GPU_ARCHS=gfx1030` shim (same gfx10.3 ISA); independently verified a gfx1030-compiled binary executes correctly on this hardware. Out-of-the-box, these kernels do **not** build on the RX 6700 XT without the shim.

## Adoption note
These verdicts **gate** future kernel integration into the engines; no kernel is wired in yet. The three ADOPT kernels are candidates for the candle/inference GPU path once the ROCm-GPU environment issues above are resolved.
