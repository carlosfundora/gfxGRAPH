<p align="center">
  <img src="docs/assets/gfxgraph-logo.png" alt="gfxGRAPH logo" width="600" />
</p>

# gfxGRAPH v0.3.1

Drop-in CUDA Graph → HIP Graph translation layer for AMD gfx1030/1031 (RDNA2). gfxGRAPH provides production-safe graph capture/replay with eager fallback, dynamic-shape bucketing, validation, and optional native acceleration paths.

## At a Glance

- **Tier 1**: pure-Python integration with monkey-patched `torch.cuda.CUDAGraph`
- **Tier 2**: native bridge for conditional nodes, rapid launch, and nested capture gaps
- **Target**: AMD Radeon RX 6700 XT / 6800 / 6900 class GPUs on ROCm
- **Focus**: transparent integration, safe fallback behavior, and practical performance on RDNA2

## Table of Contents

- [Target Hardware](#target-hardware)
- [Quick Start](#quick-start)
- [Two Operating Tiers](#two-operating-tiers)
- [Usage](#usage)
- [Architecture](#architecture)
- [Observability](#observability)
- [Troubleshooting](#troubleshooting)
- [Current Capabilities & Performance](#current-capabilities--performance-v031)
- [Documentation](#documentation)
- [License](#license)

## Target Hardware

| Component | Requirement |
|-----------|-------------|
| **GPU** | AMD Radeon RX 6700 XT / 6800 / 6900 (gfx1030, RDNA2) |
| **ROCm** | 7.2.0+ |
| **PyTorch** | 2.9+ (ROCm build) |
| **Python** | 3.12+ |

## Quick Start

If you just want gfxGRAPH working with the fewest moving parts, start with Tier 1.

### Fastest Path: Tier 1

```bash
# Install PyTorch ROCm build
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

# Install gfxGRAPH from repo root
pip install /path/to/gfxGRAPH

# Verify
python3 -c "import gfxgraph; print(gfxgraph.__version__); print(gfxgraph.health_check())"
```

Expected result:
- `native_bridge: False`
- This is normal in Tier 1
- All Python-level features still work

### Native Path: Tier 2

```bash
pip install /path/to/gfxGRAPH
pip install /path/to/gfxGRAPH/native

python3 -c "import gfxgraph; print(gfxgraph.health_check())"
```

Expected result:
- `native_bridge: True`

---

## Two Operating Tiers

gfxGRAPH works in **two tiers** depending on which dependencies you install.
**Most users only need Tier 1** because it provides the full Python-level
integration, including the monkey-patch that makes CUDA graphs work
transparently on RDNA2.

### Tier Comparison

| Tier | Install Style | What You Get | Best For |
|------|---------------|--------------|----------|
| **Tier 1** | Pure Python | Monkey-patch, eager fallback, shape bucketing, validation, stats, health checks | Most users getting started |
| **Tier 2** | Python + native companion | Native acceleration paths for routing, validation, and conditional helpers | Users who want lower Python overhead where available |

### Tier 1: Python-Only Mode

**What you get:**
- `torch.cuda.CUDAGraph → BridgedCUDAGraph` monkey-patch (transparent to callers)
- Eager fallback — capture/replay failures never crash, just run slower
- Shape bucketing — reduced graph captures for dynamic batch sizes
- VRAM safety cap — prevents graph capture OOM (`GFXGRAPH_VRAM_CAP`)
- Validation mode — catches silent HIP Graph correctness bugs (PyTorch #155684)
- Thread-safe stats: `gfxgraph.stats()` → capture/replay/fallback counts
- Health check: `gfxgraph.health_check()` → GPU info + smoke test
- Structured logging: `HGB_LOG_LEVEL=debug|info|warn|error`

**Dependencies:**
```bash
# That's it — just PyTorch (ROCm build) and Python
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

**Install gfxGRAPH:**
```bash
# Preferred source install from repo root
pip install /path/to/gfxGRAPH

# Transitional compatibility path
pip install /path/to/gfxGRAPH/python/
```

**Verify:**
```bash
python3 -c "import gfxgraph; print(gfxgraph.__version__); print(gfxgraph.health_check())"
```

You'll see `native_bridge: False` — that's expected and fine. All Python-level
features work without the native library.

### Tier 2: Full Native Mode

This is the advanced path and requires the ROCm SDK.

**What you get additionally:**
- Native helper paths for selected bridge components (`gfxgraph_rs`, `gfxgraph_stats`)
- Optional `libhipgraph_bridge.so` loading when present
- Lower Python overhead on supported paths

**System dependencies (Ubuntu/Debian):**
```bash
# ROCm SDK — the big one. Follow AMD's official guide:
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
#
# Key packages needed:
sudo apt-get install -y \
    rocm-dev \
    hip-dev \
    hipcc \
    rocm-cmake

# Build tools
sudo apt-get install -y cmake ninja-build
```

> ⚠️ **ROCm SDK installation is non-trivial.** It requires kernel-level drivers,
> specific package repositories, and careful version matching. Plan for 30-60 min
> on a fresh system. If you're running PyTorch ROCm builds, you likely already
> have `libamdhip64.so` — but you still need `hip-dev` headers and `hipcc` for
> compiling the bridge.

#### Option A: Build the Native Bridge Locally

```bash
cd /path/to/gfxGRAPH

cmake --preset release
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure
```

#### Option B: Install the Native Companion Package

```bash
pip install /path/to/gfxGRAPH
pip install /path/to/gfxGRAPH/native
```

`pip install .[native]` is intentionally **not** the supported source-install path
in this batch. Tier 2 stays a two-step flow so plain `pip install /path/to/gfxGRAPH`
remains a true pure-Python install.

gfxGRAPH checks `GFXGRAPH_LIB` first, then the canonical packaged resolver
`gfxgraph._native.library_path()`, then local `build/` outputs, and finally
standard loader paths. During this phase the companion package still owns the
actual `.so`, but runtime code treats `gfxgraph._native` as the canonical lookup.

**Verify native bridge loaded:**
```bash
python3 -c "import gfxgraph; print(gfxgraph.health_check())"
# Should show: native_bridge: True
```

---

## Usage

### Standalone (any PyTorch code)

```python
import gfxgraph
gfxgraph.enable()  # patches torch.cuda.CUDAGraph globally

# Your existing CUDA graph code works unchanged:
graph = torch.cuda.CUDAGraph()  # actually BridgedCUDAGraph
# ... capture_begin / capture_end / replay all delegate correctly
```

### With SGLang

gfxGRAPH integrates transparently with SGLang's CUDA graph runner.
Set these environment variables before launching:

```bash
# Required: enable RDNA2 kernel paths (activates gfxGRAPH)
export SGLANG_RDNA2_KERNELS=1

# Required for gfx1031 (RX 6700 XT)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030

# Optional: validation mode (catches silent graph correctness bugs)
export GFXGRAPH=validate

# Optional: debug logging
export GFXGRAPH=debug

# Optional: VRAM cap for graph capture scratch (default 0.90 = 90% of total)
export GFXGRAPH_VRAM_CAP=0.90

# Optional: disable gfxGRAPH while keeping RDNA2 kernels
export SGLANG_DISABLE_GFXGRAPH=1

# Launch SGLang
python3 -m sglang.launch_server --model-path <model> ...
```

SGLang logs gfxGRAPH status at startup:
```
INFO: gfxGRAPH v0.3.1 enabled (mode=normal, vram_cap=0.90)
INFO: gfxGRAPH health check passed: AMD Radeon RX 6700 XT (gfx1030), VRAM 10240MB free / 12288MB total
```

### Via Environment Variable (auto-enables on import)

```bash
GFXGRAPH=1 python3 my_script.py        # standard mode
GFXGRAPH=debug python3 my_script.py    # verbose logging
GFXGRAPH=validate python3 my_script.py # correctness checking
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   User Application                    │
├──────────────┬───────────────────┬───────────────────┤
│   PyTorch    │   Direct HIP C   │  Unmodified CUDA  │
├──────────────┼───────────────────┼───────────────────┤
│  Layer 2     │                   │  Layer 3          │
│  hipgraph_   │                   │  libcudagraph_    │
│  bridge/     │                   │  compat.so        │
│  (Python)    │                   │  (LD_PRELOAD)     │
├──────────────┴───────────────────┴───────────────────┤
│            Layer 1: libhipgraph_bridge.so             │
│     Gap bridges · Routing logic · Kernel pool         │
├──────────────────────────────────────────────────────┤
│         libamdhip64.so  (ROCm · 104 symbols)          │
├──────────────────────────────────────────────────────┤
│              gfx1030 · RDNA2 Hardware                 │
└──────────────────────────────────────────────────────┘
```

### Gaps Bridged

| # | Gap | Bridge Strategy | Availability |
|:-:|-----|----------------|:------------:|
| 51 | Conditional nodes | Per-branch graph dispatch with eager fallback | Tier 1/2 |
| 52 | Device-side launch | Native launch-path helpers when bridge library is present | Tier 2 |
| 53 | Dynamic input shapes | Shape bucketing with VRAM-aware capture + replay | Tier 1/2 |
| 54 | Nested capture | Native nested-capture support when bridge library is present | Tier 2 |

### Routing Strategy

| Tier | Stack | Intent |
|:----:|-------|:------:|
| 0 | `torch.compile` only | Baseline compiler path |
| 1 | HIP Graph + gfxGRAPH (Python-only) | Default production path |
| 2 | HIP Graph + gfxGRAPH (+ native companion) | Lower-overhead helper paths where available |

---

## Observability

```python
import gfxgraph

# Performance counters
gfxgraph.stats()
# → {'enabled_at': 1712..., 'capture_count': 32, 'replay_count': 1847,
#     'fallback_count': 0, 'validation_failures': 0, 'avg_replay_us': 42.3}

# Health check
gfxgraph.health_check()
# → {'ok': True, 'gpu': 'AMD Radeon RX 6700 XT', 'rocm': 'gfx1030',
#     'native_bridge': False, 'vram_total_mb': 12288, 'vram_free_mb': 10240,
#     'details': 'Graph capture/replay OK, output verified'}

# Status
gfxgraph.is_enabled()  # → True
```

---

## Troubleshooting

### "Native bridge not available" message at startup

**Expected in Tier 1.** gfxGRAPH runs in pure-Python mode — all key features work.
Build `libhipgraph_bridge.so` (see Tier 2 above) only if you need the 2 extra native-only gaps.

### Health check returns `ok: False`

- Verify ROCm is working: `rocminfo | grep gfx`
- Check HSA override: `echo $HSA_OVERRIDE_GFX_VERSION` (should be `10.3.0` for gfx1031)
- Test PyTorch: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Check for PyTorch #155684 (HIP Graph correctness bug) — use `GFXGRAPH=validate`

### CUDA graphs fail during SGLang model loading

- Set `AMD_SERIALIZE_KERNEL=3` and `AMD_SERIALIZE_COPY=3` (SGLang sets these automatically)
- Reduce `GFXGRAPH_VRAM_CAP` if running near VRAM limits
- Try `SGLANG_DISABLE_GFXGRAPH=1` to isolate whether gfxGRAPH is the issue

### Fallback count keeps increasing

- Some graph shapes may genuinely fail on HIP — eager fallback is intentional
- Check `HGB_LOG_LEVEL=debug` for detailed failure reasons
- If all captures fail, the underlying HIP Graph support may be broken

---

## Current Capabilities & Performance (v0.3.1)

### Verified capability snapshot

- `BridgedCUDAGraph` capture/replay works on gfx1030 with eager fallback safety.
- Dynamic-shape `ShapeBucketPool` capture/replay works across bucketed batch sizes.
- `ConditionalGraph` branch capture/replay works with fallback on per-branch failure.
- Rust companion modules (`gfxgraph_rs`, `gfxgraph_stats`) are optional, but when present they are used for router/validator/stats fast paths.

### Public benchmark (RX 6700 XT / gfx1030, ROCm 7.2, torch 2.11.0+rocm7.2)

Run:

```bash
PYTHONPATH=python python benchmarks/bench_readme_public.py \
  --output benchmarks/results/readme_benchmark_latest.json
```

Results from `benchmarks/results/readme_benchmark_latest.json`:

| Workload | Eager (ms/iter) | Graph (ms/iter) | Speedup |
|---|---:|---:|---:|
| decode_like_layernorm_gelu_chain_bs1_d1024 | 0.1396 | **0.1167** | **1.20x** |
| mlp_bs32_d1024 | 0.1013 | 0.1035 | 0.98x |
| mlp_bs128_d2048 | 0.6139 | 0.6161 | 1.00x |

Interpretation:
- gfxGRAPH currently shows the strongest gains on launch-bound decode-like chains.
- For heavier throughput-bound GEMM paths, replay is near parity in this environment.
- All measured runs above completed with `fallback: false` (successful graph replay path).

---

## Documentation

- [Design Specification](docs/hipgraph-bridge-design.md)
- [CUDA Parity Matrix](docs/torch-hip-rocm-graph.md)

## License

MIT
