<p align="center">
  <img src="docs/assets/gfxgraph-logo.png" alt="gfxGRAPH logo" width="600" />
</p>

# gfxGRAPH v0.5.0

Drop-in **CUDA Graph → HIP Graph** translation layer for AMD **gfx1030/1031 (RDNA2)** — safe eager
fallback, dynamic-shape bucketing, a 3-tier illegal-memory-access **GUARD**, **adaptive
hardware/ROCm-PyTorch detection**, and **always-on bilingual (English / 中文) HIP/ROCm diagnostics**.
One `pip install`; it auto-detects your GPU + ROCm-PyTorch and applies only what's relevant.

> Install: `uv pip install gfxgraph` · Quick check: `gfxgraph doctor` · Explain any ROCm error
> (from *any* engine): `your-engine 2>&1 | gfxgraph explain`

## At a Glance

- **One dynamic install** — auto-detects GPU arch, ROCm-PyTorch, and the optional native bridge, and
  applies only what's present. No manual "tier" installs.
- **Adaptive** — reads the GPU on boot (or honors `GFXGRAPH_ARCH=<gfxNNNN>`), **reports the
  ROCm-PyTorch it finds**, and **errors clearly if PyTorch isn't a ROCm build** (the common CPU/CUDA-
  wheel trap) — but only when *activating the bridge*; diagnostics stay usable without torch.
- **Bilingual diagnostics** — terse HIP/ROCm errors → cause + arch context + fix; `GFXGRAPH_LANG=zh`
  for 中文; usable from **any** engine via `gfxgraph explain` (pipe its stderr).
- **GUARD** — opt-in 3-tier illegal-memory-access safety (`GFXGRAPH_GUARD=1|2|3`) — see below.
- **Collision-safe wave64/128** — captures wave64/128 intent + plans the software-wave conversion,
  **only when your code isn't already doing it** (skips if the launch already gangs warps / the grid
  is saturated / you opt out).
- **Cross-engine** — full bridge for PyTorch engines (vLLM, sglang); diagnostics for **any** engine
  (llama.cpp, candle) via the CLI; native hipGraph interposer + MIGraphX backend on the roadmap.
- **Target**: AMD RX 6700 XT / 6800 / 6900 (RDNA2) on ROCm; adapts to other archs.

## GUARD — illegal-memory-access safety (`GFXGRAPH_GUARD`)

Most "illegal memory access" crashes on ROCm come from CUDA-graph rules ROCm users don't expect.
GUARD (off by default; set `GFXGRAPH_GUARD=1|2|3`) addresses them in three escalating tiers:

| Tier | `GFXGRAPH_GUARD` | What it does |
|---|---|---|
| 1 — auto-safe-capture | `1` / `tier1` / `safe` | Force tensors entering capture/replay to be **contiguous and own their storage** (fixes non-contiguous / broadcast-0-stride / negative-stride views). Auto-corrects the whole *capture-safety* fault family. |
| 2 — fault localization | `2` / `tier2` / `localize` | Turn a would-be SIGSEGV (`hipErrorIllegalAddress`) into a precise, catchable `GfxGraphFault` (op + every tensor's layout) + graceful eager fallback. Makes *in-kernel* OOB (a producing-code logic bug — not auto-fixable) **diagnosable** instead of fatal. |
| 3 — deep guard (opt-in, slow) | `3` / `tier3` / `deep` | `RedZone` sentinel buffers catch OOB writes past gfxGRAPH-owned buffers; disables the caching allocator so faults land at real boundaries; `compute_sanitizer_cmd()` wraps a run in compute-sanitizer / rocm-memcheck to pin the exact op. |

Higher tiers include the lower ones. Programmatic API: `gfxgraph.make_safe`, `make_capture_safe`,
`validate_layout`, `GfxGraphFault`, `localize_fault`, `RedZone`, `compute_sanitizer_cmd`,
`guard_level`.

## Diagnostics — bilingual HIP/ROCm error reporting (`gfxgraph.diagnostics`)

ROCm errors are terse ("No available kernel. Aborting execution."). gfxGRAPH translates them into
**cause + your-GPU context + a concrete fix** — and works whether or not CUDA-graphs are active
(GUARD only covers the graph path). Covers `no_kernel_image`, `out_of_memory`, `illegal_address`,
`bf16_unsupported`, `wrong_arch`, `wave64_ignored`, `aiter_on_rdna`, `invalid_configuration`.

```python
import gfxgraph
gfxgraph.install_diagnostics()           # always-on: cryptic HIP errors → explained (auto when GFXGRAPH=1)
print(gfxgraph.explain("No available kernel").format())
with gfxgraph.diagnose("decode"):        # wrap a risky block
    model.generate(...)
```
**中文:** `export GFXGRAPH_LANG=zh` switches all diagnostics to Chinese (translations live in a
separate lazily-loaded `diag_zh.py`; English users pay zero cost). See [docs/GUIDE_zh.md](docs/GUIDE_zh.md).

## Adaptive behavior

- **Reads your GPU on boot** (arch / name / CU / wavefront / VRAM) and adapts diagnostics + wave
  planning to it. Override with `GFXGRAPH_ARCH=<gfxNNNN>` to target a specific card.
- **Reports the ROCm-PyTorch it finds** (`torch X · HIP Y`) and **errors clearly if PyTorch is not a
  ROCm build** (`torch.version.hip is None` — a CPU/CUDA wheel). Fires when *activating the bridge*,
  not at import (diagnostics/wavefront stay torch-free for CI/dev boxes).
- **Collision-safe wave conversion** (`GFXGRAPH_WAVE=off|detect|auto`, default `detect`): gfxGRAPH
  **does not** apply software-wave64/128 when your code already handles it — it skips if the launch
  already gangs warps (`block > wavefront`), the grid already saturates the GPU, or you set
  `GFXGRAPH_NO_WAVE=1`. (gfx1030 is Wave32-only; ROCm drops `-mwavefrontsize64`. "Conversion" =
  gang W Wave32 warps + LDS merge — a *plan/helper*, not a runtime kernel rewrite.)

```python
gfxgraph.device_info()        # DeviceInfo(arch, name, cu, wavefront, vram…)
gfxgraph.torch_rocm_status()  # {is_rocm, torch_version, hip_version, message}
gfxgraph.should_convert(block_threads, grid_blocks)  # (apply, reason) — collision-safe gate
```

## CLI (`gfxgraph …`)

The diagnostics are framework-agnostic, so the CLI helps users of **any** engine:

```bash
gfxgraph doctor                         # full env report: GPU, ROCm-PyTorch, accelerators, engines
gfxgraph device                         # detected/overridden GPU summary
gfxgraph explain "hipErrorOutOfMemory"  # explain an error (arg) …
llama-cli … 2>&1 | gfxgraph explain     # … or pipe any engine's stderr (llama.cpp/candle/vLLM)
gfxgraph run train.py                   # run a script with the CUDA→HIP bridge enabled
```

## Cross-engine support

| Engine | gfxGRAPH support |
|---|---|
| **PyTorch engines** (vLLM, sglang, TGI) | **Full** CUDA-graph bridge + GUARD + diagnostics (via the `torch.cuda.CUDAGraph` patch). |
| **llama.cpp**, **candle** | **Diagnostics now** via `gfxgraph explain` (pipe stderr). GUARD/bridge for their *native* graphs = **roadmap** via the **hipGraph interposer** (`LD_PRELOAD` over `hipGraph*`). |
| **Any engine / language** | The `gfxgraph explain` CLI works universally. |

> Note: **hipGraph** here means the **HIP runtime graph API** (the CUDA-Graphs equivalent gfxGRAPH is
> built on) — *not* the ROCm-DS `hipGRAPH` graph-*analytics* library (unrelated). **MIGraphX**
> (detected via `gfxgraph.migraphx_available()`) is a potential ONNX/IR compile backend — roadmap;
> use AMD's ONNX-Runtime MIGraphX EP today.

## Environment variables (reference)

| Variable | Default | Purpose |
|---|---|---|
| `GFXGRAPH` | off | `1` enable bridge · `debug` · `validate` (auto-installs diagnostics when set) |
| `GFXGRAPH_GUARD` | `0` | illegal-access safety tier: `0\|1\|2\|3` (`safe`/`localize`/`deep`) |
| `GFXGRAPH_DIAG` | `1` | diagnostics output; `0` to silence |
| `GFXGRAPH_LANG` | `en` | diagnostics language; `zh` for 中文 |
| `GFXGRAPH_ARCH` | (detected) | override the target GPU arch, e.g. `gfx1100` |
| `GFXGRAPH_WAVE` | `detect` | wave64/128 conversion: `off` · `detect` (warn) · `auto` |
| `GFXGRAPH_NO_WAVE` | unset | hard opt-out of wave conversion (collision avoidance) |
| `GFXGRAPH_REPLAY_MODE` | `standard` | graph replay strategy: `standard\|adaptive\|hot` |
| `GFXGRAPH_VRAM_CAP` | `0.80` | VRAM fraction for graph-capture scratch |
| `HSA_OVERRIDE_GFX_VERSION` | — | run gfx1031 as `10.3.0` (gfx1030); set on RX 6700 XT |

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

### Building the Rust Accelerators

The Rust crates (`rs_gfxgraph`, `rs_gfxgraph_stats`) provide zero-cost architectural contracts and fast-paths for graph routing. To build them from source during development:

```bash
# Ensure maturin is installed via your environment manager (e.g., uv)
# Build and install into the current environment
maturin develop --release --manifest-path rust/rs_gfxgraph/Cargo.toml
maturin develop --release --manifest-path rust/rs_gfxgraph_stats/Cargo.toml
```

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
- Native helper paths for selected bridge components (`rs_gfxgraph`, `rs_gfxgraph_stats`)
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

# Optional: VRAM cap for graph capture scratch (default 0.80 = 80% of total)
export GFXGRAPH_VRAM_CAP=0.80

# Optional: replay hot mode (skips replay-path diagnostics for lowest overhead)
export GFXGRAPH_REPLAY_HOT_MODE=1

# Optional: unified replay mode selection (standard|adaptive|hot)
# - standard: trusted replay + sampled diagnostics
# - adaptive: enables adaptive eager/graph selection and signature winner cache
# - hot: leanest replay path (minimum replay diagnostics)
export GFXGRAPH_REPLAY_MODE=adaptive

# Optional: standard-mode trusted replay tuning (safe fallback remains enabled)
export GFXGRAPH_TRUSTED_REPLAY_THRESHOLD=16
export GFXGRAPH_TRUSTED_REPLAY_SAMPLE_INTERVAL=16

# Optional: disable gfxGRAPH while keeping RDNA2 kernels
export SGLANG_DISABLE_GFXGRAPH=1

# Launch SGLang
python3 -m sglang.launch_server --model-path <model> ...
```

SGLang logs gfxGRAPH status at startup:
```
INFO: gfxGRAPH v0.3.1 enabled (mode=normal, vram_cap=0.80)
INFO: gfxGRAPH health check passed: AMD Radeon RX 6700 XT (gfx1030), VRAM 10240MB free / 12288MB total
```

### Via Environment Variable (auto-enables on import)

```bash
GFXGRAPH=1 python3 my_script.py        # standard mode
GFXGRAPH=debug python3 my_script.py    # verbose logging
GFXGRAPH=validate python3 my_script.py # correctness checking
GFXGRAPH_REPLAY_MODE=adaptive python3 my_script.py # adaptive eager/graph mode
GFXGRAPH_REPLAY_MODE=hot python3 my_script.py      # lower-overhead replay path
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

## Current Capabilities & Performance (v0.3.4)

### Verified capability snapshot

- `BridgedCUDAGraph` capture/replay works on gfx1030 with eager fallback safety.
- Dynamic-shape `ShapeBucketPool` capture/replay works across bucketed batch sizes.
- `ConditionalGraph` branch capture/replay works with fallback on per-branch failure.
- Includes explicitly tuned RDNA2 (gfx1030) `deepspeed-hip` inference kernels (layer norm, rms norm, tiled linear) and Triton kernels.

### Public benchmark (RX 6700 XT / gfx1030, ROCm 7.2, torch 2.11.0+rocm7.2)

Run:

```bash
PYTHONPATH=python python benchmarks/bench_readme_public.py \
  --run-count 3 \
  --output benchmarks/results/readme_benchmark_latest.json
```

Results from `benchmarks/results/readme_benchmark_latest.json` (**standard mode**):

| Workload | Eager (ms/iter) | Graph (ms/iter) | Status |
|---|---:|---:|---:|
| decode_like_layernorm_gelu_chain_bs1_d1024 | 0.1395 | **0.1276** | **1.09x gain** |
| mlp_bs32_d1024 | 0.1023 | 0.1028 | 1.00x parity |
| mlp_bs128_d2048 | 0.6128 | 0.6157 | 1.00x parity |

Optional with `GFXGRAPH_REPLAY_HOT_MODE=1`:

| Workload | Eager (ms/iter) | Graph (ms/iter) | Status |
|---|---:|---:|---:|
| decode_like_layernorm_gelu_chain_bs1_d1024 | 0.1378 | **0.1335** | **1.03x gain** |
| mlp_bs32_d1024 | 0.1022 | 0.1032 | 0.99x parity |
| mlp_bs128_d2048 | 0.6130 | 0.6138 | 1.00x parity |

Interpretation:
- **Stability and Parity:** The primary value is crash-free graph behavior with eager fallback safety.
- **Modest Gains:** We see modest performance gains on launch-bound decode workloads (e.g., 1.09x), with exact parity on compute-bound tasks, as expected on RDNA2.
- Standard mode now uses trusted replay promotion with sampled diagnostics and preserved eager fallback safety.
- Hot replay mode remains available when you want the leanest replay path and can accept reduced replay-path diagnostics.
- All measured runs above completed with `fallback: false` (successful graph replay path).
- Benchmark JSON now captures provenance (`commit_sha`), ROCm runtime/driver hints, tracked environment variables, and repeated run samples for reproducibility.

---

## Documentation

- [Design Specification](docs/hipgraph-bridge-design.md)
- [CUDA Parity Matrix](docs/torch-hip-rocm-graph.md)
- [Changelog](CHANGELOG.md)
- [Security Policy](SECURITY.md)

## License

**MIT** — free for any use (commercial included), modification, and redistribution; no copyleft, no
runtime royalties. The only runtime dependency is **PyTorch** (BSD-3-Clause, also permissive), so the
full stack stays permissively licensed. See [LICENSE](LICENSE).

© 2026 **Carlos Fundora** — GitHub [@carlosfundora](https://github.com/carlosfundora) ·
Hugging Face [@carlosfundora](https://huggingface.co/carlosfundora).

## Documentation
- [docs/GUIDE_zh.md](docs/GUIDE_zh.md) — 中文使用指南 (Chinese guide)
- [docs/PUBLISHING.md](docs/PUBLISHING.md) — releasing to PyPI (Trusted Publishing, first-timer friendly)
- [CHANGELOG.md](CHANGELOG.md)
