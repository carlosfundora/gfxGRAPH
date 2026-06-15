# Changelog

All notable changes to this project are documented in this file.

## [0.5.0] - 2026-06-15

### Added
- **Adaptive hardware + ROCm-PyTorch detection** (`hipgraph_bridge.hardware`, exported from
  `gfxgraph`): `device_info()` reads the GPU on boot (arch/name/CU/wavefront/VRAM); `GFXGRAPH_ARCH`
  overrides the target card. `require_rocm_torch()` (called at `enable()`) **reports the ROCm-PyTorch
  found** (`torch X · HIP Y`) and **raises a precise error if PyTorch is missing or a CPU/CUDA wheel**
  (`torch.version.hip is None`) — fired at bridge activation, so diagnostics/wavefront stay torch-free.
  `detect_accelerators()` / `detect_engines()` discover MIGraphX/AITER/Triton/native-bridge + engines.
  Diagnostics + wavefront now **adapt to the detected/overridden arch** (no longer hardcoded gfx1030).
- **Collision-safe wave conversion**: `GFXGRAPH_WAVE=off|detect|auto` + `wavefront.should_convert()`
  — gfxGRAPH only plans software-wave64/128 when the user's code isn't already doing it (skips when
  the launch gangs warps / the grid is saturated / `GFXGRAPH_NO_WAVE=1`). `plan_software_wave` now
  defaults lanes/CU from the detected device.
- **Universal CLI** (`python -m gfxgraph` / `gfxgraph`): `explain` (arg or piped stderr — works for
  ANY engine: llama.cpp/candle/vLLM), `doctor`/`env`, `device`, `run`.
- **Cross-engine interop scaffold** (`hipgraph_bridge.interop`): `migraphx_available()`,
  `cross_engine_support()`, `hipgraph_interposer_status()`, and a clearly-experimental
  `MIGraphXBackend` stub (raises until routing lands) — the framework-agnostic hipGraph-interposer +
  MIGraphX backend are the documented roadmap; diagnostics are universal today via the CLI.
- **Diagnostics — always-on, bilingual (English / 中文) HIP/ROCm error & status reporting**
  (`hipgraph_bridge.diagnostics`, exported from `gfxgraph`). Translates terse ROCm errors into
  **cause + gfx1030/RDNA2 context + concrete fix** for confused ROCm users. Works whether or not
  CUDA-graphs are active (fills the gap where GUARD — graph-path only — stayed silent).
  - `explain(err)`, `report(err, context=...)`, `diagnose(context)` context manager,
    `install_diagnostics()` (always-on `sys.excepthook`; auto-installed when `GFXGRAPH=1` or
    `GFXGRAPH_DIAG=1`), and `environment_report()` (HSA override / arch / free VRAM / graph state).
  - 8 diagnoses grounded in real gfx1030 failures: `no_kernel_image` (AOTriton/FA3 no gfx1030
    image), `out_of_memory` (rq3/tq3 codec OOM at high mem-fraction), `illegal_address` (CUDA-graph
    stale buffer), `bf16_unsupported`, `wrong_arch`, `wave64_ignored`, `aiter_on_rdna`,
    `invalid_configuration`.
  - **Chinese (中文) toggle** via `GFXGRAPH_LANG=zh`; translations live in a **separate** lazily-
    loaded `hipgraph_bridge/diag_zh.py` so the English module stays clean (zero cost for en users).
- **Wavefront — capture wave64/128 + plan the software-wave conversion** (`hipgraph_bridge.wavefront`):
  `detect_wave64()` captures 64/128-lane intent (compile flags / `warpSize==64` / log lines);
  `plan_software_wave(rows, 64|128)` returns the gang-W-Wave32 + LDS-merge launch plan (W=2 ≈ wave64,
  W=4 ≈ wave128), adaptive to grid occupancy. NB: this is detection + a conversion *plan*, not a
  runtime kernel rewrite (ROCm fixes wavefront size at compile time).
- Chinese user guide: `docs/GUIDE_zh.md`.
- Publisher headers (Carlos Fundora · @carlosfundora) across new source files.

### Changed
- Packaging readied for PyPI (`uv pip install gfxgraph`): keywords, GPU + Chinese classifiers,
  description updated. Pure-Python; native bridge remains the optional `gfxgraph-native` companion.

## [0.4.0] - 2026-06-14

### Added
- **GUARD — 3-tier illegal-memory-access safety** (`hipgraph_bridge.guard`, exported from
  `gfxgraph`; activate with `GFXGRAPH_GUARD=1|2|3`, default off → zero overhead). For ROCm users
  unfamiliar with CUDA-graph rules:
  - **Tier 1 — auto-safe-capture**: `make_safe()` / `make_capture_safe()` force tensors entering
    capture/replay to be contiguous and own their storage (fixes non-contiguous, broadcast/0-stride,
    negative-stride views); `validate_layout()` reports issues. Auto-corrects the entire
    *capture-safety* fault family (accesses that fail only because a captured buffer was strided or
    its address wasn't stable). Wired into `BridgedCUDAGraph.replay()` input handling.
  - **Tier 2 — fault localization**: a hard `hipErrorIllegalAddress` (would-be SIGSEGV) becomes a
    precise, catchable `GfxGraphFault` carrying the op + every involved tensor's layout, then a
    graceful eager fallback. Wired into the graph-replay and eager-fallback paths so an in-kernel OOB
    (a producing-code logic bug) is *diagnosable* instead of crashing the process.
  - **Tier 3 — deep guard (opt-in)**: `RedZone` sentinel-padded buffers detect OOB writes past
    gfxGRAPH-owned buffers; `apply_deep_guard_env()` disables the caching allocator so faults land at
    real allocation boundaries; `compute_sanitizer_cmd()` wraps a command in compute-sanitizer /
    rocm-memcheck to pin the exact offending op.
  - 31 CPU-runnable unit tests (`tests/test_guard.py`).

## [Unreleased]

### Fixed
- **Packaging**: Restored the base `gfxgraph` package to a true pure-Python install (Tier 1). The 0.3.4 `setup.py`/`setuptools-rust` consolidation (below) had re-coupled the base install to a Rust + ROCm toolchain, contradicting the documented Tier 1 design and the `native/` companion that already builds `libhipgraph_bridge.so`. Native acceleration is again opt-in via the `native/` companion and source `rust/` crates; the runtime degrades gracefully when absent (`_HAS_*` guards).

## [0.3.4] - 2026-06-12

### Added
- Integrated pure-Rust companion crates `rs_gfxgraph_core` and `rs_gfxgraph_toolbox` into the `rust/` directory, stripping out irrelevant "federation" terminology.

### Fixed
- **Packaging/Build**: Fully consolidated the CMake and Rust build pipelines into a single setuptools `setup.py` that builds both the `libhipgraph_bridge.so` native bridge and the Rust extensions using `setuptools-rust`. Fixed `MANIFEST.in` to include headers and Rust source.
- **Memory Safety**: Restored dedicated `torch.cuda.graph_pool_handle()` usage in `ShapeBucketPool` and `ConditionalGraph` when capture is explicitly enabled. On ROCm/HIP, Python bridge capture now fails closed by default to avoid process-level segmentation faults observed before Python can catch an exception; low-level callers receive `RuntimeError` instead of entering the unsafe PyTorch capture path.
- **Rust/Python Parity**: Aligned the Rust `BucketRouter` signature to return `(-1, 2)` for oversized inputs instead of raising `ValueError`, ensuring exact parity with the Python router fallback.
- **Structural Fallback Metrics**: `fallback_count` is now accurately bumped on structural transitions (when a branch/bucket fails for the first time) rather than incorrectly inflating on every eager replay iteration.
- **Strict Typing**: Replaced duck-typed tensor checking (`is_cuda`) in the Rust bridge with strict PyO3 type verification (`is_instance`).
- **Exception Forwarding**: The Rust PyO3 interoperability layer now forwards Python exception types, values, and tracebacks into the `torch.no_grad.__exit__` context block.

## [0.3.2] - 2026-06-12

### Added
- Integrated `deepspeed-hip` kernels (fused layer norm, RMS norm, tiled linear) explicitly optimized for RDNA2.
- Integrated Triton RDNA2 kernels into `kernels/rdna2`.

### Fixed
- **Shape Pool Propagation**: Fixed data propagation so inputs actually flow into the shape-pool during replay.
- **Timing & Performance**: Added explicit `torch.cuda.synchronize()` calls before and after graph replay to accurately capture GPU time.
- **Memory Safety**: Resolved a cross-branch aliasing crash in `ConditionalGraph` by properly allocating dedicated memory pools per branch (`torch.cuda.graph_pool_handle()`).
- **CI Security**: Guarded self-hosted runner pipelines against execution from fork PRs.
- **Packaging**: Deleted `setup.py` so the root project can be correctly installed as pure-Python.

## [0.3.1] - 2026-05-17

### Added
- Introduced a root `SECURITY.md` with vulnerability reporting and supported version policy.
- Added benchmark provenance fields for public benchmark output: commit SHA, ROCm runtime/driver hints, tracked environment variables, and repeated-run metadata.
- Added PyPI-facing metadata in `pyproject.toml` (`project.urls` and standard classifiers).

### Changed
- Aligned CI workflows to Python 3.12 to match the package runtime requirement.
- Switched GPU-unavailable integration tests to explicit `pytest.skip(...)` signaling instead of print-and-return behavior.

### Removed
- Closed the stale PR #16 tracking branch (`rusty-stats-refactor-9436386981500045669`) to reduce release confusion.
