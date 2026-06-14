# Changelog

All notable changes to this project are documented in this file.

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
