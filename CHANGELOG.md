# Changelog

All notable changes to this project are documented in this file.

## [0.3.2] - 2026-06-12

### Added
- Integrated `deepspeed-hip` kernels (fused layer norm, RMS norm, tiled linear) explicitly optimized for RDNA2.
- Integrated Triton RDNA2 kernels into `kernels/rdna2`.
- Ported the lightweight pure-Rust federation crates (`rs_gfxgraph_core` and `rs_gfxgraph_toolbox`) to the hybrid branches to provide zero-cost architectural contracts.

### Fixed
- **Timing & Performance**: Added explicit `torch.cuda.synchronize()` calls before and after graph replay to accurately capture GPU time.
- **Eager Fallback Safety**: Implemented a graceful eager fallback for oversized input shapes by intercepting shape bucket boundaries, preventing `ValueError` panics that terminated SGLang.
- **Memory Safety**: Resolved a cross-branch aliasing crash in `ConditionalGraph` by properly allocating dedicated memory pools per branch (`torch.cuda.graph_pool_handle()`).
- **Rust/PyO3 Exception Handling**: The PyO3 interoperability layer now correctly extracts and forwards Python exception types, values, and tracebacks into the `torch.no_grad.__exit__` context block, preventing silence on failures.
- **Strict Typing**: Replaced unsafe duck-typed tensor checks (`is_cuda`) in the Rust bridge with strict `is_instance` PyO3 type verification.
- **Native Initialization Safety**: explicitly defined `argtypes` and `restype` for native C functions and deferred `ctypes` loading to prevent early initialization crashes.
- **Cold Starts**: Integrated a warmup cycle in `_enable.py` healthchecks before verification.

### Removed
- Removed the deprecated PyTorch C++ native sources (`src/` directory) to rely exclusively on the pure Rust bridge crates.

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
