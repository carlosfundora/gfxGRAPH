# Rusty Rust Refactor Report

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `gfxgraph._enable` stats tracking (`bump`, `record_replay_us`) | Python | Massive concurrency speedup by removing GIL lock contention. | Low | Low | Selected |
| 2 | `hipgraph_bridge.shape_bucketing` selection logic | Python | Minor latency reduction for bucket lookup (`bisect_left`). | Low | Medium | Rejected |
| 3 | `hipgraph_bridge.conditional` graph execution | Python | Minor latency reduction. | Low | Medium | Rejected |
| 4 | Configuration loading/validation (`GFXGRAPH_VRAM_CAP`) | Python | Lower startup overhead. | Low | Low | Rejected |
| 5 | Validation comparison `torch.allclose` wrapper | Python | Negligible, still depends on torch. | Medium | High | Rejected |

## Selected Candidate

- Path: `python/gfxgraph/_enable.py` (specifically `bump`, `record_replay_us`, `stats`, `_stats_lock`)
- Current implementation: Pure Python dictionaries wrapped in a `threading.Lock()`
- Rust replacement: `gfxgraph_stats` PyO3 extension using `std::sync::atomic` lock-free operations.
- Reason selected: Best combination of highest performance impact (removing GIL contention in a hot path) and lowest complexity/risk. SGLang uses threads heavily, and continuous stat updates block the entire process otherwise.

## Implementation Summary
Created a new Rust module `gfxgraph_stats` directly integrated into the root `pyproject.toml` utilizing `setuptools_rust` and `pyo3` to expose `bump`, `record_replay_us`, `stats`, and `reset` methods lock-free using `std::sync::atomic` and `std::sync::RwLock` for dynamic keys. Modified `python/gfxgraph/_enable.py` to optionally import and use the rust implementation, falling back to python if unavailable.

## Before Benchmark
100k single-threaded iterations: 158.41 ms
4 threads * 50k iterations: 11344.69 ms

## After Benchmark
100k single-threaded iterations: 22.66 ms
4 threads * 50k iterations: 50.40 ms

## Benchmark Delta
Single thread: 7x throughput increase.
Multi thread: 225x throughput increase.

## Tests Run
Custom benchmarks. Implemented pytest coverage in `tests/test_rust_stats.py` validating correct behavior over standard and dynamic keys. Verified backwards compatibility fallback logic.

## Files Changed
- `python/gfxgraph/_enable.py`
- `pyproject.toml`
- `src/lib.rs`
- `Cargo.toml`
- `tests/test_rust_stats.py`
- `benchmarks/bench_stats.py`
- `benchmarks/bench_stats_rust.py`

## Compatibility Notes
We kept the python implementation as a fallback. If the user doesn't install `gfxgraph_stats` or build fails, it will gracefully fall back to the old locking python logic.

## Remaining Follow-Ups
None.
