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
| 1 | `python/hipgraph_bridge/shape_bucketing.py` `select_bucket` | Python | Performance (hot loop graph routing) | Low | Low | Selected |
| 2 | `python/hipgraph_bridge/graph_manager.py` | Python | Performance (less overhead per graph launch) | High | Medium | Rejected |
| 3 | `python/hipgraph_bridge/conditional.py` | Python | Performance | Low | Low | Rejected |
| 4 | `python/gfxgraph/_enable.py` | Python | Startup time / Safety checks | Low | Low | Rejected |
| 5 | `python/hipgraph_bridge/ops.py` | Python | Minimal | Low | Low | Rejected |

## Selected Candidate

- Path: `python/hipgraph_bridge/shape_bucketing.py` (specifically `ShapeBucketPool.select_bucket`)
- Current implementation: Pure Python `bisect.bisect_left` per graph inference call.
- Rust replacement: `gfxgraph_rs.BucketSelector` implemented via PyO3 utilizing native Rust `binary_search`.
- Reason selected: The dynamic shape bucketing logic is executed in the critical path on every model forward pass. It scales with request rate. Converting it to Rust significantly reduces the Python execution overhead.

## Implementation Summary

Created a new PyO3 Rust crate `gfxgraph_rs` built with `maturin`. It exposes a `BucketSelector` class which holds the buckets array natively and exposes `select_bucket`. Modified the Python `ShapeBucketPool` class to lazily load and utilize the Rust implementation while keeping the eager Python implementation as a fallback if the library fails to import.

## Before Benchmark

```json
{
  "candidate": "python/hipgraph_bridge/shape_bucketing.py",
  "implementation": "before",
  "command": "python benchmark_bucketing.py before",
  "timestamp": "2026-05-08T20:26:06.113168",
  "iterations": 1200000,
  "input_description": "Repeated bucket selection for varied sizes",
  "duration_ms": 403.24,
  "throughput": 2975878.34,
  "notes": "Selects buckets based on bisect"
}
```

## After Benchmark

```json
{
  "candidate": "python/hipgraph_bridge/shape_bucketing.py",
  "implementation": "after",
  "command": "python benchmark_bucketing.py after",
  "timestamp": "2026-05-08T20:37:35.010390",
  "iterations": 1200000,
  "input_description": "Repeated bucket selection for varied sizes",
  "duration_ms": 196.60,
  "throughput": 6103651.41,
  "notes": "Selects buckets based on bisect"
}
```

## Benchmark Delta

Execution time decreased from 403.24ms to 196.60ms (-51.2%), increasing throughput significantly.

## Tests Run

- `tests/test_torch_integration.py` -> PASSED
- `tests/test_graph_manager.py` -> PASSED

## Files Changed

- `.jules/verification/rusty/before-benchmark.json`
- `.jules/verification/rusty/after-benchmark.json`
- `.jules/verification/rusty/benchmark-summary.md`
- `.jules/verification/rusty/rust-refactor-report.md`
- `python/hipgraph_bridge/shape_bucketing.py`
- `gfxgraph_rs/Cargo.toml`
- `gfxgraph_rs/src/lib.rs`

## Compatibility Notes

The Python module performs a conditional import (`try...except ImportError`) of `gfxgraph_rs`. If the native component cannot be loaded, it gracefully falls back to the original pure Python `bisect.bisect_left` implementation.

## Remaining Follow-Ups

- Optionally extend `gfxgraph_rs` to handle more stateful pool operations (`_warmed_up`, `_failed_buckets`).
- Integrate building the Rust library into the `pyproject.toml` workflow automatically.
