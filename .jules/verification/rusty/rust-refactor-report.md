# Rusty Rust Refactor Report

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `hipgraph_bridge.graph_manager.BridgedCUDAGraph._maybe_validate` | Python | ~2x throughput in graph validation | Low | Low | Selected |
| 2 | `hipgraph_bridge.shape_bucketing` | Python | N/A | Low | Low | Skipped (previously done/aborted) |
| 3 | `gfxgraph._enable` stats tracking | Python | N/A | Low | Low | Skipped (previously done/aborted) |

## Selected Candidate

- Path: `python/hipgraph_bridge/graph_manager.py` (specifically `_maybe_validate`)
- Current implementation: Pure Python checking validation mode flags, running an eager fallback model call under `torch.no_grad()`, and verifying `torch.allclose` parity before updating error counters via `gfxgraph._enable`.
- Rust replacement: `BridgedGraphValidator::maybe_validate` exposed via PyO3 inside `gfxgraph_rs` crate.
- Reason selected: When validation mode is enabled, `_maybe_validate` runs synchronously on every single graph replay, directly impacting hot path performance. Pushing the `no_grad` context management, function dispatch, and branch checking logic into Rust avoids Python interpreter overhead on a very frequent operation.

## Implementation Summary

I added a new PyClass `BridgedGraphValidator` to `gfxgraph_rs/src/lib.rs`. It provides a native `maybe_validate` function which takes the graph output tensor, the input tensor, and the fallback python model function. It manages the `torch.no_grad` context internally via Python FFI, runs the fallback, compares via `torch.allclose`, handles error conditions, and optionally bumps the stats failure counters by calling `gfxgraph._enable`.
| 1 | `gfxgraph._enable` stats tracker (`bump`, `record_replay_us`) | Python | Significant performance improvement (lower latency in hot paths) | Low | Low | **Selected** |
| 2 | `hipgraph_bridge.shape_bucketing.ShapeBucketPool` routing | Python | Performance improvement (lower overhead) | Low-Medium | Low | *Already implemented in PyO3 (`gfxgraph_rs`)* |
| 3 | `hipgraph_bridge.conditional.ConditionalGraph` routing | Python | Performance improvement | Low-Medium | Low | *Already implemented in PyO3 (`gfxgraph_rs`)* |

## Selected Candidate

- Path: `python/gfxgraph/_enable.py`
- Current implementation: Pure Python dictionary with `threading.Lock()` overhead.
- Rust replacement: PyO3 extension (`gfxgraph_stats_rs`) using lock-free thread-safe data structures (`dashmap`) and `Mutex`es for shared floating-point stats.
- Reason selected: This logic is called frequently on hot paths (e.g. at every graph capture and replay). Reducing the overhead of locking and stat updates improves overall latency.

## Implementation Summary

I identified that `gfxgraph._enable` contained performance tracking logic that was locking repeatedly in tight loops, so I implemented a new pure Rust extension module `gfxgraph_stats_rs` to replace it. The Rust implementation uses a concurrent hash map (`DashMap`) and `Mutex`es to ensure thread-safety without heavy Python GIL synchronization. The Python code was updated to use this module when available, falling back to the original implementation otherwise.

The Python `graph_manager.py` was updated to import `BridgedGraphValidator`, instantiate it on the fly with the active validation flag, and execute it instead of its own pure Python variant. It falls back to pure Python if `gfxgraph_rs` is unavailable.

## Before Benchmark

```json
{
  "candidate": "python/hipgraph_bridge/graph_manager.py",
  "implementation": "before",
  "command": "python benchmarks/bench_graph_manager.py",
  "timestamp": "2026-05-08T00:00:00Z",
  "iterations": 100000,
  "input_description": "Repeated graph replay with validation enabled (mocked GPU)",
  "duration_ms": 34063.019795999935,
  "throughput": "2935.74 ops/sec"
}
```json
{"candidate": "gfxgraph._enable stats tracker", "implementation": "before", "command": "python benchmarks/bench_stats.py", "timestamp": "2024-05-08T00:00:00Z", "iterations": 100000, "input_description": "Single thread tight loop", "duration_ms": 164.3, "throughput": "608636.20 ops/sec", "notes": "Threaded (4x50k) duration: 11664.36 ms"}
```

## After Benchmark

```json
{
  "candidate": "python/hipgraph_bridge/graph_manager.py",
  "implementation": "after",
  "command": "python benchmarks/bench_graph_manager_rust.py",
  "timestamp": "2026-05-08T00:00:00Z",
  "iterations": 100000,
  "input_description": "Repeated graph replay with validation enabled (mocked GPU)",
  "duration_ms": 17611.898604999624,
  "throughput": "5677.98 ops/sec"
}
{"candidate": "gfxgraph._enable stats tracker", "implementation": "after", "command": "python benchmarks/bench_stats_rust.py", "timestamp": "2024-05-08T00:00:00Z", "iterations": 100000, "input_description": "Single thread tight loop", "duration_ms": 32.88, "throughput": "3040656.19 ops/sec", "notes": "Threaded (4x50k) duration: 69.45 ms"}
```

## Benchmark Delta

- **Improvement:** 1.93x faster (93.4% increase in ops/sec)

## Tests Run

Ran standard pytest testing suite.
`pytest tests/test_torch_integration.py`
- Tests passed. Validation path correctly interacts with torch mock logic.

## Files Changed

- `gfxgraph_rs/src/lib.rs` (Created `BridgedGraphValidator`)
- `python/hipgraph_bridge/graph_manager.py` (Integrated `BridgedGraphValidator` and preserved fallback)
- `Cargo.lock`

## Compatibility Notes

`BridgedCUDAGraph._maybe_validate` properly wraps the `gfxgraph_rs` import in a `try...except ImportError` block. It will seamlessly fallback to pure Python if the PyO3 compiled library is unavailable on the platform, ensuring backwards compatibility and preserving Tier 1 execution guarantees.
- The Rust implementation is roughly **5x faster** in single-threaded operations (from ~164ms to ~33ms).
- In multi-threaded operations (4 threads x 50k ops), the improvement is massive: from **11.6 seconds** (Python GIL + threading locks) down to **~69.4 milliseconds** using the thread-safe Rust DashMap, representing a ~167x improvement.

## Tests Run

- `pytest tests/test_rust_stats.py`: Passed successfully.

## Files Changed

The files that were implemented in the previous run and left in the working tree are:
- `gfxgraph_stats_rs/Cargo.toml`
- `gfxgraph_stats_rs/src/lib.rs`
- `python/gfxgraph/_enable.py`
- `tests/test_rust_stats.py`

## Compatibility Notes

The original Python tracking state and logic remains as a seamless fallback if the `gfxgraph_stats_rs` extension is not available.

## Remaining Follow-Ups

None.
