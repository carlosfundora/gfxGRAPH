# Rusty Rust Refactor Report

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
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

## Before Benchmark

```json
{"candidate": "gfxgraph._enable stats tracker", "implementation": "before", "command": "python benchmarks/bench_stats.py", "timestamp": "2024-05-08T00:00:00Z", "iterations": 100000, "input_description": "Single thread tight loop", "duration_ms": 164.3, "throughput": "608636.20 ops/sec", "notes": "Threaded (4x50k) duration: 11664.36 ms"}
```

## After Benchmark

```json
{"candidate": "gfxgraph._enable stats tracker", "implementation": "after", "command": "python benchmarks/bench_stats_rust.py", "timestamp": "2024-05-08T00:00:00Z", "iterations": 100000, "input_description": "Single thread tight loop", "duration_ms": 32.88, "throughput": "3040656.19 ops/sec", "notes": "Threaded (4x50k) duration: 69.45 ms"}
```

## Benchmark Delta

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
