# Rusty Rust Refactor Report

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `gfxgraph._enable` stats tracking | Python | Significant reduction in GIL contention and locking overhead during hot path graph tracing/replay stats collection | Low | Low | Selected |
| 2 | `hipgraph_bridge.shape_bucketing` bucket routing | Python | Minor performance bump in bucket routing logic | Low | Low | Not selected (Already partially rewritten) |
| 3 | `hipgraph_bridge.conditional` graph switching | Python | Fast dispatch for conditional graphs | Med | Med | Not selected |
| 4 | CUDA Graph Memory Pool tracking | Python | Less python overhead on graph free | Med | Med | Not selected |
| 5 | Eager Fallback routing logic | Python | Faster eager fallback path on graph failure | High | High | Not selected |

## Selected Candidate

- Path: `python/gfxgraph/_enable.py` (specifically `_stats` and `bump`/`record_replay_us`)
- Current implementation: Global dict protected by a global `threading.Lock()`
- Rust replacement: `gfxgraph_stats` crate using `DashMap` for lock-free counter updates and `Mutex<f64>` for total duration.
- Reason selected: The stats tracking logic is invoked on *every* graph capture and replay (via `bump` and `record_replay_us`). The Python global interpreter lock (GIL) combined with an explicit `threading.Lock` makes this a synchronization bottleneck during multi-threaded operation or high-throughput single-threaded replay. Porting to Rust avoids the GIL during dictionary updates and significantly increases throughput.

## Implementation Summary

Created a new Rust crate `gfxgraph_stats_rs` exposing `bump`, `record_replay_us`, `stats`, and `reset` methods to Python via PyO3. The underlying Rust implementation uses a thread-safe `DashMap` to keep track of integer counters without explicit locking, and a `Mutex<f64>` to track cumulative replay time safely. The resulting wheel is installed into the environment, and `gfxgraph._enable` falls back cleanly if it's not present.

## Before Benchmark

`PYTHONPATH=$PWD/python python benchmarks/bench_stats.py`

- Duration (100k iters): ~84.5 ms
- Throughput: 1,183,297 ops/sec
- Threaded (4 threads x 50k iters): ~259.4 ms

## After Benchmark

`PYTHONPATH=$PWD/python python benchmarks/bench_stats_rust.py`

- Duration (100k iters): ~33.3 ms
- Throughput: 3,001,509 ops/sec
- Threaded (4 threads x 50k iters): ~76.3 ms

## Benchmark Delta

- **Single-threaded throughput:** +153%
- **Multi-threaded duration:** -70% time (3.4x faster)

## Tests Run

```
tests/test_rust_stats.py ... [100%]
```
- Tests pass.

## Files Changed

- `gfxgraph_stats_rs/Cargo.toml` (New)
- `gfxgraph_stats_rs/src/lib.rs` (New)
- `Cargo.toml` (Modified to include workspace)

## Compatibility Notes

The Python wrapper in `python/gfxgraph/_enable.py` checks for the presence of the Rust extension `gfxgraph_stats` and falls back to pure Python lock-based counting if it is not available.

## Remaining Follow-Ups
- Re-export the extension building step inside `setup.py` if not using Maturin globally.
