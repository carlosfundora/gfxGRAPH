# Benchmark Summary

## Commands

- **Before**: `python benchmarks/bench_stats.py` (with PyO3 extension disabled via mock)
- **After**: `python benchmarks/bench_stats_rust.py` (using `gfxgraph_stats_rs` extension)

## Results

| Metric | Before (Python GIL/Threading Locks) | After (Rust PyO3 `DashMap`/`Mutex`) | Delta |
|---|---|---|---|
| Single Thread Throughput | ~600,000 ops/sec | ~3,000,000 ops/sec | **~5x improvement** |
| Single Thread Duration (100k) | ~164 ms | ~33 ms | |
| Multi-Thread Duration (4x 50k ops) | ~11,600 ms | ~69 ms | **~168x improvement** |

## Notes
The massive improvement in multi-threading is due to the removal of Python's heavy `threading.Lock()` overhead and GIL contention, replacing it with the highly concurrent lock-free `dashmap` crate for tracking string-indexed counters and simple `std::sync::Mutex` wrapping floating point logic.
