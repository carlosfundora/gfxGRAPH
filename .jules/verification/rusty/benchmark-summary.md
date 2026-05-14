# Benchmark Summary

- **Candidate:** `hipgraph_bridge.graph_manager.BridgedCUDAGraph._maybe_validate`
- **Before Command:** `python benchmarks/bench_graph_manager.py`
- **After Command:** `python benchmarks/bench_graph_manager_rust.py`

## Results (Mocked GPU)
- **Before Duration (100k iters):** 34063.02 ms (2935.74 ops/sec)
- **After Duration (100k iters):** 17611.90 ms (5677.98 ops/sec)
- **Improvement:** 1.93x faster (93.4% increase in ops/sec)

## Notes
The `_maybe_validate` step of graph manager runs on the critical path of graph replay when validation mode is enabled. It involves a `no_grad` context manager, multiple torch operations, and error checks, adding significant overhead per graph replay. Re-writing this exact logic within a PyO3 Rust extension limits Python bytecode interpreter overhead, resulting in nearly doubling the validation throughput on small graphs.
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
