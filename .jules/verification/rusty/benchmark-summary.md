# Benchmark Summary

- **Candidate:** `gfxgraph._enable stats tracker`
- **Before Command:** `PYTHONPATH=$PWD/python python benchmarks/bench_stats.py`
- **After Command:** `PYTHONPATH=$PWD/python python benchmarks/bench_stats_rust.py`

## Results (Single Thread)
- **Before Duration:** ~84.5 ms (1,183,297 ops/sec)
- **After Duration:** ~33.3 ms (3,001,509 ops/sec)
- **Improvement:** ~2.5x faster (153% increase in ops/sec)

## Results (Threaded - 4x50k)
- **Before Duration:** ~259.4 ms
- **After Duration:** ~76.3 ms
- **Improvement:** ~3.4x faster (reduction in lock contention overhead)

## Notes
The pure Python implementation used `threading.Lock` which introduced significant overhead under load. The new Rust implementation uses `DashMap` and `Mutex` internally via PyO3 to provide thread-safe, high-performance counting.
