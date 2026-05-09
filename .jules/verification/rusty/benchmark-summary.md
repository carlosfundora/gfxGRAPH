# Stats Tracker Benchmark Summary

## Commands
* **Before**: `python benchmarks/bench_stats.py`
* **After**: `python benchmarks/bench_stats_rust.py`

## Results
* **Before timing (single thread, 100k iters)**: 158.41 ms (631,286 ops/sec)
* **After timing (single thread, 100k iters)**: 22.66 ms (4,413,674 ops/sec)
* **Before timing (4 threads, 50k iters each)**: 11344.69 ms
* **After timing (4 threads, 50k iters each)**: 50.40 ms

## Delta
* **Percent Change (single thread)**: ~7x throughput improvement (85% reduction in latency).
* **Percent Change (threaded)**: ~225x throughput improvement (99.5% reduction in latency).

## Notes
The Python threading lock `_stats_lock` introduced huge contention overhead when multiple threads were updating counters concurrently. The Rust implementation uses lock-free atomics (`AtomicUsize` and `AtomicU64`) and a compare-and-swap loop for the float, bypassing the GIL completely for updates and drastically reducing the synchronization bottleneck.
# Benchmark Summary

- Before command: `python3 benchmark_bucketing.py before`
- After command: `python3 benchmark_bucketing.py after`
- Before timing: 403.24 ms
- After timing: 196.60 ms
- Percent change: -51.2% (over 2x faster throughput)
- Notes on variance or limitations: The benchmark only tests the specific bucketing logic isolated from PyTorch overhead, making the performance gain clear and localized. PyTorch overhead in real use cases will dilute the end-to-end performance gain, but this confirms the Python-to-Rust migration reduces the hot-path latency.
