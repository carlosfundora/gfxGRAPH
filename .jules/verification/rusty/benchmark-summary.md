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
