# Rusty Rust Refactor Report

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `hipgraph_bridge.conditional` graph execution | Python | Major latency reduction across branch checks. | Medium | Medium | Selected |
| 2 | `hipgraph_bridge.shape_bucketing` selection logic | Python | Minor latency reduction for bucket lookup (`bisect_left`). | Low | Medium | Rejected |
| 3 | `gfxgraph._enable` stats tracking (`bump`, `record_replay_us`) | Python | Concurrency speedup by removing GIL lock contention. | Low | Low | Rejected |
| 4 | Configuration loading/validation (`GFXGRAPH_VRAM_CAP`) | Python | Lower startup overhead. | Low | Low | Rejected |
| 5 | Validation comparison `torch.allclose` wrapper | Python | Negligible, still depends on torch. | Medium | High | Rejected |

## Selected Candidate

- Path: `python/hipgraph_bridge/conditional.py`
- Current implementation: Pure Python dictionaries mapping branches to graphs.
- Rust replacement: `gfxgraph_rs.ConditionalGraphRunner` implemented via PyO3 utilizing native Rust HashMap validation and state checks.
- Reason selected: The dynamic graph routing logic is executed in the critical path on every model forward pass conditionally. Converting the `run` method to Rust significantly reduces the Python execution overhead.

## Implementation Summary
Created a new `ConditionalGraphRunner` class inside `gfxgraph_rs` PyO3 rust library. `python/hipgraph_bridge/conditional.py` now populates and relies entirely on the rust runner to perform validation on tensor sizes, verify branches, execute `torch.no_grad()` fallbacks and execute the proper graph stream. The pure python logic remains as a fallback. Interior mutability is used via `std::sync::RwLock` to mutate the failed_branches array safely. Hard panics were removed in favor of `PyRuntimeError`. `torch.no_grad` context manager exit logic utilizes proper `PyErr` value extraction.

## Before Benchmark
`python benchmarks/bench_conditional_mock.py`
Duration: 63313.80 ms
Throughput: 3158.87 ops/sec

## After Benchmark
`python benchmarks/bench_conditional_mock.py`
Duration: 9294.43 ms
Throughput: 21518.25 ops/sec

## Benchmark Delta
Execution time decreased by ~85.3% (-54019.3ms), increasing throughput significantly.

## Tests Run
`pytest tests/` -> PASSED

## Files Changed
- `gfxgraph_rs/src/lib.rs`
- `python/hipgraph_bridge/conditional.py`
- `benchmarks/bench_conditional_mock.py`
- `.jules/verification/rusty/before-benchmark.json`
- `.jules/verification/rusty/after-benchmark.json`
- `.jules/verification/rusty/benchmark-summary.md`
- `.jules/verification/rusty/rust-refactor-report.md`

## Compatibility Notes
We kept the python implementation as a fallback. If the user doesn't install `gfxgraph_rs` or build fails, it will gracefully fall back to the old locking python logic.

## Remaining Follow-Ups
None.
