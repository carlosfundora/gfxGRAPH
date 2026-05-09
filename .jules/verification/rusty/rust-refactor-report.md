# Rusty Rust Refactor Report

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
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
