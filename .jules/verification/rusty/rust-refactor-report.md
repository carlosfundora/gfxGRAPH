# Rusty Rust Refactor Report

## Candidate Ranking

| Rank | Candidate | Current Runtime | Expected Benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|---|
| 1 | `python/gfxgraph/_enable.py` `_stats` | Python | Performance (hot path threading lock) | Low | Low | Selected |
| 2 | `python/hipgraph_bridge/shape_bucketing.py` `select_bucket` | Python | Performance (hot loop graph routing) | Low | Low | Rejected (already implemented) |
| 3 | `python/hipgraph_bridge/graph_manager.py` | Python | Performance (less overhead per graph launch) | High | Medium | Rejected |
| 4 | `python/hipgraph_bridge/conditional.py` | Python | Performance | Low | Low | Rejected |
| 5 | `python/hipgraph_bridge/ops.py` | Python | Minimal | Low | Low | Rejected |

## Selected Candidate

- Path: `python/gfxgraph/_enable.py` (specifically `_stats` tracking logic)
- Current implementation: Pure Python `dict` with `threading.Lock()` per bump and replay tracking call.
- Rust replacement: `gfxgraph_rs.StatsManager` implemented via PyO3 utilizing native Rust `std::sync::Mutex` and `HashMap`.
- Reason selected: The stats tracking logic is executed on every model graph replay and capture. Converting it to Rust significantly reduces the Python execution overhead and lock contention.

## Implementation Summary

Extended the existing PyO3 Rust crate `gfxgraph_rs`. It exposes a new `StatsManager` class which holds a Mutex and HashMap natively and exposes `bump`, `record_replay_us`, `set_enabled_at`, and `stats` methods. Modified the Python `gfxgraph/_enable.py` module to conditionally utilize the Rust implementation while keeping the Python threading lock implementation as a fallback.

## Before Benchmark

```json
{
  "candidate": "python/gfxgraph/_enable.py",
  "implementation": "before",
  "command": "python benchmark_stats.py before",
  "timestamp": "2026-05-08T20:26:06.113168",
  "iterations": 1000000,
  "input_description": "Repeated stats increments for stats tracking",
  "duration_ms": 2104.83,
  "throughput": 475097.75,
  "notes": "Records stats based on Python lock dict"
}
```

## After Benchmark

```json
{
  "candidate": "python/gfxgraph/_enable.py",
  "implementation": "after",
  "command": "python benchmark_stats.py after",
  "timestamp": "2026-05-08T20:37:35.010390",
  "iterations": 1000000,
  "input_description": "Repeated stats increments for stats tracking",
  "duration_ms": 308.92,
  "throughput": 3237084.03,
  "notes": "Records stats based on Rust StatsManager"
}
```

## Benchmark Delta

Execution time decreased from 2104.83ms to 308.92ms (-85.3%), increasing throughput significantly.

## Tests Run

- `tests/test_gfxgraph_rs.py` -> PASSED
- `tests/test_torch_integration.py` -> PASSED
- `tests/test_graph_manager.py` -> PASSED

## Files Changed

- `.jules/verification/rusty/before-benchmark.json`
- `.jules/verification/rusty/after-benchmark.json`
- `.jules/verification/rusty/benchmark-summary.md`
- `.jules/verification/rusty/rust-refactor-report.md`
- `python/gfxgraph/_enable.py`
- `gfxgraph_rs/src/lib.rs`
- `gfxgraph_rs/src/stats.rs`

## Compatibility Notes

The Python module conditionally imports `StatsManager` from `gfxgraph_rs`. If the native component cannot be loaded, it gracefully falls back to the original pure Python `threading.Lock()` implementation. The JSON representations are backwards compatible.

## Remaining Follow-Ups

- Consider optimizing Rust Mutex overhead or mapping directly to native hardware atomic counters if extreme microsecond latency is required.
