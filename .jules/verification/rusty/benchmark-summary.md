# Benchmark Summary

Before command: `python3 benchmarks/bench_conditional_mock.py` (Before refactor)
After command: `python3 benchmarks/bench_conditional_mock.py` (After refactor)

## Single Thread

Before timing: 63313.80 ms
After timing: 9294.43 ms

Percent change: (63313.80 - 9294.43) / 63313.80 * 100% = ~85.3% reduction in execution time

Notes: Migrating the `ConditionalGraph.run()` method inner-loop out of pure python dictionary and execution logic to rust provides an 85% speedup on mocked bounds checks/fallbacks.

- Before command: `python benchmarks/bench_routing.py`
- After command: `python benchmarks/bench_routing_rust.py`
- Before timing: 430.36 ms
- After timing: 433.31 ms
- Percent change: +0.6%
- Notes: Evaluated the impact of consolidating `ShapeBucketPool` state management directly into the native PyO3 Rust `gfxgraph_rs` binary search iterator loop. The consolidation offsets the PyO3 serialization cost boundary crossing completely but lacks heavy math requirements necessary for Rust to definitively out-pace pure python object referencing mappings.
