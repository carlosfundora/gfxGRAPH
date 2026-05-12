# Benchmark Summary

- Before command: `python benchmarks/bench_routing.py`
- After command: `python benchmarks/bench_routing_rust.py`
- Before timing: 430.36 ms
- After timing: 433.31 ms
- Percent change: +0.6%
- Notes: Evaluated the impact of consolidating `ShapeBucketPool` state management directly into the native PyO3 Rust `gfxgraph_rs` binary search iterator loop. The consolidation offsets the PyO3 serialization cost boundary crossing completely but lacks heavy math requirements necessary for Rust to definitively out-pace pure python object referencing mappings.
