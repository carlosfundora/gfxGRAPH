---
name: gfxgraph-benchmarking
description: Benchmark gfxGRAPH internals, run the public benchmark suite, and compare performance between Python and Rust.
---

# gfxGRAPH Benchmarking

**Goal**
Run and analyze performance benchmarks on gfxGRAPH.

## When to use this skill
- When verifying the performance impact of modifications to the routing logic (e.g. conditional runner, shape bucketing).
- When validating the package's overall latency and throughput improvements.
- When generating performance report JSON files containing provenance tracking (e.g. commit SHA, ROCm parameters).

## How to use it
1. Use real hardware (e.g. AMD Radeon RX 6700 XT) under ROCm 7.2 when running GPU benchmarks.
2. For micro-benchmarking CPU/mock pathways, ensure standard mocks (like mocking `torch_cuda_execution_probe`) are set up.
3. Run the public benchmark script:
   ```bash
   PYTHONPATH=python python benchmarks/bench_readme_public.py --run-count 3 --output benchmarks/results/readme_benchmark_latest.json
   ```
4. Run python/rust micro-benchmarks:
   - Routing: `python benchmarks/bench_routing.py` vs. `python benchmarks/bench_routing_rust.py`
   - Conditional: `python benchmarks/bench_conditional_mock.py`

## Preferred checks
- Check latency output (`avg_replay_us`) in `gfxgraph.stats()`.
- Compare ops/sec throughput differences on dynamic-shape bucketed runs.

## Pass criteria
- Public benchmark completes successfully and writes a populated JSON output containing system provenance metrics (commit SHA, ROCm runtime, etc.).
- Performance regression checking shows throughput remains within target bounds (e.g. >= 90% static graph replay for shape bucketing, >= 85% for conditional branching).
