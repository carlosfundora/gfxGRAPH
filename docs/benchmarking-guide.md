# gfxGRAPH Benchmarking Guide

This guide details the benchmarking suite, script configurations, output formats, and how to verify execution performance on AMD ROCm (specifically RDNA2 hardware).

---

## 1. Native-Only Gate

The `rust-hip-cpp` branch must prove the native runtime path without importing the
Python package. Use this gate before changing runtime behavior:

```bash
cargo build -p rs_gfxgraph_native --bin gfxgraph-native-probe
cargo run -p rs_gfxgraph_bench -- \
  --repo-root . \
  --phase clean-candidate \
  --native-only \
  --run-id native-only-runtime-YYYYMMDD
```

This writes a schema-validated report under `benchmarks/results/YYYY-MM-DD/` with
`python: null`, `target_import_policy: native-only`, and a
`rust-native-runtime-cli` benchmark produced by `gfxgraph-native-probe`. The probe
loads `libhipgraph_bridge.so` directly, exercises lifecycle and profiler FFI, and
reports `python_used: false`.

Python package install/uninstall is only for compatibility baselines against the
published package. It is not the development target for this branch.

---

## 2. Micro-Benchmarks

Micro-benchmarks isolate specific components to compare the performance between the pure Python fallback logic and the PyO3 Rust extension modules.

### Shape Bucketing Router: `bench_routing.py` vs `bench_routing_rust.py`
- **File Location**: [benchmarks/bench_routing.py](file:///home/local/ai/projects/gfxGRAPH/benchmarks/bench_routing.py) and [benchmarks/bench_routing_rust.py](file:///home/local/ai/projects/gfxGRAPH/benchmarks/bench_routing_rust.py)
- **Goal**: Measure the overhead of selecting shape buckets over 1,000,000 routing iterations.
- **Pure Python Baseline**: Maps input size using Python's `bisect_left` and set lookup operations.
- **Rust Implementation**: Uses PyO3 bindings mapped to `std::collections::HashSet` and binary search arrays.
- **How to Run**:
  ```bash
  python benchmarks/bench_routing.py
  python benchmarks/bench_routing_rust.py
  ```

### Conditional Execution Mock: `bench_conditional_mock.py`
- **File Location**: [benchmarks/bench_conditional_mock.py](file:///home/local/ai/projects/gfxGRAPH/benchmarks/bench_conditional_mock.py)
- **Goal**: Measures the throughput (ops/sec) of alternating branch execution with a mocked GPU.
- **Mocks Used**: Mocks `torch` and overrides `torch_cuda_execution_probe` to simulate a contiguous CUDA environment on CPU, allowing the benchmark to execute fallbacks without hardware dependencies.
- **How to Run**:
  ```bash
  python benchmarks/bench_conditional_mock.py
  ```

---

## 3. Public GPU Benchmarks

The public benchmark suite is intended to run on a real GPU to verify overall latency and throughput improvements.

### Public Workload Benchmark: `bench_readme_public.py`
- **File Location**: [benchmarks/bench_readme_public.py](file:///home/local/ai/projects/gfxGRAPH/benchmarks/bench_readme_public.py)
- **Goal**: Measures execution duration (ms/iter) for core model layers (e.g. LayerNorm/GELU chains and MLPs) comparing direct Eager mode vs. Graph execution.
- **How to Run**:
  ```bash
  PYTHONPATH=python python benchmarks/bench_readme_public.py \
    --run-count 3 \
    --output benchmarks/results/readme_benchmark_latest.json
  ```
- **Optional Hot Replay Mode**: Enable via environment variable `GFXGRAPH_REPLAY_HOT_MODE=1` to use the optimized low-overhead native launcher path.

---

## 4. Output JSON Schema & Provenance

When running the public benchmark, results are saved to a JSON file containing provenance tracking metadata:

```json
{
  "provenance": {
    "commit_sha": "abc123def...",
    "rocm_version": "7.2.0",
    "gpu_target": "gfx1030",
    "timestamp": "2026-06-16T08:00:00Z"
  },
  "results": [
    {
      "workload": "mlp_bs32_d1024",
      "eager_ms_per_iter": 0.1023,
      "graph_ms_per_iter": 0.1028,
      "speedup": 1.00
    }
  ]
}
```

---

## 5. Targets and Performance Gates

| Benchmark Component | Baseline Reference | Target Performance |
|---------------------|--------------------|--------------------|
| Conditional 2-branch| Native HIP (no cond) | $\ge$ 85% throughput |
| Pipeline launch     | `hipGraphLaunch` (cold) | $\le$ 50 $\mu$s latency |
| Shape bucketing     | Static graph replay | $\ge$ 90% throughput |
| Composition         | Monolithic graph | $\ge$ 90% throughput |
