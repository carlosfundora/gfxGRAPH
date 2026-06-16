---
name: benchmark-validate
description: Validates and benchmarks ROCm-DS component ports against their CPU baselines
  (e.g., pandas vs hipDF). Use this to generate parity tests, verify correctness,
  and measure the performance speedups of GPU-accelerated code.
---


# Purpose

To execute and measure data science tasks across CPU and ROCm GPU backends to ensure both functional parity and measure performance acceleration.

# When to use

- After migrating a script to use ROCm-DS components (e.g., hipDF, hipRAFT) and needing to verify it produces the same results as the original.
- To quantify the wall-clock performance difference between CPU baseline and GPU accelerated scripts.

# Procedure

1. Read the provided baseline CPU script and the migrated ROCm-DS script.
2. Formulate a small test dataset if one is not provided.
3. Run the baseline CPU script and record the outputs and execution time.
4. Run the migrated ROCm script and record the outputs and execution time.
5. Verify functional parity between the two outputs (e.g., using `pandas.testing.assert_frame_equal`).
6. Calculate the acceleration factor (CPU_time / GPU_time).
7. Use the `examples/benchmark_report.md` template to generate the final report.

# Execution Support

*   **Helper Script:** `scripts/run_rocm_benchmark.sh`
    *   *Usage:* `bash scripts/run_rocm_benchmark.sh <cpu_script.py> <rocm_script.py>`
    *   This script acts as a basic wrapper to time execution and capture outputs.
*   **Report Template:** `examples/benchmark_report.md`

# Deliverables

- Output of parity check (Pass/Fail)
- Performance metrics (CPU time, GPU time, Speedup)
- Formatted `benchmark_report.md`
