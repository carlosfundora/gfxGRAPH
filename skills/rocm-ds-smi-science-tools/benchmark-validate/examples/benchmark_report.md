# ROCm-DS Benchmark Report

## Overview
**Component:** (e.g., hipDF, hipRAFT)
**Task:** (e.g., Dataframe Join, KMeans Clustering)
**Date:** YYYY-MM-DD

## Scripts Tested
*   **CPU Baseline:** `path/to/cpu_baseline.py`
*   **ROCm Script:** `path/to/rocm_accelerated.py`

## Environment Facts
*   **OS:**
*   **GPU Model:**
*   **ROCm Version:**

## Parity Validation
*   **Status:** [PASS | FAIL]
*   **Notes:** (e.g., "pandas.testing.assert_frame_equal passed with rtol=1e-5. Minor floating point differences were observed but within acceptable tolerance.")

## Performance Results
| Metric | CPU Baseline | ROCm Accelerated |
| :--- | :--- | :--- |
| Execution Time (s) | XX.XX | YY.YY |
| Speedup Factor | 1.0x | ZZ.Zx |

## Limitations & Unresolved Issues
*   (List any unsupported features encountered, memory constraints, or limitations in the porting process)
