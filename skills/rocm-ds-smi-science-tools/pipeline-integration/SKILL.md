---
name: pipeline-integration
description: Guidelines and patterns for integrating multiple ROCm-DS components (hipDF,
  hipVS, hipGRAPH, hipRAFT) into a cohesive, end-to-end data science pipeline on AMD
  GPUs. Use this skill when building multi-stage workflows that require passing data
  efficiently between different ROCm-DS tools.
---


# Purpose

This skill provides instructions on how to integrate disparate ROCm-DS components into a unified pipeline. It focuses on zero-copy data transfers and consistent environment usage when chaining tools like hipDF (tabular data), hipRAFT (clustering/primitives), and hipVS (vector search).

# When to use

- Use when you need to construct a multi-step data science workflow that spans multiple ROCm-DS libraries.
- Use when you need to ensure efficient data handoffs (avoiding GPU-to-CPU-to-GPU copies) between tools.

# When not to use

- Do not use when working with a single, isolated ROCm-DS component (refer to specific skills like `hipdf-pandas-port` or `hipraft-primitives` instead).
- Do not use for generic Python scripting unrelated to AMD GPU acceleration.

# Required inputs

- Repository path
- Relevant source files containing the pipeline logic
- Target environment facts (ROCm version, available GPUs)
- Desired outcome (e.g., "Build an end-to-end RAG pipeline using hipDF for ETL and hipVS for retrieval")

# Procedure

1.  **Identify Components**: Determine which ROCm-DS libraries are required for the pipeline (e.g., data loading with hipDF, feature extraction with hipRAFT, indexing with hipVS).
2.  **Plan Data Flow**: Map out how data will move between components. Prioritize keeping data on the GPU memory (`__device__` memory) across boundaries.
3.  **Implement Integration**:
    *   Use CuPy or DLPack as the intermediate representation when passing data between components that do not natively support direct handoffs.
    *   Ensure all components are initialized within the same ROCm stream or context if necessary to avoid synchronization bottlenecks.
4.  **Verify**: Test the pipeline with a small dataset to confirm data flows correctly and components interact without crashing.
5.  **Benchmark**: Validate the performance to ensure the integrated pipeline is faster than its CPU counterpart and doesn't suffer from excessive memory transfers.

# Deliverables

-   Concise summary of the pipeline architecture.
-   Files changed (pipeline scripts).
-   Validation and benchmarking results performed.
-   Known limitations or bottlenecks in the integration.
