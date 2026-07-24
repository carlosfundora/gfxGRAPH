---
name: rocm-ds-smi-science-tools
description: Parent overview skill for ROCm Data Science (ROCm-DS) components and
  SMI tools. Provides guidance on routing workloads, compatibility triage, and component-specific
  tasks like hipDF, hipVS, and hipMM.
---


# ROCm-DS SMI Science Tools Overview

This is the parent skill for managing tasks related to ROCm Data Science (ROCm-DS) and AMD GPU-accelerated scientific computing tools.

## When to use this skill
Use this skill to determine the appropriate nested skill to invoke when working with ROCm-DS components, porting data science workflows to ROCm, or handling installation and compatibility.

## Nested Skills Directory

- **`compatibility-triage`**: Verifies whether a requested ROCm-DS workflow is officially supported, source-build feasible, or experimental on the target system.
- **`hipdf-pandas-port`**: Migrates pandas-like workflows to hipDF and cudf.pandas style acceleration, auditing for unsupported features.
- **`hipgraph-analytics`**: Leverages GPU acceleration to process and analyze complex graph structures using hipGRAPH.
- **`hipmm-memory-ops`**: Utilizes the HIP Memory Manager (hipMM) for advanced GPU memory pooling, efficient allocation, and data movement.
- **`hipraft-primitives`**: Utilizes hipRAFT for foundational, reusable GPU-accelerated primitives like clustering, dimensionality reduction, and statistical operations.
- **`hipvs-ann`**: Selects, builds, benchmarks, and validates hipVS ANN indexes and query paths for ROCm-DS workloads.
- **`install-build`**: Handles installation and source compilation of ROCm-DS components via conda, PyPI, or source builds.
- **`pipeline-integration`**: Integrates ROCm-DS components into broader data science pipelines.
- **`benchmark-validate`**: Validates and benchmarks ROCm-DS component ports against their CPU baselines (e.g., pandas vs hipDF).
- **`tool-router`**: Classifies and routes tasks to the appropriate ROCm-DS component (hipDF, hipVS, hipGRAPH, hipRAFT, hipMM) based on workload requirements.
