# rs_gfxgraph_core

Pure-Rust shared contracts for the **gfxGRAPH** library. Defines lightweight, high-performance DTOs, schemas, and enumerations without native bindings or external execution frameworks.

---

## Architectural Role in gfxGRAPH

```text
       ┌────────────────────────┐
       │   rs_gfxgraph_core     │◄─────── Pure-Rust contracts (zero-cost, fast compile)
       └───────────┬────────────┘
                   │
       ┌───────────┼───────────────┐
       ▼           ▼               ▼
┌──────────────┐ ┌───────────────┐ 
│gfxgraph_rs   │ │gfxgraph_stats │ 
│   _pyo3      │ │   _rs         │ 
│(PyO3 bindings│ │(Observability)│ 
└──────────────┘ └───────────────┘ 
```

### Architectural Analysis

1. **`rs_gfxgraph_core` (This Crate — Core Contract Layer)**:
   - **Characteristics**: Extremely lightweight, pure Rust, minimal external dependencies (only `serde`), and near-zero compile time.
   - **Role**: Defines the core schema models (`GfxGraphNodeSpec`), telemetry storage contracts (`GfxGraphStatsSample`), and routing enums (`GfxGraphAdapterKind`). A **pure contract layer**.
   - **Modularity Rationale**: Keeping this crate pure ensures that downstream Rust systems (database interfaces, CLI parsers, metadata pipelines) can serialize, deserialize, and reference these types without compiling heavy FFI or deep-learning runtimes.

2. **`gfxgraph_rs` (Native PyO3 Execution Crate)**:
   - **Characteristics**: Coupled tightly to the Python interpreter via `PyO3`.
   - **Role**: Contains the high-performance conditional graph runner and bucket router for deep-learning inference workloads.
   - **Why Separate**: Merging with core would destroy the lightweight contract nature, introducing complex PyO3 and native library linkage.

3. **`gfxgraph_stats_rs` (Observability)**:
   - **Role**: Collects live execution statistics and provides benchmark infrastructure.
   - **Why Separate**: Decoupled to keep execution telemetry completely separate from pure schema contracts.

---

## Features

- **Schema Contracts**: `GfxGraphNodeSpec` — graph node registry specifications.
- **Observability Models**: `GfxGraphStatsSample` — bucket performance telemetry.
- **Unified Error Handling**: `GfxGraphError`.

---

Last Updated: 2026-06-12

## Generic graph lifecycle contracts

The core also exports RAII capture/replay leases, bounded capture retry policy, capture-stable persistent metadata layouts, graph update modes, and validated child-graph composition plans. Native HIP execution remains in the native bridge; workload-specific state belongs in the consuming kernel crate.
