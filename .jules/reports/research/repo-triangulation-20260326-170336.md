# Comparative Repository Audit: gfxGRAPH vs AMDMIGraphX vs vLLM

## 1. Executive Technical Summary
This report analyzes three core pieces of the AMD/ROCm ML inference ecosystem: **gfxGRAPH** (the immediate drop-in parity bridge for RDNA2), **AMDMIGraphX** (the official, heavy-duty AMD graph compiler), and **vLLM** (a major consumer of CUDA graphs employing sophisticated batch bucketing and model serving architecture).

The analysis reveals:
* **gfxGRAPH** is a highly focused, surgical shim. It achieves high value via a Python monkey-patching approach over torch.cuda.CUDAGraph, relying on small custom HIP kernels to bridge gaps in consumer RDNA2 silicon.
* **AMDMIGraphX** is a heavy, robust C++ compiler infrastructure. It provides deep optimization and target-specific routing but suffers from a steep integration curve and heavy framework dependencies (rocBLAS, MIOpen, Protocol Buffers).
* **vLLM** has the most mature and battle-tested architecture for dynamic batching using CUDA graphs, managing multiple graph instances (buckets) seamlessly.

For our system, **vLLM's shape bucketing pattern is the most valuable architectural extraction**, while **gfxGRAPH** serves as a vital enabler for running these patterns on cost-effective AMD hardware.

## 2. Repository Targets & Assumptions
* **Target A:** `gfxGRAPH` (Current Repo) - Analyzed via local source. Assumed primary purpose is bridging PyTorch graph execution on RDNA2.
* **Target B:** `AMDMIGraphX` (Cloned) - The official AMD ML graph compiler. Assumed to represent the "enterprise/heavy" graph execution path.
* **Target C:** `vLLM` (Cloned) - Specifically its `vllm/compilation/` and CUDA graph execution pathways. Assumed to represent the ideal usage/consumption pattern of graph execution.

## 3. Per-Repo Deep Audit

### Repo A: gfxGRAPH

A. Core Architecture & Logic Flow
- Entrypoints: `gfxgraph.enable()` and `BridgedCUDAGraph`.
- Data Flow: Intercepts PyTorch graph APIs, manages conditional branches using custom HIP code or pure-Python eagerly.
- Orchestration Layers: Python monkey-patch layer over C++ HIP APIs.
- Execution Engine: Underlying `torch.cuda` or `libhipgraph_bridge.so`.
- Architecture Class: CLI-first with library core / Framework-bound application shell wrapper.
- Concurrency Model: Synchronous pipeline (inherits from PyTorch CUDA streams).

B. Functional Decomposition & “The Heart”
- Hot Path: `python/hipgraph_bridge/graph_manager.py` (capture and replay interception).
- Complexity Score: 7/10. Clean abstraction depth, relies heavily on monkey-patch side effects, but internal naming clarity is high.

C. Dependency & Health Audit
- Dependencies: Lean (`torch>=2.9`).
- Dependency Tree Shape: Flat and controlled.
- Health: Fast pulse, active commits for niche hardware gap closing. MIT License ensures high embedding potential.

D. Developer Experience & Integration
- Boilerplate: Minimal (`import gfxgraph; gfxgraph.enable()`).
- Setup Friction: Low (Python-only Tier 1).
- Internal API Style: Declarative patching.
- Tests: Unit tests specifically targeting hardware parity behaviors.
- Integration Surface: SDK entrypoint (monkey-patch hook).

E. Lock-in & Migration Risk
- Migration Risk: Low.
- "How difficult would it be to migrate away from this repo in 5 years?" -> Very easy. If native hardware support arrives, the import is simply deleted and standard `torch.cuda` resumes control.

### Repo B: AMDMIGraphX

A. Core Architecture & Logic Flow
- Entrypoints: C++ API `migraphx::program` and Python bindings.
- Data Flow: Parse ONNX -> Graph IR -> Target specific compilation (GPU/CPU) -> Execution.
- Orchestration Layers: C++ compiler passes.
- Execution Engine: Direct target-specific (MIOpen / rocBLAS kernels).
- Architecture Class: Monolith compiler.
- Concurrency Model: Threading / Synchronous execution.

B. Functional Decomposition & “The Heart”
- Hot Path: `src/targets/gpu/` and `src/compile.cpp`.
- Complexity Score: 9/10. Deep abstraction nesting, tightly coupled to AMD driver components, heavy framework entanglement.

C. Dependency & Health Audit
- Dependencies: Framework-heavy (`rocm-cmake`, `miopen`, `rocblas`, `protobuf`, `half`, `sqlite3`).
- Dependency Tree Shape: Deep and fragile (relies on specific ROCm toolkit versions).
- Health: Stable enterprise pulse. MIT License.

D. Developer Experience & Integration
- Boilerplate: Extremely high (CMake dependencies, specific rbuild environments).
- Setup Friction: High.
- Internal API Style: Imperative C++.
- Tests: Heavy C++ Catch2 tests.
- Integration Surface: C++ SDK and Python bindings.

E. Lock-in & Migration Risk
- Migration Risk: Severe.
- "How difficult would it be to migrate away from this repo in 5 years?" -> Extremely difficult. Tying inference to this compiler means tying model preparation, compilation pipelines, and deployment environments strictly to AMD's specific IR and tooling.

### Repo C: vLLM

A. Core Architecture & Logic Flow
- Entrypoints: API server or `LLMEngine`.
- Data Flow: HTTP Request -> Async Engine -> Worker -> CUDA Graph replay or eager model forward.
- Orchestration Layers: AsyncIO event loop (API) -> RPC (Workers).
- Execution Engine: PyTorch / Triton Custom Kernels.
- Architecture Class: Layered app / Modular monolith.
- Concurrency Model: Async/await event loop orchestrating worker queues and synchronous GPU operations.

B. Functional Decomposition & “The Heart”
- Hot Path: `vllm/compilation/cuda_graph.py` and `vllm/worker/model_runner.py`.
- Complexity Score: 8/10. High modularity, but deep nesting due to multi-platform compatibility and tensor parallelism abstractions.

C. Dependency & Health Audit
- Dependencies: `torch`, `xformers`, `triton`, `fastapi`, etc.
- Dependency Tree Shape: Moderately layered.
- Health: Massive community support. Apache 2.0 license.

D. Developer Experience & Integration
- Boilerplate: Moderate (requires configuring engines).
- Setup Friction: Moderate (complex python environment, simple API usage).
- Internal API Style: Composable / Stateful.
- Tests: Massive End-to-end and integration suites.
- Integration Surface: HTTP API, Python Server SDK.

E. Lock-in & Migration Risk
- Migration Risk: Moderate.
- "How difficult would it be to migrate away from this repo in 5 years?" -> Moderate. The abstraction around model serving is thick. However, the models themselves remain standard PyTorch, meaning the backend engine can be swapped with significant plumbing effort.

## 4. Feature Parity Table

| Feature | gfxGRAPH | AMDMIGraphX | vLLM |
|---|---|---|---|
| C++ Core Engine | Optional (Tier 2) | Yes | Yes (C++ ops) |
| Eager Fallback | Yes | No | Yes |
| Dynamic Shape Bucketing | Yes (via vLLM pattern) | No (static bounds) | Yes |
| Transparent PyTorch Hook | Yes | No | No |
| Hardware Abstraction | No (gfx1030 specific) | Yes (gfx8+) | Yes (via Triton/Torch) |
| Test Harness Depth | High | Medium | High |
| Config Layering | Env Vars | Env/API | Config Classes |
| Plugin System | No | No | No |

## 5. Comparative Trade-off Matrix

| Metric | gfxGRAPH | AMDMIGraphX | vLLM | Explanation |
|---|---|---|---|---|
| **Architectural Clarity** | 8 | 6 | 9 | vLLM abstracts complex systems well; MIGraphX is dense; gfxGRAPH is a clean shim. |
| **Maintainability** | 9 | 4 | 7 | gfxGRAPH is tiny; vLLM is huge but well-structured; MIGraphX is complex C++. |
| **Extensibility** | 6 | 8 | 9 | vLLM allows massive extension; MIGraphX has target plugins; gfxGRAPH is single-purpose. |
| **Performance Potential** | 9 | 10 | 9 | All target maximal GPU performance, MIGraphX at the lowest level. |
| **Dependency Risk** | 2 | 9 | 5 | MIGraphX ties deeply to ROCm versions; gfxGRAPH only to PyTorch. |
| **Migration Flexibility** | 8 | 2 | 6 | gfxGRAPH is a removable patch; MIGraphX locks into AMD IR. |
| **DX / Onboarding** | 9 | 3 | 7 | gfxGRAPH is `import gfxgraph.enable()`; MIGraphX requires `rbuild`. |
| **Test Trustworthiness** | 8 | 7 | 9 | vLLM has massive E2E CI; gfxGRAPH unit tests specific hardware behaviors. |
| **Operational Maturity** | 6 | 9 | 9 | vLLM and MIGraphX are production staples. |
| **Integration Readiness** | 9 | 3 | 6 | gfxGRAPH is ready instantly; vLLM is moderate; MIGraphX takes weeks. |
| **Licensing Suitability** | 10 | 10 | 10 | All MIT / Apache 2.0. |

## 6. Integration Opportunity Mapping
* **Opportunity 1:** Extract vLLM's `BatchDescriptor` and `BatchBucketPool` (from `vllm/compilation/cuda_graph.py`) into our system to handle dynamic input sizes efficiently. *Medium Difficulty, Adopt directly.*
* **Opportunity 2:** Utilize gfxGRAPH's `_CaptureContext.__exit__` cleanup pattern to prevent SIGABRTs during failed HIP captures. *Low Difficulty, Fast Win.*
* **Opportunity 3:** Avoid AMDMIGraphX's heavy C++ compilation pipeline unless deploying static edge models. *Attractive Trap.*

## 7. Adoption Plan
1. **Foundation:** Implement a lightweight `ShapeBucketPool` inspired by vLLM in our orchestration layer.
2. **Adapter:** Wrap `torch.cuda.CUDAGraph` usages behind an interface that gracefully handles capture failures (the gfxGRAPH eager fallback pattern).
3. **Execution:** Deploy gfxGRAPH only as an environment-level dependency on AMD hardware, keeping our core codebase hardware-agnostic.

## 8. Concrete Work Items
1. **Ticket 1: GraphCaptureManager implementation**
   - Purpose: Create a context manager that intercepts graph capture failures.
   - Affected Area: AI Execution Engine.
   - Dependency Order: 1.
   - Risk Level: Low.
   - Acceptance Criteria: A capture failure does not crash the app, but falls back to eager execution.
2. **Ticket 2: ShapeBucketing for dynamic batches**
   - Purpose: Port padding and bucket selection logic from vLLM.
   - Affected Area: AI Memory Manager.
   - Dependency Order: 2.
   - Risk Level: Medium.
   - Acceptance Criteria: Tensor inputs map to the nearest power-of-two graph bucket.
3. **Ticket 3: Integrate AMD CI Pipeline**
   - Purpose: Add a GitHub Action to test graph logic on AMD hardware using gfxGRAPH.
   - Affected Area: `.github/workflows`.
   - Dependency Order: 3.
   - Risk Level: Low.
   - Acceptance Criteria: CI passes with `GFXGRAPH=1`.

* **First PR:** Introduce the safe `GraphCaptureManager` (Ticket 1). Safest high-leverage foundation to prevent graph-related crashes.
* **Second PR:** Implement `ShapeBucketing` (Ticket 2) utilizing the new capture manager.
* **What not to do:** Do not embed heavy C++ graph compilers (like MIGraphX) directly into our core Python app. Do not tightly couple to `torch.cuda` without a shim.

## 9. Final Recommendation
* **Best to Adopt Directly:** gfxGRAPH (as a transparent environmental dependency).
* **Best to Mine for Ideas:** vLLM (for its serving architecture and graph bucketing).
* **Best to Avoid:** AMDMIGraphX (too heavy, severe lock-in).

## 10. Horizon Scanning
1. **The Rising Star:** `SGLang`
   - Selected for: High-performance inference engine.
   - Category: Rapid Execution / Serving.
   - Technical Reason: RadixAttention and fast prefix matching are highly relevant to our graph usage optimizations.
   - Why no deep dive: Heavily overlaps with vLLM's space; focus was on core graph generation (MIGraphX/PyTorch).
2. **The Legacy Standard:** `ONNXRuntime`
   - Selected for: Ubiquity in ML deployment.
   - Category: Enterprise Integration.
   - Technical Reason: The cross-platform alternative to MIGraphX, providing similar graph compilation with broader hardware support.
   - Why no deep dive: Well understood, doesn't address the specific AMD RDNA2 gaps that gfxGRAPH targets.
3. **The Niche Specialist:** `TensorRT-LLM`
   - Selected for: Nvidia's highly optimized stack.
   - Category: Vendor-specific Performance.
   - Technical Reason: Nvidia's proprietary graph executor, representing the ceiling of what hardware-specific optimization looks like.
   - Why no deep dive: Not relevant to the open ROCm/AMD hardware constraints we are currently auditing.

## 11. Appendix: Evidence Notes
* gfxGRAPH's core value is its monkey-patching of PyTorch (seen in `gfxgraph._enable.py` and `bridge.py` specs).
* AMDMIGraphX requires `rbuild` and deep ROCm dependencies (seen in its `CMakeLists.txt` and `README.md`).
* vLLM's graph management is highly sophisticated, maintaining a `compilation_counter` and `CUDAGraphMode` configs (seen in `vllm/compilation/cuda_graph.py`).
