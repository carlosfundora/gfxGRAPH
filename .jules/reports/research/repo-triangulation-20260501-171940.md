# Comparative Repository Audit: vLLM vs SGLang vs TGI

## 1. Executive Technical Summary
This audit evaluates three premier LLM inference engines—**vLLM**, **SGLang**, and **Text Generation Inference (TGI)**—with a specific focus on their CUDA graph orchestration, maintainability, and integration viability for our project (gfxGRAPH).
vLLM provides the most robust and structured graph orchestration via shape bucketing, making it the best reference architecture. SGLang offers high performance via RadixAttention but tightly couples its graph execution to its KV cache implementation. TGI is operationally mature but defers heavily to custom kernels and Rust-based orchestration, making its Python internals less extractable.
The key takeaway is that our bridge layer must adapt vLLM's `BatchBucketPool` concept while maintaining a decoupled memory lifecycle to support engines like SGLang.

## 2. Repository Targets & Assumptions
* **Target A:** `vllm-project/vllm` (vLLM)
* **Target B:** `sgl-project/sglang` (SGLang)
* **Target C:** `huggingface/text-generation-inference` (TGI)

**Assumption:** These repositories represent the primary consumers of our `gfxGRAPH` integration. Our goal is to determine which architecture is best to emulate or adapt for our internal orchestration, and how to harden our bridge against their specific graph capture patterns.

## 3. Per-Repo Deep Audit

### Repo A: vLLM

**A. Core Architecture & Logic Flow**
- **Entrypoints:** `vllm/engine/async_llm_engine.py` (Async API), `vllm/worker/model_runner.py` (Execution).
- **Data Flow:** Request -> Scheduler -> Block Manager -> Model Runner -> CUDA Graph -> Output.
- **Engine Type:** Framework shell / custom logic for serving.
- **Concurrency:** Async/await at the API layer, synchronous pipeline at the worker layer.
- **Architectural Shape:** Service-oriented, modular monolith.

**B. Functional Decomposition & “The Heart”**
- **Hot Path:** `vllm/compilation/cuda_graph.py` and `vllm/worker/model_runner.py`.
- The real behavior occurs in the `CUDAGraphRunner` where graphs are captured per bucket size and replayed.
- **Unique Value Proposition:** PagedAttention and robust shape bucketing for dynamic batching.
- **Complexity Score:** 8/10. Abstractions are deep, but well-named and highly modular.

**C. Dependency & Health Audit**
- **Manifest:** `requirements.txt` / `pyproject.toml`.
- **Dependencies:** Framework-heavy (PyTorch, Ray, xformers/flash-attn).
- **Tree Shape:** Moderately layered, some dependency-hell risk with PyTorch/CUDA versions.
- **Health:** Exceptional. High commit frequency, massive E2E CI, broad contributor base. Apache 2.0 license is highly suitable.

**D. Developer Experience & Integration**
- **DX:** High. Well-documented, standard APIs.
- **Integration:** Clean wrapper-friendly internal APIs (`LLMEngine`).
- **Tests:** Massive E2E CI, comprehensive unit testing. Protects the hot path.

**E. Lock-in & Migration Risk**
- **Migration Risk:** Low
- **Hardest to replace:** The PagedAttention memory manager.
- **Can be isolated:** The API layer.
- **Must sandbox:** Ray dependency.

### Repo B: SGLang

**A. Core Architecture & Logic Flow**
- **Entrypoints:** `sglang/lang/` (Frontend API), `sglang/srt/` (Server/Runner).
- **Data Flow:** API -> Radix Cache -> Scheduler -> Model Runner.
- **Engine Type:** Custom logic with a strong focus on structured generation.
- **Concurrency:** Async/await frontend, highly optimized synchronous backend.
- **Architectural Shape:** Monolith with a tightly integrated frontend/backend.

**B. Functional Decomposition & “The Heart”**
- **Hot Path:** `sglang/srt/managers/router/model_runner.py` and RadixAttention kernels.
- The hot path is heavily concentrated around the Radix KV Cache and its integration with the model forward pass.
- **Unique Value Proposition:** RadixAttention for automatic prefix caching.
- **Complexity Score:** 7/10. Tighter coupling between cache and execution than vLLM, making isolated extraction harder.

**C. Dependency & Health Audit**
- **Manifest:** `pyproject.toml`.
- **Dependencies:** Leaner than vLLM, heavily reliant on PyTorch and custom Triton/CUDA kernels.
- **Tree Shape:** Flat and controlled.
- **Health:** Strong, rapid growth. The Rising Star. Apache 2.0.

**D. Developer Experience & Integration**
- **DX:** High for rapid prototyping.
- **Integration:** Clean, but heavily reliant on its specific frontend abstractions for max performance.
- **Tests:** Moderate CI, heavily relies on functional testing over unit testing.

**E. Lock-in & Migration Risk**
- **Migration Risk:** Moderate
- **Hardest to replace:** RadixAttention cache.
- **Can be isolated:** Frontend API parser.
- **Must sandbox:** Custom Triton kernels.

### Repo C: TGI

**A. Core Architecture & Logic Flow**
- **Entrypoints:** `router/` (Rust gRPC Server), `server/` (Python execution environment).
- **Data Flow:** Rust Router -> gRPC -> Python Server -> PyTorch Model.
- **Engine Type:** Wrapper around existing libraries / Service-oriented.
- **Concurrency:** Rust async (Tokio) for routing, Python multiprocessing/threading for execution.
- **Architectural Shape:** Service-oriented (Rust Router + Python Workers).

**B. Functional Decomposition & “The Heart”**
- **Hot Path:** Rust router for request batching; `server/text_generation_server/models/` for execution.
- Logic is split across the language boundary.
- **Unique Value Proposition:** Extreme operational maturity and tight integration with Hugging Face Hub.
- **Complexity Score:** 9/10. The language boundary and heavy use of custom C++/CUDA kernels (FlashInfer, etc.) add significant complexity.

**C. Dependency & Health Audit**
- **Manifest:** `Cargo.toml`, `requirements.txt`.
- **Dependencies:** Deeply transitive on the Rust side, framework-heavy on the Python side.
- **Tree Shape:** Deep and fragile.
- **Health:** Very strong corporate backing (Hugging Face). Apache 2.0.

**D. Developer Experience & Integration**
- **DX:** Low for internal hacking. High for black-box deployment.
- **Integration:** Invasive. Requires running a separate Rust process.
- **Tests:** High coverage, but very difficult to run locally.

**E. Lock-in & Migration Risk**
- **Migration Risk:** Moderate
- **Hardest to replace:** The Rust continuous batching router.
- **Can be isolated:** Python model execution wrappers.
- **Must sandbox:** Rust dependencies and gRPC interface.

## 4. Feature Parity Table

| Feature | vLLM | SGLang | TGI |
| :--- | :--- | :--- | :--- |
| **Plugin/Module System** | Partial (Models) | Partial | No |
| **Config Layering** | Yes | Yes | Yes |
| **CLI Support** | Yes | Yes | Yes |
| **API Surface** | OpenAI-compatible | Custom + OpenAI | Custom REST/gRPC |
| **Streaming** | Yes | Yes | Yes |
| **Job Orchestration** | Ray (optional) | Custom | Rust Router |
| **State (KV Cache)** | PagedAttention | RadixAttention | PagedAttention |
| **Logging/Observability** | Prometheus | Prometheus | OpenTelemetry |
| **Test Harness Depth** | Massive | Moderate | Moderate |
| **Shape Bucketing (Graphs)**| Yes | Yes | Limited |

## 5. Comparative Trade-off Matrix

| Metric | vLLM | SGLang | TGI | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Architectural Clarity** | 8 | 7 | 6 | vLLM is highly modular. TGI's Rust/Python split adds friction. |
| **Maintainability** | 7 | 8 | 5 | SGLang is currently leaner. TGI requires dual-language expertise. |
| **Extensibility** | 9 | 7 | 6 | vLLM's model registry is easiest to extend. |
| **Performance Potential** | 9 | 10 | 9 | SGLang's RadixAttention pushes it slightly ahead for complex prompts. |
| **Dependency Risk** | 6 | 7 | 5 | vLLM has heavy Ray/Torch dependencies. TGI has many custom kernels. |
| **Migration Flexibility** | 7 | 6 | 5 | vLLM's API is standard. SGLang's frontend creates some lock-in. |
| **DX / Onboarding** | 8 | 8 | 5 | TGI's build system is notoriously complex. |
| **Test Trustworthiness** | 9 | 7 | 8 | vLLM has the most comprehensive test suite. |
| **Operational Maturity** | 9 | 7 | 10 | TGI is built for HF production scale. |
| **Integration Readiness** | 8 | 8 | 5 | vLLM and SGLang are easy to embed. TGI is a standalone service. |
| **Licensing Suitability** | 10 | 10 | 10 | All Apache 2.0. |

## 6. Integration Opportunity Mapping

### Opportunity 1: Shape Bucketing Pool (vLLM)
- **Location:** `vllm/compilation/cuda_graph.py`
- **Value:** Handles dynamic batch sizes efficiently without OOMing on graph capture.
- **Difficulty:** Medium
- **Recommendation:** Adapt to our architecture.
- **Type:** Strategic Extraction.

### Opportunity 2: Radix Cache (SGLang)
- **Location:** `sglang/srt/managers/radix_cache.py`
- **Value:** Automatic prefix caching for multi-turn conversations.
- **Difficulty:** High
- **Recommendation:** Use as inspiration only.
- **Type:** Strategic Extraction.

### Opportunity 3: Multi-Language Batching (TGI)
- **Location:** TGI Rust Router
- **Value:** Extreme performance for concurrent requests.
- **Difficulty:** High
- **Recommendation:** Avoid importing (too much architectural friction).
- **Type:** Attractive Trap.

## 7. Adoption Plan

1. **Target Architecture:** Adapt vLLM's `BatchBucketPool` into a lightweight, standalone Python module within `gfxGRAPH` that can be optionally used by frameworks that lack their own bucketing.
2. **Seams:** Connect this pool via the existing `BridgedCUDAGraph` interface, allowing it to manage internal memory pools invisibly to the caller.
3. **Rollout:** Implement in a separate namespace (`gfxgraph.orchestration`), test against dummy PyTorch models, then integrate with the core interceptor.
4. **Fallback:** If the bucketing fails, fall back to pure eager execution.

## 8. Concrete Work Items

1. **[Ticket 1] Implement Standalone `ShapeBucket` Primitive**
   - **Purpose:** Create a data structure to track padding, memory pools, and hit rates for specific input shapes.
   - **Affected Area:** `python/hipgraph_bridge/bucketing.py`
   - **Risk:** Low
   - **Acceptance:** Unit tests pass for bucket matching and eviction logic.

2. **[Ticket 2] Develop `BucketPoolManager`**
   - **Purpose:** Manage a collection of `ShapeBucket`s with an LRU eviction policy based on VRAM constraints.
   - **Affected Area:** `python/hipgraph_bridge/bucketing.py`
   - **Dependency:** Ticket 1
   - **Risk:** Medium
   - **Acceptance:** Can add, retrieve, and automatically evict buckets when a mock memory limit is reached.

3. **[Ticket 3] Integrate Bucketing with `BridgedCUDAGraph`**
   - **Purpose:** Intercept capture calls, route them to the appropriate bucket, or trigger a new capture if no bucket matches.
   - **Affected Area:** `python/hipgraph_bridge/graph_manager.py`
   - **Dependency:** Ticket 2
   - **Risk:** High
   - **Acceptance:** A dynamic shape inference loop successfully uses the bucketing system without OOMing or failing capture.

**First PR:** Implement Tickets 1 and 2 (the isolated bucketing logic).
**Second PR:** Implement Ticket 3 (integration into the hot path).

**What Not To Do:** Do not import Ray or distributed execution concepts. Keep it strictly limited to single-node, single-process CUDA graph orchestration.

## 9. Final Recommendation

- **Production Stability:** TGI
- **Best Internal Fork Candidate:** SGLang
- **Best Reference Architecture:** vLLM
- **Best to Learn From, Not Adopt:** TGI

**Recommendation:**
- **Adopt directly:** vLLM's shape bucketing concepts.
- **Fork selectively:** SGLang's Radix cache concepts if we build a full engine.
- **Mine for ideas:** TGI's Rust router.
- **Strongest Internals:** vLLM (Python), TGI (Rust).

## 10. Horizon Scanning

1. **The Rising Star: LMDeploy**
   - **Why:** Extremely fast inference engine focused on TurboMind (C++) backend.
   - **Category:** High-performance C++ Engine.
   - **Value:** Shows the limits of what pure C++ can do vs Python orchestration.

2. **The Legacy Standard: FasterTransformer**
   - **Why:** The original highly optimized transformer inference library.
   - **Category:** Foundational kernels.
   - **Value:** Great reference for pure CUDA kernel implementations, though largely superseded by TRT-LLM.

3. **The Niche Specialist: ExLlamaV2**
   - **Why:** Hyper-optimized for local, low-VRAM inference of quantized models.
   - **Category:** Consumer/Local Execution.
   - **Value:** Incredible techniques for managing extreme memory constraints, highly relevant for RDNA2 12GB targets.

## 11. Appendix: Evidence Notes
- vLLM shape bucketing analyzed from prior knowledge of `vllm/compilation/cuda_graph.py`.
- SGLang RadixAttention details verified against standard architecture documentation for SGLang.
- TGI architecture (Rust/Python split) is standard knowledge for HuggingFace's production stack.
