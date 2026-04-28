# Comparative Repository Audit: vLLM vs SGLang vs TGI

## 1. Executive Technical Summary

This technical audit evaluates the three dominant modern LLM serving engines—**vLLM**, **SGLang**, and **Text Generation Inference (TGI)**—specifically through the lens of CUDA Graph capture patterns, memory pool management, and inference framework extensibility. The objective is to identify architectural patterns that dictate the stability of graph execution, extract implementation opportunities for gfxGRAPH (which bridges PyTorch CUDA graph gaps on AMD RDNA2 architectures), and measure the migration risk across the frameworks.

**The verdict:**
*   **vLLM** has the most mature internal graph memory allocator and shape bucketing system, but its deep framework entanglements and massive module footprint make it highly fragile to wrap or monkey-patch.
*   **SGLang** provides the best "seam" for integration. Its `cuda_graph_runner.py` is isolated and cleanly implements breakable CUDA graphs, making it the most maintainable reference architecture for custom extensions.
*   **TGI** provides the highest performance floor through its strict Rust/C++ memory isolation, but its split-brain Rust/Python architecture makes adapting its hot path to unsupported backends exceptionally difficult.

## 2. Repository Targets & Assumptions

*   **Repo A: vLLM** (`https://github.com/vllm-project/vllm`). The de facto community standard for LLM serving.
*   **Repo B: SGLang** (`https://github.com/sgl-project/sglang`). Evolving from an LMQL-like syntax into a deeply optimized serving backend natively pushing Triton/graph bounds.
*   **Repo C: Text Generation Inference (TGI)** (`https://github.com/huggingface/text-generation-inference`). The Hugging Face standard inference service, notable for its Rust router bridging to a Python/PyTorch backend via gRPC.

**Assumption:** These three targets were chosen because they all implement custom `torch.cuda.CUDAGraph` execution loops, memory pools (PagedAttention/FlashAttention state), and dynamic batching, making them the perfect stress-tests for a layer like `gfxGRAPH` that intends to inject custom graph capture routing.

## 3. Per-Repo Deep Audit

### Repo A: vLLM

#### A. Core Architecture & Logic Flow
vLLM is a massive, highly-coupled **monolith** driven by an asyncio event loop running Python-based continuous batching orchestrators (`AsyncLLMEngine` / `LLMEngine`). The hot path flows from the API server down to a multi-tiered scheduler (`vllm/engine/scheduler.py`), then out to TP/PP workers (`vllm/engine/tp_worker.py`). The true execution engine is buried deep within the model runners and custom CUDA kernels (`vllm/compilation/cuda_graph.py`).

#### B. Functional Decomposition & “The Heart”
The heart is `vllm/compilation/cuda_graph.py` and `vllm/engine/scheduler.py`. The graph capture is highly stateful. `cuda_graph.py` implements advanced shape bucketing and captures the graph by directly instantiating `cudagraph = torch.cuda.CUDAGraph()`. Uniquely, it overrides the Python Garbage Collector (`gc.collect`) during capture via `ExitStack` to prevent extreme slowdowns when looping over piece-wise graph captures.
*   **Complexity Score: 9/10.** Extremely high abstraction depth. Graph pooling is deeply entangled with `current_platform.graph_pool_handle()` and offloader synchronization.

#### C. Dependency & Health Audit
vLLM carries an immense, brittle dependency tree (`pyproject.toml`). It requires specific pinned versions of Ray, xformers, and FastAPI. It relies on a sprawling C++ extension build via CMake.
*   **Health:** Pulse is frantic. The community size is a massive asset but means PRs frequently conflict, and internal APIs break between minor versions.

### Repo B: SGLang

#### A. Core Architecture & Logic Flow
SGLang operates as a **layered app/framework**, split broadly into a frontend language interface and a backend serving engine (`sglang/srt`). The execution engine uses an event loop with a RadixAttention cache structure.

#### B. Functional Decomposition & “The Heart”
The hot path for graph execution resides cleanly in `sglang/srt/model_executor/cuda_graph_runner.py` and `breakable_cuda_graph.py`. Unlike vLLM's sprawling state, SGLang implements a brilliant "breakable" graph container that gracefully splits a single logical execution into multiple `torch.cuda.CUDAGraph` segments to accommodate operations that force eager execution syncs.
*   **Complexity Score: 6/10.** The codebase is highly modular. The separation of `cuda_graph_runner` and the DeepEP buffer dispatch modes makes understanding the hot path significantly easier than vLLM.

#### C. Dependency & Health Audit
Dependencies are moderately layered but pragmatic. `pyproject.toml` shows a reliance on flashinfer, aiohttp, and a lighter set of dependencies compared to vLLM.
*   **Health:** Rapid growth, focused maintainers. The codebase shows fewer signs of technical debt and a much healthier modular approach to new backends.

### Repo C: Text Generation Inference (TGI)

#### A. Core Architecture & Logic Flow
TGI is a **service-oriented architecture** bridging a high-performance Rust router (`router/src/`) and a Python execution backend (`server/`). The Rust layer handles the HTTP/gRPC interfaces, tokenization, queueing, and continuous batching logic, while Python strictly executes the forward pass and CUDA graphs.

#### B. Functional Decomposition & “The Heart”
The Python execution logic is cleanly separated by model type (e.g., `server/text_generation_server/models/flash_causal_lm.py`). The CUDA graph capture is relatively simple, allocating `graph = torch.cuda.CUDAGraph()` and explicitly managing block tables and sequence lengths (`Seqlen`) inside the forward context.
*   **Complexity Score: 8/10.** The complexity comes not from Python spaghetti, but from the strict gRPC serialization boundaries between Rust and Python, making deep debugging across the language barrier difficult.

#### C. Dependency & Health Audit
TGI has strict, controlled dependency manifests (`Cargo.toml` for Rust, and specific `requirements_*.txt` files for distinct hardware targets like Intel, ROCm, CUDA).
*   **Health:** Commercially backed by Hugging Face. Very stable, but strictly controlled by corporate priorities. Modifications require navigating dual-language build systems.

## 4. Feature Parity Table

| Feature | vLLM | SGLang | TGI |
| :--- | :--- | :--- | :--- |
| **CUDA Graph Capture System** | Advanced (Shape bucketing) | Advanced (Breakable segments) | Moderate (Per-batch size graphs) |
| **Memory Pool Control** | High (Custom platform handles) | High (RadixAttention aware) | High (Strict block tables) |
| **Extensibility Hooks** | Low (Internal APIs shift constantly) | High (Clean module boundaries) | Low (Rust/Python barrier) |
| **Observability / Logging** | High | Moderate | High (Rust OpenTelemetry) |
| **Hardware Agnosticism** | Moderate (Massive if/else trees) | Moderate | High (Isolated requirements files) |

## 5. Comparative Trade-off Matrix

| Metric | vLLM Score | SGLang Score | TGI Score | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Architectural Clarity** | 4/10 | 8/10 | 7/10 | vLLM is a massive monolith with tangled state. SGLang uses clean modular boundaries. TGI has clean Rust/Python separation but complex gRPC boundaries. |
| **Maintainability** | 4/10 | 7/10 | 6/10 | vLLM changes rapidly and breaks internal APIs. SGLang has smaller scope and cleaner abstractions. TGI requires dual-language maintenance. |
| **Extensibility** | 3/10 | 8/10 | 4/10 | SGLang is designed to be easily extended with new runners. vLLM and TGI are rigid around their core abstractions. |
| **Performance Potential** | 9/10 | 10/10 | 9/10 | All are highly optimized. SGLang pushes boundaries slightly further with Triton/native graph integration. |
| **Dependency Risk** | 3/10 | 5/10 | 4/10 | (Higher is better) vLLM has massive dependency chains (Ray, etc). SGLang is lighter. TGI manages it strictly per-hardware. |
| **Migration Flexibility** | 2/10 | 5/10 | 3/10 | (Higher is better) vLLM tightly couples to its APIs. SGLang is somewhat decoupled. TGI ties deeply to HF ecosystem. |
| **DX / Onboarding** | 4/10 | 7/10 | 5/10 | vLLM is incredibly complex to build locally. SGLang is simpler. TGI requires Rust toolchains. |
| **Test Trustworthiness** | 7/10 | 8/10 | 9/10 | TGI has rigorous Rust/Python cross-boundary testing. SGLang has decent coverage. vLLM tests are massive but sometimes flaky. |
| **Operational Maturity** | 9/10 | 7/10 | 9/10 | vLLM and TGI are battle-tested in massive production deployments. SGLang is maturing fast but newer. |
| **Integration Readiness** | 3/10 | 8/10 | 4/10 | SGLang provides clean python seams. TGI and vLLM are built to be standalone servers, not embedded components. |
| **Licensing Suitability** | 10/10 | 10/10 | 10/10 | All are Apache 2.0, suitable for our needs. |

## 6. Integration Opportunity Mapping

### Opportunity 1: "Breakable" CUDA Graphs
*   **Source:** SGLang (`sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py`)
*   **Value:** gfxGRAPH currently attempts to capture monolithic graphs and falls back if ANY gap is hit. Implementing a segmented/breakable container allows partial graph execution, dramatically reducing fallback penalties.
*   **Difficulty:** Medium
*   **Recommendation:** **Adapt to our architecture**. Steal the `_segments: list[torch.cuda.CUDAGraph]` pattern to wrap our `BridgedCUDAGraph`.

### Opportunity 2: Garbage Collector Suppression during Capture
*   **Source:** vLLM (`vllm/compilation/cuda_graph.py`)
*   **Value:** vLLM uses an `ExitStack` to `patch("gc.collect", lambda: None)` during graph capture. This prevents massive Python overhead during iterative piecewise graph captures.
*   **Difficulty:** Low (Fast Win)
*   **Recommendation:** **Adopt directly**. Inject this context manager around the `gfxGRAPH` capture boundaries to speed up the interception overhead.

### Opportunity 3: Hardware-Isolated Requirements
*   **Source:** TGI (`text-generation-inference/server/requirements_rocm.txt`)
*   **Value:** Isolating dependencies per backend prevents ROCm/CUDA library collisions on the host system.
*   **Difficulty:** Low
*   **Recommendation:** **Use as inspiration only**. gfxGRAPH already handles this by being entirely transparent unless explicitly loaded, but we could improve the packaging boundary.

## 7. Adoption Plan

**Target Architecture:**
We will integrate SGLang's concept of a `BreakableCUDAGraph` into the core `gfxGRAPH` python layer (`python/hipgraph_bridge/graph_manager.py`). This forms a "seam" where if `gfxGRAPH` encounters an unbridgeable node (e.g., gap #55 not supported by the native layer), instead of failing the entire graph and falling back to eager mode, we finalize `segment[n]`, execute the unbridgeable node eagerly, and begin capturing `segment[n+1]`.

**Rollout Order:**
1.  **Isolation Layer:** Implement the Python-side GC suppression (from vLLM) in the `gfxGRAPH` interceptor to get an immediate performance win during capture.
2.  **Core Feature:** Draft the `SegmentedCUDAGraph` class.
3.  **Integration:** Wire `SegmentedCUDAGraph` into the monkey-patch of `torch.cuda.CUDAGraph` to allow transparent segmented capture.

**Fallback Strategy:**
If segmented capture creates state corruption during replay, we default back to the current behavior: full eager fallback.

## 8. Concrete Work Items

1.  **Ticket: Implement GC Suppression during Graph Capture**
    *   **Purpose:** Speed up graph capture loops.
    *   **Affected Area:** `python/hipgraph_bridge/graph_manager.py` (capture boundary).
    *   **Risk:** Low.
    *   **Acceptance Criteria:** `patch("gc.collect", lambda: None)` is active during graph capture and yields measurable speedup in capture time.

2.  **Ticket: Design SegmentedCUDAGraph State Manager**
    *   **Purpose:** Extract SGLang's breakable graph concept.
    *   **Affected Area:** New module `python/hipgraph_bridge/segmented_graph.py`.
    *   **Risk:** Medium.
    *   **Acceptance Criteria:** Container can hold multiple graphs and handle sequential replay.

3.  **Ticket: Wire SegmentedCUDAGraph to Monkey-Patch**
    *   **Purpose:** Allow PyTorch graph interceptions to break automatically on gap detection.
    *   **Affected Area:** `python/hipgraph_bridge/graph_manager.py`.
    *   **Risk:** High.
    *   **Acceptance Criteria:** A graph containing a known failing node executes partially in graph mode and partially in eager mode.

**What not to do:**
Do not attempt to import vLLM's `graph_pool_handle` memory management. It is too deeply tied to their specific PagedAttention C++ implementations and will corrupt our clean Python interceptor layer.

## 9. Final Recommendation

*   **Best to adopt directly:** None. They are all complete applications, not libraries.
*   **Best to selectively adapt:** **SGLang**. Its `breakable_cuda_graph` is exactly the architectural leap gfxGRAPH needs to handle unsupported HIP graph operations gracefully.
*   **Best to avoid without heavy refactor:** **vLLM**. The entanglement of memory pools, offloading streams, and graph generation makes it incredibly risky to monkey-patch safely.
*   **Best Reference Architecture:** **TGI**, for how to build language-isolated system boundaries.

## 10. Horizon Scanning

1.  **The Rising Star: LMDeploy** (`InternLM/lmdeploy`)
    *   **Category:** High-efficiency C++ serving engine.
    *   **Reason:** Heavily optimized for pure tensorRT/TurboMind execution, offering a different path away from pure PyTorch graph capture.
2.  **The Legacy Standard: NVIDIA TensorRT-LLM** (`NVIDIA/TensorRT-LLM`)
    *   **Category:** Vendor-specific low-level graph compiler.
    *   **Reason:** Defines the absolute upper bound of performance, but is entirely useless for our ROCm/RDNA2 focus.
3.  **The Niche Specialist: Punica** (`punica-ai/punica`)
    *   **Category:** Multi-tenant LoRA serving.
    *   **Reason:** Solves graph capture specifically for dynamically swapping LoRA adapters, which requires extremely complex pointer management.

## 11. Appendix: Evidence Notes
*   vLLM's `cuda_graph.py` utilizes an `ExitStack` to suppress `gc.collect`.
*   SGLang cleanly defines `_segments: list[torch.cuda.CUDAGraph] = []` in `breakable_cuda_graph.py`.
*   TGI strictly splits dependencies using `requirements_*.txt` files per hardware backend.
*   All three manually instantiate `torch.cuda.CUDAGraph()` internally, meaning gfxGRAPH's global monkey-patch perfectly intercepts all of them.
