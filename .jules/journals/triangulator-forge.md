
2026-03-26 - vLLM vs gfxGRAPH Architectural Constraints
Learning: vLLM's `cuda_graph.py` implements a robust shape bucketing pool, but relies heavily on `CUDAGraphMode` and static memory pools. gfxGRAPH intercepts these exact patterns via Python monkey-patching in `python/hipgraph_bridge/graph_manager.py`, intercepting `torch.cuda.CUDAGraph`. This reveals a major constraint: AMD graph execution requires intercepting and neutralizing capture failures (eager fallback) because features like NGRAM in vLLM trigger silent HIP capture faults.
Action: Adopt vLLM's shape bucketing but wrap graph generation in a failure-resistant interceptor similar to gfxGRAPH's `BridgedCUDAGraph`.

2026-05-01 - AMD Graph Integration Surface Constraints
Learning: A persistent constraint across frameworks targeting AMD RDNA2/gfx1030 is the tension between heavy, enterprise-grade compilation pipelines (e.g., AMDMIGraphX, requiring rocBLAS, MIOpen, C++ toolchains) and lightweight, environment-level intercepts (e.g., gfxGRAPH's Python monkey-patching and vLLM's abstracted cuda_graph.py). Integrating robust CUDA graph parity on lower-tier hardware necessitates avoiding monolith compilers in favor of modular shims and dynamic shape bucketing. Relying directly on vendor C++ IR compilers introduces severe lock-in and high setup friction.
Action: Abstract any complex graph orchestration (like vLLM's batch bucketing) behind a Python-native interface, and utilize environmental shims (like gfxGRAPH) for hardware compatibility, avoiding direct integration with heavy compiler toolchains like MIGraphX.
 2026-05-01 - Inference Engine Graph Layer Fragmentation
Learning: Across vLLM, SGLang, and TGI, there is a recurring architectural fragmentation in how CUDA graphs are orchestrated. vLLM uses a strict bucketed pool in `vllm/compilation/cuda_graph.py`, SGLang relies on deep integration with RadixAttention for graph capture, and TGI defers heavily to external custom kernels or PyTorch native compilation. This means any bridge layer (like gfxGRAPH) cannot assume a unified `torch.cuda.CUDAGraph` lifecycle; it must actively intercept or mock differing context managers and memory pool behaviors.
Action: Any adoption of graph orchestration logic from these repos must isolate the memory pool lifecycle from the graph capture context manager to ensure cross-compatibility with differing engine designs.

 2026-05-02 - Extracting Independent Shape Bucketing
Learning: Inference engines (vLLM, SGLang) heavily couple shape bucketing to PyTorch's native CUDA graph orchestration, making them brittle on AMD eager fallbacks.
Action: Extracted knowledge from vLLM's `cuda_graph.py` to propose a pure-Python, framework-agnostic bucketing pool for `gfxGRAPH` that decouples memory management from the graph capture context.
