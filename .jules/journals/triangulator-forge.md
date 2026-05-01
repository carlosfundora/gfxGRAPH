
2026-03-26 - vLLM vs gfxGRAPH Architectural Constraints
Learning: vLLM's `cuda_graph.py` implements a robust shape bucketing pool, but relies heavily on `CUDAGraphMode` and static memory pools. gfxGRAPH intercepts these exact patterns via Python monkey-patching in `python/hipgraph_bridge/graph_manager.py`, intercepting `torch.cuda.CUDAGraph`. This reveals a major constraint: AMD graph execution requires intercepting and neutralizing capture failures (eager fallback) because features like NGRAM in vLLM trigger silent HIP capture faults.
Action: Adopt vLLM's shape bucketing but wrap graph generation in a failure-resistant interceptor similar to gfxGRAPH's `BridgedCUDAGraph`.

2026-05-01 - AMD Graph Integration Surface Constraints
Learning: A persistent constraint across frameworks targeting AMD RDNA2/gfx1030 is the tension between heavy, enterprise-grade compilation pipelines (e.g., AMDMIGraphX, requiring rocBLAS, MIOpen, C++ toolchains) and lightweight, environment-level intercepts (e.g., gfxGRAPH's Python monkey-patching and vLLM's abstracted cuda_graph.py). Integrating robust CUDA graph parity on lower-tier hardware necessitates avoiding monolith compilers in favor of modular shims and dynamic shape bucketing. Relying directly on vendor C++ IR compilers introduces severe lock-in and high setup friction.
Action: Abstract any complex graph orchestration (like vLLM's batch bucketing) behind a Python-native interface, and utilize environmental shims (like gfxGRAPH) for hardware compatibility, avoiding direct integration with heavy compiler toolchains like MIGraphX.
