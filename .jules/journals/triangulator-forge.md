
2026-03-26 - vLLM vs gfxGRAPH Architectural Constraints
Learning: vLLM's `cuda_graph.py` implements a robust shape bucketing pool, but relies heavily on `CUDAGraphMode` and static memory pools. gfxGRAPH intercepts these exact patterns via Python monkey-patching in `python/hipgraph_bridge/graph_manager.py`, intercepting `torch.cuda.CUDAGraph`. This reveals a major constraint: AMD graph execution requires intercepting and neutralizing capture failures (eager fallback) because features like NGRAM in vLLM trigger silent HIP capture faults.
Action: Adopt vLLM's shape bucketing but wrap graph generation in a failure-resistant interceptor similar to gfxGRAPH's `BridgedCUDAGraph`.

2026-04-28 - Breakable CUDA Graphs in SGLang
Learning: SGLang implements a `BreakableCUDAGraph` in `sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py` which divides a single logical forward pass into an array of smaller `torch.cuda.CUDAGraph` segments. This prevents full execution failure when encountering a node that forces eager execution, a massive architectural win for fallback resilience on ROCm.
Action: Design a `SegmentedCUDAGraph` container within gfxGRAPH to intercept and partition graphs that contain unsupported HIP nodes.
