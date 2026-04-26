
### Run: Optimize gfxGRAPH shape bucketing and VRAM allocation

**What changed**:
1. Fixed `model_fn` passing from `BridgedCUDAGraph.capture` to `ShapeBucketPool` ensuring shape bucketing runs properly without unconditionally failing into eager fallback mode.
2. Modified shape bucketing dynamic input allocation handling to correctly accept 2D+ shapes by reading the actual `example_input` instead of defaulting to a 1D tensor buffer.
3. Overhauled shape bucketing to allocate a single shared `_max_static_input` buffer that maps to the maximum bucket size, which is sliced for smaller buckets instead of creating N separate duplicate tensors for each size.
4. Passed `pool=torch.cuda.graph_pool_handle()` downstream to `torch.cuda.graph()` context manager. PyTorch caching allocator will now share output buffers and intermediates, limiting overall VRAM consumed by N buckets purely to the peak intermediate usage of the largest bucket instead of summing linearly.
5. Re-wrote `memory_overhead` tracker property to reflect actual max usage sizes across shared pools.

**Why it matters on gfx1030**:
RDNA2 consumer GPUs like the RX 6700 XT are strictly VRAM bottlenecked (12GB VRAM cap limit). A linear overhead for CUDA graph capture per bucket size meant dynamic shape sizes consumed unnecessary GBs of precious VRAM. Sharing the memory pool lets us expand our buckets granularly without exhausting memory and dropping our KV cache limits.

**What was learned**:
The codebase contains a great abstraction layer for bridging HIP capabilities, but the Python overhead allocation logic was clearly lifted from early static assumptions. PyTorch CUDA graphs provide very mature memory-pool primitives which work transparently with ROCm if utilized properly.

**Next steps**:
Look closer into `launch_pipeline.hip` for Gap 52 kernel launch latency when used with high-frequency tiny graph executions (like continuous small batch decode generation cycles).
