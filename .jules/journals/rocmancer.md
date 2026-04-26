
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

### Run: Optimize conditional graph branching memory allocations

**What changed**:
1. Following the identical principle from our shape bucketing VRAM optimization, `ConditionalGraph` (Gap 51) was updated to share intermediate memory.
2. Initialized `_mempool = torch.cuda.graph_pool_handle()` and attached it to all branch graph captures, reducing the VRAM footprint of control-flow ops to simply `max(branch_mem)`.
3. Replaced the `_static_inputs` mapping dict with a single `_shared_input` tensor allocation since branching executes mutually exclusively, saving more GBs of memory waste when inputs are large block states.

**Why it matters on gfx1030**:
RDNA2 conditional node dispatches are emulated entirely through a Python "supergraph" equivalent that pre-captures a graph for every branch condition (a huge weakness compared to proper hardware dispatching). Until native hardware control flow exists for these GPUs, minimizing the duplicate cost of every emulated branch is strictly critical to maintaining baseline KV-cache space on consumer 12GB cards.
