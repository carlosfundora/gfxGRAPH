---
name: gfxgraph-development
description: >-
  Develop and extend gfxGRAPH internals — add gap bridges, fix HIP/ROCm issues, debug graph
  capture failures, and understand the architecture. Covers BridgedCUDAGraph, ShapeBucketPool,
  ConditionalGraph, native bridge, and the monkey-patch system.
  USE FOR: extend gfxGRAPH, add new gap bridge, fix gfxGRAPH bug, understand gfxGRAPH architecture,
  modify BridgedCUDAGraph, add gfxGRAPH feature, debug graph capture internals, improve gfxGRAPH
  performance, gfxGRAPH C++ native bridge, write gfxGRAPH tests.
  DO NOT USE FOR: just using gfxGRAPH in a project (use gfxgraph-integration), general ROCm issues.
---

# gfxGRAPH Development Guide

This skill provides deep knowledge of gfxGRAPH internals for agents who need to modify,
extend, or debug the library itself.

## Repository Layout

```
gfxGRAPH/
├── python/
│   ├── gfxgraph/                    # Public package (user-facing)
│   │   ├── __init__.py              # Re-exports + auto-enable via GFXGRAPH env var
│   │   ├── __main__.py              # CLI: python -m gfxgraph <script.py>
│   │   └── _enable.py               # enable()/disable()/stats()/health_check()
│   ├── hipgraph_bridge/             # Internal implementation (the actual bridge)
│   │   ├── __init__.py              # Exports BridgedCUDAGraph, ShapeBucketPool
│   │   ├── graph_manager.py         # BridgedCUDAGraph — core drop-in replacement
│   │   ├── shape_bucketing.py       # ShapeBucketPool — multi-bucket graph pool
│   │   ├── conditional.py           # ConditionalGraph — per-branch dispatch (Gap 51)
│   │   ├── compile_backend.py       # torch.compile backend registration
│   │   ├── ops.py                   # Custom torch.library ops (conditional_select)
│   │   └── _C.py                    # ctypes bindings to native .so (optional)
│   └── pyproject.toml
├── src/                             # C/HIP native bridge source
│   └── hipgraph_bridge/
│       └── bridge.hip               # Gap bridges 52, 54 (device-side launch, nested capture)
├── include/                         # C headers
├── tests/                           # HIP C++ tests + Python integration test
├── benchmarks/                      # Performance benchmarks
├── docs/
│   ├── hipgraph-bridge-design.md    # Full design specification
│   └── torch-hip-rocm-graph.md      # CUDA Graph parity matrix (HIP/Torch/MIGraphX)
├── CMakeLists.txt                   # Native build (requires ROCm SDK)
└── .github/
    ├── workflows/                   # CI
    └── skills/                      # These skill files
```

## Architecture

### Two-Package Design

- **`gfxgraph`** — Public API package. Users `import gfxgraph` and call `gfxgraph.enable()`.
  This package re-exports from `hipgraph_bridge` and manages the monkey-patch lifecycle.
- **`hipgraph_bridge`** — Internal implementation. Contains `BridgedCUDAGraph`, `ShapeBucketPool`,
  `ConditionalGraph`, the native `.so` bindings, and all bridge logic.

This separation means `hipgraph_bridge` can be imported directly by code that needs
specific classes without triggering auto-enable behavior.

### Monkey-Patch System (`_enable.py`)

`gfxgraph.enable()` does:
1. Stashes original `torch.cuda.CUDAGraph` in `_originals["CUDAGraph"]`
2. Replaces `torch.cuda.CUDAGraph` with `BridgedCUDAGraph`
3. Optionally registers the `hipgraph_bridge` torch.compile backend
4. Sets up atexit cleanup handler
5. Optionally enables validation mode (graph-vs-eager comparison)

`gfxgraph.disable()` restores originals. Thread-safe stats via `_stats_lock`.

### BridgedCUDAGraph (`graph_manager.py`)

The core class. Key design decisions:

**Low-Level API Compatibility:**
Methods `capture_begin()`, `capture_end()`, `replay()`, `pool()` delegate directly to a
real `_OriginalCUDAGraph` instance. This makes BridgedCUDAGraph a true drop-in for SGLang's
`cuda_graph_runner.py` which uses the low-level API.

**`_OriginalCUDAGraph` Reference:**
Captured at module load time BEFORE any monkey-patching:
```python
_OriginalCUDAGraph = torch.cuda.CUDAGraph
```
This ensures BridgedCUDAGraph always creates real CUDAGraph instances internally,
even after `torch.cuda.CUDAGraph` has been patched.

**Eager Fallback:**
When graph capture fails, `_eager_fallback = True` is set, and subsequent `replay()` calls
execute `model_fn` eagerly. This is the HIGH-LEVEL API path only. The low-level API
(used by SGLang) does NOT fall back — it raises exceptions that the caller handles.

**SIGABRT Prevention (CRITICAL):**
The `_CaptureContext.__exit__` must properly end HIP stream capture BEFORE dropping the
graph reference. Without this:
1. `self.parent._graph = None` triggers Python GC
2. `_OriginalCUDAGraph.__del__` → `at::cuda::CUDAGraph::~CUDAGraph()`
3. C++ destructor calls `hipStreamEndCapture` on stream still in capture mode
4. C++ exception in destructor → `std::terminate()` → SIGABRT

**Fix pattern (apply to ALL error paths in capture):**
```python
graph = self.parent._graph
if graph is not None:
    try:
        graph.capture_end()
    except Exception:
        pass
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
self.parent._graph = None  # NOW safe
```

### ShapeBucketPool (`shape_bucketing.py`)

- Uses `bisect.bisect_left` to find smallest bucket ≥ input size
- Lazy capture: only captures a bucket graph when first needed
- VRAM safety: checks `torch.cuda.mem_get_info()` vs `GFXGRAPH_VRAM_CAP` before capture
- Per-bucket failure tracking: failed buckets use eager fallback permanently

### ConditionalGraph (`conditional.py`)

Gap 51 bridge. Uses per-branch dispatch (separate CUDAGraph per branch) rather than
`hipGraphNodeSetEnabled` (unreliable on gfx1030 for child graph nodes).

### Native Bridge (`_C.py` + `src/`)

Optional. The `.so` provides C-level implementations of gaps 52 (device-side launch via
`hipGraphUpload` + rapid host pipeline) and 54 (nested capture via sequential capture +
child graph nodes). Auto-discovered via build directory, `LD_LIBRARY_PATH`, or
`GFXGRAPH_LIB` env var.

## Gaps Bridged

| # | CUDA Feature | HIP Status | Bridge Strategy | Tier |
|:-:|-------------|-----------|-----------------|:----:|
| 51 | Conditional nodes | Missing | Per-branch dispatch / `hipGraphNodeSetEnabled` | 1/2 |
| 52 | Device-side launch | Missing | `hipGraphUpload` + rapid host pipeline | 2 |
| 53 | Dynamic input shapes | Partial | Shape bucketing + parameter update | 1 |
| 54 | Nested capture | Missing | Sequential capture + child graph nodes | 2 |

## Adding a New Gap Bridge

1. **Identify the gap**: Check `docs/torch-hip-rocm-graph.md` parity matrix
2. **Choose tier**: Python-only (Tier 1) or requires native code (Tier 2)
3. **Implement in `hipgraph_bridge/`**: New module or extend existing class
4. **Wire into BridgedCUDAGraph**: Add detection + routing in `capture()` or `replay()`
5. **Add counter integration**: Use `_bump_capture()`, `_bump_fallback()`, `_record_replay()`
6. **Add tests**: Python test in `tests/test_torch_integration.py`, HIP test if native
7. **Update health check**: Add smoke test case in `_enable.py::health_check()`
8. **Update docs**: parity matrix + README

## Counter/Observability System

All counters are thread-safe (protected by `_stats_lock` in `_enable.py`):

```python
from gfxgraph._enable import bump, record_replay_us

bump("capture_count")       # increment capture counter
bump("fallback_count")      # increment fallback counter  
record_replay_us(42.3)      # record replay latency, updates running average
```

Bridge modules import these via try/except to remain import-safe when `gfxgraph`
package isn't installed (only `hipgraph_bridge` is).

## Testing

### Python Integration Tests

```bash
cd gfxGRAPH
python3 -m pytest tests/test_torch_integration.py -v
```

### Native Bridge Tests (requires ROCm SDK)

```bash
cd gfxGRAPH
cmake -B build -GNinja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_HIP_ARCHITECTURES=gfx1030
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Health Check Smoke Test

```bash
python3 -c "import gfxgraph; gfxgraph.enable(); print(gfxgraph.health_check())"
```

## SGLang Integration Points

When gfxGRAPH is used with SGLang, the key integration file is:
`sglang/srt/model_executor/cuda_graph_runner.py`

### How SGLang Uses BridgedCUDAGraph

1. **Capture**: `cuda_graph_runner._capture_graph()` creates `BridgedCUDAGraph()`,
   calls `capture_begin()` / `capture_end()` (low-level API)
2. **Run check**: `cuda_graph_runner.can_run()` checks if graph exists for batch size
3. **Replay**: `cuda_graph_runner.replay()` calls `graph.replay()` on the captured graph
4. **Routing**: `model_runner._forward_raw()` decides graph vs direct forward based on `can_run()`

### Handling Capture Failures in SGLang

When capture fails (e.g., NGRAM speculative decoding uses unsupported ops):
- `_capture_graph()` should RAISE RuntimeError (not silent eager fallback)
- The outer handler sets `self.graphs[key] = None`
- `can_run()` returns False for None graphs
- SGLang routes through the direct (non-graph) forward path

**Critical: SGLang's `can_run()` must handle BOTH padding modes:**
```python
# disable_padding=True: direct key lookup
if self.graphs[graph_key] is None:
    return False

# disable_padding=False: compute padded batch size first
padded_idx = bisect.bisect_left(self.capture_bs, cuda_graph_bs)
padded_key = ... # look up actual padded key
if self.graphs.get(padded_key) is None:
    return False
```

## Common Development Tasks

### Debugging Graph Capture Failures

1. Set `HGB_LOG_LEVEL=debug` or `GFXGRAPH=debug`
2. Check `gfxgraph.stats()` for `fallback_count`
3. Enable validation: `GFXGRAPH=validate` to catch silent correctness bugs
4. Add `torch.cuda.synchronize()` before/after suspected failure points
5. Check ROCm: `rocminfo | grep gfx`, verify `HSA_OVERRIDE_GFX_VERSION`

### Adding Torch Library Ops

Register new ops in `hipgraph_bridge/ops.py`:
```python
LIBRARY.define("my_op(Tensor x) -> Tensor")

@torch.library.impl(LIBRARY, "my_op", "CUDA")
def my_op_cuda(x):
    return ...

@torch.library.impl(LIBRARY, "my_op", "Meta")
def my_op_meta(x):
    return torch.empty_like(x)  # shape inference
```

### Performance Profiling

```python
import gfxgraph
gfxgraph.enable()

# ... run workload ...

s = gfxgraph.stats()
print(f"Captures: {s['capture_count']}")
print(f"Replays: {s['replay_count']}")
print(f"Fallbacks: {s['fallback_count']}")
print(f"Avg replay: {s['avg_replay_us']:.1f} µs")
```

## Thread Safety

- `BridgedCUDAGraph`: NOT thread-safe per instance (designed for single-thread use per graph)
- `gfxgraph.stats()`: Thread-safe (uses `_stats_lock`)
- `ShapeBucketPool`: NOT thread-safe (designed for single-thread capture/replay loop)
- `ConditionalGraph`: NOT thread-safe per instance
- The monkey-patch itself (`enable`/`disable`): Thread-safe (one-time setup)
- HIP Graph capture: Must be done on the thread that owns the CUDA stream

## Version History

- **v0.3.0** — Current. SIGABRT fix, SGLang low-level API compat, VRAM safety, validation mode
- **v0.1.0** — Initial. Basic BridgedCUDAGraph with eager fallback
