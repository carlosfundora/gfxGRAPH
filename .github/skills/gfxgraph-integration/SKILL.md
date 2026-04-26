---
name: gfxgraph-integration
description: >-
  Integrate gfxGRAPH into PyTorch or SGLang projects on AMD ROCm (RDNA2/gfx1030).
  Provides drop-in CUDA Graph parity, eager fallback, shape bucketing, and VRAM safety.
  USE FOR: enable CUDA graphs on AMD, integrate gfxGRAPH, configure gfxGRAPH for SGLang,
  fix graph capture failures on ROCm, RDNA2 CUDA graph compatibility, gfx1030 graph support,
  HIP graph bridge, monkey-patch torch.cuda.CUDAGraph, shape bucketing for dynamic batches.
  DO NOT USE FOR: developing or extending gfxGRAPH internals (use gfxgraph-development),
  general PyTorch ROCm issues unrelated to graph capture.
---

# gfxGRAPH Integration Guide

Use gfxGRAPH to enable transparent CUDA Graph capture/replay on AMD RDNA2 GPUs (gfx1030/gfx1031)
where native HIP Graph support has parity gaps. gfxGRAPH is a drop-in replacement — existing
`torch.cuda.CUDAGraph` code works unchanged after `gfxgraph.enable()`.

## Quick Start

### Installation

```bash
# From source (editable — recommended for development)
pip install -e /path/to/gfxGRAPH/python/

# Standard install
pip install /path/to/gfxGRAPH/python/
```

### Verify Installation

```python
import gfxgraph
print(gfxgraph.__version__)       # → "0.3.0"
print(gfxgraph.health_check())    # → {'ok': True, 'gpu': 'AMD Radeon RX 6700 XT', ...}
```

### Enable (Standalone PyTorch)

```python
import gfxgraph
gfxgraph.enable()  # patches torch.cuda.CUDAGraph globally

# All existing CUDA graph code now works on RDNA2:
graph = torch.cuda.CUDAGraph()  # actually BridgedCUDAGraph
```

### Enable (SGLang)

Set environment variables before launching:

```bash
export SGLANG_RDNA2_KERNELS=1           # activates gfxGRAPH in SGLang
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # required for gfx1031 → gfx1030 mapping
export PYTORCH_ROCM_ARCH=gfx1030
export AMD_SERIALIZE_KERNEL=3
export AMD_SERIALIZE_COPY=3

python3 -m sglang.launch_server --model-path <model> ...
```

### Enable (Environment Variable)

```bash
GFXGRAPH=1 python3 my_script.py         # standard mode
GFXGRAPH=debug python3 my_script.py     # verbose logging
GFXGRAPH=validate python3 my_script.py  # correctness checking mode
```

## Core API

### `gfxgraph.enable(**kwargs)`

Monkey-patches `torch.cuda.CUDAGraph` → `BridgedCUDAGraph`. Options:
- `debug=True` — verbose logging (also via `HGB_LOG_LEVEL=debug`)
- `validate=True` — compares graph output vs eager to catch silent bugs (PyTorch #155684)

### `gfxgraph.disable()`

Restores original `torch.cuda.CUDAGraph`. Safe to call even if not enabled.

### `gfxgraph.stats()`

Thread-safe performance counters:
```python
{'enabled_at': 1712..., 'capture_count': 32, 'replay_count': 1847,
 'fallback_count': 0, 'validation_failures': 0, 'avg_replay_us': 42.3}
```

### `gfxgraph.health_check()`

Quick GPU smoke test — captures and replays a trivial graph:
```python
{'ok': True, 'gpu': 'AMD Radeon RX 6700 XT', 'rocm': 'gfx1030',
 'native_bridge': False, 'vram_total_mb': 12288, 'vram_free_mb': 10240,
 'details': 'Graph capture/replay OK, output verified'}
```

## BridgedCUDAGraph

Drop-in replacement for `torch.cuda.CUDAGraph`. Supports both high-level and low-level APIs.

### Low-Level API (SGLang/vLLM compatible)

```python
from hipgraph_bridge import BridgedCUDAGraph

g = BridgedCUDAGraph()
g.capture_begin()
# ... operations ...
g.capture_end()
g.replay()
g.pool()  # returns mempool id
```

### High-Level API (shape bucketing + eager fallback)

```python
g = BridgedCUDAGraph()
with g.capture(dynamic_shapes=True, buckets=[1, 4, 8, 16, 32], model_fn=my_model):
    output = my_model(static_input)
g.replay(batch_size=12)  # auto-selects bucket 16, pads input
```

### Eager Fallback Behavior

When graph capture fails (common on RDNA2 for certain operations):
1. gfxGRAPH logs a warning
2. Sets `_eager_fallback = True`
3. Subsequent `replay()` calls execute the model eagerly via `model_fn`
4. **No crash, no SIGABRT** — the CRITICAL fix is proper HIP state cleanup before
   dropping graph references (prevents C++ destructor from throwing)

## ShapeBucketPool

Manages multiple CUDAGraph instances across batch size buckets:

```python
from hipgraph_bridge import ShapeBucketPool

pool = ShapeBucketPool(model_fn=my_model, buckets=[1, 4, 8, 16, 32, 64])
output = pool(input_tensor)  # auto-selects bucket, pads, replays
```

Features:
- VRAM monitoring — skips capture if usage exceeds `GFXGRAPH_VRAM_CAP` (default 80%)
- Lazy bucket instantiation — only captures when first needed
- Eager fallback per-bucket on capture failure

## ConditionalGraph

Per-branch graph dispatch (Gap 51 bridge):

```python
from hipgraph_bridge import ConditionalGraph

cg = ConditionalGraph()
cg.add_branch("training", train_fn)
cg.add_branch("inference", infer_fn)
cg.capture(example_input)
output = cg.run("inference", input_tensor)
```

## Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `GFXGRAPH` | `1`, `debug`, `validate` | Auto-enable on import |
| `GFXGRAPH_VRAM_CAP` | `0.0`–`1.0` (default `0.80`) | Max VRAM fraction for graph capture |
| `HGB_LOG_LEVEL` | `debug`, `info`, `warn`, `error` | Structured logging level |
| `SGLANG_RDNA2_KERNELS` | `1` | Enable gfxGRAPH in SGLang |
| `SGLANG_DISABLE_GFXGRAPH` | `1` | Disable gfxGRAPH in SGLang |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | Map gfx1031 → gfx1030 |

## Known Limitations and Workarounds

### NGRAM Speculative Decoding + Graph Capture

NGRAM's forward path uses operations incompatible with `hipStreamBeginCapture`. When NGRAM
is active, **all batch sizes will fail graph capture**. The correct handling is:

1. gfxGRAPH catches the capture failure (no SIGABRT)
2. SGLang's `cuda_graph_runner.can_run()` returns False for failed captures
3. SGLang routes through `_forward_raw`'s direct (non-graph) forward path
4. Performance: ~6-7 t/s with NGRAM (vs ~25 t/s with graphs on normal decode)

**Key code in SGLang's cuda_graph_runner.py:**
- `_capture_graph()`: raises RuntimeError on capture failure (not silent eager)
- `can_run()`: rejects batch sizes where `self.graphs[key] is None`
- `replay()`: safety guard against None graphs

### HIP Stream Capture SIGABRT Prevention

The CRITICAL fix in `_CaptureContext.__exit__`:
```python
# BEFORE dropping graph reference, end HIP capture state:
try:
    graph.capture_end()
except Exception:
    pass
torch.cuda.synchronize()
# NOW safe to drop:
self.parent._graph = None
```

Without this, the C++ destructor calls `hipStreamEndCapture` on a stream still in
capture mode → C++ exception → `terminate()` → SIGABRT (exit code -6).

### RDNA2 Masked-Lane Address Validation

RDNA2 hardware validates virtual addresses for ALL wavefront lanes, even exec-masked
(inactive) ones. This means:
- All `tl.load/tl.store` in Triton kernels need clamped offsets on partial tiles
- Use pattern: `offs_safe = tl.where(mask, offs, 0)` before pointer arithmetic
- This is NOT a gfxGRAPH issue but affects all Triton code on RDNA2

## Two Operating Tiers

| Tier | What | Capabilities |
|:----:|-------|:------------:|
| 1 | Python-only (`pip install`) | 52/54 CUDA Graph parity |
| 2 | Full native (`libhipgraph_bridge.so`) | 54/54 CUDA Graph parity |

Most users only need Tier 1. Tier 2 requires ROCm SDK headers and `hipcc` compiler.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `fallback_count` increasing | Graph capture failing for some shapes | Normal on RDNA2; eager fallback is intentional |
| SIGABRT (exit -6) during capture | C++ destructor throws during HIP cleanup | Update gfxGRAPH — fixed in v0.3.0+ |
| `hipErrorIllegalAddress` on replay | Eager fallback through graph runner static buffers | Route failed captures through direct forward, not eager |
| Health check `ok: False` | ROCm/HIP not working | Check `rocminfo`, `HSA_OVERRIDE_GFX_VERSION`, PyTorch CUDA available |
| All captures fail with NGRAM | NGRAM ops incompatible with stream capture | Expected — use `--disable-cuda-graph` or let routing handle it |
