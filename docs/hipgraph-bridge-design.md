# HIP Graph Bridge — Design Specification

> **Purpose:** Drop-in translation layer for gfx1030 (RDNA2) that bridges
> all 4 CUDA Graph parity gaps, enabling dynamic CUDA→HIP graph call
> routing with maximum performance and minimum user-facing complexity.
>
> **Target:** AMD Radeon RX 6700 XT · gfx1030 · ROCm 7.2.0  
> **Companion:** See `torch-hip-rocm-graph.md` for the parity matrix.

---

## Routing Strategy: Path of Least Resistance

Use the **highest abstraction layer** that satisfies the workload.
Drop to lower layers only when parity or performance demands it.

```
┌─────────────────────────────────────────────────┐
│                User Workload                     │
└─────────────────────┬───────────────────────────┘
                      ▼
          ┌───── Can torch.compile ─────┐
          │     handle it natively?     │
          └──────┬──────────────┬───────┘
              YES│              │NO
                 ▼              ▼
          ┌──────────┐  ┌─────────────────────┐
          │ Tier 0   │  │ Does it need a gap   │
          │ Torch    │  │ capability (51-54)?  │
          │ only     │  └──────┬─────────┬─────┘
          └──────────┘      YES│         │NO
                               ▼         ▼
                    ┌────────────┐  ┌──────────┐
                    │ Tier 2     │  │ Tier 1   │
                    │ HIP Graph  │  │ HIP Graph│
                    │ + Bridge   │  │ native   │
                    │ .so        │  │          │
                    └────────────┘  └──────────┘
```

### Performance Tiers

| Tier | Layer | Capabilities | Performance | User Effort |
|:----:|-------|:------------:|:-----------:|:-----------:|
| 0 | `torch.compile` only | 31/54 | Baseline | None |
| 1 | HIP Graph native | 50/54 | +15–40% | Low |
| 2 | HIP Graph + bridge `.so` | 54/54 | +10–35% | Low (drop-in) |

### When to Escalate

| Signal | Action |
|--------|--------|
| `torch.compile` covers all ops | Stay at Tier 0 |
| Need graph replay perf (< 50μs launch) | Escalate to Tier 1 |
| Need conditional branching in graph | Escalate to Tier 2 (gap 51) |
| Need varying batch sizes in graph | Escalate to Tier 2 (gap 53) |
| Need graph-launches-graph | Escalate to Tier 2 (gap 52) |
| Need nested capture composition | Escalate to Tier 2 (gap 54) |
| Running unmodified CUDA code on ROCm | Use Layer 3 compat shim |

---

## Architecture Overview

Three layers, each optional. Load only what you need.

```
┌──────────────────────────────────────────────────────┐
│                   User Application                    │
├──────────────┬───────────────────┬───────────────────┤
│   PyTorch    │   Direct HIP C   │  Unmodified CUDA  │
│   torch.*    │   hip*() calls   │  cuda*() calls    │
├──────────────┼───────────────────┼───────────────────┤
│  Layer 2     │                   │  Layer 3          │
│  hipgraph_   │                   │  libcudagraph_    │
│  bridge/     │                   │  compat.so        │
│  (Python)    │                   │  (LD_PRELOAD)     │
├──────────────┴───────────────────┴───────────────────┤
│            Layer 1: libhipgraph_bridge.so             │
│     Gap bridges · Routing logic · Kernel pool         │
├──────────────────────────────────────────────────────┤
│         libamdhip64.so (ROCm 7.2 · 104 symbols)      │
├──────────────────────────────────────────────────────┤
│              gfx1030 · RDNA2 Hardware                 │
└──────────────────────────────────────────────────────┘
```

### Design Principles

1. **Zero overhead for native paths** — the 50 natively-supported
   capabilities pass straight through to libamdhip64.so
2. **Bridge only when needed** — gap code activates only when a gap
   capability is requested
3. **Fail-safe** — if a bridge cannot handle a pattern, fall back to
   eager execution rather than crash
4. **Cache everything** — compiled graphs, bucketed instances, and
   bridge kernel objects are all pooled and reused

---

## Layer 1: `libhipgraph_bridge.so`

The core C/HIP shared library. Exports bridge functions that
implement the 4 gap capabilities using verified workarounds.

### Build Target

```
libhipgraph_bridge.so
  Compiler:  hipcc --offload-arch=gfx1030
  Link:      -lamdhip64 -lhiprtc
  Standard:  C++17 / HIP
  Size est:  ~200–400 KB
```

### Public API: `hipgraph_bridge.h`

```c
#pragma once
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version & Init ─────────────────────────────── */

typedef struct {
    int major;
    int minor;
    int patch;
    const char* gfx_target;   /* "gfx1030" */
    const char* rocm_version; /* "7.2.0"   */
} hgb_version_t;

hgb_version_t hgb_get_version(void);
hipError_t    hgb_init(void);
void          hgb_shutdown(void);

/* ── Gap 51: Conditional Execution ──────────────── */

typedef struct {
    hipGraphExec_t    exec;
    hipGraphNode_t*   branch_nodes;   /* array of nodes per branch */
    int*              branch_sizes;   /* node count per branch     */
    int               num_branches;
} hgb_conditional_t;

/**
 * Build a supergraph containing all branches.
 * Branches are disabled by default.
 *
 * @param graphs   Array of sub-graphs (one per branch)
 * @param count    Number of branches (typically 2: if/else)
 * @param out      Populated conditional handle
 */
hipError_t hgb_conditional_create(
    hipGraph_t*         graphs,
    int                 count,
    hgb_conditional_t*  out
);

/**
 * Select and launch a branch.
 *
 * Disables all branches, enables branch[index],
 * then launches the exec on the given stream.
 */
hipError_t hgb_conditional_launch(
    hgb_conditional_t*  cond,
    int                 branch_index,
    hipStream_t         stream
);

/**
 * Select branch from device-side flag (async read).
 *
 * Reads *d_flag from device memory, uses it as branch index.
 * Requires a host-sync point (hipStreamSynchronize or event).
 */
hipError_t hgb_conditional_launch_by_flag(
    hgb_conditional_t*  cond,
    int*                d_flag,
    hipStream_t         stream
);

void hgb_conditional_destroy(hgb_conditional_t* cond);

/* ── Gap 52: Rapid Graph Launch Pipeline ────────── */

typedef struct {
    hipGraphExec_t exec[2];       /* double-buffered */
    hipStream_t    streams[2];
    int            active;        /* 0 or 1          */
    int            uploaded;      /* bitmask          */
} hgb_pipeline_t;

/**
 * Create a double-buffered graph launch pipeline.
 *
 * Pre-uploads graph to GPU memory. Alternates between
 * two exec instances for overlapped launch+update.
 */
hipError_t hgb_pipeline_create(
    hipGraph_t       graph,
    hgb_pipeline_t*  out
);

/**
 * Launch the active graph and swap buffers.
 * ~44μs per launch (measured on gfx1030).
 */
hipError_t hgb_pipeline_launch(hgb_pipeline_t* pipe);

/**
 * Update kernel params on the inactive buffer.
 * Takes effect on next hgb_pipeline_launch().
 */
hipError_t hgb_pipeline_update_kernel(
    hgb_pipeline_t*       pipe,
    hipGraphNode_t        node,
    hipKernelNodeParams*  params
);

void hgb_pipeline_destroy(hgb_pipeline_t* pipe);

/* ── Gap 53: Dynamic Shape Management ───────────── */

typedef struct {
    int*             bucket_sizes;
    int              num_buckets;
    hipGraphExec_t*  execs;       /* one per bucket     */
    hipGraph_t*      graphs;      /* one per bucket     */
    void**           static_bufs; /* pinned I/O buffers */
} hgb_shape_pool_t;

/**
 * Create a shape-bucketed graph pool.
 *
 * @param graph_fn    Callback that captures a graph for a given size.
 *                    Signature: hipError_t fn(int size, hipGraph_t* out)
 * @param buckets     Array of bucket sizes (e.g., {1,2,4,8,16,32,64})
 * @param num_buckets Length of buckets array
 * @param out         Populated shape pool
 */
typedef hipError_t (*hgb_capture_fn)(int size, hipGraph_t* out, void* ctx);

hipError_t hgb_shape_pool_create(
    hgb_capture_fn    graph_fn,
    void*             ctx,
    const int*        buckets,
    int               num_buckets,
    hgb_shape_pool_t* out
);

/**
 * Launch the graph for the given input size.
 * Selects smallest bucket >= size, pads if needed.
 * Returns the actual bucket size used.
 */
hipError_t hgb_shape_pool_launch(
    hgb_shape_pool_t* pool,
    int               input_size,
    hipStream_t       stream,
    int*              actual_bucket
);

/**
 * Update kernel params in-place for a specific bucket.
 * Uses hipGraphExecKernelNodeSetParams (no re-instantiate).
 */
hipError_t hgb_shape_pool_update_params(
    hgb_shape_pool_t*     pool,
    int                   bucket_index,
    hipGraphNode_t        node,
    hipKernelNodeParams*  params
);

void hgb_shape_pool_destroy(hgb_shape_pool_t* pool);

/* ── Gap 54: Capture Compositor ─────────────────── */

typedef struct {
    hipGraph_t      composed;
    hipGraphExec_t  exec;
    hipGraphNode_t* child_nodes;
    int             num_children;
} hgb_composed_graph_t;

/**
 * Compose multiple captured sub-graphs into one.
 *
 * Captures each sub-graph independently, then attaches
 * them as child graph nodes in a parent graph.
 * Dependencies are expressed via the deps array.
 */
hipError_t hgb_compose_graphs(
    hipGraph_t*           sub_graphs,
    int                   count,
    const int*            deps,      /* dep[i] = parent of i, -1=root */
    hgb_composed_graph_t* out
);

/**
 * Update a child sub-graph without re-instantiation.
 */
hipError_t hgb_compose_update_child(
    hgb_composed_graph_t* comp,
    int                   child_index,
    hipGraph_t            new_sub_graph
);

hipError_t hgb_compose_launch(
    hgb_composed_graph_t* comp,
    hipStream_t           stream
);

void hgb_compose_destroy(hgb_composed_graph_t* comp);

#ifdef __cplusplus
}
#endif
```

### Kernel Specifications

#### Gap 51 — Conditional Dispatch Kernel

```
File:       src/conditional_bridge.hip
Purpose:    Read device-side condition flag + toggle graph nodes
Algorithm:
  1. hgb_conditional_create():
     - For each branch graph, instantiate into the supergraph
       via hipGraphAddChildGraphNode
     - Record node handles per branch
     - hipGraphInstantiate the supergraph
     - Disable all branch nodes via hipGraphNodeSetEnabled(0)
  2. hgb_conditional_launch():
     - Disable all branches: hipGraphNodeSetEnabled(exec, node, 0)
     - Enable selected branch: hipGraphNodeSetEnabled(exec, node, 1)
     - hipGraphLaunch(exec, stream)
  3. hgb_conditional_launch_by_flag():
     - hipMemcpyDtoH(&flag, d_flag, sizeof(int))  // async copy
     - hipStreamSynchronize(stream)
     - Dispatch to hgb_conditional_launch(cond, flag, stream)

Performance:  ~90% of native conditional nodes for if/else
Limitation:   While-loop semantics NOT supported (no device-side loop)
Memory:       ~2× graph size (all branches allocated)
```

#### Gap 52 — Rapid Launch Pipeline Kernel

```
File:       src/launch_pipeline.hip
Purpose:    Minimize graph launch latency via pre-upload + double-buffer
Algorithm:
  1. hgb_pipeline_create():
     - hipGraphInstantiate × 2 (double buffer)
     - hipStreamCreate × 2
     - hipGraphUpload both execs to respective streams
  2. hgb_pipeline_launch():
     - hipGraphLaunch(exec[active], streams[active])
     - active ^= 1  (swap buffer)
     - hipGraphUpload(exec[active], streams[active])  // pre-stage next
  3. hgb_pipeline_update_kernel():
     - hipGraphExecKernelNodeSetParams(exec[!active], node, params)
     - Updates staged on inactive buffer, live on next launch

Performance:  ~44μs per launch (measured), ~95% with double-buffer
Limitation:   Host round-trip required (no device-side graph trigger)
Memory:       2× exec memory (double-buffered)
```

#### Gap 53 — Shape Bucketing Manager

```
File:       src/shape_manager.hip
Purpose:    Manage graph instances across dynamic input shapes
Algorithm:
  1. hgb_shape_pool_create():
     - For each bucket size, call graph_fn(size, &graph)
     - hipGraphInstantiate each graph
     - Allocate pinned I/O buffer per bucket (hipMalloc)
  2. hgb_shape_pool_launch():
     - Binary search for smallest bucket >= input_size
     - Copy input to static_bufs[bucket] (only input_size elements)
     - hipGraphLaunch(execs[bucket], stream)
     - Return actual_bucket for output slicing
  3. hgb_shape_pool_update_params():
     - hipGraphExecKernelNodeSetParams on the target bucket exec
     - Updates gridDim, blockDim, kernelParams without re-instantiate

Performance:  ~90-95% of static (padding overhead 5-10%)
Limitation:   Bucket count × graph memory; known size range required
Memory:       num_buckets × (graph_exec + I/O buffer)
```

#### Gap 54 — Capture Compositor

```
File:       src/capture_compositor.hip
Purpose:    Compose sub-graphs into a single launchable graph
Algorithm:
  1. hgb_compose_graphs():
     - hipGraphCreate parent graph
     - For each sub_graph[i]:
       - hipGraphAddChildGraphNode(&child_nodes[i], parent, ...)
       - If deps[i] >= 0, add dependency edge to child_nodes[deps[i]]
     - hipGraphInstantiate(parent, &exec)
  2. hgb_compose_update_child():
     - hipGraphExecChildGraphNodeSetParams(exec, child_nodes[i], new)
     - No re-instantiation needed
  3. hgb_compose_launch():
     - hipGraphLaunch(exec, stream)

Performance:  ~95% of nested capture
Limitation:   Requires pre-captured sub-graphs (can't nest captures)
Memory:       Parent graph + child references (lightweight)
```

### Internal Helpers

```
File:       src/graph_utils.hip
Purpose:    Shared utilities across bridge kernels

Functions:
  hgb_find_kernel_nodes()   — Walk graph, return all kernel nodes
  hgb_clone_and_modify()    — Clone graph + apply param modifications
  hgb_validate_gfx1030()    — Check GPU arch, return error if wrong
  hgb_pool_alloc/free()     — Thread-safe graph exec pool
  hgb_log()                 — Debug logging (env HGB_DEBUG=1)
```

---

## Layer 2: `hipgraph_bridge/` (Python)

PyTorch integration module. Registers custom ops that are
transparent to `torch.compile` and `torch.cuda.graph()`.

### Module Structure

```python
hipgraph_bridge/
├── __init__.py             # Version, auto-init Layer 1
├── _C.py                   # ctypes bindings to libhipgraph_bridge.so
├── conditional.py          # torch.cond-compatible graph branching
├── shape_bucketing.py      # CUDAGraph pool keyed by shape
├── graph_manager.py        # Drop-in CUDAGraph replacement
├── compile_backend.py      # Custom torch.compile backend
└── ops.py                  # torch.library custom op registration
```

### Key Classes

#### `BridgedCUDAGraph` — Drop-in CUDAGraph Replacement

```python
class BridgedCUDAGraph:
    """Drop-in replacement for torch.cuda.CUDAGraph that
    automatically applies bridge strategies when needed.

    Usage:
        g = BridgedCUDAGraph()
        with g.capture(dynamic_shapes=True, buckets=[1,4,8,16,32]):
            output = model(static_input)
        g.replay(actual_batch_size=12)  # auto-selects bucket 16
    """

    def capture(self, *, dynamic_shapes=False, buckets=None,
                conditional_branches=None):
        """Enhanced capture context manager.

        Args:
            dynamic_shapes: Enable shape bucketing (gap 53)
            buckets: List of bucket sizes (default: [1,2,4,8,16,32,64])
            conditional_branches: Dict of {name: callable} for
                conditional execution (gap 51)
        """
        ...

    def replay(self, *, batch_size=None, branch=None):
        """Launch with optional dynamic dispatch.

        Args:
            batch_size: If dynamic_shapes, selects appropriate bucket
            branch: If conditional, selects branch by name/index
        """
        ...
```

#### `ShapeBucketPool` — Shape-Aware Graph Pool

```python
class ShapeBucketPool:
    """Manages a pool of CUDAGraph instances across shape buckets.

    Follows the vLLM/SGLang pattern. Captures one graph per bucket
    at first use, then replays the cached graph on subsequent calls.

    Usage:
        pool = ShapeBucketPool(model, buckets=[1, 4, 8, 16, 32, 64])
        output = pool(input_tensor)  # auto-selects bucket, pads, runs
    """

    def __init__(self, model_fn, buckets, warmup=True):
        ...

    def __call__(self, *args):
        """Auto-bucket, pad, replay, slice output."""
        ...

    @property
    def memory_overhead(self):
        """Returns total GPU memory used by all bucket graphs."""
        ...
```

#### Custom Op Registration

```python
# ops.py — Register with torch.library for torch.compile compat

@torch.library.custom_op("hipgraph_bridge::conditional_launch",
                          mutates_args=())
def conditional_launch(
    pred: torch.Tensor,
    true_graph: torch.ScriptObject,
    false_graph: torch.ScriptObject,
) -> torch.Tensor:
    """Launch one of two pre-captured graphs based on pred.

    Works with torch.compile — the dispatcher routes to the
    bridge .so at runtime.
    """
    ...

@conditional_launch.register_fake
def _(pred, true_graph, false_graph):
    """Shape inference for torch.compile tracing."""
    return torch.empty_like(pred)
```

#### Compile Backend

```python
# compile_backend.py — Optional torch.compile backend override

def hipgraph_bridge_backend(gm, example_inputs):
    """Custom torch.compile backend that:

    1. Analyzes the FX graph for gap capabilities
    2. If all ops are natively supported → delegate to inductor
    3. If gap ops detected → wrap in bridge, compile remainder
    4. Returns optimized callable

    Usage:
        model = torch.compile(model, backend="hipgraph_bridge")
    """
    ...
```

---

## Layer 3: `libcudagraph_compat.so` (Optional)

LD_PRELOAD shim for running **unmodified CUDA binaries** on gfx1030.
Intercepts `cuda*` Graph API symbols and routes them to HIP + bridge.

> ⚠️ This layer is optional. Most users on ROCm already use HIP via
> PyTorch and do not need CUDA symbol interception.

### Mechanism

```c
/* cuda_intercept.c — LD_PRELOAD symbol interception */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include "hipgraph_bridge.h"

/*
 * Intercept cudaGraphCreate → hipGraphCreate
 * (1:1 mapping, 50/54 capabilities)
 */
cudaError_t cudaGraphCreate(cudaGraph_t* graph, unsigned int flags) {
    return (cudaError_t)hipGraphCreate(
        (hipGraph_t*)graph, flags
    );
}

/*
 * Intercept cudaGraphConditionalHandleCreate → bridge
 * (Gap 51: no native HIP equivalent)
 */
cudaError_t cudaGraphConditionalHandleCreate(
    /* CUDA 12.3 conditional params */
    ...
) {
    /* Route to Layer 1 bridge */
    return (cudaError_t)hgb_conditional_create(...);
}

/* ... ~90 more symbol mappings ... */
```

### Symbol Routing Table

| CUDA Symbol | Route | Layer |
|-------------|-------|:-----:|
| `cudaGraphCreate` | `hipGraphCreate` | native |
| `cudaGraphLaunch` | `hipGraphLaunch` | native |
| `cudaGraphInstantiate` | `hipGraphInstantiate` | native |
| ... (40+ native 1:1 maps) | ... | native |
| `cudaGraphAddDependencies` | `hipGraphAddDependencies` | partial |
| ... (6 partial maps) | ... | partial |
| `cudaGraphConditionalHandleCreate` | `hgb_conditional_create` | bridge |
| `cudaGraphInstantiateFlagDeviceLaunch` | `hgb_pipeline_create` | bridge |
| ... (4 gap bridges) | ... | bridge |
| `cudaGraphAddDependencies_v2` | error + log | unsupported |
| ... (20+ v2 APIs) | ... | unsupported |

### Usage

```bash
# Run unmodified CUDA binary on gfx1030
LD_PRELOAD=libcudagraph_compat.so ./my_cuda_app

# With debug logging
HGB_DEBUG=1 LD_PRELOAD=libcudagraph_compat.so ./my_cuda_app
```

---

## File Tree

```
ai/build/src/hipgraph-bridge/
├── CMakeLists.txt
├── README.md
├── include/
│   └── hipgraph_bridge.h              # Public C API (see Layer 1)
├── src/
│   ├── init.cpp                       # hgb_init/shutdown, version
│   ├── conditional_bridge.hip         # Gap 51 implementation
│   ├── launch_pipeline.hip            # Gap 52 implementation
│   ├── shape_manager.hip              # Gap 53 implementation
│   ├── capture_compositor.hip         # Gap 54 implementation
│   ├── graph_utils.hip                # Shared helpers
│   └── compat/
│       └── cuda_intercept.c           # Layer 3 LD_PRELOAD shim
├── python/
│   └── hipgraph_bridge/
│       ├── __init__.py
│       ├── _C.py                      # ctypes bindings
│       ├── conditional.py
│       ├── shape_bucketing.py
│       ├── graph_manager.py
│       ├── compile_backend.py
│       └── ops.py                     # torch.library registration
├── tests/
│   ├── test_conditional.hip           # Gap 51 C tests
│   ├── test_pipeline.hip              # Gap 52 C tests
│   ├── test_shapes.hip                # Gap 53 C tests
│   ├── test_compositor.hip            # Gap 54 C tests
│   ├── test_compat.py                 # Layer 3 integration
│   └── test_torch_integration.py      # Layer 2 PyTorch tests
└── benchmarks/
    ├── bench_conditional.py
    ├── bench_pipeline.py
    ├── bench_shapes.py
    └── bench_native_comparison.py     # vs native HIP baseline
```

---

## Build System

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.21)
project(hipgraph_bridge LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_ARCHITECTURES gfx1030)

# ── Layer 1: Core bridge library ────────────────────
add_library(hipgraph_bridge SHARED
    src/init.cpp
    src/conditional_bridge.hip
    src/launch_pipeline.hip
    src/shape_manager.hip
    src/capture_compositor.hip
    src/graph_utils.hip
)
target_include_directories(hipgraph_bridge PUBLIC include/)
target_link_libraries(hipgraph_bridge PRIVATE
    hip::amdhip64
    hip::hiprtc
)
set_target_properties(hipgraph_bridge PROPERTIES
    VERSION 0.1.0
    SOVERSION 0
    OUTPUT_NAME hipgraph_bridge
)

# ── Layer 3: CUDA compat shim (optional) ────────────
option(BUILD_CUDA_COMPAT "Build CUDA compatibility shim" OFF)
if(BUILD_CUDA_COMPAT)
    add_library(cudagraph_compat SHARED
        src/compat/cuda_intercept.c
    )
    target_link_libraries(cudagraph_compat PRIVATE
        hipgraph_bridge
        ${CMAKE_DL_LIBS}
    )
endif()

# ── Tests ───────────────────────────────────────────
enable_testing()
add_executable(test_conditional tests/test_conditional.hip)
add_executable(test_pipeline    tests/test_pipeline.hip)
add_executable(test_shapes      tests/test_shapes.hip)
add_executable(test_compositor  tests/test_compositor.hip)
foreach(test IN ITEMS conditional pipeline shapes compositor)
    target_link_libraries(test_${test} PRIVATE hipgraph_bridge)
    add_test(NAME ${test} COMMAND test_${test})
endforeach()
```

### Build Commands

```bash
cd ai/build/src/hipgraph-bridge

# Configure
cmake -B build -GNinja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_HIP_ARCHITECTURES=gfx1030

# Build
cmake --build build -j$(nproc)

# Test
cd build && ctest --output-on-failure

# Install (optional)
cmake --install build --prefix /opt/rocm/lib/hipgraph_bridge

# Build with CUDA compat layer
cmake -B build -GNinja -DBUILD_CUDA_COMPAT=ON ...
```

### Python Install

```bash
cd ai/build/src/hipgraph-bridge
pip install -e python/   # or: uv pip install -e python/ --no-deps
```

---

## Test Matrix

| Test | Gap | What It Verifies | Pass Criteria |
|------|:---:|------------------|---------------|
| `test_conditional` | 51 | 2-branch if/else supergraph | Correct branch selected, output matches |
| `test_conditional_flag` | 51 | Device-side flag dispatch | Flag=0 → branch A, flag=1 → branch B |
| `test_pipeline_latency` | 52 | Launch latency ≤ 50μs | Mean < 50μs over 1000 iterations |
| `test_pipeline_double` | 52 | Double-buffer correctness | Output matches sequential execution |
| `test_shape_buckets` | 53 | Bucket selection + padding | All sizes 1–64 produce correct output |
| `test_shape_update` | 53 | In-place param update | gridDim changes without re-instantiate |
| `test_compose_linear` | 54 | Linear sub-graph chain | A→B→C composition matches sequential |
| `test_compose_update` | 54 | Child graph hot-swap | Updated child produces new output |
| `test_torch_bridged` | all | BridgedCUDAGraph end-to-end | PyTorch model runs with all bridges |
| `test_compile_backend` | all | torch.compile integration | Compiled model produces correct output |
| `test_compat_shim` | all | LD_PRELOAD interception | CUDA calls route to HIP correctly |

### Benchmark Targets

| Benchmark | Baseline | Target |
|-----------|----------|--------|
| Conditional 2-branch | native HIP (no cond) | ≥ 85% throughput |
| Pipeline launch | hipGraphLaunch (cold) | ≤ 50μs |
| Shape bucketing (7 buckets) | static graph replay | ≥ 90% throughput |
| Composition (3 sub-graphs) | monolithic graph | ≥ 90% throughput |

---

## Implementation Priority

| Phase | Scope | Effort | Impact |
|:-----:|-------|:------:|:------:|
| 1 | Gap 53 shape manager + Python bucketing | 1 week | **High** — most common gap |
| 2 | Gap 51 conditional bridge | 1 week | **High** — enables branching |
| 3 | Gap 52 pipeline + Gap 54 compositor | 3 days | Medium — lower severity gaps |
| 4 | Python integration (Layer 2) | 3 days | **High** — user-facing API |
| 5 | CUDA compat shim (Layer 3) | 2 days | Low — niche use case |
| 6 | Benchmarks + documentation | 2 days | Medium — validation |

**Total estimated scope: ~4 weeks for full implementation.**

Phase 1 + 2 alone cover the two **High severity** gaps and deliver
~96% effective workload coverage via the Python layer.

---

## Local Resources for Implementation

| Resource | Path | Provides |
|----------|------|----------|
| HIP headers | `ai/build/resources/hip/include/` | API definitions |
| HIPIFY mappings | `ai/build/tools/HIPIFY/src/` | CUDA→HIP symbol table |
| aiter JIT | `ai/build/resources/aiter/aiter/jit/` | .so build pattern |
| AOTriton | `ai/build/resources/aotriton/` | Drop-in .so pattern |
| Triton cache | `ai/build/resources/kernel-cache/triton-gfx1031/` | HSACO examples |
| kernel-learning | `ai/build/resources/kernel-learning/09-pytorch-custom-ops/` | Op registration |
| TheRock CMake | `ai/build/resources/TheRock/cmake/` | Build infrastructure |
| hipGraph report | `ai/build/resources/hipgraph-analysis-report.md` | Performance data |
