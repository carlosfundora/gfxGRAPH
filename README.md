# gfxGRAPH

Drop-in CUDA Graph → HIP Graph translation layer for AMD gfx1030 (RDNA2).

Bridges all 4 CUDA Graph parity gaps on ROCm, enabling dynamic CUDA→HIP
graph call routing with maximum performance and minimum complexity.

## Target

- **GPU:** AMD Radeon RX 6700 XT (gfx1030, RDNA2)
- **ROCm:** 7.2.0+
- **PyTorch:** 2.9+ (ROCm build)

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   User Application                    │
├──────────────┬───────────────────┬───────────────────┤
│   PyTorch    │   Direct HIP C   │  Unmodified CUDA  │
├──────────────┼───────────────────┼───────────────────┤
│  Layer 2     │                   │  Layer 3          │
│  hipgraph_   │                   │  libcudagraph_    │
│  bridge/     │                   │  compat.so        │
│  (Python)    │                   │  (LD_PRELOAD)     │
├──────────────┴───────────────────┴───────────────────┤
│            Layer 1: libhipgraph_bridge.so             │
│     Gap bridges · Routing logic · Kernel pool         │
├──────────────────────────────────────────────────────┤
│         libamdhip64.so  (ROCm · 104 symbols)          │
├──────────────────────────────────────────────────────┤
│              gfx1030 · RDNA2 Hardware                 │
└──────────────────────────────────────────────────────┘
```

### Three Layers

| Layer | Component | Purpose |
|:-----:|-----------|---------|
| 1 | `libhipgraph_bridge.so` | C/HIP library bridging 4 CUDA Graph gaps |
| 2 | `hipgraph_bridge/` (Python) | PyTorch integration, torch.compile compat |
| 3 | `libcudagraph_compat.so` | Optional LD_PRELOAD CUDA→HIP interception |

### Gaps Bridged

| # | Gap | Bridge Strategy | Perf |
|:-:|-----|----------------|:----:|
| 51 | Conditional nodes | Supergraph + `hipGraphNodeSetEnabled` | ~90% |
| 52 | Device-side launch | `hipGraphUpload` + rapid host pipeline | ~95% |
| 53 | Dynamic input shapes | Shape bucketing + param update | ~90-95% |
| 54 | Nested capture | Sequential capture + child graph nodes | ~95% |

## Quick Start

### Build

```bash
cmake -B build -GNinja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_HIP_ARCHITECTURES=gfx1030

cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Python

```bash
pip install -e python/
```

### Usage (Python)

```python
from hipgraph_bridge import BridgedCUDAGraph

g = BridgedCUDAGraph()
with g.capture(dynamic_shapes=True, buckets=[1, 4, 8, 16, 32]):
    output = model(static_input)
g.replay(batch_size=12)  # auto-selects bucket 16
```

### Usage (C/HIP)

```c
#include <hipgraph_bridge.h>

hgb_init();

hgb_shape_pool_t pool;
int buckets[] = {1, 4, 8, 16, 32, 64};
hgb_shape_pool_create(my_capture_fn, NULL, buckets, 6, &pool);
hgb_shape_pool_launch(&pool, actual_size, stream, NULL);

hgb_shutdown();
```

## Routing Strategy

Use the highest abstraction that meets the workload:

| Tier | Stack | Capabilities |
|:----:|-------|:------------:|
| 0 | `torch.compile` only | 31/54 |
| 1 | HIP Graph native | 50/54 |
| 2 | HIP Graph + gfxGRAPH | **54/54** |

## Documentation

- [Design Specification](docs/hipgraph-bridge-design.md)
- [CUDA Parity Matrix](docs/torch-hip-rocm-graph.md)

## License

MIT
