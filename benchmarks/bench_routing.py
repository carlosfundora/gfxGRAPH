import time
import json
import sys
from unittest.mock import MagicMock
import torch

mock_torch = MagicMock()
mock_torch.Tensor = MagicMock
sys.modules["torch"] = mock_torch
sys.modules["torch.cuda"] = mock_torch.cuda

import gfxgraph_rs
_HAS_RUST_EXT = False
sys.modules["gfxgraph_rs"] = MagicMock()

from hipgraph_bridge.shape_bucketing import ShapeBucketPool

pool = ShapeBucketPool(lambda x: x, buckets=[1, 4, 8, 16, 32, 64])
pool._router = None
pool._warmed_up = {1, 4, 8, 16, 32, 64}

iters = 1000000

t0 = time.perf_counter()
for _ in range(iters):
    pool.route_bucket(31)
t1 = time.perf_counter()

duration = (t1 - t0) * 1000

print(json.dumps({
    "candidate": "python/hipgraph_bridge/shape_bucketing.py",
    "implementation": "before",
    "command": "python benchmarks/bench_routing.py",
    "timestamp": "2024-05-08T00:00:00Z",
    "iterations": iters,
    "input_description": "Repeated bucket size lookup and state validation",
    "duration_ms": duration,
    "throughput": f"{iters / (duration / 1000):.2f} ops/sec"
}))
