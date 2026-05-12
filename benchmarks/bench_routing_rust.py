import time
import json
import sys
from unittest.mock import MagicMock
import torch

mock_torch = MagicMock()
mock_torch.Tensor = MagicMock
sys.modules["torch"] = mock_torch
sys.modules["torch.cuda"] = mock_torch.cuda

from hipgraph_bridge.shape_bucketing import ShapeBucketPool
import gfxgraph_rs

pool = ShapeBucketPool(lambda x: x, buckets=[1, 4, 8, 16, 32, 64])
pool._router.mark_warmed_up(1)
pool._router.mark_warmed_up(4)
pool._router.mark_warmed_up(8)
pool._router.mark_warmed_up(16)
pool._router.mark_warmed_up(32)
pool._router.mark_warmed_up(64)

iters = 1000000

t0 = time.perf_counter()
for _ in range(iters):
    pool.route_bucket(31)
t1 = time.perf_counter()

duration = (t1 - t0) * 1000

print(json.dumps({
    "candidate": "python/hipgraph_bridge/shape_bucketing.py",
    "implementation": "after",
    "command": "python benchmarks/bench_routing_rust.py",
    "timestamp": "2024-05-08T00:00:00Z",
    "iterations": iters,
    "input_description": "Repeated bucket size lookup and state validation",
    "duration_ms": duration,
    "throughput": f"{iters / (duration / 1000):.2f} ops/sec"
}))
