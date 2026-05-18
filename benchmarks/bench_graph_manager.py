import time
import json
import sys
from unittest.mock import MagicMock
import torch

mock_torch = MagicMock()
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available.return_value = False
class MockTensor:
    def __init__(self):
        self.is_cuda = True
    def clone(self):
        return MockTensor()
    def copy_(self, other):
        pass
    def contiguous(self):
        return self
    def is_contiguous(self):
        return True
    def abs(self):
        return self
    def max(self):
        return self
    def item(self):
        return 0.0

mock_torch.Tensor = MockTensor
mock_torch.__mock__ = True
mock_torch.allclose.return_value = True
mock_torch.no_grad = MagicMock()
sys.modules['torch'] = mock_torch

from hipgraph_bridge.graph_manager import BridgedCUDAGraph

def main():
    def model_fn(x):
        return mock_torch.Tensor()

    # Disable Rust extension for the baseline
    sys.modules['rs_gfxgraph'] = None

    import gfxgraph._enable
    gfxgraph._enable._validate_mode = True

    g = BridgedCUDAGraph()
    with g.capture(model_fn=model_fn):
        g._static_output = mock_torch.Tensor()

    iters = 100000

    input_tensor = mock_torch.Tensor()

    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay(input_tensor=input_tensor)
    t1 = time.perf_counter()

    duration = t1 - t0

    result = {
        "candidate": "python/hipgraph_bridge/graph_manager.py",
        "implementation": "before",
        "command": "python benchmarks/bench_graph_manager.py",
        "timestamp": "2026-05-08T00:00:00Z",
        "iterations": iters,
        "input_description": "Repeated graph replay with validation enabled (mocked GPU)",
        "duration_ms": duration * 1000,
        "throughput": f"{iters / duration:.2f} ops/sec",
    }

    with open(".jules/verification/rusty/before-benchmark.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Benchmark saved to .jules/verification/rusty/before-benchmark.json")

if __name__ == "__main__":
    main()
