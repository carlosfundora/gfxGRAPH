import time
import json
import sys
from unittest.mock import MagicMock

# Mock torch before anything imports it
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

mock_torch.Tensor = MockTensor
mock_torch.__mock__ = True
sys.modules['torch'] = mock_torch

from hipgraph_bridge.conditional import ConditionalGraph

def main():
    cg = ConditionalGraph()
    cg.add_branch("a", lambda x: "result_a")
    cg.add_branch("b", lambda x: "result_b")

    example = mock_torch.Tensor()
    cg.capture(example)

    iters = 100000
    input_tensor = mock_torch.Tensor()

    t0 = time.perf_counter()
    for _ in range(iters):
        cg.run("a", input_tensor)
        cg.run("b", input_tensor)
    t1 = time.perf_counter()
    duration = t1 - t0

    result = {
        "candidate": "python/hipgraph_bridge/conditional.py",
        "implementation": "after",
        "command": "python benchmarks/bench_conditional_mock.py",
        "timestamp": "2026-05-08T00:00:00Z",
        "iterations": iters * 2,
        "input_description": "Alternating branch execution (mocked GPU)",
        "duration_ms": duration * 1000,
        "throughput": f"{(iters * 2) / duration:.2f} ops/sec",
    }

    with open(".jules/verification/rusty/after-benchmark.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Benchmark saved to .jules/verification/rusty/after-benchmark.json")

if __name__ == "__main__":
    main()
