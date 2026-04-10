"""
End-to-end PyTorch integration test for gfxGRAPH.
"""
import torch
from hipgraph_bridge import BridgedCUDAGraph
from hipgraph_bridge.shape_bucketing import ShapeBucketPool
from hipgraph_bridge.conditional import ConditionalGraph


def test_shape_bucket_pool():
    """Test ShapeBucketPool with a simple model."""
    if not torch.cuda.is_available():
        print("SKIP: no GPU")
        return

    model = torch.nn.Linear(64, 64).cuda().eval()

    def model_fn(x):
        return model(x)

    pool = ShapeBucketPool(model_fn, buckets=[1, 4, 8, 16, 32])

    # Test various input sizes
    for size in [1, 3, 7, 16, 30]:
        x = torch.randn(size, 64, device="cuda")
        out = pool(x)
        assert out.shape == (size, 64), f"Expected ({size}, 64), got {out.shape}"
        print(f"  Size {size} → bucket {pool.select_bucket(size)}: OK")

    print("  test_shape_bucket_pool PASSED")


def test_conditional_graph():
    """Test ConditionalGraph branch selection."""
    if not torch.cuda.is_available():
        print("SKIP: no GPU")
        return

    def branch_a(x):
        return x * 2.0

    def branch_b(x):
        return x + 100.0

    cg = ConditionalGraph()
    cg.add_branch("double", branch_a)
    cg.add_branch("add100", branch_b)

    example = torch.ones(16, device="cuda")
    cg.capture(example)

    out_a = cg.run("double", torch.ones(16, device="cuda"))
    assert torch.allclose(out_a, torch.full((16,), 2.0, device="cuda"))
    print("  Branch 'double': OK")

    out_b = cg.run("add100", torch.ones(16, device="cuda"))
    assert torch.allclose(out_b, torch.full((16,), 101.0, device="cuda"))
    print("  Branch 'add100': OK")

    print("  test_conditional_graph PASSED")


def test_bridged_cuda_graph():
    """Test BridgedCUDAGraph standard capture."""
    if not torch.cuda.is_available():
        print("SKIP: no GPU")
        return

    model = torch.nn.Linear(32, 32).cuda().eval()
    static_input = torch.randn(8, 32, device="cuda")

    # Warmup
    with torch.no_grad():
        _ = model(static_input)
    torch.cuda.synchronize()

    g = BridgedCUDAGraph()
    with g.capture():
        g._static_output = model(static_input)

    g.replay()
    torch.cuda.synchronize()
    print("  Standard capture+replay: OK")

    print("  test_bridged_cuda_graph PASSED")


if __name__ == "__main__":
    print("=== test_torch_integration ===")
    test_shape_bucket_pool()
    test_conditional_graph()
    test_bridged_cuda_graph()
    print("ALL PASSED")
