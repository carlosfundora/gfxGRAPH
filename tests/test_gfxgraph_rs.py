import pytest

try:
    from rs_gfxgraph import BucketRouter
    _HAS_RUST_EXT = True
except ImportError:
    _HAS_RUST_EXT = False

@pytest.mark.skipif(not _HAS_RUST_EXT, reason="rs_gfxgraph not built")
def test_rust_bucket_selector():
    # Unit test for Rust component
    router = BucketRouter([1, 4, 8, 16, 32])

    # Exact match
    assert router.route(4) == (4, 1)

    # Next larger match
    assert router.route(2) == (4, 1)
    assert router.route(5) == (8, 1)

    # First bucket
    assert router.route(1) == (1, 1)

    # State validation
    router.mark_warmed_up(4)
    assert router.route(2) == (4, 0)

    router.mark_failed(8)
    assert router.route(5) == (8, 2)

    # Error path test for invalid input (too large)
    with pytest.raises(ValueError, match="Input size 33 exceeds largest bucket 32. Add a larger bucket."):
        router.route(33)
