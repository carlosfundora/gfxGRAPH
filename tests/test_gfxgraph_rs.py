import pytest

try:
    import gfxgraph_rs
    _HAS_RUST_EXT = True
except ImportError:
    _HAS_RUST_EXT = False

@pytest.mark.skipif(not _HAS_RUST_EXT, reason="gfxgraph_rs not built")
def test_rust_bucket_selector():
    # Unit test for Rust component
    selector = gfxgraph_rs.BucketSelector([1, 4, 8, 16, 32])

    # Exact match
    assert selector.select_bucket(4) == 4

    # Next larger match
    assert selector.select_bucket(2) == 4
    assert selector.select_bucket(5) == 8

    # First bucket
    assert selector.select_bucket(1) == 1

    # Error path test for invalid input (too large)
    with pytest.raises(ValueError, match="Input size 33 exceeds largest bucket 32. Add a larger bucket."):
        selector.select_bucket(33)
