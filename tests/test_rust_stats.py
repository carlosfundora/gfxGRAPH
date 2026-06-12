import pytest
from gfxgraph._enable import bump, record_replay_us, stats, _HAS_RUST_STATS

def test_rust_stats_available():
    assert _HAS_RUST_STATS is True, "Rust stats module should be loaded"

pytestmark = pytest.mark.skipif(
    not _HAS_RUST_STATS, reason="Rust stats module not installed (Tier 1 mode)"
)

def test_bump():
    from rs_gfxgraph_stats import reset
    reset()

    bump("capture_count", 5)
    bump("replay_count", 1)
    bump("fallback_count")
    bump("custom_metric", 10)

    s = stats()
    assert s["capture_count"] == 5
    assert s["replay_count"] == 1
    assert s["fallback_count"] == 1
    assert s["validation_failures"] == 0
    assert s["custom_metric"] == 10

def test_record_replay_us():
    from rs_gfxgraph_stats import reset
    reset()

    record_replay_us(100.0)
    record_replay_us(200.0)

    s = stats()
    assert s["replay_count"] == 2
    assert s["avg_replay_us"] == 150.0
