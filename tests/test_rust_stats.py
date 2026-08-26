
from gfxgraph._enable import bump, record_replay_us, stats, _HAS_RUST_STATS

def test_rust_stats_aliased():
    if not _HAS_RUST_STATS:
        return
    import rs_gfxgraph_stats
    assert bump == rs_gfxgraph_stats.bump
    assert record_replay_us == rs_gfxgraph_stats.record_replay_us
