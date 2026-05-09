import time
import sys

def benchmark_python():
    import threading
    _stats_lock = threading.Lock()
    _stats = {
        "enabled_at": None,
        "capture_count": 0,
        "replay_count": 0,
        "fallback_count": 0,
        "validation_failures": 0,
        "avg_replay_us": 0.0,
        "_total_replay_us": 0.0,
    }

    def bump(counter: str, amount: int = 1) -> None:
        with _stats_lock:
            _stats[counter] = _stats.get(counter, 0) + amount

    def record_replay_us(us: float) -> None:
        with _stats_lock:
            _stats["replay_count"] += 1
            _stats["_total_replay_us"] += us
            _stats["avg_replay_us"] = (
                _stats["_total_replay_us"] / _stats["replay_count"]
            )

    start = time.perf_counter()
    for _ in range(1_000_000):
        bump("capture_count", 1)
        record_replay_us(10.0)
    end = time.perf_counter()
    return (end - start) * 1000

def benchmark_rust():
    sys.path.insert(0, './gfxgraph_rs/target/release/')
    try:
        import gfxgraph_rs
    except ImportError:
        import os
        os.system("cp gfxgraph_rs/target/release/libgfxgraph_rs.so gfxgraph_rs/target/release/gfxgraph_rs.so")
        import gfxgraph_rs

    manager = gfxgraph_rs.StatsManager()
    start = time.perf_counter()
    for _ in range(1_000_000):
        manager.bump("capture_count", 1)
        manager.record_replay_us(10.0)
    end = time.perf_counter()
    return (end - start) * 1000

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'before':
        print(f"Python: {benchmark_python():.2f} ms")
    elif len(sys.argv) > 1 and sys.argv[1] == 'after':
        print(f"Rust: {benchmark_rust():.2f} ms")
