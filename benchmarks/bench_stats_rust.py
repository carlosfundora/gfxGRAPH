import time
import threading

import rs_gfxgraph_stats

def run_bench(iterations):
    t0 = time.perf_counter()
    for _ in range(iterations):
        rs_gfxgraph_stats.bump("capture_count", 1)
        rs_gfxgraph_stats.record_replay_us(42.5)
    t1 = time.perf_counter()
    return t1 - t0

def run_threaded_bench(iterations, threads):
    threads_list = []

    def worker():
        for _ in range(iterations):
            rs_gfxgraph_stats.bump("replay_count", 1)
            rs_gfxgraph_stats.record_replay_us(15.0)

    t0 = time.perf_counter()
    for _ in range(threads):
        t = threading.Thread(target=worker)
        threads_list.append(t)
        t.start()

    for t in threads_list:
        t.join()
    t1 = time.perf_counter()
    return t1 - t0

if __name__ == "__main__":
    import json
    # Reset stats
    rs_gfxgraph_stats.reset()

    # Warmup
    run_bench(1000)

    iters = 100000
    duration = run_bench(iters)

    threaded_iters = 50000
    threads = 4
    duration_threaded = run_threaded_bench(threaded_iters, threads)

    print(json.dumps({
        "candidate": "gfxgraph._enable stats tracker",
        "implementation": "after",
        "command": "python benchmarks/bench_stats_rust.py",
        "timestamp": "2024-05-08T00:00:00Z",
        "iterations": iters,
        "input_description": "Single thread tight loop",
        "duration_ms": duration * 1000,
        "throughput": f"{iters / duration:.2f} ops/sec",
        "notes": f"Threaded (4x50k) duration: {duration_threaded * 1000:.2f} ms"
    }))
