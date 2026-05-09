# Benchmark Summary

- Before command: `python3 benchmark_bucketing.py before`
- After command: `python3 benchmark_bucketing.py after`
- Before timing: 403.24 ms
- After timing: 196.60 ms
- Percent change: -51.2% (over 2x faster throughput)
- Notes on variance or limitations: The benchmark only tests the specific bucketing logic isolated from PyTorch overhead, making the performance gain clear and localized. PyTorch overhead in real use cases will dilute the end-to-end performance gain, but this confirms the Python-to-Rust migration reduces the hot-path latency.
