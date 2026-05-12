# Benchmark Summary

Before command: `python3 benchmarks/bench_conditional_mock.py` (Before refactor)
After command: `python3 benchmarks/bench_conditional_mock.py` (After refactor)

## Single Thread

Before timing: 63313.80 ms
After timing: 9294.43 ms

Percent change: (63313.80 - 9294.43) / 63313.80 * 100% = ~85.3% reduction in execution time

Notes: Migrating the `ConditionalGraph.run()` method inner-loop out of pure python dictionary and execution logic to rust provides an 85% speedup on mocked bounds checks/fallbacks.
