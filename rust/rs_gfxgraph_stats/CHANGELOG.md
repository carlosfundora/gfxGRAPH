# Changelog

All notable crate-specific changes for `rs_gfxgraph_stats` are recorded here.

## [Unreleased]

### Added

- Initial crate scaffold: process-global `StatsCore` backed by `DashMap` and `Mutex<f64>` for lock-free counter increments and serialized replay-latency accumulation. PyO3 functions: `bump`, `record_replay_us`, `stats` (returns dict with `avg_replay_us` and all counters, filling defaults for required counters), and `reset`.
