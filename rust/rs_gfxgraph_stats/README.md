# rs_gfxgraph_stats

Rust crate in the gfxGRAPH workspace. Thread-safe global statistics collector for gfxGRAPH graph execution — tracks capture count, replay count, fallback count, validation failures, total replay latency, and per-replay average microseconds via a process-global `DashMap`-backed store.

## Navigation

- Package manifest: `Cargo.toml`
- Change history: `CHANGELOG.md`
- Canonical repository documentation: consult the nearest repository `docs/` directory and workspace-level architecture notes.

## Maintenance

Keep this README as a crate-local routing page. Put durable design details in canonical repository documentation and record crate-specific changes in `CHANGELOG.md`.
