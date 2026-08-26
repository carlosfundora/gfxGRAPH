## 2024-05-24 - [Optimize Telemetry Function Calls]
**Learning:** Found that telemetry and stat tracking calls (`bump`, `record_replay_us`) in hot paths perform a global boolean fallback check (`if _HAS_RUST_STATS:`) on every invocation, adding unnecessary overhead.
**Action:** Aliased the underlying Rust functions directly upon import to bypass the fallback check entirely, significantly reducing per-call overhead.
