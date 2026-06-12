# rs_gfxgraph_toolbox

Policy-free developer toolbox around `rs_gfxgraph_core`.

This crate contains agent/developer-facing helpers for shape bucketing, graph-capture layout analysis, RDNA2 launch planning, and abstract signal-domain helpers such as Hann windows and frequency-bin mapping. It intentionally has no `rs_policy_mesh`, PyO3, audio runtime, voice runtime, or logly dependency so it can be governed as a separate tool boundary.
