---
name: gfxgraph-configure
description: Configure or reconfigure gfxGRAPH runtime behavior, environment variables, and replay modes after installation.
---

# gfxGRAPH Configuration

**Goal**
Set up or adjust gfxGRAPH runtime behavior after installation.

## When to use this skill
- When a user wants to enable, disable, or tune gfxGRAPH behavior.
- When switching between standard, adaptive, and hot replay modes.
- When adjusting VRAM caps, logging, or ROCm environment variables.
- When an installed environment needs to be reconfigured without reinstalling it.

## Do not use this skill for
- Checking whether the environment is ready to install.
- Installing gfxGRAPH into a target directory.
- Confirming functionality after a change.

## How to use it
1. Inspect the current runtime mode and relevant environment variables.
2. Apply the requested configuration changes for the target process or shell.
3. Restart the process if the setting is process-startup-only.
4. Confirm the new configuration with `gfxgraph.stats()` or `gfxgraph.health_check()`.

## Configuration areas
- Replay mode: `GFXGRAPH_REPLAY_MODE=standard|adaptive|hot`
- Legacy hot toggle: `GFXGRAPH_REPLAY_HOT_MODE=1`
- VRAM safety: `GFXGRAPH_VRAM_CAP`
- Logging: `HGB_LOG_LEVEL`
- ROCm device mapping: `HSA_OVERRIDE_GFX_VERSION`

## Reconfiguration notes
- Preserve eager fallback behavior when changing replay settings.
- Prefer explicit mode changes over hidden env drift.
- Recheck the effective mode after every change.
- Keep legacy compatibility toggles only when the user explicitly relies on them.

## Common reconfiguration cases
- Switch `GFXGRAPH_REPLAY_MODE` between `standard`, `adaptive`, and `hot`.
- Reduce `GFXGRAPH_VRAM_CAP` if capture pressure is too high.
- Increase logging for debugging, then restore a quieter level.
