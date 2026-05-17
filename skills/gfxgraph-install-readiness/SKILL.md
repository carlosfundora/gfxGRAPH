---
name: gfxgraph-install-readiness
description: Assess whether a target environment is ready for gfxGRAPH installation and surface blockers before any changes are made.
---

# gfxGRAPH Installation Readiness

**Goal**
Check whether gfxGRAPH can be installed cleanly in the requested target environment.

## When to use this skill
- Before installing gfxGRAPH into any target directory or venv.
- When the environment, ROCm stack, or Python path needs a readiness check first.
- When you need a clear go/no-go verdict plus blockers.
- When you need to avoid mutating a target until the install path is safe.

## Do not use this skill for
- Performing the install itself.
- Reconfiguring an already installed gfxGRAPH environment.
- Runtime testing after installation.

## How to use it
1. Confirm the repo path and the exact target directory the user wants to install into.
2. Check the basics: `uv` availability, Python 3.12+, ROCm GPU visibility, and whether the target venv already exists.
3. Verify the target can import from the gfxGRAPH package path (`python/`) without conflicting installs.
4. Report readiness with any blockers, then stop before making changes.

## Decision tree
- If the target is missing prerequisites, report `not ready` and list the blockers.
- If the target already contains a working gfxGRAPH install, confirm that the user wants to replace it before proceeding.
- If ROCm or Python is missing, stop and point to the install or configuration skill instead.

## Readiness checks
- Confirm the target directory is writable.
- Confirm the target environment is using ROCm, not CUDA.
- Confirm any existing gfxGRAPH install is intentional and not stale.
- Confirm the package path and runtime match the expected repo layout.
- Confirm the target environment can run the repo's Python tests later.
