---
name: gfxgraph-install
description: Install gfxGRAPH into a user-specified target directory or environment using the repo's supported uv-based workflow.
---

# gfxGRAPH Installation

**Goal**
Install gfxGRAPH into the target directory specified by the user.

## When to use this skill
- When the user wants gfxGRAPH installed into a specific directory or venv.
- When you need to perform the actual install after readiness checks pass.
- When an install must be done with the repo's supported uv-based workflow.
- When a target already exists and you need to populate it safely.

## Do not use this skill for
- Readiness assessment before install.
- Changing runtime settings after install.
- Verifying runtime behavior after install.

## How to use it
1. Resolve the exact target directory from the user's request.
2. Create or reuse the target venv with `uv venv --seed --relocatable` if needed.
3. Install gfxGRAPH from the repo's package path with `uv sync` in the target environment.
4. Verify the install by importing `gfxgraph` from the target environment.

## Installation rules
- Use `uv`, not `pip`, `poetry`, or manual wheel copying.
- Use the repo's `python/` package entrypoint unless the user's target layout requires a different install path.
- Keep ROCm PyTorch and gfxGRAPH aligned in the same target environment.
- Do not overwrite an existing target install unless the user explicitly wants a reinstall.
- Preserve the user's requested target path exactly.

## Suggested install flow
1. Confirm the target path and whether it should be created or reused.
2. Create the target environment if needed.
3. Sync the package into the target.
4. Run a minimal import check before handing off.
