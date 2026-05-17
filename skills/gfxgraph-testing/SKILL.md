---
name: gfxgraph-testing
description: Test gfxGRAPH on real hardware and confirm it is functioning after install or configuration changes.
---

# gfxGRAPH Testing

**Goal**
Confirm gfxGRAPH is functioning on the target machine after install or reconfiguration.

## When to use this skill
- After installing gfxGRAPH.
- After changing configuration or replay modes.
- When you need to confirm the package actually works on a real ROCm GPU.
- When you need a pass/fail answer for the current install state.

## Do not use this skill for
- Preparing the environment for install.
- Applying configuration changes.
- Debugging unrelated ROCm failures before the install is proven.

## How to use it
1. Run the relevant Python integration tests from the repo.
2. Run a direct smoke check with `gfxgraph.health_check()`.
3. Confirm the runtime reports the expected GPU and replay behavior.
4. If a failure appears, capture the exact error before changing anything else.

## Preferred checks
- `python3 -m pytest tests/test_torch_integration.py -v`
- `python3 -c "import gfxgraph; gfxgraph.enable(); print(gfxgraph.health_check())"`
- `gfxgraph.stats()` after a real capture/replay path

## Test rules
- Use real hardware when available.
- Do not rely on mocks for GPU availability in the success path.
- Confirm both importability and runtime behavior.
- Prefer a minimal smoke test first, then expand to the repo's integration test path.

## Pass criteria
- `import gfxgraph` succeeds.
- `gfxgraph.health_check()` reports `ok: True`.
- The runtime sees the intended ROCm GPU.
