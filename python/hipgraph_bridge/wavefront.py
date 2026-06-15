# gfxGRAPH — wavefront: capture wave64/128 usage + plan the software-wave conversion
# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora
# SPDX-License-Identifier: MIT
"""gfxGRAPH wavefront — capture wave64/128 usage and plan the software-wave conversion.

gfx1030 (RDNA2) executes **Wave32 only**; ROCm silently drops `-mwavefrontsize64`. gfxGRAPH
cannot rewrite a compiled kernel's wavefront size at runtime — but it CAN:

  1. CAPTURE wave64/128 intent — scan build flags / source / launch shapes for 64-lane
     assumptions and surface a precise diagnosis (so the failure isn't silent).
  2. CONVERT (plan) — compute the **software-wave** launch: gang `W = target/32` Wave32 warps
     per work-item and merge once via LDS (W=2 ≈ wave64, W=4 ≈ wave128). This is the proven
     flash-decode-hip `W∈{1,2,4}` pattern; opt-in kernels call `plan_software_wave()` and use
     the returned (grid, block) + W. Adaptive: W>1 only helps when the grid is underfilled.

So "automatic" = automatic detection + a correct conversion *plan*, not a kernel rewrite.
"""

from __future__ import annotations

import os
import re

from .diagnostics import Diagnosis, explain

_WAVE64 = re.compile(r"mwavefrontsize64|wave_?64|wavefront.*64|__AMDGCN_WAVEFRONT_SIZE\s*==?\s*64|warpSize\s*==?\s*64", re.I)
_WAVE128 = re.compile(r"wave_?128|wavefront.*128", re.I)

_WAVE_ALIASES = {"": 0, "0": 0, "off": 0, "none": 0, "false": 0,
                 "1": 1, "detect": 1, "warn": 1,
                 "2": 2, "auto": 2, "convert": 2}


def wave_mode() -> int:
    """Active wave-conversion mode from GFXGRAPH_WAVE: 0=off, 1=detect/warn (default-safe),
    2=auto (apply the collision-safe plan). Mirrors `guard.guard_level()`."""
    return _WAVE_ALIASES.get(os.environ.get("GFXGRAPH_WAVE", "detect").strip().lower(), 1)


def _device_defaults(lanes, cu):
    """Fill lanes/cu from the detected GPU when not explicitly given (adaptive)."""
    if lanes is not None and cu is not None:
        return lanes, cu
    try:
        from .hardware import device_info
        d = device_info()
        if d is not None:
            lanes = lanes if lanes is not None else d.wavefront
            cu = cu if cu is not None else (d.cu or 20)
    except Exception:
        pass
    return (lanes if lanes is not None else 32), (cu if cu is not None else 20)


def detect_wave64(text: str) -> "Diagnosis | None":
    """Capture a wave64/128 request/assumption in compile flags, source, or a log line.

    Returns the `wave64_ignored` Diagnosis when found (call `.format()` to print), else None.
    """
    if not text:
        return None
    if _WAVE64.search(text) or _WAVE128.search(text):
        return explain("mwavefrontsize64")  # → the wave64_ignored entry
    return None


def should_convert(block_threads: int, grid_blocks: int, *, lanes: int | None = None,
                   cu: int | None = None) -> tuple[bool, str]:
    """Collision-safe gate: should gfxGRAPH apply software-wave conversion to this launch?

    Returns (apply, reason). gfxGRAPH NO-OPs (apply=False) when the user already handles wave
    width or conversion would not help — preventing double-application/collisions:
      • GFXGRAPH_WAVE=off                      → user disabled it.
      • GFXGRAPH_NO_WAVE=1                      → explicit opt-out.
      • block_threads > lanes                  → the launch already gangs >1 wavefront (user did it).
      • grid_blocks already saturates the GPU  → extra warps add no parallelism.
    Only when none of these hold does conversion (a wave64/128 plan) actually help.
    """
    lanes, cu = _device_defaults(lanes, cu)
    if wave_mode() == 0:
        return False, "GFXGRAPH_WAVE=off"
    if os.environ.get("GFXGRAPH_NO_WAVE", "").strip().lower() in ("1", "on", "true"):
        return False, "GFXGRAPH_NO_WAVE opt-out"
    if block_threads > lanes:
        return False, f"user already gangs wavefronts (block={block_threads} > Wave{lanes})"
    if grid_blocks >= cu * 32:
        return False, f"grid already saturates the GPU ({grid_blocks} blocks ≥ {cu}×32)"
    return True, "underfilled single-wavefront launch — software-wave conversion helps"


def plan_software_wave(total_rows: int, target_width: int = 64, *, lanes: int | None = None,
                       cu: int | None = None, min_rows_per_lane: int = 8,
                       max_kv_len: int | None = None) -> dict:
    """Plan the software-wave (gang-W-Wave32) launch that emulates a `target_width`-lane wavefront.

    Args:
        total_rows: independent work-items (e.g. batch*heads*queries).
        target_width: desired logical wavefront (64 → wave64, 128 → wave128); rounded to a
            multiple of `lanes`.
        lanes: hardware wavefront; **defaults to the detected GPU's** (32 RDNA / 64 CDNA).
        cu: compute units; **defaults to the detected GPU's** (`multiProcessorCount`, else 20).
        max_kv_len: optional reduction length; if given, W is capped so each warp keeps
            >= min_rows_per_lane elements (avoids over-splitting short reductions).

    Returns dict: {W, block, grid, lanes, emulated_width, underfilled, note}.
        block = lanes*W threads; grid = total_rows; each block gangs W Wave32 warps + 1 LDS merge.
    """
    lanes, cu = _device_defaults(lanes, cu)
    W_req = max(1, round(target_width / lanes))
    # Only gang extra warps when the plain Wave32 grid doesn't already fill the GPU
    # (W>1 adds threads/work-item but not independent parallelism — see flash-decode-hip).
    saturated = total_rows >= cu * 32  # ~32 resident waves/CU rule of thumb
    W = 1 if saturated else W_req
    if max_kv_len is not None and W > 1:
        while W > 1 and max_kv_len < min_rows_per_lane * W:
            W //= 2
    return {
        "W": W,
        "block": lanes * W,
        "grid": int(total_rows),
        "lanes": lanes,
        "emulated_width": lanes * W,
        "underfilled": not saturated,
        "note": (
            f"software-wave: gang W={W} Wave32 warp(s)/work-item (block={lanes*W} threads) "
            + ("+ LDS merge" if W > 1 else "(plain Wave32; grid already saturates the GPU)")
            + (f"; emulates wave{lanes*W}" if W > 1 else "")
        ),
    }
