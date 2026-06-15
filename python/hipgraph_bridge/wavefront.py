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

import re

from .diagnostics import Diagnosis, explain

_WAVE64 = re.compile(r"mwavefrontsize64|wave_?64|wavefront.*64|__AMDGCN_WAVEFRONT_SIZE\s*==?\s*64|warpSize\s*==?\s*64", re.I)
_WAVE128 = re.compile(r"wave_?128|wavefront.*128", re.I)


def detect_wave64(text: str) -> "Diagnosis | None":
    """Capture a wave64/128 request/assumption in compile flags, source, or a log line.

    Returns the `wave64_ignored` Diagnosis when found (call `.format()` to print), else None.
    """
    if not text:
        return None
    if _WAVE64.search(text) or _WAVE128.search(text):
        return explain("mwavefrontsize64")  # → the wave64_ignored entry
    return None


def plan_software_wave(total_rows: int, target_width: int = 64, *, lanes: int = 32,
                       cu: int = 20, min_rows_per_lane: int = 8, max_kv_len: int | None = None) -> dict:
    """Plan the software-wave (gang-W-Wave32) launch that emulates a `target_width`-lane wavefront.

    Args:
        total_rows: independent work-items (e.g. batch*heads*queries).
        target_width: desired logical wavefront (64 → wave64, 128 → wave128); rounded to a
            multiple of `lanes`.
        lanes: hardware wavefront (32 on gfx1030).
        cu: compute units (20 on RX 6700 XT) — used to decide if the grid is underfilled.
        max_kv_len: optional reduction length; if given, W is capped so each warp keeps
            >= min_rows_per_lane elements (avoids over-splitting short reductions).

    Returns dict: {W, block, grid, lanes, emulated_width, underfilled, note}.
        block = lanes*W threads; grid = total_rows; each block gangs W Wave32 warps + 1 LDS merge.
    """
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
