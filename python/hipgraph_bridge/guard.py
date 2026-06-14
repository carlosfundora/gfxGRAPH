"""gfxGRAPH GUARD — 3-tier illegal-memory-access safety for HIP/CUDA graph capture.

Most "illegal memory access" failures on ROCm fall into two families. This module
handles both, since ROCm users are often unfamiliar with CUDA-graph rules:

  Tier 1 — auto-safe-capture (the "convert unsafe access pattern -> correct" tier).
    Force every tensor entering capture/replay to be CONTIGUOUS and own its storage
    (no broadcast/0-stride, negative-stride, or non-contiguous views), and validate
    dtype/device/layout. This eliminates the entire *capture-safety* family — the
    accesses that go bad only because a captured buffer was strided or its address
    wasn't stable. The arithmetic was fine; the buffer was the problem, so it IS
    auto-correctable.

  Tier 2 — fault localization. Turn a hard hipErrorIllegalAddress (which would be an
    opaque SIGSEGV) into a precise, catchable GfxGraphFault carrying the op + the
    layout of every tensor involved, then fall back to eager. For *in-kernel* OOB
    (e.g. a kernel indexing kv[stale_index]) the bad address is computed from data
    inside the kernel — not auto-correctable — but Tier 2 makes it diagnosable
    instead of crashing.

  Tier 3 — deep guard (opt-in, slow). Red-zone canary buffers (detect OOB writes
    past gfxGRAPH-owned buffers) + disable the caching allocator so faults land at
    real allocation boundaries + a compute-sanitizer/rocm-memcheck launcher to pin
    the exact offending op for the Tier-2 (logic-bug) class.

Activation: GFXGRAPH_GUARD = off | 1|tier1|safe | 2|tier2|localize | 3|tier3|deep
(default off — zero overhead). Higher tiers include the lower ones.
"""

import logging
import os

import torch

_log = logging.getLogger("gfxgraph")

_GUARD_ALIASES = {
    "": 0, "0": 0, "off": 0, "none": 0, "false": 0,
    "1": 1, "tier1": 1, "safe": 1, "capture": 1,
    "2": 2, "tier2": 2, "localize": 2,
    "3": 3, "tier3": 3, "deep": 3, "all": 3,
}


def guard_level() -> int:
    """Resolve the active GUARD tier (0..3) from GFXGRAPH_GUARD."""
    return _GUARD_ALIASES.get(os.environ.get("GFXGRAPH_GUARD", "").strip().lower(), 0)


# --------------------------------------------------------------------------
# Tier 1 — auto-safe-capture
# --------------------------------------------------------------------------

def validate_layout(t, name: str = "tensor") -> list:
    """Return a list of capture-safety issues for a tensor (empty == safe)."""
    issues = []
    if not isinstance(t, torch.Tensor):
        return issues
    try:
        strides = tuple(t.stride())
    except Exception:
        strides = ()
    if not t.is_contiguous():
        issues.append(f"{name}: non-contiguous (stride={strides})")
    if t.numel() > 1 and any(s == 0 for s in strides):
        issues.append(f"{name}: broadcast/0-stride view (stride={strides})")
    if any(s < 0 for s in strides):
        issues.append(f"{name}: negative stride (stride={strides})")
    return issues


def make_safe(t, name: str = "tensor"):
    """Return a graph-capture-safe version of `t` (contiguous, own storage).

    Returns (safe_tensor, converted: bool). Non-tensors pass through unchanged.
    This is the auto-correction for the capture-safety fault family.
    """
    if not isinstance(t, torch.Tensor):
        return t, False
    try:
        strides = tuple(t.stride())
    except Exception:
        strides = ()
    unsafe = (
        (not t.is_contiguous())
        or (t.numel() > 1 and any(s == 0 for s in strides))
        or any(s < 0 for s in strides)
    )
    if unsafe:
        safe = t.contiguous().clone() if any(s == 0 for s in strides) else t.contiguous()
        _log.debug("GUARD tier1: made %s capture-safe (was stride=%s)", name, strides)
        return safe, True
    return t, False


def make_capture_safe(obj):
    """Recursively make every tensor in obj (tensor/list/tuple/dict) capture-safe.

    Returns (new_obj, num_converted). Use at the boundary of capture/replay.
    """
    converted = [0]

    def rec(o):
        if isinstance(o, torch.Tensor):
            s, c = make_safe(o)
            if c:
                converted[0] += 1
            return s
        if isinstance(o, (list, tuple)):
            return type(o)(rec(x) for x in o)
        if isinstance(o, dict):
            return {k: rec(v) for k, v in o.items()}
        return o

    return rec(obj), converted[0]


# --------------------------------------------------------------------------
# Tier 2 — fault localization
# --------------------------------------------------------------------------

class GfxGraphFault(RuntimeError):
    """A localized illegal-memory-access fault with op + tensor context."""

    def __init__(self, message, *, op=None, tensors=None, original=None):
        self.op = op
        self.tensors = tensors or []
        self.original = original
        super().__init__(message)


_ILLEGAL_MARKERS = (
    "illegal memory access",
    "hiperrorillegaladdress",
    "misaligned address",
    "an illegal instruction",
)


def is_illegal_access(exc) -> bool:
    """True if exc looks like a GPU illegal/misaligned memory access."""
    s = str(exc).lower()
    return any(m in s for m in _ILLEGAL_MARKERS)


def _describe(tensors) -> str:
    lines = []
    for nm, t in tensors or []:
        if isinstance(t, torch.Tensor):
            try:
                lines.append(
                    f"{nm}: shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} "
                    f"contig={t.is_contiguous()} stride={tuple(t.stride())}"
                )
            except Exception:
                lines.append(f"{nm}: <tensor; introspection failed>")
    return "\n   ".join(lines)


def localize_fault(exc, *, op=None, tensors=None) -> "GfxGraphFault":
    """Build a precise GfxGraphFault from an illegal-access exception + context."""
    desc = _describe(tensors)
    msg = (
        "gfxGRAPH GUARD localized an illegal memory access"
        + (f" in op '{op}'" if op else "")
        + f".\n  Underlying: {exc}"
        + (f"\n  Tensors:\n   {desc}" if desc else "")
        + "\n  Class: in-kernel out-of-bounds (a producing-code logic bug, e.g. a "
        "stale/garbage index). Tier-2 detects+reports; it is not auto-fixable. "
        "Run GFXGRAPH_GUARD=3 (+ compute-sanitizer) to pin the exact op."
    )
    return GfxGraphFault(
        msg, op=op, tensors=[t for _, t in (tensors or [])], original=exc
    )


# --------------------------------------------------------------------------
# Tier 3 — deep guard (opt-in)
# --------------------------------------------------------------------------

def deep_guard_env() -> dict:
    """Env that makes OOB faults land at real allocation boundaries.

    Disabling the caching allocator removes red-zone-hiding reuse so an illegal
    access faults at the true buffer edge instead of silently hitting a neighbour.
    """
    return {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}


def apply_deep_guard_env() -> list:
    """Set the deep-guard env (without clobbering operator-set values)."""
    applied = []
    for k, v in deep_guard_env().items():
        if k not in os.environ:
            os.environ[k] = v
            applied.append(f"{k}={v}")
    if applied:
        _log.warning("GUARD tier3: applied deep-guard env: %s", ", ".join(applied))
    return applied


class RedZone:
    """Allocate a buffer padded with sentinel red-zones; detect OOB writes.

    Use `rz.tensor` as the working buffer; call `rz.check_intact()` after a
    replay/kernel to detect writes that ran past either end (gfxGRAPH-owned
    buffers only — cannot guard arbitrary user allocations).
    """

    SENTINEL = 127  # 0x7F — recognizable in fp/int dtypes

    def __init__(self, shape, dtype, device, pad_elems: int = 128):
        self.pad = int(pad_elems)
        n = 1
        for s in shape:
            n *= int(s)
        self._n = n
        self._flat = torch.empty(n + 2 * self.pad, dtype=dtype, device=device)
        self._fill_sentinel()
        self.tensor = self._flat[self.pad : self.pad + n].view(*shape)

    def _fill_sentinel(self):
        self._flat[: self.pad].fill_(self.SENTINEL)
        self._flat[self.pad + self._n :].fill_(self.SENTINEL)

    def check_intact(self) -> bool:
        lo_ok = bool((self._flat[: self.pad] == self.SENTINEL).all().item())
        hi_ok = bool((self._flat[self.pad + self._n :] == self.SENTINEL).all().item())
        if not (lo_ok and hi_ok):
            _log.error(
                "GUARD tier3: RED-ZONE VIOLATION — out-of-bounds write detected "
                "(lo_ok=%s hi_ok=%s)",
                lo_ok,
                hi_ok,
            )
        return lo_ok and hi_ok

    def reset(self):
        self._fill_sentinel()


def compute_sanitizer_cmd(argv) -> list:
    """Wrap argv in compute-sanitizer / rocm memcheck if available, else None."""
    import shutil

    candidates = (
        ("compute-sanitizer", ["--tool", "memcheck", "--launch-timeout", "0"]),
        ("rocm-memcheck", []),
        ("roc-memcheck", []),
    )
    for tool, opts in candidates:
        path = shutil.which(tool)
        if path:
            return [path, *opts, *list(argv)]
    return None


__all__ = [
    "guard_level",
    "validate_layout",
    "make_safe",
    "make_capture_safe",
    "GfxGraphFault",
    "is_illegal_access",
    "localize_fault",
    "deep_guard_env",
    "apply_deep_guard_env",
    "RedZone",
    "compute_sanitizer_cmd",
]
