# gfxGRAPH — diagnostics: always-on CUDA/HIP error & status reporting (bilingual en/zh)
# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora
# SPDX-License-Identifier: MIT
"""gfxGRAPH diagnostics — robust, always-on CUDA/HIP error & status reporting.

ROCm/HIP error messages are famously terse ("No available kernel. Aborting execution.",
"invalid device function", a bare SIGSEGV). gfxGRAPH's GUARD already turns *graph-capture*
illegal-access into precise faults — but most real crashes happen OUTSIDE graph capture
(allocator OOM, missing kernel images, wrong-arch launches, bf16 on RDNA2). This module is
the general layer: it translates a HIP/ROCm error into a plain-English **cause + RDNA2/gfx1030
context + concrete fix**, works whether or not CUDA-graphs are active, and can install an
always-on excepthook so a confused ROCm user gets the explanation instead of a cryptic abort.

Quick use:
    import gfxgraph
    gfxgraph.install_diagnostics()            # always-on: cryptic HIP errors -> explained
    print(gfxgraph.environment_report())      # HSA override? arch? free VRAM? graph state?
    with gfxgraph.diagnose("decode forward"): # wrap a risky block
        model.generate(...)

Pure-Python; `torch`/`rocm-smi` are imported lazily so the table works without a GPU.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

_log = logging.getLogger("gfxgraph")

# gfx1030 / RX 6700 XT facts referenced by several entries (single source of truth).
_GFX1030 = "gfx1030 (RDNA2, Wave32, no matrix/MFMA cores; this box runs gfx1031 as gfx1030)"
_HSA = "HSA_OVERRIDE_GFX_VERSION=10.3.0"


def lang() -> str:
    """Active diagnostics language: 'zh' if GFXGRAPH_LANG starts with zh/cn, else 'en'."""
    v = os.environ.get("GFXGRAPH_LANG", "en").strip().lower()
    return "zh" if v.startswith(("zh", "cn", "chinese", "中")) else "en"


# English frame labels. Chinese labels + messages live in `diag_zh.py`, loaded lazily ONLY when
# GFXGRAPH_LANG=zh — keeps this English module clean and zero-import-cost for English users.
_LABELS_EN = {"diag": "gfxGRAPH diagnosis", "cause": "Cause", "ctx": "ROCm/gfx1030 context",
              "fix": "Fix", "silence": "set GFXGRAPH_DIAG=0 to silence"}

_ZH_PACK = None  # cached (LABELS, MESSAGES) from diag_zh on first Chinese use


def _zh_pack():
    global _ZH_PACK
    if _ZH_PACK is None:
        try:
            from . import diag_zh
            _ZH_PACK = (diag_zh.LABELS, diag_zh.MESSAGES)
        except Exception:  # pragma: no cover - fall back to English if the pack is missing
            _ZH_PACK = (_LABELS_EN, {})
    return _ZH_PACK


def _labels(lang_: str) -> dict:
    return _zh_pack()[0] if lang_ == "zh" else _LABELS_EN


@dataclass
class Diagnosis:
    """A translated, ROCm-newbie-friendly explanation of a HIP/CUDA condition.

    English fields are stored directly; Chinese is looked up by `code` from `_ZH` so the same
    Diagnosis renders in either language via `format(lang=...)` / `GFXGRAPH_LANG=zh`.
    """

    code: str          # canonical key, e.g. "illegal_address"
    severity: str      # "error" | "warning" | "info"
    summary: str       # one-line headline (English)
    cause: str         # why it happens (English)
    rocm_context: str  # gfx1030/RDNA2-specific context (English)
    fix: str           # concrete, actionable next step (English)

    def localized(self, lang_: str) -> dict:
        if lang_ == "zh":
            msgs = _zh_pack()[1]
            if self.code in msgs:
                return msgs[self.code]
        return {"summary": self.summary, "cause": self.cause,
                "rocm_context": self.rocm_context, "fix": self.fix}

    def format(self, lang_: "str | None" = None) -> str:
        lg = lang_ or lang()
        L = _labels(lg)
        t = self.localized(lg)
        return (
            f"\n┌─ {L['diag']} [{self.severity.upper()}: {self.code}]\n"
            f"│ {t['summary']}\n"
            f"│  • {L['cause']}: {t['cause']}\n"
            f"│  • {L['ctx']}: {t['rocm_context']}\n"
            f"│  • {L['fix']}: {t['fix']}\n"
            f"└─ (gfxGRAPH; {L['silence']})"
        )

    def __str__(self) -> str:
        return self.format()


# Ordered table: (canonical code, severity, [regex patterns matched against the error text],
# summary, cause, rocm_context, fix). First match wins; order most-specific first.
_TABLE: list[tuple] = [
    (
        "no_kernel_image", "error",
        [r"no available kernel", r"invalid device function", r"no kernel image",
         r"hipErrorNoBinaryForGpu", r"shared object init"],
        "A kernel has no binary built for this GPU architecture.",
        "The op (often torch FLASH/AOTriton SDPA, FA3/FA4, or a CUTLASS/CK kernel) was AOT-compiled "
        "for archs that don't include gfx1030, so the runtime finds no usable image.",
        f"On {_GFX1030}, AOTriton/AITER flash-attention and tensor-core kernels frequently ship no "
        "gfx1030 image (and FA3/FA4 need Hopper/Blackwell). It is NOT a bug in your code.",
        "Use a JIT-Triton path instead (sglang `--attention-backend triton`, or the in-house "
        "flash-decode-hip / flash-attn-prefill-hip), or rebuild the kernel with "
        f"`--offload-arch=gfx1030` and run with {_HSA}.",
    ),
    (
        "out_of_memory", "error",
        [r"out of memory", r"hipErrorOutOfMemory", r"HIPCachingAllocator.*OOM", r"CUDA out of memory"],
        "GPU VRAM exhausted.",
        "The requested allocation exceeds free VRAM — often a KV/activation pool sized by "
        "`--mem-fraction-static`, or a quant codec allocating extra buffers on top of that pool.",
        f"{_GFX1030} has ~12 GB. RotorQuant/TurboQuant (rq3/tq3) KV codecs allocate compressed "
        "buffers + rotation matrices BEYOND the base pool, so a high mem-fraction (e.g. 0.85) OOMs "
        "even though plain f16 fits; other resident services may also hold VRAM.",
        "Lower `--mem-fraction-static` (~0.45 for rq*/tq* codecs — sglang auto-lowers this on RDNA2), "
        "free competing GPU services, or reduce context/batch. Check free VRAM with "
        "`rocm-smi --showmeminfo vram` / `gfxgraph.environment_report()`.",
    ),
    (
        "illegal_address", "error",
        [r"illegal memory access", r"hipErrorIllegalAddress", r"misaligned address"],
        "Illegal/out-of-bounds GPU memory access (would be an opaque SIGSEGV).",
        "Either a buffer captured into a CUDA/HIP graph was read at a stale/freed address on replay "
        "(capture-safety), or a kernel indexed out of bounds (logic bug).",
        f"On {_GFX1030} this is the #1 CUDA-graph failure; that's why cuda-graph is DEFAULT-OFF here. "
        "Non-contiguous/strided/0-stride tensors captured into a graph are the usual capture-safety "
        "cause.",
        "Enable GUARD (`GFXGRAPH_GUARD=1` auto-makes capture inputs safe; `=2` localizes the faulting "
        "op + tensor layouts; `=3` deep red-zone/sanitizer). For the logic-bug class, run under "
        "compute-sanitizer to pin the op.",
    ),
    (
        "bf16_unsupported", "error",
        [r"fdot2\.bf16", r"bf16.*not supported", r"bfloat16.*unsupported", r"v_dot2.*bf16"],
        "bfloat16 GEMM/dot is unsupported on this GPU.",
        "The kernel emitted a bf16 dot-product instruction the hardware lacks.",
        f"{_GFX1030} has no `fdot2.bf16.bf16` — bf16 matmul crashes. (CDNA/RDNA3 have it; RDNA2 "
        "doesn't.)",
        "Run in fp16 instead of bf16 (sglang/llama.cpp auto-override bf16→fp16 on gfx1030; ensure "
        "that override is active, or pass `--dtype float16`).",
    ),
    (
        "wrong_arch", "warning",
        [r"HSA_OVERRIDE", r"gfx900.*gfx1030", r"unsupported.*gfx", r"Cannot find.*gfx"],
        "GPU architecture mismatch / override not set.",
        "The toolchain or runtime resolved an arch that doesn't match the physical GPU.",
        "This box is physically gfx1031 but is run as gfx1030 (best-supported RDNA2 target); without "
        "the override, ROCm may misdetect it.",
        f"Export {_HSA} (and build with `--offload-arch=gfx1030`). Verify with `rocminfo | grep gfx`.",
    ),
    (
        "wave64_ignored", "info",
        [r"mwavefrontsize64", r"wave64", r"wavefront.*64"],
        "Wave64 was requested but gfx1030 only executes Wave32.",
        "ROCm's gfx1030 backend silently ignores `-mwavefrontsize64`; you always get Wave32.",
        f"{_GFX1030} is Wave32-only. 'Going wider' than 32 lanes must be done in software.",
        "Use software split-K: gang W Wave32 warps per work-item with an LDS merge (W=2 ≈ wave64, "
        "W=4 ≈ wave128) — as in flash-decode-hip's adaptive `W∈{1,2,4}`. Adopt W>1 only when the "
        "grid is underfilled.",
    ),
    (
        "aiter_on_rdna", "warning",
        [r"aiter.*not.*support", r"aiter.*rdna", r"flydsl\.moe_common", r"CK.*not available"],
        "AITER (AMD CK/ASM kernels) is not available/optimal on this GPU.",
        "AITER's ASM/CK kernels target CDNA (MI2xx/MI3xx); on RDNA they're missing or fall back.",
        f"{_GFX1030} is RDNA2 — AITER attention routes to (slower) Triton, and AITER MoE/flydsl bits "
        "may be absent. Not an error, just unsupported hardware.",
        "Prefer the Triton path on RDNA (sglang auto-routes `aiter`→triton on gfx10xx). Don't expect "
        "CK/ASM acceleration here.",
    ),
    (
        "invalid_configuration", "error",
        [r"hipErrorInvalidConfiguration", r"invalid configuration argument", r"too many resources"],
        "Invalid kernel launch configuration.",
        "Block/grid dims or shared-memory/register request exceed device limits.",
        f"{_GFX1030}: max 1024 threads/block, 64 KB LDS/CU; high per-lane VGPR use (e.g. a per-lane "
        "O[D] accumulator) collapses occupancy.",
        "Reduce block size / LDS / per-thread registers. For attention, prefer a Wave32 warp that "
        "distributes the head-dim reduction over 32 lanes rather than huge per-lane accumulators.",
    ),
]

_COMPILED = [
    (code, sev, [re.compile(p, re.I) for p in pats], summ, cause, ctx, fix)
    for (code, sev, pats, summ, cause, ctx, fix) in _TABLE
]



def explain(error) -> "Diagnosis | None":
    """Translate a HIP/CUDA error (Exception, str, or hipError int) into a Diagnosis.

    Returns None if nothing in the table matches (caller can fall back to the raw error).
    """
    text = _error_text(error)
    if not text:
        return None
    for code, sev, pats, summ, cause, ctx, fix in _COMPILED:
        if any(p.search(text) for p in pats):
            return Diagnosis(code, sev, summ, cause, ctx, fix)
    return None


def _error_text(error) -> str:
    if error is None:
        return ""
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return str(error)


def diag_enabled() -> bool:
    """Diagnostics emit unless GFXGRAPH_DIAG is explicitly off (default on)."""
    return os.environ.get("GFXGRAPH_DIAG", "1").strip().lower() not in ("0", "off", "false", "no")


def report(error, *, context: str | None = None, logger: logging.Logger | None = None) -> "Diagnosis | None":
    """Explain `error` and emit it (logged at the diagnosis severity). Returns the Diagnosis."""
    d = explain(error)
    if d is None or not diag_enabled():
        return d
    lg = logger or _log
    head = f"[{context}] " if context else ""
    msg = head + d.format()
    {"error": lg.error, "warning": lg.warning}.get(d.severity, lg.info)(msg)
    return d


class diagnose:
    """Context manager: on a HIP/CUDA exception inside the block, emit a diagnosis then re-raise.

        with gfxgraph.diagnose("rdna2 decode"):
            out = model.forward(...)
    """

    def __init__(self, context: str | None = None):
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            report(exc, context=self.context)
        return False  # never suppress


_PREV_EXCEPTHOOK = None


def install_diagnostics() -> bool:
    """Install an always-on `sys.excepthook` that appends a gfxGRAPH diagnosis to any uncaught
    HIP/CUDA error. Idempotent; returns True if installed. This is the 'alert us even when
    cuda-graphs are off' hook."""
    import sys

    global _PREV_EXCEPTHOOK
    if _PREV_EXCEPTHOOK is not None:
        return False
    _PREV_EXCEPTHOOK = sys.excepthook

    def _hook(exc_type, exc, tb):
        try:
            d = explain(exc)
            if d is not None and diag_enabled():
                print(d.format(), file=sys.stderr)
        except Exception:
            pass
        _PREV_EXCEPTHOOK(exc_type, exc, tb)

    sys.excepthook = _hook
    _log.info("gfxGRAPH diagnostics installed (excepthook); set GFXGRAPH_DIAG=0 to silence")
    return True


def environment_report() -> str:
    """Human-readable snapshot for confused ROCm users: arch override, torch/HIP, free VRAM,
    and gfxGRAPH/GUARD state. Degrades gracefully when torch/rocm-smi are absent."""
    lines = ["gfxGRAPH environment report:"]
    lines.append(f"  HSA_OVERRIDE_GFX_VERSION = {os.environ.get('HSA_OVERRIDE_GFX_VERSION', '(unset!)')}")
    lines.append(f"  PYTORCH_ROCM_ARCH        = {os.environ.get('PYTORCH_ROCM_ARCH', '(unset)')}")
    lines.append(f"  GFXGRAPH / GUARD / DIAG  = {os.environ.get('GFXGRAPH','0')} / "
                 f"{os.environ.get('GFXGRAPH_GUARD','0')} / {os.environ.get('GFXGRAPH_DIAG','1')}")
    try:
        import torch
        hip = getattr(torch.version, "hip", None)
        lines.append(f"  torch                    = {torch.__version__} (hip={hip})")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            lines.append(f"  device                   = {torch.cuda.get_device_name(0)}")
            lines.append(f"  VRAM free/total          = {free/2**30:.2f} / {total/2**30:.2f} GiB")
        else:
            lines.append("  device                   = (torch.cuda not available — CPU/no-ROCm)")
    except Exception as e:  # pragma: no cover
        lines.append(f"  torch                    = (not importable: {e})")
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION") != "10.3.0":
        lines.append("  ⚠ gfx1030 boxes usually need HSA_OVERRIDE_GFX_VERSION=10.3.0")
    return "\n".join(lines)
