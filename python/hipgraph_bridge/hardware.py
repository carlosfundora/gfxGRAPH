# gfxGRAPH — hardware: boot-time device + ROCm-PyTorch detection (the adaptive foundation)
# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora
# SPDX-License-Identifier: MIT
"""gfxGRAPH hardware detection — read the GPU + ROCm-PyTorch environment on boot and adapt.

Design goals (user-driven):
  • Dynamic install: one `pip install gfxgraph`; on activation gfxGRAPH auto-detects the GPU arch,
    ROCm-PyTorch, and native bridge, and applies ONLY what's relevant — no manual tier installs.
  • Report the ROCm-PyTorch it finds (torch X · HIP Y).
  • Error clearly if ROCm-PyTorch is absent — including the common trap where torch IS installed but
    is a CPU/CUDA wheel (`torch.version.hip is None`).

Torch is imported lazily so this module (and the diagnostics that reuse `arch_descriptor()`) stays
usable on a CI/dev box without torch. The ROCm-torch *requirement* is enforced at `enable()` (bridge
activation), never at import.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

_log = logging.getLogger("gfxgraph")


class RocmTorchError(RuntimeError):
    """Raised at bridge activation when a usable ROCm build of PyTorch is not present."""


# CDNA archs are wave64; RDNA (gfx10xx/11xx/12xx) is wave32.
_CDNA_PREFIXES = ("gfx90", "gfx94", "gfx95", "gfx908", "gfx906", "gfx900")


def _wavefront_for(arch: str) -> int:
    a = (arch or "").lower()
    return 64 if any(a.startswith(p) for p in _CDNA_PREFIXES) else 32


@dataclass
class DeviceInfo:
    arch: str            # gcnArchName, e.g. "gfx1030" (or the GFXGRAPH_ARCH override)
    name: str            # marketing name, e.g. "AMD Radeon RX 6700 XT"
    cu: int              # compute units (multiProcessorCount); 0 if unknown
    wavefront: int       # 32 (RDNA) / 64 (CDNA)
    vram_total_gib: float
    vram_free_gib: float
    overridden: bool     # arch came from GFXGRAPH_ARCH, not the live device

    @property
    def is_rdna(self) -> bool:
        return self.wavefront == 32

    @property
    def is_gfx1030(self) -> bool:
        return self.arch.lower().startswith("gfx1030")

    def summary(self) -> str:
        src = " (GFXGRAPH_ARCH override)" if self.overridden else ""
        return (f"{self.arch}{src} · {self.name} · {self.cu} CU · Wave{self.wavefront} · "
                f"VRAM {self.vram_free_gib:.1f}/{self.vram_total_gib:.1f} GiB")


def torch_rocm_status() -> dict:
    """Probe PyTorch without raising. Returns:
    {installed, is_rocm, torch_version, hip_version, cuda_available, message}."""
    try:
        import torch
    except Exception as e:  # not installed / broken
        return {"installed": False, "is_rocm": False, "torch_version": None,
                "hip_version": None, "cuda_available": False,
                "message": f"PyTorch is not importable ({e})."}
    hip = getattr(torch.version, "hip", None)
    ver = getattr(torch, "__version__", "?")
    avail = False
    try:
        avail = bool(torch.cuda.is_available())
    except Exception:
        pass
    if hip:
        msg = f"ROCm PyTorch {ver} (HIP {hip}) detected; torch.cuda available={avail}."
    else:
        msg = (f"PyTorch {ver} is installed but is NOT a ROCm build "
               f"(torch.version.hip is None — this looks like a CPU/CUDA wheel).")
    return {"installed": True, "is_rocm": bool(hip), "torch_version": ver,
            "hip_version": hip, "cuda_available": avail, "message": msg}


def require_rocm_torch(strict: bool = True) -> dict:
    """Report the ROCm PyTorch found, or raise/log a precise error if it's absent/wrong-backend.

    Returns the `torch_rocm_status()` dict. With `strict=True` (the default at bridge activation),
    raises `RocmTorchError` when torch is missing or is a non-ROCm wheel. Diagnostics/wavefront do
    NOT call this — only `enable()` does — so the package stays import-safe without torch.
    """
    st = torch_rocm_status()
    if st["is_rocm"]:
        _log.info("gfxGRAPH: %s", st["message"])
        return st
    # Build an actionable error.
    if not st["installed"]:
        hint = ("Install a ROCm build of PyTorch (https://pytorch.org → ROCm wheel), e.g. "
                "`pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`.")
    else:
        hint = ("You have a CPU/CUDA PyTorch; reinstall the ROCm wheel for your ROCm version, then "
                "verify `python -c \"import torch; print(torch.version.hip)\"` prints a HIP version.")
    msg = (f"gfxGRAPH bridge needs ROCm PyTorch. {st['message']} {hint} "
           "(Diagnostics still work without it; only the CUDA→HIP bridge is disabled.)")
    if strict:
        raise RocmTorchError(msg)
    _log.error(msg)
    return st


_DEVICE: "DeviceInfo | None" = None
_DEVICE_PROBED = False


def detect_device(refresh: bool = False) -> "DeviceInfo | None":
    """Detect the active GPU (cached). Honors `GFXGRAPH_ARCH=<gfxNNNN>` to override the arch (for
    users targeting a specific card / cross-config). Returns None if no device is visible."""
    global _DEVICE, _DEVICE_PROBED
    if _DEVICE_PROBED and not refresh:
        return _DEVICE
    _DEVICE_PROBED = True
    override = os.environ.get("GFXGRAPH_ARCH", "").strip().lower()
    arch, name, cu, vtot, vfree = "", "", 0, 0.0, 0.0
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            arch = getattr(props, "gcnArchName", "") or ""
            name = getattr(props, "name", "") or torch.cuda.get_device_name(0)
            cu = int(getattr(props, "multi_processor_count", 0) or 0)
            try:
                free, total = torch.cuda.mem_get_info(0)
                vfree, vtot = free / 2**30, total / 2**30
            except Exception:
                vtot = getattr(props, "total_memory", 0) / 2**30
    except Exception as e:  # pragma: no cover
        _log.debug("gfxGRAPH device probe failed: %s", e)
    if override:
        arch = override
    if not arch:
        if not override:
            _DEVICE = None
            return None
    _DEVICE = DeviceInfo(arch=arch, name=name or "(unknown GPU)", cu=cu,
                         wavefront=_wavefront_for(arch), vram_total_gib=vtot,
                         vram_free_gib=vfree, overridden=bool(override))
    return _DEVICE


def device_info() -> "DeviceInfo | None":
    """Public accessor for the detected/overridden device (cached). See `detect_device`."""
    return detect_device()


# gfx1030 stays the rich default descriptor (this is the reference box); other arches get an
# adaptive line. Diagnostics import this so their context de-hardcodes from a fixed gfx1030 string.
_GFX1030_DEFAULT = "gfx1030 (RDNA2, Wave32, no matrix/MFMA cores; this box runs gfx1031 as gfx1030)"


def arch_descriptor() -> str:
    """One-line architecture descriptor for diagnostics, adapted to the detected/overridden GPU.
    Falls back to the gfx1030 reference string when no device is known (torch-free contexts)."""
    dev = device_info()
    if dev is None:
        return _GFX1030_DEFAULT
    if dev.is_gfx1030:
        return _GFX1030_DEFAULT if not dev.overridden else \
            "gfx1030 (RDNA2, Wave32, no matrix/MFMA cores)"
    kind = "RDNA (Wave32, no matrix cores)" if dev.is_rdna else "CDNA (Wave64, has matrix cores)"
    return f"{dev.arch} — {kind}, {dev.cu} CU ({dev.name})"


def _has(mod: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def detect_accelerators() -> dict:
    """Probe ROCm graph/accel libraries present (capability discovery → 'apply only what's there').

    Reports, for adaptive behavior + `environment_report`:
      torch_cuda_graph/torch_compile (what the bridge hooks), migraphx (AMD graph compiler —
      a potential alt backend), aiter/triton/aotriton (kernel backends), onnxruntime (MIGraphX EP
      host), gfxgraph_native (our optional native .so).
    """
    import shutil
    acc = {
        "torch_cuda_graph": False, "torch_compile": False,
        "migraphx": _has("migraphx") or bool(shutil.which("migraphx-driver")),
        "aiter": _has("aiter"), "triton": _has("triton"), "aotriton": _has("aotriton"),
        "onnxruntime": _has("onnxruntime"), "gfxgraph_native": False,
    }
    try:
        import torch
        acc["torch_cuda_graph"] = hasattr(torch.cuda, "CUDAGraph")
        acc["torch_compile"] = hasattr(torch, "compile")
    except Exception:
        pass
    try:
        from . import _C
        acc["gfxgraph_native"] = getattr(_C, "lib", None) is not None
    except Exception:
        pass
    return acc


def detect_engines() -> dict:
    """Probe known inference engines present. PyTorch engines (vllm/sglang) get the CUDA-graph
    BRIDGE for free; native engines (llama.cpp/candle) get the diagnostics via the CLI/log-pipe
    (the bridge needs per-runtime native hooks — see docs)."""
    import shutil
    return {
        "vllm": _has("vllm"),                # torch → bridge + diagnostics
        "sglang": _has("sglang"),            # torch → bridge + diagnostics
        "llama_cpp_python": _has("llama_cpp"),
        "llama_cpp_bin": bool(shutil.which("llama-cli") or shutil.which("llama-server")
                              or shutil.which("llama-completion")),  # native → diagnostics via CLI
        "candle_bin": bool(shutil.which("candle")),  # rust → diagnostics via CLI (best-effort)
    }
