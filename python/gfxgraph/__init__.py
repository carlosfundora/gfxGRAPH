"""
gfxgraph — Drop-in CUDA Graph parity for AMD gfx1030 (RDNA2).

Activation:
    import gfxgraph
    gfxgraph.enable()          # patches torch transparently
    gfxgraph.enable(validate=True)  # + correctness validation

    # Or via env var (auto-enables on import):
    # GFXGRAPH=1 python my_script.py
    # GFXGRAPH=debug python my_script.py
    # GFXGRAPH=validate python my_script.py
"""

__version__ = "1.1.0"

import os as _os
import logging as _logging

_log = _logging.getLogger("gfxgraph")
_handler = _logging.StreamHandler()
_handler.setFormatter(_logging.Formatter("[gfxgRAPH] %(levelname)s: %(message)s"))
_log.addHandler(_handler)
_log.setLevel(_logging.WARNING)

# Re-export public API from internal modules
from hipgraph_bridge.graph_manager import BridgedCUDAGraph
from hipgraph_bridge.shape_bucketing import ShapeBucketPool
from hipgraph_bridge.conditional import ConditionalGraph

# Enable/disable machinery
from gfxgraph._enable import enable, disable, is_enabled, stats, health_check

# GUARD — 3-tier illegal-memory-access safety (GFXGRAPH_GUARD=1|2|3)
from hipgraph_bridge.guard import (
    GfxGraphFault,
    RedZone,
    apply_deep_guard_env,
    compute_sanitizer_cmd,
    guard_level,
    is_illegal_access,
    localize_fault,
    make_capture_safe,
    make_safe,
    validate_layout,
)
from hipgraph_bridge.diagnostics import (
    Diagnosis,
    diagnose,
    environment_report,
    explain,
    install_diagnostics,
    lang as diag_lang,
    report,
)
from hipgraph_bridge.wavefront import detect_wave64, plan_software_wave, should_convert, wave_mode
from hipgraph_bridge.hardware import (
    DeviceInfo,
    RocmTorchError,
    detect_accelerators,
    detect_engines,
    device_info,
    require_rocm_torch,
    torch_rocm_status,
)
from hipgraph_bridge.interop import (
    MIGraphXBackend,
    cross_engine_support,
    hipgraph_interposer_status,
    migraphx_available,
)

__all__ = [
    "enable",
    "disable",
    "is_enabled",
    "stats",
    "health_check",
    "BridgedCUDAGraph",
    "ShapeBucketPool",
    "ConditionalGraph",
    # GUARD
    "guard_level",
    "validate_layout",
    "make_safe",
    "make_capture_safe",
    "GfxGraphFault",
    "is_illegal_access",
    "localize_fault",
    "RedZone",
    "apply_deep_guard_env",
    "compute_sanitizer_cmd",
    # Diagnostics (always-on HIP/ROCm error & status reporting; bilingual en/zh)
    "explain",
    "report",
    "diagnose",
    "install_diagnostics",
    "environment_report",
    "Diagnosis",
    "diag_lang",
    # Wavefront (capture wave64/128 + plan + collision-safe gate)
    "detect_wave64",
    "plan_software_wave",
    "should_convert",
    "wave_mode",
    # Hardware (boot-time device + ROCm-PyTorch detection; user-card override)
    "device_info",
    "DeviceInfo",
    "torch_rocm_status",
    "require_rocm_torch",
    "RocmTorchError",
    "detect_accelerators",
    "detect_engines",
    # Cross-engine interop (MIGraphX + hipGraph interposer — detection now, routing roadmap)
    "migraphx_available",
    "MIGraphXBackend",
    "hipgraph_interposer_status",
    "cross_engine_support",
]

# Auto-enable via environment variable
_env = _os.environ.get("GFXGRAPH", "").lower()
if _env == "1":
    enable()
elif _env == "debug":
    _log.setLevel(_logging.DEBUG)
    enable(debug=True)
elif _env == "validate":
    enable(validate=True)

# Always-on HIP/ROCm error reporting: auto-install the diagnostics excepthook when gfxGRAPH is
# enabled, or when GFXGRAPH_DIAG is explicitly requested. Honors GFXGRAPH_LANG (en|zh).
if _env in ("1", "debug", "validate") or _os.environ.get("GFXGRAPH_DIAG", "").lower() in ("1", "on", "true"):
    try:
        install_diagnostics()
    except Exception:  # pragma: no cover - never let diagnostics break import
        pass
