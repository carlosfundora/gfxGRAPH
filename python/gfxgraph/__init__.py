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

__version__ = "0.4.0"

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
