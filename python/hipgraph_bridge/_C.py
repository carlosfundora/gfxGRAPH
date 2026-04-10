"""
ctypes bindings to libhipgraph_bridge.so
"""

import ctypes
import os
from pathlib import Path

_LIB_NAME = "libhipgraph_bridge.so"


def _find_lib():
    """Search for the bridge .so in common locations."""
    search_paths = [
        # Build directory (development)
        Path(__file__).parent.parent.parent / "build",
        # Installed location
        Path("/opt/rocm/lib/hipgraph_bridge"),
        Path("/usr/local/lib"),
        # LD_LIBRARY_PATH entries
    ]

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    for p in ld_path.split(":"):
        if p:
            search_paths.append(Path(p))

    for base in search_paths:
        candidate = base / _LIB_NAME
        if candidate.exists():
            return str(candidate)

    # Fallback: let ctypes search system paths
    return _LIB_NAME


try:
    lib = ctypes.CDLL(_find_lib())
except OSError:
    lib = None  # Bridge .so not built yet — Python-only mode


# Type definitions matching hipgraph_bridge.h
if lib is not None:
    # hgb_init
    lib.hgb_init.restype = ctypes.c_int
    lib.hgb_init.argtypes = []

    # hgb_shutdown
    lib.hgb_shutdown.restype = None
    lib.hgb_shutdown.argtypes = []

    # hgb_set_debug
    lib.hgb_set_debug.restype = None
    lib.hgb_set_debug.argtypes = [ctypes.c_int]
