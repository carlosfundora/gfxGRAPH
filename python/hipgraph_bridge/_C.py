"""
ctypes bindings to libhipgraph_bridge.so

Error handling: All HIP-facing C functions return hipError_t (int).
The Python layer checks every return code — non-zero triggers logging
and/or fallback. lib=None is safe (pure-Python mode).
"""

import ctypes
import logging
import os
from pathlib import Path

from gfxgraph._native import library_path as packaged_library_path

_log = logging.getLogger("gfxgraph")
_LIB_NAME = "libhipgraph_bridge.so"


def _find_lib():
    """Search for the bridge .so in common locations."""
    # Check explicit env override first
    explicit = os.environ.get("GFXGRAPH_LIB")
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        _log.warning("GFXGRAPH_LIB=%s not found, searching defaults", explicit)

    packaged = packaged_library_path()
    if packaged is not None:
        return str(packaged)

    repo_root = Path(__file__).resolve().parents[2]
    search_paths = [
        # Build directory (development)
        repo_root / "build",
        repo_root / "build" / "lib",
        repo_root / "build" / "Release",
    ]

    for base in search_paths:
        candidate = base / _LIB_NAME
        if candidate.exists():
            return str(candidate)

    # Fallback: let ctypes search system paths
    return _LIB_NAME


try:
    _lib_path = _find_lib()
    lib = ctypes.CDLL(_lib_path)
    _log.debug("Loaded native bridge from %s", _lib_path)
except OSError:
    lib = None  # Bridge .so not built yet — Python-only mode


# Type definitions matching hipgraph_bridge.h
if lib is not None:
    # hgb_init → int (hipError_t)
    lib.hgb_init.restype = ctypes.c_int
    lib.hgb_init.argtypes = []

    # hgb_shutdown → void
    lib.hgb_shutdown.restype = None
    lib.hgb_shutdown.argtypes = []

    # hgb_set_debug → void
    lib.hgb_set_debug.restype = None
    lib.hgb_set_debug.argtypes = [ctypes.c_int]


def call_native(func_name: str, *args) -> int:
    """Safely call a native bridge function. Returns 0 on success.

    If the native lib is not loaded, returns -1 (not available).
    Logs warnings on non-zero return codes.
    """
    if lib is None:
        return -1
    fn = getattr(lib, func_name, None)
    if fn is None:
        _log.warning("Native function %s not found in bridge .so", func_name)
        return -1
    rc = fn(*args)
    if rc != 0:
        _log.warning("Native %s returned error code %d", func_name, rc)
    return rc
