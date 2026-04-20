"""Canonical native-library lookup for gfxGRAPH packaging."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

_LIB_NAME = "libhipgraph_bridge.so"


def _candidate_from_spec(name: str) -> Path | None:
    try:
        spec = find_spec(name)
    except ModuleNotFoundError:
        return None

    if spec is None or not spec.submodule_search_locations:
        return None

    for location in spec.submodule_search_locations:
        candidate = Path(location) / _LIB_NAME
        if candidate.is_file():
            return candidate

    return None


def library_path() -> Path | None:
    """Return the packaged native bridge location when available.

    Resolution order:
    1. Bundled `gfxgraph._native/libhipgraph_bridge.so`
    2. Companion-owned `gfxgraph_native._native/libhipgraph_bridge.so`
    3. `None` when no packaged native bridge is installed
    """
    bundled = _candidate_from_spec(__name__)
    if bundled is not None:
        return bundled

    return _candidate_from_spec("gfxgraph_native._native")
