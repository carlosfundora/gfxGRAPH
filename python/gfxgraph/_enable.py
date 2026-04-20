"""
gfxgraph.enable() / disable() — transparent monkey-patching of torch for gfx1030.

Calling enable() patches:
  - torch.cuda.CUDAGraph → BridgedCUDAGraph (gap-aware drop-in)
  - torch.compile default backend → hipgraph_bridge (optional)
  - Installs atexit handler for clean shutdown

Calling disable() restores all originals.

Observability:
  - gfxgraph.stats()        → performance counters
  - gfxgraph.health_check() → quick smoke test
  - HGB_LOG_LEVEL=debug|info|warn|error  → structured logging
"""

import atexit
import logging
import os
import threading
import time
from typing import Optional

_log = logging.getLogger("gfxgraph")

# ---------- Structured logging (obs-logging) ----------

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _configure_logging() -> None:
    """Apply HGB_LOG_LEVEL from environment if set."""
    level_str = os.environ.get("HGB_LOG_LEVEL", "").lower()
    if level_str in _LOG_LEVELS:
        _log.setLevel(_LOG_LEVELS[level_str])


_configure_logging()

# ---------- State ----------

_enabled = False
_validate_mode = False
_originals = {}  # stash for monkey-patched originals
_stats_lock = threading.Lock()
_stats = {
    "enabled_at": None,
    "capture_count": 0,
    "replay_count": 0,
    "fallback_count": 0,
    "validation_failures": 0,
    "avg_replay_us": 0.0,
    "_total_replay_us": 0.0,
}
_atexit_registered = False


# ---------- Counter helpers (obs-counters) ----------

def bump(counter: str, amount: int = 1) -> None:
    """Thread-safe counter increment. Used by bridge modules."""
    with _stats_lock:
        _stats[counter] = _stats.get(counter, 0) + amount


def record_replay_us(us: float) -> None:
    """Record a replay duration in microseconds, update running average."""
    with _stats_lock:
        _stats["replay_count"] += 1
        _stats["_total_replay_us"] += us
        _stats["avg_replay_us"] = (
            _stats["_total_replay_us"] / _stats["replay_count"]
        )


def get_validate_mode() -> bool:
    """Check whether validation mode is active."""
    return _validate_mode


# ---------- Public API ----------

def enable(*, debug: bool = False, validate: bool = False) -> None:
    """Activate gfxGRAPH — patches torch for transparent CUDA Graph parity.

    Args:
        debug: Enable verbose debug logging (also via HGB_DEBUG=1)
        validate: Enable validation mode — compares graph output vs eager
                  output to catch silent correctness bugs (PyTorch #155684).
                  Also activatable via GFXGRAPH=validate.
    """
    global _enabled, _validate_mode, _atexit_registered

    if _enabled:
        _log.info("gfxGRAPH already enabled")
        return

    import torch

    if debug:
        _log.setLevel(logging.DEBUG)
        os.environ["HGB_DEBUG"] = "1"

    _log.info("Enabling gfxGRAPH v0.3.1 for gfx1030/RDNA2")

    # Try to init native bridge (non-fatal if .so not built)
    _init_native(debug)

    # --- Monkey-patch torch.cuda.CUDAGraph ---
    from hipgraph_bridge.graph_manager import BridgedCUDAGraph

    _originals["CUDAGraph"] = torch.cuda.CUDAGraph
    torch.cuda.CUDAGraph = BridgedCUDAGraph
    _log.debug("Patched torch.cuda.CUDAGraph → BridgedCUDAGraph")

    # --- Register compile backend ---
    try:
        import hipgraph_bridge.compile_backend  # noqa: F401 — registers on import
        _log.debug("Registered hipgraph_bridge torch.compile backend")
    except Exception as e:
        _log.debug("torch.compile backend registration skipped: %s", e)

    # --- Validation mode ---
    _validate_mode = validate
    if validate:
        _log.info("Validation mode ON — graph outputs will be checked vs eager")

    # --- atexit cleanup ---
    if not _atexit_registered:
        atexit.register(_shutdown)
        _atexit_registered = True

    with _stats_lock:
        _stats["enabled_at"] = time.time()
    _enabled = True
    _log.info("gfxGRAPH enabled successfully")


def disable() -> None:
    """Restore original torch behavior. Safe to call even if not enabled."""
    global _enabled, _validate_mode

    if not _enabled:
        return

    import torch

    if "CUDAGraph" in _originals:
        torch.cuda.CUDAGraph = _originals.pop("CUDAGraph")
        _log.debug("Restored original torch.cuda.CUDAGraph")

    _enabled = False
    _validate_mode = False
    _log.info("gfxGRAPH disabled")


def is_enabled() -> bool:
    """Check if gfxGRAPH is currently active."""
    return _enabled


def stats() -> dict:
    """Return performance/diagnostic counters (thread-safe snapshot)."""
    with _stats_lock:
        out = {k: v for k, v in _stats.items() if not k.startswith("_")}
    return out


def health_check() -> dict:
    """Quick smoke test: capture + replay a trivial graph on gfx1030.

    Returns dict with 'ok', 'gpu', 'rocm', 'native_bridge',
    'vram_total_mb', 'vram_free_mb', 'details'.
    """
    result = {
        "ok": False,
        "gpu": "unknown",
        "rocm": "unknown",
        "native_bridge": False,
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "details": "",
    }

    try:
        import torch
        if not torch.cuda.is_available():
            result["details"] = "No CUDA/ROCm device available"
            return result

        result["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        result["rocm"] = getattr(props, "gcnArchName", "unknown")

        # VRAM info
        try:
            free, total = torch.cuda.mem_get_info(0)
            result["vram_total_mb"] = round(total / (1024 * 1024))
            result["vram_free_mb"] = round(free / (1024 * 1024))
        except Exception:
            pass

        # Test basic graph capture/replay
        x = torch.ones(4, device="cuda")
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            torch.cuda.synchronize()
            g.capture_begin()
            y = x * 2
            g.capture_end()
        g.replay()
        torch.cuda.synchronize()

        # Verify correctness
        expected = torch.ones(4, device="cuda") * 2
        if torch.allclose(y, expected):
            result["ok"] = True
            result["details"] = "Graph capture/replay OK, output verified"
        else:
            result["details"] = (
                "Graph replay produced wrong output — "
                "possible PyTorch #155684 (HIP Graph correctness bug)"
            )

    except Exception as e:
        result["details"] = f"Health check failed: {e}"

    # Check native bridge
    try:
        from hipgraph_bridge._C import lib
        if lib is not None:
            result["native_bridge"] = True
    except Exception:
        pass

    return result


def _init_native(debug: bool) -> None:
    """Try to load and init the native .so bridge. Non-fatal on failure."""
    try:
        from hipgraph_bridge._C import lib
        if lib is not None:
            rc = lib.hgb_init()
            if rc == 0:
                _log.info("Native bridge (libhipgraph_bridge.so) loaded")
                if debug:
                    lib.hgb_set_debug(1)
            else:
                _log.warning("Native bridge init returned error %d", rc)
        else:
            _log.info(
                "Native bridge not available (libhipgraph_bridge.so not found). "
                "Running in pure-Python mode. To build: cd gfxGRAPH && "
                "cmake --preset release && cmake --build build -j$(nproc)"
            )
    except Exception as e:
        _log.info("Native bridge load failed: %s. Pure-Python mode.", e)


def _shutdown() -> None:
    """atexit handler — clean up native resources."""
    try:
        from hipgraph_bridge._C import lib
        if lib is not None:
            lib.hgb_shutdown()
            _log.debug("Native bridge shutdown complete")
    except Exception:
        pass
