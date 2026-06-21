"""Runtime guardrails for high-level PyTorch graph capture."""

import contextlib
import os
import subprocess
import sys
from functools import lru_cache

import torch

try:
    import rs_gfxgraph as _rs_gfxgraph
except Exception:
    _rs_gfxgraph = None

# Process-global hipGraph capture serialization lives in Rust
# (rs_gfxgraph_core::capture_gate). On gfx1030/RDNA2, HIP capture bookkeeping is
# process-global, so two in-process LLM sessions capturing at once corrupt each
# other. These thin wrappers serialize capture across sessions while leaving
# replay concurrent; when the native extension is absent (pure-Python install)
# they degrade to a no-op, matching the rest of the optional-native pattern.
_HAS_CAPTURE_GATE = _rs_gfxgraph is not None and hasattr(_rs_gfxgraph, "CaptureLock")


def capture_lock():
    """Return the process-global capture lock (exclusive) as a context manager."""
    if _HAS_CAPTURE_GATE:
        return _rs_gfxgraph.CaptureLock()
    return contextlib.nullcontext()


def replay_lock():
    """Return the shared replay lock as a context manager (excluded during capture)."""
    if _HAS_CAPTURE_GATE:
        return _rs_gfxgraph.ReplayLock()
    return contextlib.nullcontext()


def acquire_capture_lock():
    """Acquire the capture lock for the begin/end API (spans two calls)."""
    if _HAS_CAPTURE_GATE:
        _rs_gfxgraph.acquire_capture_lock()


def release_capture_lock():
    """Release the lock taken by :func:`acquire_capture_lock`."""
    if _HAS_CAPTURE_GATE:
        _rs_gfxgraph.release_capture_lock()


@lru_cache(maxsize=1)
def torch_cuda_execution_probe() -> tuple[bool, str]:
    """Check CUDA/HIP execution in a subprocess so crashes do not kill us."""
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false"

    code = """
import torch
x = torch.ones(1, device="cuda")
y = x + 1
torch.cuda.synchronize()
raise SystemExit(0 if float(y.cpu()[0]) == 2.0 else 2)
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(os.environ.get("GFXGRAPH_GPU_PROBE_TIMEOUT", "20")),
        )
    except subprocess.TimeoutExpired:
        return False, "CUDA execution probe timed out"
    except Exception as exc:
        return False, f"CUDA execution probe failed to run: {exc}"

    if completed.returncode == 0:
        return True, "CUDA execution probe passed"
    if completed.returncode < 0:
        return False, f"CUDA execution probe crashed with signal {-completed.returncode}"
    stderr = completed.stderr.strip().splitlines()
    detail = stderr[-1] if stderr else f"exit code {completed.returncode}"
    return False, f"CUDA execution probe failed: {detail}"


def torch_cuda_execution_usable() -> bool:
    return torch_cuda_execution_probe()[0]


def torch_cuda_execution_error() -> str:
    return torch_cuda_execution_probe()[1]


def unsafe_torch_graph_capture_enabled() -> bool:
    """Return True when high-level torch.cuda.graph capture is allowed.

    ROCm builds can segfault before Python can catch an exception on some
    consumer GPU/runtime combinations. Keep the high-level bridge fail-closed
    there unless explicitly enabled by an operator.
    """
    raw = os.environ.get("GFXGRAPH_ENABLE_UNSAFE_TORCH_GRAPH_CAPTURE", "")
    if raw.lower() in {"1", "true", "yes", "on"}:
        return torch_cuda_execution_usable()
    return not bool(getattr(torch.version, "hip", None))


def torch_graph_capture_block_reason() -> str:
    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        return (
            "high-level torch.cuda.graph capture is disabled by default on "
            f"ROCm/HIP {hip_version}; set "
            "GFXGRAPH_ENABLE_UNSAFE_TORCH_GRAPH_CAPTURE=1 to opt in"
        )
    return "high-level torch.cuda.graph capture is disabled"
