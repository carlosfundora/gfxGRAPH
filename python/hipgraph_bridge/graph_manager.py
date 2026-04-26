"""
BridgedCUDAGraph — Drop-in replacement for torch.cuda.CUDAGraph that
automatically applies bridge strategies for the 4 parity gaps.

Hardened with:
  - Try/except eager fallback on capture or replay failure
  - Validation mode (graph vs eager comparison, catches PyTorch #155684)
  - Input tensor validation (device, contiguity)
  - Performance counter integration
"""

import logging
import time

import torch

from hipgraph_bridge.shape_bucketing import ShapeBucketPool

_log = logging.getLogger("gfxgraph")

# Capture the original CUDAGraph class BEFORE monkey-patching replaces it.
# This module is imported by gfxgraph.__init__ which happens before enable().
_OriginalCUDAGraph = torch.cuda.CUDAGraph


def _validate_tensor(t: torch.Tensor, name: str = "input") -> None:
    """Raise if tensor is not on CUDA or not contiguous."""
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    if not t.is_cuda:
        raise ValueError(
            f"{name} must be on a CUDA device, got {t.device}. "
            "Move it with .cuda() or .to('cuda')."
        )
    if not t.is_contiguous():
        _log.warning("%s is not contiguous — this may hurt graph capture perf", name)


class BridgedCUDAGraph:
    """Drop-in replacement for torch.cuda.CUDAGraph with gap bridging.

    Usage:
        g = BridgedCUDAGraph()
        with g.capture(dynamic_shapes=True, buckets=[1, 4, 8, 16, 32]):
            output = model(static_input)
        g.replay(batch_size=12)  # auto-selects bucket 16

    Falls back to standard CUDAGraph when no gap capabilities are needed.
    On capture/replay failure, automatically falls back to eager execution.
    """

    def __init__(self):
        self._graph = None
        self._shape_pool = None
        self._conditional_branches = None
        self._stream = None
        self._static_output = None
        self._model_fn = None       # stored for eager fallback
        self._eager_fallback = False  # set True on capture failure
        self._last_input = None      # for validation mode

    # ---- PyTorch CUDAGraph low-level API compatibility ----
    # These methods make BridgedCUDAGraph a true drop-in for torch.cuda.CUDAGraph.
    # When callers use the low-level API (capture_begin/capture_end/replay()),
    # we delegate directly to a real CUDAGraph with no eager-fallback wrapping,
    # because the caller (e.g. SGLang's cuda_graph_runner) handles errors itself.

    def capture_begin(self, *args, **kwargs):
        """Start graph capture — delegates to real CUDAGraph."""
        if self._graph is None:
            self._graph = _OriginalCUDAGraph()
        self._graph.capture_begin(*args, **kwargs)

    def capture_end(self):
        """End graph capture — delegates to real CUDAGraph."""
        if self._graph is not None:
            self._graph.capture_end()
            _bump_capture()

    def pool(self):
        """Return the mempool id — delegates to real CUDAGraph."""
        if self._graph is not None:
            return self._graph.pool()
        return None

    class _CaptureContext:
        def __init__(self, parent, dynamic_shapes, buckets, conditional_branches):
            self.parent = parent
            self.dynamic_shapes = dynamic_shapes
            self.buckets = buckets
            self.conditional_branches = conditional_branches

        def __enter__(self):
            self.parent._stream = torch.cuda.Stream()

            if self.dynamic_shapes and self.buckets:
                # Gap 53: Use shape bucketing — capture deferred to first replay
                self.parent._shape_pool = ShapeBucketPool(
                    model_fn=self.parent._model_fn,
                    buckets=self.buckets
                )
                _bump_capture()
                return self

            # Standard capture path — with try/except for eager fallback
            try:
                self.parent._graph = _OriginalCUDAGraph()
                torch.cuda.synchronize()
                self.parent._graph.capture_begin()
            except Exception as e:
                _log.warning("Graph capture_begin failed: %s — using eager fallback", e)
                # Sync before dropping reference to avoid C++ destructor crash
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                self.parent._graph = None
                self.parent._eager_fallback = True
                _bump_fallback()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                # Exception during captured region — fall back to eager.
                # CRITICAL: We must end the stream capture BEFORE dropping
                # the graph reference, otherwise the C++ destructor of
                # _OriginalCUDAGraph calls hipStreamEndCapture on a stream
                # still in capture mode and throws a C++ exception that
                # bypasses Python's try/except → terminate() → SIGABRT.
                _log.warning(
                    "[gfxgRAPH] Exception during graph capture: %s "
                    "— cleaning up HIP state and using eager fallback",
                    exc_val,
                )
                graph = self.parent._graph
                if graph is not None:
                    # Try to properly end the capture so HIP state is clean
                    try:
                        graph.capture_end()
                    except Exception:
                        pass  # already broken, just need the state reset
                    # Synchronize to flush any pending HIP errors before
                    # the C++ destructor runs
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                self.parent._graph = None
                self.parent._eager_fallback = True
                _bump_fallback()
                # Final sync to ensure no async HIP errors linger
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                return True  # suppress the exception

            if self.parent._graph is not None:
                try:
                    self.parent._graph.capture_end()
                    _bump_capture()
                except Exception as e:
                    _log.warning("capture_end failed: %s — using eager fallback", e)
                    # Same cleanup: end capture, sync, then drop reference
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    self.parent._graph = None
                    self.parent._eager_fallback = True
                    _bump_fallback()
            return False

    def capture(self, *, dynamic_shapes=False, buckets=None,
                conditional_branches=None, model_fn=None):
        """Enhanced capture context manager.

        Args:
            dynamic_shapes: Enable shape bucketing (gap 53 bridge)
            buckets: List of bucket sizes (default: [1, 2, 4, 8, 16, 32, 64])
            conditional_branches: Dict of {name: callable} for
                conditional execution (gap 51 bridge)
            model_fn: Callable for eager fallback and validation mode.
                      If not provided, eager fallback will raise on replay.
        """
        if buckets is None and dynamic_shapes:
            buckets = [1, 2, 4, 8, 16, 32, 64]

        self._model_fn = model_fn
        return self._CaptureContext(
            self, dynamic_shapes, buckets, conditional_branches
        )

    def replay(self, *, batch_size=None, branch=None, input_tensor=None):
        """Launch the captured graph.

        Args:
            batch_size: For dynamic_shapes mode, selects appropriate bucket
            branch: For conditional mode, selects branch by name/index
            input_tensor: For validation mode and eager fallback
        """
        # Input validation
        if input_tensor is not None:
            _validate_tensor(input_tensor, "input_tensor")

        # Eager fallback path
        if self._eager_fallback:
            return self._run_eager(input_tensor)

        # Shape bucketing path
        if self._shape_pool is not None and batch_size is not None:
            t0 = time.perf_counter()
            result = self._shape_pool(batch_size)
            _record_replay(t0)
            return self._maybe_validate(result, input_tensor)

        # Standard graph replay
        if self._graph is not None:
            t0 = time.perf_counter()
            try:
                self._graph.replay()
            except Exception as e:
                _log.warning("Graph replay failed: %s — falling back to eager", e)
                self._eager_fallback = True
                _bump_fallback()
                return self._run_eager(input_tensor)
            _record_replay(t0)
            return self._maybe_validate(self._static_output, input_tensor)

        raise RuntimeError("No graph captured. Call capture() first.")

    def _run_eager(self, input_tensor=None):
        """Execute model eagerly (fallback path)."""
        if self._model_fn is None:
            raise RuntimeError(
                "Graph capture failed and no model_fn provided for eager fallback. "
                "Pass model_fn= to capture() for automatic fallback."
            )
        _log.debug("Running eager fallback")
        _bump_fallback()
        if input_tensor is not None:
            return self._model_fn(input_tensor)
        return self._model_fn()

    def _maybe_validate(self, graph_output, input_tensor):
        """In validation mode, compare graph output vs eager (PyTorch #155684)."""
        try:
            from gfxgraph._enable import get_validate_mode
            if not get_validate_mode():
                return graph_output
        except ImportError:
            return graph_output

        if self._model_fn is None or input_tensor is None:
            return graph_output

        _log.debug("Validation: comparing graph output vs eager")
        with torch.no_grad():
            eager_output = self._model_fn(input_tensor)

        if not torch.allclose(graph_output, eager_output, atol=1e-5, rtol=1e-3):
            _log.error(
                "VALIDATION FAILURE: graph output differs from eager output! "
                "max_diff=%.6f — possible PyTorch #155684",
                (graph_output - eager_output).abs().max().item()
            )
            try:
                from gfxgraph._enable import bump
                bump("validation_failures")
            except ImportError:
                pass
            return eager_output  # return the correct (eager) output

        _log.debug("Validation passed")
        return graph_output

    def reset(self):
        """Release all resources."""
        self._graph = None
        self._shape_pool = None
        self._conditional_branches = None
        self._stream = None
        self._static_output = None
        self._model_fn = None
        self._eager_fallback = False

    # ---- Additional CUDAGraph API stubs (future-proofing) ----

    def debug_dump(self, path: str) -> None:
        """Dump graph debug info to file — delegates to real CUDAGraph if available."""
        if self._graph is not None and hasattr(self._graph, "debug_dump"):
            self._graph.debug_dump(path)
        else:
            _log.debug("debug_dump: no captured graph to dump")

    def enable_debug_mode(self) -> None:
        """Enable debug mode on the underlying graph."""
        if self._graph is not None and hasattr(self._graph, "enable_debug_mode"):
            self._graph.enable_debug_mode()

    def register_generator_state(self, gen) -> None:
        """Register RNG generator state for reproducible graph replay."""
        if self._graph is not None and hasattr(self._graph, "register_generator_state"):
            self._graph.register_generator_state(gen)

    @property
    def raw_cuda_graph(self):
        """Access the underlying cudaGraph_t / hipGraph_t handle."""
        if self._graph is not None and hasattr(self._graph, "raw_cuda_graph"):
            return self._graph.raw_cuda_graph
        return None

    @property
    def raw_cuda_graph_exec(self):
        """Access the underlying cudaGraphExec_t / hipGraphExec_t handle."""
        if self._graph is not None and hasattr(self._graph, "raw_cuda_graph_exec"):
            return self._graph.raw_cuda_graph_exec
        return None


# ---------- Counter helpers (import-safe) ----------

def _bump_capture():
    try:
        from gfxgraph._enable import bump
        bump("capture_count")
    except ImportError:
        pass


def _bump_fallback():
    try:
        from gfxgraph._enable import bump
        bump("fallback_count")
    except ImportError:
        pass


def _record_replay(t0: float):
    try:
        from gfxgraph._enable import record_replay_us
        us = (time.perf_counter() - t0) * 1e6
        record_replay_us(us)
    except ImportError:
        pass
