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
import os
import threading
import time

import torch

from hipgraph_bridge.shape_bucketing import ShapeBucketPool

try:
    from gfxgraph._enable import bump as _bump
except ImportError:
    _bump = None

try:
    from gfxgraph._enable import record_replay_us as _record_replay_us
except ImportError:
    _record_replay_us = None

try:
    from gfxgraph._enable import get_validate_mode as _get_validate_mode
except ImportError:
    _get_validate_mode = None

try:
    import gfxgraph_rs as _gfxgraph_rs
except Exception:
    _gfxgraph_rs = None

_HAS_BRIDGED_VALIDATOR = _gfxgraph_rs is not None and hasattr(_gfxgraph_rs, "BridgedGraphValidator")

_log = logging.getLogger("gfxgraph")

def _resolve_replay_mode(mode_env: str, legacy_hot: bool) -> str:
    valid_modes = {"standard", "adaptive", "hot"}
    raw = (mode_env or "").strip().lower()
    if raw:
        return raw if raw in valid_modes else "standard"
    if legacy_hot:
        return "hot"
    return "standard"


_ADAPTIVE_REPLAYS = max(0, int(os.environ.get("GFXGRAPH_ADAPTIVE_REPLAYS", "8")))
_STATS_FLUSH_INTERVAL = max(1, int(os.environ.get("GFXGRAPH_REPLAY_STATS_FLUSH", "32")))
_ADAPTIVE_EAGER_BIAS = max(1.0, float(os.environ.get("GFXGRAPH_ADAPTIVE_EAGER_BIAS", "1.10")))
_ADAPTIVE_GRAPH_ADVANTAGE = min(1.0, max(0.5, float(os.environ.get("GFXGRAPH_ADAPTIVE_GRAPH_ADVANTAGE", "0.98"))))
_ADAPTIVE_SIGNATURE_CACHE_MAX = max(32, int(os.environ.get("GFXGRAPH_ADAPTIVE_SIGNATURE_CACHE_MAX", "512")))
_ADAPTIVE_SPIKE_SWITCH_FACTOR = max(1.0, float(os.environ.get("GFXGRAPH_ADAPTIVE_SPIKE_SWITCH_FACTOR", "1.20")))
_ADAPTIVE_SPIKE_SWITCH_HITS = max(1, int(os.environ.get("GFXGRAPH_ADAPTIVE_SPIKE_SWITCH_HITS", "2")))
_LEGACY_HOT_REPLAY_MODE = os.environ.get("GFXGRAPH_REPLAY_HOT_MODE", "").strip() in {"1", "true", "TRUE", "yes", "on"}
_REPLAY_MODE = _resolve_replay_mode(os.environ.get("GFXGRAPH_REPLAY_MODE", ""), _LEGACY_HOT_REPLAY_MODE)
_HOT_REPLAY_MODE = _REPLAY_MODE == "hot"
_ADAPTIVE_REPLAY_MODE = _REPLAY_MODE == "adaptive"
_TRUSTED_REPLAY_THRESHOLD = max(0, int(os.environ.get("GFXGRAPH_TRUSTED_REPLAY_THRESHOLD", "16")))
_TRUSTED_REPLAY_SAMPLE_INTERVAL = max(1, int(os.environ.get("GFXGRAPH_TRUSTED_REPLAY_SAMPLE_INTERVAL", "16")))
_ADAPTIVE_SIGNATURE_CACHE_LOCK = threading.Lock()
_ADAPTIVE_SIGNATURE_CACHE = {}

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
        self._capture_ctx = None
        self._shape_pool = None
        self._conditional_branches = None
        self._stream = None
        self._static_output = None
        self._model_fn = None       # stored for eager fallback
        self._eager_fallback = False  # set True on capture failure
        self._last_input = None      # for validation mode
        self._prefer_eager = False
        self._adaptive_replay_samples = 0
        self._adaptive_graph_total_us = 0.0
        self._adaptive_eager_total_us = 0.0
        self._adaptive_disabled = False
        self._validation_enabled_cached = False
        self._replay_stat_buffer_count = 0
        self._replay_stat_buffer_us = 0.0
        self._adaptive_signature = None
        self._eager_baseline_us = None
        self._adaptive_graph_spikes = 0
        self._trusted_replay_active = False
        self._trusted_replay_count = 0

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
                self.parent._capture_ctx = torch.cuda.graph(
                    self.parent._graph, stream=self.parent._stream
                )
                self.parent._capture_ctx.__enter__()
            except Exception as e:
                _log.warning("Graph capture_begin failed: %s — using eager fallback", e)
                # Sync before dropping reference to avoid C++ destructor crash
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                self.parent._capture_ctx = None
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
                if self.parent._capture_ctx is not None:
                    try:
                        self.parent._capture_ctx.__exit__(exc_type, exc_val, exc_tb)
                    except Exception:
                        pass
                    self.parent._capture_ctx = None
                elif graph is not None:
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
                    if self.parent._capture_ctx is not None:
                        self.parent._capture_ctx.__exit__(None, None, None)
                        self.parent._capture_ctx = None
                    else:
                        self.parent._graph.capture_end()
                    _bump_capture()
                except Exception as e:
                    _log.warning("capture_end failed: %s — using eager fallback", e)
                    # Same cleanup: end capture, sync, then drop reference
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    self.parent._capture_ctx = None
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
        self._prefer_eager = False
        self._adaptive_replay_samples = 0
        self._adaptive_graph_total_us = 0.0
        self._adaptive_eager_total_us = 0.0
        self._adaptive_disabled = False
        self._replay_stat_buffer_count = 0
        self._replay_stat_buffer_us = 0.0
        self._validation_enabled_cached = bool(_get_validate_mode and _get_validate_mode())
        self._adaptive_signature = None
        self._eager_baseline_us = None
        self._adaptive_graph_spikes = 0
        self._trusted_replay_active = False
        self._trusted_replay_count = 0
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
        if _ADAPTIVE_REPLAY_MODE:
            self._maybe_load_cached_decision(batch_size, input_tensor)
        if _ADAPTIVE_REPLAY_MODE and self._prefer_eager:
            return self._run_preferred_eager(input_tensor)

        # Shape bucketing path
        if self._shape_pool is not None and batch_size is not None:
            t0 = time.perf_counter()
            result = self._shape_pool(batch_size)
            if not _HOT_REPLAY_MODE:
                self._record_replay_sample(t0)
            return self._maybe_validate(result, input_tensor)

        # Standard graph replay
        if self._graph is not None:
            if _HOT_REPLAY_MODE:
                self._graph.replay()
                return self._static_output
            sample_diagnostics = True
            if _TRUSTED_REPLAY_THRESHOLD > 0 and self._trusted_replay_active:
                sample_diagnostics = (self._trusted_replay_count % _TRUSTED_REPLAY_SAMPLE_INTERVAL) == 0
            t0 = time.perf_counter() if sample_diagnostics else 0.0
            try:
                self._graph.replay()
            except Exception as e:
                _log.warning("Graph replay failed: %s — falling back to eager", e)
                self._eager_fallback = True
                _bump_fallback()
                return self._run_eager(input_tensor)
            self._trusted_replay_count += 1
            if (
                _TRUSTED_REPLAY_THRESHOLD > 0
                and not self._trusted_replay_active
                and self._trusted_replay_count >= _TRUSTED_REPLAY_THRESHOLD
            ):
                self._trusted_replay_active = True
            if sample_diagnostics:
                self._record_replay_sample(t0)
                if _ADAPTIVE_REPLAY_MODE:
                    self._maybe_update_adaptive_mode(input_tensor)
                return self._maybe_validate(self._static_output, input_tensor)
            return self._static_output

        raise RuntimeError("No graph captured. Call capture() first.")

    def _run_eager(self, input_tensor=None):
        """Execute model eagerly (fallback path)."""
        if self._model_fn is None:
            if input_tensor is None and self._static_output is not None:
                return self._static_output
            raise RuntimeError(
                "Graph capture failed and no model_fn provided for eager fallback. "
                "Pass model_fn= to capture() for automatic fallback."
            )
        _log.debug("Running eager fallback")
        _bump_fallback()
        if input_tensor is not None:
            return self._model_fn(input_tensor)
        return self._model_fn()

    def _run_preferred_eager(self, input_tensor=None):
        """Execute eager path selected by adaptive policy (non-fallback)."""
        if input_tensor is not None:
            return self._model_fn(input_tensor)
        return self._model_fn()

    def _run_eager_timed(self, input_tensor=None):
        t0 = time.perf_counter()
        with torch.no_grad():
            if input_tensor is not None:
                out = self._model_fn(input_tensor)
            else:
                out = self._model_fn()
        us = (time.perf_counter() - t0) * 1e6
        return out, us

    def _maybe_update_adaptive_mode(self, input_tensor):
        if not _ADAPTIVE_REPLAY_MODE:
            return
        if _ADAPTIVE_REPLAYS <= 0 or self._adaptive_disabled or self._model_fn is None:
            return
        if self._adaptive_replay_samples >= _ADAPTIVE_REPLAYS:
            return
        if self._adaptive_signature is None:
            self._adaptive_signature = self._build_adaptive_signature(None, input_tensor)

        try:
            _, eager_us = self._run_eager_timed(input_tensor)
        except TypeError:
            # model_fn requires an input tensor, but replay caller did not provide one.
            self._adaptive_disabled = True
            return
        except Exception as exc:
            _log.debug("Adaptive eager probe failed: %s", exc)
            self._adaptive_disabled = True
            return

        self._adaptive_eager_total_us += eager_us
        self._adaptive_replay_samples += 1
        if self._adaptive_replay_samples >= _ADAPTIVE_REPLAYS:
            graph_total = self._adaptive_graph_total_us
            eager_total = self._adaptive_eager_total_us
            if graph_total <= (eager_total * _ADAPTIVE_GRAPH_ADVANTAGE):
                self._prefer_eager = False
                _log.debug(
                    "Adaptive path kept graph: eager=%.2fus graph=%.2fus over %d samples (graph_adv=%.3f)",
                    eager_total,
                    graph_total,
                    self._adaptive_replay_samples,
                    _ADAPTIVE_GRAPH_ADVANTAGE,
                )
            elif eager_total <= (graph_total * _ADAPTIVE_EAGER_BIAS):
                self._prefer_eager = True
                _log.debug(
                    "Adaptive path selected eager: eager=%.2fus graph=%.2fus over %d samples (bias=%.3f)",
                    eager_total,
                    graph_total,
                    self._adaptive_replay_samples,
                    _ADAPTIVE_EAGER_BIAS,
                )
            else:
                self._prefer_eager = False
                _log.debug(
                    "Adaptive path kept graph: eager=%.2fus graph=%.2fus over %d samples",
                    eager_total,
                    graph_total,
                    self._adaptive_replay_samples,
                )
            if self._adaptive_signature is not None:
                _set_adaptive_signature_decision(
                    self._adaptive_signature,
                    "eager" if self._prefer_eager else "graph",
                    eager_total / self._adaptive_replay_samples,
                    graph_total / self._adaptive_replay_samples,
                )
            self._eager_baseline_us = eager_total / self._adaptive_replay_samples

    def _build_adaptive_signature(self, batch_size, input_tensor):
        if self._model_fn is None:
            return None

        fn_obj = getattr(self._model_fn, "__func__", self._model_fn)
        fn_module = getattr(fn_obj, "__module__", type(fn_obj).__module__)
        fn_name = getattr(fn_obj, "__qualname__", getattr(fn_obj, "__name__", type(fn_obj).__name__))
        fn_code = getattr(fn_obj, "__code__", None)
        if fn_code is not None:
            fn_sig = f"{fn_module}:{fn_name}:{fn_code.co_firstlineno}:{fn_code.co_argcount}:{len(fn_code.co_code)}"
        else:
            fn_sig = f"{fn_module}:{fn_name}:{type(fn_obj).__name__}"

        ref_tensor = input_tensor
        if ref_tensor is None and self._static_output is not None:
            ref_tensor = self._static_output

        shape = tuple(int(dim) for dim in getattr(ref_tensor, "shape", ())) if ref_tensor is not None else ()
        dtype = str(getattr(ref_tensor, "dtype", "unknown")) if ref_tensor is not None else "unknown"
        device = str(getattr(ref_tensor, "device", "unknown")) if ref_tensor is not None else "unknown"
        stride_obj = getattr(ref_tensor, "stride", None) if ref_tensor is not None else None
        if callable(stride_obj):
            try:
                stride = tuple(int(dim) for dim in stride_obj())
            except Exception:
                stride = ()
        else:
            stride = ()

        batch_hint = batch_size
        if batch_hint is None and shape:
            batch_hint = shape[0]

        return f"{fn_sig}|b={batch_hint}|shape={shape}|dtype={dtype}|device={device}|stride={stride}"

    def _maybe_load_cached_decision(self, batch_size, input_tensor):
        if not _ADAPTIVE_REPLAY_MODE:
            return
        if self._adaptive_disabled:
            return
        if self._adaptive_signature is None:
            self._adaptive_signature = self._build_adaptive_signature(batch_size, input_tensor)
        if self._adaptive_signature is None:
            return
        cached = _get_adaptive_signature_decision(self._adaptive_signature)
        if cached is None:
            return
        self._prefer_eager = cached.get("mode") == "eager"
        self._adaptive_disabled = True
        self._eager_baseline_us = cached.get("eager_avg_us")

    def _record_replay_sample(self, t0: float):
        us = (time.perf_counter() - t0) * 1e6
        self._adaptive_graph_total_us += us
        self._replay_stat_buffer_us += us
        self._replay_stat_buffer_count += 1
        if not self._prefer_eager and self._eager_baseline_us:
            if us > (self._eager_baseline_us * _ADAPTIVE_SPIKE_SWITCH_FACTOR):
                self._adaptive_graph_spikes += 1
                if self._adaptive_graph_spikes >= _ADAPTIVE_SPIKE_SWITCH_HITS:
                    self._prefer_eager = True
                    self._adaptive_disabled = True
                    if self._adaptive_signature is not None:
                        _set_adaptive_signature_decision(
                            self._adaptive_signature,
                            "eager",
                            self._eager_baseline_us,
                            us,
                        )
                    _log.debug(
                        "Adaptive spike guard switched to eager: replay_us=%.2f baseline_us=%.2f factor=%.2f",
                        us,
                        self._eager_baseline_us,
                        _ADAPTIVE_SPIKE_SWITCH_FACTOR,
                    )
            elif self._adaptive_graph_spikes > 0:
                self._adaptive_graph_spikes -= 1
        if self._replay_stat_buffer_count >= _STATS_FLUSH_INTERVAL:
            self._flush_replay_stats()

    def _flush_replay_stats(self):
        if self._replay_stat_buffer_count == 0:
            return
        count = self._replay_stat_buffer_count
        avg_us = self._replay_stat_buffer_us / count
        if _record_replay_us is not None:
            for _ in range(count):
                _record_replay_us(avg_us)
        self._replay_stat_buffer_count = 0
        self._replay_stat_buffer_us = 0.0

    def _maybe_validate(self, graph_output, input_tensor):
        """In validation mode, compare graph output vs eager (PyTorch #155684)."""
        if _HOT_REPLAY_MODE:
            return graph_output
        validation_enabled = self._validation_enabled_cached

        if not validation_enabled or self._model_fn is None or input_tensor is None:
            return graph_output

        if _HAS_BRIDGED_VALIDATOR:
            validator = _gfxgraph_rs.BridgedGraphValidator(validation_enabled)
            return validator.maybe_validate(graph_output, input_tensor, self._model_fn)

        _log.debug("Validation: comparing graph output vs eager")
        with torch.no_grad():
            eager_output = self._model_fn(input_tensor)

        if not torch.allclose(graph_output, eager_output, atol=1e-5, rtol=1e-3):
            _log.error(
                "VALIDATION FAILURE: graph output differs from eager output! "
                "max_diff=%.6f — possible PyTorch #155684",
                (graph_output - eager_output).abs().max().item()
            )
            if _bump is not None:
                _bump("validation_failures")
            return eager_output  # return the correct (eager) output

        _log.debug("Validation passed")
        return graph_output

    def reset(self):
        """Release all resources."""
        self._flush_replay_stats()
        self._graph = None
        self._shape_pool = None
        self._conditional_branches = None
        self._stream = None
        self._static_output = None
        self._model_fn = None
        self._eager_fallback = False
        self._prefer_eager = False
        self._adaptive_replay_samples = 0
        self._adaptive_graph_total_us = 0.0
        self._adaptive_eager_total_us = 0.0
        self._adaptive_disabled = False
        self._validation_enabled_cached = False
        self._adaptive_signature = None
        self._eager_baseline_us = None
        self._adaptive_graph_spikes = 0
        self._trusted_replay_active = False
        self._trusted_replay_count = 0

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
    if _bump is not None:
        _bump("capture_count")


def _bump_fallback():
    if _bump is not None:
        _bump("fallback_count")


def _record_replay(t0: float):
    if _record_replay_us is not None:
        us = (time.perf_counter() - t0) * 1e6
        _record_replay_us(us)


def _set_adaptive_signature_decision(
    signature: str,
    decision: str,
    eager_avg_us: float | None = None,
    graph_avg_us: float | None = None,
):
    if not signature:
        return
    payload = {
        "mode": decision,
        "eager_avg_us": eager_avg_us,
        "graph_avg_us": graph_avg_us,
    }
    with _ADAPTIVE_SIGNATURE_CACHE_LOCK:
        if signature in _ADAPTIVE_SIGNATURE_CACHE:
            _ADAPTIVE_SIGNATURE_CACHE.pop(signature, None)
        _ADAPTIVE_SIGNATURE_CACHE[signature] = payload
        while len(_ADAPTIVE_SIGNATURE_CACHE) > _ADAPTIVE_SIGNATURE_CACHE_MAX:
            _ADAPTIVE_SIGNATURE_CACHE.pop(next(iter(_ADAPTIVE_SIGNATURE_CACHE)))


def _get_adaptive_signature_decision(signature: str):
    if not signature:
        return None
    with _ADAPTIVE_SIGNATURE_CACHE_LOCK:
        return _ADAPTIVE_SIGNATURE_CACHE.get(signature)
