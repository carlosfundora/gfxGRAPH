"""
ShapeBucketPool — Manages a pool of CUDAGraph instances across shape buckets.

Follows the vLLM/SGLang shape-bucketing pattern: pre-capture one graph per
bucket size, then at runtime select the smallest bucket >= input_size.

Hardened with:
  - VRAM monitoring before bucket capture (cap at configurable %)
  - Lazy bucket instantiation (only capture when first needed)
  - Eager fallback on capture failure
  - Input tensor validation
"""

import bisect
import weakref

try:
    import rs_gfxgraph as _rs_gfxgraph
except Exception:
    _rs_gfxgraph = None

_HAS_BUCKET_ROUTER = _rs_gfxgraph is not None and hasattr(_rs_gfxgraph, "BucketRouter")

import logging
import os
from typing import Callable, List, Optional, Tuple

import torch

from hipgraph_bridge.capture_safety import (
    torch_graph_capture_block_reason,
    torch_cuda_execution_error,
    torch_cuda_execution_usable,
    unsafe_torch_graph_capture_enabled,
)

try:
    from gfxgraph._enable import bump as _bump
except ImportError:
    _bump = None

_log = logging.getLogger("gfxgraph")
if _rs_gfxgraph is not None and not _HAS_BUCKET_ROUTER:
    _log.warning("rs_gfxgraph loaded without BucketRouter; using Python router fallback")

# Configurable VRAM cap (fraction, 0.0-1.0). Default: 80%.
_VRAM_CAP = float(os.environ.get("GFXGRAPH_VRAM_CAP", "0.80"))


def _vram_available() -> tuple:
    """Return (free_bytes, total_bytes) or (0, 0) on error."""
    try:
        return torch.cuda.mem_get_info(0)
    except Exception:
        return (0, 0)


class ShapeBucketPool:
    """Shape-aware CUDAGraph pool for dynamic input sizes.

    Usage:
        pool = ShapeBucketPool(model_fn, buckets=[1, 4, 8, 16, 32, 64])
        output = pool(input_tensor)  # auto-selects bucket, pads, runs

    VRAM safety: Will not capture a new bucket if doing so would exceed
    GFXGRAPH_VRAM_CAP (default 80%) of total GPU memory. Falls back to
    eager execution in that case.
    """

    def __init__(
        self,
        model_fn: Optional[Callable] = None,
        buckets: Optional[List[int]] = None,
        warmup: bool = False,
    ):
        if model_fn is not None and not callable(model_fn):
            raise TypeError("model_fn must be callable")

        self.model_fn = model_fn
        self.buckets = sorted(buckets or [1, 2, 4, 8, 16, 32, 64])
        if _HAS_BUCKET_ROUTER:
            self._router = _rs_gfxgraph.BucketRouter(self.buckets)
            self._warmed_up = None
            self._failed_buckets = None
        else:
            self._router = None
            self._warmed_up = set()
            self._failed_buckets = set()  # buckets that failed capture

        self._graphs = {}      # bucket_size -> CUDAGraph
        self._graph_pools = {}  # bucket_size -> graph_pool_handle()
        self._static_inputs = {}  # bucket_size -> static input tensor
        self._max_static_input = None  # legacy alias for older callers
        self._static_outputs = {}  # bucket_size → static output tensor
        # Clone-free output materialization ring (automatic, no user config).
        self._output_slots = {}      # bucket_size -> List[Tensor]
        self._output_slot_refs = {}  # bucket_size -> List[weakref | None]
        self._output_slot_cursor = {}  # bucket_size -> next slot index
        self._copy_fastpath_disabled = set()  # bucket sizes forced to clone fallback

        self._cached_vram = None

        if warmup and model_fn is not None:
            self._warmup_all()

    def _get_vram(self) -> tuple:
        if self._cached_vram is not None:
            return self._cached_vram
        return _vram_available()

    def _warmup_all(self):
        """Pre-capture graphs for all bucket sizes (respects VRAM cap)."""
        self._cached_vram = _vram_available()
        try:
            if not self._check_vram():
                return
            for i, size in enumerate(self.buckets):
                if i > 0 and i % 5 == 0:
                    self._cached_vram = _vram_available()
                self._capture_bucket(size, skip_vram_check=True)
        finally:
            self._cached_vram = None

    def _check_vram(self) -> bool:
        """Return True if we have headroom below the VRAM cap."""
        free, total = self._get_vram()
        if total == 0:
            return True  # can't check — allow capture
        used_frac = 1.0 - (free / total)
        if used_frac >= _VRAM_CAP:
            _log.warning(
                "VRAM usage %.1f%% exceeds cap %.0f%% — skipping graph capture",
                used_frac * 100, _VRAM_CAP * 100,
            )
            return False
        return True

    def _is_warmed_up(self, bucket_size: int) -> bool:
        if self._router is not None:
            # We don't expose pure `is_warmed_up` directly anymore from the router natively,
            # but we can check via route
            bucket, state = self._router.route(bucket_size)
            return state == 0
        return bucket_size in self._warmed_up

    def _is_failed(self, bucket_size: int) -> bool:
        if self._router is not None:
            bucket, state = self._router.route(bucket_size)
            return state == 2
        return bucket_size in self._failed_buckets

    def _mark_warmed_up(self, bucket_size: int):
        if self._router is not None:
            self._router.mark_warmed_up(bucket_size)
        else:
            self._warmed_up.add(bucket_size)

    def _mark_failed(self, bucket_size: int):
        if self._router is not None:
            self._router.mark_failed(bucket_size)
        else:
            self._failed_buckets.add(bucket_size)

    def _capture_bucket(
        self,
        bucket_size: int,
        example_input=None,
        skip_vram_check: bool = False,
    ) -> bool:
        """Capture a CUDAGraph for the given bucket size. Returns True on success."""
        # For direct capture calls (e.g. warmup_all), we evaluate state manually
        if self._is_warmed_up(bucket_size):
            return True
        if self._is_failed(bucket_size):
            return False
        if self.model_fn is None:
            return False
        if not unsafe_torch_graph_capture_enabled():
            _log.warning(
                "Graph capture skipped for bucket %d: %s",
                bucket_size,
                torch_graph_capture_block_reason(),
            )
            self._mark_failed(bucket_size)
            if _bump is not None:
                _bump("fallback_count")
            return False

        clear_cache = False
        if self._cached_vram is None:
            self._cached_vram = _vram_available()
            clear_cache = True

        try:
            if not skip_vram_check and not self._check_vram():
                return False

            device = torch.device("cuda")

            if bucket_size not in self._static_inputs:
                if example_input is not None and isinstance(example_input, torch.Tensor):
                    shape = list(example_input.shape)
                    shape[0] = bucket_size
                    static_input = torch.zeros(shape, dtype=example_input.dtype, device=device)
                else:
                    static_input = torch.zeros(bucket_size, device=device)
                self._static_inputs[bucket_size] = static_input
                if self._max_static_input is None or bucket_size == self.buckets[-1]:
                    self._max_static_input = static_input
            else:
                static_input = self._static_inputs[bucket_size]

            # Warmup run (required before capture)
            torch.cuda.synchronize()
            with torch.no_grad():
                _ = self.model_fn(static_input)
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            self._graph_pools[bucket_size] = pool
            with torch.cuda.graph(graph, pool=pool):
                static_output = self.model_fn(static_input)

            self._graphs[bucket_size] = graph
            self._static_outputs[bucket_size] = static_output
            self._init_output_ring(bucket_size, static_output)
            self._mark_warmed_up(bucket_size)
            _log.debug("Captured graph for bucket size %d", bucket_size)
            return True

        except Exception as e:
            _log.warning(
                "Graph capture failed for bucket %d: %s — will use eager fallback",
                bucket_size, e,
            )
            self._mark_failed(bucket_size)
            # Clean up partial state
            self._static_outputs.pop(bucket_size, None)
            self._graphs.pop(bucket_size, None)
            self._graph_pools.pop(bucket_size, None)
            self._static_inputs.pop(bucket_size, None)
            if _bump is not None:
                _bump("fallback_count")
            return False
        finally:
            if clear_cache:
                self._cached_vram = None

    def route_bucket(self, input_size: int) -> Tuple[int, int]:
        """Find the smallest bucket >= input_size and return its current state.
        State: 0=Ready, 1=NeedsWarmup, 2=Failed.
        """
        if self._router is not None:
            return self._router.route(input_size)

        idx = bisect.bisect_left(self.buckets, input_size)
        if idx >= len(self.buckets):
            _log.warning(
                "Input size %d exceeds largest bucket %d. Falling back to eager.",
                input_size, self.buckets[-1]
            )
            return -1, 2
        bucket = self.buckets[idx]
        if bucket in self._warmed_up:
            state = 0
        elif bucket in self._failed_buckets:
            state = 2
        else:
            state = 1
        return bucket, state

    def __call__(self, input_tensor_or_size) -> torch.Tensor:
        """Run with automatic bucket selection.

        Args:
            input_tensor_or_size: Either a torch.Tensor (uses shape[0])
                                  or an int (batch size).
        """
        if isinstance(input_tensor_or_size, torch.Tensor):
            input_tensor = input_tensor_or_size
            input_size = input_tensor.shape[0]
            if input_tensor.is_cuda and not input_tensor.is_contiguous():
                _log.warning("Input tensor not contiguous — this may hurt perf")
        elif isinstance(input_tensor_or_size, int):
            input_size = input_tensor_or_size
            input_tensor = None
        else:
            raise TypeError(
                f"Expected torch.Tensor or int, got {type(input_tensor_or_size).__name__}"
            )

        bucket, state = self.route_bucket(input_size)

        # Lazy capture if not warmed up
        if state == 1:
            if not self._capture_bucket(bucket, input_tensor):
                # Capture failed or VRAM exceeded — eager fallback
                return self._eager_fallback(input_tensor, input_size, bump=False)
        elif state == 2:
            return self._eager_fallback(input_tensor, input_size, bump=False)

        if bucket not in self._graphs:
            return self._eager_fallback(input_tensor, input_size, bump=False)

        # Copy input to static buffer (only input_size elements)
        if input_tensor is not None:
            static_in = self._static_inputs[bucket]
            static_in[:input_size].copy_(input_tensor[:input_size])
            if input_size < bucket:
                static_in[input_size:].zero_()  # Zero-pad

        # Replay
        self._graphs[bucket].replay()

        return self._materialize_output(bucket, input_size)

    def _ring_slot_count(self, static_output: torch.Tensor) -> int:
        # Prefer 3 slots, but scale down under low free VRAM.
        try:
            bytes_per_slot = static_output.nelement() * static_output.element_size()
        except Exception:
            return 1
        free, _ = self._get_vram()
        if free <= 0:
            return 2
        if free >= bytes_per_slot * 8:
            return 3
        if free >= bytes_per_slot * 4:
            return 2
        return 1

    def _init_output_ring(self, bucket_size: int, static_output):
        if bucket_size in self._output_slots:
            return
        if not isinstance(static_output, torch.Tensor):
            self._copy_fastpath_disabled.add(bucket_size)
            self._output_slots[bucket_size] = []
            self._output_slot_refs[bucket_size] = []
            self._output_slot_cursor[bucket_size] = 0
            return
        slot_count = self._ring_slot_count(static_output)
        slots = [torch.empty_like(static_output) for _ in range(slot_count)]
        refs = [None] * slot_count
        self._output_slots[bucket_size] = slots
        self._output_slot_refs[bucket_size] = refs
        self._output_slot_cursor[bucket_size] = 0

    def _materialize_output(self, bucket_size: int, input_size: int):
        static_out = self._static_outputs[bucket_size]
        if not isinstance(static_out, torch.Tensor):
            return static_out

        # Safe fallback for buckets where ring logic cannot be guaranteed.
        if bucket_size in self._copy_fastpath_disabled:
            return static_out[:input_size].clone()

        slots = self._output_slots.get(bucket_size)
        refs = self._output_slot_refs.get(bucket_size)
        if not slots or refs is None:
            return static_out[:input_size].clone()

        n = len(slots)
        cursor = self._output_slot_cursor.get(bucket_size, 0)
        chosen = None
        for i in range(n):
            idx = (cursor + i) % n
            ref = refs[idx]
            if ref is None or ref() is None:
                chosen = idx
                break

        # All slots currently referenced by caller => preserve correctness.
        if chosen is None:
            return static_out[:input_size].clone()

        try:
            slots[chosen].copy_(static_out)
            result = slots[chosen][:input_size]
            refs[chosen] = weakref.ref(result)
            self._output_slot_cursor[bucket_size] = (chosen + 1) % n
            return result
        except Exception:
            self._copy_fastpath_disabled.add(bucket_size)
            return static_out[:input_size].clone()

    def _eager_fallback(self, input_tensor, input_size, *, bump: bool = True):
        """Run model eagerly when graph isn't available."""
        if self.model_fn is None:
            raise RuntimeError(
                "No graph available and no model_fn for eager fallback"
            )
        _log.debug("Shape pool eager fallback for size %d", input_size)
        if bump and _bump is not None:
            _bump("fallback_count")
        if input_tensor is not None and input_tensor.is_cuda and not torch_cuda_execution_usable():
            raise RuntimeError(
                "Cannot run eager fallback because CUDA/HIP execution is not usable: "
                f"{torch_cuda_execution_error()}"
            )

        if input_tensor is not None:
            with torch.no_grad():
                return self.model_fn(input_tensor)
        else:
            dummy = torch.zeros(input_size, device="cuda")
            with torch.no_grad():
                return self.model_fn(dummy)

    @property
    def memory_overhead(self) -> int:
        """Total GPU memory used by all bucket graphs (bytes)."""
        total = 0
        for static_input in self._static_inputs.values():
            total += static_input.nelement() * static_input.element_size()
        for out in self._static_outputs.values():
            total += out.nelement() * out.element_size()
        for slots in self._output_slots.values():
            for slot in slots:
                total += slot.nelement() * slot.element_size()
        return total

    @property
    def captured_buckets(self) -> list:
        """List of bucket sizes that have been successfully captured."""
        if self._router is not None:
            return self._router.warmed_up_list()
        return sorted(self._warmed_up)

    @property
    def failed_buckets(self) -> list:
        """List of bucket sizes that failed to capture."""
        if self._router is not None:
            return self._router.failed_list()
        return sorted(self._failed_buckets)

    # Maintain legacy property for compatibility during refactoring tools
    @property
    def select_bucket(self):
        def _wrapper(input_size):
            b, _ = self.route_bucket(input_size)
            return b
        return _wrapper
