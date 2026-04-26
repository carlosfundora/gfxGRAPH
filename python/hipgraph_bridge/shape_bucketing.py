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
import logging
import os
from typing import Callable, List, Optional

import torch

_log = logging.getLogger("gfxgraph")

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
        self._graphs = {}      # bucket_size → CUDAGraph
        self._max_static_input = None
        self._static_outputs = {}  # bucket_size → static output tensor
        self._warmed_up = set()
        self._failed_buckets = set()  # buckets that failed capture
        self._mempool = torch.cuda.graph_pool_handle()

        if warmup and model_fn is not None:
            self._warmup_all()

    def _warmup_all(self):
        """Pre-capture graphs for all bucket sizes (respects VRAM cap)."""
        for size in self.buckets:
            self._capture_bucket(size)

    def _check_vram(self) -> bool:
        """Return True if we have headroom below the VRAM cap."""
        free, total = _vram_available()
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

    def _capture_bucket(self, bucket_size: int, example_input=None) -> bool:
        """Capture a CUDAGraph for the given bucket size. Returns True on success."""
        if bucket_size in self._warmed_up:
            return True
        if bucket_size in self._failed_buckets:
            return False
        if self.model_fn is None:
            return False
        if not self._check_vram():
            return False

        device = torch.device("cuda")

        try:
            # Check if we need to initialize or resize the shared max static input
            max_bucket = self.buckets[-1]
            if self._max_static_input is None:
                if example_input is not None and isinstance(example_input, torch.Tensor):
                    shape = list(example_input.shape)
                    shape[0] = max_bucket
                    self._max_static_input = torch.zeros(shape, dtype=example_input.dtype, device=device)
                else:
                    self._max_static_input = torch.zeros(max_bucket, device=device)

            # Use a slice of the max static input for this bucket's capture
            static_input = self._max_static_input[:bucket_size]

            # Warmup run (required before capture)
            torch.cuda.synchronize()
            with torch.no_grad():
                _ = self.model_fn(static_input)
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._mempool):
                static_output = self.model_fn(static_input)

            self._graphs[bucket_size] = graph
            self._static_outputs[bucket_size] = static_output
            self._warmed_up.add(bucket_size)
            _log.debug("Captured graph for bucket size %d", bucket_size)
            return True

        except Exception as e:
            _log.warning(
                "Graph capture failed for bucket %d: %s — will use eager fallback",
                bucket_size, e,
            )
            self._failed_buckets.add(bucket_size)
            # Clean up partial state
            self._static_outputs.pop(bucket_size, None)
            self._graphs.pop(bucket_size, None)
            return False

    def select_bucket(self, input_size: int) -> int:
        """Find the smallest bucket >= input_size."""
        idx = bisect.bisect_left(self.buckets, input_size)
        if idx >= len(self.buckets):
            raise ValueError(
                f"Input size {input_size} exceeds largest bucket "
                f"{self.buckets[-1]}. Add a larger bucket."
            )
        return self.buckets[idx]

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

        bucket = self.select_bucket(input_size)

        # Lazy capture if not warmed up
        if bucket not in self._warmed_up:
            if not self._capture_bucket(bucket, input_tensor):
                # Capture failed or VRAM exceeded — eager fallback
                return self._eager_fallback(input_tensor, input_size)

        if bucket not in self._graphs:
            return self._eager_fallback(input_tensor, input_size)

        # Copy input to static buffer (only input_size elements)
        if input_tensor is not None:
            static_in = self._max_static_input[:bucket]
            static_in[:input_size].copy_(input_tensor[:input_size])
            if input_size < bucket:
                static_in[input_size:].zero_()  # Zero-pad

        # Replay
        self._graphs[bucket].replay()

        # Slice output to actual size
        return self._static_outputs[bucket][:input_size].clone()

    def _eager_fallback(self, input_tensor, input_size):
        """Run model eagerly when graph isn't available."""
        if self.model_fn is None:
            raise RuntimeError(
                "No graph available and no model_fn for eager fallback"
            )
        _log.debug("Shape pool eager fallback for size %d", input_size)
        try:
            from gfxgraph._enable import bump
            bump("fallback_count")
        except ImportError:
            pass

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
        if self._max_static_input is not None:
            total += self._max_static_input.nelement() * self._max_static_input.element_size()
        # With CUDA graph pools, output memory is reused across buckets.
        # We approximate it by taking the max size.
        max_out = 0
        for out in self._static_outputs.values():
            sz = out.nelement() * out.element_size()
            if sz > max_out: max_out = sz
        total += max_out
        return total

    @property
    def captured_buckets(self) -> list:
        """List of bucket sizes that have been successfully captured."""
        return sorted(self._warmed_up)

    @property
    def failed_buckets(self) -> list:
        """List of bucket sizes that failed to capture."""
        return sorted(self._failed_buckets)
