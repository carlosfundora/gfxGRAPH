"""
ShapeBucketPool — Manages a pool of CUDAGraph instances across shape buckets.

Follows the vLLM/SGLang shape-bucketing pattern: pre-capture one graph per
bucket size, then at runtime select the smallest bucket >= input_size.
"""

import bisect
from typing import Callable, List, Optional

import torch


class ShapeBucketPool:
    """Shape-aware CUDAGraph pool for dynamic input sizes.

    Usage:
        pool = ShapeBucketPool(model_fn, buckets=[1, 4, 8, 16, 32, 64])
        output = pool(input_tensor)  # auto-selects bucket, pads, runs
    """

    def __init__(
        self,
        model_fn: Optional[Callable] = None,
        buckets: Optional[List[int]] = None,
        warmup: bool = True,
    ):
        self.model_fn = model_fn
        self.buckets = sorted(buckets or [1, 2, 4, 8, 16, 32, 64])
        self._graphs = {}      # bucket_size → CUDAGraph
        self._static_inputs = {}   # bucket_size → static input tensor
        self._static_outputs = {}  # bucket_size → static output tensor
        self._warmed_up = set()

        if warmup and model_fn is not None:
            self._warmup_all()

    def _warmup_all(self):
        """Pre-capture graphs for all bucket sizes."""
        for size in self.buckets:
            self._capture_bucket(size)

    def _capture_bucket(self, bucket_size: int):
        """Capture a CUDAGraph for the given bucket size."""
        if bucket_size in self._warmed_up:
            return

        if self.model_fn is None:
            return

        device = torch.device("cuda")

        # Allocate static input
        static_input = torch.zeros(bucket_size, device=device)
        self._static_inputs[bucket_size] = static_input

        # Warmup run (required before capture)
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = self.model_fn(static_input)
        torch.cuda.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.model_fn(static_input)

        self._graphs[bucket_size] = graph
        self._static_outputs[bucket_size] = static_output
        self._warmed_up.add(bucket_size)

    def select_bucket(self, input_size: int) -> int:
        """Find the smallest bucket >= input_size."""
        idx = bisect.bisect_left(self.buckets, input_size)
        if idx >= len(self.buckets):
            raise ValueError(
                f"Input size {input_size} exceeds largest bucket "
                f"{self.buckets[-1]}. Add a larger bucket."
            )
        return self.buckets[idx]

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run with automatic bucket selection."""
        input_size = input_tensor.shape[0]
        bucket = self.select_bucket(input_size)

        # Lazy capture if not warmed up
        if bucket not in self._warmed_up:
            self._capture_bucket(bucket)

        if bucket not in self._graphs:
            raise RuntimeError(
                f"Graph for bucket {bucket} not captured. "
                f"Ensure model_fn is set."
            )

        # Copy input to static buffer (only input_size elements)
        static_in = self._static_inputs[bucket]
        static_in[:input_size].copy_(input_tensor)
        if input_size < bucket:
            static_in[input_size:].zero_()  # Zero-pad

        # Replay
        self._graphs[bucket].replay()

        # Slice output to actual size
        return self._static_outputs[bucket][:input_size].clone()

    @property
    def memory_overhead(self) -> int:
        """Total GPU memory used by all bucket graphs (bytes)."""
        total = 0
        for size, inp in self._static_inputs.items():
            total += inp.nelement() * inp.element_size()
        for size, out in self._static_outputs.items():
            total += out.nelement() * out.element_size()
        return total
