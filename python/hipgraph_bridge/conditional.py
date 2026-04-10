"""
Conditional graph execution for PyTorch (Gap 51 bridge).

Uses per-branch graph dispatch (NOT supergraph + hipGraphNodeSetEnabled,
which is unreliable on gfx1030 for child graph nodes).

Hardened with:
  - Eager fallback on capture failure
  - Input tensor validation
  - Performance counter integration
"""

import logging
import time

import torch
from typing import Callable, Dict, Optional

_log = logging.getLogger("gfxgraph")


class ConditionalGraph:
    """Execute one of several pre-captured graphs based on a condition.

    Uses per-branch graph dispatch: each branch is captured as a separate
    CUDAGraph, and at runtime the requested branch's graph is replayed.
    This avoids hipGraphNodeSetEnabled limitations on gfx1030.

    Usage:
        cg = ConditionalGraph()
        cg.add_branch("training", training_fn)
        cg.add_branch("inference", inference_fn)
        cg.capture(example_input)

        # At runtime:
        output = cg.run("training", input_tensor)
    """

    def __init__(self):
        self._branches: Dict[str, Callable] = {}
        self._graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self._static_inputs: Dict[str, torch.Tensor] = {}
        self._static_outputs: Dict[str, torch.Tensor] = {}
        self._captured = False
        self._failed_branches: set = set()

    def add_branch(self, name: str, fn: Callable):
        """Register a branch function."""
        if self._captured:
            raise RuntimeError("Cannot add branches after capture")
        if not callable(fn):
            raise TypeError(f"Branch function must be callable, got {type(fn).__name__}")
        self._branches[name] = fn

    def capture(self, example_input: torch.Tensor):
        """Capture all branches as separate CUDAGraphs.

        If capture fails for a branch, it is marked as failed and will
        use eager fallback on run().
        """
        if not isinstance(example_input, torch.Tensor):
            raise TypeError(f"example_input must be a torch.Tensor, got {type(example_input).__name__}")
        if example_input.is_cuda and not example_input.is_contiguous():
            _log.warning("example_input not contiguous — cloning for capture")
            example_input = example_input.contiguous()

        for name, fn in self._branches.items():
            static_input = example_input.clone()
            self._static_inputs[name] = static_input

            try:
                # Warmup
                torch.cuda.synchronize()
                with torch.no_grad():
                    _ = fn(static_input)
                torch.cuda.synchronize()

                # Capture
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = fn(static_input)

                self._graphs[name] = graph
                self._static_outputs[name] = static_output
                _log.debug("Captured branch '%s'", name)

            except Exception as e:
                _log.warning(
                    "Capture failed for branch '%s': %s — will use eager fallback",
                    name, e,
                )
                self._failed_branches.add(name)
                try:
                    from gfxgraph._enable import bump
                    bump("fallback_count")
                except ImportError:
                    pass

        self._captured = True
        try:
            from gfxgraph._enable import bump
            bump("capture_count", len(self._graphs))
        except ImportError:
            pass

    def run(self, branch: str, input_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute a specific branch.

        Args:
            branch: Name of the branch to execute
            input_tensor: If provided, copied to static input before replay.
                         Must be on CUDA device.
        """
        if not self._captured:
            raise RuntimeError("Call capture() first")
        if branch not in self._branches:
            raise KeyError(
                f"Unknown branch '{branch}'. "
                f"Available: {list(self._branches.keys())}"
            )

        # Input validation
        if input_tensor is not None:
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError(f"input_tensor must be a torch.Tensor, got {type(input_tensor).__name__}")
            if not input_tensor.is_cuda:
                raise ValueError("input_tensor must be on CUDA device")

        # Failed branch → eager fallback
        if branch in self._failed_branches:
            return self._eager_fallback(branch, input_tensor)

        if input_tensor is not None:
            self._static_inputs[branch].copy_(input_tensor)

        t0 = time.perf_counter()
        try:
            self._graphs[branch].replay()
        except Exception as e:
            _log.warning("Replay failed for branch '%s': %s — eager fallback", branch, e)
            self._failed_branches.add(branch)
            return self._eager_fallback(branch, input_tensor)

        try:
            from gfxgraph._enable import record_replay_us
            us = (time.perf_counter() - t0) * 1e6
            record_replay_us(us)
        except ImportError:
            pass

        return self._static_outputs[branch]

    def _eager_fallback(self, branch: str, input_tensor: Optional[torch.Tensor]):
        """Run branch function eagerly."""
        fn = self._branches[branch]
        _log.debug("Eager fallback for branch '%s'", branch)
        try:
            from gfxgraph._enable import bump
            bump("fallback_count")
        except ImportError:
            pass

        if input_tensor is not None:
            with torch.no_grad():
                return fn(input_tensor)
        elif branch in self._static_inputs:
            with torch.no_grad():
                return fn(self._static_inputs[branch])
        else:
            raise RuntimeError(f"No input available for branch '{branch}'")

    @property
    def branches(self) -> list:
        """List of registered branch names."""
        return list(self._branches.keys())

    @property
    def failed_branches(self) -> list:
        """List of branches that failed capture and use eager fallback."""
        return sorted(self._failed_branches)
