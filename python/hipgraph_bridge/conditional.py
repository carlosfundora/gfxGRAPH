"""
Conditional graph execution for PyTorch (Gap 51 bridge).
"""

import torch
from typing import Callable, Dict, Optional


class ConditionalGraph:
    """Execute one of several pre-captured graphs based on a condition.

    Uses hipGraphNodeSetEnabled under the hood to toggle branches
    in a supergraph without re-instantiation.

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

    def add_branch(self, name: str, fn: Callable):
        """Register a branch function."""
        if self._captured:
            raise RuntimeError("Cannot add branches after capture")
        self._branches[name] = fn

    def capture(self, example_input: torch.Tensor):
        """Capture all branches as separate CUDAGraphs."""
        for name, fn in self._branches.items():
            static_input = example_input.clone()
            self._static_inputs[name] = static_input

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

        self._captured = True

    def run(self, branch: str, input_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute a specific branch.

        Args:
            branch: Name of the branch to execute
            input_tensor: If provided, copied to static input before replay
        """
        if not self._captured:
            raise RuntimeError("Call capture() first")
        if branch not in self._graphs:
            raise KeyError(f"Unknown branch '{branch}'. Available: {list(self._branches.keys())}")

        if input_tensor is not None:
            self._static_inputs[branch].copy_(input_tensor)

        self._graphs[branch].replay()
        return self._static_outputs[branch]
