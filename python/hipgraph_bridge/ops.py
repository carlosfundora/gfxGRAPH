"""
Custom torch.library op registration for gfxGRAPH.

Registers bridge operations with the PyTorch dispatcher so they
are compatible with torch.compile.
"""

import torch

# Namespace for all gfxGRAPH ops
LIBRARY = torch.library.Library("gfxgraph", "DEF")

# Define conditional_select: pick a tensor based on a boolean predicate
LIBRARY.define("conditional_select(Tensor pred, Tensor a, Tensor b) -> Tensor")


@torch.library.impl(LIBRARY, "conditional_select", "CUDA")
def conditional_select_cuda(pred, a, b):
    """Select tensor a or b based on pred (scalar bool tensor)."""
    return torch.where(pred.bool(), a, b)


@torch.library.impl(LIBRARY, "conditional_select", "Meta")
def conditional_select_meta(pred, a, b):
    """Shape inference for torch.compile tracing."""
    return torch.empty_like(a)
