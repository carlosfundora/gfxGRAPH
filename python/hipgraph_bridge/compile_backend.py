"""
Custom torch.compile backend for gfxGRAPH.

Analyzes FX graphs for gap capabilities and applies bridge strategies
before delegating to the inductor backend.
"""

import torch
from torch._dynamo.backends.common import aot_autograd


def hipgraph_bridge_backend(gm, example_inputs):
    """Custom torch.compile backend that:

    1. Analyzes the FX graph for gap capabilities
    2. If all ops are natively supported → delegate to inductor
    3. If gap ops detected → wrap with bridge, compile remainder
    4. Returns optimized callable

    Usage:
        model = torch.compile(model, backend="hipgraph_bridge")
    """
    # For now, delegate entirely to inductor.
    # Future: analyze gm.graph for gap-related patterns and
    # inject bridge wrappers around those subgraphs.
    from torch._inductor.compile_fx import compile_fx
    return compile_fx(gm, example_inputs)


# Register as a named backend
torch._dynamo.register_backend(
    name="hipgraph_bridge",
    compiler_fn=hipgraph_bridge_backend,
)
