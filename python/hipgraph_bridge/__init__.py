"""
gfxGRAPH — CUDA Graph → HIP Graph translation layer for gfx1030 RDNA2

Provides drop-in bridge for all 4 CUDA Graph parity gaps on AMD ROCm.
"""

__version__ = "0.3.1"

from hipgraph_bridge._C import lib as _lib
from hipgraph_bridge.graph_manager import BridgedCUDAGraph
from hipgraph_bridge.shape_bucketing import ShapeBucketPool

__all__ = ["BridgedCUDAGraph", "ShapeBucketPool"]


def init():
    """Initialize the native bridge library."""
    _lib.hgb_init()


def shutdown():
    """Release all native bridge resources."""
    _lib.hgb_shutdown()
