import sys
from unittest.mock import MagicMock, patch

# Mock torch before importing BridgedCUDAGraph
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["torch.cuda"] = mock_torch.cuda

# Mock hipgraph_bridge.shape_bucketing
sys.modules["hipgraph_bridge.shape_bucketing"] = MagicMock()

import hipgraph_bridge.graph_manager as graph_manager
from hipgraph_bridge.graph_manager import BridgedCUDAGraph

def test_debug_dump_no_graph():
    """Test debug_dump when no graph is captured."""
    g = BridgedCUDAGraph()
    # Should not raise, should log to debug
    with patch("hipgraph_bridge.graph_manager._log") as mock_log:
        g.debug_dump("test_path.dot")
        mock_log.debug.assert_called_with("debug_dump: no captured graph to dump")

def test_debug_dump_with_graph_no_method():
    """Test debug_dump when graph exists but has no debug_dump method."""
    g = BridgedCUDAGraph()
    g._graph = MagicMock(spec=[]) # No debug_dump attribute

    with patch("hipgraph_bridge.graph_manager._log") as mock_log:
        g.debug_dump("test_path.dot")
        mock_log.debug.assert_called_with("debug_dump: no captured graph to dump")

def test_debug_dump_success():
    """Test debug_dump successfully delegates to the underlying graph."""
    g = BridgedCUDAGraph()
    mock_inner_graph = MagicMock()
    # Add debug_dump to the mock
    mock_inner_graph.debug_dump = MagicMock()
    g._graph = mock_inner_graph

    path = "/tmp/graph.dot"
    g.debug_dump(path)
    mock_inner_graph.debug_dump.assert_called_once_with(path)

def test_enable_debug_mode_no_graph():
    """Test enable_debug_mode when no graph is captured."""
    g = BridgedCUDAGraph()
    # Should be a no-op
    g.enable_debug_mode()

def test_enable_debug_mode_success():
    """Test enable_debug_mode successfully delegates to the underlying graph."""
    g = BridgedCUDAGraph()
    mock_inner_graph = MagicMock()
    mock_inner_graph.enable_debug_mode = MagicMock()
    g._graph = mock_inner_graph

    g.enable_debug_mode()
    mock_inner_graph.enable_debug_mode.assert_called_once()


def test_adaptive_signature_varies_by_shape():
    g = BridgedCUDAGraph()
    g._model_fn = lambda x: x

    t1 = MagicMock()
    t1.shape = (1, 1024)
    t1.dtype = "torch.float16"
    t1.device = "cuda:0"
    t1.stride.return_value = (1024, 1)

    t2 = MagicMock()
    t2.shape = (32, 1024)
    t2.dtype = "torch.float16"
    t2.device = "cuda:0"
    t2.stride.return_value = (1024, 1)

    s1 = g._build_adaptive_signature(None, t1)
    s2 = g._build_adaptive_signature(None, t2)
    assert s1 != s2


def test_cached_adaptive_decision_sets_preferred_eager():
    g = BridgedCUDAGraph()
    g._model_fn = lambda x: x

    t = MagicMock()
    t.shape = (8, 1024)
    t.dtype = "torch.float16"
    t.device = "cuda:0"
    t.stride.return_value = (1024, 1)

    sig = g._build_adaptive_signature(None, t)
    graph_manager._set_adaptive_signature_decision(sig, "eager")

    g._maybe_load_cached_decision(None, t)
    assert g._prefer_eager is True
    assert g._adaptive_disabled is True
