import sys
from unittest.mock import MagicMock, patch

# Mock torch before importing BridgedCUDAGraph
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["torch.cuda"] = mock_torch.cuda

# Mock hipgraph_bridge.shape_bucketing
sys.modules["hipgraph_bridge.shape_bucketing"] = MagicMock()

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
