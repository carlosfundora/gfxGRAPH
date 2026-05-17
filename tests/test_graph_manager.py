from unittest.mock import MagicMock, patch

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


def test_hot_replay_mode_bypasses_validation():
    original_hot_mode = graph_manager._HOT_REPLAY_MODE
    graph_manager._HOT_REPLAY_MODE = True
    try:
        g = BridgedCUDAGraph()
        g._validation_enabled_cached = True
        g._model_fn = MagicMock()
        graph_output = object()
        out = g._maybe_validate(graph_output, MagicMock())
        assert out is graph_output
        g._model_fn.assert_not_called()
    finally:
        graph_manager._HOT_REPLAY_MODE = original_hot_mode


def test_trusted_replay_promotes_and_skips_non_sampled_diagnostics():
    original_hot_mode = graph_manager._HOT_REPLAY_MODE
    original_threshold = graph_manager._TRUSTED_REPLAY_THRESHOLD
    original_interval = graph_manager._TRUSTED_REPLAY_SAMPLE_INTERVAL
    graph_manager._HOT_REPLAY_MODE = False
    graph_manager._TRUSTED_REPLAY_THRESHOLD = 2
    graph_manager._TRUSTED_REPLAY_SAMPLE_INTERVAL = 3
    try:
        g = BridgedCUDAGraph()
        g._graph = MagicMock()
        g._static_output = object()
        g._record_replay_sample = MagicMock()
        g._maybe_validate = MagicMock(side_effect=lambda out, _: out)

        out1 = g.replay()
        out2 = g.replay()
        out3 = g.replay()

        assert out1 is g._static_output
        assert out2 is g._static_output
        assert out3 is g._static_output
        assert g._trusted_replay_active is True
        # replay #3 in trusted mode is not a sampled replay with interval=3
        assert g._record_replay_sample.call_count == 2
        assert g._maybe_validate.call_count == 2
    finally:
        graph_manager._HOT_REPLAY_MODE = original_hot_mode
        graph_manager._TRUSTED_REPLAY_THRESHOLD = original_threshold
        graph_manager._TRUSTED_REPLAY_SAMPLE_INTERVAL = original_interval


def test_trusted_replay_keeps_fallback_on_replay_error():
    original_hot_mode = graph_manager._HOT_REPLAY_MODE
    graph_manager._HOT_REPLAY_MODE = False
    try:
        g = BridgedCUDAGraph()
        mock_graph = MagicMock()
        mock_graph.replay.side_effect = RuntimeError("boom")
        g._graph = mock_graph
        g._model_fn = MagicMock(return_value="eager")
        out = g.replay()
        assert out == "eager"
        assert g._eager_fallback is True
        g._model_fn.assert_called_once()
    finally:
        graph_manager._HOT_REPLAY_MODE = original_hot_mode
