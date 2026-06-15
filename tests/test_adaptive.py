# gfxGRAPH — tests for adaptive hardware/ROCm-torch detection, collision-safe wave, interop, CLI
# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora
# SPDX-License-Identifier: MIT
"""Pure-Python tests (no GPU required) for the adaptive/cross-engine surface."""

import os
import subprocess
import sys

from hipgraph_bridge import hardware as hw
from hipgraph_bridge import wavefront as wf
from hipgraph_bridge import interop


def test_torch_rocm_status_shape():
    st = hw.torch_rocm_status()
    assert set(st) >= {"installed", "is_rocm", "torch_version", "hip_version", "message"}
    assert isinstance(st["message"], str) and st["message"]


def test_device_info_optional():
    d = hw.device_info()
    if d is not None:
        assert d.wavefront in (32, 64)
        assert d.arch and isinstance(d.summary(), str)


def test_arch_override(monkeypatch):
    monkeypatch.setenv("GFXGRAPH_ARCH", "gfx942")  # pretend CDNA card
    d = hw.detect_device(refresh=True)
    assert d is not None and d.arch == "gfx942" and d.wavefront == 64 and d.overridden
    monkeypatch.delenv("GFXGRAPH_ARCH", raising=False)
    hw.detect_device(refresh=True)  # reset cache


def test_wave_mode_and_collision_gate(monkeypatch):
    monkeypatch.setenv("GFXGRAPH_WAVE", "off")
    assert wf.wave_mode() == 0
    assert wf.should_convert(32, 4, lanes=32, cu=20)[0] is False  # off
    monkeypatch.setenv("GFXGRAPH_WAVE", "auto")
    assert wf.wave_mode() == 2
    # collision avoidance: user already gangs wavefronts → skip
    assert wf.should_convert(64, 4, lanes=32, cu=20)[0] is False
    # grid already saturates the GPU → skip
    assert wf.should_convert(32, 20 * 32, lanes=32, cu=20)[0] is False
    # genuinely underfilled single-wavefront launch → convert
    assert wf.should_convert(32, 4, lanes=32, cu=20)[0] is True
    # explicit opt-out → skip
    monkeypatch.setenv("GFXGRAPH_NO_WAVE", "1")
    assert wf.should_convert(32, 4, lanes=32, cu=20)[0] is False


def test_plan_uses_given_specs():
    p = wf.plan_software_wave(4, 64, lanes=32, cu=20)
    assert p["W"] == 2 and p["block"] == 64
    p2 = wf.plan_software_wave(4, 128, lanes=64, cu=20)  # CDNA: wave64 hardware
    assert p2["lanes"] == 64


def test_detect_accelerators_and_engines():
    acc, eng = hw.detect_accelerators(), hw.detect_engines()
    assert isinstance(acc, dict) and "migraphx" in acc and "torch_cuda_graph" in acc
    assert isinstance(eng, dict) and "vllm" in eng and "llama_cpp_bin" in eng


def test_migraphx_backend_is_honest_stub():
    import pytest
    assert isinstance(interop.migraphx_available(), dict)
    with pytest.raises(NotImplementedError):
        interop.MIGraphXBackend().compile()
    assert "llama.cpp" in interop.cross_engine_support()


def test_cli_explain_subprocess():
    env = {**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "python")}
    r = subprocess.run([sys.executable, "-m", "gfxgraph", "explain", "hipErrorOutOfMemory: out of memory"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "out_of_memory" in r.stdout
