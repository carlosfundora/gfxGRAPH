"""Tests for gfxGRAPH GUARD (3-tier illegal-memory-access safety).

CPU-runnable: layout validation, capture-safe conversion, fault localization, and
red-zone canary logic all work on CPU tensors (contiguity/stride/clone semantics are
device-independent). GPU-only behavior (real illegal-access interception) is covered
by the live server tests in the sglang fork, not here.
"""

import os

import pytest
import torch

from hipgraph_bridge import guard


# ---- guard_level / env parsing ----

@pytest.mark.parametrize(
    "val,expected",
    [
        ("", 0), ("off", 0), ("0", 0), ("none", 0),
        ("1", 1), ("tier1", 1), ("safe", 1),
        ("2", 2), ("tier2", 2), ("localize", 2),
        ("3", 3), ("tier3", 3), ("deep", 3),
        ("garbage", 0),
    ],
)
def test_guard_level(monkeypatch, val, expected):
    monkeypatch.setenv("GFXGRAPH_GUARD", val)
    assert guard.guard_level() == expected


# ---- Tier 1: validate_layout / make_safe ----

def test_validate_layout_contiguous_ok():
    t = torch.zeros(4, 8)
    assert guard.validate_layout(t, "x") == []


def test_validate_layout_flags_noncontiguous():
    t = torch.zeros(4, 8).t()  # transpose -> non-contiguous
    issues = guard.validate_layout(t, "x")
    assert any("non-contiguous" in i for i in issues)


def test_validate_layout_flags_broadcast():
    t = torch.zeros(1, 8).expand(4, 8)  # 0-stride broadcast
    issues = guard.validate_layout(t, "x")
    assert any("broadcast" in i or "0-stride" in i for i in issues)


def test_make_safe_passes_contiguous_through():
    t = torch.arange(12).reshape(3, 4)
    out, converted = guard.make_safe(t)
    assert not converted
    assert out is t  # no copy when already safe


def test_make_safe_fixes_noncontiguous():
    t = torch.arange(12).reshape(3, 4).t()  # non-contiguous
    out, converted = guard.make_safe(t)
    assert converted
    assert out.is_contiguous()
    assert torch.equal(out, t)  # same values, safe layout


def test_make_safe_fixes_broadcast_clone():
    base = torch.arange(8).reshape(1, 8)
    t = base.expand(4, 8)  # 0-stride
    out, converted = guard.make_safe(t)
    assert converted
    assert out.is_contiguous()
    assert all(s != 0 for s in out.stride())
    assert torch.equal(out, t)


def test_make_capture_safe_nested():
    a = torch.zeros(4, 4).t()           # unsafe
    b = torch.zeros(4, 4)               # safe
    obj = {"q": a, "kv": [b, a], "scale": 0.5}
    out, n = guard.make_capture_safe(obj)
    assert n == 2  # a appears twice, both converted
    assert out["q"].is_contiguous()
    assert out["kv"][0].is_contiguous()
    assert out["scale"] == 0.5


def test_make_safe_non_tensor_passthrough():
    out, converted = guard.make_safe(5)
    assert out == 5 and not converted


# ---- Tier 2: fault localization ----

def test_is_illegal_access():
    assert guard.is_illegal_access(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert guard.is_illegal_access(RuntimeError("hipErrorIllegalAddress"))
    assert guard.is_illegal_access(RuntimeError("misaligned address"))
    assert not guard.is_illegal_access(RuntimeError("out of memory"))


def test_localize_fault_carries_context():
    q = torch.zeros(2, 64)
    orig = RuntimeError("CUDA error: an illegal memory access was encountered")
    fault = guard.localize_fault(orig, op="decode", tensors=[("q", q)])
    assert isinstance(fault, guard.GfxGraphFault)
    assert fault.op == "decode"
    assert fault.original is orig
    s = str(fault)
    assert "illegal memory access" in s
    assert "q: shape=(2, 64)" in s
    assert "logic bug" in s.lower()


# ---- Tier 3: red-zone canary ----

def test_redzone_intact_when_in_bounds():
    rz = guard.RedZone((4, 8), torch.float32, "cpu", pad_elems=16)
    rz.tensor.fill_(1.0)  # write only within the usable region
    assert rz.check_intact()


def test_redzone_detects_overflow():
    rz = guard.RedZone((4, 8), torch.float32, "cpu", pad_elems=16)
    # Simulate an OOB write past the end of the usable region into the red zone.
    rz._flat[rz.pad + rz._n + 2] = 0.0  # clobber a sentinel
    assert not rz.check_intact()


def test_redzone_reset():
    rz = guard.RedZone((2, 2), torch.int32, "cpu", pad_elems=8)
    rz._flat[0] = 0
    assert not rz.check_intact()
    rz.reset()
    assert rz.check_intact()


# ---- deep guard env ----

def test_deep_guard_env_keys():
    env = guard.deep_guard_env()
    assert env.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"


def test_apply_deep_guard_env_no_clobber(monkeypatch):
    monkeypatch.setenv("PYTORCH_NO_CUDA_MEMORY_CACHING", "operator-value")
    guard.apply_deep_guard_env()
    assert os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "operator-value"


def test_compute_sanitizer_cmd_none_when_absent(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _t: None)
    assert guard.compute_sanitizer_cmd(["python", "x.py"]) is None


def test_public_api_exports():
    import gfxgraph
    for name in ("guard_level", "make_safe", "make_capture_safe", "GfxGraphFault",
                 "validate_layout", "localize_fault", "RedZone", "is_illegal_access"):
        assert hasattr(gfxgraph, name), name
