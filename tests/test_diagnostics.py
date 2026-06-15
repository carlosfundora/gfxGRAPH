# gfxGRAPH — tests for diagnostics + wavefront (bilingual error reporting, software-wave plan)
# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora
# SPDX-License-Identifier: MIT
"""Pure-Python tests (no GPU/torch required) for the diagnostics + wavefront modules."""

from hipgraph_bridge import diagnostics as d
from hipgraph_bridge import wavefront as w


def test_explain_matches_all_codes():
    cases = {
        "No available kernel. Aborting execution.": "no_kernel_image",
        "hipErrorOutOfMemory: out of memory": "out_of_memory",
        "CUDA error: an illegal memory access was encountered": "illegal_address",
        "missing fdot2.bf16.bf16": "bf16_unsupported",
        "please set HSA_OVERRIDE_GFX_VERSION": "wrong_arch",
        "warning: -mwavefrontsize64 ignored": "wave64_ignored",
        "No module named 'aiter.ops.flydsl.moe_common'": "aiter_on_rdna",
        "hipErrorInvalidConfiguration": "invalid_configuration",
    }
    for text, code in cases.items():
        dg = d.explain(text)
        assert dg is not None and dg.code == code, (text, dg)


def test_explain_none_on_unknown():
    assert d.explain("totally unrelated message") is None
    assert d.explain(None) is None


def test_explain_accepts_exception():
    dg = d.explain(RuntimeError("CUDA out of memory. Tried to allocate ..."))
    assert dg is not None and dg.code == "out_of_memory"


def test_bilingual_format():
    dg = d.explain("No available kernel")
    en, zh = dg.format("en"), dg.format("zh")
    assert "gfxGRAPH diagnosis" in en and "Cause:" in en
    assert "gfxGRAPH 诊断" in zh and "原因" in zh
    # every table code has a zh translation (now in the separate diag_zh pack)
    from hipgraph_bridge import diag_zh
    for code in (c for c, *_ in d._TABLE):
        assert code in diag_zh.MESSAGES, f"missing zh translation for {code}"


def test_lang_resolver(monkeypatch):
    monkeypatch.setenv("GFXGRAPH_LANG", "zh-CN")
    assert d.lang() == "zh"
    monkeypatch.setenv("GFXGRAPH_LANG", "en")
    assert d.lang() == "en"


def test_diagnose_context_manager_reraises():
    import pytest
    with pytest.raises(RuntimeError):
        with d.diagnose("unit"):
            raise RuntimeError("illegal memory access")


def test_wavefront_detect_and_plan():
    assert w.detect_wave64("-mwavefrontsize64").code == "wave64_ignored"
    assert w.detect_wave64("warpSize == 64").code == "wave64_ignored"
    assert w.detect_wave64("nothing here") is None
    # underfilled grid -> gang warps (wave64/wave128); saturated -> plain Wave32
    assert w.plan_software_wave(4, 64)["W"] == 2
    assert w.plan_software_wave(4, 128)["W"] == 4
    assert w.plan_software_wave(8192, 64)["W"] == 1
    # short reduction caps W so each warp keeps >= min_rows_per_lane
    assert w.plan_software_wave(4, 128, max_kv_len=16)["W"] <= 2


def test_environment_report_runs():
    rep = d.environment_report()
    assert "gfxGRAPH environment report" in rep and "HSA_OVERRIDE_GFX_VERSION" in rep
