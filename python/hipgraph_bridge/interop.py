# gfxGRAPH — interop: cross-engine support (MIGraphX backend + hipGraph interposer) — EXPERIMENTAL
# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora
# SPDX-License-Identifier: MIT
"""gfxGRAPH cross-engine interop — detection now, deep routing on the roadmap (honest scaffold).

Two extension paths beyond the torch CUDA-graph bridge. **Detection works today; the deep routing
is documented + stubbed, NOT a finished/verified execution backend** — it raises a clear
`NotImplementedError` rather than pretend.

  1) MIGraphX backend — AMD's graph compile/execute engine (consumes ONNX / MIGraphX IR, reachable
     via ONNX-Runtime's MIGraphX EP). A *potential* alternative to graph-capture for whole-model
     inference. `migraphx_available()` detects it; `MIGraphXBackend` is the roadmap entry point.

  2) hipGraph interposer — the framework-AGNOSTIC route. gfxGRAPH already operates on the **HIP
     runtime graph API** (`hipGraph*`), and native engines (**llama.cpp** `GGML_CUDA_GRAPHS`,
     **candle**) call that API directly. An `LD_PRELOAD` interposer over the real `hipGraph*` /
     `hipStreamBeginCapture` symbols would apply GUARD + the bridge to their *native* graphs — no
     per-engine reimplementation. (NB: hipGraph here = the HIP runtime graph API, the CUDA-Graphs
     equivalent — NOT the ROCm-DS `hipGRAPH` graph-analytics library, which is unrelated.)

Universally available regardless of engine/language: the **diagnostics CLI** (`gfxgraph explain`)
— pipe any engine's stderr through it.
"""

from __future__ import annotations

import shutil


def migraphx_available() -> dict:
    """Detect MIGraphX (python module / driver / C lib). Returns {available, python, driver, note}."""
    from .hardware import _has
    py = _has("migraphx")
    drv = bool(shutil.which("migraphx-driver"))
    return {"available": py or drv, "python": py, "driver": drv,
            "note": "MIGraphX present — usable as an ONNX/IR graph-compile backend (roadmap "
                    "routing in gfxGRAPH; for now, AMD's ONNX-Runtime MIGraphX EP is the direct path)."
                    if (py or drv) else "MIGraphX not found."}


class MIGraphXBackend:
    """EXPERIMENTAL roadmap entry point for routing graphs through MIGraphX. Not yet implemented —
    MIGraphX consumes ONNX/MIGraphX-IR, not torch CUDA-graph capture, so this is a substantial
    adapter (export → migraphx.parse → compile → run). Use AMD's ONNX-Runtime MIGraphX EP today."""

    def __init__(self):
        self.status = migraphx_available()

    def compile(self, *_args, **_kwargs):
        raise NotImplementedError(
            "gfxGRAPH MIGraphXBackend is a roadmap stub (v0.5.0). MIGraphX routing requires an "
            "ONNX/IR adapter; track it on the repo, or use onnxruntime with the 'MIGraphXExecutionProvider'."
        )

    run = compile


def hipgraph_interposer_status() -> dict:
    """Status of the framework-agnostic hipGraph LD_PRELOAD interposer (the native-engine route).
    Returns {built, lib, note}. The interposer .so is not shipped in the pure-Python wheel."""
    from . import _C
    lib = getattr(_C, "library_path", None)
    built = False
    try:
        from ._C import library_path as _lp  # type: ignore
        built = _lp() is not None
    except Exception:
        built = getattr(_C, "lib", None) is not None
    return {"built": built, "lib": str(lib) if lib else None,
            "note": "hipGraph interposer = the LD_PRELOAD route to apply GUARD/bridge to native "
                    "engines (llama.cpp GGML_CUDA_GRAPHS, candle) that call hipGraph* directly. "
                    "Roadmap; today native engines get the diagnostics via `gfxgraph explain`."}


def cross_engine_support() -> dict:
    """Map each engine to its current gfxGRAPH support level (honest)."""
    from .hardware import detect_engines
    eng = detect_engines()
    return {
        "torch_engines (vllm, sglang, TGI)": "FULL bridge + GUARD + diagnostics (torch.cuda.CUDAGraph patch)",
        "llama.cpp": ("diagnostics via `gfxgraph explain` NOW; GUARD/bridge via the hipGraph "
                      "interposer = ROADMAP" + ("  [detected]" if eng.get("llama_cpp_bin") or eng.get("llama_cpp_python") else "")),
        "candle": "diagnostics via `gfxgraph explain` NOW; native hipGraph hooks = ROADMAP",
        "any engine": "diagnostics CLI works universally — pipe stderr through `gfxgraph explain`",
    }
