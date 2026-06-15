# gfxGRAPH — CLI: universal, framework-agnostic ROCm diagnostics + env doctor + script runner
# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora
# SPDX-License-Identifier: MIT
"""gfxGRAPH command-line interface.

Subcommands (the diagnostics ones are torch-free, so they work for ANY engine — llama.cpp,
candle, vLLM, sglang — by piping that engine's error/log through gfxGRAPH):

  gfxgraph explain ["<error text>"]   Explain a HIP/ROCm error (arg or piped stdin) with
                                      cause + gfx-arch context + fix. Honors GFXGRAPH_LANG=zh.
                                      e.g.  llama-cli ... 2>&1 | gfxgraph explain
  gfxgraph doctor | env               Print the environment report (GPU, ROCm-PyTorch,
                                      accelerators, engines, config).
  gfxgraph device                     Print the detected/overridden GPU summary.
  gfxgraph run <script.py> [args...]  Run a script with the CUDA→HIP graph bridge enabled.
  python -m gfxgraph <script.py> ...  Back-compat alias for `run`.
"""

import os
import sys


def _print_help():
    print(__doc__.strip())
    print("\nEnv: GFXGRAPH=1|debug|validate · GFXGRAPH_GUARD=1|2|3 · GFXGRAPH_DIAG=0 · "
          "GFXGRAPH_LANG=zh · GFXGRAPH_WAVE=off|detect|auto · GFXGRAPH_ARCH=gfxNNNN")


def _explain(args) -> int:
    text = " ".join(args).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read()
    if not text:
        print("usage: gfxgraph explain \"<error text>\"   (or pipe a log via stdin)", file=sys.stderr)
        return 2
    from hipgraph_bridge.diagnostics import explain
    d = explain(text)
    if d is None:
        print("gfxGRAPH: no known diagnosis matched this text.", file=sys.stderr)
        return 1
    print(d.format())
    return 0


def _run(args) -> int:
    if not args:
        print("usage: gfxgraph run <script.py> [args...]", file=sys.stderr)
        return 2
    import runpy
    os.environ.setdefault("GFXGRAPH", "1")
    import gfxgraph  # noqa: F401 — triggers auto-enable + diagnostics
    script = args[0]
    sys.argv = list(args)
    runpy.run_path(script, run_name="__main__")
    return 0


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "explain":
        sys.exit(_explain(rest))
    elif cmd in ("doctor", "env", "status"):
        from hipgraph_bridge.diagnostics import environment_report
        print(environment_report())
    elif cmd == "device":
        from hipgraph_bridge.hardware import device_info
        d = device_info()
        print(d.summary() if d else "(no GPU detected)")
    elif cmd == "run":
        sys.exit(_run(rest))
    elif cmd.endswith(".py"):
        sys.exit(_run(argv))  # back-compat: python -m gfxgraph script.py
    else:
        print(f"gfxgraph: unknown command '{cmd}'\n", file=sys.stderr)
        _print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
