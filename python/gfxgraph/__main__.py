"""
CLI entry point: python -m gfxgraph [script.py] [args...]

Enables gfxGRAPH transparently, then runs the user's script.
"""

import os
import sys
import runpy


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m gfxgraph <script.py> [args...]")
        print()
        print("Enables gfxGRAPH CUDA Graph parity for gfx1030/RDNA2,")
        print("then runs your script with torch patched transparently.")
        print()
        print("Options (via env vars):")
        print("  GFXGRAPH=debug     Enable debug logging")
        print("  GFXGRAPH=validate  Enable graph-vs-eager validation")
        sys.exit(0)

    # Enable gfxGRAPH
    os.environ.setdefault("GFXGRAPH", "1")
    import gfxgraph  # noqa: F401 — triggers auto-enable via env var

    # Run user script
    script = sys.argv[1]
    sys.argv = sys.argv[1:]  # shift argv so script sees correct args

    # Use runpy to execute the script in __main__ namespace
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
