# gfxgraph base package — pure-Python install (Tier 1).
#
# Native acceleration is intentionally NOT built here, so
# `pip install /path/to/gfxGRAPH` is a true pure-Python install:
#   - The HIP bridge (libhipgraph_bridge.so) ships via the `native/`
#     companion package (`gfxgraph-native`, scikit-build/CMake).
#   - The Rust contract crates (rs_gfxgraph, rs_gfxgraph_stats) build from
#     source under `rust/` for users who want them.
# The Python runtime imports both optionally and falls back to pure-Python
# implementations when absent (see the _HAS_* guards in
# hipgraph_bridge/{graph_manager,conditional,shape_bucketing}.py and
# gfxgraph/_enable.py). Tier 2 native acceleration:
#   pip install /path/to/gfxGRAPH/native
#
# All project metadata (packages, scripts, dependencies) lives in
# pyproject.toml; this file exists only for tooling that expects a setup.py.
from setuptools import setup

setup()
