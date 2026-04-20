# gfxGRAPH Python Package

Python integration layer for the gfxGRAPH CUDA→HIP graph bridge.

## Install

```bash
pip install /path/to/gfxGRAPH
pip install /path/to/gfxGRAPH/native   # optional native companion

# Transitional compatibility path
pip install /path/to/gfxGRAPH/python
```

`pip install .[native]` is not the supported source-install contract in this phase.
Tier-2 remains a two-step install so the repo-root package stays pure Python.

## Requirements

- PyTorch 2.9+ (ROCm build)
- ROCm 7.2.0+
- AMD GPU (gfx1030 target)
