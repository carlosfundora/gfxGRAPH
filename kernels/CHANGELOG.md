# Changelog

## 2026-05-19

### Rebuild audit: ATOM / gfxATOM-Rust integration check

- Audited kernel roots used by the current WIP stack:
  - `/home/local/ai/build/kernels`
  - `/mnt/ai/build/kernels`
- Verified `/home` and `/mnt` `rdna2/` kernel files are hash-identical.
- Compared WIP SGLang RDNA2 kernel sources against `build/kernels/rdna2`; observed only local formatting/docstring differences in `dispatch.py`, `fused_qknorm_rope.py`, and `__init__.py`.
- Determined no kernel rebuild action is required from this audit pass.
