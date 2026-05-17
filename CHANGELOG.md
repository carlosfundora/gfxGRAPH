# Changelog

All notable changes to this project are documented in this file.

## [0.3.1] - 2026-05-17

### Added
- Introduced a root `SECURITY.md` with vulnerability reporting and supported version policy.
- Added benchmark provenance fields for public benchmark output: commit SHA, ROCm runtime/driver hints, tracked environment variables, and repeated-run metadata.
- Added PyPI-facing metadata in `pyproject.toml` (`project.urls` and standard classifiers).

### Changed
- Aligned CI workflows to Python 3.12 to match the package runtime requirement.
- Switched GPU-unavailable integration tests to explicit `pytest.skip(...)` signaling instead of print-and-return behavior.

### Removed
- Closed the stale PR #16 tracking branch (`rusty-stats-refactor-9436386981500045669`) to reduce release confusion.
