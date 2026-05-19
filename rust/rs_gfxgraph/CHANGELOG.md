# Changelog

All notable crate-specific changes for `rs_gfxgraph` are recorded here.

## [Unreleased]

### Added

- Initial crate scaffold: `BucketRouter` for shape-bucketed graph selection with binary-search dispatch and warmup/failed-bucket state tracking, `ConditionalGraphRunner` for branch-conditional CUDA graph execution with fallback callbacks and shared input tensor support. Exposed as a PyO3 extension module (`cdylib`).
