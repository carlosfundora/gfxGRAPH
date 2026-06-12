# Changelog

## [Unreleased]
- Added geometry/shape/layout/wave primitive coverage for engine-safe graph contracts, including `Dim`, `Rank`, `BatchShape`, `Contiguity`, tiled/page layout specs, and `OccupancyHint`.
- Added `convert` contracts for shape/layout conversion plans, page transforms, stride transforms, and validation-only dtype conversion.
- Added `registry` contracts with a baseline graph capability registry for geometry, shape, layout, wave, conversion, runner, validator, adapter, and signal capabilities.
- Added robust, unified error handling definitions (`GfxGraphError`) and error reporting (`report_error`).
- Implemented conditional compilation integration (via optional `"logly"` feature) with `rs_logly_logger`.
- Provided a zero-cost system-level stderr fallback for non-logly environments.
- Updated documentation and README with comprehensive architecture and usage guidelines.
- Added the canonical `rs_gfxgraph_core` crate with schema, stats, and adapter modules.
