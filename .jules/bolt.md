## 2024-05-17 - Optimize VRAM info polling to reduce GPU synchronization
**Learning:** PyTorch's `torch.cuda.mem_get_info()` causes GPU synchronization which can introduce performance bottlenecks when repeatedly called in quick succession, such as during the warmup phase of CUDA graphs in a loop over bucket sizes.
**Action:** When performing bulk operations or checks that require VRAM polling, cache the result of `torch.cuda.mem_get_info()` and reuse it across closely-spaced queries. Alternatively, query the info periodically (e.g., every 5 items in a loop) instead of on every iteration to minimize synchronization overhead while still respecting dynamic constraints like VRAM caps.

## 2024-05-18 - Cache PyO3 Rust extension objects to eliminate FFI instantiation overhead in hot paths
**Learning:** Instantiating PyO3 Rust extension objects (like `BridgedGraphValidator`) in Python on every forward pass crossing the FFI boundary introduces unnecessary serialization and overhead that can outweigh the speed benefits of the native extension itself, especially during high-frequency execution like graph replay.
**Action:** When a Python wrapper calls into a PyO3 Rust class repeatedly, the wrapper should instantiate the class once and cache it on the Python instance (`self._validator = _gfxgraph_rs.BridgedGraphValidator(...)`) so that subsequent calls only invoke methods on the pre-initialized object, strictly limiting FFI overhead.

## 2024-05-19 - Cache Python imports inside PyO3 Rust structs for FFI hot paths
**Learning:** PyO3 extensions that dynamically import Python modules or access Python attributes inside tight loops (like `py.import("torch")` or `torch.call_method0("no_grad")` during graph replay) incur high FFI serialization and Python import mechanism overhead.
**Action:** When creating PyO3 extensions wrapping Python callables, import and store `PyObject` references to the required Python modules and functions inside the Rust struct's constructor (`#[new]`). Then, invoke these pre-cached callables (`self.torch_no_grad.call0(py)?`) directly within the hot path methods.

## 2024-05-19 - Eliminate dynamic getattr checks and conditional aliasing in Python hot paths
**Learning:** Using `getattr(self, "attr", None)` with a fallback inside a hot path (e.g. `_maybe_validate`) takes roughly 2.5x longer than a direct attribute check against a pre-initialized class field (`if self.attr is None:`). Similarly, wrapping hot-path functions (like telemetry) in a Python condition (`if _HAS_RUST_STATS: rust_fn() else: py_fn()`) adds overhead.
**Action:** Pre-initialize all expected instance attributes to `None` inside `__init__`. For telemetry functions wrapping native extensions, directly alias the Rust functions to the Python module-level namespace at import-time to completely bypass Python execution frames.
