## 2024-05-14 - PyO3 FFI Overhead in Hot Paths

**Learning:** Crossing the FFI boundary for granular operations like importing a module (`py.import("time")`) and calling python methods (`time_mod.call_method0("perf_counter")`) adds significant overhead inside hot paths (like graph replay). In our case it added ~1.5us to a fast path that should be entirely native.
**Action:** Replace `py.import("time")` and `perf_counter` with native Rust `std::time::Instant::now()` to measure time inside PyO3 extensions without re-entering the Python interpreter.
