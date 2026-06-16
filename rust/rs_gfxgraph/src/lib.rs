use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyKeyError, PyRuntimeError, PyTypeError};
use pyo3::types::PyDict;
use std::collections::HashSet;
use std::sync::RwLock;
use std::sync::OnceLock;

/// Stores the result of system-level environment and core-affinity initializations.
static INIT_RESULT: OnceLock<Result<(Option<String>, Option<Vec<usize>>), String>> = OnceLock::new();

/// Helper to parse standard Linux sysfs CPU lists (e.g. "0-2,12-14").
#[cfg(target_os = "linux")]
fn parse_cpu_list(list_str: &str) -> Vec<usize> {
    let mut cores = Vec::new();
    for part in list_str.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let mut range_parts = part.split('-');
            if let (Some(start_str), Some(end_str)) = (range_parts.next(), range_parts.next()) {
                if let (Ok(start), Ok(end)) = (start_str.parse::<usize>(), end_str.parse::<usize>()) {
                    for core in start..=end {
                        cores.push(core);
                    }
                }
            }
        } else if let Ok(core) = part.parse::<usize>() {
            cores.push(core);
        }
    }
    cores
}

/// Dynamic GPU detection and thread-affinity alignment optimized for AMD systems.
///
/// 1. Detects if an RDNA2 card (gfx1030/1031) is installed on Linux. If so, and
///    HSA_OVERRIDE_GFX_VERSION is unset, it programmatically sets it to "10.3.0".
/// 2. Resolves CPU Core Complex (CCX) cache-sharing topology via sysfs and pins
///    the calling thread to local CPU cores to minimize Infinity Fabric latency.
#[cfg(target_os = "linux")]
fn init_environment_and_affinity() -> Result<(Option<String>, Option<Vec<usize>>), String> {
    // 1. Programmatic HSA Override for gfx1030/1031 (Navi 21 / Navi 22)
    let mut hsa_overridden = None;
    if std::env::var("HSA_OVERRIDE_GFX_VERSION").is_err() {
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with("card") {
                    let dev_path = path.join("device/device");
                    let vend_path = path.join("device/vendor");
                    if let (Ok(dev_str), Ok(vend_str)) = (std::fs::read_to_string(dev_path), std::fs::read_to_string(vend_path)) {
                        let dev_id = dev_str.trim().to_lowercase();
                        let vend_id = vend_str.trim().to_lowercase();
                        // 0x1002 is the AMD PCI vendor ID
                        if vend_id.contains("1002") {
                            // Check for Navi 22 (0x73df - RX 6700 XT) or Navi 21 (0x73a5, 0x73a0, 0x73bf)
                            if dev_id.contains("73df") || dev_id.contains("73a5") || dev_id.contains("73a0") || dev_id.contains("73bf") {
                                std::env::set_var("HSA_OVERRIDE_GFX_VERSION", "10.3.0");
                                hsa_overridden = Some("10.3.0".to_string());
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Local L3/CCX Cache Pinning
    let mut pinned_cores = None;
    unsafe {
        let cpu = libc::sched_getcpu();
        if cpu >= 0 {
            let cache_list_path = format!("/sys/devices/system/cpu/cpu{}/cache/index3/shared_cpu_list", cpu);
            if let Ok(list_str) = std::fs::read_to_string(&cache_list_path) {
                let cores = parse_cpu_list(&list_str);
                if !cores.is_empty() {
                    let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
                    for &core in &cores {
                        if core < 1024 {
                            libc::CPU_SET(core, &mut cpuset);
                        }
                    }
                    let res = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
                    if res == 0 {
                        pinned_cores = Some(cores);
                    }
                }
            }
        }
    }

    Ok((hsa_overridden, pinned_cores))
}

#[cfg(not(target_os = "linux"))]
fn init_environment_and_affinity() -> Result<(Option<String>, Option<Vec<usize>>), String> {
    Err("HSA overrides and CPU pinning are only supported on Linux targets.".to_string())
}

/// Exposes the CPU core indices that the current process thread was pinned to.
#[pyfunction]
fn pinned_cpu_cores() -> PyResult<Option<Vec<usize>>> {
    match INIT_RESULT.get() {
        Some(Ok((_, Some(cores)))) => Ok(Some(cores.clone())),
        _ => Ok(None),
    }
}

/// Exposes the value of HSA_OVERRIDE_GFX_VERSION if programmatically set on boot.
#[pyfunction]
fn hsa_override_applied() -> PyResult<Option<String>> {
    match INIT_RESULT.get() {
        Some(Ok((Some(val), _))) => Ok(Some(val.clone())),
        _ => Ok(None),
    }
}

/// Returns a descriptive diagnostic message of the system initialization achievements.
#[pyfunction]
fn init_status_message() -> PyResult<String> {
    match INIT_RESULT.get() {
        Some(Ok((override_val, pinned))) => {
            let mut msg = String::new();
            if let Some(val) = override_val {
                msg.push_str(&format!("Applied HSA override value '{}' for RDNA2. ", val));
            } else {
                msg.push_str("No programmatic HSA overrides needed. ");
            }
            if let Some(cores) = pinned {
                msg.push_str(&format!("Successfully pinned thread to local L3 cache/CCX cores: {:?}", cores));
            } else {
                msg.push_str("Affinity core-pinning not executed or bypassed.");
            }
            Ok(msg)
        }
        Some(Err(e)) => Ok(format!("Initialization failed with warning: {}", e)),
        None => Ok("Initialization has not run yet.".to_string()),
    }
}

/// High-performance bucket selection and state tracker for dynamic-shape inputs.
///
/// Implemented natively in Rust to bypass Python Bisect and interpreter boundary crossings.
#[pyclass]
pub struct BucketRouter {
    buckets: Vec<usize>,
    warmed_up: HashSet<usize>,
    failed_buckets: HashSet<usize>,
}

#[pymethods]
impl BucketRouter {
    #[new]
    fn new(buckets: Vec<usize>) -> Self {
        BucketRouter {
            buckets,
            warmed_up: HashSet::new(),
            failed_buckets: HashSet::new(),
        }
    }

    /// Selects the smallest bucket size >= input_size and returns (bucket_size, state_code).
    ///
    /// State Codes:
    /// - 0: Ready (Graph is warmed up)
    /// - 1: NeedsWarmup (Graph needs capture/replay warmup)
    /// - 2: Failed (Graph capture or execution previously failed)
    fn route(&self, input_size: usize) -> PyResult<(usize, u8)> {
        let bucket = match self.buckets.binary_search(&input_size) {
            Ok(idx) => self.buckets[idx],
            Err(idx) => {
                if idx < self.buckets.len() {
                    self.buckets[idx]
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Input size {} exceeds largest bucket {}. Add a larger bucket.",
                        input_size,
                        self.buckets.last().unwrap_or(&0)
                    )));
                }
            }
        };

        let state = if self.warmed_up.contains(&bucket) {
            0
        } else if self.failed_buckets.contains(&bucket) {
            2
        } else {
            1
        };

        Ok((bucket, state))
    }

    /// Marks a bucket size as successfully warmed up.
    fn mark_warmed_up(&mut self, bucket_size: usize) {
        self.warmed_up.insert(bucket_size);
    }

    /// Marks a bucket size as failed (forcing eager execution routing).
    fn mark_failed(&mut self, bucket_size: usize) {
        self.failed_buckets.insert(bucket_size);
    }

    /// Returns the sorted list of warmed up bucket sizes.
    fn warmed_up_list(&self) -> Vec<usize> {
        let mut list: Vec<usize> = self.warmed_up.iter().copied().collect();
        list.sort_unstable();
        list
    }

    /// Returns the sorted list of failed bucket sizes.
    fn failed_list(&self) -> Vec<usize> {
        let mut list: Vec<usize> = self.failed_buckets.iter().copied().collect();
        list.sort_unstable();
        list
    }
}

/// Safe RAII guard to track currently replaying branches in ConditionalGraphRunner.
/// Prevents thread re-entrancy collision by removing the branch from the running registry when dropped.
struct ReentrancyGuard<'a> {
    runner: &'a ConditionalGraphRunner,
    branch: &'a str,
}

impl<'a> Drop for ReentrancyGuard<'a> {
    fn drop(&mut self) {
        if let Ok(mut lock) = self.runner.running_branches.write() {
            lock.remove(self.branch);
        }
    }
}

/// Native conditional graph execution manager.
///
/// Bypasses Python dictionary mapping checks and provides thread-safe validation,
/// eager callbacks, performance logging, and atomic thread re-entrancy collision protection.
#[pyclass]
pub struct ConditionalGraphRunner {
    branches: Vec<String>,
    graphs: PyObject, // dict branch_name -> CUDAGraph
    static_outputs: PyObject, // dict branch_name -> static output tensor
    failed_branches: RwLock<HashSet<String>>,
    running_branches: RwLock<HashSet<String>>, // Keeps track of currently replaying branches
    shared_input: PyObject, // optional shared tensor
    branches_callbacks: PyObject, // dict branch_name -> callable fallback
}

#[pymethods]
impl ConditionalGraphRunner {
    #[new]
    fn new(
        branches: Vec<String>,
        graphs: PyObject,
        static_outputs: PyObject,
        failed_branches: Vec<String>,
        shared_input: PyObject,
        branches_callbacks: PyObject,
    ) -> Self {
        let mut failed = HashSet::new();
        for b in failed_branches {
            failed.insert(b);
        }
        ConditionalGraphRunner {
            branches,
            graphs,
            static_outputs,
            failed_branches: RwLock::new(failed),
            running_branches: RwLock::new(HashSet::new()),
            shared_input,
            branches_callbacks,
        }
    }

    /// Runs a specific branch's graph capture.
    ///
    /// Validates inputs, handles eager fallbacks, records run latency, and protects
    /// against parallel re-entrancy conflicts on the GPU.
    fn run<'py>(
        &self,
        py: Python<'py>,
        branch: &str,
        input_tensor: Option<PyObject>,
    ) -> PyResult<PyObject> {
        if !self.branches.iter().any(|b| b == branch) {
            return Err(PyKeyError::new_err(format!(
                "Unknown branch '{}'. Available: {:?}",
                branch,
                self.branches
            )));
        }

        // Validate input tensor type and location if provided
        if let Some(ref input) = input_tensor {
            let torch_mod = py.import("torch")?;
            let tensor_cls = torch_mod.getattr("Tensor")?;
            if !input.bind(py).is_instance(&tensor_cls)? {
                return Err(PyTypeError::new_err("input_tensor must be a torch.Tensor"));
            }
            let is_cuda: bool = input.getattr(py, "is_cuda")?.extract(py)?;
            if !is_cuda {
                return Err(PyValueError::new_err("input_tensor must be on CUDA/HIP device"));
            }
        }

        // If the branch is already marked as failed, immediately route to eager fallback
        let failed = {
            let lock = self.failed_branches.read().unwrap();
            lock.contains(branch)
        };
        if failed {
            return self.eager_fallback(py, branch, input_tensor);
        }

        // Re-entrancy / Thread Collision Guard:
        // If this branch is currently running on another thread/stream, bypass the graph
        // replay and route to eager fallback to prevent memory space address conflicts.
        {
            let lock = self.running_branches.read().unwrap();
            if lock.contains(branch) {
                let log_mod = py.import("logging")?;
                let logger = log_mod.call_method1("getLogger", ("gfxgraph",))?;
                logger.call_method1("warning", (format!("Parallel re-entrancy collision detected on branch '{}' — bypassing graph replay for safety", branch),))?;
                return self.eager_fallback(py, branch, input_tensor);
            }
        }

        // Register that the branch is running
        {
            let mut lock = self.running_branches.write().unwrap();
            lock.insert(branch.to_string());
        }

        // Establish the RAII drop guard to release this branch execution lock on exit
        let _guard = ReentrancyGuard { runner: self, branch };

        // Copy input to shared static buffer if necessary
        if let Some(ref input) = input_tensor {
            if !self.shared_input.is_none(py) {
                self.shared_input.call_method1(py, "copy_", (input,))?;
            }
        }

        let time_mod = py.import("time")?;
        let t0: f64 = time_mod.call_method0("perf_counter")?.extract()?;

        let graphs_dict = self.graphs.downcast_bound::<PyDict>(py)
            .map_err(|_| PyRuntimeError::new_err("Invalid state: graphs must be a dict"))?;
        let graph = graphs_dict.get_item(branch)?;
        if let Some(g) = graph {
            if let Err(e) = g.call_method0("replay") {
                let log_mod = py.import("logging")?;
                let logger = log_mod.call_method1("getLogger", ("gfxgraph",))?;
                logger.call_method1("warning", (format!("Replay failed for branch '{}': {:?} — eager fallback", branch, e),))?;

                if let Ok(mut lock) = self.failed_branches.write() {
                    lock.insert(branch.to_string());
                }
                return self.eager_fallback(py, branch, input_tensor);
            }
        }

        let us = (time_mod.call_method0("perf_counter")?.extract::<f64>()? - t0) * 1e6;

        if let Ok(enable_mod) = py.import("gfxgraph._enable") {
            let _ = enable_mod.call_method1("record_replay_us", (us,));
        }

        let outputs_dict = self.static_outputs.downcast_bound::<PyDict>(py)
            .map_err(|_| PyRuntimeError::new_err("Invalid state: static_outputs must be a dict"))?;
        let output = outputs_dict.get_item(branch)?;
        if let Some(out) = output {
            Ok(out.into())
        } else {
            Err(PyRuntimeError::new_err("Static output buffer not found for branch"))
        }
    }

    /// Evaluates branch callback function eagerly inside a PyTorch no_grad context.
    fn eager_fallback<'py>(&self, py: Python<'py>, branch: &str, input_tensor: Option<PyObject>) -> PyResult<PyObject> {
        let callbacks_dict = self.branches_callbacks.downcast_bound::<PyDict>(py)
            .map_err(|_| PyRuntimeError::new_err("Invalid state: branches_callbacks must be a dict"))?;
        let fn_obj = callbacks_dict.get_item(branch)?.ok_or_else(|| PyRuntimeError::new_err("Branch fallback not found"))?;

        if let Ok(enable_mod) = py.import("gfxgraph._enable") {
            let _ = enable_mod.call_method1("bump", ("fallback_count",));
        }

        let torch_mod = py.import("torch")?;
        let no_grad_ctx = torch_mod.call_method0("no_grad")?;

        let _ = no_grad_ctx.call_method0("__enter__")?;

        let result = if let Some(ref input) = input_tensor {
            fn_obj.call1((input,))
        } else if !self.shared_input.is_none(py) {
            fn_obj.call1((&self.shared_input,))
        } else {
            Err(PyRuntimeError::new_err(format!("No input available for branch '{}'", branch)))
        };

        match result {
            Ok(val) => {
                let _ = no_grad_ctx.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
                Ok(val.into())
            }
            Err(e) => {
                let exc_type = e.get_type(py).into_any().unbind();
                let exc_value = e.value(py).clone().into_any().unbind();
                let exc_traceback = match e.traceback(py) {
                    Some(tb) => tb.clone().into_any().unbind(),
                    None => py.None(),
                };
                let _ = no_grad_ctx.call_method1("__exit__", (exc_type, exc_value, exc_traceback));
                Err(e)
            }
        }
    }
}

/// Exposes the compiled Rust components as a python module.
#[pymodule]
fn rs_gfxgraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Perform one-time initialization of core-affinity and HSA overrides on boot
    let _ = INIT_RESULT.get_or_init(init_environment_and_affinity);

    m.add_class::<BucketRouter>()?;
    m.add_class::<ConditionalGraphRunner>()?;
    m.add_class::<BridgedGraphValidator>()?;
    
    // Register system info functions
    m.add_function(wrap_pyfunction!(pinned_cpu_cores, m)?)?;
    m.add_function(wrap_pyfunction!(hsa_override_applied, m)?)?;
    m.add_function(wrap_pyfunction!(init_status_message, m)?)?;
    
    Ok(())
}

/// Evaluates graph outputs against eager outputs to validate capture correctness.
///
/// Designed to catch silent incorrect replays on AMD platforms (e.g. PyTorch #155684).
#[pyclass]
pub struct BridgedGraphValidator {
    validation_enabled: bool,
}

#[pymethods]
impl BridgedGraphValidator {
    #[new]
    fn new(validation_enabled: bool) -> Self {
        BridgedGraphValidator { validation_enabled }
    }

    /// Verifies if graph execution matches eager execution when validation mode is on.
    fn maybe_validate<'py>(
        &self,
        py: Python<'py>,
        graph_output: PyObject,
        input_tensor: Option<PyObject>,
        model_fn: Option<PyObject>,
    ) -> PyResult<PyObject> {
        if !self.validation_enabled {
            return Ok(graph_output);
        }

        let model = match model_fn {
            Some(m) => m,
            None => return Ok(graph_output),
        };

        let input = match input_tensor {
            Some(i) => i,
            None => return Ok(graph_output),
        };

        let torch = py.import("torch")?;
        let no_grad_ctx = torch.call_method0("no_grad")?;
        let _ = no_grad_ctx.call_method0("__enter__")?;

        let eager_output = model.call1(py, (input,))?;

        let _ = no_grad_ctx.call_method1("__exit__", (py.None(), py.None(), py.None()))?;

        let allclose = torch.call_method1("allclose", (&graph_output, &eager_output))?;
        let is_close: bool = allclose.extract()?;

        if !is_close {
            let log_mod = py.import("logging")?;
            let logger = log_mod.call_method1("getLogger", ("gfxgraph",))?;
            logger.call_method1("error", ("VALIDATION FAILURE: graph output differs from eager output! — possible PyTorch #155684",))?;

            let enable_mod = py.import("gfxgraph._enable").ok();
            if let Some(m) = enable_mod {
                let _ = m.call_method1("bump", ("validation_failures",));
            }
            return Ok(eager_output);
        }

        let log_mod = py.import("logging")?;
        let logger = log_mod.call_method1("getLogger", ("gfxgraph",))?;
        logger.call_method1("debug", ("Validation passed",))?;

        Ok(graph_output)
    }
}
