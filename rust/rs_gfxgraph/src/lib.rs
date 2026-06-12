use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyKeyError, PyRuntimeError, PyTypeError};
use pyo3::types::{PyAny, PyDict};
use std::collections::HashSet;
use std::sync::RwLock;

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

    fn route(&self, input_size: usize) -> PyResult<(i64, u8)> {
        let bucket_val = match self.buckets.binary_search(&input_size) {
            Ok(idx) => self.buckets[idx],
            Err(idx) => {
                if idx < self.buckets.len() {
                    self.buckets[idx]
                } else {
                    return Ok((-1, 2));
                }
            }
        };

        let state = if self.warmed_up.contains(&bucket_val) {
            0 // Ready
        } else if self.failed_buckets.contains(&bucket_val) {
            2 // Failed
        } else {
            1 // NeedsWarmup
        };

        Ok((bucket_val as i64, state))
    }

    fn mark_warmed_up(&mut self, bucket_size: usize) {
        self.warmed_up.insert(bucket_size);
    }

    fn mark_failed(&mut self, bucket_size: usize) {
        self.failed_buckets.insert(bucket_size);
    }

    fn warmed_up_list(&self) -> Vec<usize> {
        let mut list: Vec<usize> = self.warmed_up.iter().copied().collect();
        list.sort_unstable();
        list
    }

    fn failed_list(&self) -> Vec<usize> {
        let mut list: Vec<usize> = self.failed_buckets.iter().copied().collect();
        list.sort_unstable();
        list
    }
}

#[pyclass]
pub struct ConditionalGraphRunner {
    branches: Vec<String>,
    graphs: Py<PyAny>, // dict branch_name -> CUDAGraph
    static_outputs: Py<PyAny>, // dict branch_name -> static output tensor
    failed_branches: RwLock<HashSet<String>>,
    shared_input: Py<PyAny>, // optional shared tensor
    branches_callbacks: Py<PyAny>, // dict branch_name -> callable fallback
}

#[pymethods]
impl ConditionalGraphRunner {
    #[new]
    fn new(
        branches: Vec<String>,
        graphs: Py<PyAny>,
        static_outputs: Py<PyAny>,
        failed_branches: Vec<String>,
        shared_input: Py<PyAny>,
        branches_callbacks: Py<PyAny>,
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
            shared_input,
            branches_callbacks,
        }
    }

    fn run<'py>(
        &self,
        py: Python<'py>,
        branch: &str,
        input_tensor: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if !self.branches.iter().any(|b| b == branch) {
            return Err(PyKeyError::new_err(format!(
                "Unknown branch '{}'. Available: {:?}",
                branch,
                self.branches
            )));
        }

        if let Some(ref input) = input_tensor {
            let torch_mod = py.import("torch")?;
            let tensor_cls = torch_mod.getattr("Tensor")?;
            if !input.bind(py).is_instance(&tensor_cls)? {
                return Err(PyTypeError::new_err("input_tensor must be a torch.Tensor"));
            }
            let is_cuda: bool = input.getattr(py, "is_cuda")?.extract(py)?;
            if !is_cuda {
                return Err(PyValueError::new_err("input_tensor must be on CUDA device"));
            }
        }

        let failed = {
            let lock = self.failed_branches.read().unwrap();
            lock.contains(branch)
        };
        if failed {
            return self.eager_fallback(py, branch, input_tensor);
        }

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
                    if lock.insert(branch.to_string()) {
                        if let Ok(enable_mod) = py.import("gfxgraph._enable") {
                            let _ = enable_mod.call_method1("bump", ("fallback_count",));
                        }
                    }
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
            Err(PyRuntimeError::new_err("Output not found"))
        }
    }

    fn eager_fallback<'py>(&self, py: Python<'py>, branch: &str, input_tensor: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let callbacks_dict = self.branches_callbacks.downcast_bound::<PyDict>(py)
            .map_err(|_| PyRuntimeError::new_err("Invalid state: branches_callbacks must be a dict"))?;
        let fn_obj = callbacks_dict.get_item(branch)?.ok_or_else(|| PyRuntimeError::new_err("Branch fallback not found"))?;

        // We no longer bump fallback_count here, it's structurally bumped on transition.

        let torch_mod = py.import("torch")?;
        let no_grad_ctx = torch_mod.call_method0("no_grad")?;

        // Use no_grad context
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
                return Err(e);
            }
        }
    }
}


#[pymodule]
fn rs_gfxgraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BucketRouter>()?;
    m.add_class::<ConditionalGraphRunner>()?;
    m.add_class::<BridgedGraphValidator>()?;
    Ok(())
}

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

    fn maybe_validate<'py>(
        &self,
        py: Python<'py>,
        graph_output: Py<PyAny>,
        input_tensor: Option<Py<PyAny>>,
        model_fn: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
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

        // simplified for now, use default tolerances or implement kwargs equivalent
        let allclose = torch.call_method1("allclose", (&graph_output, &eager_output))?;
        let is_close: bool = allclose.extract()?;

        if !is_close {
            let log_mod = py.import("logging")?;
            let logger = log_mod.call_method1("getLogger", ("gfxgraph",))?;
            logger.call_method1("error", ("VALIDATION FAILURE: graph output differs from eager output! \u{2014} possible PyTorch #155684",))?;

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
