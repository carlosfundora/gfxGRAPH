use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyKeyError, PyRuntimeError, PyTypeError};
use pyo3::types::{PyDict, PyString};
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
            0 // Ready
        } else if self.failed_buckets.contains(&bucket) {
            2 // Failed
        } else {
            1 // NeedsWarmup
        };

        Ok((bucket, state))
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
    graphs: PyObject, // dict branch_name -> CUDAGraph
    static_outputs: PyObject, // dict branch_name -> static output tensor
    failed_branches: RwLock<HashSet<String>>,
    shared_input: PyObject, // optional shared tensor
    branches_callbacks: PyObject, // dict branch_name -> callable fallback
    time_perf_counter: PyObject,
    logger_warning: PyObject,
    record_replay_us_fn: Option<PyObject>,
    bump_fn: Option<PyObject>,
    torch_no_grad: PyObject,
}

#[pymethods]
impl ConditionalGraphRunner {
    #[new]
    fn new(
        py: Python<'_>,
        branches: Vec<String>,
        graphs: PyObject,
        static_outputs: PyObject,
        failed_branches: Vec<String>,
        shared_input: PyObject,
        branches_callbacks: PyObject,
    ) -> PyResult<Self> {
        let mut failed = HashSet::new();
        for b in failed_branches {
            failed.insert(b);
        }

        let time_mod = py.import("time")?;
        let time_perf_counter = time_mod.getattr("perf_counter")?.into();

        let log_mod = py.import("logging")?;
        let logger = log_mod.call_method1("getLogger", ("gfxgraph",))?;
        let logger_warning = logger.getattr("warning")?.into();

        let (record_replay_us_fn, bump_fn) = if let Ok(enable_mod) = py.import("gfxgraph._enable") {
            let record = enable_mod.getattr("record_replay_us").ok().map(|r| r.into());
            let bump = enable_mod.getattr("bump").ok().map(|b| b.into());
            (record, bump)
        } else {
            (None, None)
        };

        let torch = py.import("torch")?;
        let torch_no_grad = torch.getattr("no_grad")?.into();

        Ok(ConditionalGraphRunner {
            branches,
            graphs,
            static_outputs,
            failed_branches: RwLock::new(failed),
            shared_input,
            branches_callbacks,
            time_perf_counter,
            logger_warning,
            record_replay_us_fn,
            bump_fn,
            torch_no_grad,
        })
    }

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

        if let Some(ref input) = input_tensor {
            let is_tensor = input.getattr(py, "is_cuda").is_ok();
            if !is_tensor {
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

        let t0: f64 = self.time_perf_counter.call0(py)?.extract(py)?;

        let graphs_dict = self.graphs.downcast_bound::<PyDict>(py)
            .map_err(|_| PyRuntimeError::new_err("Invalid state: graphs must be a dict"))?;
        let graph = graphs_dict.get_item(branch)?;
        if let Some(g) = graph {
            if let Err(e) = g.call_method0("replay") {
                let _ = self.logger_warning.call1(py, (format!("Replay failed for branch '{}': {:?} — eager fallback", branch, e),));

                if let Ok(mut lock) = self.failed_branches.write() {
                    lock.insert(branch.to_string());
                }
                return self.eager_fallback(py, branch, input_tensor);
            }
        }

        let us = (self.time_perf_counter.call0(py)?.extract::<f64>(py)? - t0) * 1e6;

        if let Some(f) = &self.record_replay_us_fn {
            let _ = f.call1(py, (us,));
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

    fn eager_fallback<'py>(&self, py: Python<'py>, branch: &str, input_tensor: Option<PyObject>) -> PyResult<PyObject> {
        let callbacks_dict = self.branches_callbacks.downcast_bound::<PyDict>(py)
            .map_err(|_| PyRuntimeError::new_err("Invalid state: branches_callbacks must be a dict"))?;
        let fn_obj = callbacks_dict.get_item(branch)?.ok_or_else(|| PyRuntimeError::new_err("Branch fallback not found"))?;

        if let Some(b) = &self.bump_fn {
            let _ = b.call1(py, ("fallback_count",));
        }

        let no_grad_ctx = self.torch_no_grad.call0(py)?;

        // Use no_grad context
        let _ = no_grad_ctx.call_method0(py, "__enter__")?;

        let result = if let Some(ref input) = input_tensor {
            fn_obj.call1((input,))
        } else if !self.shared_input.is_none(py) {
            fn_obj.call1((&self.shared_input,))
        } else {
            Err(PyRuntimeError::new_err(format!("No input available for branch '{}'", branch)))
        };

        match result {
            Ok(val) => {
                let _ = no_grad_ctx.call_method1(py, "__exit__", (py.None(), py.None(), py.None()))?;
                Ok(val.into())
            }
            Err(e) => {
                // We just let the exception propagate, torch.no_grad.__exit__ must be called
                // Since we don't have the traceback, we'll just pass None for all.
                let _ = no_grad_ctx.call_method1(py, "__exit__", (py.None(), py.None(), py.None()));
                return Err(e);
            }
        }
    }
}


#[pymodule]
fn gfxgraph_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BucketRouter>()?;
    m.add_class::<ConditionalGraphRunner>()?;
    m.add_class::<BridgedGraphValidator>()?;
    Ok(())
}

#[pyclass]
pub struct BridgedGraphValidator {
    validation_enabled: bool,
    torch_no_grad: PyObject,
    torch_allclose: PyObject,
    logger_error: PyObject,
    logger_debug: PyObject,
    bump_fn: Option<PyObject>,
}

#[pymethods]
impl BridgedGraphValidator {
    #[new]
    fn new(py: Python<'_>, validation_enabled: bool) -> PyResult<Self> {
        let torch = py.import("torch")?;
        let torch_no_grad = torch.getattr("no_grad")?.into();
        let torch_allclose = torch.getattr("allclose")?.into();

        let log_mod = py.import("logging")?;
        let logger = log_mod.call_method1("getLogger", ("gfxgraph",))?;
        let logger_error = logger.getattr("error")?.into();
        let logger_debug = logger.getattr("debug")?.into();

        let bump_fn = if let Ok(enable_mod) = py.import("gfxgraph._enable") {
            enable_mod.getattr("bump").ok().map(|b| b.into())
        } else {
            None
        };

        Ok(BridgedGraphValidator {
            validation_enabled,
            torch_no_grad,
            torch_allclose,
            logger_error,
            logger_debug,
            bump_fn,
        })
    }

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

        let no_grad_ctx = self.torch_no_grad.call0(py)?;
        let _ = no_grad_ctx.call_method0(py, "__enter__")?;

        let eager_output = model.call1(py, (input,))?;

        let _ = no_grad_ctx.call_method1(py, "__exit__", (py.None(), py.None(), py.None()))?;

        // simplified for now, use default tolerances or implement kwargs equivalent
        let allclose = self.torch_allclose.call1(py, (&graph_output, &eager_output))?;
        let is_close: bool = allclose.extract(py)?;

        if !is_close {
            let _ = self.logger_error.call1(py, ("VALIDATION FAILURE: graph output differs from eager output! \u{2014} possible PyTorch #155684",));

            if let Some(b) = &self.bump_fn {
                let _ = b.call1(py, ("validation_failures",));
            }
            return Ok(eager_output);
        }

        let _ = self.logger_debug.call1(py, ("Validation passed",));

        Ok(graph_output)
    }
}
