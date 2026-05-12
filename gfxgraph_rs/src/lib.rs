use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyKeyError, PyRuntimeError, PyTypeError};
use pyo3::types::{PyDict, PyString};
use std::collections::HashSet;
use std::sync::RwLock;

#[pyclass]
pub struct BucketSelector {
    buckets: Vec<usize>,
}

#[pymethods]
impl BucketSelector {
    #[new]
    fn new(buckets: Vec<usize>) -> Self {
        BucketSelector { buckets }
    }

    fn select_bucket(&self, input_size: usize) -> PyResult<usize> {
        match self.buckets.binary_search(&input_size) {
            Ok(idx) => Ok(self.buckets[idx]),
            Err(idx) => {
                if idx < self.buckets.len() {
                    Ok(self.buckets[idx])
                } else {
                    Err(PyValueError::new_err(format!(
                        "Input size {} exceeds largest bucket {}. Add a larger bucket.",
                        input_size,
                        self.buckets.last().unwrap_or(&0)
                    )))
                }
            }
        }
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
            shared_input,
            branches_callbacks,
        }
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
            Err(PyRuntimeError::new_err("Output not found"))
        }
    }

    fn eager_fallback<'py>(&self, py: Python<'py>, branch: &str, input_tensor: Option<PyObject>) -> PyResult<PyObject> {
        let callbacks_dict = self.branches_callbacks.downcast_bound::<PyDict>(py)
            .map_err(|_| PyRuntimeError::new_err("Invalid state: branches_callbacks must be a dict"))?;
        let fn_obj = callbacks_dict.get_item(branch)?.ok_or_else(|| PyRuntimeError::new_err("Branch fallback not found"))?;

        if let Ok(enable_mod) = py.import("gfxgraph._enable") {
            let _ = enable_mod.call_method1("bump", ("fallback_count",));
        }

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
                // We just let the exception propagate, torch.no_grad.__exit__ must be called
                // Since we don't have the traceback, we'll just pass None for all.
                let _ = no_grad_ctx.call_method1("__exit__", (py.None(), py.None(), py.None()));
                return Err(e);
            }
        }
    }
}


#[pymodule]
fn gfxgraph_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BucketSelector>()?;
    m.add_class::<ConditionalGraphRunner>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_bucket() {
        let selector = BucketSelector::new(vec![1, 4, 8, 16, 32]);
        assert_eq!(selector.select_bucket(1).unwrap(), 1);
        assert_eq!(selector.select_bucket(2).unwrap(), 4);
        assert_eq!(selector.select_bucket(4).unwrap(), 4);
        assert_eq!(selector.select_bucket(5).unwrap(), 8);
        assert_eq!(selector.select_bucket(32).unwrap(), 32);

        // Error case
        assert!(selector.select_bucket(33).is_err());
    }
}
