use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashSet;

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

#[pymodule]
fn gfxgraph_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BucketRouter>()?;
    Ok(())
}
