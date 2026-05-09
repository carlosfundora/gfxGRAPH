use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

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

mod stats;
use stats::StatsManager;

#[pymodule]
fn gfxgraph_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BucketSelector>()?;
    m.add_class::<StatsManager>()?;
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
