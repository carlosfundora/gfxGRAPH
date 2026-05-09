use pyo3::prelude::*;
use std::sync::Mutex;
use std::collections::HashMap;

#[pyclass]
pub struct StatsManager {
    stats: Mutex<Stats>,
}

struct Stats {
    counts: HashMap<String, i64>,
    replay_count: u64,
    total_replay_us: f64,
    enabled_at: Option<f64>,
}

#[pymethods]
impl StatsManager {
    #[new]
    fn new() -> Self {
        StatsManager {
            stats: Mutex::new(Stats {
                counts: HashMap::new(),
                replay_count: 0,
                total_replay_us: 0.0,
                enabled_at: None,
            }),
        }
    }

    fn bump(&self, counter: String, amount: i64) {
        let mut stats = self.stats.lock().unwrap();
        *stats.counts.entry(counter).or_insert(0) += amount;
    }

    fn record_replay_us(&self, us: f64) {
        let mut stats = self.stats.lock().unwrap();
        stats.replay_count += 1;
        stats.total_replay_us += us;
    }

    fn set_enabled_at(&self, time: f64) {
        let mut stats = self.stats.lock().unwrap();
        stats.enabled_at = Some(time);
    }

    fn stats(&self, py: Python) -> PyResult<PyObject> {
        let stats = self.stats.lock().unwrap();
        let dict = pyo3::types::PyDict::new_bound(py);

        // Initialize defaults that might be missing
        dict.set_item("capture_count", stats.counts.get("capture_count").unwrap_or(&0))?;
        dict.set_item("fallback_count", stats.counts.get("fallback_count").unwrap_or(&0))?;
        dict.set_item("validation_failures", stats.counts.get("validation_failures").unwrap_or(&0))?;

        // Add everything else
        for (k, v) in &stats.counts {
            if k != "capture_count" && k != "fallback_count" && k != "validation_failures" {
                dict.set_item(k, v)?;
            }
        }

        dict.set_item("replay_count", stats.replay_count)?;

        let avg = if stats.replay_count > 0 {
            stats.total_replay_us / stats.replay_count as f64
        } else {
            0.0
        };
        dict.set_item("avg_replay_us", avg)?;

        match stats.enabled_at {
            Some(time) => dict.set_item("enabled_at", time)?,
            None => dict.set_item("enabled_at", py.None())?,
        }

        Ok(dict.into())
    }
}
