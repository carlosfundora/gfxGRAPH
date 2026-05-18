use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use dashmap::DashMap;

// We use an internal struct to hold the metrics to ensure thread safety
struct StatsCore {
    counters: DashMap<String, i64>,
    total_replay_us: Mutex<f64>,
}

impl StatsCore {
    fn new() -> Self {
        Self {
            counters: DashMap::new(),
            total_replay_us: Mutex::new(0.0),
        }
    }
}

lazy_static::lazy_static! {
    static ref STATS_CORE: Arc<StatsCore> = Arc::new(StatsCore::new());
}

#[pyfunction]
#[pyo3(signature = (counter, amount=1))]
fn bump(counter: &str, amount: i64) -> PyResult<()> {
    *STATS_CORE.counters.entry(counter.to_string()).or_insert(0) += amount;
    Ok(())
}

#[pyfunction]
fn record_replay_us(us: f64) -> PyResult<()> {
    *STATS_CORE.counters.entry("replay_count".to_string()).or_insert(0) += 1;
    let mut total = STATS_CORE.total_replay_us.lock().unwrap();
    *total += us;
    Ok(())
}

#[pyfunction]
fn stats(py: Python<'_>) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    let dict = PyDict::new(py);

    for entry in STATS_CORE.counters.iter() {
        dict.set_item(entry.key(), *entry.value())?;
    }

    // Fill in default required counters if they don't exist
    for default_counter in &["capture_count", "replay_count", "fallback_count", "validation_failures"] {
        if !dict.contains(default_counter)? {
            dict.set_item(default_counter, 0)?;
        }
    }

    let total = *STATS_CORE.total_replay_us.lock().unwrap();
    let count: i64 = match dict.get_item("replay_count")? {
        Some(val) => val.extract()?,
        None => 0,
    };

    let avg = if count > 0 { total / (count as f64) } else { 0.0 };

    dict.set_item("avg_replay_us", avg)?;

    Ok(dict.into())
}

#[pyfunction]
fn reset() -> PyResult<()> {
    STATS_CORE.counters.clear();
    let mut total = STATS_CORE.total_replay_us.lock().unwrap();
    *total = 0.0;
    Ok(())
}

#[pymodule]
fn rs_gfxgraph_stats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bump, m)?)?;
    m.add_function(wrap_pyfunction!(record_replay_us, m)?)?;
    m.add_function(wrap_pyfunction!(stats, m)?)?;
    m.add_function(wrap_pyfunction!(reset, m)?)?;
    Ok(())
}
