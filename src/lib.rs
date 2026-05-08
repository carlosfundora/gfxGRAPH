use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::sync::RwLock;
use std::collections::HashMap;

static CAPTURE_COUNT: AtomicUsize = AtomicUsize::new(0);
static REPLAY_COUNT: AtomicUsize = AtomicUsize::new(0);
static FALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);
static VALIDATION_FAILURES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_REPLAY_US: AtomicU64 = AtomicU64::new(0);

lazy_static::lazy_static! {
    static ref DYNAMIC_STATS: RwLock<HashMap<String, usize>> = RwLock::new(HashMap::new());
}

fn add_to_total_us(us: f64) {
    let mut current = TOTAL_REPLAY_US.load(Ordering::Relaxed);
    loop {
        let current_f = f64::from_bits(current);
        let new_f = current_f + us;
        let new_bits = new_f.to_bits();
        match TOTAL_REPLAY_US.compare_exchange_weak(current, new_bits, Ordering::SeqCst, Ordering::Relaxed) {
            Ok(_) => break,
            Err(x) => current = x,
        }
    }
}

#[pyfunction]
fn bump(counter: &str, amount: Option<usize>) {
    let amt = amount.unwrap_or(1);
    match counter {
        "capture_count" => { CAPTURE_COUNT.fetch_add(amt, Ordering::SeqCst); },
        "replay_count" => { REPLAY_COUNT.fetch_add(amt, Ordering::SeqCst); },
        "fallback_count" => { FALLBACK_COUNT.fetch_add(amt, Ordering::SeqCst); },
        "validation_failures" => { VALIDATION_FAILURES.fetch_add(amt, Ordering::SeqCst); },
        _ => {
            let mut dyn_stats = DYNAMIC_STATS.write().unwrap();
            let entry = dyn_stats.entry(counter.to_string()).or_insert(0);
            *entry += amt;
        }
    }
}

#[pyfunction]
fn record_replay_us(us: f64) {
    REPLAY_COUNT.fetch_add(1, Ordering::SeqCst);
    add_to_total_us(us);
}

#[pyfunction]
fn stats(py: Python) -> PyResult<PyObject> {
    let captures = CAPTURE_COUNT.load(Ordering::SeqCst);
    let replays = REPLAY_COUNT.load(Ordering::SeqCst);
    let fallbacks = FALLBACK_COUNT.load(Ordering::SeqCst);
    let failures = VALIDATION_FAILURES.load(Ordering::SeqCst);
    let total_us = f64::from_bits(TOTAL_REPLAY_US.load(Ordering::SeqCst));

    let avg = if replays > 0 { total_us / replays as f64 } else { 0.0 };

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("capture_count", captures)?;
    dict.set_item("replay_count", replays)?;
    dict.set_item("fallback_count", fallbacks)?;
    dict.set_item("validation_failures", failures)?;
    dict.set_item("avg_replay_us", avg)?;

    let dyn_stats = DYNAMIC_STATS.read().unwrap();
    for (k, v) in dyn_stats.iter() {
        dict.set_item(k, v)?;
    }

    Ok(dict.into())
}

#[pyfunction]
fn reset() {
    CAPTURE_COUNT.store(0, Ordering::SeqCst);
    REPLAY_COUNT.store(0, Ordering::SeqCst);
    FALLBACK_COUNT.store(0, Ordering::SeqCst);
    VALIDATION_FAILURES.store(0, Ordering::SeqCst);
    TOTAL_REPLAY_US.store(0.0f64.to_bits(), Ordering::SeqCst);
    let mut dyn_stats = DYNAMIC_STATS.write().unwrap();
    dyn_stats.clear();
}

#[pymodule]
fn gfxgraph_stats(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bump, m)?)?;
    m.add_function(wrap_pyfunction!(record_replay_us, m)?)?;
    m.add_function(wrap_pyfunction!(stats, m)?)?;
    m.add_function(wrap_pyfunction!(reset, m)?)?;
    Ok(())
}
