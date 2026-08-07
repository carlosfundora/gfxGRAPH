//! Process-global hipGraph capture serialization gate.
//!
//! On AMD gfx1030 / RDNA2 under ROCm, HIP stream-capture bookkeeping is
//! process-global even with `hipStreamCaptureModeThreadLocal`. Two LLM sessions
//! in one process that capture graphs at the same time corrupt each other's
//! output (NaN). This module provides ONE process-wide reader/writer lock so
//! capture is serialized across every session, while replay stays concurrent:
//!
//!   * capture -> write (exclusive): one capture at a time, and excludes any
//!     in-flight replay during the brief one-time capture window.
//!   * replay  -> read  (shared):    replays run concurrently; blocked only
//!     while a capture holds the write lock.
//!
//! The lock is a module-level `static` on purpose: a per-object lock would give
//! each session its own lock and therefore no mutual exclusion -- which is
//! exactly the bug this prevents.
//!
//! The PyO3 layer (`rs_gfxgraph::CaptureLock` / `ReplayLock`) wraps these
//! functions in RAII context managers and releases the GIL while blocking.

use parking_lot::lock_api::RawRwLock as _;
use parking_lot::RawRwLock;

/// The single process-wide capture/replay gate.
static CAPTURE_GATE: RawRwLock = RawRwLock::INIT;

/// RAII lease for process-global HIP graph capture.
#[derive(Debug)]
pub struct CaptureLease {
    _private: (),
}

impl CaptureLease {
    #[inline]
    #[must_use]
    pub fn acquire() -> Self {
        lock_capture();
        Self { _private: () }
    }
}

impl Drop for CaptureLease {
    #[inline]
    fn drop(&mut self) {
        unlock_capture();
    }
}

/// RAII lease for a process-global HIP graph replay.
#[derive(Debug)]
pub struct ReplayLease {
    _private: (),
}

impl ReplayLease {
    #[inline]
    #[must_use]
    pub fn acquire() -> Self {
        lock_replay();
        Self { _private: () }
    }
}

impl Drop for ReplayLease {
    #[inline]
    fn drop(&mut self) {
        unlock_replay();
    }
}

/// Acquire the exclusive capture lock.
///
/// Blocks until no other capture (writer) and no in-flight replay (reader)
/// holds the gate. Pair exactly once with [`unlock_capture`].
#[inline]
pub fn lock_capture() {
    CAPTURE_GATE.lock_exclusive();
}

/// Release the exclusive capture lock.
#[inline]
pub fn unlock_capture() {
    // SAFETY: the raw lock has no poisoning; callers (the PyO3 `CaptureLock`
    // RAII guard / `acquire_capture_lock`+`release_capture_lock` pair) call this
    // exactly once per successful `lock_capture()`.
    unsafe {
        CAPTURE_GATE.unlock_exclusive();
    }
}

/// Acquire a shared replay lock.
///
/// Multiple replays may hold it concurrently; excluded only while a capture
/// holds the write lock. Pair exactly once with [`unlock_replay`].
#[inline]
pub fn lock_replay() {
    CAPTURE_GATE.lock_shared();
}

/// Release a shared replay lock.
#[inline]
pub fn unlock_replay() {
    // SAFETY: paired exactly once with a prior `lock_replay()` by the PyO3
    // `ReplayLock` RAII guard.
    unsafe {
        CAPTURE_GATE.unlock_shared();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn capture_serializes_and_replay_shares() {
        // Part 1: the exclusive capture lock admits at most one holder, even
        // under contention from many threads. This is the actual corruption
        // fix -- concurrent capture must never overlap.
        let holders = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let holders = Arc::clone(&holders);
            let max_seen = Arc::clone(&max_seen);
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    lock_capture();
                    let now = holders.fetch_add(1, Ordering::SeqCst) + 1;
                    max_seen.fetch_max(now, Ordering::SeqCst);
                    assert_eq!(now, 1, "more than one capture holder at once");
                    thread::sleep(Duration::from_micros(20));
                    holders.fetch_sub(1, Ordering::SeqCst);
                    unlock_capture();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            1,
            "capture lock was held by >1 thread simultaneously"
        );

        // Part 2: two replay (shared) holders coexist. If the lock were
        // exclusive for readers, the barrier below would deadlock because the
        // second thread could never enter while the first holds the lock.
        let barrier = Arc::new(Barrier::new(2));
        let mut replay_handles = Vec::new();
        for _ in 0..2 {
            let barrier = Arc::clone(&barrier);
            replay_handles.push(thread::spawn(move || {
                lock_replay();
                barrier.wait(); // both readers must be inside simultaneously
                unlock_replay();
            }));
        }
        for h in replay_handles {
            h.join().unwrap();
        }
    }
}
