//! Process-global capture/replay compatibility serialization gate.
//!
//! Historical graph-capture backends had process-global bookkeeping. Two sessions
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
use std::cell::Cell;

/// The single process-wide capture/replay gate.
static CAPTURE_GATE: RawRwLock = RawRwLock::INIT;

thread_local! {
    /// How many replay (shared) leases *this thread* is holding.
    ///
    /// The gate is one reader/writer lock, so a thread that takes the exclusive
    /// side while holding the shared side deadlocks against itself. Whether a
    /// caller is nested is not a local property — a resident ring is built
    /// inside the kernel call, which runs lease-free on a capture epoch and
    /// under a shared replay lease otherwise — so the gate tracks it instead of
    /// asking callers to know.
    static REPLAY_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Replay leases held by the calling thread.
#[inline]
#[must_use]
pub fn replay_depth() -> usize {
    REPLAY_DEPTH.with(Cell::get)
}

/// RAII lease for the process-global capture compatibility gate.
///
/// Capture is exclusive with all graph replay. Dropping the lease releases the
/// process-global gate, so consumers do not need to pair raw lock/unlock calls.
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

    /// Acquire the capture lease only if it is free right now.
    ///
    /// For callers that may already be holding a *replay* lease on the same
    /// thread. The gate is one reader/writer lock, so an exclusive acquire
    /// nested inside a shared one deadlocks against itself — and whether a
    /// caller is nested is not always a local property. A resident ring, for
    /// instance, is built inside the kernel call, which runs lease-free on a
    /// capture epoch but under a shared replay lease otherwise; those two
    /// conditions are computed independently and are only correlated in
    /// practice.
    ///
    /// Returning `None` is not a failure. Capture is an optimisation, so a
    /// caller that cannot take the gate should launch eagerly instead — which
    /// is the same fallback it already needs for a capture that fails outright.
    #[inline]
    #[must_use]
    pub fn try_acquire() -> Option<Self> {
        try_lock_capture().then_some(Self { _private: () })
    }

    /// Wait for the capture lease unless this thread already holds a replay
    /// lease, in which case waiting would self-deadlock and this returns
    /// `None`.
    ///
    /// Cross-thread replay/capture ordering is handled separately by the
    /// authoritative replay-gate -> wave-gang-slot order in
    /// [`ConcurrentWaveGangGate`](super::wave_gang::ConcurrentWaveGangGate).
    /// This method addresses the distinct same-thread self-nesting case.
    #[inline]
    #[must_use]
    pub fn acquire_unless_nested() -> Option<Self> {
        (replay_depth() == 0).then(Self::acquire)
    }
}

impl Drop for CaptureLease {
    #[inline]
    fn drop(&mut self) {
        unlock_capture();
    }
}

/// RAII lease for a process-global replay compatibility gate.
///
/// Replays share the gate with other replays and are blocked only while an
/// exclusive capture lease is active.
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

/// Acquire the exclusive capture lock if it is uncontended, without blocking.
///
/// Returns whether the lock was taken; pair a `true` exactly once with
/// [`unlock_capture`]. Prefer [`CaptureLease::try_acquire`], which pairs them
/// for you.
#[inline]
pub fn try_lock_capture() -> bool {
    CAPTURE_GATE.try_lock_exclusive()
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
    REPLAY_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
}

/// Release a shared replay lock.
#[inline]
pub fn unlock_replay() {
    REPLAY_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
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

    /// A capture nested inside this thread's own replay lease must decline
    /// rather than wait. Waiting is a self-deadlock: the gate is one lock, and
    /// the thread asking for the exclusive side is the thread holding the
    /// shared side. If this ever blocks, the test hangs rather than fails,
    /// which is the honest signal.
    #[test]
    fn capture_declines_when_nested_in_this_threads_replay() {
        assert_eq!(replay_depth(), 0);
        let replay = ReplayLease::acquire();
        assert_eq!(replay_depth(), 1);
        assert!(
            CaptureLease::acquire_unless_nested().is_none(),
            "a capture nested inside a replay lease must decline"
        );
        drop(replay);
        assert_eq!(replay_depth(), 0);

        // Unnested, the same call must succeed -- otherwise "decline when
        // nested" would be indistinguishable from "never capture".
        let capture = CaptureLease::acquire_unless_nested();
        assert!(capture.is_some(), "an unnested capture must be granted");
        drop(capture);
    }
}
