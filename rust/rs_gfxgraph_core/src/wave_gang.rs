//! Concurrent graph-replay admission for kernel-owned Wave32 gangs.
//!
//! gfxGRAPH does not choose kernel launch geometry. A kernel supplies its
//! native wave width, waves per workgroup, and workgroups per replay. This
//! module only controls how many already-captured graph replays may execute
//! concurrently. Concurrent replays multiply the number of active kernel-owned
//! wave gangs without pretending that the hardware has a wider physical wave.

use crate::capture_gate::{lock_replay, unlock_replay};
use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};

const MAX_WORKGROUP_THREADS: u32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConcurrentWaveGangPlan {
    pub native_wave_size: u32,
    pub waves_per_workgroup: u32,
    pub workgroups_per_replay: u32,
    pub concurrent_replays: u32,
}

/// Policy for tiling a very large logical Wave32 workload into bounded replay gangs.
///
/// The planner never changes the hardware-native wave width. It only chooses how
/// many workgroups are submitted per replay and how many replays may be admitted
/// concurrently under an explicit in-flight-wave budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaveGangScalePolicy {
    /// Maximum workgroups packed into one captured/replayed graph invocation.
    pub max_workgroups_per_replay: u32,
    /// Maximum number of concurrent replay leases, irrespective of wave budget.
    pub max_concurrent_replays: u32,
    /// Maximum logical waves admitted across all concurrent replay leases.
    pub max_in_flight_waves: u64,
}

impl WaveGangScalePolicy {
    /// Conservative gfx1030 policy for large streams. The wave budget is explicit
    /// so callers may replace this with a measured device-specific value.
    pub fn gfx1030(max_in_flight_waves: u64) -> Result<Self, WaveGangError> {
        if max_in_flight_waves == 0 {
            return Err(WaveGangError::InvalidWaveBudget(0));
        }
        Ok(Self {
            max_workgroups_per_replay: 1024,
            max_concurrent_replays: 32,
            max_in_flight_waves,
        })
    }

    pub fn validate(self) -> Result<(), WaveGangError> {
        if self.max_workgroups_per_replay == 0 {
            return Err(WaveGangError::InvalidWorkgroupsPerReplay(0));
        }
        if self.max_concurrent_replays == 0 {
            return Err(WaveGangError::InvalidConcurrentReplays(0));
        }
        if self.max_in_flight_waves == 0 {
            return Err(WaveGangError::InvalidWaveBudget(0));
        }
        Ok(())
    }
}

/// Result of scaling an arbitrarily large logical workload into bounded replay tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScaledWaveGangPlan {
    /// Admission plan used by the replay gate.
    pub replay: ConcurrentWaveGangPlan,
    /// Total logical workgroups in the workload.
    pub total_workgroups: u64,
    /// Number of replay tiles required to cover the whole workload exactly.
    pub replay_tiles: u64,
    /// Number of workgroups in the final replay tile.
    pub tail_workgroups: u32,
}

impl ScaledWaveGangPlan {
    /// Build a Wave32 replay tiling for a large workload while respecting both
    /// the concurrency cap and the explicit in-flight-wave budget.
    pub fn rdna2_wave32(
        waves_per_workgroup: u32,
        total_workgroups: u64,
        policy: WaveGangScalePolicy,
    ) -> Result<Self, WaveGangError> {
        if total_workgroups == 0 {
            return Err(WaveGangError::InvalidTotalWorkgroups(0));
        }
        policy.validate()?;
        let max_wg = u64::from(policy.max_workgroups_per_replay).min(total_workgroups);
        let workgroups_per_replay =
            u32::try_from(max_wg).map_err(|_| WaveGangError::ArithmeticOverflow)?;
        let waves_per_replay = u64::from(waves_per_workgroup)
            .checked_mul(u64::from(workgroups_per_replay))
            .ok_or(WaveGangError::ArithmeticOverflow)?;
        if waves_per_replay == 0 || waves_per_replay > policy.max_in_flight_waves {
            return Err(WaveGangError::ReplayExceedsWaveBudget {
                waves_per_replay,
                maximum: policy.max_in_flight_waves,
            });
        }
        let budget_concurrency = policy.max_in_flight_waves / waves_per_replay;
        let concurrent_replays = u32::try_from(
            budget_concurrency
                .max(1)
                .min(u64::from(policy.max_concurrent_replays)),
        )
        .map_err(|_| WaveGangError::ArithmeticOverflow)?;
        let replay = ConcurrentWaveGangPlan::rdna2_wave32(
            waves_per_workgroup,
            workgroups_per_replay,
            concurrent_replays,
        )?;
        let replay_tiles = total_workgroups.div_ceil(u64::from(workgroups_per_replay));
        let tail = total_workgroups % u64::from(workgroups_per_replay);
        let tail_workgroups = if tail == 0 {
            workgroups_per_replay
        } else {
            u32::try_from(tail).map_err(|_| WaveGangError::ArithmeticOverflow)?
        };
        Ok(Self {
            replay,
            total_workgroups,
            replay_tiles,
            tail_workgroups,
        })
    }
}

impl ConcurrentWaveGangPlan {
    pub fn new(
        native_wave_size: u32,
        waves_per_workgroup: u32,
        workgroups_per_replay: u32,
        concurrent_replays: u32,
    ) -> Result<Self, WaveGangError> {
        let plan = Self {
            native_wave_size,
            waves_per_workgroup,
            workgroups_per_replay,
            concurrent_replays,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn rdna2_wave32(
        waves_per_workgroup: u32,
        workgroups_per_replay: u32,
        concurrent_replays: u32,
    ) -> Result<Self, WaveGangError> {
        Self::new(
            32,
            waves_per_workgroup,
            workgroups_per_replay,
            concurrent_replays,
        )
    }

    pub fn validate(self) -> Result<(), WaveGangError> {
        if self.native_wave_size == 0 {
            return Err(WaveGangError::InvalidNativeWaveSize(0));
        }
        if self.waves_per_workgroup == 0 {
            return Err(WaveGangError::InvalidWavesPerWorkgroup(0));
        }
        if self.workgroups_per_replay == 0 {
            return Err(WaveGangError::InvalidWorkgroupsPerReplay(0));
        }
        if self.concurrent_replays == 0 {
            return Err(WaveGangError::InvalidConcurrentReplays(0));
        }
        let threads = self
            .native_wave_size
            .checked_mul(self.waves_per_workgroup)
            .ok_or(WaveGangError::ArithmeticOverflow)?;
        if threads > MAX_WORKGROUP_THREADS {
            return Err(WaveGangError::WorkgroupTooLarge {
                threads,
                maximum: MAX_WORKGROUP_THREADS,
            });
        }
        self.waves_per_replay()?;
        self.peak_in_flight_waves()?;
        Ok(())
    }

    pub fn threads_per_workgroup(self) -> Result<u32, WaveGangError> {
        self.native_wave_size
            .checked_mul(self.waves_per_workgroup)
            .ok_or(WaveGangError::ArithmeticOverflow)
    }

    pub fn waves_per_replay(self) -> Result<u64, WaveGangError> {
        u64::from(self.waves_per_workgroup)
            .checked_mul(u64::from(self.workgroups_per_replay))
            .ok_or(WaveGangError::ArithmeticOverflow)
    }

    pub fn peak_in_flight_waves(self) -> Result<u64, WaveGangError> {
        self.waves_per_replay()?
            .checked_mul(u64::from(self.concurrent_replays))
            .ok_or(WaveGangError::ArithmeticOverflow)
    }

    pub fn peak_in_flight_threads(self) -> Result<u64, WaveGangError> {
        u64::from(self.threads_per_workgroup()?)
            .checked_mul(u64::from(self.workgroups_per_replay))
            .and_then(|value| value.checked_mul(u64::from(self.concurrent_replays)))
            .ok_or(WaveGangError::ArithmeticOverflow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveGangReplaySnapshot {
    pub capacity: u32,
    pub in_flight: u32,
    pub waiters: u32,
    pub peak_in_flight: u32,
    pub total_admissions: u64,
}

#[derive(Debug)]
struct GateState {
    in_flight: u32,
    waiters: u32,
    peak_in_flight: u32,
    total_admissions: u64,
}

#[derive(Debug)]
pub struct ConcurrentWaveGangGate {
    plan: ConcurrentWaveGangPlan,
    state: Mutex<GateState>,
    available: Condvar,
}

impl ConcurrentWaveGangGate {
    pub fn new(plan: ConcurrentWaveGangPlan) -> Result<Self, WaveGangError> {
        plan.validate()?;
        Ok(Self {
            plan,
            state: Mutex::new(GateState {
                in_flight: 0,
                waiters: 0,
                peak_in_flight: 0,
                total_admissions: 0,
            }),
            available: Condvar::new(),
        })
    }

    pub fn plan(&self) -> ConcurrentWaveGangPlan {
        self.plan
    }

    pub fn acquire(&self) -> WaveGangLease<'_> {
        let mut state = self.state.lock();
        state.waiters = state.waiters.saturating_add(1);
        while state.in_flight >= self.plan.concurrent_replays {
            self.available.wait(&mut state);
        }
        state.waiters = state.waiters.saturating_sub(1);
        state.in_flight += 1;
        state.peak_in_flight = state.peak_in_flight.max(state.in_flight);
        state.total_admissions = state.total_admissions.saturating_add(1);
        drop(state);
        WaveGangLease {
            gate: self,
            released: false,
        }
    }

    pub fn try_acquire(&self) -> Option<WaveGangLease<'_>> {
        let mut state = self.state.lock();
        if state.in_flight >= self.plan.concurrent_replays {
            return None;
        }
        state.in_flight += 1;
        state.peak_in_flight = state.peak_in_flight.max(state.in_flight);
        state.total_admissions = state.total_admissions.saturating_add(1);
        drop(state);
        Some(WaveGangLease {
            gate: self,
            released: false,
        })
    }

    pub fn acquire_replay(&self) -> ConcurrentWaveReplayLease<'_> {
        // Global order is replay-gate -> wave-gang slot. Acquiring the gang first
        // can deadlock behind a queued capture writer while occupying capacity
        // that an existing replay reader needs in order to drain.
        lock_replay();
        let gang = self.acquire();
        ConcurrentWaveReplayLease {
            gang: Some(gang),
            replay_locked: true,
        }
    }

    pub fn snapshot(&self) -> WaveGangReplaySnapshot {
        let state = self.state.lock();
        WaveGangReplaySnapshot {
            capacity: self.plan.concurrent_replays,
            in_flight: state.in_flight,
            waiters: state.waiters,
            peak_in_flight: state.peak_in_flight,
            total_admissions: state.total_admissions,
        }
    }

    fn release(&self) {
        let mut state = self.state.lock();
        debug_assert!(state.in_flight > 0);
        state.in_flight = state.in_flight.saturating_sub(1);
        self.available.notify_one();
    }
}

#[derive(Debug)]
pub struct WaveGangLease<'a> {
    gate: &'a ConcurrentWaveGangGate,
    released: bool,
}

impl WaveGangLease<'_> {
    pub fn release(mut self) {
        if !self.released {
            self.gate.release();
            self.released = true;
        }
    }
}

impl Drop for WaveGangLease<'_> {
    fn drop(&mut self) {
        if !self.released {
            self.gate.release();
            self.released = true;
        }
    }
}

#[derive(Debug)]
pub struct ConcurrentWaveReplayLease<'a> {
    gang: Option<WaveGangLease<'a>>,
    replay_locked: bool,
}

impl Drop for ConcurrentWaveReplayLease<'_> {
    fn drop(&mut self) {
        // Reverse acquisition order: release the gang slot before replay gate.
        self.gang.take();
        if self.replay_locked {
            unlock_replay();
            self.replay_locked = false;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveGangError {
    InvalidNativeWaveSize(u32),
    InvalidWavesPerWorkgroup(u32),
    InvalidWorkgroupsPerReplay(u32),
    InvalidConcurrentReplays(u32),
    InvalidTotalWorkgroups(u64),
    InvalidWaveBudget(u64),
    ReplayExceedsWaveBudget { waves_per_replay: u64, maximum: u64 },
    WorkgroupTooLarge { threads: u32, maximum: u32 },
    ArithmeticOverflow,
}

impl std::fmt::Display for WaveGangError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNativeWaveSize(value) => {
                write!(formatter, "invalid native wave size {value}")
            }
            Self::InvalidWavesPerWorkgroup(value) => {
                write!(formatter, "invalid waves per workgroup {value}")
            }
            Self::InvalidWorkgroupsPerReplay(value) => {
                write!(formatter, "invalid workgroups per replay {value}")
            }
            Self::InvalidConcurrentReplays(value) => {
                write!(formatter, "invalid concurrent replay count {value}")
            }
            Self::InvalidTotalWorkgroups(value) => {
                write!(formatter, "invalid total workgroup count {value}")
            }
            Self::InvalidWaveBudget(value) => write!(formatter, "invalid wave budget {value}"),
            Self::ReplayExceedsWaveBudget {
                waves_per_replay,
                maximum,
            } => write!(
                formatter,
                "one replay needs {waves_per_replay} waves, exceeding wave budget {maximum}"
            ),
            Self::WorkgroupTooLarge { threads, maximum } => write!(
                formatter,
                "workgroup requires {threads} threads, exceeding maximum {maximum}"
            ),
            Self::ArithmeticOverflow => formatter.write_str("wave-gang arithmetic overflow"),
        }
    }
}

impl std::error::Error for WaveGangError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn rdna2_plan_multiplies_kernel_owned_wave_gangs() {
        let plan = ConcurrentWaveGangPlan::rdna2_wave32(16, 80, 10).unwrap();
        assert_eq!(plan.threads_per_workgroup().unwrap(), 512);
        assert_eq!(plan.waves_per_replay().unwrap(), 1_280);
        assert_eq!(plan.peak_in_flight_waves().unwrap(), 12_800);
        assert_eq!(plan.peak_in_flight_threads().unwrap(), 409_600);
    }

    #[test]
    fn rejects_workgroups_larger_than_hardware_contract() {
        let error = ConcurrentWaveGangPlan::rdna2_wave32(33, 1, 1).unwrap_err();
        assert_eq!(
            error,
            WaveGangError::WorkgroupTooLarge {
                threads: 1_056,
                maximum: 1_024,
            }
        );
    }

    #[test]
    fn scaled_wave32_plan_tiles_massive_work_without_widening_the_wave() {
        let policy = WaveGangScalePolicy {
            max_workgroups_per_replay: 256,
            max_concurrent_replays: 8,
            max_in_flight_waves: 4096,
        };
        let plan = ScaledWaveGangPlan::rdna2_wave32(4, 1_000_000, policy).unwrap();
        assert_eq!(plan.replay.native_wave_size, 32);
        assert_eq!(plan.replay.workgroups_per_replay, 256);
        assert_eq!(plan.replay.waves_per_replay().unwrap(), 1024);
        assert_eq!(plan.replay.concurrent_replays, 4);
        assert_eq!(plan.replay.peak_in_flight_waves().unwrap(), 4096);
        assert_eq!(plan.replay_tiles, 3907);
        assert_eq!(plan.tail_workgroups, 64);
    }

    #[test]
    fn scaled_plan_rejects_one_replay_larger_than_wave_budget() {
        let policy = WaveGangScalePolicy {
            max_workgroups_per_replay: 64,
            max_concurrent_replays: 8,
            max_in_flight_waves: 127,
        };
        assert_eq!(
            ScaledWaveGangPlan::rdna2_wave32(2, 4096, policy).unwrap_err(),
            WaveGangError::ReplayExceedsWaveBudget {
                waves_per_replay: 128,
                maximum: 127,
            }
        );
    }

    #[test]
    fn gate_allows_bounded_concurrent_replays() {
        let plan = ConcurrentWaveGangPlan::rdna2_wave32(16, 1, 4).unwrap();
        let gate = Arc::new(ConcurrentWaveGangGate::new(plan).unwrap());
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let inside = Arc::new(Barrier::new(4));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let gate = Arc::clone(&gate);
            let active = Arc::clone(&active);
            let maximum = Arc::clone(&maximum);
            let inside = Arc::clone(&inside);
            handles.push(thread::spawn(move || {
                let _lease = gate.acquire();
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum.fetch_max(current, Ordering::SeqCst);
                inside.wait();
                thread::sleep(Duration::from_millis(2));
                active.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let snapshot = gate.snapshot();
        assert_eq!(maximum.load(Ordering::SeqCst), 4);
        assert_eq!(snapshot.peak_in_flight, 4);
        assert_eq!(snapshot.in_flight, 0);
        assert_eq!(snapshot.total_admissions, 4);
    }

    #[test]
    fn try_acquire_never_exceeds_capacity() {
        let plan = ConcurrentWaveGangPlan::rdna2_wave32(8, 1, 2).unwrap();
        let gate = ConcurrentWaveGangGate::new(plan).unwrap();
        let first = gate.try_acquire().unwrap();
        let second = gate.try_acquire().unwrap();
        assert!(gate.try_acquire().is_none());
        drop(first);
        assert!(gate.try_acquire().is_some());
        drop(second);
    }
}
