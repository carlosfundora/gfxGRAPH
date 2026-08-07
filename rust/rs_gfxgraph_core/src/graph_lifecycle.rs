// SPDX-License-Identifier: //! Generic HIP graph lifecycle and update contracts.
//!
//! These types describe graph-safe execution topology without knowing the
//! workload inhabiting the graph. They are derived from gfxGRAPH's native
//! shape-pool, persistent-metadata, kernel-node update, and child-graph
//! composition facilities.

use std::collections::HashSet;

/// Generic retry policy for one-time graph capture/instantiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphCapturePolicy {
    pub create_attempts: usize,
}

impl Default for GraphCapturePolicy {
    fn default() -> Self {
        Self { create_attempts: 3 }
    }
}

impl GraphCapturePolicy {
    pub fn validate(self) -> Result<Self, GraphLifecycleError> {
        if self.create_attempts == 0 {
            return Err(GraphLifecycleError::ZeroCaptureAttempts);
        }
        Ok(self)
    }
}

/// How replay-visible state changes after a graph has been captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphUpdateMode {
    /// Pointer arguments remain fixed; only persistent buffer contents change.
    PersistentBufferRefresh,
    /// A captured kernel node receives new launch parameters in place.
    KernelNodeParams,
    /// One child graph in a composed executable is replaced.
    ChildGraphReplacement,
}

/// One fixed-address metadata region read by captured graph nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistentMetadataRegion {
    pub element_bytes: usize,
    pub capacity_elements: usize,
    pub alignment_bytes: usize,
}

impl PersistentMetadataRegion {
    pub fn byte_len(self) -> Result<usize, GraphLifecycleError> {
        if self.element_bytes == 0 || self.capacity_elements == 0 {
            return Err(GraphLifecycleError::EmptyMetadataRegion);
        }
        if self.alignment_bytes == 0 || !self.alignment_bytes.is_power_of_two() {
            return Err(GraphLifecycleError::InvalidAlignment(self.alignment_bytes));
        }
        self.element_bytes
            .checked_mul(self.capacity_elements)
            .ok_or(GraphLifecycleError::SizeOverflow)
    }
}

/// Fixed-address metadata layout shared by all captures in a graph family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureStableMetadataPlan {
    pub regions: Vec<PersistentMetadataRegion>,
}

impl CaptureStableMetadataPlan {
    pub fn total_bytes(&self) -> Result<usize, GraphLifecycleError> {
        if self.regions.is_empty() {
            return Err(GraphLifecycleError::NoMetadataRegions);
        }
        self.regions.iter().try_fold(0usize, |total, region| {
            total
                .checked_add(region.byte_len()?)
                .ok_or(GraphLifecycleError::SizeOverflow)
        })
    }
}

/// Parent relation for a composed graph. `None` denotes a root child graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphCompositionPlan {
    pub parents: Vec<Option<usize>>,
}

impl GraphCompositionPlan {
    pub fn validate(&self) -> Result<(), GraphLifecycleError> {
        if self.parents.is_empty() {
            return Err(GraphLifecycleError::EmptyComposition);
        }
        for (child, parent) in self.parents.iter().copied().enumerate() {
            if parent == Some(child) {
                return Err(GraphLifecycleError::SelfDependency(child));
            }
            match parent {
                Some(parent) if parent >= self.parents.len() => {
                    return Err(GraphLifecycleError::DependencyOutOfBounds { child, parent });
                }
                _ => {}
            }
            let mut cursor = parent;
            let mut seen = HashSet::new();
            while let Some(node) = cursor {
                if !seen.insert(node) {
                    return Err(GraphLifecycleError::DependencyCycle(child));
                }
                cursor = self.parents[node];
            }
        }
        Ok(())
    }

    pub fn roots(&self) -> impl Iterator<Item = usize> + '_ {
        self.parents
            .iter()
            .enumerate()
            .filter_map(|(index, parent)| parent.is_none().then_some(index))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphLifecycleError {
    ZeroCaptureAttempts,
    EmptyMetadataRegion,
    InvalidAlignment(usize),
    SizeOverflow,
    NoMetadataRegions,
    EmptyComposition,
    SelfDependency(usize),
    DependencyOutOfBounds { child: usize, parent: usize },
    DependencyCycle(usize),
}

impl core::fmt::Display for GraphLifecycleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroCaptureAttempts => write!(f, "graph capture attempts must be nonzero"),
            Self::EmptyMetadataRegion => write!(
                f,
                "persistent metadata region must contain at least one nonzero-sized element"
            ),
            Self::InvalidAlignment(value) => write!(
                f,
                "persistent metadata alignment must be a nonzero power of two, got {value}"
            ),
            Self::SizeOverflow => write!(f, "persistent metadata size overflow"),
            Self::NoMetadataRegions => write!(
                f,
                "capture-stable metadata plan must contain at least one region"
            ),
            Self::EmptyComposition => write!(f, "composed graph must contain at least one child"),
            Self::SelfDependency(child) => write!(f, "graph child {child} cannot depend on itself"),
            Self::DependencyOutOfBounds { child, parent } => write!(
                f,
                "graph child {child} depends on out-of-range child {parent}"
            ),
            Self::DependencyCycle(child) => {
                write!(f, "graph child {child} participates in a dependency cycle")
            }
        }
    }
}

impl std::error::Error for GraphLifecycleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_metadata_plan_counts_fixed_address_storage() {
        let plan = CaptureStableMetadataPlan {
            regions: vec![
                PersistentMetadataRegion {
                    element_bytes: 4,
                    capacity_elements: 64,
                    alignment_bytes: 64,
                },
                PersistentMetadataRegion {
                    element_bytes: 8,
                    capacity_elements: 32,
                    alignment_bytes: 64,
                },
            ],
        };
        assert_eq!(plan.total_bytes().unwrap(), 512);
    }

    #[test]
    fn composition_rejects_cycles_and_accepts_forest() {
        let good = GraphCompositionPlan {
            parents: vec![None, Some(0), Some(1), None],
        };
        good.validate().unwrap();
        assert_eq!(good.roots().collect::<Vec<_>>(), vec![0, 3]);

        let bad = GraphCompositionPlan {
            parents: vec![Some(1), Some(0)],
        };
        assert!(matches!(
            bad.validate(),
            Err(GraphLifecycleError::DependencyCycle(_))
        ));
    }
}
