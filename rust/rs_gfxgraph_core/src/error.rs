use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GfxGraphError {
    InvalidNode { name: String, reason: String },
    StatsCollectionFailed { reason: String },
    ExecutionError { branch: String, reason: String },
    Generic(String),
}

impl fmt::Display for GfxGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GfxGraphError::InvalidNode { name, reason } => {
                write!(f, "Invalid node spec '{}': {}", name, reason)
            }
            GfxGraphError::StatsCollectionFailed { reason } => {
                write!(f, "Stats collection failed: {}", reason)
            }
            GfxGraphError::ExecutionError { branch, reason } => {
                write!(f, "Execution error on branch '{}': {}", branch, reason)
            }
            GfxGraphError::Generic(msg) => write!(f, "GfxGraph core error: {}", msg),
        }
    }
}

impl std::error::Error for GfxGraphError {}

/// Report an error using the logly mesh if a `_logly` sidecar is available,
/// or fall back to system-level stderr.
pub fn report_error(error: &dyn std::error::Error, category: &str) {
    let msg = error.to_string();

    // Emit normal system-level error to stderr. Mesh-specific reporting is kept
    // out of this portable companion crate until the mesh crates are present.
    eprintln!("[ERROR] [rs_gfxgraph_core] {}: {}", category, msg);
}
