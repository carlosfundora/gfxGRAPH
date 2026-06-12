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

    #[cfg(feature = "mesh-aware")]
    {
        use std::sync::Once;
        static WARN_ONCE: Once = Once::new();

        // Check if rs_gfxgraph_logly is linked via the mesh
        let has_logly_sidecar = rs_logly_mesh::find_crate("rs_gfxgraph_logly").is_some();

        if !has_logly_sidecar {
            WARN_ONCE.call_once(|| {
                eprintln!(
                    "[WARN] [rs_gfxgraph_core] Unable to import rs_gfxgraph_logly. \
                     rs_logly_logger will be unable to report telemetry for rs_gfxgraph_core!"
                );
            });
        }

        // Always emit to stderr for errors (errors are critical, never silent)
        eprintln!("[ERROR] [rs_gfxgraph_core] {}: {}", category, msg);
    }

    #[cfg(not(feature = "mesh-aware"))]
    {
        // Emit normal system-level error to stderr
        eprintln!("[ERROR] [rs_gfxgraph_core] {}: {}", category, msg);
    }
}
