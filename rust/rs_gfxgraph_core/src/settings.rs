#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CrateSettings {
    pub crate_id: String,
    pub policy_id: String,
    pub policy_zone: String,
    pub logly_enabled: bool,
    pub logly_crate_id: Option<String>,
    pub pyo3_binding_enabled: bool,
    pub pyo3_crate_id: Option<String>,
    pub python_module_name: Option<String>,
    pub node_binding_enabled: bool,
    pub node_crate_id: Option<String>,
    pub node_module_name: Option<String>,
}

impl Default for CrateSettings {
    fn default() -> Self {
        Self {
            crate_id: "rs_gfxgraph_core".to_string(),
            policy_id: "policy.rs_gfxgraph_core.v1".to_string(),
            policy_zone: "development".to_string(),
            logly_enabled: true,
            logly_crate_id: Some("rs_gfxgraph_core_logly".to_string()),
            pyo3_binding_enabled: false,
            pyo3_crate_id: Some("rs_gfxgraph_core_pyo3".to_string()),
            python_module_name: Some("gfxgraph_core".to_string()),
            node_binding_enabled: false,
            node_crate_id: Some("rs_gfxgraph_core_node".to_string()),
            node_module_name: Some("rs-gfxgraph-core".to_string()),
        }
    }
}
