use rs_policy_mesh::{
    ActorKind, CrateLockState, CratePolicyContract, CriticalityLevel, MethodPolicyContract,
    PermissionMode, PolicyZone, PolicyZoneType, RiskLevel, ZoneEnforcementMode,
};
use std::collections::{BTreeMap, BTreeSet};

pub const CRATE_ID: &str = "rs_gfxgraph_core";
pub const POLICY_ID: &str = "policy.rs_gfxgraph_core.v1";
pub const MESH_KEY_ID: &str = "mesh-root-2026";
pub const EMBEDDED_MESH_PUBLIC_KEY_B64: &str = "ao7WdvfNLPcMhAAkRQSNFphDNtqcdvB6SKYnmjANEoQ=";

pub fn crate_policy_contract() -> CratePolicyContract {
    let zone = PolicyZone::new(
        "development",
        PolicyZoneType::Custom,
        ZoneEnforcementMode::RequireApproval,
    );
    let mut methods = BTreeMap::new();
    methods.insert(
        "health".to_string(),
        MethodPolicyContract {
            method_id: "health".to_string(),
            capability: format!("{CRATE_ID}.health"),
            policy_zone: zone.clone(),
            allowed_modes: BTreeSet::from([PermissionMode::Allow, PermissionMode::Test]),
            required_scopes: BTreeSet::from([format!("{CRATE_ID}:read")]),
            zones_allowed: BTreeSet::from([zone.clone()]),
            zones_denied: BTreeSet::new(),
            allowed_actor_kinds: BTreeSet::from([
                ActorKind::Agent,
                ActorKind::HumanUser,
                ActorKind::NetworkService,
                ActorKind::SystemDaemon,
            ]),
            risk_level: RiskLevel::Low,
            criticality: CriticalityLevel::Soft,
            trust_level: 0.9,
            mutates_state: false,
            network_required: false,
            filesystem_required: false,
            input_schema_hash: None,
            output_schema_hash: None,
            conditions: Vec::new(),
            max_payload_bytes: 1024,
            timeout_ms: 1_000,
            idempotent: true,
        },
    );

    CratePolicyContract {
        policy_id: POLICY_ID.to_string(),
        crate_id: CRATE_ID.to_string(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        contract_version: "1".to_string(),
        policy_zone: zone,
        contract_hash: "blake3:0eb65e83da15ca54410f60ada5ee02fc440409b5fc1a3fc9a105963e0e5e0ca5"
            .to_string(),
        signer_key_id: MESH_KEY_ID.to_string(),
        signature_b64: "embedded-contract-signature-pending".to_string(),
        default_state: CrateLockState::ReadOnly,
        methods,
    }
}
