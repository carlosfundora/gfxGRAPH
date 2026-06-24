use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use rs_gfxgraph_core::{BucketRouterCore, GfxGraphStatsSample};
use rs_gfxgraph_native::{default_library_candidates, NativeBridge, ProfileEventKind};
use serde::Serialize;
use serde_json::{json, Value};

const SCHEMA_VERSION: &str = "gfxgraph-benchmark-v1";
const REPORT_KIND: &str = "benchmark_gate_phase_report";
const SCHEMA_RELATIVE_PATH: &str = "benchmarks/schemas/gfxgraph-benchmark-v1.schema.json";

#[derive(Parser, Debug)]
#[command(
    name = "gfxgraph-bench-gate",
    about = "Rust-owned gfxGRAPH benchmark gate for installed and candidate package phases"
)]
struct Cli {
    #[arg(long)]
    repo_root: Option<PathBuf>,

    #[arg(long)]
    output_root: Option<PathBuf>,

    #[arg(long)]
    python: Option<PathBuf>,

    #[arg(long = "phase", value_enum)]
    phases: Vec<Phase>,

    #[arg(long)]
    matrix: bool,

    #[arg(long)]
    allow_package_changes: bool,

    #[arg(long)]
    no_enforce_import_isolation: bool,

    #[arg(long)]
    skip_public: bool,

    #[arg(long)]
    skip_python_micro: bool,

    #[arg(long)]
    include_python_micro: bool,

    #[arg(long)]
    skip_hip: bool,

    #[arg(long)]
    native_only: bool,

    #[arg(long, default_value_t = 3)]
    public_run_count: u32,

    #[arg(long, default_value_t = 200_000)]
    rust_iterations: u64,

    #[arg(long)]
    run_id: Option<String>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Phase {
    InstalledBaseline,
    InstalledGraphEnabled,
    CleanCandidate,
    GraphCandidate,
}

impl Phase {
    fn as_str(self) -> &'static str {
        match self {
            Self::InstalledBaseline => "installed-baseline",
            Self::InstalledGraphEnabled => "installed-graph-enabled",
            Self::CleanCandidate => "clean-candidate",
            Self::GraphCandidate => "graph-candidate",
        }
    }

    fn graph_enabled(self) -> bool {
        matches!(self, Self::InstalledGraphEnabled | Self::GraphCandidate)
    }

    fn candidate(self) -> bool {
        matches!(self, Self::CleanCandidate | Self::GraphCandidate)
    }

    fn target_import_policy(self, enforce: bool) -> &'static str {
        if !enforce {
            "not-enforced"
        } else if self.candidate() {
            "repo"
        } else {
            "site-packages"
        }
    }
}

#[derive(Serialize)]
struct BenchmarkReport {
    schema_version: &'static str,
    report_kind: &'static str,
    run_id: String,
    phase: &'static str,
    timestamp_utc: String,
    report_date: String,
    schema_path: String,
    repo: RepoProvenance,
    environment: EnvironmentProvenance,
    python: Option<Value>,
    package_state: PackageState,
    gate_checks: Vec<GateCheck>,
    benchmarks: Vec<BenchmarkRecord>,
}

#[derive(Serialize)]
struct RepoProvenance {
    root: String,
    branch: String,
    commit: String,
    tracked_dirty: bool,
}

#[derive(Serialize)]
struct EnvironmentProvenance {
    os: String,
    rocm_path: Option<String>,
    hipcc_version: Option<String>,
    env: BTreeMap<String, Option<String>>,
}

#[derive(Serialize)]
struct PackageState {
    target_import_policy: &'static str,
    graph_enabled: bool,
    allow_package_changes: bool,
    package_actions: Vec<CommandRecord>,
}

#[derive(Serialize)]
struct CommandRecord {
    name: String,
    command: String,
    status: &'static str,
    duration_ms: f64,
    exit_code: Option<i32>,
    stdout_tail: String,
    stderr_tail: String,
}

#[derive(Serialize)]
struct GateCheck {
    name: String,
    status: &'static str,
    message: String,
    details: Value,
}

#[derive(Serialize)]
struct BenchmarkRecord {
    name: String,
    kind: String,
    status: &'static str,
    command: String,
    iterations: Option<u64>,
    duration_ms: Option<f64>,
    throughput_ops_per_sec: Option<f64>,
    stdout_tail: String,
    stderr_tail: String,
    metrics: BTreeMap<String, Value>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("gfxgraph-bench-gate: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let repo_root = cli
        .repo_root
        .clone()
        .unwrap_or_else(|| PathBuf::from("."))
        .canonicalize()?;
    let python_candidate = cli
        .python
        .clone()
        .unwrap_or_else(|| default_python(&repo_root));
    let python = absolute_path(&repo_root, python_candidate);
    let output_root = cli
        .output_root
        .clone()
        .unwrap_or_else(|| repo_root.join("benchmarks/results"));
    let clock = UtcClock::read();
    let run_id = cli.run_id.clone().unwrap_or_else(|| {
        clock
            .timestamp_utc
            .replace(['-', ':'], "")
            .replace('Z', "Z")
    });
    let phases = resolve_phases(&cli);
    let output_dir = output_root.join(&clock.report_date);
    fs::create_dir_all(&output_dir)?;

    let mut any_failed = false;
    for phase in phases {
        let report = run_phase(&cli, &repo_root, &python, phase, &clock, &run_id)?;
        validate_report(&report)?;
        let report_path =
            output_dir.join(format!("{SCHEMA_VERSION}-{}-{run_id}.json", phase.as_str()));
        let json = serde_json::to_string_pretty(&report)?;
        fs::write(&report_path, format!("{json}\n"))?;
        println!("wrote {}", report_path.display());

        any_failed |= report
            .gate_checks
            .iter()
            .any(|check| check.status == "failed");
        any_failed |= report
            .benchmarks
            .iter()
            .any(|benchmark| benchmark.status == "failed");
    }

    if any_failed {
        std::process::exit(1);
    }
    Ok(())
}

fn resolve_phases(cli: &Cli) -> Vec<Phase> {
    if !cli.phases.is_empty() {
        return cli.phases.clone();
    }
    if cli.matrix {
        return vec![
            Phase::InstalledBaseline,
            Phase::InstalledGraphEnabled,
            Phase::CleanCandidate,
            Phase::GraphCandidate,
        ];
    }
    vec![Phase::InstalledBaseline, Phase::InstalledGraphEnabled]
}

fn run_phase(
    cli: &Cli,
    repo_root: &Path,
    python: &Path,
    phase: Phase,
    clock: &UtcClock,
    run_id: &str,
) -> Result<BenchmarkReport, Box<dyn Error>> {
    let enforce_import_isolation = !cli.no_enforce_import_isolation;
    let candidate_transition_allowed = !phase.candidate() || cli.allow_package_changes;
    let effective_import_enforcement = enforce_import_isolation && candidate_transition_allowed;
    let mut package_actions = Vec::new();
    let mut gate_checks = Vec::new();
    let mut benchmarks = Vec::new();
    let phase_env = phase_env(phase);

    let mut python_probe = None;

    if cli.native_only {
        gate_checks.push(GateCheck {
            name: "native-runtime-only".to_string(),
            status: "passed",
            message: "native-only gate skips Python package transition and import provenance"
                .to_string(),
            details: json!({
                "phase": phase.as_str(),
                "python": Value::Null,
            }),
        });
    } else if phase.candidate() && !cli.allow_package_changes {
        gate_checks.push(GateCheck {
            name: "candidate-package-transition".to_string(),
            status: "skipped",
            message: "candidate phase requires --allow-package-changes before UV uninstall/install"
                .to_string(),
            details: json!({
                "phase": phase.as_str(),
                "python": python.display().to_string()
            }),
        });
        benchmarks.push(skipped_benchmark(
            "phase-benchmarks",
            "phase-control",
            "candidate package transition was not allowed",
        ));
    } else if phase.candidate() {
        package_actions.extend(prepare_candidate_packages(repo_root, python));
        gate_checks.push(check_candidate_package_actions(&package_actions));
    }

    if cli.native_only {
        benchmarks.push(skipped_benchmark(
            "python-import-provenance",
            "package-provenance",
            "skipped by --native-only",
        ));
    } else {
        let (probe, python_command) = collect_python_provenance(python, repo_root, &phase_env);
        python_probe = probe;
        benchmarks.push(command_as_benchmark(
            "python-import-provenance",
            "package-provenance",
            python_command,
        ));
        gate_checks.push(check_python_import_policy(
            phase,
            repo_root,
            python_probe.as_ref(),
            effective_import_enforcement,
        ));
    }

    if !phase.candidate() || cli.allow_package_changes || cli.native_only {
        if cli.native_only {
            benchmarks.push(skipped_benchmark(
                "torch-capture-policy",
                "capture-policy",
                "skipped by --native-only",
            ));
        } else {
            benchmarks.push(run_capture_policy_probe(repo_root, python, &phase_env));
        }
        benchmarks.push(bench_rust_router(cli.rust_iterations));
        benchmarks.push(bench_rust_stats(cli.rust_iterations));

        if cli.skip_hip {
            benchmarks.push(skipped_benchmark(
                "native-hip-benchmark",
                "hip-native",
                "skipped by --skip-hip",
            ));
            benchmarks.push(skipped_benchmark(
                "native-hip-tests",
                "hip-native",
                "skipped by --skip-hip",
            ));
        } else {
            benchmarks.push(run_native_runtime_cli(repo_root));
            benchmarks.push(run_native_runtime_ffi(repo_root));
            benchmarks.push(run_hip_benchmark(repo_root));
            benchmarks.push(run_ctest(repo_root));
        }

        if cli.native_only {
            benchmarks.push(skipped_benchmark(
                "readme-public-benchmark",
                "python-public",
                "skipped by --native-only",
            ));
        } else if cli.skip_public {
            benchmarks.push(skipped_benchmark(
                "readme-public-benchmark",
                "python-public",
                "skipped by --skip-public",
            ));
        } else {
            let public_record = run_python_script(
                "readme-public-benchmark",
                "python-public",
                repo_root,
                python,
                "benchmarks/bench_readme_public.py",
                &[
                    "--output".to_string(),
                    format!(
                        "benchmarks/results/{}/readme-public-{}-{run_id}.json",
                        clock.report_date,
                        phase.as_str()
                    ),
                    "--run-count".to_string(),
                    cli.public_run_count.to_string(),
                ],
                &phase_env,
            );
            benchmarks.push(public_benchmark_record(public_record));
        }

        if cli.native_only {
            benchmarks.push(skipped_benchmark(
                "legacy-python-microbenchmarks",
                "python-micro",
                "skipped by --native-only",
            ));
        } else if cli.skip_python_micro || !cli.include_python_micro {
            benchmarks.push(skipped_benchmark(
                "legacy-python-microbenchmarks",
                "python-micro",
                "skipped; pass --include-python-micro to run legacy Python microbenchmarks",
            ));
        } else {
            for script in [
                "benchmarks/bench_routing.py",
                "benchmarks/bench_routing_rust.py",
                "benchmarks/bench_stats.py",
                "benchmarks/bench_stats_rust.py",
            ] {
                benchmarks.push(run_python_script(
                    script,
                    "python-micro",
                    repo_root,
                    python,
                    script,
                    &[],
                    &phase_env,
                ));
            }
        }
    }

    Ok(BenchmarkReport {
        schema_version: SCHEMA_VERSION,
        report_kind: REPORT_KIND,
        run_id: run_id.to_string(),
        phase: phase.as_str(),
        timestamp_utc: clock.timestamp_utc.clone(),
        report_date: clock.report_date.clone(),
        schema_path: SCHEMA_RELATIVE_PATH.to_string(),
        repo: repo_provenance(repo_root),
        environment: environment_provenance(),
        python: python_probe,
        package_state: PackageState {
            target_import_policy: if cli.native_only {
                "native-only"
            } else {
                phase.target_import_policy(effective_import_enforcement)
            },
            graph_enabled: phase.graph_enabled(),
            allow_package_changes: cli.allow_package_changes,
            package_actions,
        },
        gate_checks,
        benchmarks,
    })
}

fn prepare_candidate_packages(repo_root: &Path, python: &Path) -> Vec<CommandRecord> {
    let python = python.display().to_string();
    let repo = repo_root.display().to_string();
    let native = repo_root.join("native").display().to_string();
    let rs_gfxgraph_dir = repo_root.join("rust/rs_gfxgraph");
    let rs_gfxgraph_stats_dir = repo_root.join("rust/rs_gfxgraph_stats");
    let maturin_env = maturin_env(Path::new(&python));

    vec![
        run_command(
            "uv-uninstall-installed-python-packages",
            "uv",
            &[
                "pip".to_string(),
                "uninstall".to_string(),
                "--python".to_string(),
                python.clone(),
                "gfxgraph".to_string(),
                "gfxgraph-native".to_string(),
                "rs-gfxgraph".to_string(),
                "rs-gfxgraph-stats".to_string(),
            ],
            Some(repo_root),
            &[],
        ),
        run_command(
            "uv-install-candidate-gfxgraph",
            "uv",
            &[
                "pip".to_string(),
                "install".to_string(),
                "--python".to_string(),
                python.clone(),
                "--no-deps".to_string(),
                "-e".to_string(),
                repo,
            ],
            Some(repo_root),
            &[],
        ),
        run_command(
            "uv-install-candidate-gfxgraph-native",
            "uv",
            &[
                "pip".to_string(),
                "install".to_string(),
                "--python".to_string(),
                python,
                "--no-deps".to_string(),
                "-e".to_string(),
                native,
            ],
            Some(repo_root),
            &[],
        ),
        run_command(
            "maturin-develop-rs-gfxgraph-uv",
            "maturin",
            &[
                "develop".to_string(),
                "--uv".to_string(),
                "--release".to_string(),
            ],
            Some(&rs_gfxgraph_dir),
            &maturin_env,
        ),
        run_command(
            "maturin-develop-rs-gfxgraph-stats-uv",
            "maturin",
            &[
                "develop".to_string(),
                "--uv".to_string(),
                "--release".to_string(),
            ],
            Some(&rs_gfxgraph_stats_dir),
            &maturin_env,
        ),
    ]
}

fn check_candidate_package_actions(package_actions: &[CommandRecord]) -> GateCheck {
    let failed: Vec<_> = package_actions
        .iter()
        .filter(|action| action.status == "failed")
        .map(|action| action.name.as_str())
        .collect();
    let actions: Vec<_> = package_actions
        .iter()
        .map(|action| {
            json!({
                "name": action.name,
                "status": action.status,
                "exit_code": action.exit_code
            })
        })
        .collect();

    if failed.is_empty() {
        GateCheck {
            name: "candidate-package-transition".to_string(),
            status: "passed",
            message: "UV package uninstall/install transition completed".to_string(),
            details: json!({ "actions": actions }),
        }
    } else {
        GateCheck {
            name: "candidate-package-transition".to_string(),
            status: "failed",
            message: format!("UV package transition failed: {}", failed.join(", ")),
            details: json!({ "actions": actions, "failed": failed }),
        }
    }
}

fn collect_python_provenance(
    python: &Path,
    repo_root: &Path,
    envs: &[(String, String)],
) -> (Option<Value>, CommandRecord) {
    let script = r#"
import importlib.metadata as md
import importlib.util
import json
import sys

metadata_names = {
    "gfxgraph": "gfxgraph",
    "hipgraph_bridge": "gfxgraph",
    "gfxgraph_native": "gfxgraph-native",
    "rs_gfxgraph": "rs-gfxgraph",
    "rs_gfxgraph_stats": "rs-gfxgraph-stats",
}

payload = {
    "executable": sys.executable,
    "version": sys.version,
    "prefix": sys.prefix,
    "imports": [],
}

for module, distribution in metadata_names.items():
    item = {"name": module, "distribution": distribution}
    try:
        spec = importlib.util.find_spec(module)
        item["origin"] = None if spec is None else spec.origin
        locations = None if spec is None else spec.submodule_search_locations
        item["search_locations"] = None if locations is None else list(locations)
        item["importable"] = spec is not None
    except Exception as exc:
        item["importable"] = False
        item["origin_error"] = repr(exc)
    try:
        item["version"] = md.version(distribution)
        dist = md.distribution(distribution)
        item["distribution_location"] = str(dist.locate_file(""))
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            item["direct_url"] = json.loads(direct_url)
    except Exception as exc:
        item["version_error"] = repr(exc)
    payload["imports"].append(item)

print(json.dumps(payload, sort_keys=True))
"#;
    let python_display = python.display().to_string();
    let args = vec!["-c".to_string(), script.to_string()];
    let command_display =
        display_command(&python_display, &["-c".to_string(), "<probe>".to_string()]);
    let start = Instant::now();
    let mut command = Command::new(python);
    command
        .args(&args)
        .current_dir(std::env::temp_dir())
        .env_remove("PYTHONPATH")
        .env("GFXGRAPH_REPO_ROOT", repo_root);
    for (key, value) in envs {
        command.env(key, value);
    }

    let record = match command.output() {
        Ok(output) => {
            let status = if output.status.success() {
                "passed"
            } else {
                "failed"
            };
            CommandRecord {
                name: "python-import-provenance".to_string(),
                command: command_display,
                status,
                duration_ms: elapsed_ms(start.elapsed()),
                exit_code: output.status.code(),
                stdout_tail: tail_text(&String::from_utf8_lossy(&output.stdout), 16_384),
                stderr_tail: tail_text(&String::from_utf8_lossy(&output.stderr), 16_384),
            }
        }
        Err(error) => CommandRecord {
            name: "python-import-provenance".to_string(),
            command: command_display,
            status: "failed",
            duration_ms: elapsed_ms(start.elapsed()),
            exit_code: None,
            stdout_tail: String::new(),
            stderr_tail: error.to_string(),
        },
    };

    let parsed = if record.status == "passed" {
        serde_json::from_str(record.stdout_tail.trim()).ok()
    } else {
        None
    };
    (parsed, record)
}

fn run_capture_policy_probe(
    repo_root: &Path,
    python: &Path,
    envs: &[(String, String)],
) -> BenchmarkRecord {
    let script = r#"
import json
import os
import torch
from hipgraph_bridge.capture_safety import (
    torch_cuda_execution_error,
    torch_cuda_execution_usable,
    torch_graph_capture_block_reason,
    unsafe_torch_graph_capture_enabled,
)

print(json.dumps({
    "torch": torch.__version__,
    "torch_hip": getattr(torch.version, "hip", None),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cuda_available": torch.cuda.is_available(),
    "cuda_execution_usable": torch_cuda_execution_usable(),
    "cuda_execution_error": torch_cuda_execution_error(),
    "unsafe_torch_graph_capture_enabled": unsafe_torch_graph_capture_enabled(),
    "torch_graph_capture_block_reason": torch_graph_capture_block_reason(),
    "GFXGRAPH_ENABLE_UNSAFE_TORCH_GRAPH_CAPTURE": os.environ.get("GFXGRAPH_ENABLE_UNSAFE_TORCH_GRAPH_CAPTURE"),
}, sort_keys=True))
"#;
    let python_display = python.display().to_string();
    let command_display = display_command(
        &python_display,
        &["-c".to_string(), "<capture-policy-probe>".to_string()],
    );
    let start = Instant::now();
    let mut command = Command::new(python);
    command
        .arg("-c")
        .arg(script)
        .current_dir(std::env::temp_dir())
        .env_remove("PYTHONPATH")
        .env("GFXGRAPH_REPO_ROOT", repo_root);
    for (key, value) in envs {
        command.env(key, value);
    }

    let record = match command.output() {
        Ok(output) => CommandRecord {
            name: "torch-capture-policy".to_string(),
            command: command_display,
            status: if output.status.success() {
                "passed"
            } else {
                "failed"
            },
            duration_ms: elapsed_ms(start.elapsed()),
            exit_code: output.status.code(),
            stdout_tail: tail_text(&String::from_utf8_lossy(&output.stdout), 16_384),
            stderr_tail: tail_text(&String::from_utf8_lossy(&output.stderr), 16_384),
        },
        Err(error) => CommandRecord {
            name: "torch-capture-policy".to_string(),
            command: command_display,
            status: "failed",
            duration_ms: elapsed_ms(start.elapsed()),
            exit_code: None,
            stdout_tail: String::new(),
            stderr_tail: error.to_string(),
        },
    };

    let mut benchmark = command_as_benchmark("torch-capture-policy", "capture-policy", record);
    if let Ok(payload) = serde_json::from_str::<Value>(benchmark.stdout_tail.trim()) {
        benchmark.metrics.insert("probe".to_string(), payload);
    }
    benchmark
}

fn check_python_import_policy(
    phase: Phase,
    repo_root: &Path,
    python: Option<&Value>,
    enforce: bool,
) -> GateCheck {
    if !enforce {
        return GateCheck {
            name: "python-import-isolation".to_string(),
            status: "skipped",
            message: "import isolation enforcement disabled by --no-enforce-import-isolation"
                .to_string(),
            details: json!({ "phase": phase.as_str() }),
        };
    }

    let Some(python) = python else {
        return GateCheck {
            name: "python-import-isolation".to_string(),
            status: "failed",
            message: "python import provenance probe did not produce JSON".to_string(),
            details: json!({ "phase": phase.as_str() }),
        };
    };

    let required_modules = ["gfxgraph", "hipgraph_bridge", "gfxgraph_native"];
    let mut failures = Vec::new();
    let mut checked = Vec::new();
    let imports = python
        .get("imports")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    for module in required_modules {
        let import = imports.iter().find(|entry| {
            entry
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| name == module)
        });
        let Some(import) = import else {
            failures.push(format!("{module}: missing from provenance"));
            continue;
        };

        let importable = import
            .get("importable")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let origin = import.get("origin").and_then(Value::as_str).unwrap_or("");
        let in_repo = path_string_is_repo_source(origin, repo_root);
        checked.push(json!({
            "module": module,
            "origin": origin,
            "in_repo": in_repo,
            "importable": importable
        }));

        if !importable {
            failures.push(format!("{module}: not importable"));
        } else if phase.candidate() && !in_repo {
            failures.push(format!("{module}: expected repo import, got {origin}"));
        } else if !phase.candidate() && in_repo {
            failures.push(format!(
                "{module}: expected site-packages import, got {origin}"
            ));
        }
    }

    if failures.is_empty() {
        GateCheck {
            name: "python-import-isolation".to_string(),
            status: "passed",
            message: format!(
                "{} imports match {} policy",
                phase.as_str(),
                phase.target_import_policy(true)
            ),
            details: json!({ "checked": checked }),
        }
    } else {
        GateCheck {
            name: "python-import-isolation".to_string(),
            status: "failed",
            message: failures.join("; "),
            details: json!({ "checked": checked, "failures": failures }),
        }
    }
}

fn bench_rust_router(iterations: u64) -> BenchmarkRecord {
    let mut router = BucketRouterCore::new(vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512]);
    for bucket in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
        router.mark_warmed_up(bucket);
    }

    let start = Instant::now();
    let mut ready_count = 0_u64;
    for i in 0..iterations {
        let input = ((i as usize) % 512) + 1;
        let (_, state) = router
            .route(input)
            .expect("router input should fit buckets");
        if matches!(state, rs_gfxgraph_core::BucketState::Ready) {
            ready_count += 1;
        }
    }
    black_box(ready_count);
    let duration = start.elapsed();

    let mut metrics = BTreeMap::new();
    metrics.insert("ready_routes".to_string(), json!(ready_count));
    metrics.insert("bucket_count".to_string(), json!(router.buckets().len()));
    BenchmarkRecord {
        name: "rust-bucket-router-core".to_string(),
        kind: "rust-micro".to_string(),
        status: "passed",
        command: "in-process BucketRouterCore::route loop".to_string(),
        iterations: Some(iterations),
        duration_ms: Some(elapsed_ms(duration)),
        throughput_ops_per_sec: Some(throughput(iterations, duration)),
        stdout_tail: String::new(),
        stderr_tail: String::new(),
        metrics,
    }
}

fn bench_rust_stats(iterations: u64) -> BenchmarkRecord {
    let start = Instant::now();
    let mut sample = GfxGraphStatsSample::default();
    for i in 0..iterations {
        sample.samples += 1;
        if i % 1024 == 0 {
            sample.failures += 1;
        }
    }
    black_box(&sample);
    let duration = start.elapsed();

    let mut metrics = BTreeMap::new();
    metrics.insert("samples".to_string(), json!(sample.samples));
    metrics.insert("failures".to_string(), json!(sample.failures));
    BenchmarkRecord {
        name: "rust-stats-sample-update".to_string(),
        kind: "rust-micro".to_string(),
        status: "passed",
        command: "in-process GfxGraphStatsSample update loop".to_string(),
        iterations: Some(iterations),
        duration_ms: Some(elapsed_ms(duration)),
        throughput_ops_per_sec: Some(throughput(iterations, duration)),
        stdout_tail: String::new(),
        stderr_tail: String::new(),
        metrics,
    }
}

fn run_native_runtime_ffi(repo_root: &Path) -> BenchmarkRecord {
    let candidates = default_library_candidates(repo_root);
    if !candidates.iter().any(|candidate| candidate.exists()) {
        let mut benchmark = skipped_benchmark(
            "rust-native-runtime-ffi",
            "rust-native",
            "libhipgraph_bridge.so does not exist; build with CMake first",
        );
        benchmark
            .metrics
            .insert("candidates".to_string(), json!(path_strings(&candidates)));
        return benchmark;
    }

    let start = Instant::now();
    match NativeBridge::open_default(repo_root) {
        Ok(bridge) => {
            let init_result = bridge.init();
            let version = bridge.version();
            let initialized = bridge.is_initialized();
            bridge.profiler_reset();
            let seq = bridge.profiler_record(ProfileEventKind::Unknown, 1, 2, 3);
            let (samples, counters) = bridge.profiler_snapshot(8);
            bridge.shutdown();

            let status = if init_result.is_ok() && initialized && !samples.is_empty() {
                "passed"
            } else {
                "failed"
            };
            let mut metrics = BTreeMap::new();
            metrics.insert(
                "library_path".to_string(),
                json!(bridge.library_path().display().to_string()),
            );
            metrics.insert(
                "version".to_string(),
                json!({
                    "major": version.major,
                    "minor": version.minor,
                    "patch": version.patch,
                    "gfx_target": version.gfx_target,
                    "rocm_version": version.rocm_version,
                }),
            );
            metrics.insert(
                "init_error".to_string(),
                json!(init_result.err().map(|error| error.to_string())),
            );
            metrics.insert("initialized".to_string(), json!(initialized));
            metrics.insert("recorded_seq".to_string(), json!(seq));
            metrics.insert("sample_count".to_string(), json!(samples.len()));
            metrics.insert(
                "profiler_counters".to_string(),
                json!({
                    "written": counters.written,
                    "dropped": counters.dropped,
                    "capacity": counters.capacity,
                }),
            );
            BenchmarkRecord {
                name: "rust-native-runtime-ffi".to_string(),
                kind: "rust-native".to_string(),
                status,
                command: format!("NativeBridge::open_default({})", repo_root.display()),
                iterations: None,
                duration_ms: Some(elapsed_ms(start.elapsed())),
                throughput_ops_per_sec: None,
                stdout_tail: String::new(),
                stderr_tail: String::new(),
                metrics,
            }
        }
        Err(error) => BenchmarkRecord {
            name: "rust-native-runtime-ffi".to_string(),
            kind: "rust-native".to_string(),
            status: "failed",
            command: format!("NativeBridge::open_default({})", repo_root.display()),
            iterations: None,
            duration_ms: Some(elapsed_ms(start.elapsed())),
            throughput_ops_per_sec: None,
            stdout_tail: String::new(),
            stderr_tail: error.to_string(),
            metrics: BTreeMap::from([("candidates".to_string(), json!(path_strings(&candidates)))]),
        },
    }
}

fn run_native_runtime_cli(repo_root: &Path) -> BenchmarkRecord {
    let probe_args = vec![
        "--repo-root".to_string(),
        repo_root.display().to_string(),
        "--event-count".to_string(),
        "3".to_string(),
        "--sample-count".to_string(),
        "8".to_string(),
    ];
    let release_probe = repo_root.join("target/release/gfxgraph-native-probe");
    let debug_probe = repo_root.join("target/debug/gfxgraph-native-probe");
    let record = if release_probe.exists() {
        run_command(
            "rust-native-runtime-cli",
            &release_probe.display().to_string(),
            &probe_args,
            Some(repo_root),
            &[],
        )
    } else if debug_probe.exists() {
        run_command(
            "rust-native-runtime-cli",
            &debug_probe.display().to_string(),
            &probe_args,
            Some(repo_root),
            &[],
        )
    } else {
        let mut cargo_args = vec![
            "run".to_string(),
            "--quiet".to_string(),
            "-p".to_string(),
            "rs_gfxgraph_native".to_string(),
            "--bin".to_string(),
            "gfxgraph-native-probe".to_string(),
            "--".to_string(),
        ];
        cargo_args.extend(probe_args);
        run_command(
            "rust-native-runtime-cli",
            "cargo",
            &cargo_args,
            Some(repo_root),
            &[],
        )
    };
    let mut benchmark = command_as_benchmark("rust-native-runtime-cli", "rust-native", record);
    if let Ok(payload) = serde_json::from_str::<Value>(benchmark.stdout_tail.trim()) {
        benchmark
            .metrics
            .insert("payload".to_string(), payload.clone());
        for key in [
            "python_used",
            "init_ok",
            "initialized",
            "library_path",
            "native_contracts",
            "event_count",
            "snapshot_sample_count",
            "profiler_counters",
            "version",
        ] {
            if let Some(value) = payload.get(key) {
                benchmark.metrics.insert(key.to_string(), value.clone());
            }
        }
    }
    benchmark
}

fn run_hip_benchmark(repo_root: &Path) -> BenchmarkRecord {
    let executable = repo_root.join("build/benchmark_pipeline");
    if !executable.exists() {
        return skipped_benchmark(
            "native-hip-benchmark",
            "hip-native",
            "build/benchmark_pipeline does not exist; build with CMake BUILD_BENCHMARKS=ON",
        );
    }
    let record = run_command(
        "native-hip-benchmark",
        &executable.display().to_string(),
        &[],
        Some(repo_root),
        &[],
    );
    let mut benchmark = command_as_benchmark("native-hip-benchmark", "hip-native", record);
    parse_native_pipeline_metrics(&mut benchmark);
    benchmark
}

fn path_strings(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect()
}

fn run_ctest(repo_root: &Path) -> BenchmarkRecord {
    let build_dir = repo_root.join("build");
    if !build_dir.exists() {
        return skipped_benchmark(
            "native-hip-tests",
            "hip-native",
            "build directory does not exist; configure CMake native tests first",
        );
    }
    let record = run_command(
        "native-hip-tests",
        "ctest",
        &[
            "--test-dir".to_string(),
            build_dir.display().to_string(),
            "--output-on-failure".to_string(),
        ],
        Some(repo_root),
        &[],
    );
    command_as_benchmark("native-hip-tests", "hip-native", record)
}

fn run_python_script(
    name: &str,
    kind: &str,
    repo_root: &Path,
    python: &Path,
    script: &str,
    script_args: &[String],
    envs: &[(String, String)],
) -> BenchmarkRecord {
    let script_path = repo_root.join(script);
    if !script_path.exists() {
        return skipped_benchmark(name, kind, "script does not exist");
    }
    let mut args = Vec::with_capacity(script_args.len() + 1);
    args.push(script.to_string());
    args.extend(script_args.iter().cloned());
    let record = run_command(
        name,
        &python.display().to_string(),
        &args,
        Some(repo_root),
        envs,
    );
    command_as_benchmark(name, kind, record)
}

fn run_command(
    name: &str,
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    envs: &[(String, String)],
) -> CommandRecord {
    let command_display = display_command(program, args);
    let start = Instant::now();
    let mut command = Command::new(program);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    for (key, value) in envs {
        command.env(key, value);
    }

    match command.output() {
        Ok(output) => CommandRecord {
            name: name.to_string(),
            command: command_display,
            status: if output.status.success() {
                "passed"
            } else {
                "failed"
            },
            duration_ms: elapsed_ms(start.elapsed()),
            exit_code: output.status.code(),
            stdout_tail: tail_text(&String::from_utf8_lossy(&output.stdout), 16_384),
            stderr_tail: tail_text(&String::from_utf8_lossy(&output.stderr), 16_384),
        },
        Err(error) => CommandRecord {
            name: name.to_string(),
            command: command_display,
            status: "failed",
            duration_ms: elapsed_ms(start.elapsed()),
            exit_code: None,
            stdout_tail: String::new(),
            stderr_tail: error.to_string(),
        },
    }
}

fn command_as_benchmark(name: &str, kind: &str, command: CommandRecord) -> BenchmarkRecord {
    BenchmarkRecord {
        name: name.to_string(),
        kind: kind.to_string(),
        status: command.status,
        command: command.command,
        iterations: None,
        duration_ms: Some(command.duration_ms),
        throughput_ops_per_sec: None,
        stdout_tail: command.stdout_tail,
        stderr_tail: command.stderr_tail,
        metrics: BTreeMap::new(),
    }
}

fn public_benchmark_record(mut benchmark: BenchmarkRecord) -> BenchmarkRecord {
    if benchmark.status != "passed" {
        return benchmark;
    }
    let Ok(payload) = serde_json::from_str::<Value>(benchmark.stdout_tail.trim()) else {
        benchmark.metrics.insert(
            "parse_warning".to_string(),
            json!("stdout was not parseable readme benchmark JSON"),
        );
        return benchmark;
    };

    let results = payload
        .get("results")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let fallback_workloads: Vec<Value> = results
        .iter()
        .filter(|entry| {
            entry
                .get("fallback")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .map(|entry| {
            json!({
                "workload": entry.get("workload").and_then(Value::as_str).unwrap_or("unknown"),
                "speedup_x": entry.get("speedup_x").cloned().unwrap_or(Value::Null),
                "eager_ms_per_iter": entry.get("eager_ms_per_iter").cloned().unwrap_or(Value::Null),
                "graph_ms_per_iter": entry.get("graph_ms_per_iter").cloned().unwrap_or(Value::Null)
            })
        })
        .collect();

    benchmark.metrics.insert(
        "fallback_workload_count".to_string(),
        json!(fallback_workloads.len()),
    );
    benchmark
        .metrics
        .insert("workload_count".to_string(), json!(results.len()));
    benchmark.metrics.insert(
        "all_workloads_fallback".to_string(),
        json!(!results.is_empty() && fallback_workloads.len() == results.len()),
    );
    benchmark
        .metrics
        .insert("fallback_workloads".to_string(), json!(fallback_workloads));
    benchmark.metrics.insert("payload".to_string(), payload);
    benchmark
}

fn parse_native_pipeline_metrics(benchmark: &mut BenchmarkRecord) {
    if benchmark.status != "passed" {
        return;
    }

    for line in benchmark.stdout_tail.lines() {
        if let Some(value) = line.strip_prefix("Warmup launches: ") {
            if let Ok(parsed) = value.trim().parse::<u64>() {
                benchmark
                    .metrics
                    .insert("warmup_launches".to_string(), json!(parsed));
            }
        } else if let Some(value) = line.strip_prefix("Measured launches: ") {
            if let Ok(parsed) = value.trim().parse::<u64>() {
                benchmark.iterations = Some(parsed);
                benchmark
                    .metrics
                    .insert("measured_launches".to_string(), json!(parsed));
            }
        } else if let Some(value) = line.strip_prefix("Average pipeline launch latency: ") {
            let trimmed = value.trim().trim_end_matches(" us").trim();
            if let Ok(parsed) = trimmed.parse::<f64>() {
                benchmark.metrics.insert(
                    "average_pipeline_launch_latency_us".to_string(),
                    json!(parsed),
                );
            }
        } else if let Some(value) = line.strip_prefix("Device clock64 probe cycles: ") {
            if let Ok(parsed) = value.trim().parse::<u64>() {
                benchmark
                    .metrics
                    .insert("device_clock64_probe_cycles".to_string(), json!(parsed));
            }
        } else if let Some(value) = line.strip_prefix("Profiler samples copied: ") {
            if let Ok(parsed) = value.trim().parse::<u64>() {
                benchmark
                    .metrics
                    .insert("profiler_samples_copied".to_string(), json!(parsed));
            }
        } else if let Some(value) = line.strip_prefix("Profiler events written: ") {
            if let Ok(parsed) = value.trim().parse::<u64>() {
                benchmark
                    .metrics
                    .insert("profiler_events_written".to_string(), json!(parsed));
            }
        } else if let Some(value) = line.strip_prefix("Profiler events dropped: ") {
            if let Ok(parsed) = value.trim().parse::<u64>() {
                benchmark
                    .metrics
                    .insert("profiler_events_dropped".to_string(), json!(parsed));
            }
        }
    }
}

fn skipped_benchmark(name: &str, kind: &str, reason: &str) -> BenchmarkRecord {
    let mut metrics = BTreeMap::new();
    metrics.insert("skip_reason".to_string(), json!(reason));
    BenchmarkRecord {
        name: name.to_string(),
        kind: kind.to_string(),
        status: "skipped",
        command: String::new(),
        iterations: None,
        duration_ms: None,
        throughput_ops_per_sec: None,
        stdout_tail: String::new(),
        stderr_tail: String::new(),
        metrics,
    }
}

fn validate_report(report: &BenchmarkReport) -> Result<(), Box<dyn Error>> {
    if report.schema_version != SCHEMA_VERSION {
        return Err("schema_version mismatch".into());
    }
    if report.report_kind != REPORT_KIND {
        return Err("report_kind mismatch".into());
    }
    if report.run_id.trim().is_empty() {
        return Err("run_id is empty".into());
    }
    if report.timestamp_utc.trim().is_empty() {
        return Err("timestamp_utc is empty".into());
    }
    if report.report_date.len() != 10 {
        return Err("report_date must use YYYY-MM-DD".into());
    }
    if report.gate_checks.is_empty() {
        return Err("gate_checks must not be empty".into());
    }
    if report.benchmarks.is_empty() {
        return Err("benchmarks must not be empty".into());
    }
    for check in &report.gate_checks {
        validate_status(check.status)?;
    }
    for benchmark in &report.benchmarks {
        validate_status(benchmark.status)?;
    }
    Ok(())
}

fn validate_status(status: &str) -> Result<(), Box<dyn Error>> {
    match status {
        "passed" | "failed" | "skipped" => Ok(()),
        _ => Err(format!("invalid status: {status}").into()),
    }
}

fn repo_provenance(repo_root: &Path) -> RepoProvenance {
    let branch = command_stdout(
        "git",
        &[
            "-C".to_string(),
            repo_root.display().to_string(),
            "branch".to_string(),
            "--show-current".to_string(),
        ],
    )
    .unwrap_or_else(|| "unknown".to_string());
    let commit = command_stdout(
        "git",
        &[
            "-C".to_string(),
            repo_root.display().to_string(),
            "rev-parse".to_string(),
            "HEAD".to_string(),
        ],
    )
    .unwrap_or_else(|| "unknown".to_string());
    let tracked_status = command_stdout(
        "git",
        &[
            "-C".to_string(),
            repo_root.display().to_string(),
            "status".to_string(),
            "--short".to_string(),
            "--untracked-files=no".to_string(),
        ],
    )
    .unwrap_or_default();

    RepoProvenance {
        root: repo_root.display().to_string(),
        branch,
        commit,
        tracked_dirty: !tracked_status.trim().is_empty(),
    }
}

fn environment_provenance() -> EnvironmentProvenance {
    let mut env = BTreeMap::new();
    for key in [
        "GFXGRAPH",
        "GFXGRAPH_VALIDATE",
        "GFXGRAPH_VRAM_CAP",
        "HIP_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "ROCM_PATH",
    ] {
        env.insert(key.to_string(), std::env::var(key).ok());
    }

    EnvironmentProvenance {
        os: command_stdout("uname", &["-a".to_string()]).unwrap_or_else(|| "unknown".to_string()),
        rocm_path: std::env::var("ROCM_PATH").ok(),
        hipcc_version: command_stdout("hipcc", &["--version".to_string()]),
        env,
    }
}

fn command_stdout(program: &str, args: &[String]) -> Option<String> {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn default_python(repo_root: &Path) -> PathBuf {
    let local = repo_root.join(".venv/bin/python");
    if local.exists() {
        local
    } else {
        PathBuf::from("python3")
    }
}

fn phase_env(phase: Phase) -> Vec<(String, String)> {
    if phase.graph_enabled() {
        vec![("GFXGRAPH".to_string(), "1".to_string())]
    } else {
        Vec::new()
    }
}

fn maturin_env(python: &Path) -> Vec<(String, String)> {
    let Some(bin_dir) = python.parent() else {
        return Vec::new();
    };
    if bin_dir.file_name().and_then(|name| name.to_str()) != Some("bin") {
        return Vec::new();
    }
    let Some(venv_root) = bin_dir.parent() else {
        return Vec::new();
    };

    let path = std::env::var("PATH")
        .map(|current| format!("{}:{current}", bin_dir.display()))
        .unwrap_or_else(|_| bin_dir.display().to_string());
    vec![
        ("VIRTUAL_ENV".to_string(), venv_root.display().to_string()),
        ("PATH".to_string(), path),
    ]
}

fn path_string_is_repo_source(path: &str, root: &Path) -> bool {
    if path.is_empty() {
        return false;
    }
    let candidate = PathBuf::from(path);
    if let Ok(candidate) = candidate.canonicalize() {
        return candidate.starts_with(root) && !path_is_installed_site_package(&candidate);
    }
    path.contains(&root.display().to_string()) && !path.contains("/site-packages/")
}

fn path_is_installed_site_package(path: &Path) -> bool {
    path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|part| part == "site-packages" || part == "dist-packages")
    })
}

fn absolute_path(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn display_command(program: &str, args: &[String]) -> String {
    std::iter::once(program.to_string())
        .chain(args.iter().cloned())
        .map(|part| {
            if part.contains(char::is_whitespace) {
                format!("{part:?}")
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn tail_text(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    text.chars().skip(char_count - max_chars).collect()
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn throughput(iterations: u64, duration: Duration) -> f64 {
    if duration.is_zero() {
        return 0.0;
    }
    iterations as f64 / duration.as_secs_f64()
}

struct UtcClock {
    timestamp_utc: String,
    report_date: String,
}

impl UtcClock {
    fn read() -> Self {
        let timestamp_utc = command_stdout(
            "date",
            &["-u".to_string(), "+%Y-%m-%dT%H:%M:%SZ".to_string()],
        )
        .unwrap_or_else(|| {
            let seconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            format!("unix-{seconds}")
        });
        let report_date = command_stdout("date", &["-u".to_string(), "+%Y-%m-%d".to_string()])
            .unwrap_or_else(|| timestamp_utc.chars().take(10).collect());
        Self {
            timestamp_utc,
            report_date,
        }
    }
}
