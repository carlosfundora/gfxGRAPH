use std::env;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use rs_gfxgraph_native::{default_library_candidates, NativeBridge, ProfileEventKind};
use serde_json::json;

#[derive(Debug)]
struct Args {
    repo_root: PathBuf,
    library: Option<PathBuf>,
    event_count: u64,
    sample_count: usize,
}

fn main() {
    let started = Instant::now();
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            println!(
                "{}",
                json!({
                    "kind": "gfxgraph-native-probe-v1",
                    "status": "failed",
                    "error": error,
                    "python_used": false,
                })
            );
            process::exit(2);
        }
    };

    let candidates = if let Some(library) = &args.library {
        vec![library.clone()]
    } else {
        default_library_candidates(&args.repo_root)
    };

    let bridge_result = if let Some(library) = &args.library {
        NativeBridge::open(library)
    } else {
        NativeBridge::open_default(&args.repo_root)
    };

    let bridge = match bridge_result {
        Ok(bridge) => bridge,
        Err(error) => {
            println!(
                "{}",
                json!({
                    "kind": "gfxgraph-native-probe-v1",
                    "status": "failed",
                    "error": error.to_string(),
                    "repo_root": args.repo_root,
                    "candidate_libraries": path_strings(&candidates),
                    "python_used": false,
                    "duration_ms": elapsed_ms(started),
                })
            );
            process::exit(1);
        }
    };

    let init_result = bridge.init();
    let version = bridge.version();
    let initialized = bridge.is_initialized();
    bridge.profiler_reset();

    let mut sequences = Vec::new();
    if init_result.is_ok() {
        for i in 0..args.event_count {
            sequences.push(bridge.profiler_record(ProfileEventKind::Unknown, i + 1, i + 2, i + 3));
        }
    }
    let (samples, counters) = bridge.profiler_snapshot(args.sample_count);
    bridge.shutdown();

    let status = if init_result.is_ok()
        && initialized
        && samples.len() == args.event_count.min(args.sample_count as u64) as usize
    {
        "passed"
    } else {
        "failed"
    };

    println!(
        "{}",
        json!({
            "kind": "gfxgraph-native-probe-v1",
            "status": status,
            "repo_root": args.repo_root,
            "library_path": bridge.library_path(),
            "candidate_libraries": path_strings(&candidates),
            "python_used": false,
            "native_contracts": [
                "lifecycle",
                "profiler",
                "pipeline_handles",
                "composed_handles"
            ],
            "version": {
                "major": version.major,
                "minor": version.minor,
                "patch": version.patch,
                "gfx_target": version.gfx_target,
                "rocm_version": version.rocm_version,
            },
            "init_ok": init_result.is_ok(),
            "init_error": init_result.err().map(|error| error.to_string()),
            "initialized": initialized,
            "event_count": args.event_count,
            "recorded_sequences": sequences,
            "snapshot_sample_count": samples.len(),
            "profiler_counters": {
                "written": counters.written,
                "dropped": counters.dropped,
                "capacity": counters.capacity,
            },
            "duration_ms": elapsed_ms(started),
        })
    );

    if status != "passed" {
        process::exit(1);
    }
}

fn parse_args() -> Result<Args, String> {
    let mut repo_root = PathBuf::from(".");
    let mut library = None;
    let mut event_count = 3_u64;
    let mut sample_count = 8_usize;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--repo-root" => {
                let Some(value) = iter.next() else {
                    return Err("--repo-root requires a path".to_string());
                };
                repo_root = PathBuf::from(value);
            }
            "--library" => {
                let Some(value) = iter.next() else {
                    return Err("--library requires a path".to_string());
                };
                library = Some(PathBuf::from(value));
            }
            "--event-count" => {
                let Some(value) = iter.next() else {
                    return Err("--event-count requires an integer".to_string());
                };
                event_count = value
                    .parse()
                    .map_err(|_| format!("invalid --event-count value: {value}"))?;
            }
            "--sample-count" => {
                let Some(value) = iter.next() else {
                    return Err("--sample-count requires an integer".to_string());
                };
                sample_count = value
                    .parse()
                    .map_err(|_| format!("invalid --sample-count value: {value}"))?;
            }
            "--help" | "-h" => {
                return Err(
                    "usage: gfxgraph-native-probe [--repo-root PATH] [--library PATH] [--event-count N] [--sample-count N]"
                        .to_string(),
                );
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    if event_count == 0 {
        return Err("--event-count must be greater than zero".to_string());
    }
    if sample_count == 0 {
        return Err("--sample-count must be greater than zero".to_string());
    }

    Ok(Args {
        repo_root,
        library,
        event_count,
        sample_count,
    })
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn path_strings(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect()
}
