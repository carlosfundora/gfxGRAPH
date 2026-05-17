#!/usr/bin/env python3
"""
Public README benchmark for gfxGRAPH on AMD RDNA2.

Runs reproducible eager-vs-graph microbenchmarks on real GPU workloads and
writes a JSON report suitable for README updates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import statistics
from pathlib import Path
from typing import Callable

import torch

from hipgraph_bridge.graph_manager import BridgedCUDAGraph


def _safe_command_output(command: list[str]) -> str | None:
    try:
        out = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception:
        return None


def _current_commit_sha() -> str | None:
    return _safe_command_output(["git", "rev-parse", "HEAD"])


def _rocm_driver_info() -> dict:
    info: dict[str, str | None] = {
        "runtime_from_torch": getattr(torch.version, "hip", None),
        "driver_from_hipconfig": None,
        "runtime_from_rocminfo": None,
    }
    hipconfig = _safe_command_output(["hipconfig", "--version"])
    if hipconfig:
        info["driver_from_hipconfig"] = hipconfig.splitlines()[0]
    rocminfo = _safe_command_output(["rocminfo"])
    if rocminfo:
        for line in rocminfo.splitlines():
            lower = line.lower()
            if "runtime version" in lower:
                info["runtime_from_rocminfo"] = line.strip()
                break
    return info


def _tracked_env_vars() -> dict[str, str | None]:
    keys = [
        "HSA_OVERRIDE_GFX_VERSION",
        "PYTORCH_ROCM_ARCH",
        "GFXGRAPH",
        "GFXGRAPH_VRAM_CAP",
        "SGLANG_RDNA2_KERNELS",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {key: os.environ.get(key) for key in keys}


def _bench_eager_vs_graph(
    name: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    iters: int,
    warmup: int,
    run_count: int,
) -> dict:
    eager_ms_runs: list[float] = []
    graph_ms_runs: list[float] = []
    fallback_observed = False

    for _ in range(run_count):
        with torch.no_grad():
            for _ in range(warmup):
                fn(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                fn(x)
        torch.cuda.synchronize()
        eager_s = time.perf_counter() - t0

        g = BridgedCUDAGraph()
        eager_fn = lambda: fn(x)
        with g.capture(model_fn=eager_fn):
            g._static_output = fn(x)

        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        graph_s = time.perf_counter() - t0
        fallback_observed = fallback_observed or bool(g._eager_fallback)

        eager_ms_runs.append((eager_s * 1000.0) / iters)
        graph_ms_runs.append((graph_s * 1000.0) / iters)

    eager_ms = statistics.median(eager_ms_runs)
    graph_ms = statistics.median(graph_ms_runs)
    speedup = (eager_ms / graph_ms) if graph_ms else None

    return {
        "workload": name,
        "iters": iters,
        "run_count": run_count,
        "eager_ms_per_iter": eager_ms,
        "graph_ms_per_iter": graph_ms,
        "eager_ms_per_iter_runs": eager_ms_runs,
        "graph_ms_per_iter_runs": graph_ms_runs,
        "speedup_x": speedup,
        "fallback": fallback_observed,
    }


def run(run_count: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is not available")

    device_info = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "rocm": _rocm_driver_info(),
        "env": _tracked_env_vars(),
    }

    # Workload A: launch-bound decode-like chain
    ln = torch.nn.LayerNorm(1024).cuda().eval()
    x_a = torch.randn(1, 1024, device="cuda")

    def workload_a(x: torch.Tensor) -> torch.Tensor:
        y = x
        for _ in range(8):
            y = ln(y)
            y = torch.nn.functional.gelu(y)
        return y

    # Workload B: typical MLP block
    mlp_1k = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 1024),
    ).cuda().eval()
    x_b = torch.randn(32, 1024, device="cuda")

    # Workload C: heavier throughput path
    mlp_2k = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 2048),
    ).cuda().eval()
    x_c = torch.randn(128, 2048, device="cuda")

    results = [
        _bench_eager_vs_graph(
            "decode_like_layernorm_gelu_chain_bs1_d1024",
            workload_a,
            x_a,
            iters=2000,
            warmup=200,
            run_count=run_count,
        ),
        _bench_eager_vs_graph(
            "mlp_bs32_d1024",
            lambda x: mlp_1k(x),
            x_b,
            iters=1500,
            warmup=100,
            run_count=run_count,
        ),
        _bench_eager_vs_graph(
            "mlp_bs128_d2048",
            lambda x: mlp_2k(x),
            x_c,
            iters=300,
            warmup=30,
            run_count=run_count,
        ),
    ]

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit_sha": _current_commit_sha(),
        "run_count": run_count,
        "env": device_info,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="benchmarks/results/readme_benchmark_latest.json",
        help="Path to write benchmark JSON output",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        default=3,
        help="How many repeated timed runs to record per workload",
    )
    args = parser.parse_args()

    payload = run(run_count=args.run_count)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
