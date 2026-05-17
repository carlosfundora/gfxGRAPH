#!/usr/bin/env python3
"""
Public README benchmark for gfxGRAPH on AMD RDNA2.

Runs reproducible eager-vs-graph microbenchmarks on real GPU workloads and
writes a JSON report suitable for README updates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch

from hipgraph_bridge.graph_manager import BridgedCUDAGraph


def _bench_eager_vs_graph(
    name: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    iters: int,
    warmup: int,
) -> dict:
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
    with g.capture(model_fn=fn):
        g._static_output = fn(x)

    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    graph_s = time.perf_counter() - t0

    return {
        "workload": name,
        "iters": iters,
        "eager_ms_per_iter": (eager_s * 1000.0) / iters,
        "graph_ms_per_iter": (graph_s * 1000.0) / iters,
        "speedup_x": eager_s / graph_s if graph_s else None,
        "fallback": bool(g._eager_fallback),
    }


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is not available")

    device_info = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
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
        ),
        _bench_eager_vs_graph(
            "mlp_bs32_d1024",
            lambda x: mlp_1k(x),
            x_b,
            iters=1500,
            warmup=100,
        ),
        _bench_eager_vs_graph(
            "mlp_bs128_d2048",
            lambda x: mlp_2k(x),
            x_c,
            iters=300,
            warmup=30,
        ),
    ]

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
    args = parser.parse_args()

    payload = run()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
