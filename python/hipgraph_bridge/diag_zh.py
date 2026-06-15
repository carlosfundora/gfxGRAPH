# gfxGRAPH — diagnostics Chinese (zh) translations; loaded on demand when GFXGRAPH_LANG=zh
# Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora
# SPDX-License-Identifier: MIT
"""Chinese (zh) message pack for gfxGRAPH diagnostics.

Kept in a separate file so the English diagnostics module stays clean; imported lazily by
`diagnostics.py` only when `GFXGRAPH_LANG=zh`. Keys in `MESSAGES` match the canonical error
codes in `diagnostics._TABLE`. Self-contained (no import from diagnostics).
"""

from __future__ import annotations

# Frame labels (the box around each diagnosis).
LABELS = {
    "diag": "gfxGRAPH 诊断", "cause": "原因", "ctx": "ROCm/gfx1030 背景",
    "fix": "解决方法", "silence": "设置 GFXGRAPH_DIAG=0 可静音",
}

# Arch descriptor is a token («arch») substituted at format() time so zh output adapts to the
# detected/overridden GPU (GFXGRAPH_ARCH), same mechanism as the English table. _HSA is literal.
_GFX = "«arch»"
_HSA = "HSA_OVERRIDE_GFX_VERSION=10.3.0"

# Per-code messages — same canonical codes as diagnostics._TABLE.
MESSAGES: dict[str, dict] = {
    "no_kernel_image": {
        "summary": "某内核没有为当前 GPU 架构编译的二进制镜像。",
        "cause": "该算子（常见于 torch FLASH/AOTriton SDPA、FA3/FA4，或 CUTLASS/CK 内核）是为不含 "
                 "gfx1030 的架构提前编译（AOT）的，运行时找不到可用镜像。",
        "rocm_context": f"在 {_GFX} 上，AOTriton/AITER 的 flash-attention 与张量核内核通常不提供 "
                        "gfx1030 镜像（FA3/FA4 还需要 Hopper/Blackwell）。这不是你代码的 bug。",
        "fix": "改用 JIT-Triton 路径（sglang `--attention-backend triton`，或自研 flash-decode-hip / "
               f"flash-attn-prefill-hip），或用 `--offload-arch=gfx1030` 重新编译并在 {_HSA} 下运行。",
    },
    "out_of_memory": {
        "summary": "GPU 显存耗尽（OOM）。",
        "cause": "申请的显存超过空闲量——通常是由 `--mem-fraction-static` 决定的 KV/激活池，或量化编解码器在该池之外额外申请的缓冲区。",
        "rocm_context": f"{_GFX} 约有 12 GB 显存。RotorQuant/TurboQuant（rq3/tq3）KV 编解码器会在基础池之外"
                        "额外分配压缩缓冲区与旋转矩阵，因此较高的 mem-fraction（如 0.85）即使纯 f16 能放下也会 OOM；"
                        "其它常驻服务也可能占用显存。",
        "fix": "降低 `--mem-fraction-static`（rq*/tq* 编解码器约用 0.45——sglang 在 RDNA2 上会自动下调）、"
               "释放竞争的 GPU 服务，或减小上下文/批大小。用 `rocm-smi --showmeminfo vram` / "
               "`gfxgraph.environment_report()` 查看空闲显存。",
    },
    "illegal_address": {
        "summary": "非法/越界 GPU 内存访问（否则会是不透明的 SIGSEGV）。",
        "cause": "要么是被捕获进 CUDA/HIP 图的缓冲区在重放时读到了失效/已释放的地址（捕获安全性问题），"
                 "要么是内核索引越界（逻辑 bug）。",
        "rocm_context": f"在 {_GFX} 上这是头号 CUDA-graph 失败原因，因此这里默认关闭 cuda-graph。"
                        "把非连续/带步幅/0 步幅的张量捕获进图是常见的捕获安全性诱因。",
        "fix": "启用 GUARD（`GFXGRAPH_GUARD=1` 自动使捕获输入安全；`=2` 定位出错算子与张量布局；`=3` 深度红区/sanitizer）。"
               "对于逻辑 bug 类，请在 compute-sanitizer 下运行以定位具体算子。",
    },
    "bf16_unsupported": {
        "summary": "此 GPU 不支持 bfloat16 GEMM/点积。",
        "cause": "内核发出了硬件不具备的 bf16 点积指令。",
        "rocm_context": f"{_GFX} 没有 `fdot2.bf16.bf16`——bf16 矩阵乘会崩溃。（CDNA/RDNA3 有，RDNA2 没有。）",
        "fix": "改用 fp16 而非 bf16（sglang/llama.cpp 在 gfx1030 上会自动把 bf16→fp16；确认该覆盖生效，或传 `--dtype float16`）。",
    },
    "wrong_arch": {
        "summary": "GPU 架构不匹配/未设置覆盖。",
        "cause": "工具链或运行时解析出的架构与物理 GPU 不一致。",
        "rocm_context": "本机物理上是 gfx1031，但按 gfx1030（受支持最好的 RDNA2 目标）运行；不设覆盖时 ROCm 可能误判。",
        "fix": f"导出 {_HSA}（并用 `--offload-arch=gfx1030` 构建）。用 `rocminfo | grep gfx` 验证。",
    },
    "wave64_ignored": {
        "summary": "请求了 Wave64，但 gfx1030 只能以 Wave32 执行。",
        "cause": "ROCm 的 gfx1030 后端会静默忽略 `-mwavefrontsize64`；你始终得到 Wave32。",
        "rocm_context": f"{_GFX} 仅支持 Wave32。要“超过 32 通道”必须用软件实现。",
        "fix": "使用软件 split-K：每个工作项联合 W 个 Wave32 波并经 LDS 归并（W=2≈wave64，W=4≈wave128）——"
               "如 flash-decode-hip 的自适应 `W∈{1,2,4}`。仅在网格未填满时采用 W>1。",
    },
    "aiter_on_rdna": {
        "summary": "AITER（AMD CK/ASM 内核）在此 GPU 上不可用/非最优。",
        "cause": "AITER 的 ASM/CK 内核面向 CDNA（MI2xx/MI3xx）；在 RDNA 上缺失或回退。",
        "rocm_context": f"{_GFX} 是 RDNA2——AITER 注意力会回退到（较慢的）Triton，AITER MoE/flydsl 组件可能缺失。"
                        "这不是错误，只是硬件不支持。",
        "fix": "在 RDNA 上优先 Triton 路径（sglang 在 gfx10xx 上会自动把 `aiter`→triton）。不要指望这里有 CK/ASM 加速。",
    },
    "invalid_configuration": {
        "summary": "无效的内核启动配置。",
        "cause": "线程块/网格维度或共享内存/寄存器需求超过设备上限。",
        "rocm_context": f"{_GFX}：每块最多 1024 线程、每 CU 64 KB LDS；过高的每通道 VGPR 占用"
                        "（如每通道 O[D] 累加器）会使占用率崩溃。",
        "fix": "减小块大小/LDS/每线程寄存器。注意力建议用一个 Wave32 波把 head-dim 归约分摊到 32 个通道，"
               "而非使用巨大的每通道累加器。",
    },
}
