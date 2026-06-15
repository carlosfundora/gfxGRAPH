<!-- 发布者 / Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora · MIT -->
# gfxGRAPH 中文使用指南

**gfxGRAPH** 是面向 AMD **gfx1030/1031（RDNA2）** 的 CUDA Graph → HIP Graph 转换层，提供安全的
eager 回退、动态形状分桶、illegal-memory-access 防护（GUARD），以及**始终在线的 HIP/ROCm 错误与状态报告（中英双语）**。

本指南为中文用户而写。**所有报错信息也可切换为中文**（见下文“中文切换”）。

> 边界说明：gfxGRAPH 工作在 CUDA-graph / torch 层，**无法在运行时改写已编译内核的波前大小**。对
> wave64/wave128，它做的是**捕获（检测）+ 转换规划**（软件波 = 联合 W 个 Wave32 波 + LDS 归并），
> 而非“魔法改写”。

## 安装

```bash
uv pip install gfxgraph        # 或: pip install gfxgraph
```
纯 Python 包，处处可装；原生 HIP 加速桥为可选的 `gfxgraph-native` 伴随包，缺失时自动回退，不影响导入。

## 快速开始

```python
import gfxgraph

gfxgraph.install_diagnostics()          # 始终在线：把晦涩的 HIP 报错翻译成可读的诊断
print(gfxgraph.environment_report())    # 是否设了 HSA 覆盖？架构？空闲显存？图状态？

with gfxgraph.diagnose("解码前向"):      # 包裹有风险的代码块；出错时打印诊断后再抛出
    out = model.generate(...)
```

## 诊断（错误/警告/状态报告）

ROCm 的报错往往非常简短（例如 “No available kernel. Aborting execution.”）。gfxGRAPH 会把它翻译成
**原因 + gfx1030/RDNA2 背景 + 具体解决方法**。覆盖的常见情形：

| 代码 | 含义（gfx1030 场景） |
|---|---|
| `no_kernel_image` | 内核无 gfx1030 镜像（AOTriton/FA3/CUTLASS）→ 改用 Triton 路径 |
| `out_of_memory` | 显存 OOM；rq3/tq3 编解码器需额外缓冲 → 降低 `--mem-fraction-static`（约 0.45） |
| `illegal_address` | 非法访问（多为 CUDA-graph 失效缓冲）→ 启用 `GFXGRAPH_GUARD` |
| `bf16_unsupported` | gfx1030 无 `fdot2.bf16` → 改用 fp16 |
| `wrong_arch` | 架构不匹配 → 设 `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| `wave64_ignored` | Wave64 被忽略（仅 Wave32）→ 用软件波 split-K |
| `aiter_on_rdna` | AITER 面向 CDNA，RDNA 上回退 → 用 Triton |
| `invalid_configuration` | 启动配置超限 → 减小块/LDS/寄存器 |

```python
d = gfxgraph.explain("No available kernel. Aborting execution.")
print(d.format("zh"))   # 强制中文；或设 GFXGRAPH_LANG=zh 全局生效
```

## 中文切换

```bash
export GFXGRAPH_LANG=zh   # 所有诊断输出切换为中文（默认 en）
export GFXGRAPH_DIAG=0    # 关闭诊断输出（默认开启）
```
中文文案放在独立文件 `hipgraph_bridge/diag_zh.py`，仅在 `GFXGRAPH_LANG=zh` 时按需加载——保持英文代码整洁，英文用户零额外开销。

## GUARD（图捕获的非法访问防护）

```bash
export GFXGRAPH_GUARD=1   # 一级：自动使捕获输入连续/安全
export GFXGRAPH_GUARD=2   # 二级：把 SIGSEGV 定位为可捕获的 GfxGraphFault（含算子+张量布局）
export GFXGRAPH_GUARD=3   # 三级：红区金丝雀 + compute-sanitizer（慢，排深层 bug）
```

## wavefront：捕获 wave64/128 并规划软件波转换

```python
import gfxgraph
gfxgraph.detect_wave64("-mwavefrontsize64")          # 捕获 wave64/128 意图 → 诊断
plan = gfxgraph.plan_software_wave(total_rows, 64)    # 规划软件 wave64（W=2 Wave32 + LDS 归并）
# plan = {'W':2,'block':64,'grid':...,'note':'...emulates wave64'}；网格已填满时 W=1
```
仅当网格未填满（如解码/短提示）时 `W>1` 才有收益——与 flash-decode-hip 的自适应 `W∈{1,2,4}` 一致。

## 环境变量一览

| 变量 | 作用 |
|---|---|
| `GFXGRAPH=1` | 启用 CUDA→HIP 图桥（并自动安装诊断） |
| `GFXGRAPH_GUARD=1\|2\|3` | 非法访问防护层级 |
| `GFXGRAPH_DIAG=0` | 关闭诊断输出 |
| `GFXGRAPH_LANG=zh` | 诊断输出切换为中文 |
| `HSA_OVERRIDE_GFX_VERSION=10.3.0` | gfx1031 按 gfx1030 运行（gfx1030 框必设） |

---
发布者：Carlos Fundora（GitHub/Hugging Face：@carlosfundora）· 许可证：MIT
