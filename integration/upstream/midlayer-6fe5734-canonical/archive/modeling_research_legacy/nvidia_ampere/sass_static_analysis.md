# Static SASS Analysis for Kernel Behavior Prediction

## Idea

从编译产物（SASS）静态提取 kernel 的结构信息，用于预测 per-kernel 的 L2 effective BW，
替代 NCU profiling（不需要 GPU 执行）。

## 动机

当前 V2 resource model 用固定 L2 BW = 2500 B/clk，实际 achieved BW 范围 1316-3735（2.8x）。
这是 ALL 13.8% MAPE 的主要来源。Oracle L2 BW → 1.3% 证明公式结构无问题，
唯一的差距就是"预测每个 kernel 实际能打到多少 BW"。

SASS 是 GPU 编译的最终产物，包含完整的指令调度、寄存器分配、内存访问模式。
与 AMALi 的运行时 trace 不同，SASS 可以在 CPU 上静态获取（cuobjdump / Triton JIT cache）。

## 可获取性

```
Kernel 来源             SASS 获取方式                    是否需要 GPU
──────────────────────────────────────────────────────────────────────
cuBLAS                  cuobjdump 从 fatbinary 提取      否 (CPU)
Triton JIT             编译 cache (~/.triton/cache/)     否 (CPU)
CUTLASS                 编译产物 .cubin                  否 (CPU)
自定义 CUDA             ptxas 编译 (只需 toolkit)         否 (CPU)
```

ptxas 是 CPU 程序，只需 CUDA toolkit + 目标架构标记 (sm_80)。
速度: ~1s per kernel variant (cuobjdump dominated)。

## SASS 可提取的信息

```
信息                              提取方式                          对建模的作用
─────────────────────────────────────────────────────────────────────────────────
tile dims (Mb, Nb, Kb)            HMMA layout + LDGSTS count/iter   → working set → miss rate
pipeline stages                   barrier-to-barrier 间距            → OLP per warp
loads per stage                   LDGSTS 指令计数                    → OLP per warp
regs per thread                   SASS header (.nv.info section)     → occupancy
smem per block                    SASS header                        → occupancy
cache hint (.cg/.ca/.cs)          load 指令修饰符                    → L1 bypass 行为
循环结构 (unroll factor, K-loop)  branch pattern + loop back-edge    → 总 load 次数
```

## 验证实验结果 (2026-05-20)

用 NCU 已有数据（occupancy、grid size、miss rate）作为 proxy，验证 OLP 和 achieved BW 的相关性。

### L2 achieved BW 的驱动因素

```
Regime              N     BW 范围 (B/clk)   主驱动         r²
──────────────────────────────────────────────────────────────
Full SM (=108)     484    718-3749          miss_rate      0.664
Non-full SM (<108) 144     72-2788          active_sms     0.677
  tiny (≤32)        37     72-612           OLP            0.726
```

### 关键发现

1. **77% 的 kernel 跑满 108 SM，OLP 早已饱和**
   - median OLP = 864 warps on chip
   - pipeline depth (3 vs 5 stages) 对 achieved BW 无影响
   - BW 变异完全由 miss_rate (L2 refill contention) 驱动

2. **23% non-full SM kernel (主要是 FA decode)**
   - SM injection bandwidth 是瓶颈，BW 正比于 active_sms
   - active_sms = min(grid, 108) trivially 已知，不需要 SASS

3. **tiny grid (≤32 blocks) 子集里 OLP 确实 matters (r²=0.726)**
   - 但只有 37 个 kernel，对整体 MAPE 影响 < 1pp

### 组合模型

```
BW_eff = min(BW_injection, BW_L2_refill)

BW_injection = 23.8 × active_sms              (SM-side, linear fit r²=0.677)
BW_L2_refill = 2821 - 1439 × miss_rate        (L2-side, linear fit r²=0.664)

Combined R² = 0.694, BW prediction MAPE = 16.7%
```

## SASS vs Tile-Level Reuse Distance

```
方法                  需要 GPU?   精度              用途
──────────────────────────────────────────────────────────────────
Address-level (AMALi) 需要       精确              已有，慢且依赖硬件
Tile-level symbolic   不需要     足够 (7.4% MAPE)  forward model 核心
Op-level analytical   不需要     同上              当前 v5 LRU model
```

AMALi 运行时截获每条 load 的真实地址做 PARDA。SASS 做不到这个（地址是运行时值）。

SASS 可以做 tile-level symbolic reuse distance: 从静态指令结构提取 tile dims + pipeline stages，
然后用 tile 粒度模拟 LRU。这和已有的 tile LRU model 本质一样，区别是 tile dims 来源更可靠。

## 当前结论与优先级

对当前阶段（Phase 2 forward model），SASS 分析不是 bottleneck:
- miss_rate 预测 → 来自 tile LRU model (tile dims 已能从 kernel name/grid 推断)
- active_sms → trivially 已知
- OLP → 对 full-SM kernel 无影响

**SASS 的真正价值场景（未来）：**
1. 自定义 Triton kernel（没有 CUTLASS 命名规范，无法从 kernel name 推断 tile dims）
2. 新硬件（L2 更小 / latency 更高 → OLP 不再普遍饱和）
3. 跨硬件通用性（ISA parser per-arch，但产出统一 KernelStaticProfile）
4. 消除对 NCU profiling 的最后依赖（完全 offline 预测）

## 未来实施方案

### SASS Parser 产出的统一中间表示

```python
@dataclass
class KernelStaticProfile:
    tile_dims: Tuple[int, int, int]    # (Mb, Nb, Kb) from HMMA structure
    pipeline_stages: int                # barrier-to-barrier count in main loop
    loads_per_stage: int                # LDGSTS/LDG count per stage
    regs_per_thread: int                # from .nv.info
    smem_per_block: int                 # from .nv.info
    cache_bypass_ratio: float           # fraction of .cg loads
    loop_iterations: int                # K / Kb (combined with op shape)
    unroll_factor: int                  # from loop structure
```

### 跨硬件设计

ISA parser 是硬件相关的，但 KernelStaticProfile 是通用的:
- NVIDIA: cuobjdump + SASS parser
- 沐曦 C500: 对应 ISA disassembler + parser
- AMD CDNA: llvm-objdump + CDNA ISA parser

profile 提取后喂给同一套 resource model 公式，不需要改公式。
