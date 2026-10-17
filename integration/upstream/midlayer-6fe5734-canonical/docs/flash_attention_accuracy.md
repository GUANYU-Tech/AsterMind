# FlashAttention 解析模型精度与机制记录

本文保存 FlashAttention（FA）解析模型中已经验证、可复用的物理结论，以及仍未闭合的问题和已否定的建模方式。它不是一次会话流水账；后续精度优化应从这里列出的证据链和剩余误差继续，而不是重新按最终 MAPE 猜公式。

通用 Kernel 模型和证据规则见 [Kernel 解析性能模型](kernel_performance_model.md)，C500 profiler 的 timing/counter 边界见 [MetaX C500 kernel profiling](metax_c500/profiling.md)。

## 1. 范围与结果口径

本轮工作从 `research/fa-accuracy` 在 `5d5ac948` 上创建开始，最终已验证实现节点为 `b43d5c46`。分支期间多次 rebase，历史提交 hash 仍可通过 `git reflog show research/fa-accuracy` 追溯，不能只用当前 ancestry 判断已经完成的工作。

表中指标均为 294 个 FA task 的 task-level aggregate 误差，不是单个内部 kernel 的平均值。MAPE 包含最大离群 case；实测时间只用于 evaluation，不是 Forward 参数来源。

| 硬件 | 起始 MAPE | 起始 Max APE | 当前 MAPE | 当前 Max APE | 当前最差 case |
|---|---:|---:|---:|---:|---|
| A800 | 12.3% | 42.2% | 11.2% | 44.4% | `FLASHATTN_B16_S512_HQ64_HKV8_D128_C0` |
| C500 | 25.1% | 72.5% | 13.9% | 33.2% | `FLASHATTN_B32_S512_HQ40_HKV8_D128_C0` |
| 910B | 26.8% | 78.5% | 16.0% | 52.4% | `FLASHATTN_B2_S512_HQ32_HKV8_D64_C0` |

可复查的本地 evaluation 记录为：起点 `experiment_logs/kernel/fa_accuracy/20260828_baseline_5d5ac948/`，A800 当前口径 `20260829_post_rebase_c08dabb6/nvidia_a800.log`，C500 当前口径 `20260828_910b_reduce1024/metax_c500.log`，910B 当前口径 `20260829_stride_l2resident_candidate/ascend_910b.log`。这些日志位于被忽略的 `experiment_logs/`，上表保留了长期需要的结果。

C500 的起始值和当前值不是纯公式消融：期间 canonical FA golden 切换为 294/294 case 的 serving 相关 CUDA-Graph replay + mcTracer 逐 kernel GPU duration，14 个原 launch-mismatch case 同时更新 launch metadata 和 layer-1 counters。因此 C500 的最终精度可以用于当前验收，但不能把 `25.1% → 13.9%` 全部归因于 Forward 修改。新数据基线为 `research/kernel-golden-v2@3a05fd2e`，不得再以旧 `e41456cc` 数据得出结论。

三种硬件的 timing target 以 canonical dataset 逐 entry 声明的 `op_info.evaluation.timing` 为准：A800 是 `kernel` 粒度的 `nsys_nvtx_kernel_median`（334 个 FA 行里 294 个如此，其余 40 个 FA_SERVING 行是 `op` 粒度），C500 是 `op` 粒度的 `device_event_op_median`，910B 是 `op` 粒度的 `msprof_timeline_op_median`。本行此前写 C500 为 strict Graph `mcTracer` kernel duration 之和、910B 为 `msprof_task_duration`，与 evaluation 实际消费的量不符，已按数据订正；`mcTracer` 仍是 C500 `kernel` 粒度的合法 source，只是当前 dataset 不用它。C500 的 mcProfiler `during` 是插桩 replay 时间，不能替代 production/Graph mcTracer 时间。三卡分母不同一事的影响范围见 `docs/cross_card_span_semantics.md`。

## 2. 跨硬件通用结论

FA 的大误差不能只用 `bytes / bandwidth` 或统一 launch 常数解释。真正影响短任务和少 wave 的是离散 launch、首个依赖、有限 buffer、跨域 handoff、barrier、尾部工作和 task 外层 envelope；这些必须按源码时序组成。

应始终区分三类量：物理 launch/issue 下界、真正进入主流水的 active work、以及 source dependency tail。Padded worker 即使提前退出仍可能支付 issue，但不能被当作完整 active pipeline；反过来，resident tail 虽然不扩大 steady-state 主流水，却必须接在最后一个依赖阶段之后。

同一资源的机制下界通常与 byte-capacity 下界取 `max`，不是无条件相加。首个 descriptor 的 completion latency、后续 ping-pong descriptor 的 stream interval、以及 compulsory DRAM traffic 是不同约束；把它们全部串行相加会重复计价，把它们全部当 steady stream 又会漏掉首用依赖。

所有生产时延修复必须来自源码语义、launch observation 或与 evaluation workload 独立的机制微基准。目标 FA case 的 task/kernel duration 只能用于排序误差和验证变化，不能用来拟合 residual、阈值或 per-shape 修正。

## 3. C500 已闭合机制

### 3.1 Ragged Q64 的 active work 与 physical issue

C500 Q64 ragged grid 中存在源码级 padded CTA early exit。预测时完整 physical grid 仍贡献 launch/issue 下界，但只有真实非空 CTA 进入 attention 主流水。把 padded CTA 全部展开为有效工作会系统性高估 ragged prefill；完全删除 padded CTA 又会漏掉物理发射成本。对应修复为历史提交 `9aaa4030`。

### 3.2 Direct CTA 与 packed persistent queue

Packed Q64 不是一个固定执行模型。实际 launch metadata 能证明两类路径：direct CTA grid 应按离散 CTA/wave 组成，较小固定 worker pool 承载 padded logical jobs 时才按 persistent queue 组成。将所有 packed Q64 一律展开为 direct CTA，或一律按 persistent queue，都与 launch 语义冲突。相关修复为 `3d6bcb38`、`fff71168`；launch selector 和 observation normalizer 必须继续保持这一区分。

### 3.3 D128 Q64 CTA barrier

D128 Q64 source 中存在明确 CTA barrier，不能用普通 arithmetic/traffic 吞吐隐式吸收。独立微基准在 256 threads、32 KiB dynamic shared memory、每 AP 两个 resident CTA 的生产拓扑下测得 `56.640625 cycles / CTA barrier`；冻结验证为 0.034% MAPE、0.309% Max APE。该参数位于 `configs/hardware/metax_c500.yaml`，微基准位于 `microbench/metax_c500/runtime/synchronization/`，原始报告为 `profiler_output/microbench/metax_c500/cta_barrier_cc95ff7f_20260828_r1/report.json`（本地产物，未被 git 跟踪）。相关提交为 `00e82fd7`、`d75717b9`。

### 3.4 Issue 只能计价一次

C500 的 issue instruction 既可能出现在 aggregate resource lower bound，也可能出现在显式 CTA DAG。两处同时加入会重复收费。当前语义是在拥有显式 CTA/source DAG 时由该 DAG 承担 issue 依赖，aggregate 只保留尚未被 DAG覆盖的资源约束。相关修复为 `98e6efec`。

### 3.5 Resident/page-table tail

Resident tail、page-table load 等工作是最后一批 source task 的依赖尾部，不是经验 kernel residual。它们应连接到实际 producer/consumer 之后，再与相应资源容量下界组合；不能简单加到所有 CTA，也不能仅按总 bytes 平摊进 steady-state。相关修复为 `b26b28cd`，strict Graph golden 下 C500 最终达到 13.9% MAPE、33.2% Max APE。

## 4. 910B 已闭合机制

### 4.1 AIV SyncAll 与 mixed-task outer envelope

CANN FIA source 要求所有 active AIV 在首段主工作之前执行一次 `SyncAll`。独立 mixed 1-AIC:2-AIV 微基准得到：

```text
SyncAll(active_aiv_workers)
= 265.326 ns + 59.387 ns * active_aiv_workers
```

它是 source-mandatory phase，不是对目标 kernel 拟合的 residual。正式证据为 `microbench/ascend_910b/references/fia_aiv_sync_all_cann_8_5_1.json`，相关提交为 `158e2ac8`、`f74a2635`、`5a44d2ee`。

更大的早期漏项是完整 mixed-core task 在源码可见阶段之外的外层 envelope。独立实验用 complete task event time 减去同时测得的 `InitOutput + SyncAll` 内部区间，得到：

```text
outer_envelope(active_aiv_workers)
= 5213.005 ns + 55.681 ns * active_aiv_workers
```

该 envelope 只能在 source DAG 外组合一次；完整 empty-task floor、SyncAll、控制链和 outer envelope 不能互相重复相加。正式证据为 `microbench/ascend_910b/references/fia_mixed_task_outer_envelope_cann_8_5_1.json`，相关提交为 `18ee8f49`、`67af78ca`。这是本轮最大的单项改进，使 910B MAPE 约从 26.8% 降到 19.6%。

### 4.2 Source-ordered FIA DAG

910B FIA 不能只将全局 Cube、Vector、MTE2、MTE3 资源时间取最大值。当前模型按 CANN source 建立 actor/domain 有序 DAG，分别表达 AIC/AIV tiling materialization、MTE2 input、MTE1 load pair、Cube、Fixpipe、Vector、workspace transfer 和三次 AIC/AIV handoff。

AIC 与 AIV 的 1560-byte tiling materialization 同时开始，各自只 gate 本 domain 的第一个 source stage；不能把两个 domain 串行。AIC completion 为 12.222 ns，AIV 为 `42.420 + 37.246 * active_aiv_workers ns`。证据为 `fia_tiling_data_materialization_cann_8_5_1.json`。

MTE1 的 L1→L0 load pair 使用 source-shaped completion lower bound，并与同 stage byte-capacity time 取 `max`。D64/D128 下 MM1 分别约为 81.092/161.663 ns，MM2 分别约为 88.328/161.667 ns。证据为 `aic_mte1_load_pair_cann_8_5_1.json`。

空的 MM1→Vec1→MM2→Vec2 三次 handoff ring 只提供整个 source program span 的锚定约束，不是三个可分别相加的固定延时。正式证据为 `fia_cross_domain_three_handoff_cann_8_5_1.json`；当前配置将其标记为不直接消费 duration，避免把 whole-program span 与已展开 DAG 重复收费。

相关实现提交为 `7641ef0c`、`8ef12f92`、`bce59f74`。这些机制与后续 AIV workspace 完成约束一起将 910B MAPE 推进到约 16.7%。

### 4.3 SplitFuse MTE2 的首用与稳态

SplitFuse 的 MTE2 descriptor 不是全部串行，也不是全部 steady stream。首个 descriptor 必须等待真实 completion，后续两个 buffer 交替的 descriptors 使用 back-to-back stream interval，最后 drain。把所有 descriptor 串行会明显过慢；把首个 descriptor 也按 stream interval 处理会漏掉 first-use dependency。相关修复为 `6cad1d5c`、`4f652ed0`。

### 4.4 AIV workspace completion

Vec1/Vec2 workspace load/store 链存在 source-byte-matched completion 下界。它应与同一工作量的 read+write bandwidth 下界取 `max`，不能作为 additive correction 再叠加到完整 Vector/MTE pipeline。正式参数在 `configs/hardware/ascend_910b.yaml` 的 `fia_aiv_workspace_transfer_service`，相关提交为 `9557e32d` 至 `5e115bbc`。

### 4.5 FlashDecode terminal prelude

FlashDecode terminal 路径在最终 reduction 前有独立的 14 个 AIV buffer reset/init、两次 instruction-cache preload 和一次全 actor `SyncAll`。该 phase 与普通 FIA 开头的 output barrier 不同，必须只在 FlashDecode terminal 路径启用。相关提交为 `9e8c40dd`、`a2849d05`；加入后 910B MAPE 约为 15.9%。

### 4.6 Row reduction 范围

`CalcLocalRowSum` 的实际列数可到 1024；只覆盖较短列数会低估长 reduction。扩展独立 reduction evidence 和配置范围后，910B 达到 15.7% MAPE、52.1% Max APE。相关提交为 `743f049a`、`f2865038`。

### 4.7 BSH K/V strided ND2NZ

标准 GQA 的 BSH K/V 源行并非按 `D` contiguous，而是按 `HKV * D` stride。当前模型对首个 descriptor 使用 serial completion，对后续 ping-pong descriptors 使用 stream interval；strided local L2→L1 service 与 compulsory DRAM traffic 是同一执行的两个下界，应取 `max`，不能相加，也不能把 DRAM 冷流量伪装成 local MTE2 service。

有效的 strided probe 使用不同 worker partition 上互不重叠的源地址，总 footprint 为 128 MiB，在 warm-up 后可驻留于公开的 192 MiB shared L2。相关提交为历史 `4a1cdee7`、`285e6dd9`、`3ee3fae1` 及当前 `5c2dc705`、`b43d5c46`。这项物理修正使整体 MAPE 从此前 15.7% 变为 16.0%，但修正了错误的 contiguous/local-service 语义；不能因为 evaluation 数字轻微回退就恢复错误公式。

### 4.8 Cold MTE2 的 resident 与 DRAM-miss 约束按 max 组合

`aic_mte2_nd2nz_cold_completion` 此前按 `resident + miss_fraction * (cold - resident)` 相加。同一个 descriptor 的 L2 命中行与 DRAM 缺失行走的是同一条 MTE2 流：缺失行同样要经过 L2→L1，所以把完整 resident completion 再加上一份 DRAM 代价会对缺失行的 L2→L1 段重复计价。§4.4 与 §4.7 对同一对约束已经确立了 `max` 规则，本节把 FIA cold 路径也统一到该规则。

同时 resident 项改用几何匹配的 strided 服务，而不是 `max(等字节的 contiguous serial, strided)`：等字节的 contiguous 条目描述的是另一种 descriptor 形状，不是本 descriptor 的替代下界。

独立验证：本轮 drained descriptor 探针在 100% DRAM-cold、8 AIC、`256x128 stride 1024` 上测得 1243.9 ns/descriptor，即每 AIC 52.7 GB/s、合计 **421 GB/s**；而生产 `FLASHATTN_B1_S8192_HQ32_HKV8_D128_C0` 由其自身 missed bytes 与 `aic_mte2_ratio` 推出的合计 DRAM 速率为 **409 GB/s**，二者相差 3%。相加形式无法同时满足这两个数，`max` 形式可以。

该修正只影响走 cold-completion 分支的 descriptor（334 行中 5 行）：`PAGEDVARLEN_SKEW_B128/B96/B64_S512` 从 35.7%/33.7%/18.0% 降到 5.5%/4.6%/6.5%，而 `B32/B16_S512` 从 2.2%/13.7% 变为 16.4%/26.3% —— 后两者落进本来就存在的低估主体（见 §7.9），不是新缺陷。整体 910B FLASHATTN 从 17.28% 降到 **17.15%**，P90 34.6→34.2，short KV 20.65→19.99，paged_varlen 20.41→19.99；A800 与 C500 逐位不变。

## 5. A800 当前状态

A800 从起始 12.3% MAPE、42.2% Max APE 变为 11.2% MAPE、44.4% Max APE。变化主要来自 rebase 后吸收的已有集成改动，本分支没有完成一轮可单独归因的大规模 A800 FA 机制修复。

A800 当前整体 MAPE 已低于 15%，但最大 case 仍超过 40%。此前逐 CTA 全展开方案不满足所需的 analytical 执行速度，不应恢复。若继续处理 A800，应保持共享的 source/launch/pipeline 抽象，用按同构 CTA 类别聚合、wave/尾波和有限并发的闭式组成代替逐 CTA 枚举。

## 6. 已否定的建模方式

- **用 production duration 校准 residual、斜率或 per-shape 参数**：违反零拟合边界，即使能降低当前 MAPE 也不能进入 Forward。
- **把 910B MTE2 descriptor 全串行**：模型过慢，曾得到约 17.8% MAPE，且不符合源码双 buffer/ping-pong 时序。
- **把所有 MTE2 descriptor 全按 stream interval**：漏掉第一个 consumer 前必须完成的首用依赖。
- **把 mixed empty floor、outer envelope、SyncAll、tiling 和完整控制链全部相加**：这些量的计时边界互相包含，会重复计价。
- **把 whole-program handoff span 拆成三个固定 additive handoff**：微基准只测得完整 ring span，没有证明每一段可独立拆分。
- **把 C500 packed Q64 统一当 direct CTA 或统一当 persistent queue**：两者都忽略实际 launch/grid 的两类语义。
- **C500 同时在 aggregate lower bound 和 CTA DAG 计 issue**：同一指令重复收费。
- **910B strided probe 让多个 core 共享同一源地址**：会产生非生产语义的跨核 L2 reuse，不能代表独立 worker stream。
- **用 512 MiB disjoint footprint 推导 local MTE2 service**：footprint 超过 shared L2，混入 compulsory DRAM 冷流量，不能作为 local L2→L1 参数。
- **把 910B 最差短 KV 残差归为普通 launch latency**：独立 AIC Init/buffer/event startup 的稳定中位数约 0.42 µs，远不足以解释约 14 µs 的缺口。
- **按 evaluation MAPE 选择物理候选**：例如 strided ND2NZ 修正略微增加整体 MAPE，但证据表明旧 contiguous 假设错误；最终误差只能评价，不能决定保留哪套机制。
- **按 aggregate stage 求和的大小挑修复目标**：910B FA 的 fixpipe stage 求和比实测 occupancy 高 8.6×，但敏感性实验证明它完全被 overlap、不在关键路径上（乘 0.116 后 MAPE 反而从 17.28% 变 17.38%）。stage 求和是诊断量，不是关键路径。
- **把 910B FA 残差当成单一速率问题**：334 行中 216 行低估、118 行高估，中位带符号 −6.69%。全局调整 MTE2 服务只会在两端之间搬运误差（×0.5 时 MAPE 17.28→30.42）。
- **假设 910B FIA 漏掉了 K 的下一 tile 预取**：source 中该预取以 `qkvDSameFlag == false` 为条件，标准 GQA/MHA 不满足，逐 descriptor drain 是 source 正确的。

## 7. 当前主要剩余误差

本节于 `1b6bf7206` 基线重测，取代此前基于 294-case 的陈述。评测口径为 **334 行**（294 个带 per-kernel 记录的 case + 40 个只有 op-level 证据的 `FA_SERVING_*`），910B FLASHATTN/EAGERATTN Forward MAPE **17.15%**（本轮起始 17.28%）、P90 34.2%、Max APE 72.5%，最差 case 为 `FLASHATTN_PAGEDVARLEN_SKEW_B8_S8192_HQ32_HKV8_D128_C0`（actual 88760 ns、predicted 153123 ns）。对照 A800 6.01%、C500 7.92%。

§7 此前写的最差 case `FLASHATTN_B2_S512_HQ32_HKV8_D64_C0` 与桶分布（short KV 25.7%、short Q 21.2%、contiguous 19.2%、decode 18.2%）已经过时，不要再引用。当前桶分布为 short Q 28.29%、very-long KV 23.82%、short KV 20.65%、paged_varlen 20.41%，而 long Q 仅 3.84%。

### 7.1 残差是双向的，不能用单一速率解释

334 行中 118 行高估、216 行低估，中位带符号误差为 **−6.69%**。两端的结构相反：

| 桶 | n | MAPE | 中位带符号 |
|---|---:|---:|---:|
| short Q (2–512) | 20 | 28.29 | −29.18 |
| short KV (≤512) | 66 | 20.65 | −18.09 |
| very-long KV (≥8193) | 10 | 23.82 | +25.36 |
| long Q (≥2049) | 14 | 3.84 | +1.07 |

也就是说：**主体是低估，尾部是高估**。把任何一个全局服务速率调快都会同时改善高估尾部、恶化低估主体。这一点已用受控实验证实（见 7.3），因此"910B MTE2 偏慢、调快即可"是错误结论，不要再按这个方向尝试。

### 7.2 流量 demand 不是误差来源

Forward 的 MTE2 与 DRAM 字节需求与 canonical counter 逐 case 吻合到 **1% 以内**，包括全部高估最严重的 case：

| case | 带符号误差 | 模型/实测 `gm_to_l1` | 模型/实测 `dram_read` |
|---|---:|---:|---:|
| `FLASHATTN_B1_S512_HQ32_HKV8_D128_C0` | −6.68% | 0.96 | 0.99 |
| `FLASHATTN_B1_S4096_HQ32_HKV8_D128_C0` | +32.06% | 1.01 | 1.00 |
| `FLASHATTN_B1_S8192_HQ32_HKV8_D128_C0` | +48.61% | 1.01 | 1.00 |
| `FLASHATTN_B1_S32768_HQ32_HKV8_D128_C0` | +50.56% | 1.01 | 1.00 |

因此 910B FA 的时延残差**不属于**流量 demand 问题，全部落在时间组成上。

### 7.3 已被本轮证伪的候选

以下候选都有量化证据，不要重复尝试：

- **指令 cache 冷填**：canonical counter 的 `aic_icache_miss_rate` 恒为 `0.0`。
- **频率下降（DVFS）**：per-block layer 显示整个 B1→B32 扫描中 AIC 时钟恒为 1.800 GHz，增长的是 cycles 而不是每 cycle 时间。
- **通用 task dispatch 的 per-block 代价**：纯 vector kernel（`op_type=vector`）在 `block_dim` 1→48 上没有 floor 增长（最小 3040 ns → 3840 ns）。per-block 效应只出现在 `op_type=mix` 上。
- **Fixpipe 服务偏慢**：aggregate stage 求和确实比实测 occupancy 高 **8.6×**（B1_S8192：模型 45720 ns/AIC，实测 `aic_fixpipe_ratio` × duration = 5318 ns/AIC），但这是 aggregate-vs-occupancy 的对照假象，不是预测驱动量。把 fixpipe 服务整体乘 0.116 后，MAPE 从 17.28% 变为 **17.38%**（Max 72.5→72.2）——fixpipe 在 source DAG 中被完全 overlap，从不在关键路径上。**先做敏感性实验再改服务**，不要按 stage 求和的大小挑修复目标。
- **整体加快 MTE2**：MTE2 确实是关键路径（把全部 ND2NZ 服务乘 0.5，MAPE 17.28→30.42、中位带符号 −6.69→−32.49、Max 72.5→53.6）。但正因为主体是低估，任何全局 MTE2 提速都会把主体推得更差。MTE2 是 load-bearing 的资源，却不是可以用单一系数关闭的误差。
- **exact-count 门是瓶颈**：见 7.4，实测为否。
- **AIC scalar 控制前端随 active AIC group 数增长，解释低估主体**：按冻结的 mix 1-AIC:2-AIV 探针实测（groups 1/4/8/16/22/24 × iters 0/16/64/256，各 3 个 fresh process），source slot lifetime 控制骨架的服务为 **7.1875 ns/iteration**，在 1→24 groups 上的跨度仅 **1.09%**，`iters=256` 的 6 个 group cell 全部为 1860 ns、process 跨度 0.00%。它**完全不随 active group 数增长**。按生产规模折算：`FLASHATTN_B1_S8192_HQ32_HKV8_D128_C0` 每 AIC 约 66 个 descriptor，控制骨架合计约 0.5 µs，而该 case 实测 51096 ns——量级差两个数量级，不可能解释 short Q / short KV 每多 8 个 block 增加约 10 µs 的行为。该候选按我提出的形式被否决。
- **MM1 的 K 流与 MM2 的 V 流跨边界重叠，使 MTE2 维持约两个 outstanding descriptor**：按冻结的双流 cell 实测（geometry 256x128 stride1024、count 64、groups 8/16/24、各 3 个 fresh process），`streams=2` 与 `streams=1` 的每 descriptor 服务分别为 1245.0 vs 1243.9、1573.4 vs 1556.9、2302.0 vs 2302.7 ns，全部在预先声明的 5% 判据之内。drained descriptor 链确实是串行的，该候选被否决。
- **cross-domain three-handoff ring span 是高估来源**：把 `fia_cross_domain_three_handoff_time_ns` 整体置 0，或只保留单 task 跨度、去掉 per-task steady 外推，MAPE/P90/Max/各桶**一位不变**（17.28 / 34.6 / 72.5）。该 ring span 在当前 334 行上从不成为 `max()` 的胜者，是完全惰性的约束，不要再把它当作调节量。

### 7.4 drained descriptor 的 count-independence（本轮新增机制）

CANN FIA 在 `qkvDSameFlag == true`（标准 GQA/MHA，Q/K/V 同 `D`，即本数据集绝大多数 shape）时跳过 `pfa_service_matmul_kv_nd.h` 中 `mL0 == 2` 的下一 tile K 预取，`CopyKToL1`/`CopyVToL1` 之后紧跟其 `WaitFlag<HardEvent::MTE2_MTE1>`。因此**逐 descriptor drain 是 source 正确的**，之前怀疑的"漏掉 K 预取"不成立。

已发布的 `aic_mte2_fia_gqa_finite_drain*` 证据声明为 `exact_measured_geometry_and_descriptor_count_only`，Forward 不得插值。本轮用独立微基准测量了同一 drained 边界在固定 geometry/并发下对 descriptor count 的依赖：在 count 8→128（16 倍）范围内，每 descriptor 服务的跨 count 跨度为 **0.79%–6.16%**，全部通过预先声明的 8% 门。据此发布 `aic_mte2_fia_gqa_drained_descriptor_count_law_cann_8_5_1.json`，它只授权把已发布的 exact-count 程序按 descriptor 应用到该范围内的任意 count，**不提供绝对服务水平**（绝对值仍取自 exact-count 参考），也不授权跨 geometry 或跨并发插值。

40 个 cell 中 3 个（均为 d64 geometry）未通过 5% 的 process-median 稳定门，其所属的两个 (geometry, groups) cell 已整体排除，只发布 6 个 cell。微基准与冻结计时边界见 `microbench/ascend_910b/operators/flash_attn/gqa_drain_count_law/`。

**该机制当前对评测是惰性的**：接入后 1027 个 case 中预测值发生变化的有 **0 个**——FIA source program 产生的 descriptor run 本来就落在已发布的 exact count（8/16/32 等）上，exact-count 门并不是瓶颈。它是一个 fail-closed 的正确性扩展与已验证的边界，不是本轮的精度改进。

### 7.5 低估主体的真实瓶颈：固定的每 block AIC scalar 时间

把 canonical counter 的 `aic_scalar_ratio` 乘以实测 duration 换成绝对时间后，低估主体的瓶颈是明确的。在 `contiguous D128 HQ32 HKV8 decode`、`block_dim=16`、每 block 1 个 source task 上：

| B | KV | duration µs | **scalar busy µs** | mte2 busy µs | cube µs |
|---:|---:|---:|---:|---:|---:|
| 2 | 512 | 25.78 | **14.86** | 6.43 | 0.43 |
| 2 | 1024 | 27.38 | **13.23** | 10.46 | 0.78 |
| 2 | 2048 | 32.40 | **13.21** | 16.50 | 1.35 |
| 2 | 4096 | 39.62 | **13.17** | 26.13 | 2.27 |

AIC scalar busy 在 KV 512→4096（工作量 8×）上几乎是常数 **13.2 µs**。用同 `block_dim` 的两对 case 分离固定项与每 task 项：`block_dim=16` 得固定 13.12 µs + 每 task 1.74 µs；`block_dim=22` 得固定 14.40 µs + 每 task 2.98 µs；`block_dim=8` 的固定项约 8.5 µs。

该两点分离已用全部 200 个有 per-invocation 记录的 decode case 复核。在固定 `(block_dim, tasks/block)` 的 cell 内按每 block descriptor 数分桶，scalar busy 在很宽的工作量区间上确实是一个地板：

| cell | scalar 地板 | 地板成立的 desc/block 区间 |
|---|---:|---|
| `bd=8, t=1` | 8.0–8.5 µs | 30→100 |
| `bd=16, t=1` | 13.0 µs | 30→180 |
| `bd=16, t=2` | 15.5–16.6 µs | 50→220 |
| `bd=22, t=3` | 21 µs | 80→520（6.5×） |
| `bd=22, t=6` | 29 µs | 150→320 |
| `bd=24, t=11` | 48 µs | 270→590 |

两点分离的外推与之一致：`bd=16` 预测 14.86 / 16.60 µs（实测中位 14.07 / 16.60），`bd=22, t=3` 预测 23.3 µs（实测 21–24）。

两点需要注意。其一，descriptor 数足够大时会出现另一个与工作量成正比的区段并盖过地板（`bd=22, t=3` 在 desc/block 1030 处为 62.4 µs，`bd=22, t=6` 在 2060 处为 114.6 µs），所以这不是纯固定项，而更接近 `max(地板, 与工作量成正比项)`。其二，`aic_scalar_ratio × duration` 与 duration 本身的相关系数为 0.958，每 descriptor 的 scalar 时间跨度达 16.5×——这个量不能当作一个干净的、可直接消费的机制量，它在地板区间内稳定，在别处不稳定。

这一项解释了低估主体的全部特征：它在 short KV 占 duration 的 54%（模型完全没有这一项，§4.2 的 AIC tiling materialization 只有 12.222 ns）；在 `B1/KV8192` 被 MTE2 的 41.1 µs 盖过（scalar 仅 13.15 µs），所以 long KV 准确；工作量翻倍时它不动，所以同 `block_dim` 下 duration 平在 1.9% 以内；它也正是 `aic_scalar_ratio` 在这些桶为 0.52–0.68 而在准确桶只有 0.25 的原因。它还随 active block 数增长（8.5 → 13.2 → 14.4 µs，对应 `block_dim` 8 → 16 → 22）。

### 7.6 该固定项不是 scalar 指令发射（反汇编取证）

生产 kernel 二进制为 `FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da.o`（CANN 8.5.1，SHA256 `743e08ea63810b669400f3988c83e373ee9166c56b900e6b866b16ed16dee1b4`），与 canonical dataset 的 kernel 名逐字一致。数据集实际发射的 tiling key 对应符号 `..._103000000010622221_mix_aic`，位于 `.text` 偏移 `0x8fa314`，长度 `0x3398` = **13208 字节**；`llvm-objdump` 与 `msdebug`(lldb) 两个工具都按 **4 字节定长**逐条列出，共 **3302 条指令**（13208/3302 = 4.0）。对应的 AIV 符号为 `..._103000000010622221_mix_aiv`，长度 `0x73ac` = 29612 字节。

由此得到一个不需要解码指令就成立的量：整个 AIC 函数**至多** 3302 条静态指令，若每条一周期发射，在 1.8 GHz 下把函数完整走一遍是 **1.84 µs**。而 §7.5 测得的固定项是 8.5 µs（`block_dim=8`）到 14.4 µs（`block_dim=22`）。

**订正一处我自己先前的论证。** 早先把它写成「固定项比整个 kernel 的指令总量还多 4.6 至 7.8 倍，所以不可能是指令发射」，这个推理是不严谨的：函数里有循环，**动态**指令数并不被静态条数 3302 界住，走 5 到 8 遍循环体就能凑出那么多发射。正确的论证是另一条，而且更强：循环次数随工作量增长，动态指令数也随之增长，但**固定项在 KV 512 到 4096 上是平的**（§7.5：`bd=16, t=1` 上为 14.86、13.23、13.21、13.17 µs，工作量 8 倍而几乎不动）。一个随工作量增长的量解释不了一个不随工作量变化的地板，因此固定项不是指令发射。静态条数 3302 仍然有用——它说明不存在一段几千条指令的巨型前导段——但它本身不是那个论证。

**因此这个固定项不可能是 scalar 指令发射，只能是 scalar 流水线内部的阻塞/延迟。** 这同时解释了为什么 §7.8 的 mix 探针只测到 7.1875 ns/iteration：scalar 的吞吐确实很快，`aic_scalar_ratio` 计的是"scalar 流水线被占用"，包含等待，不是指令条数。

**用差分解码绕过缺失的解码器。** CANN 8.5.1 在本机没有 AI core 指令解码器（`llvm-objdump` 的 `hiipu64` target 对全部 3302 条指令输出 `<not available>`，各 `--mcpu` 均同；`msobjdump` 只做 fatbin/ELF 信息；`msdebug` 助记符为空；`ccec -S` 直接报 `unsupported option '-S'`）。但不需要完整解码器也能认出特定指令类：用 `ccec` 以同一份 AscendC 探针、仅改展开次数 `NREP` 编译两次，对 `.text` 的四字节小端字做直方图差分，计数恰好变化 `ΔNREP` 的那一类就是被测操作；寄存器位在低位，因此要按高位前缀聚合而不是整字。方法与脚本见 `microbench/ascend_910b/operators/flash_attn/isa_differential_decoding/`。

已识别（CANN 8.5.1、`dav-c220-cube`、-O3）：`GlobalTensor<int64_t>::GetValue` 的依赖式标量 GM 读为每次迭代 1 条指令，整字 `01c41100`、类为高 10 位等于 `7`；`GlobalTensor<int32_t>::GetValue` 连同索引运算为每次迭代 4 条，前缀为 `414`、`028`、`038` 与 `000`。两者在 `ΔNREP = 16` 上都给出干净的 `+16`。

把这些类数到生产 kernel 的 AIC 函数（3302 条指令）上：

| 类 | 条数 | 占比 | 按八分位分布 |
|---|---:|---:|---|
| int64 标量 GM 读（高 10 位 `7`） | 1 | 0.0% | `[0,0,1,0,0,0,0,0]` |
| int32 读族 `414` | 55 | 1.7% | `[0,5,10,8,3,11,10,8]` |
| 族 `028` | 176 | 5.3% | `[3,12,30,30,29,23,33,16]` |
| 族 `038` | 20 | 0.6% | `[7,7,0,2,1,2,1,0]` |

**订正：先前「没有密集的标量 GM 读前导段」的结论是错的。** 我最初只识别出一种标量 GM 读编码（依赖式寄存器索引 `01c4`），据此数到 1 条，并由此写下「无前导段聚集」。继续做差分后发现标量 GM 读至少有三种编码，取决于索引形式：依赖式寄存器索引为 `01c4`、计算出的寄存器偏移为 `414`、常量立即偏移为 `080` 与 `09c`/`09d`。按全部四类重数：

| 类 | 条数 |
|---|---:|
| `080` | 138 |
| `414` | 55 |
| `09d` | 19 |
| `09c` | 17 |
| **合计** | **229（6.9%）** |

按八分位分布为 `[82, 33, 17, 18, 16, 19, 18, 26]`——**明显前置**，229 条里有 115 条落在前四分之一，前 256 条指令内就有 43 条。这正是「开头集中读 tiling 结构体」的形态。按 §7.8 实测的每读约 100 ns，229 条完全串行为 **22.9 µs**，而地板是 8.5 µs（`bd=8`）到 14.4 µs（`bd=22`）——量级第一次落到了同一区间内。

**先前的否决确实用错了 regime，但在正确 regime 下重测仍然否决，且理由不同。** §7.8 的并发扫描用 worker 互不相交的地址，测得每读约 100 ns、并发只涨 1.187 倍。生产里 24 个 AIC 读的是同一份 1560 字节 tiling 结构体，是共享行；据此把该候选重新打开，并按冻结 cell 改成「所有 core 走同一段 4 KiB 依赖链」重扫（`gm_reads` 0/512 × groups 1/8/16/22/24，各 3 个 process）。

| groups | 共享行每读 | process 跨度 |
|---:|---:|---:|
| 1 | 7.42 ns | 5% |
| 8 | 7.85 ns | 81% |
| 16 | 6.17 ns | 31% |
| 22 | 7.46 ns | 19% |
| 24 | 6.88 ns | 121% |

`c(16)/c(8) = 0.79`，远低于预先声明的 1.25，**候选第二次被否决**，这次是在生产实际使用的 regime 下。更关键的是绝对值：共享小区域被所有 core 反复读，本来就是 cache 驻留的，每读约 7 ns，比互不相交的冷读便宜 13 倍。按 7 ns 计，229 条读合计只有 **1.6 µs**，而不是按冷读 100 ns 估出的 22.9 µs——那个量级估计用错了速率。地板 8.5 到 14.4 µs 仍然没有着落。

诚实记一笔：这三个 process 的跨度在 19% 到 121%，远未通过 5% 稳定门，因此上表不作为服务值发布。但结论对这点噪声是稳的——共享读比冷读便宜一个数量级，且不随并发增长，两者的效应量都远大于跨度。

因此 §7.10 的 G6 判定恢复，已否决候选回到八项。

边界：指令类是由 `ccec` 对该构造在该优化级与架构下的产物识别出来的，其他寻址形式或宽度可能使用尚未识别的编码，因此条数是下界；而「按位置散布、无前导段聚集」这一结论不依赖于是否穷举了全部编码，是更稳的那部分。

仍未解决的是：`llvm-objdump` 注册了 `hiipu64` target，但对全部 3302 条指令一律输出 `<not available>`（`--mcpu=dav-c220-cube/-vec/dav-c220/dav-m200/dav-c100` 均相同）；`msobjdump` 只是 fatbin/ELF 信息工具，没有反汇编功能；`msdebug`(lldb) 能解析地址但助记符为空。C500 那条路径（反汇编 device 镜像读出 operand load path）在 910B 上不成立，不是方法问题而是工具缺失。要继续需要带指令表的 ccec 工具链，本机没有。

### 7.7 AIC scalar 控制骨架与 mix-task envelope 的实测边界

为分辨"低估主体来自 AIC scalar 控制程序长度"与"来自 mix-task envelope 被低估"两种解释，本轮用一个 mix 1-AIC:2-AIV 探针**独立扫描 group 数与控制程序长度**，两者互不耦合。冻结定义、cell 与门见 `microbench/ascend_910b/operators/flash_attn/aic_scalar_control_floor/REGISTRATION.md`。

两种解释都被证伪：

- **控制程序长度**：slot lifetime 控制骨架为 7.1875 ns/iteration，跨 1→24 groups 跨度 1.09%，与 group 数无关。生产规模下合计约 0.5 µs。
- **mix-task envelope**：`iters=0` 的 host span 中位数在 groups 1→24 上为 27660→29000 ns，全域范围只有 1.34 µs 且**没有随 group 数单调增长**；而低估主体所需的增长约为 27 µs。host span 未通过预先声明的 5% 稳定门（3 个 process 下 aclrtEvent 噪声最大到 82%），因此**不发布任何 envelope 参数**，上述只能作为量级上的反证，不能作为精确测量。

正式证据为 `microbench/ascend_910b/references/aic_scalar_control_slot_lifetime_cann_8_5_1.json`，其 `forward_eligible` 显式为 `false`：该值真实可复测，但只覆盖最小控制骨架、不覆盖 FIA source 的完整 per-iteration scalar 程序，且量级上不可能改变任何预测，故**不接入 Forward**，只作为证伪证据保留。

因此生产 counter 里 `aic_scalar_ratio` 的 0.52–0.68 仍然没有被解释：本轮只证明它不是由 slot lifetime 控制骨架、也不是由 active group 数驱动的。要继续这条线，需要 FIA kernel 二进制的实际 scalar 指令数，或从 source 推出的 per-iteration scalar 指令计数——两者本轮都不具备。

### 7.8 低估主体：unsupported 判定（G6）

**结论：910B FA 低估主体（short Q、short KV、高 B，即 scalar 地板占优的区间）在当前证据下无法用被允许的来源关闭。** 这是一个针对该子问题的 unsupported 判定，不涵盖 §7.9 的高估尾部，也不是对整个 910B FA 残差的 WONTFIX。

**受影响的域边界。** `aic_scalar_ratio × duration` 形成地板的区间，即每 block descriptor 数低于约 500–1000 的 decode launch。该区间覆盖 short Q（20 行，MAPE 28.29%）、short KV（66 行，20.65%）与高 B 的 paged_varlen。descriptor 数更高时另一个与工作量成正比的区段接管，模型在那里已经准确（long Q 3.84%）。

**要关闭它需要的量。** 地板本身：`bd=8,t=1` 8.0–8.5 µs、`bd=16,t=1` 13.0 µs、`bd=16,t=2` 15.5–16.6 µs、`bd=22,t=3` 21 µs、`bd=22,t=6` 29 µs、`bd=24,t=11` 48 µs（§7.5 已按 200 个 case 扫过）。它随 active block 数增长：固定 `tasks=1` 时 `bd` 8→16 使地板从 8.0 涨到 13.0 µs，即 1.63×。

**为什么它不能直接进 Forward。** 这些数值全部由 `aic_scalar_ratio × 实测 duration` 得到，是 production duration 的派生量。按零拟合底线，它只能用于排序误差和验证变化，不能作为 Forward 参数、residual 或 LUT。必须由独立来源重新取得。

**本轮否定的候选及其证据（十项）。**

- **scalar 指令发射**：生产 kernel 的 AIC 函数至多 3302 条指令（13208 字节，两个工具均按 4 字节定长列出），全部执行一遍在 1.8 GHz 下仅 1.84 µs，比地板小 4.6–7.8 倍（§7.6）。
- **source slot lifetime 控制骨架**：独立 mix 探针测得 7.1875 ns/iteration，跨 1→24 groups 跨度 1.09%，与 group 数完全无关；生产规模下合计约 0.5 µs（§7.7）。
- **mix-task outer envelope 被低估**：同一探针的 `iters=0` host span 在 groups 1→24 上为 27660→29000 ns，无单调增长，全域范围 1.34 µs，而所需增长约 27 µs（§7.7）。
- **序列化 scalar GM 读**：按冻结 cell 实测（`gm_reads` 0/32/128/512 × groups 1/8/16/22/24），每读成本为 101.3 ns(g=1) → 120.2 ns(g=24)，`c(24)/c(1) = 1.187`，低于预先声明的 1.25 判据。仅 `bd` 8→16 的地板增长就需要 1.63×，而实测只有 1.05×。该绝对值同时独立复现了已发布的 `aic_scalar_global_get_value_serialized`（93.317 ns）到 8% 以内，说明探针本身是准的。候选被否决。

- **地板其实是 task startup / launch 开销**：把同一个 mix 探针的空任务（`iters=0`、`gm_reads=0`）放到 canonical 评测所用的同一个仪器下测量——`msprof --ai-core=on` 的 `op_summary` `Task Duration(us)` 列，每次 capture 20 次发射、丢弃首次、取中位，3 次 capture——得到 `D_empty(g)`：

  | groups | `D_empty` | 已发布 envelope `E(g)` | `D/E` | capture 跨度 |
  |---:|---:|---:|---:|---:|
  | 1 | 5.31 µs | 5.32 | 1.00 | 2.6% |
  | 4 | 2.57 | 5.66 | 0.45 | 2.6% |
  | 8 | 3.88 | 6.10 | 0.64 | 3.6% |
  | 16 | **6.13** | 6.99 | 0.88 | 5.7% |
  | 22 | 8.22 | 7.66 | 1.07 | 9.5% |
  | 24 | 7.86 | 7.89 | 1.00 | 5.4% |

  `D_empty` 在任何 group 数上都没有超过已发布 envelope 的 25%（预先声明的接受判据），在 `g=4/8` 反而明显低于它。`bd=16` 的空 mix 任务只要 6.13 µs，而 Forward 已经计了 6.99 µs——**13 µs 的 scalar 地板不是 task startup，它确实发生在 kernel 内部**。该候选被否决。顺带地，这条测量在第二个独立仪器（msprof task duration）上交叉验证了已发布的 `fia_mixed_task_outer_envelope`：`g=1/16/22/24` 的比值为 1.00/0.88/1.07/1.00。（一次只有 2 个样本、且含 warmup 的探索性读数曾给出 8.7/17.5 µs，与此不符；以本表的 20 次发射中位为准。）

- **AIC 指令 fetch**：`aic_icache_miss_rate` 在本数据集中是被填充的计数器（vector 与 flash_attn 的 AIV 侧出现过 0.139、0.167、1.657、2.398 等非零值），而 291 个 FA case 的 AIC 侧一律为 `0.0`。AIC 取指未命中确实为零，该候选被证伪；此前的证伪现在有了「计数器确实会非零」的佐证，不再是空字段推论。

- **AIC 与 AIV 之间的 handoff round trip**：这是唯一通过并发判据的候选，但仍未通过幅度判据。用源码自己的 `FIA_SYNC_MODE2`、`CrossCoreSetFlag<2, PIPE_FIX>` 与 `CrossCoreWaitFlag` 构造纯 round trip 探针（无 DataCopy、Mmad、Fixpipe、Vector 与访存），按 `handoffs` 0 到 128 的斜率测得每次 round trip 时延：`g=16` 为 902.5 ns、`g=22` 为 1305.8 ns、`g=24` 为 1442.8 ns（process 跨度 1.18%、1.53%、2.37%，通过 5% 稳定门）；`g=1` 与 `g=8` 的 515.0 与 669.7 ns 跨度为 28.06% 与 11.96%，未通过稳定门，不作为服务值发布。仅用稳定 cell，`h(24)/h(16) = 1.599`，高于预先声明的 1.25——**cross-domain round trip 的时延确实随并发增长**。

  但幅度不够。PFA 源码把 `QK_READY`、`SOFTMAX_READY`、`UPDATE_READY` 三次跨域信号放在 per-KV-block 的 `nIdx` 循环内（约 3 次 crossing，合 1.5 次 round trip 每 KV block），而不是每 task 固定一段。KV=512 时每 task 约 3 到 4.5 次 round trip，按 `h(16)` 折合 2.7 到 4.1 µs，对 13.0 µs 的地板差 3.2 到 4.8 倍，超出预先声明的 2 倍判据。复合判据要求两半同时成立，因此该候选**不能**解释地板，地板仍未闭合。

  该参数本身是合法测得、可复现的独立机制，已发布为 `microbench/ascend_910b/references/fia_cross_domain_round_trip_latency_cann_8_5_1.json`，`forward_eligible` 为 `false`：接入还需要由源码严格推出的每 task round trip 数，且它按 KV block 增长，会给本就高估的 long-KV 尾部再加时间。

- **固定次数的 SyncAll**：这是形态上最贴合的候选——`fia_aiv_sync_all_service` 为 `265.3 + 59.4 × n_aiv` ns，而 `n_aiv = 2 × block_dim`，因此它天然「与工作量无关、随 block 数增长」，正是地板的两个特征。算术上也几乎对得上：若每个 AIC 执行 6 次 SyncAll，`bd=8` 为 7.3 µs、`bd=16` 为 13.0 µs，而实测地板是 8.5 µs 与 13.0 µs。

  用二进制直接数否掉了它。差分识别出 `CrossCoreSetFlag` 为 `41f`（每次 1 条）、`SyncAll` 为 `41f` + `40e` + `41e`（每次 3 条，其中 `41e` 是 SyncAll 特有）。数到生产函数上：AIC 函数里 `41e` 只有 **1** 条、`41f` 3 条；AIV 函数里 `41e` 3 条、`41f` 2 条。每个 AIC 只做一次 SyncAll，不是六次；`bd=16` 下一次 SyncAll 为 2.17 µs，对 13.0 µs 的地板差 6 倍。该候选被否决。

  附带一个正面结果：§4.1 的「每个 AIV 在首段主工作前执行一次 SyncAll」此前只来自源码阅读，现在由二进制里 AIC 侧恰好 1 条 `41e` 独立复证。

- **AIC 在等 AIV**：AIC 只做一次 SyncAll，而 SyncAll 会一直阻塞到最慢的 AIV 到达，所以 AIC 的固定项有可能就是它等掉的 AIV 前导段。用解码器数 AIV 函数（7403 条指令）：第一次 SyncAll 出现在第 **587** 条，即前导段占函数的 7.9%，其中有 62 条标量 GM 读、12 条 flag。587 条指令的发射约 0.33 µs，62 条读按共享命中的 7 ns 算约 0.43 µs、按冷读 100 ns 算约 6.2 µs。即便取冷读上界，AIV 前导段加上一次 SyncAll 在 `bd=8` 约 7.7 µs、`bd=16` 约 8.7 µs，而实测地板是 8.5 与 13.0 µs：**量级接近但增长不对**（预测涨 1.13 倍，实测涨 1.53 倍）。AIV 前导段本身解释不了地板随 block 数的增长。

  但这次数出一个 Forward 层面的真实差异，值得单独跟进：**AIV 函数里有 3 条 `41e`（SyncAll），分别位于函数的 7.9%、28.6%、68.4% 处，而 Forward 只计一次** （`formula.py` 的 `initial_sync_all_s`），另有一次只在 FlashDecode 路径上由 §4.5 的 terminal prelude 单独计。第三处目前没有对应项。按 `265.3 + 59.4 × n_aiv`，多计两次在 `bd=8` 为 +2.4 µs、`bd=16` 为 +4.3 µs——方向对（低估主体需要加时间）且随 block 数增长。

  **已由源码语义澄清：不该加，Forward 现在是对的。** 静态条数不等于执行次数，而 PFA X910 路径的三个 SyncAll 点在源码里各有条件：`InitOutputSingleCore()` 里 `InitOutput` 之后是**无条件** `SyncAll()`——这正是 §4.1 建模、`formula.py` 计的那一次；`InitLseOutputSingleCore()` 里的 `SyncAll()` 由 `isSoftMaxLseEnable == true` 守卫；第三处属于另一个模板特化 / FlashDecode 路径。canonical FA case 的 `Attr` 里没有任何 LSE 请求字段（只有 B、D、HKV、HQ、KV_LEN、Q_LEN、causal、kv_layout、split_policy 等），因此 `isSoftMaxLseEnable` 为假，那一次 SyncAll 不执行。对低估主体的这些 shape，AIV 在 init 路径上就是执行一次 SyncAll，与 Forward 当前一致。该候选**撤回**，不是被幅度否决，而是可达性根本不成立；先前没有按「加上去正好补掉误差」直接采纳，是对的。

**为什么在当前证据下关不掉它。** 三条合法来源全部被实际走过，结果都是否决而不是归因。

*(1) ISA 静态语义——本轮从「工具缺失」变成「已走通并用尽」。* 早先的判定里这条写的是「本机没有 AI core 指令解码器」。那已经不成立：差分解码不需要解码器也能认出指令类（§7.6），本轮据此识别出标量 GM 读的三种编码、`SetFlag`/`WaitFlag`、MTE1 `LoadData`、`SetFmatrix`、`CrossCoreSetFlag` 与 `SyncAll`，并在生产二进制上数出：AIC 函数 3302 条指令、229 条标量 GM 读（前四分之一占 115 条）、56 条 `WaitFlag`（37 条在末尾八分之一）、1 条 `SyncAll`；AIV 函数 7403 条指令，首个 `SyncAll` 在第 587 条。这条路径现在是**开着的**，它给出的是一连串否决：前导段指令便宜（256 条约 0.14 µs）、核内同步稀疏且靠后、SyncAll 只有一次、标量读在共享 regime 下只值 1.6 µs。**它没有给出地板的归因。**

*(2) 算子源码静态语义。* source 里可见的阶段——tiling materialization、InitOutput、SyncAll、slot lifetime、三次 handoff ring——都已建模，且其中四项已被本轮探针逐一证伪。源码还澄清了两件事：`qkvDSameFlag == true` 时逐 descriptor drain 是正确的（§7.4）；第二次 AIV SyncAll 由 `isSoftMaxLseEnable` 守卫、对本数据集不执行。停顿不在 source 暴露的结构里。

*(3) 独立机制微基准。* 要复现一个停顿，必须先知道它停在什么上。本轮能命名并探测的停顿源已全部被否决，每一项都对着采集前写死的判据：

| # | 候选 | 判据与结果 |
|---:|---|---|
| 1 | 指令发射 | 地板与工作量无关而动态指令数随工作量增长（§7.6 订正后的论证） |
| 2 | slot lifetime 控制骨架 | 7.1875 ns/iteration，跨 1→24 groups 跨度 1.09% |
| 3 | mix-task envelope 被低估 | 1→24 groups 上 host span 无单调增长 |
| 4 | 序列化标量 GM 读（互不相交） | `c(24)/c(8)=1.187` vs 判据 1.25 |
| 5 | task startup | 空 mix 任务 `bd=16` 6.13 µs，Forward 已计 6.99 µs |
| 6 | AIC 指令 fetch | `aic_icache_miss_rate` 全 291 case 为 0.0，且该计数器会非零 |
| 7 | AIC↔AIV handoff round trip | 并发判据过（1.599≥1.25），幅度判据不过（2.7–4.1 µs vs 13.0 µs） |
| 8 | 序列化标量 GM 读（共享行） | `c(16)/c(8)=0.79` vs 判据 1.25；共享读 ~7 ns，229 条仅 1.6 µs |
| 9 | 固定次数 SyncAll | 形态完全吻合、六次即 13.0 µs，但二进制里 AIC 只有 1 条 |
| 10 | AIC 在等 AIV 前导段 | 幅度接近（冷读上界 8.7 µs）但增长不对（1.13× vs 实测 1.53×） |
| — | 多出来的 AIV SyncAll | 撤回：`isSoftMaxLseEnable` 守卫，对本数据集不可达 |

第 9 与第 11 项特别值得留着：两者的量级与缩放都几乎正好补上残差，若按「加上去误差就没了」采纳，进 Forward 的会是 kernel 根本不执行的项。这正是零拟合底线要防的东西。

**重开条件 (b) 的兑现：剩余未识别类已全部认出，结论是否定（2026-09-24，wt-910b-fa-duration）。** 用前述 -O0 差分解码流程（`ccec --cce-aicore-arch=dav-c220-cube --cce-aicore-only`，模板递归展开 × 指令字直方图）把重开条件里点名的五个类逐一认出，并对每个类给出「kernel 是否真执行它」的独立证据与收益上限：

| 类 | 识别结果 | 生产 AIC 静态条数 | 是否被执行 | 收益上限 |
|---|---|---:|---|---|
| `03c` | LocalTensor **UB 位置**标量 load（`GetValue` 解引用路径，-O0 每次 14 条含地址生成） | 390 | **否**：CANN 8.5.1 `kernel_tensor_impl.h` 的 `GetValue` 契约——AIC 且位置为 UB 时 `return 0`，是显式 stub | 0 |
| `04c` | LocalTensor **UB 位置**标量 store（`SetValue`，-O0 差分 3:3 精确匹配） | 237 | **否**：同一契约，AIC+UB 时 `return`，no-op | 0 |
| `080`/`09c`/`09d`/`414` | 标量 GM 读族（§7.6 已识别） | 229 | 是，但共享行 ~7 ns/读 | ~1.6 µs（候选 #8 已否决） |
| `720` | 16 条中 15 条为**同一字** `720a9310`，全部落在 AIC 函数前 270 字节的 prologue 内、8 字节等间距 | 16 | 至多每 call 一次 | ~20 ns 量级，五个数量级低于地板 |
| `020` | scalar ALU 族（`dd_branchy` 探针携带 31 条、约 1.3 条/步） | 174 | 是（普通算术/分支） | 发射受限，已被候选 #1（指令发射）整体否决覆盖 |

`03c`/`04c` 的排除有两条独立证据：其一，CANN 8.5.1 契约源码（`asc/impl/basic_api/kernel_tensor_impl.h`，`GetValue`/`SetValue` 的 `if ASCEND_IS_AIC { if (GetPhyType(GetPosition()) == Hardware::UB) return 0 / return; }` 守卫）——UB 位置的标量访问在 AIC 核上是被契约抹掉的；其二，FIA 源码全量审计（kernel、vec/cube block、matmul service、buffer 管理层）：**零处 LocalTensor 标量访问**，源码里所有标量 `GetValue` 都是 GlobalTensor（GM）读，即 §7.6 已计数的 229 读族。生产二进制里的 390+237 条 `03c`/`04c` 是框架模板实例化携带的编译产物（其 UB 形态被契约 stub、AIV 分支不可达）。

**编码分离证明。** 为排除「生产里的 `03c`/`04c` 其实是 L1（A1）位置的有效解引用」这一残留可能，另建 L1 位置探针（`TBuf<TPosition::A1>` + `SetValue`/`GetValue` 依赖链，-O0 与 -O3 各一份）：L1 位置的标量解引用编译为**不同的指令类**（探针字直方图中 `03c`/`04c` 计数为零，L1 解引用落在 `2c`/`30`/`18`/`20`/`10`/`f0` 等 *e 族），因此生产二进制里的 `03c`/`04c` 不可能是 L1 访问。附带测量：AIC-only kernel 上 L1 窗口依赖读链 7.19 ns/step vs 纯算术控制孪生 2.20 ns/step（差值 ~5 ns 为 L1 窗口 load-to-use），且链上读回值为 0——即便把全部 627 条静态 `03c`/`04c` 按最有利口径当作活 L1 访问执行一遍，上限也只有 627 × 7.2 ns ≈ 4.5 µs，仍低于 16.54 µs 的 decode 缺口，且方向上没有任何源码可执行证据支持。

**结论：** 重开条件 (b)——「把 `04c`、`03c`、`080`、`720`、`020` 认出来后指向某个长延迟结构」——已按字面兑现，答案是**没有任何一个类是被执行的长延迟结构**。剩余的重开条件只有 (a) 能分辨执行次数的运行时探针与 (c) 新的独立可复现停顿源。

**边界，按用户的原话划。** 这是「我能命名并探测的候选已用尽、三条来源都走过且都只给出否决」，**不是**「没有人能拿到这个数」。地板本身已经被量化并扫过（§7.5），它只是还没有被归因。若出现下列任一项，本判定应当被重新打开：能分辨执行次数而非静态条数的运行时探针；~~把 `04c`、`03c`、`080`、`720`、`020` 这些仍未识别的类认出来后指向某个长延迟结构~~（2026-09-24 已认出全部五类并逐类排除，见上表）；或任何新的、可独立复现的停顿源。在此之前，不得用任何按误差反推的系数覆盖这个桶——本轮已有两次「数值正好对上但机制不成立」的先例。

### 7.9 仍然未闭合的部分（高估尾部）

高估尾部与低估主体需要**两个不同的机制**，本轮只排除了候选、没有关闭任何一端：

- **低估主体（short Q / short KV / 高 B）**：瓶颈已定位为固定的每 block AIC scalar 时间（§7.5：`block_dim=16` 下固定 13.12 µs + 每 task 1.74 µs，`block_dim=8` 约 8.5 µs，随 active block 数增长），模型完全没有这一项。已排除的解释：不是 scalar 指令发射（§7.6，整个 AIC 函数至多 3302 条指令、全部执行一遍仅 1.84 µs，比固定项小 4.6–7.8 倍），不是 slot lifetime 控制骨架（§7.7，7.1875 ns/iteration 且与 group 数无关），不是 mix-task envelope（§7.7，1→24 groups 上 host span 无单调增长）。剩下的只能是 scalar 流水线内部的阻塞。下一步需要能区分阻塞来源的证据——序列化的 scalar GM 读、instruction fetch、或跨核共享结构上的争用；本机无 AI core 指令解码器（§7.6），因此需要带指令表的 ccec 工具链，或一个能独立复现"固定的每 block scalar 阻塞随 active block 数增长"的探针。

- **高估尾部（very-long KV / paged_varlen）**：关键路径为 source compute DAG，其中 MTE2 是唯一 load-bearing 的资源（fixpipe 与 ring span 均已实测证伪为惰性）。字节正确、drain 语义 source 正确、count 依赖已测。剩下的矛盾是：已发布的 drained 服务在 8 AIC 上为 1388 ns/64 KiB descriptor（blended 约 1096 ns），而生产实测约 618 ns/descriptor，快约 1.8×；且模型/实测 MTE2 occupancy 比值随 KV 长度从 1.43 单调升到 2.27，说明这是**并发效应而不是固定系数**。最可能的物理来源是 FIA 每个 task 同时存在 MM1 的 K 与 MM2 的 V 两条 descriptor 流、跨 MM1/MM2 边界重叠，使 MTE2 实际维持约两个 outstanding descriptor，而模型按逐个完全 drain 计价。本轮的微基准只有单条 descriptor 流（因此测得 count-independent），下一步应做双流变体来判定。注意：任何全局 MTE2 提速都会恶化低估主体，必须是随 outstanding 深度变化的模型，不能是常数。

## 8. 推荐的后续推进顺序

本节按 `1b6bf7206` 的实测重写，取代旧的 control-front-end 优先顺序。

1. **先做敏感性实验，再选修复目标。** 910B FA 的 stage 求和与关键路径严重不一致（fixpipe 求和高 8.6× 却完全不 load-bearing）。任何候选在改服务之前，先用整体缩放做一次同口径 334 行对照，确认它能移动 MAPE 和 Max APE。
2. **优先处理低估主体，而不是最大 APE 的那几个 case。** 216/334 低估、中位 −6.69%，而 very-long KV 只有 10 行。下一个机制应针对 short Q / short KV / 高 B：测 AIC scalar/控制前端服务随 active AIC group 数的变化，并把 `configs/hardware/ascend_910b.yaml` 中 `fma_pipe` 的 `[NEEDS_MICROBENCH] 占位` 换成实测值。判据是固定每 block 工作量下时延随 active block 数的增长（实测约 1.3 µs/block）。
3. **高估尾部需要先判定 drained 服务是否被测得过慢**，而不是直接改系数。已发布的 cold drained 服务比生产实测慢约 1.8×，而字节、drain 语义与 count 依赖都已分别验证正确。需要一次 2-slot ping-pong 并发探针来确定 MTE2 在 drain 边界下实际能领先多少。
4. **不要再用 evaluation MAPE 反推 910B 的 MTE2 速率。** 这条路径已在本轮被量化否决（见 §6、§7.3）。
5. C500 与 A800 当前分别为 7.92% 和 6.01%，均低于 910B，除非出现新 golden 或明确 source discrepancy，不优先继续调参。

上一轮未提交的微基准源码（`aic_control_frontend`、`aic_tile_chain`、`aiv_stage_chain`、`runtime/cross_domain`）位于 910B 执行副本 `910b-1:/gpfs/midlayer_yxh/microbench/ascend_910b/` 下，可以阅读并据此重建实验设计，但它们没有发布 reference、没有 Forward 消费者，也不得从远程执行副本原样反向合并。旧 §7 提到的 `fia_zero_work/` 与 `runtime/cross_domain_nonempty/` 在 910B 上并不存在，那份清单本身不准。

## 9. 关键提交与证据索引

| 主题 | 历史提交 | 长期证据/实现入口 |
|---|---|---|
| C500 ragged/direct/packed launch | `9aaa4030`, `3d6bcb38`, `fff71168` | `src/kernel/predictors/flash_attn.py`, `src/kernel/launch_config/metax_c500/flash_attn_launch.py` |
| C500 CTA barrier | `00e82fd7`, `d75717b9` | `configs/hardware/metax_c500.yaml`, `microbench/metax_c500/runtime/synchronization/` |
| C500 issue 与 resident tail | `98e6efec`, `b26b28cd` | `src/kernel/predictors/flash_attn.py`, `src/kernel/formula.py` |
| 910B SyncAll | `158e2ac8`, `f74a2635`, `5a44d2ee` | `fia_aiv_sync_all_cann_8_5_1.json` |
| 910B mixed outer envelope | `18ee8f49`, `67af78ca` | `fia_mixed_task_outer_envelope_cann_8_5_1.json` |
| 910B SplitFuse MTE2 | `6cad1d5c`, `4f652ed0` | `src/kernel/formula.py`, `src/kernel/predictors/flash_attn.py` |
| 910B source-ordered DAG | `7641ef0c`, `8ef12f92`, `bce59f74` | `src/kernel/launch_config/ascend_910b/flash_attn_launch.py`, `fia_tiling_data_materialization_cann_8_5_1.json`, `aic_mte1_load_pair_cann_8_5_1.json` |
| 910B AIV workspace | `9557e32d`–`5e115bbc` | `configs/hardware/ascend_910b.yaml` 的 `fia_aiv_workspace_transfer_service` |
| 910B FlashDecode terminal | `9e8c40dd`, `a2849d05` | `configs/hardware/ascend_910b.yaml` 的 `fia_flash_decode_terminal_prelude_service` |
| 910B reduction 1024 columns | `743f049a`, `f2865038` | `microbench/ascend_910b/operators/flash_attn/block_reduce_sum/` |
| 910B strided ND2NZ | `4a1cdee7`, `285e6dd9`, `3ee3fae1`, `5c2dc705`, `b43d5c46` | `configs/hardware/ascend_910b.yaml` 的 `aic_mte2_nd2nz_service` |
| 910B drained descriptor count law | 本轮 | `microbench/ascend_910b/references/aic_mte2_fia_gqa_drained_descriptor_count_law_cann_8_5_1.json`, `microbench/ascend_910b/operators/flash_attn/gqa_drain_count_law/` |

历史 hash 用于定位当时的实验和设计决策；rebase 后同一语义可能以不同 hash 出现在当前主线。判断现状时以当前源码、正式 reference JSON 和 HWSpec 为准，不以旧 commit message 替代代码检查。
