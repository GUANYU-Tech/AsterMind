# Kernel golden 计时口径与三硬件数据结论

本文记录 `kernel_dataset/` 中 NVIDIA A800、Ascend 910B 与 MetaX C500 的逻辑算子计时合同、重采调查、当前 canonical 数据边界、相对旧 golden 的变化，以及后续 Kernel 预测精度工作的使用规则。2026-09-13 的 A800 全量 counter/timing 发布以 main 父提交 `a121512b0b705b470d614fc12adf2a34d66850cc` 为 before baseline；同日 910B 全量 timing 发布使用 capture baseline `9c2926f93a7f29c7cd970b13d0c3148f522f7c6f`。本文中明确标记为 `7233a42f`、上一版或 legacy 的表格仍是历史对比，不应解读为当前 canonical。

## 结论摘要

面向真实 LLM Serving 仿真，Kernel 层应定价的是一次逻辑 public OP 在设备上的执行时间，而不是 Python/CPU 提交耗时，也不是 profiler 插桩时间。若一个 OP 含多个 device kernel，理想 target 是该 OP 的设备执行区间；在尚未具备可靠 OP 区间测量时，逐 kernel duration 之和只能作为显式标记的兼容 target，不能伪装成直接测得的 OP elapsed。

当前每种硬件均有 987 个相同 case：296 个 Vector、294 个 FlashAttention、221 个 GEMM、176 个 FusedMoE。三种硬件都有完整的 evaluation total，但测量成熟度不同：

| 硬件 | 当前可评分 case | 直接 OP timing | 兼容 kernel-sum timing | 推荐用途 |
|---|---:|---:|---:|---|
| A800 | 987 | 0 | 987 个 nsys kernel median sum | 987 个均为三独立进程窗口的中位窗口，最大窗口差均 ≤5%；这些值是 kernel duration sum，不是直接 OP interval |
| 910B | 987 | 987 个 msprof Timeline public-call device-task timing | 0 | 每个 case 选择三个最一致独立窗口并取中位 observation；987/987 两两相对差 ≤5%，最大 4.993% |
| C500 | 987 | 643 个 Graph Event | 344 个 mcTracer kernel sum | 643 个有 OP Event 值，但只有 494 个通过统一的 CV/max-deviation 5% 检查；全量 987 是 mixed-source 过渡集 |

因此，本次结论不是“所有旧值都错”或“所有新值都同等可靠”。推荐的 canonical 使用方式是：A800 使用当前 987 个同一口径 nsys target，并明确其全部为 kernel-sum 边界；910B 使用当前三窗口一致的 msprof device-task OP timing，GEMM 已有同边界 Serving 对账，FA/MoE/Vector 仍缺逐族同边界 Serving 直接验证；C500 至少拆成 494 个 strict-stable Event、149 个 Vector Event stability warning 和 344 个 mcTracer fallback 三组，不能把全量 987 MAPE 称为全量最终稳定 OP golden 精度。

## 为什么目标是 OP 设备时间

Serving 调度层产生的是 GEMM、FlashAttention、FusedMoE、Norm、RoPE、Activation、Elementwise 和 KV-cache 等逻辑调用。一次逻辑调用可能只启动一个 kernel，也可能启动多个 kernel；Serving 仿真需要为这个调用定价，再由调度和资源时间线处理调用间的 CPU 提交、同步、排队、重叠和框架开销。

设备 Event 本身不会“计入 CPU 执行时间”。问题出在计时边界：若对短 eager 调用逐次 record Event、同步并提交，CPU 的提交节奏可能让设备队列出现空洞，最终测到的是受 host pacing 影响的设备时间线，而不是 Serving Graph 内稳定的增量 OP 成本。Graph replay 将 launch 序列预先固化，外部 Event 只覆盖设备 replay；差分 Graph 进一步用两张共享前缀的 Graph 相减，消除 replay 固定成本。

本文采用以下定价边界：

```text
logical OP device price
  = device execution interval attributable to one public OP
  ≠ Python wall time
  ≠ eager submit interval
  ≠ profiler-instrumented duration
  ≠ Serving scheduler/runtime fixed overhead
```

Graph 与 eager 在数学算子语义上相同，但在短算子的实际时间线上可能不同，因为 eager host 提交间隙、同步位置、缓存复用、功耗状态和 Graph replay 固定开销都会改变观测值。Serving runtime 固定成本应在 Serving 层建模，不能同时摊进每个 OP golden，否则会重复计费。A800 当前 canonical 为了四族统一边界，已从上一版 Vector 差分 Graph Event 切换为全量 nsys kernel sum。

## Canonical timing contract

评分只读取每个 case 的 `op_info.evaluation.timing`：

- `granularity="op"` 表示直接 public OP timing，当前来源为 `device_event_op_median` 或 `msprof_timeline_op_median`。
- `granularity="kernel"` 表示逐 kernel observation 汇总的兼容 target，当前来源为 `nsys_nvtx_kernel_median` 或 `mcTracer`；`total_duration_ns` 等于 `kernels[].duration_ns` 之和。
- `total_duration_ns` 是 evaluation target；旧 profiler payload 中的 kernel duration、counter 或插桩字段不能越过该边界成为另一套评分真值。
- `provenance` 固定 baseline、runner、source snapshot、timer sanity 和 artifact SHA；存在该字段不等于 raw artifact 已随 Git 发布，证据可移植性边界见后文。

提交 `7233a42f` 相对父提交 `3a05fd2e` 的 12 个 canonical JSON 逐 case 去掉 `op_info.evaluation` 后完全相同：case ID、Attr、kernel name、grid/block、counter 和其他 profiler payload均未因 OP timing 发布而改变。该提交实际替换的 timing 是 A800 Vector 296 个、910B 全部 987 个、C500 FA 57 个、GEMM 208 个、MoE 82 个；C500 Vector 296 个已在更早的 `add047a3` 发布，因此在本提交中没有再次变化。910B FlashAttention launch facts 只更新了与新 source 对应的 provenance digest，launch facts 本身未改。

2026-09-13 的 A800 发布同时替换 987 个 case 的 NCU physical-counter payload 和 987 个 `op_info.evaluation.timing`。Counter candidate 已排除所有 duration 命名字段；timing candidate 与新 counter 的 Type、Attr、kernel name、grid、block 和 ordinal 在 987/987 case 上精确匹配。Forward、HWSpec、910B 和 C500 未在该发布中修改。

Golden duration 只能进入 `evaluation/`。它们不得进入 production Forward、HWSpec 或 production launch selector，不得用于训练、回归、插值、分段点选择、per-shape correction 或 residual correction，也不得因为某个参数降低 MAPE 就被视为合法。Forward 只能使用硬件规格、编译器/算子源码静态语义、与 evaluation case 独立的机制微基准，以及不含目标 duration 的 launch/counter observation。

## 测量方案

### A800 全量：三窗口 NSYS device kernel timing

A800 当前 987 个 canonical case 统一使用 `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --export=sqlite` 采集的 CUDA device kernel `start→end`。每个 case 有三个独立进程窗口，按整个 public call 的 kernel duration 总和选择中位窗口，不取最快值；三窗口最大相对差 `(max-min)/max` 必须 ≤5%。最终 GEMM 221/221、FlashAttention 294/294、FusedMoE 176/176、Vector 296/296 全部通过，四族最大窗口差分别为 4.737%、4.967%、3.175% 和 4.491%。多 kernel OP 的 `total_duration_ns` 是已验证 launch group 的逐 ordinal duration 之和，不包含 kernel 之间的 device gap。

### A800 Vector 上一版：差分 CUDA Graph Event

A800 Vector 的 296 个 case 使用 `torch.cuda.Event`，不挂 profiler。每个 case 构建共享相同前缀的 lower-call 与 higher-call 两张 CUDA Graph；每对样本先计算：

```text
op_time_i = (T_high_i - T_low_i) / (N_high - N_low)
```

Golden 是 paired `op_time_i` 的中位数。实际采集交替执行 lower→higher 与 higher→lower，以抵消慢速漂移。Graph 内每个 public call 使用独立 storage；lower 和新增 suffix 各自至少覆盖 320 MiB，实际独立输入为 320–416 MiB，是 A800 40 MiB L2 的至少 8 倍。调用数由 `ceil(320 MiB / per-call input bytes)` 决定，不按目标耗时挑选。每个 case 有 5 次 warmup 和 50 个 paired samples；采集开始前执行 1000 次 4096³ BF16 GEMM，并在每次 replay 的 start Event 前同流执行一次 GEMM 维持功耗状态，该 GEMM不在 Event 区间内。

CUDA Graph capture、Python submit、终止 Event 的 host synchronize 和 host interval均不计入 golden。该采集快照中的每个 Vector public OP 对应一个 canonical kernel，但定价仍来自 OP 级差分 Event，而不是旧 nsys kernel duration。

最终 296/296 accepted，0 unstable，0 unsupported；285 个来自 full shards，11 个来自 retry shards。门槛为 CV ≤5% 且任一样本相对中位数最大偏差 ≤5%；实际最大 CV 为 1.693%，实际最大偏差为 4.865%，中位 CV 为 0.156%，P90 CV 为 0.766%。Timer sanity 的 4096³ BF16 GEMM 中位数为 635904.014 ns，CV 为 0.090%。

### A800 非 Vector 上一版：保留 nsys kernel median sum

A800 的 691 个非 Vector case 仍使用旧 `nsys_nvtx_kernel_median`。采集在 warmup 后把一次 public call 放入 NVTX range，严格关联 CUPTI runtime 到 kernel，并按 kernel identity、grid 和 block 对齐 NCU；每个 ordinal 跨 samples 取 kernel median，OP evaluation total 是这些 ordinal median 的和。

这 691 个 case 可用于当前 OP 评分，但不是直接测得的“首 kernel start 到末 kernel end”区间，因此不包含多 kernel 之间的 device gap。当前 FA 有 124 个单 kernel、170 个多 kernel case；GEMM 有 195 个单 kernel、26 个多 kernel case；MoE 176 个全是多 kernel case。它们也没有像新 Vector 一样在 canonical timing 中嵌入完整 per-case provenance。

本轮另做了未发布的 A800 非 Vector 复测，用来判断是否必须替换 canonical：GEMM 221 个全部 strict stable，候选相对当前 canonical 的绝对变化中位/P90/最大为 0.342%/0.967%/2.492%；FA 为 278 strict stable + 16 cross-run-median usable，0 unusable，绝对变化中位/P90/最大为 0.340%/1.385%/9.869%；MoE 为 165 strict stable + 11 cross-run-median usable，0 unusable，绝对变化中位/P90/最大为 4.464%/17.721%/24.714%。MoE 的 prefill bucket 有符号中位变化为 −14.548%，tiny bucket 为 +6.642%，存在明显 regime-dependent shift；因此这些候选保留为调查证据，没有写入 `7233a42f`。

### 910B 全量：三窗口 msprof device-task timing

910B 当前 987 个 case 全部使用 `msprof_timeline_op_median`。GEMM、FlashAttention 和 Vector 的一次 public call 各对应一个 Timeline device task；MoE 的一次 public call 对应同 stream 上的一组 device tasks，其 OP interval 从首 task start 到末 task end，包含 task 间 device gap，不包含 Python/CPU wall time。GEMM/FA/MoE 在 runner warmup 后测 ordinary-warm 状态；Vector 每次 public call 前执行 2 GiB BF16 Fill，并在每个窗口内执行 31 次 public call。

采集先完成三轮全量独立窗口，每轮 987/987。初始有 183 个 case 不满足“选中三个窗口任意两者相对差 ≤5%”；之后仅定向补测失败项，依次收缩到 84、7、0 个。最终每个 case 从所有独立窗口中选择最一致的三个，再取三者中位数对应的真实 observation，不取最快值。最终 GEMM、FA、MoE、Vector 共 987/987 通过，最大两两相对差为 4.992968%。

GEMM 已有同边界 Serving 对账：Serving device task 为 1600.850 µs，standalone ordinary-warm/no-sweep 为 1599.820 µs，差 −0.06%；2 GiB sweep 为 1685.030 µs，差 +5.26%。这直接支持 GEMM 的 ordinary-warm/no-sweep 边界。FA、MoE 和 Vector 当前只证明了三窗口重复性，尚无逐族同边界 Serving device-task trace 的直接外部验证；该限制必须与稳定性结论分开报告。

### C500：Graph Event 与 mcTracer/mcProfiler 的职责分离

C500 的直接 OP timing 使用外部 device Event 包围 CUDA Graph replay。Event 负责 OP 时间；Graph mcTracer 负责正常执行下的 kernel name、grid、block 和 kernel wall-time；mcProfiler 只提供 counters。mcProfiler 的 `during` 是插桩执行时间，已观察到比正常 launch 放大几十至上千倍，绝不能作为 golden。完整工具约束见 [MetaX C500 profiling](metax_c500/profiling.md)。

C500 的 FA/GEMM/MoE Event 对同一个 OP Graph 使用两个 repetition 档，每个样本先用 Event replay 总时间除以 repetition 得到单 OP 时间，再比较两档中位数：绝对差必须 ≤ `max(500 ns, 5% × 较大中位数)`，且 selected 样本 CV ≤5%、任一样本相对中位数最大偏差 ≤5%。FA/GEMM 使用 50/200 次 replay，MoE 使用 5/20 次 replay。每次 replay 前有 2 GiB device sweep，但同一次 Graph replay 内会复用该 OP 的输入，因此该值代表 Graph steady-state/intra-replay reuse 条件，不应描述为每个 OP 都读取全新冷数据。按这套完整发布门槛通过的是 FA 57、GEMM 208、MoE 82，共 347 个。

C500 Vector 296 个来自更早的 paired-differential Graph 合同：50 个 paired samples、轮换 buffer，lower 和新增 suffix 的 distinct input 各自至少为 C500 8 MiB L2 的 8 倍。Canonical value 与 296 个 raw median 全部一致，但该批次没有经过后来 FA/GEMM/MoE 的 completion manifest 发布门禁；按相同的 CV ≤5% 和最大样本偏差 ≤5% 重新审计，只有 147 个通过，149 个告警，其中 66 个 CV >5%，149 个最大偏差 >5%。最差 case `VECTOR_ELEMENTWISE_T1024_H8192_B1_MUL` 的 CV 为 23.97%、最大偏差为 140.49%。所以 C500 有 643 个直接 OP Event 值，但当前只能称 494 个通过统一的严格样本稳定性检查，不能再写成“643 个全部严格稳定”。

剩余 344 个仍保留 mcTracer kernel timing：FA 237、GEMM 13、MoE 94。FA 的 294 个 case 已有完整 Graph mcTracer launch evidence，Attr 全部一致，历史 14 个 launch mismatch 已修正，因此 237 个只需要补 Event，不需要重跑 mcTracer；GEMM 13 个也只需补 Event；MoE 的 94 个中有 10 个 Event 不稳定，另 84 个缺完整 Graph mcTracer + Event。completion manifest 已将待办拆成 `flash_attn=237`、`gemm=13`、`fused_moe_unstable=10`、`fused_moe_missing=84`。

C500 FA 的历史完整 Event raw 严格复算中只有 57/294 通过：95 个只是不满足两档中位数收敛，81 个同时不收敛且最大样本偏差 >5%，61 个同时不收敛、CV >5% 且最大偏差 >5%。因此不能把“已有 Event 文件”解释为“已有稳定 OP golden”，也不能把 294 个 Graph mcTracer duration 全部标成 Event OP timing。

## 当前数据矩阵

| 硬件 | Family | Case | 当前 source | 直接 OP | 稳定性/保留状态 |
|---|---|---:|---|---|---|
| A800 | Vector | 296 | `nsys_nvtx_kernel_median` | 否，kernel sum | 296/296 三窗口 ≤5%，最大 4.491% |
| A800 | FlashAttention | 294 | `nsys_nvtx_kernel_median` | 否，kernel sum | 294/294 三窗口 ≤5%，最大 4.967% |
| A800 | GEMM | 221 | `nsys_nvtx_kernel_median` | 否，kernel sum | 221/221 三窗口 ≤5%，最大 4.737% |
| A800 | FusedMoE | 176 | `nsys_nvtx_kernel_median` | 否，kernel sum | 176/176 三窗口 ≤5%，最大 3.175% |
| 910B | Vector | 296 | `msprof_timeline_op_median` | 是 | 296/296 三窗口 ≤5%；每 call 前 2 GiB Fill |
| 910B | FlashAttention | 294 | `msprof_timeline_op_median` | 是 | 294/294 三窗口 ≤5% |
| 910B | GEMM | 221 | `msprof_timeline_op_median` | 是 | 221/221 三窗口 ≤5%；已有 Serving 对账 |
| 910B | FusedMoE | 176 | `msprof_timeline_op_median` | 是 | 176/176 三窗口 ≤5% |
| C500 | Vector | 296 | `device_event_op_median` | 是 | 147 strict；149 个 CV/max-deviation 告警 |
| C500 | FlashAttention | 294 | 57 Event + 237 `mcTracer` | 部分 | 57 strict；237 待 Event |
| C500 | GEMM | 221 | 208 Event + 13 `mcTracer` | 部分 | 208 strict；13 待 Event |
| C500 | FusedMoE | 176 | 82 Event + 94 `mcTracer` | 部分 | 82 strict；10 Event unstable，84 缺 Graph trace + Event |

## 相对旧 golden 的变化

2026-09-13 A800 发布相对其直接 before canonical（main `28525a0b`）的 timing 变化如下；逐 case 数值见 `experiment_logs/kernel_golden/nvidia_a800_device_golden_20260912/a800_timing_changes.csv`。

| Family | Case | 有符号中位 | 绝对变化中位 | 绝对变化 P95 | 最大绝对变化 |
|---|---:|---:|---:|---:|---:|
| GEMM | 221 | +0.128% | 0.459% | 2.589% | 4.867% |
| FlashAttention | 294 | +0.237% | 0.503% | 2.444% | 16.973% |
| FusedMoE | 176 | +0.016% | 0.150% | 1.037% | 15.366% |
| Vector | 296 | +0.550% | 18.304% | 35.497% | 42.407% |

2026-09-13 910B 发布相对其直接 before canonical 的 timing 变化如下；逐 case 数值见 `experiment_logs/kernel_golden/ascend_910b_device_golden_20260912/ascend_910b_timing_changes.csv`，完整分桶见同目录的 `ascend_910b_timing_bucket_changes.md`。

| Family | Case | 变快 | 变慢 | 有符号中位 | 绝对变化中位 | 绝对变化 P95 | 最大绝对变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GEMM | 221 | 214 | 7 | −51.867% | 51.867% | 80.914% | 81.736% |
| FlashAttention | 294 | 278 | 16 | −21.410% | 21.410% | 66.102% | 72.033% |
| FusedMoE | 176 | 149 | 27 | −4.826% | 4.989% | 23.663% | 44.969% |
| Vector | 296 | 6 | 290 | +15.334% | 15.334% | 78.853% | 162.529% |

以下累计对比以 `28577f792c51f6951c7dfb0f49703ee7fd4519cd` 为 legacy baseline，只记录 `7233a42f` 阶段的历史结果，不代表 2026-09-13 A800/910B 全量重采后的当前值。变化定义为 `(7233a42f - legacy) / legacy`；正值表示该历史 golden 更慢，负值表示更快。

### Family 总览

| 硬件 | Family | Case | 变化 case | 有符号中位 | 绝对变化中位 | 绝对变化 P90 | 最大绝对变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| A800 | Vector | 296 | 296 | −1.06% | 17.64% | 34.41% | 74.53% |
| A800 | FlashAttention | 294 | 0 | 0% | 0% | 0% | 0% |
| A800 | GEMM | 221 | 0 | 0% | 0% | 0% | 0% |
| A800 | FusedMoE | 176 | 0 | 0% | 0% | 0% | 0% |
| 910B | Vector | 296 | 296 | −6.33% | 15.70% | 65.00% | 161.38% |
| 910B | FlashAttention | 294 | 294 | +5.62% | 6.17% | 68.10% | 93.00% |
| 910B | GEMM | 221 | 221 | +61.57% | 61.57% | 133.57% | 146.77% |
| 910B | FusedMoE | 176 | 176 | +3.50% | 3.56% | 10.87% | 21.12% |
| C500 | Vector | 296 | 296 | −57.62% | 57.62% | 73.16% | 80.51% |
| C500 | FlashAttention | 294 | 294 | −0.63% | 1.66% | 16.07% | 491.52% |
| C500 | GEMM | 221 | 208 | −0.05% | 0.99% | 10.47% | 89.54% |
| C500 | FusedMoE | 176 | 82 | 0.00% | 0.00% | 9.77% | 76.72% |

C500 的累计变化表混合了多个阶段：Vector 在 `add047a3` 切换到 Event；FA/GEMM 在 `3a05fd2e` 发布完整 Graph mcTracer launch/timing evidence；`7233a42f` 又只把满足严格门槛的 57 FA、208 GEMM 和 82 MoE 替换为 Event。不能仅凭“相对 legacy 已变化”推断该 case 已是 Event source，必须读取当前 `timing.source`。

只看 `7233a42f` 相对直接父提交 `3a05fd2e` 的 C500 Event replacement，FA 57 个的绝对变化中位/P90/最大为 1.23%/2.54%/2.85%，且 Event 均比同批 Graph mcTracer kernel sum 更长，绝对差中位/P90/最大为 25.51/29.23/31.18 µs；GEMM 208 个为 1.18%/10.29%/40.30%，最大项之一 `GEMM_B32_M1_K5120_N1536` 从 38.144 µs 降到 22.772 µs；MoE 82 个为 4.78%/27.33%/76.72%，最大项 `FUSEDMOE_TOK8_E512_TOPK10_H2048_INT512` 从 1977.6 µs 降到 460.301 µs。它们说明“OP interval 与 kernel sum 的差”不是一个可全局相加的常数，尤其 GEMM/MoE 不能用固定 correction 替代真实测量。

### A800 Vector 分桶

| Bucket | Case | 有符号中位 | 绝对变化中位 | 绝对变化 P90 | 最大绝对变化 |
|---|---:|---:|---:|---:|---:|
| Activation | 56 | +23.25% | 23.25% | 35.10% | 44.37% |
| Elementwise | 96 | −16.23% | 18.75% | 55.04% | 74.53% |
| KV-cache | 16 | −6.32% | 6.79% | 12.55% | 17.65% |
| Norm | 96 | −5.43% | 15.16% | 24.57% | 52.05% |
| RoPE | 32 | +15.02% | 15.02% | 31.16% | 56.19% |

A800 最大相对变化是 `VECTOR_ELEMENTWISE_T1024_H2560_B1_MUL`，6784.000 → 11840.001 ns，+74.53%；最大绝对变化是 `VECTOR_SILU_T1024_I53248_B1`，339645.000 → 299519.986 ns，−40125.014 ns。相同 Forward prediction 下，Vector MAPE 从旧 target 的 16.50% 降到新 target 的 11.23%，但 WAPE 从 14.28% 小幅升到 14.66%；不能只凭 MAPE 宣称所有尺度都改善。Activation MAPE 反而从 11.71% 升到 22.52%，说明旧 golden 上形成的 subtype 结论也需要重建。

### 910B 分桶（`7233a42f` 历史）

| Family/Bucket | Case | 有符号中位 | 绝对变化中位 | 绝对变化 P90 | 最大绝对变化 |
|---|---:|---:|---:|---:|---:|
| Vector / Norm | 96 | −9.21% | 11.68% | 47.55% | 67.55% |
| Vector / RoPE | 32 | +80.25% | 80.25% | 120.23% | 161.38% |
| Vector / Activation | 56 | +0.79% | 9.53% | 61.58% | 95.17% |
| Vector / Elementwise | 96 | −11.69% | 16.42% | 52.49% | 77.78% |
| Vector / KV-cache | 16 | −9.32% | 11.83% | 41.98% | 45.14% |
| FA / Q=1 | 200 | +23.21% | 23.21% | 75.89% | 93.00% |
| FA / short Q 2–512 | 15 | +1.44% | 1.77% | 9.15% | 13.59% |
| FA / medium Q 513–2048 | 68 | +1.29% | 1.50% | 5.10% | 8.09% |
| FA / long Q ≥2049 | 11 | +0.93% | 0.93% | 1.46% | 2.33% |
| GEMM / tiny B×M≤16 | 65 | +117.00% | 117.00% | 140.70% | 146.77% |
| GEMM / small 17–64 | 39 | +112.10% | 112.10% | 133.95% | 140.83% |
| GEMM / medium 65–2047 | 39 | +53.49% | 53.49% | 110.85% | 120.91% |
| GEMM / prefill ≥2048 | 78 | +4.27% | 4.44% | 18.03% | 34.18% |
| MoE / tiny TOK≤16 | 55 | +1.79% | 1.79% | 5.76% | 20.30% |
| MoE / small TOK 17–64 | 22 | +3.28% | 3.28% | 4.87% | 9.67% |
| MoE / medium TOK 65–2047 | 44 | +5.36% | 5.36% | 11.89% | 20.11% |
| MoE / prefill TOK≥2048 | 55 | +4.67% | 4.67% | 13.77% | 21.12% |

上述 910B 表是 `7233a42f` 的历史变化，不是当前全量重采结果。当前 910B 的主分桶变化是：GEMM tiny/small/medium/prefill 有符号中位分别为 −73.260%/−74.602%/−53.562%/−3.812%，FA Q=1 为 −40.048%，Vector Norm/RoPE/Activation/Elementwise/KV-cache 分别为 +14.376%/+9.797%/+13.232%/+18.539%/+16.673%。

### C500 分桶

| Family/Bucket | Case | 变化 case | 有符号中位 | 绝对变化中位 | 绝对变化 P90 | 最大绝对变化 |
|---|---:|---:|---:|---:|---:|---:|
| Vector / Norm | 96 | 96 | −25.86% | 26.07% | 58.31% | 75.29% |
| Vector / RoPE | 32 | 32 | −20.40% | 20.40% | 39.29% | 58.50% |
| Vector / Activation | 56 | 56 | −64.75% | 64.75% | 68.32% | 69.73% |
| Vector / Elementwise | 96 | 96 | −69.42% | 69.42% | 73.80% | 76.00% |
| Vector / KV-cache | 16 | 16 | −72.58% | 72.58% | 79.73% | 80.51% |
| FA / Q=1 | 200 | 200 | −1.94% | 2.29% | 12.53% | 58.47% |
| FA / short Q 2–512 | 15 | 15 | −0.33% | 0.75% | 60.78% | 491.52% |
| FA / medium Q 513–2048 | 68 | 68 | +0.54% | 0.90% | 106.98% | 297.16% |
| FA / long Q ≥2049 | 11 | 11 | +0.63% | 0.63% | 2.42% | 90.48% |
| GEMM / tiny B×M≤16 | 65 | 55 | 0.00% | 1.46% | 17.51% | 32.49% |
| GEMM / small 17–64 | 39 | 37 | −0.10% | 2.55% | 8.66% | 40.30% |
| GEMM / medium 65–2047 | 39 | 38 | −1.79% | 1.87% | 12.37% | 25.11% |
| GEMM / prefill ≥2048 | 78 | 78 | −0.00% | 0.15% | 1.42% | 89.54% |
| MoE / tiny TOK≤16 | 55 | 54 | +3.86% | 6.55% | 45.76% | 76.72% |
| MoE / small TOK 17–64 | 22 | 21 | +2.02% | 2.02% | 7.02% | 8.89% |
| MoE / medium TOK 65–2047 | 44 | 7 | 0.00% | 0.00% | 1.04% | 10.12% |
| MoE / prefill TOK≥2048 | 55 | 0 | 0.00% | 0.00% | 0.00% | 0.00% |

C500 分布具有“中位变化很小、少量 case 变化极大”的特征，尤其 FA 和 GEMM，不能用 family median 掩盖离群点。`FLASHATTN_CHUNKED_SKEW_PREFILL_Q2048_B4_S4096_HQ32_HKV8_D128_C1` 是典型 provenance 冲突案例：legacy 为 103.680 µs，当前为 613.286 µs，约 +491.5%；当前 source 仍是 `mcTracer`，并未通过 Event 稳定门槛。该 case 应保留为待补测/归因项，不能据它反调 FA Forward 公式。

## 本轮调查中否决的方案

- eager 单调用 Event：短 OP 会受 host pacing、同步和设备队列空洞影响，不代表 Graph-covered Serving 中的增量设备成本。
- 只测一次 Event：无法识别频率状态、缓存条件和偶发调度波动，也无法判断中位数是否稳定。
- 轮换 buffer 只做到“接近 L2”：仍可能发生严重缓存复用；最终 A800 使用每个差分区段至少 320 MiB，910B 每次 public call 前使用 2 GiB sweep。
- 直接采用 Graph replay 总时间除调用数：会把 Graph replay 固定成本摊入 OP；短 Vector 使用 lower/higher 差分。
- 使用 profiler `during`：C500 mcProfiler 的插桩会显著放大执行时间，`during` 只可作诊断 provenance。
- 把多 kernel duration 简单求和称为直接 OP elapsed：kernel sum 不含 kernel 间隙；只有在 source 明确为兼容 target 时才可使用。
- 看到 Event 文件就发布：C500 FA 294 个都有历史 Event raw，但严格门槛只接受 57 个。
- 用差异最大的 case 选择计时方案：极端 case 容易受到 provenance 或 workload mismatch 影响，方法必须先由 Serving 边界和稳定性合同决定，再看分布。
- 用新/旧 golden 哪个让 MAPE 更低来决定真值：这会把 evaluator 变成拟合器，违反零拟合原则。
- 用 target duration 反推出 Forward 的 launch floor、分段点或 per-shape correction：旧 golden 的误差不构成物理证据，新 golden 也不构成。

## 其他调查发现

C500 旧 FA raw 的 `final_full_audit.json` 末尾带字面量 `\\n`，原文件不是合法 JSON；它只能在明确修复尾部后做人工追溯，不能作为 canonical 自动化输入。该旧 audit 在 mcTracer-Graph timing 合同下曾把 47 个 case 标为 duration-only candidate、14 个标为 launch+duration candidate；这 47 个不是最终 Event stable 子集，不能与后来严格 Event audit 的 57 个混为一谈。

C500 有 12 个 Q64 FA counter case 出现 `Achieved waves < Dispatched waves`，raw MMA/MTE 完整语义工作量低约 1.9%–2.18%；按 wave coverage 归一化后最大差约 0.275%。这是 counter coverage 问题，不是 launch 或 timing 错误，不能用它反向调 Forward。另一个 GEMM L2 case 的 analytical bytes 为 3,875,536,896，raw observed 为 3,687,243,264，raw gap 为 5.107%，wave coverage 归一化后约 3.112%；当前测试同时冻结两者，仍不允许用目标 counter 拟合模型。

`3a05fd2e` 阶段的 C500 Graph mcTracer 更新也不只是“47 个特殊 case”：FA 294 个全部获得完整 Graph trace，其中 280 个只更新 duration/Elapsed Cycles/evaluation timing，14 个历史 launch mismatch 同时更新 fresh launch/counter；GEMM 有两个 launch 发生真实修正，`GEMM_B1_M2048_K13824_N5120` 的 grid 从 `(16,64,1)` 变为 `(32,20,1)`，`GEMM_B1_M4096_K4096_N14336` 从 `(128,20,1)` 变为 `(32,28,1)`。与之不同，`7233a42f` 的 57 FA、208 GEMM、82 MoE Event 发布只改 `op_info.evaluation.timing`。

910B 当前候选 `910b_r18_three_window_candidate.json` 覆盖 987/987 个 case，`consensus_selection.json` 记录所有基础轮次、定向复测轮次和逐 case 三窗口选择；两者 SHA 分别为 `d9051de2…` 与 `b95e7c47…`。A800 Vector selected raw 296/296 也均存在并通过 SHA、runner、timer sanity 和 Attr 校验。C500 Vector raw 当前位于 canonical checkout 外部的 `experiment_logs/kernel_golden/metax_c500/20260827_vector_cold_diff_8x_full_d494ee96/`，296 个 canonical value 可由 raw 复算，但 canonical 中统一记录的 `source_artifact_sha256=23d8be05…` 没有在本 worktree 内对应一个已提交 manifest，因此当前 provenance 不能仅靠此 worktree 自闭合。`experiment_logs/` 与 `profiler_output/` 均被 Git 忽略，所以本文固化的摘要和 SHA 是必要索引，但不能替代长期 raw artifact 归档。

## 对既有精度结论的影响

此前许多 Kernel 精度优化面对的是 legacy kernel-duration golden。当前 timing target 变化远大于普通测量噪声：A800 当前 Vector 相对直接 before 的绝对变化中位 18.304%，910B tiny/small GEMM 有符号中位分别为 −73.260%/−74.602%，910B Q=1 FA 为 −40.048%，C500 Vector 的历史累计变化中位为 57.62%。因此旧报告中的 MAPE、误差排序、worst case、阈值选择和“某机制已经解决”的判断均不能直接横向比较。

在 A800 与 910B 全量发布候选写入后，用同一版 Forward 重新运行完整评估，结果为：

| 硬件 | 全部 MAPE | GEMM | FusedMoE | FlashAttention | Vector | timing source 备注 |
|---|---:|---:|---:|---:|---:|---|
| A800 | 11.38% | 12.82% | 7.36% | 5.46% | 18.56% | 987 nsys kernel sum |
| 910B | 57.71% | 147.84% | 12.83% | 58.28% | 16.52% | 987 three-window msprof OP timing |
| C500 | 6.85% | 7.28% | 8.62% | 7.11% | 5.23% | 643 OP Event + 344 mcTracer kernel sum |

C500 的 6.85% 不能与 A800/910B 的全量指标作同成熟度比较，因为它仍混合两种 timing granularity。910B 发布后 MAPE 大幅上升主要由 tiny/small GEMM 和 decode FA 的真值显著缩短造成；这说明 Forward 尚未解释新测量边界，不说明新真值更差。不得根据 MAPE 选择或回退 golden。

## 调查过程与决策演进

1. `28577f79` 先把 evaluation timing 从 profiler payload 中显式分离，成为本轮累计对比的 legacy baseline。这一步解决了“评分代码究竟读哪个 duration”的合同问题，但当时很多 target 仍是 kernel duration 或 kernel sum。
2. 短 Vector 的单调用 eager Event 试验显示，测量会强烈受 host 提交、同步位置、功耗状态和缓存复用影响。后续方案改为 Graph replay，并在 A800/910B 上进一步采用 lower/higher Graph 差分；输入工作集也从“接近 L2”收紧为明显大于 L2。
3. C500 同时比较 Event、mcTracer 和 mcProfiler 后确认三者不可互换：Event 用于 Graph OP 时间，mcTracer 用于正常 kernel wall-time 与 launch，mcProfiler只用于 counter。`add047a3` 先发布完整 Vector Event；`a2f8c1a6` 与 `5d5ac948` 修正 packed Q64 的 launch/timing provenance；`3a05fd2e` 发布 C500 Graph mcTracer evidence；最终只把满足严格 Event 门槛的子集切换为 OP timing。
4. 910B 曾尝试 NPUGraph 差分 Event 与早期 msprof Timeline 合同；最终发布改为四族统一的 msprof Timeline device-task OP timing，并通过三轮全量加失败项定向补测，为每个 case 选择三个一致独立窗口，不取最快值。
5. `7233a42f` 阶段只发布了 A800 Vector differential Event，保留 691 个既有 nsys target；这是历史决策，不再是当前 A800 canonical 边界。
6. 2026-09-13 A800 发布将四族 987 个 timing 统一为三窗口 nsys kernel sum，并同时替换 987 个排除 duration 的 NCU counter payload；发布前后的 Type、Attr 和 physical launch identity 全量匹配。
7. 2026-09-13 910B 发布只替换 987 个 golden timing，counter、launch、Attr、Forward 和 HWSpec 保持不变；全部 case 的三个选中窗口两两差 ≤5%，最大 4.993%。

## 后续 Kernel 精度优化工作流

1. 从当前 canonical commit 与 Git LFS OID 重新建立 baseline；旧报告保留并标记为 `legacy-golden`，不能删除，也不能继续作为当前验收线。
2. 每个 before/after 对比必须固定同一代码 commit、同一 case set、同一 canonical LFS OID 和同一 timing-source mask。Golden 变化与 Forward 代码变化必须分开报告。
3. 报告 overall、family 和 primary bucket 的 MAPE、MedAPE、P90、Max、WMAE/WAPE，并按 absolute error contribution 排序；总 MAPE 不能掩盖权重大 case 或 regime-dependent shift。
4. 每份报告都输出 timing source count，并区分 direct OP、kernel-sum fallback、`complete`、`partial` 和 `aggregate_only` support；禁止默认把缺失 provenance 视为相同口径。
5. 最大误差 case 至少展示 measured、predicted、绝对误差贡献、launch、timing source 与 provenance。先排查 workload、public-call 边界、Graph/eager、cache condition、功耗状态、kernel sequence 和 launch mismatch，再讨论公式。
6. A800 当前不需要全量重采；后续精度工作必须使用本次新 canonical，并对 Vector 的大幅 timing 边界变化单独分析。
7. 910B 无需再做全量稳定性重采；后续优先为 FA、MoE、Vector 各补同边界 Serving device-task 对账，再依据独立物理证据分析 tiny/small/medium GEMM、decode FA、Norm/Elementwise/Activation 和 MoE tiny 的 Forward 误差。
8. C500 先定向复测 149 个 Vector stability warning，再补 GEMM 13、MoE unstable 10、FA 237 和 MoE missing 84；补测前不得把 remaining mcTracer sum 改标为 OP Event。FA strict Event 相对同批 Graph mcTracer kernel sum 固定多出约 25 µs（绝对差中位 25.51 µs、P90 29.23 µs），应通过独立 runtime 机制微基准解释，不能把该差值塞入 kernel 公式。
9. Runtime/launch 固定开销只在 Serving 层建模。若一个机制需要独立微基准，结果必须归一化为有单位、有物理意义、可复测的硬件参数，不能复刻完整 production OP 后把其 latency 填回 Forward。
10. 任何公式、特征、阈值或参数选择都不得以降低这些 evaluation duration 上的误差为依据；若无法用独立物理证据解释，应保留误差并报告 partial、unsupported 或 WONTFIX。

## 复核命令

运行当前三硬件 Forward baseline：

```bash
python scripts/evaluation/kernel/run_kernel_evaluation.py --hardware-id nvidia_a800 --mode forward --jobs 32
python scripts/evaluation/kernel/run_kernel_evaluation.py --hardware-id ascend_910b --mode forward --jobs 32
python scripts/evaluation/kernel/run_kernel_evaluation.py --hardware-id metax_c500 --mode forward --jobs 32
```

统计当前 timing source：

```bash
for hardware in nvidia_a800 ascend_910b metax_c500; do
  for family in vector flash_attn gemm fused_moe; do
    printf '%s %s ' "$hardware" "$family"
    zstd -dc "kernel_dataset/$hardware/$family.json.zst" | jq -r '[.[].op_info.evaluation.timing.source] | group_by(.) | map({source: .[0], count: length})'
  done
done
```

检查 910B 最终三窗口一致性门槛：

```bash
jq -r '[.cases[] | select(.maximum_relative_deviation_pct > 5)] | length' \
  profiler_output/kernel_dataset/ascend_910b/ascend_910b_device_golden_20260912/consensus_selection.json
```

检查 canonical 文件 SHA：

```bash
sha256sum kernel_dataset/{nvidia_a800,ascend_910b,metax_c500}/{vector,flash_attn,gemm,fused_moe}.json.zst
```

## Evidence 与版本身份

上一次三硬件 canonical timing 发布提交为 `7233a42f8b88b5dd8774362894d7217fb1743504`；2026-09-13 A800 全量发布的 before baseline 为 main `a121512b0b705b470d614fc12adf2a34d66850cc`，累计 legacy baseline 为 `28577f792c51f6951c7dfb0f49703ee7fd4519cd`。

| Evidence | 路径 | SHA256 |
|---|---|---|
| A800 2026-09-13 publication manifest | `experiment_logs/kernel_golden/nvidia_a800_device_golden_20260912/publication_manifest.json` | `65e3fac19269c13ce4418d5b6a2de4c620b827cca2e0309e2db8f6f60993944a` |
| A800 2026-09-13 per-case timing delta | `experiment_logs/kernel_golden/nvidia_a800_device_golden_20260912/a800_timing_changes.csv` | `e1bc37e30d76ce34b99bae8663f44b217dec7f4b2301ead0f9bef3786b69e4fe` |
| A800 Vector final report | `experiment_logs/kernel_golden/a800_vector_differential_power_04d6310f/op_candidate_report.final.json` | `2fd5493be0c9674dd12ea475a485af5dbdd7a957083421afe941fe27fca7d41c` |
| A800 Vector selection | `experiment_logs/kernel_golden/a800_vector_differential_power_04d6310f/selection_manifest.final.json` | `99b980fde0f62323cbf4356e4ba654f3bca88314fb199b4a86756c3c6f38e0ba` |
| A800 non-Vector candidate audit | `experiment_logs/kernel_golden/a800_op_timing_bucket_audit_652e3d87/a800_op_timing_primary_bucket_audit.json` | `cb6989b49d121c62d6a796ef3ff48b7a3afa6529ec4c9112f4d9e42e737739f1` |
| 910B 2026-09-13 canonical candidate | `profiler_output/kernel_dataset/ascend_910b/candidate/910b_r18_three_window_candidate.json` | `d9051de201d2254215f948a5bc1266ee3382072fd2e89296650d102fb099fc5e` |
| 910B 2026-09-13 consensus selection | `profiler_output/kernel_dataset/ascend_910b/ascend_910b_device_golden_20260912/consensus_selection.json` | `b95e7c471239b7ad8d74278c30e4e124d0536403c355f9197411bf20aac18061` |
| 910B 2026-09-13 per-case timing delta | `experiment_logs/kernel_golden/ascend_910b_device_golden_20260912/ascend_910b_timing_changes.csv` | `6fedc49147d171ee43e3b7c915e0faef3d9826a9c1e88ef6613714422450b994` |
| 910B previous publication audit（历史） | `experiment_logs/kernel_golden/ascend_910b_op_timing_full_7417f8e7/audit.json` | `2ecffe40739ff837f5be3271338fbd77b99f08c6ccc7dc7aca2d724eadb9871f` |
| C500 remaining audit | `experiment_logs/kernel_golden/metax_c500_full_graph_v2/c500_remaining_audit.json` | `6c50baf0e7d8c5c0665947136de1f15204c12aec8944cc64b701b6b762eb637c` |
| C500 stable FA selection | `experiment_logs/kernel_golden/metax_c500_flash_graph_partial/audit.json` | `568bd17490324301cd7972f001559dfdf6d74cfb5f7ea8c5f394fd82ba096940` |
| C500 completion manifest | `experiment_logs/kernel_golden/metax_c500_completion_v2/completion_manifest.json` | `c89df8137a2dec4917eab329f00e3c21073f35fa7c3916a0a9268d776699077e` |

主要 execution provenance 为：A800 Vector capture baseline `04d6310f15d1223718c986b3082c944452e7c1b2`、source snapshot SHA `573795566e66658b7027b3f1711ad8e64852904cd0dea460df20abc23ec59e34`、runner SHA `9ccead03afcf7992713cf53ac9c7a908e334a23a3dc0e1b7422ec9eec14539b9`；910B 全量 capture baseline `9c2926f93a7f29c7cd970b13d0c3148f522f7c6f`、source snapshot SHA `f52e081af29088c89f5773d04b81bde086af3527c2df0b9599bca568bc54cfec`、candidate tree SHA `d9051de201d2254215f948a5bc1266ee3382072fd2e89296650d102fb099fc5e`；C500 completion execution baseline `65cb316d871ca04fabce0cf1afa11437ca9da2a2`、source snapshot SHA `c91464ab199a0f689f7d66f9dee299114c19411adb4fc88a2c75101996f8d216`、timer sanity SHA `fadc3ec2fe8fc630bb3657cf6161c47aecce59e209b870fbc21c0d36b3602626`。

当前 canonical Git LFS object SHA：

| 硬件 | Vector | FlashAttention | GEMM | FusedMoE |
|---|---|---|---|---|
| A800 | `6e0b0dda1783217501dd84e166225a6c8f2e2d56d13153b8b6546334d3c12b27` | `a4dcd9c6e197fe252cc7709503137dd1017fe13a49e483a864e08a4407bbf12e` | `daa6a1de2a8c80715cc6e4c11f47955beec0804aa04c4caa50b8766cdac8a197` | `9ffaff5f2220758ae9abc0487cf86cf02305ded239a31207e1bdcb0cef7484a1` |
| 910B | `2c855fe7d3cac321705bd06f511466c47b3163b23e94f34dff143915e027a5d0` | `15835b2fc5f4fb601481cbb248f040a771d4a657e77d2eb7659a9a8bdfc85474` | `7617c601ea166cf8eb29461431b87845d6eede93c7142084972b1e01f1fb13e7` | `b4609fd3659feb8737984bbb3f4f823355a79e75791a385804b3f3abd021e64d` |
| C500 | `5ed91693686fa7f24f9c576a8459271e6eb00b1a1c5610b1910ec019d9047d3c` | `504499cbd833dfe5288412c23b1517fc7bac2d1fe40d6a07ad919541bc6464ad` | `b1e8fbe39fafc8dfe914f1411caeb0f96b70c40576a82e85f2b2ec1fa6da1ba0` | `65490cc7296487d776b5d50ca84b5dc73d9880642385430a2ccfbcdfbc614c58` |

上述 `experiment_logs/` evidence 和仓外 execution snapshot 不受 Git/LFS 跟踪；它们的路径与 SHA 是本地证据索引，而不是可移植发布包。未来重采必须保留 source snapshot、raw artifact、audit 和 canonical candidate 的闭合 checksum chain；若 raw 被归档或迁移，必须更新可解析位置和 manifest，不能只在文档中保留失效绝对路径。
