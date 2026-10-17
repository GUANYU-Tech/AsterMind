# Serving 误差定位流程

本文规定 Midlayer 从请求级 TTFT、TPOT、E2E 和吞吐误差定位到 Scheduler step、运行时区间及单 Kernel 机制的标准流程。目标是建立可复现的因果链，而不是根据 Serving 误差调整 Forward 参数。Serving 真值只用于 evaluation、排序和验证；任何生产预测公式仍必须满足零拟合约束。

## 1. 诊断链路

```text
请求级 paired 指标
        │
        ▼
固定真实 Scheduler 轨迹重放
        │
        ├── 误差显著缩小 ──► 到达、排队或动态 batching 反馈
        │
        ▼
Prefill / Mixed / Decode step 分解
        │
        ▼
代表 step 的真实工作量与预测时间线
        │
        ▼
同运行 profiler：Device task、runtime gap、Host 边界
        │
        ▼
Kernel family → shape / variant / launch 排序
        │
        ▼
独立单 Kernel Event / counter / 机制微基准
        │
        ▼
输入错误、lowering 缺失、launch 错误、公式错误或 runtime 缺失
```

每一级只能回答自己的问题。请求级误差不能直接证明某个 Kernel 错；预测侧 Kernel 占比不能替代实测侧 profiler；单 Kernel Event 准确也不能证明 Scheduler、Graph 或 runtime 时间线准确。

已结论不再重复诊断：C500 真实 vLLM 0.24 Scheduler 重放对比证明 Scheduler 选择对 Serving 结果无实质影响，三硬件统一使用 `src/serving/backend/vllm.py` 的单一解析 Scheduler（含 vLLM 0.11 与 0.24 的 async terminal 提交差异，该差异同样无影响）；0.24 Scheduler 桥工具链已归档至 `archive/serving_legacy/vllm_024_scheduler_bridge/`，仅供未来 vLLM 升级时取回复验。 MetaX C500 的 TTFT/TPOT 误差随 QPS 单调收敛这一现象已在第 13 节收口，含已证伪的候选与剩余阻塞，不要重新按 batch 去找机制。

## 2. 第一级：确认请求级误差

先运行或刷新标准 Serving report：

```bash
python scripts/evaluation/run_standard_evaluation_report.py --serving-only
```

请求级指标必须按同一个 `request_idx` 配对，不能用“预测均值对实测均值”代替 paired comparison。对任一可配对指标 $x$：

$$
e_i = \frac{x_i^{pred}-x_i^{obs}}{x_i^{obs}}, \qquad APE_i=|e_i|, \qquad MAPE=\frac{1}{N}\sum_i APE_i
$$

正 signed error 表示预测偏慢，负值表示预测偏快。TPOT 只统计实际输出 token 数大于 1 的请求，并按同一请求从首 token 到末 token 的间隔除以输出间隔数；TTFT 从实际请求发出到首个可见 token；throughput 使用同一次运行的完整观测窗口。报告至少保留 paired MAPE、signed error、P50/P90 APE 和绝对时间差。

选择诊断 case 时，同时考虑绝对时间贡献和相对误差。优先选择误差稳定、贡献大的 case，并保留一个低 QPS case以减少排队反馈干扰；不能只挑百分比最大的极短请求。

### 本级检查项

- 实测与预测使用相同请求文件、模型 revision、dtype、TP/PP/EP、Graph 模式、Scheduler 配置和硬件。
- 请求按 `request_idx` 一一对应，actual output token 数一致。
- 使用实际发送时间而不是计划到达时间计算 TTFT。
- 明确报告的是 MAPE、mean signed error 还是聚合总量误差，三者不能混称。
- 不用整场完成时间代替 TTFT 或 TPOT。

## 3. 第二级：固定真实 Scheduler 轨迹

Serving 是闭环系统：step 时间改变下一次 Scheduler 决策时间，进而改变 batch、Kernel shape、排队和后续 step。为了把“单步执行误差”和“调度反馈误差”分开，必须重放 canonical `scheduler_steps.jsonl.zst` 解压后的真实 step 顺序、batch membership 和 token progress，并只用当前 Forward 重新计算每个 step 的时间。

单 case 入口：

```bash
python scripts/evaluation/serving/scheduler/run_scheduling_trajectory_replay.py \
  --case ascend_910b/qwen3-8b/tp1_qps4 \
  --host-launch-profile configs/host_timing/host_launch_profiles/ascend_910b.json
```

批量 signed-error 分解入口：

```bash
python scripts/evaluation/serving/diagnostics/run_scheduler_depth_breakdown.py \
  --run-id serving-error-diagnosis \
  --case ascend_910b/qwen3-8b/tp1_qps4
```

轨迹重放输出中的 `forward_duration_observation_consumed` 必须为 `false`：实测 step duration只能在 Forward 完成后用于 comparison，不能进入预测。

### 如何判断

| 现象 | 优先结论 |
|---|---|
| 固定轨迹后请求级误差大幅下降 | 动态 batching、排队或到达反馈是重要来源 |
| 固定轨迹后误差仍接近原报告 | step service 预测是主要来源 |
| step 数或 token 进度不守恒 | 先修输入、Scheduler 或 trace contract，不能继续归因 Kernel |
| 只有高 QPS 明显变差 | 优先检查容量边界、排队放大和轨迹反馈 |
| 低 QPS 也出现同方向大误差 | 优先检查每步 execution、runtime 和 Kernel |

这里的 observed serialized service 定义为：

```text
update_start - max(schedule_end, previous_update_start)
```

它表示时间线上串行暴露给 Scheduler 的 service，不是纯 Device 时间。异步 RPC、pipeline depth 和 Host/Device 重叠可能改变归属，因此只能用它定位 phase 和 step 模式，不能据此直接宣称某个 Kernel 的硬件时间已经被证明。

## 4. 第三级：按 phase 和 step 定位

将固定轨迹结果按 `prefill`、`mixed` 和 `decode` 分开，分别汇总 observed service、predicted device、signed error 和绝对误差贡献。先判断错误集中在哪个 phase，再在该 phase 内按下式排序代表 step：

$$
C_{step}=|T_{step}^{pred}-T_{step}^{obs}|\times \text{occurrence count}
$$

这是实际选择步骤，不是建议：必须先枚举选中 case 内该 phase 的全部 step，逐条输出 `step_id`、请求集合、token 数、观测边界、预测边界、绝对差和 `C_step`，再选择 `C_step` 最大且 workload 正常的 step。若因 trace 缺失、硬件限制或控制实验而选择非第一名，报告必须同时列出第一名、选择理由和两者完整 workload；不得把人工挑选的“大 batch”称为“最大误差 step”。

代表 step 必须用稳定的 `step_id` 连接以下事实：scheduled request IDs、每个请求的 prefill/decode token 数、总 token 数、KV 长度、并行 rank、Graph/eager 路径和模型层数。不能因为两个 step 都是“第 40 步”或 M 相同就认为负载相同。

在启动 profiler 前，必须先记录原 trace 的 `Graph/eager` 状态、capture bucket 和 fallback 原因，并在报告中并列记录 profiler 运行的同三个字段。要做原 Serving step 的 kernel 归因，profiler 运行必须使用相同 Graph/eager 状态；Graph Full step 必须在相同 capture bucket 下采集并解析 Graph node。若为了枚举内部 kernel 额外关闭 Graph，只能作为独立 eager replay：可以验证算子/launch，不能与原 Graph step 的时间相减、不能进入 family 误差、Top-5 或根因结论。

### Step 闭合表

每个代表 step 至少报告：

| 字段 | 含义 |
|---|---|
| observed full step | Scheduler step 完整墙钟区间 |
| observed serialized service | 时间线上未被前一 step 隐藏的 service |
| RPC envelope | Scheduler 调 executor 到返回的区间，不等于纯 Device |
| worker submit | Host 提交 Graph 或 eager kernels 的区间 |
| device task union | 同一 step、同一 device 上 task 区间的 overlap-safe union |
| inter-task/runtime gap | service 中未被 device task union 覆盖的区间，原因未证明时保持 unknown |
| predicted timeline | 当前 workload、lowering、launch 和 predictor 生成的时间线 |

所有区间必须来自同一次运行和可证明的时钟映射。不同运行、不同 clock domain 或不同 step 的数字禁止直接相减。

## 5. 第四级：从 step 分解到 Kernel family

对目标 step 保留完整 `workload graph → KernelPlan → predictions → execution timeline`，同时输出两类视图：

1. 预测侧 additive contribution：按 GEMM、FlashAttention、Fused-MoE、Vector、KV cache、communication 和 transfer 汇总调用时间，用于确定优先检查对象。
2. 时间线侧 critical-path contribution：按真实依赖和资源重叠计算对 step 完成时间的贡献，用于解释为什么某个高占比调用未必等量影响 step。

`predicted_additive_share` 只是候选排序依据，不是实测误差归因。TP/EP rank 上并发的调用不能重复累加到 step 关键路径；同一个 batch Kernel 也不能按请求数重复计入系统总时间。

对每个 family 按以下键继续分桶：

```text
KernelKind × variant × dtype × physical shape × normalized launch program
```

排序优先级是 `绝对误差贡献 > 是否系统同向 > 相对误差`。一个短 Kernel 的高 APE 通常不应优先于跨几十层重复、累计贡献数毫秒的 GEMM。

## 6. 第五级：核对 Dataset 与 Serving 是否真的是同一个输入

“调用同一个 predictor 函数”不等于输入一致。对候选 Kernel 必须逐字段比较：

- `KernelSpec`：kind、variant、dtype、B/M/N/K 或其他完整物理 shape；
- Tensor 语义：rank、view shape、storage shape、stride、layout、transpose、bias 和 epilogue；
- Launch：algorithm、kernel family/symbol、grid、block、tile、pipeline depth、split/reduction、worker 数、single-core 几何、遍历顺序和 L1/L2 区域；
- 调用边界：一个 public op 是否展开为多个 device task；
- 执行条件：Graph/eager、warm/cold cache、功耗状态、并发 stream 和频率 contract；
- Predictor：最终都经 `src.kernel.kernel_base.predict(KernelSpec, HWSpec, method)`，且没有 Dataset/Serving 专属公式。

结果分为三类：

- exact coverage：Dataset 与 Serving 的完整物理输入和 normalized launch program 相同；
- shape-only coverage：shape相同但 launch、rank、layout 或调用边界不同，不能继承精度结论；
- extrapolation：Dataset 没有相同物理 case，应优先对高贡献 case做独立硬件闭环。

## 7. 第六级：同运行 profiler 分清 Kernel 与 runtime

对一个代表 step 做窄 profiler，而不是直接 profile 全矩阵。Profiler 运行必须保存与无 profiler golden 相同的 request-to-step 身份，并检查 observer effect；若 profiler 使 step 或请求指标明显变化，它只能提供机制线索，不能作为生产时间真值。

A800 可在同一 Serving 采集入口设置 `MIDLAYER_A800_NSYS_STEP_NVTX=1` 和绝对路径 `MIDLAYER_A800_NSYS_OUTPUT`，使 worker 用 `midlayer-step:<step_id>` NVTX range标记正式 step并由 Nsight Systems包裹服务进程；未设置时不改变正常采集路径。C500 和 910B 应使用各自受支持的 profiler correlation，并遵守对应硬件 profiling contract。

应从时间线得到：

```text
device busy = union(device kernel intervals, device copy intervals)
runtime/gap = observed service - overlap-safe covered intervals
```

`runtime/gap` 不是一个物理机制名称。它可能包含 Host 未及时提交、Graph replay 边界、同步、driver/runtime、跨 stream 依赖、collective 或 profiler 无法关联的 task；在缺少证据时必须保持 unknown。Kernel duration 也必须按区间求 union，不能把重叠 task 的 duration直接求和。

判定规则：

| 证据 | 结论 |
|---|---|
| 实测 device task union 与预测 step 接近，但 service 仍有大 gap | 优先检查 runtime、Graph 和 Host/Device 编排 |
| 实测 device task union 本身与预测差距大 | 继续进入单 Kernel 归因 |
| 个别 family 的实测总时长与预测同方向偏差且贡献大 | 对该 family 的代表 shape做独立 Event |
| profiler 运行显著变慢 | 只保留 launch、顺序和非 duration counter，重新取得无扰动 timing 证据 |

## 8. 第七级：单 Kernel 硬件闭环

对贡献最大的少量 shape 使用对应 `scripts/kernel_dataset/<hardware>/` runner 做独立 Event 测量，原始结果写入 `profiler_output/`，不得直接修改 canonical Dataset。测量必须复现 Serving 的 public-op 语义，而不仅是数学 shape。

每个候选 Kernel 应同时列出四个数字：

| 数值 | 用途 |
|---|---|
| Midlayer prediction | 被验证对象 |
| exact canonical Dataset golden | 仅在完整物理输入一致时作为对照 |
| same-step profiler kernel/task | 验证真实 Serving 内部执行，但受 profiler 扰动和边界限制 |
| independent Event | 验证隔离 public op；需匹配 rank/layout/launch/cache 等 contract |

这些数字不要求天然相等。Dataset golden可能是 public-op Event，profiler可能是单个 kernel body，Serving 可能执行多个 task并含 task 间空隙；只有边界一致时才能计算误差。若独立 Event 与 same-step profiler显著不同，先排查调用边界、Graph/eager、缓存、频率、tensor rank/layout/stride 和实际 launch，不能先改公式。

一个单 Kernel 的 step 影响按调用次数和时间线语义计算。串行重复时：

$$
\Delta T_{step}\approx (T_{kernel}^{pred}-T_{kernel}^{meas})\times N_{critical\ calls}
$$

几十到上百微秒的单次误差经过几十层重复后会成为数毫秒到数十毫秒的 TPOT 误差。并行 rank只按关键路径计一次，不能乘以 rank 数。

## 9. 第八级：定位物理机制

单 Kernel 误差确认后，按以下顺序检查，不按最终 APE 猜公式：

1. 工作量是否正确：shape、rank、layout、stride、dtype、epilogue、token/expert 数量。
2. Launch 是否正确：实际 kernel、grid/block、tile、pipeline depth、split、single-core 工作和遍历顺序。
3. Traffic 是否正确：DRAM/L2/L1/UB/SMEM bytes、cache residency、重复读取、writeback 和尾块。
4. Pipeline 是否正确：启动、稳态重叠、barrier、尾部排空、descriptor 数和 completion 顺序。
5. Resource envelope 是否正确：计算、带宽、局部容量和并发资源是否使用正确单位与并行度。
6. Runtime 边界是否缺失：Graph replay、launch、同步、copy、collective 和 kernel-between gap是否属于 Serving 层。

使用 profiler 非 duration counter、固定版本编译器/算子源码的静态语义和独立机制微基准证明原因。目标 Kernel 或 Serving duration只能验证修复影响，禁止用于拟合参数、选择阈值、增加 per-shape correction 或决定 `max/add`。

## 10. 根因分类与交付标准

最终结论必须落到以下一种或多种类别：

- input mismatch：实测与预测请求、step 或 tensor 输入不同；
- lowering omission：实际执行的 operator/kernel/copy 没有进入 KernelPlan；
- launch mismatch：shape正确，但 normalized launch program错误；
- predictor mechanism：相同 KernelSpec 与 launch 下，traffic、pipeline 或 resource公式错误；
- runtime/scheduling：device kernel之外的 Graph、Host、RPC、同步、通信或排队机制缺失；
- measurement mismatch：Event、kernel body、task union或墙钟边界不一致；
- unsupported：证据不足，不能继续作确定归因。

证据等级必须写清：

- confirmed：同一输入身份已闭合，并有独立硬件证据直接支持；
- strongly localized：step 与模型组件已稳定定位，但缺少最后一项硬件或 counter证据；
- hypothesis：只有相关性或预测侧占比；
- unsupported：关键输入或计时边界不可证明。

一次完整定位至少交付：

1. 选中 case及其请求级 paired 指标；
2. 固定轨迹前后误差；
3. Prefill/Mixed/Decode 分解；
4. 一个代表 step 的精确输入、observed service、predicted timeline和误差；
5. 预测侧 family 与 shape/launch贡献排序；
6. 同运行 profiler 的 device union与 runtime/gap边界，或明确说明 unavailable；
7. 最高贡献 Kernel 的 prediction、Dataset、same-step profiler和独立 Event对比；
8. 单 Kernel 误差乘调用次数后的 step贡献与 closure residual；
9. 根因类别、证据等级、下一项最小实验或物理修复方向；
10. 明确说明所有 duration只用于 evaluation，未进入 Forward拟合。

### 强制最终汇总格式

每个被诊断硬件必须各自给出一张完整的闭合表，禁止只给全局 MAPE、只给预测侧占比，或把其他 case 的 Kernel 数据混入。表中固定使用同一个 selected case、同一个 selected `step_id` 和同一次同输入 profiler；所有时间均标注边界和单位。

| 必填项 | 必须报告的内容 |
|---|---|
| 选中 case | `hardware/model/TP/QPS`、选择依据（TPOT 绝对误差）及 paired request 数 |
| TPOT 分布 | observed 与 predicted 的 P0、P50、mean、P100，另列 mean signed error 与 paired MAPE；不得将 mean 比值称为 MAPE |
| step 排名与选中 step | 全部候选 step 的 `C_step` 排名；选中 `step_id`、phase、request 数、prefill/decode token 数、总 token、每 request 的 KV/context 摘要；非第一名时必须列第一名和选择理由 |
| Graph/Profiler 一致性 | 原 trace 与 profiler 各自的 Graph/eager 状态、capture bucket、fallback 原因；不一致时，明确标为独立 replay，禁止进入原 step 的时间误差计算 |
| step 边界 | observed full step、serialized service、RPC envelope、worker submit、device task union、runtime/inter-task gap（无法证明时为 `unknown`）、predicted timeline；禁止跨运行或跨时钟相减 |
| family 闭合 | GEMM、Attention、Vector/Norm、KV/cache、copy/communication、other/runtime：每项都列 Midlayer predicted、same-step measured task-union、signed error、APE、调用数与关键路径/加法语义 |
| 单 kernel 闭合 | 每项列完整 KernelKind/variant、shape、rank/layout/stride、normalized launch、调用数、Midlayer prediction、same-step profiler body/task、独立 Event（若有）、边界是否可比、signed error、APE、累计 step 贡献 |
| Top-5 | 按 `abs(per-call predicted - measured) × critical-call count` 排序，列出最大的五个可比 kernel；若 kernel body、public op 或 launch 边界不可比，必须排除出误差排序并单列为 `measurement mismatch` |
| 结论 | 每个 family 和 Top-5 都标 `confirmed`、`strongly localized`、`hypothesis` 或 `unsupported`；没有同输入 measurement 不得报告为“具体 kernel 误差” |

如果设备 profiler 不能完整枚举 Graph 内部节点，则仍须报告该限制、Graph task union 和可见任务；不得用 eager 的不同 Scheduler step 补成 Graph step 的单 kernel 误差。正确替代是冻结 selected step 的完整 token/context/launch workload，单独做 eager injection，并把它标为独立边界而非原 Serving step。

## 11. 常见错误

- 用 mean prediction 对 mean observation计算“MAPE”。
- 只看 E2E 或整场 wall time推断 TTFT/TPOT。
- 把 RPC envelope、worker submit或 serialized service称为纯 Device 时间。
- 把 profiler 中所有 task duration直接相加，忽略重叠。
- 关闭 Graph 后直接比较同一序号 step，忽略 Scheduler batch已经变化。
- 看到预测侧 GEMM占比高就宣称 GEMM是实测主因。
- 只比较 shape，不比较 tensor rank/layout/stride和 normalized launch。
- 把 Dataset 内相邻 shape精度推广到 Dataset 外 Serving shape。
- 用目标 duration拟合倍率、阈值、残差或 shape特判。
- 将无法解释的 gap强行归入某个 Kernel。

## 12. 最小执行清单

```text
[ ] 固定 commit、case、dataset identity 和配置
[ ] 生成 paired TTFT/TPOT/E2E/throughput 结果
[ ] 选择一个高绝对贡献 case和一个低负载控制 case
[ ] 固定 canonical Scheduler 轨迹并验证 request/token/step 守恒
[ ] 按 Prefill/Mixed/Decode 排序误差
[ ] 枚举该 phase 的全部 step，按 `C_step` 排名；若不选第一名，记录第一名与明确理由
[ ] 锁定一个稳定 step_id，不按 step序号猜负载
[ ] 在 profiler 前记录原 step 与 profiler run 的 Graph/eager、capture bucket 和 fallback；不一致时只作为独立 replay
[ ] 输出该 step 的 workload、KernelPlan、预测时间线和 family贡献
[ ] 用同运行 profiler区分 device task union与 runtime/gap
[ ] 对最高贡献的少量 Kernel核对完整物理输入与 launch
[ ] 使用独立 Event和非 duration counter完成硬件闭环
[ ] 计算单 Kernel误差对 step和TPOT的累计贡献
[ ] 以 confirmed / strongly localized / hypothesis / unsupported 标注结论
[ ] 只用合法物理证据设计修复，并用原口径复测
```

## 13. 已收口的结论：MetaX C500 的 TTFT/TPOT 误差随 QPS 单调收敛

这一节是 unsupported 边界结论的留档，目的是防止重复诊断。结论适用范围是 `metax_c500` 的 ShareGPT 15 case；`nvidia_a800` 与 `ascend_910b` 不适用，三者形态不同。

### 13.1 现象与它的真实形状

C500 的 TTFT/TPOT signed error 按 QPS 分组均值从 qps1 的 −24.83% / −23.59% 单调收敛到 qps64 的 −3.79% / −3.39%。这个分组表同时混淆了模型构成：qps1 档只有 qwen3-32b/tp2_ep1 与 qwen3-30b-a3b/tp2_ep2 两个 TP2 大模型，qps64 档只有 llama-3.1-8b、qwen3-4b、qwen3-8b 三个 TP1 dense 模型。逐模型看趋势仍然存在（五个模型中四个单调），但不能按分组均值解释。

### 13.2 层级定位

用第二级的 Scheduling Trajectory Replay 把真实 step 顺序、batch membership 和 token progress 钉死后，逐 step serialized service 的欠预测量级与闭环 TPOT signed 同号同量级；把逐 step 缺口按 decode token 数加权（TPOT 的正确权重）得到的相对误差，与闭环 TPOT signed 逐 case 对得上。**误差在每步 service，不在调度，也不在 serving 组装。** 低负载没有 idle 污染观测量：decode step 中 service 超过中位数两倍的比例，qwen3-32b/qps1 为 0.0%、qwen3-4b/qps4 为 0.7%（占 decode 总时长 0.9%）、qwen3-4b/qps64 为 0.2%。

### 13.3 单调性的机制

在 decode 内部，C500 每步的**绝对**缺口随 batch 增长得远比 step 时长慢；分母（实测 step service）随 batch 显著增长。相对误差等于缺口除以分母，于是随 QPS 单调收敛。例如 qwen3-8b/tp1_qps64 按 obs 八分位，obs 从 13.86 ms 涨到 24.34 ms（1.76 倍）而缺口从 1.916 ms 降到 1.421 ms。

这不是一个 batch 机制。原因是 **decode 的投影 GEMV 由权重流量主导，工作量本身几乎不随 batch 增长**：Forward 自己对 qwen3-4b `gate_up`（k=2560, n=19456）的定价从 m=1 的 60.3 µs 到 m=32 的 64.6 µs 只涨 7%，m>32 之后才开始明显上升。所以任何落在权重流量上的定价误差都会表现成 batch 无关的缺口。**「缺口与 batch 无关」不能用来排除 Kernel 定价类的候选。**

三卡形态差异由两个因子解释：缺口随 batch 的走向，以及分母随 batch 的增长率。C500 的缺口基本不随 batch 增长；`ascend_910b` 的缺口强烈随 batch 增长（llama-3.1-8b/tp1_qps64 从 B=1-2 的 0.653 ms 涨到 B=65-96 的 7.990 ms）；`nvidia_a800` 的分母增长最慢（qwen3-4b 从 B=1-2 到最大桶只涨 1.63 倍，C500 是 2.32 倍）且绝对缺口本来就小。**三卡不同源，不能互相套用结论。**

### 13.4 已证伪的候选

- **权重流带宽的统一 derate**：qwen3-8b 的权重字节是 qwen3-4b 的两倍（16.4 GB 对 8.05 GB），缺口却更小（1.95 ms 对 2.36 ms）。若是统一的效率误差，缺口应与权重字节成正比。
- **qwen3 与 llama 的 rope 发射数差异不是 lowering 缺陷**：llama 每层发 14 个 elementwise 加 1 个 gather，qwen3 每层发 1 个 fused `rotary`，这是 `src/serving/workload/rope_dispatch.py` 已 source-pin 的真实差异（`Llama3RotaryEmbedding` 不在 `vllm_metax` 的 OOT 注册表里，落到 `forward_native` 的 composite）。
- **每步固定 runtime/host 成本**（sampler、output 回读、graph replay 同步、executor RPC）：三条独立证据证伪。其一，同一 case 内跨 phase 缺口差 3.4 到 6 倍（qwen3-4b：decode 2.316 ms、mixed 4.484 ms、prefill 8.144 ms），而一次 step 的固定成本在三种 phase 里是同一份。其二，C500 TP1 dense 三个 case 的 decode `device_execution` host 包络几乎相同（1.281 / 1.173 / 1.201 ms），缺口却差 2.8 倍（2.316 / 1.943 / 0.837 ms）。其三，缺口没有正的下界：`llama-3.1-8b/tp1_qps64` 最大 obs 分位的缺口是 −0.765 ms，`qwen3-30b-a3b/tp2_ep2_qps16` 是 −1.234 ms，存在过预测分位就不存在所有 case 共享的正常数。

### 13.5 为什么现在关不掉：候选与各自的阻塞

缺口跟随工作量而不是跟随 step 数：跨 phase 时 token 数从 10 涨到 729，缺口从 2.32 ms 涨到 8.14 ms；decode 内部也是在投影脱离纯权重流区间（m>32）之后缺口才上升。剩余候选与阻塞如下。

| 候选 | 状态 | 证据或阻塞 |
|---|---|---|
| C500 decode 投影的 mcBLAS 定价 | **已实测确认，系数待提取** | decode 预测的 61–76% 落在 GEMM 上。2026-09-23 用 mcTracer 加 MACA graph replay 逐 shape 独立测量部署的 13 个投影 (K,N) × 9 个 M：canonical GEMM launch selection 的 tile、split-K 与归约与设备 trace **117/117 一致**（含全部 37 个 split-K cell；canonical 每个部署 shape 恰好一行，probe 的九个调用方键字段与之逐一相同，`variant` 由厂商选择而非调用方给定，分 114 个 gemm 与 3 个 gemv），所以不是 launch selection 错误；而 Forward 对同一批 launch 系统性偏快，窄输出宽度最严重（`Forward/measured` 在 N=2560 时 0.64、N=6144 时 0.75、N≥128256 时 1.00）。按层数折算，三个 dense 模型的投影定价缺口都是 1.0–1.1 ms/step，占 B=1 时 serving decode 缺口的 55–77%。物理形态是输出宽度决定 tile 数、tile 数决定并发 CTA 数、并发 CTA 数决定达成的权重读带宽，正对应 `gpu_memory_source` 自己声明缺失的 `independent_requests_per_producer` 与 `producers_per_unit` 契约 |
| VECTOR glue 的 source-bound contract 覆盖率 | 开放 | B=1 decode step 中 75% 的预测时间没有 contract；部署的 vLLM 0.9.1 树不含 `csrc`，拿不到 kernel 源码 |
| 每步固定 runtime/host 成本 | **已证伪** | 见 13.4 |

`tn_128x64x256_2m1n8k_256t` 这一个岛的 operand pipeline 已经在非生产几何上测出并 held-out 打分：538 个计时点、42 组、逐点查身份，机制对岛内 32 个部署 cell 给 3.85% MAPE，当前 Forward 是 5.46%。物理形态是一次 C 个 CTA 的发射在 104 个 AP 上占 `ceil(C/104)` 波、最后一波只有 `C mod 104` 宽，达成的权重读速率由波的填充度决定，残余的波数效应在满填充时消失、在低填充时显著（填充 1.000 时 1 波 1572、3 波 1562 GB/s；填充 0.775 时 1 波 1509、4 波 1346）。该参考标 NOT WIRED，接线属实现与三卡复验，不再需要新测量。其余岛的系数仍未发布，原因是选择面本身：部署的 13 个 (K,N) × 9 个 M 上出现 **22 个不同的编译程序**，M=1 一律走 `b16gemvt` GEMV 而不是 tiled 程序，选择器对 M 还不单调（K=4096、N=2560 在 M=2/4/8/32/128 走 split-K，在 M=16/64/96 不走）。既有的两份 mcBLAS 参考只覆盖 `tn_64x32x256_4m1n8k_256t` 一个岛（N 2048–4608、K 2048–3584、M 4–32），不覆盖 dense TP1 的任何部署投影。跨越这些不连续点拟合出来的系数不描述任何一个程序。选择面已作为纯离散 launch 事实发布在 `microbench/metax_c500/references/mcblas_decode_projection_program_surface_maca_3_7_2.json`，它给出下一轮可以合法扫描的岛边界：在每个岛内部用非生产几何扫 K 与 N、逐点查身份，提取 fill 与 per-round service，生产 shape 只做 held-out 打分；M=1 的 GEMV 族必须单独做。

在逐程序系数提取完成之前，这个缺口还没有可以接线的零拟合机制。**不得用 serving capture 的 duration 反推一个常数或倍率再发布**：那等于在评估真值上调参，即使它来自真实运行也不合法。

### 13.6 按收益上限排序：下一个落点该选谁

在继续投入任何一个算子族之前先做这一步，AGENTS.md 要求如此。全部数字用同一个探针（非生产几何、逐点查身份），权重用 C500 九个 TP1 dense cell 的 decode 墙钟。基准是**墙钟加权的 decode service 误差 −10.85%**。

| 候选 | 收益上限 | 说明 |
|---|---:|---|
| 部署的整个 GEMM 投影族 | **占缺口的 58%**（约 6.3 pp） | 探针覆盖了 23–33% 的 decode 墙钟，在覆盖区内 GEMM 占实测缺口的 0.43–0.79，均值 0.58 |
| 其余全部（VECTOR / FLASH_ATTN / MEMCPY / KV_CACHE） | 占缺口的 42%（约 4.6 pp） | 无独立探针；B=1 步里 75% 的预测时间没有 source-bound contract，阻塞项是部署的 vLLM 0.9.1 树不含 `csrc` |
| 其中 M=1 的 GEMV 子族 | **1.31 pp（TTFT 口径）/ 0.05 pp（TPOT 口径）** | 见下 |

**M=1 GEMV 子族的上限为什么这么小。** 探针实测它在 B=1 步里占该步缺口的 55–77%（qwen3-4b 1.051 ms、qwen3-8b 1.109、llama 0.972，每步），听起来很大；但只有 B=1 的 decode 步才把投影跑在 M=1，而**一个 B=1 步只产 1 个 token 却占掉一整步墙钟**。于是两种加权差一个数量级：token 加权下 B=1 步只占 0.4–1.3% 的 token，上限 **0.05 pp**；墙钟加权下它在 qps16/qps64 占 25–28% 的步数、18–22% 的墙钟，上限 **1.31 pp**，且 qps4 只有 0.35 pp。这个子族要做完需要覆盖六个不同的模板实例化（`b16gemvt_kernel<64,8,4,4>` / `<64,8,8,4>` / `<64,8,8,8>` / `<128,8,8,4>` / `<512,8,8,4>`、`b16gemvt_wave_kernel<64,2,8>` / `<64,8,4>` / `<512,4,4>`），每个都要单独测选择面加非生产几何，工作量与一个 tiled 岛同量级而上限小一个数量级。**按 AGENTS.md 的上限规则，不做。**

**逐程序分片接线是反效果的。** 13.7 记录的那次尝试覆盖了 decode GEMM 预测时间的 41.3%，但因为适用域限制只在 117 个部署 cell 中精确命中 27 个，结果 serving 的 TTFT 聚合反而 −0.24 pp。原因是 serving 总误差里存在方向相反的分量：改准一部分会让原先互相抵消的误差暴露出来。**结论是：这个族要么成体系地覆盖，要么不要接线。** 只测一个程序就接，期望值是零甚至负的。成体系覆盖意味着 decode 表面上那 22 个编译程序里至少要拿下占 74% 时间的前五个 tiled 程序加六个 GEMV 实例化，每个都要独立的选择面测绘与非生产几何扫描。

### 13.7 M=1 GEMV 族查到的真正缺口：C500 没有 finite-read-stream ramp

按 13.6 的上限排序转到 M=1 GEMV 族后，查到的不是八个模板实例化各自的定价问题，而是一个**硬件机制在 C500 上整个缺失**。

`dram_finite_read_stream_ramp` 在 `configs/hardware/nvidia_a800.yaml` 里已配置并由 `src/kernel/predictors/gemm.py` 通用消费（`gpu_mshr` 内存模型、variant 为 gemm/batched_gemm/moe_router），**C500 没有任何 reference**，所以这台设备上每个权重流发射都按完整的已发布容量（1653.1 GB/s）计价。实测的部署 M=1 投影根本达不到：21 MB 的投影只有 1061 GB/s，要到 1245 MB 才爬到 1652。`Forward/measured` 因此从 0.642 到 1.000 单调排列，与权重总量一一对应。

已实测并发布 `microbench/metax_c500/references/dram_finite_read_stream_ramp_maca_3_7_2.json`（探针 `microbench/metax_c500/memory/dram/finite_read_stream_ramp.cpp`，MACA graph replay 计时，grid 64–76032 × 每 CTA 8–512 KiB × 每 CTA 64/128/512 线程，三档共 522 个 cell）。

**请求深度必须进 key，这是这次测量最重要的一条。** A800 那份 reference 自己就写明不得外推到探针请求深度之外。固定 128 线程的探针对着十三个部署 GEMV 打分是 0.660–1.005，其中 grid 80 / block 512 那一格塌到 0.660；改成按部署 kernel 自己的 threads-per-CTA 运行后收到 **0.874–1.005、中位 0.971**。同一条曲线在不同请求深度上不是同一条。

**为什么这个落点比逐编译程序好。** 它是硬件机制而不是某个编译程序的系数，所以覆盖的是整个族而不是一个切片——13.7 记录的那次失败正是因为只覆盖了 117 个部署 cell 中的 27 个，改准一部分反而让原先互相抵消的误差暴露。消费代码也已经存在且通用，不需要新的 predictor 分支。

**已接线并三卡验证。** schema 扩成可选的请求深度轴（v2_threaded），A800 的 v1 走原路径、逐字节不变；消费点的查表键加上发射的 threads-per-CTA，查不到就返回原值而不取最近深度。

接线过程中查出两件事，都比结果本身更值得记：

其一，**这个机制在 C500 上原本不可达**。ramp floor 只挂在 `elif pipeline_depth is not None` 且 `_uses_k_tiled_pipeline` 的分支里，而 C500 的 mcBLAS launch 全部走前面的 `mcblas_rolling_pipeline` 分支。第一次接线后三卡 45 个 case 全部逐字段不变，原因就是这个，不是机制无效。ramp 按其 docstring 是以 max 组合的完成时间下界，所以挂到 mcBLAS 分支上是托底而非替换。

其二，**模型侧与实测侧的 launch 深度是两个不同的事实**。实测部署的 `b16gemvt` kernel 用 block 64/128/512，而 Midlayer 的 launch 模型对 decode 投影发 block 256——追踪显示 243 次查表落在未测的 256 上。补测 256 之后才真正命中。两者都要覆盖。

**适用域按被拒的区间扩张过三轮，直到追踪里不再有拒绝。** 第一次发布的面只覆盖 grid 64–76032、每 CTA 8–512 KiB，逐次查表打点显示 C500 serving 路径上 **651 次因每 CTA 长度超过 512 KiB 被拒**（512–2068 KiB，中位 1024），**17 次因 grid 40 低于 64 下限**被拒，深度轴则已够用（命中全在 256 与 512，无 depth miss）。于是把 grid 轴下探到 32、长度轴上抬到 2560 KiB 重扫四档，每档 cell 从 175 涨到 286。**域的扩张对着被拒的区间，不是按残差挑的。**

每一档深度都带自己的 `source_commit`、源码与快照 SHA256、产生它的二进制 SHA256、原始报告 SHA256，以及自己的 `production_or_evaluation_duration_consumed`；normalizer 要求这组字段精确相等并逐项校验，缺任一项即拒绝加载。

**三卡 ShareGPT 15 case 对照**（before 配置只移除这一个机制项）：A800 与 910B **15 个 case 每一个上报字段都相同**，三卡 kernel evaluation 也逐字节相同。C500 15 个 case 全部变化：

| | before | after | Δ |
|---|---:|---:|---:|
| TTFT paired MAPE | 20.89% | **19.41%** | **−1.49 pp** |
| TPOT paired MAPE | 12.36% | **8.97%** | **−3.39 pp** |

**TPOT 十五个 case 全部改善**，最大是 `qwen3-4b/tp1_qps4` 的 −20.75 → −9.48（+11.27 pp）。TTFT 十三改二退，最大改善是 `qwen3-4b/tp1_qps64` 的 −13.46 → −6.92（+6.54 pp），退步只剩 `qwen3-8b/tp1_qps64`（−1.68）与 `llama-3.1-8b/tp1_qps16`（−1.14）。

**按 QPS 分档，四档全部改善，两项指标都是**（机制的适用域在评测前已声明，四档均被覆盖）：

| | qps1 | qps4 | qps16 | qps64 |
|---|---:|---:|---:|---:|
| TTFT signed 前 | −24.83% | −19.97% | −17.27% | −3.79% |
| TTFT signed 后 | **−22.17%** | **−18.04%** | **−16.12%** | **−2.05%** |
| TPOT signed 前 | −23.59% | −12.77% | −10.34% | −3.39% |
| TPOT signed 后 | **−18.12%** | **−7.39%** | **−8.40%** | **−0.95%** |

TPOT 的单调性已被打破（qps4 的 −7.39 不再比 qps16 的 −8.40 更差），TTFT 的单调形态仍在但整体抬升。剩余部分按 13.6 的上限排序仍在 GEMM 族与非 GEMM 类里，后者仍受部署版本 kernel 源码不可得的阻塞。

**域覆盖的演进值得单独记，它说明机制对与机制承重是两件事。** 第一版面只覆盖 grid 64–76032 × 8–512 KiB，serving 侧 651 次因长度、17 次因 grid 被 fail-closed 退回，机制虽对却几乎不承重，聚合只动了 −0.09 / −0.34 pp。扩到 32 CTA / 2560 KiB 后升到 −1.37 / −3.12 pp，追踪显示 1639 命中对 80 拒绝，且拒绝全集中在两个 TP2 模型。再扩到 16 CTA / 5120 KiB 后，同样五个代表 case 的追踪是 **1719 命中、零拒绝**，聚合到 −1.49 / −3.39 pp。**fail-closed 保证没测过的地方不瞎猜，代价是覆盖不足时收益看不见；把域扩到发射真正被拒的地方，才让机制从"正确"变成"承重"。**

这同时验证了 13.1 到 13.3 的定位：单调性来自每步 service 的缺口，而这个缺口的一大块是设备的有限读流爬坡从未被建模。按 QPS 分组的 TTFT signed 均值仍在，但 TPOT 的 signed 已从 −23.59/−12.77/−10.34/−3.39 整体收窄。

**改善幅度小，且不要把它读成机制已被全面验证**：held-out 打分只覆盖十三个 M=1 GEMV cell。独立探针另外证明部署的 prefill 投影是 compute-bound 而非权重流 bound（weight GB/s 从 m=729 的 252 掉到 m=2048 的 98），其 ramp 下界 26.5 µs 远低于实测 124.95 µs，因此在 prefill 上不 binding——这一轮的变化来自 decode。

### 13.8 tiled 族接线的一次完整尝试与撤回

`tn_128x64x256_2m1n8k_256t` 这个岛的机制已经测完并 held-out 打分（见 13.5），接线也已完整实现、三卡全量验证并**撤回**。撤的只是接线代码，`microbench/metax_c500/references/mcblas_tn_128x64x256_operand_pipeline_maca_3_7_2.json` 及其测量结论保留在 main，标记 NOT WIRED。下面三件事是这次尝试的真正产出，重开此题的人应先读它们。

**一、岛内 held-out 与 serving 聚合彼此矛盾。** 同一套系数，在岛内 32 个 held-out 部署 cell 上给 3.85% MAPE 对当时 Forward 的 5.46%，接进 serving 后 C500 的 TTFT paired MAPE 却从 20.89% 变成 **21.13%（+0.24 pp，变差）**，TPOT 从 12.36% 到 12.29%（−0.07 pp），Req/s 从 6.79% 到 6.68%。**把一个算子族改准，可以在 serving 层被别处的补偿性误差吃掉。** 这不是测量错了，而是 serving 的总误差里存在方向相反、互相抵消的分量；任何只在单族上验证就宣告改善的做法都会被这一条推翻。接线前必须准备好 serving 端的同口径对照，不能用族内分数代替它。

**二、逐 case 是分化的，不是一致变差，而且分化跟着 gate 命中的波几何走。** C500 15 个 case 有 12 个变化，方向两边都有：

| case | TTFT before → after | TPOT before → after |
|---|---|---|
| llama-3.1-8b / tp1_qps4 | −14.25 → **−8.05**（+6.20） | −5.89 → **+6.59**（+12.48） |
| qwen3-8b / tp1_qps16 | −19.82 → **−14.60**（+5.22） | −13.06 → −11.58（+1.48） |
| qwen3-8b / tp1_qps4 | −23.34 → −21.04（+2.30） | −14.18 → −13.10（+1.08） |
| llama-3.1-8b / tp1_qps64 | −2.14 → **−5.62**（−3.48） | +6.08 → +7.44（+1.36） |
| qwen3-4b / tp1_qps16 | −26.20 → **−28.55**（−2.35） | −18.64 → −18.49（+0.15） |
| qwen3-32b / tp2_ep1_qps4 | −14.67 → −15.63（−0.96） | −13.61 → −13.82（−0.21） |

gate 在 117 个部署投影 cell 上精确命中 27 个，它们的波几何与所加的 DRAM 容量因子是：qwen3-4b o_proj（80 CTA，1 波，填充 0.769，因子 0.836）、qwen3-4b down（100 CTA，0.962，0.933）、qwen3-8b 与 llama 的 down（96 CTA，0.923，0.935）、qkv 在 M=64 时（96 CTA，0.923，0.935）、llama gate_up（**224 CTA，3 波，填充 0.718，因子 0.774**）。llama 的 gate_up 是被命中集合里减速最重的一个，而 llama 也是投影时间被 gate 覆盖最多的模型——它在 qps4 上大幅改善、在 qps64 上变差，因为 qps64 的 batch 更大，gate_up 在 M>32 会换到 `tn_256x128x64` 而离开 gate，命中构成随之改变。**所以逐 case 的方向由该 case 的 batch 分布决定哪些 cell 落进 gate，而不是由模型或 QPS 本身决定。** 注意 llama tp1_qps4 的 TPOT 从 −5.89 越过零点变成 +6.59：那不是改善，是把误差推到了另一侧。

**三、未闭合的残余波数依赖是这次接不上的直接原因。** 13.5 记录的隔离实验显示，同一波填充度下 1 波与 4 波的速率差在低填充处达 11%（填充 0.775：1 波 1509 GB/s，4 波 1346），到满填充才收敛到 1% 以内。当前表只有 **1/3/4 波**三行，**2 波一行从未测出**，所以 105–208 CTA 的发射一律被拒。要闭合它需要的新测量是：(a) 补 2 波区间，即让 105–208 CTA 的几何落在本程序内——这要先解决选择器在该 N 段会换程序的问题；(b) 在低填充端（0.67–0.80）把每个波数都测到，而不是只有 1 波有 0.673、3 波有 0.692；(c) 把 M 轴单独扫开，现在的表在 M∈{8,32,48,64} 上是池化的，而同几何下 M=8→M=64 的 round 会涨 0.6–9.2%。

**接线形态的教训**：C500 的 mcBLAS GEMM 由一条 source-derived `PipelineWorkPlan` 定价，不是字节/速率模型；而效率是对着裸 `capacity` 定义的（`measured_rate ÷ effective_capacity`）。把效率乘进那条 pipeline 的 DRAM 容量，得到的不是探针上量到的时间。机制本身在隔离测量里成立，错的是它与既有 pipeline 的组合方式。下次接线要么让消费点本身是字节/速率的，要么重新在 pipeline 自己的参数化下取系数。

**跨硬件隔离已实证成立**：A800 与 910B 的 15 个 case 每一个上报字段都与 before 完全相同，三卡 kernel evaluation 也逐字节相同。证据在 `experiment_logs/serving/c500_batch_scaling_20260922/wiring_eval/`。

### 13.9 收口：原始命题现在的答案，以及剩下的是什么

本节回答 13.1 提的那个问题在 ramp 接线之后的状态。口径与 13.1–13.3 完全一致。

#### 单调形态：TPOT 的已破，TTFT 的仍在

按 QPS 分档（15 个 C500 case，signed error 均值）：

| | qps1 | qps4 | qps16 | qps64 | 档间跨度 | 是否单调 |
|---|---:|---:|---:|---:|---:|---|
| TTFT 前 | −24.83% | −19.97% | −17.27% | −3.79% | 21.04 pp | 是 |
| TTFT 后 | −22.17% | −18.04% | −16.12% | −2.05% | **20.13 pp** | **仍是** |
| TPOT 前 | −23.59% | −12.77% | −10.34% | −3.39% | 20.20 pp | 是 |
| TPOT 后 | −18.12% | −7.39% | −8.40% | −0.95% | **17.18 pp** | **已破** |

分档会混淆模型构成（13.1），所以同时按逐模型看，结论一致：**TPOT 的单调性在 qwen3-4b、qwen3-8b、llama-3.1-8b 三个模型上从"单调"变成"已破"**（qwen3-32b 仍单调，qwen3-30b-a3b 本来就不单调）；**TTFT 的单调性五个模型一个都没变**，四个仍单调、30b-a3b 仍不单调。

这条分裂是有解释的，也是本 Goal 最该留下的结论：**TPOT 基本就是每步 device 时间的累积，ramp 修的正是这一项，所以它的单调形态随之消失；TTFT 还额外承载排队与 prefill 路径，它的单调形态来源不同或不止一个，没有被这次修动到。** 原始命题因此是一半被关闭、一半仍开放，而不是整体被关闭。

残差对 batch 的形状也变了。13.3 量到的是"gap 几乎不随 batch 变而分母在变"，现在逐 batch 重量：llama 的残差已在零附近震荡（−1.02 到 +1.25 ms，无系统形态），qwen3 两个模型变成 **U 形**——小 batch 最差（−18 至 −20%）、中段最好（−2 至 −8%）、大 batch 回落（−13 至 −18%）。**"batch 无关常数 ÷ 增长分母"这个描述对剩下的部分已不再成立**：它描述的那一部分正是 ramp 关掉的那一部分。

#### 剩余误差的族分布

deployed GEMM 投影有独立探针，可按当前 Forward 价逐 step 实测其份额，其余为余项：

| | 占剩余 decode 缺口 |
|---|---:|
| deployed GEMM 投影族 | **44%**（接线前 58%） |
| GEMM 之外全部 | **56%** |

逐 case 份额差异很大（0.07–0.77），且与 QPS 相关：qps4 的 case 只有 0.07–0.24，qps16/qps64 有 0.44–0.77。低 QPS 覆盖到的是小 batch 步，那里 GEMM 已被 ramp 修准，残差几乎全在别处。

预测 decode 时间的族构成：llama 与 qwen3-32b 的 GEMM 占 72–79%，qwen3-8b 66–73%，qwen3-4b 55–66%；VECTOR 一律 11–18%，FLASH_ATTN 4–21%，MEMCPY 2–6%，KV_CACHE 约 1%。**qwen3-30b-a3b 与其余四个不是同一形态**：qps4/qps16 下它的 GEMM 只占 8–9%，qps1 下也只有 14.5%，decode 由 FUSED_MOE 主导（qps1 为 46.7%），其次是 VECTOR（qps1 为 27.8%）。**针对 GEMM 的工作不会显著移动它**——这与实测一致，ramp 对它的 TTFT 只动了 +0.74 pp，是五个模型里最小的；而 VECTOR 在它这里占比远高于其余模型，是会移动它的。

#### 为什么 FLASH_ATTN 不能复用 ramp：这是机制类别边界，不是人力或源码问题

ramp 描述的是**连续的有限读流**，其 `allowed_use` 限定为 CTA grid、每 CTA 流式字节与每 CTA 线程数，`forbidden_use` 第一条就禁止外推到实测网格与每 CTA 长度域之外。而 FLASH_ATTN 的 KV 读是**分页且带间接**的：`src/kernel/predictors/flash_attn.py` 的 geometry 本身就携带 `block_table_lookup_count`（1166–1186 行），evaluation 侧 `paged_varlen` 是与 contiguous 并列的独立桶、有自己的 MAPE 与误差行为（见 `docs/flash_attention_accuracy.md`，该文记录 paged_varlen 20.41% → 19.99% 与其余桶分别统计）。

把连续流的实测曲面套到分页间接访问上，就是跨机制类别外推，被这份 reference 自己的 `forbidden_use` 挡住。**gate 只能限制影响面，不能解决"测的不是同一个东西"。** 要走这条路，前置是先证明分页 KV 读与连续读流属于同一机制类，而现有证据指向相反。**这个边界是结构性的：即使将来拿到部署版本的 kernel 源码，它也不会因此消失。**

#### VECTOR 的阻塞性质不同

VECTOR 那一块（占预测 decode 时间 11–18%）卡的是另一回事：**部署的 vLLM 0.9.1 树不含 `csrc`，拿不到 kernel 源码**，因此无法从源码推出访问模式契约（13.5）。这是取材阻塞，拿到源码即可解除，与上面那条机制类别边界不是同一性质，不要混为一谈。

### 13.10 剩余缺口的落点：三卡同源的 helper/glue 发射未被 lowering 产生

13.9 留下的那一半（TTFT 的单调形态未被 ramp 动到）追到这里。本节是新 Goal 的起点，不是本 Goal 的修复。

#### 逐类排除：缺口不在任何已建模的 kernel 类

承载 TTFT 的是 mixed 步而不是 prefill 步——一个 case 里 prefill 步恒为 1，mixed 步有 14–124 个，请求绝大多数在 mixed 步拿到首 token。mixed 步的误差随**融合 token 数**走而不随 prefill 占比走：1–512 token 区间欠预测 −14% 到 −44%，513–1024 区间翻成过预测 +2% 到 +27%。注意 mixed 步是一次融合 forward pass（`model_input_tokens == scheduled_tokens`），投影是单个 M = 总 token 的 GEMM，**prefill 与 decode token 共用同一 kernel 实例，按分量归因在 GEMM 上不成立**。

逐类份额全部用实测值算，不用预测占比——被欠预测的类在预测里本就被压低，用预测占比找元凶会系统性低估真凶：

| 类 | 占 mixed 步缺口 | 实测依据 |
|---|---:|---|
| 融合 GEMM 投影 | 0–27%，多数 <10% | 密网格探针，14 shape × M 8–1024 共 350 cell |
| FLASH_ATTN | 5–11%（下有界，见下） | canonical 14 条 `FA_SERVING_*_MIXED_*` |
| VECTOR | **−2~−3%** | canonical 296 条 |
| 其余 | **约 65–90%** | 余项 |

VECTOR 是负份额：整体 measured/predicted 为 **0.984**，Forward 实际偏慢 1.7%（ROPE 0.909 偏慢 10%），**它在抵消别人的欠预测而不是制造缺口**。

两条口径判定，因为对短 kernel 它们决定结论：C500 的 VECTOR 296 条全部是 `paired_differential_events_around_cuda_graph_replay`，graph replay 与 serving 同执行模式、配对差分抵消固定发射开销，所以两侧可减；`cross_card_span_semantics.md` 记的那个"VECTOR 偏移可达六成"是在 **A800 的仪器对**上量的，不能未经检验套到这个设计（旁证：dataset 最短 VECTOR 条目 2080 ns，同 shape graph replay 实测 2270 ns）。反过来，`flash_attn.json.zst` 有 40 条**没有 provenance 块**，恰好全是 `FA_SERVING_*`（含所用的 14 条 MIXED），只记 `granularity/source/total_duration_ns`，**测法未声明**；因此 FA 的 5–11% 只能作为下界，若那批含 eager 间隙则实测被抬高、FA 真实份额更低——方向上加强而非削弱"FA 不是元凶"。

KV_CACHE 与 MEMCPY 按预测占比合计只有 3–7%，全错也补不上缺口。**所以缺口不是"某个已建模类定价错"，而是"步里有时间没有被任何 kernel 计价"。**

#### 三卡发射计数：模型少发，不是仪器少记

方向必须先定：模型少发射是模型缺陷，trace 少记录是仪器缺陷，两者相反。判据是 canonical 侧是否声明完整、签名是否带 kernel 名与真实几何、以及差值方向。三卡的 `alignment_steps` 全部 `coverage=complete`，签名全部带 kernel 名（A800 65/65、C500 72/72、910B 50/50；910B 的 geometry 为 0 是 msprof 不记 grid/block，不影响计数），且 **canonical 均多于模型**——与仪器缺陷的方向相反（对照：MoE 那条 17–20% 未被 plugin 捕获是 trace 少于模型）。

qwen3-4b / tp1_qps4，rank 0：

| 卡 | phase | 模型发射 | canonical | 欠账 |
|---|---|---:|---:|---:|
| metax_c500 | prefill | 579 | 700 | 121（17%） |
| | **mixed** | 507–579 | 772 | **193–265（25–34%）** |
| | decode | 579–651 | 663 | 12–84（2–13%） |
| ascend_910b | prefill | 434 | 664 | **230（35%）** |
| | mixed | 434 | 635 | **201（32%）** |
| | decode | 434 | 561 | **127（23%）** |
| nvidia_a800 | prefill / mixed / decode | 待测 | 550 / 627 / 771 | — |

A800 的模型侧当时取不到，因为 trajectory replay 入口在九个 TP1 case 上抛 `Scheduler over-scheduled request output tokens`；该故障此后已由 `6f6cb7d72` 修复（按模式而非按硬件 gate：只有该请求已出现在更早某步的 `finished_request_indexes` 里才判为 async-discard placeholder，其余超预算情形原样抛错，被折算 token 计入守恒校验），根因是 vLLM 版本差异而非硬件——A800 采集用的 0.11.0 缺 `request.max_tokens` guard，C500 的 0.24.0 与 910B 的 0.18.0 都有，故那两卡 `placeholder_steps=0`。**A800 的模型侧计数因此现在可补，新 Goal 不必先修工具。**

#### 未计价的是同一批 helper/glue，三卡同源

把三卡族构成对齐后，**骨架类逐项一致**——每卡都是 `rms_norm 145 / rotary 36 / silu 36` 加各自的 GEMM 与 attention。差别只在 glue 的命名与聚合：

| | A800 | C500 | 910B |
|---|---|---|---|
| GEMM | gemm 145 | mcblas 144–145 | 并入 aclnn |
| attention | flash 72–108 | flash 72–144 | FusedInferAttentionScore 36 |
| norm | rms_norm 145 | rms_norm 145 | RmsNorm 73 + Add 72 |
| rope / act | rotary 36 / silu 36 | rotary 36 / silu 36 | _triton_rope 36 / SwiGlu 36 |
| **glue** | **elementwise 115–226 + reduce_kernel 1–73** | **elementwise 191–192 + cub:: 0–72** | **aclnn 267–372** |

欠账三卡都落在 glue 那一格，数量级相当（115–299 / 191–264 / 267–372），而骨架类三卡几乎逐项相同——**lowering 对算子主体是对的，缺的是 helper/glue 发射**。C500 的 `cub::` 前缀和是其中 phase 特异性最强的一项：prefill 72、mixed 72、**decode 0**，只在有 prefill token 的步出现，与"缺口集中在 mixed/prefill 而 decode 已被 ramp 修准"逐项对应；模型侧没有任何类别与之对应。

**判断是同源、且 hardware-independent。** 决定性的一条是欠账比例 **910B（23–35%）高于 C500（2–34%）**，直接否掉"C500 专有"；加上骨架类三卡对齐说明差异不在算子建模，以及欠账三卡同时存在且都落在 glue/helper。

这与 `wt-graph-padding-inversion` 独立得到的结论指向同一个缺口：那条线的 `repair_status = blocked_pending_source_authored_helper_contract`，要的是 Serving/lowering 里 source-authored 的 0/1/N helper-launch contract 而非参数化，其 `forbidden_shortcut` 明令禁止用拟合闭合缺口或加 case/model 常数。**两个入口、一个缺口。**

**因此这条不在 C500 线里就地改**：修它会同时改三卡预测，验收口径也不同（需三卡同时前后对照而非 C500 单卡），owner 应是 lowering 而非本 Goal 的 kernel 侧。

### 13.11 复算入口

Trajectory replay 单 case 入口见本文第 3 节。TP2 case 必须用模型专属 suite（`configs/evaluation/experiments/metax_c500_qwen3_32b_suite.json`、`metax_c500_qwen3_30b_a3b_suite.json`），默认的 `sharegpt_phase1.json` 只包含 TP1 cell。判断缺口是常数还是随工作量增长时，必须同时看跨 phase 与跨模型，只看跨 batch 会把 Kernel 定价误差误判成固定开销。

## 14. Goal 3 交付：三卡 helper/glue 调用点 0/1/N 发射计数谓词表

本节是 `wt-c500-batch-scaling` Goal 3 的交付物：从三份源码树（A800 vLLM 0.11.0、C500 vLLM 0.24.0+vllm_metax、910B vllm 0.18.0 core+vllm_ascend，`vllm_src_3ver/{a800, c500_0.24, 910b, 910b_ascend}`）逐树导出的 helper/glue 调用点发射计数谓词，以及逐步逐 phase 的 canonical 验证结论。谓词与逐项证据在 `experiment_logs/serving/helper_launch_20260923/{a800,c500,910b}_attention_predicates.txt`，验证脚本与完整报告在同目录 `g3_validation.py` / `g3_validation_report.txt`（8 case × 3 phase，全部 canonical `alignment_steps.json.zst`，只读）。910B 列以 `docs/serving_helper_launch_contract.md`（main，59fbd1c1e）§2.1/2.2/2.3 为起点验证；该文档属 `wt-graph-padding-inversion`，本 Goal 未改，发现的错误见 14.6。

### 14.1 谓词表（调用点 × 卡）

计数表达式单位：每 attention/MoE 层每 rank 每 step，除非另注。L = attention 层数，M = MoE 层数（qwen3-30b-a3b tp2_ep2 为 48）。

| 类 | 调用点（源 file:line） | 分支谓词 | A800 (0.11) | C500 (0.24+metax) | 910B (0.18+ascend) |
|---|---|---|---|---|---|
| 1 attention 输出写回 | A800 `flash_attn_varlen_func` out= 路径 / C500 `flash_attn` python 包装 slice-assign / 910B `attention_v1.py:839` `output[:n] = attn_output[:n]` | 910B：非 capturing、非 DecodeOnly+sliding、`sinks is None` → fused 路径 | 0/层（out= 原位写） | 源码 1/2/1 每层（prefill/mixed/decode 的 ATen slice-assign） | 源码 1/层（prefill/mixed）；decode 两候选分支（paged out= / fused 走 aclnn）下计数均为 0 |
| 1b qkv-split q/k 物化 | `qwen3.py` qkv.split + q_norm/k_norm（llama-3.1 无此结构 → 0） | 模型结构谓词（qk_norm 有无） | 2/层 | 2/层（llama 0） | 2/层 |
| 1b 邻位 | 910B rope 后 InplaceCopy_Slice 1/层、cache 写后 Contiguous 1/层（prefill/mixed） | 不可判定（aclnn 内部） | — | — | G5 |
| 2 MoE 路由 mask | A800 `moe_align_block_size.py:85` `expert_ids = expert_map[expert_ids]` | `expert_map is not None`（EP 开） | 1/层 index_elementwise（grid [1,1,1]）；**无 triple**（0.11 triton 路径在 kernel 内 `off_experts == -1` 置零，fused_moe.py:377-381） | — | — |
| 2 | C500 `moe_align_block_size.py:101`（metax `_prepare_expert_assignment` 保持 `ignore_invalid_experts=False` → 该行必执行） | `expert_map is not None` | — | 1/层 index_elementwise_kernel_1_1（位置规则：moe_align C++ 尾 kernel 之后、fused_moe_kernel 之前）；**无 triple**（kernel 内置零 :181-185/:486-490） | — |
| 2 | 910B `token_dispatcher.py:344-348` dispatcher 级 triple | EP 开（dispatcher 层） | — | — | **连续 4 kernel 串** Index→NotEqual→Cast→Mul 各 1/层；另有 MoeGatingTopK 1/层（gating 侧） |
| 2 邻位 | `ops.moe_align_block_size` C++ 内部分解（三树均无 csrc） | 不可判定（launcher 级） | 9/层 | 8/层（含 moe_align_block_size_opt_128_expert_kernel） | 7/层 |
| 3 模型输入输出边界 | embedding lookup + `hidden_states[logits_indices]`（A800 gpu_model_runner.py:2330/:2348、C500 :4366/:4372、910B model_runner_v1.py:1398/:1405/:1409） | 每 step 恒 1+1 | singleton 族已由模型边界发射覆盖 | 同左 | aclnnIndexSelect 2/step（已验证 MATCH） |
| cub:: DeviceScan | C500 varlen metadata（A800 host 侧计算 query_start_loc） | 该 step 含 prefill token | 0 | **2/层 iff 有 prefill token**（decode 0） | — |

### 14.2 G3 验证结论（逐步逐 phase）

`g3_validation_report.txt`：**MATCH 71 / MISMATCH(documented) 22 / 无未标注 mismatch**。canonical 每 phase 恰一个代表 step，故逐 step 验证即逐 phase 代表 step 验证。

- **G3 硬 gate 全过**：C500 `cub::` prefill 72/mixed 72/decode 0（4b/8b/30b）、llama 64/64/0——四个 C500 case 全部逐 phase 精确；910B `aclnn` triple 以连续 4 kernel 串做位置隔离后 prefill/mixed/decode 全部 48/48/48/48（原始名字计数 49 的 +1 是边界 singleton，位置规则消解）；A800 expert_map index 48 全 phase。
- 22 个 MISMATCH 全部是**已归因的 G5 项**，无一通过调数闭合：C500 写回 0（vendor elision，窗口内已证实无任何 copy 族 kernel）、910B 写回 2L−1 / 0（aclnn 层）、A800 decode 写回 2L 与 class1b 3L（graph 内部）、910B prefill class1b 3L。骨架谓词（class1b 2/层、class2、cub、IndexSelect）在三卡全 phase MATCH。

### 14.3 G4 零拟合来源表

每个计数只来自三处来源，无一项来自 duration、MAPE、残差或"凑总数"：(a) 源码分支谓词（上表"分支谓词"列，含 `expert_map is not None`、`ignore_invalid_experts=False`、qk_norm 结构谓词、attn 分支链）；(b) ATen/torch_npu 语义（view=0：dim-0 slice、`out=`；copy=1：slice-assign、跨布局 copy_；self-assignment copy_ src==dst=0——910B 外层 `forward` :1021-1024 即此，已单列）；(c) trace 测量只作验证，过滤条件是结构性的（kernel 名/几何/位置规则），在比较前固定，见 g3_validation.py。位置规则（preceded-by/followed-by、连续 4 串）只隔离同一调用点的发射，不改变计数语义；边界 singleton 的消解依据是位置结构而非数值凑合。

### 14.4 G5 不可判定清单

- C500 写回 elision：窗口内无任何 copy 族 kernel，机制在 MACA torch fork（maca3.8.2.5.torch2.10.0）编译路径，树内无源；已证伪 `maybe_contiguous`、无 `out=`、`copy_` import 三条归因。需 MACA torch 源或运行时归因实测（须主 agent 授权）。
- 910B 写回发射律：同一 python 位点 `:839` 按 case/phase 发 0/2L−1/2L−1/0，机制在 torch_npu/aclnn 派发；decode 走哪条 python 分支（paged 还是 fused-elided）静态不可判（`pa_shape_list` 成员与 cudagraph_mode 均非 case manifest 字段）——计数两分支下都是 0，计数可判、机制不可判。
- 910B 邻位 glue：InplaceCopy_Slice 1/层、cache 后 Contiguous 1/层（prefill/mixed）、v 物化 +1/层（prefill）——launcher 在 aclnn 算子边界内，python 不可见。
- A800 decode 写回 2L 与 class1b 3L：cudagraph 捕获内部发射，图外源码不可判。
- 三卡 `ops.moe_align_block_size` C++ 内部 9/8/7 launches/层：launcher 不可归因（无 csrc）。
- EP-off 分支：三卡均无非 EP MoE canonical case（30b-a3b 只有 tp2_ep2），mask-triple/index 的 0 分支不可从 trace 验证。

### 14.5 G6 有界不可判结论

三卡骨架发射谓词已闭合并逐 phase 验证；剩余不可判项全部是"python 源码之下的厂商层（MACA torch / torch_npu/aclnn / cudagraph 捕获 / C++ op）发射律"，不是源码判错。要把它们也闭合，需要的不是更多 trace，而是：厂商层源码（MACA torch、torch_npu aclnn dispatch）、0.18/0.11 的 moe_align csrc、manifest 记录 runtime 配置（cudagraph_mode、pa_shape_list）、以及一个非 EP MoE case。在此之前，这些项应按"结构已知、计数实测、launcher 不可归因"进入实现侧 owner 的待办，而不是用常数或拟合补上——这正是 `wt-graph-padding-inversion` `forbidden_shortcut` 所禁止的路径。

### 14.6 呈报主 agent：contract §2.1 的错误

`docs/serving_helper_launch_contract.md` §2.1 写"每次 attention layer 调用恰好 1 次，无例外"。三卡 canonical 实测证伪：910B 全部 case 的 decode 为 0；910B 30b prefill 为 0；4b/8b prefill/mixed 为 2 而非 1（其一在写回位置、另一 unattributed）。该文档的分支清单缺 paged `out=` 分支与上述相位律，且其 decode=0 的机制归因（若写了 paged 路径）与 trace 不符——910B 任何 trace 中都没有名字含 Paged 的 kernel，decode 跑的是 FusedInferAttentionScore。该文档属 `wt-graph-padding-inversion` 分支，本 Goal 未改；按 Goal 红线在此报告，请主 agent 裁决转交。
