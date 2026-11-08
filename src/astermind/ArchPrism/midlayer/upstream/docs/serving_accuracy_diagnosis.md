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

这一节是 unsupported 边界结论的留档，目的是防止重复诊断。结论适用范围是 `metax_c500` 的 ShareGPT 15 case；`nvidia_a800` 与 `ascend_910b` 不适用，三者形态不同。全部证据在 `profiler_output/serving/metax_c500/c500_batch_scaling_20260922/`（接线对照在其 `wiring_eval/`）。

### 13.1 现象、定位与机制

C500 的 TTFT/TPOT signed error 按 QPS 分组均值从 qps1 的 −24.83% / −23.59% 单调收敛到 qps64 的 −3.79% / −3.39%（这是 ramp 接线前的口径，接线后的形态见 13.6）。分组均值混淆模型构成（qps1 档只有 qwen3-32b/tp2_ep1 与 qwen3-30b-a3b/tp2_ep2 两个 TP2 大模型，qps64 档只有 llama-3.1-8b、qwen3-4b、qwen3-8b 三个 TP1 dense），但逐模型看趋势仍在：五个模型中四个单调。

Trajectory Replay 钉死真实 step 顺序、batch membership 和 token progress 后，逐 step serialized service 的欠预测量级与闭环 TPOT signed 同号同量级，按 decode token 数加权（TPOT 的正确权重）的相对误差与闭环 TPOT signed 逐 case 对得上：**误差在每步 service，不在调度，也不在 serving 组装。**低负载没有 idle 污染：decode step 中 service 超过中位数两倍的比例，qwen3-32b/qps1 为 0.0%、qwen3-4b/qps4 为 0.7%、qwen3-4b/qps64 为 0.2%。

机制不是一个 batch 机制：decode 的投影 GEMV 由权重流量主导，工作量本身几乎不随 batch 增长（Forward 对 qwen3-4b `gate_up` k=2560/n=19456 的定价从 m=1 的 60.3 µs 到 m=32 的 64.6 µs 只涨 7%，m>32 后才明显上升），落在权重流量上的定价误差因此表现成 batch 无关的缺口；每步绝对缺口随 batch 增长得远比 step 时长慢，而分母（实测 step service）随 batch 显著增长，相对误差随 QPS 单调收敛。**「缺口与 batch 无关」不能用来排除 Kernel 定价类候选。**三卡形态差异由缺口随 batch 的走向与分母增长率两个因子解释（`ascend_910b` 缺口强烈随 batch 增长，`nvidia_a800` 分母增长最慢且绝对缺口本来就小），**三卡不同源，不能互相套用结论。**

### 13.2 已证伪的候选

- **权重流带宽的统一 derate**：qwen3-8b 的权重字节是 qwen3-4b 的两倍（16.4 GB 对 8.05 GB），缺口却更小（1.95 ms 对 2.36 ms）；统一效率误差应与权重字节成正比。
- **qwen3 与 llama 的 rope 发射数差异**：不是 lowering 缺陷，是 `src/serving/workload/rope_dispatch.py` 已 source-pin 的真实差异（`Llama3RotaryEmbedding` 不在 `vllm_metax` 的 OOT 注册表，落到 `forward_native` 的 composite）。
- **每步固定 runtime/host 成本**（sampler、output 回读、graph replay 同步、executor RPC）：三条独立证据——同一 case 跨 phase 缺口差 3.4 到 6 倍（qwen3-4b：decode 2.316 ms、mixed 4.484 ms、prefill 8.144 ms）而固定成本在三种 phase 是同一份；C500 三个 TP1 dense case 的 decode host 包络几乎相同（1.281/1.173/1.201 ms）而缺口差 2.8 倍；缺口没有正的下界（llama-3.1-8b/tp1_qps64 最大 obs 分位为 −0.765 ms、qwen3-30b-a3b/tp2_ep2_qps16 为 −1.234 ms），存在过预测分位就不存在所有 case 共享的正常数。

### 13.3 当前候选与阻塞

缺口跟随工作量而不是 step 数：跨 phase token 数从 10 涨到 729 时缺口从 2.32 ms 涨到 8.14 ms；decode 内部也是投影脱离纯权重流区间（m>32）后缺口才上升。

| 候选 | 状态 | 证据或阻塞 |
|---|---|---|
| C500 decode 投影的 mcBLAS 定价 | **已实测确认，系数待提取** | decode 预测的 61–76% 落在 GEMM。逐 shape 独立测量部署的 13 个投影 (K,N) × 9 个 M：canonical GEMM launch 的 tile、split-K 与归约和设备 trace **117/117 一致**（含全部 37 个 split-K cell），不是 launch selection 错误；Forward 对同一批 launch 系统性偏快，窄输出宽度最严重（`Forward/measured`：N=2560 时 0.64、N=6144 时 0.75、N≥128256 时 1.00）。按层数折算三个 dense 模型的投影缺口均 1.0–1.1 ms/step，占 B=1 时 decode 缺口的 55–77%。物理形态是输出宽度决定 tile 数、tile 数决定并发 CTA 数、并发 CTA 数决定达成的权重读带宽，对应 `gpu_memory_source` 自己声明缺失的 `independent_requests_per_producer` 与 `producers_per_unit` 契约 |
| VECTOR glue 的 source-bound contract 覆盖率 | 开放 | B=1 decode step 中 75% 的预测时间没有 contract；部署的 vLLM 0.9.1 树不含 `csrc`，拿不到 kernel 源码 |
| 每步固定 runtime/host 成本 | **已证伪** | 见 13.2 |

收益上限已排过序（基准为墙钟加权 decode service 误差 −10.85%）：整个 GEMM 投影族占缺口的 58%（约 6.3 pp），其余全部（VECTOR/FLASH_ATTN/MEMCPY/KV_CACHE）占 42%（约 4.6 pp）；M=1 GEMV 子族上限只有 1.31 pp（TTFT 墙钟口径）/ 0.05 pp（TPOT token 口径——一个 B=1 步只产 1 个 token 却占掉一整步墙钟），却需逐个测绘六个 GEMV 模板实例化，按上限规则不做。逐程序分片接线是反效果的：覆盖 decode GEMM 预测时间 41.3% 的那次尝试只在 117 个部署 cell 命中 27 个，serving TTFT 聚合反而变差，因为总误差里存在方向相反、互相抵消的分量——**这个族要么成体系覆盖，要么不要接线。**选择面已作为纯离散 launch 事实发布在 `microbench/metax_c500/references/mcblas_decode_projection_program_surface_maca_3_7_2.json`（部署的 13 个 (K,N) × 9 个 M 上出现 22 个不同编译程序，M=1 一律走 `b16gemvt` GEMV，选择器对 M 不单调），跨越这些不连续点拟合的系数不描述任何一个程序；在逐程序系数提取完成之前，**不得用 serving capture 的 duration 反推常数或倍率再发布**。

### 13.4 已接线：dram_finite_read_stream_ramp

查到的真正缺口不是逐个 GEMV 模板实例化的定价，而是**有限读流爬坡这一硬件机制在 C500 上整个缺失**：`dram_finite_read_stream_ramp` 在 `configs/hardware/nvidia_a800.yaml:1830` 已配置并由 `src/kernel/predictors/gemm.py` 通用消费，而 C500 没有 reference，每个权重流发射都按完整已发布容量（1653.1 GB/s）计价；实测部署 M=1 投影 21 MB 只有 1061 GB/s，要到 1245 MB 才爬到 1652，`Forward/measured` 从 0.642 到 1.000 单调排列、与权重总量一一对应。参考已发布：`microbench/metax_c500/references/dram_finite_read_stream_ramp_maca_3_7_2.json`（探针 `microbench/metax_c500/memory/dram/finite_read_stream_ramp.cpp`）。

两条接线要点：**请求深度必须进 key**——固定 128 线程的探针对十三个部署 GEMV 打分 0.660–1.005，改成按部署 kernel 自己的 threads-per-CTA 运行后收到 0.874–1.005、中位 0.971，同一条曲线在不同请求深度上不是同一条；**模型侧与实测侧的 launch 深度是两个事实**——实测 `b16gemvt` 用 block 64/128/512，而 launch 模型对 decode 投影发 block 256，两者都要覆盖。该机制在 C500 上原本不可达：ramp floor 只挂在 k-tiled pipeline 分支，mcBLAS launch 全走 `mcblas_rolling_pipeline` 分支，第一次接线三卡 45 个 case 逐字段不变的原因就是这个，不是机制无效；ramp 是以 max 组合的完成时间下界，挂到 mcBLAS 分支是托底而非替换。schema 扩成可选请求深度轴（v2_threaded），A800 v1 走原路径逐字节不变，查不到深度就返回原值而不取最近。

接线后 C500 15 case：TTFT paired MAPE 20.89% → **19.41%**（−1.49 pp），TPOT 12.36% → **8.97%**（−3.39 pp）；TPOT 十五个 case 全部改善（最大 qwen3-4b/tp1_qps4 的 −20.75 → −9.48），TTFT 十三改二退。A800 与 910B 15 个 case 每个上报字段都与 before 相同，三卡 kernel evaluation 逐字节相同。覆盖演进的教训：**机制对与机制承重是两件事**——fail-closed 保证没测过的地方不瞎猜，但域不足时收益看不见（第一版面 651 次因每 CTA 长度、17 次因 grid 被拒，聚合只动 −0.09/−0.34 pp；把域扩到发射真正被拒的区间后 1719 命中零拒绝，才到 −1.49/−3.39 pp）；域的扩张对着被拒区间，不按残差挑。改善幅度小且 held-out 只覆盖十三个 M=1 GEMV cell，**不要读成机制已被全面验证**；部署的 prefill 投影是 compute-bound（ramp 下界 26.5 µs 远低于实测 124.95 µs），这一轮的变化来自 decode。

### 13.5 tiled 族接线撤回的教训

`tn_128x64x256_2m1n8k_256t` 岛的机制已测完并 held-out 打分（岛内 32 个部署 cell 3.85% MAPE，当时 Forward 5.46%），接线完整实现、三卡全量验证后**撤回**；撤的只是接线代码，测量 `microbench/metax_c500/references/mcblas_tn_128x64x256_operand_pipeline_maca_3_7_2.json` 保留在 main，标 NOT WIRED。重开此题的人应先读这三条：

1. **岛内 held-out 与 serving 聚合彼此矛盾。**同一套系数接进 serving 后 C500 TTFT paired MAPE 从 20.89% 变 21.13%（+0.24 pp，变差），TPOT 从 12.36% 到 12.29%（−0.07 pp）。**把一个算子族改准，可以在 serving 层被别处的补偿性误差吃掉**；接线前必须准备 serving 端同口径对照，不能用族内分数代替。
2. **逐 case 分化跟着 gate 命中的波几何走。**gate 在 117 个部署 cell 命中 27 个，逐 case 方向由该 case 的 batch 分布决定哪些 cell 落进 gate，不是由模型或 QPS 本身决定；llama tp1_qps4 的 TPOT 从 −5.89 越过零点变 +6.59 不是改善，是把误差推到另一侧。
3. **未闭合的残余波数依赖是接不上的直接原因。**同一填充度下 1 波与 4 波速率差在低填充处达 11%，满填充才收敛到 1% 以内；表只有 1/3/4 波三行，2 波从未测出，105–208 CTA 的发射一律被拒。闭合需要：补 2 波区间（先解决选择器在该 N 段换程序的问题）、低填充端每个波数都测到、M 轴单独扫开（同几何 M=8→64 的 round 涨 0.6–9.2%）。

**接线形态教训**：C500 的 mcBLAS GEMM 由 source-derived `PipelineWorkPlan` 定价，不是字节/速率模型，而效率是对着裸 `capacity` 定义的（`measured_rate ÷ effective_capacity`）；把效率乘进 pipeline 的 DRAM 容量得到的不是探针量到的时间。机制本身在隔离测量里成立，错的是与既有 pipeline 的组合方式——下次接线要么让消费点本身是字节/速率的，要么在 pipeline 自己的参数化下取系数。跨硬件隔离已实证：A800 与 910B 15 case 每个上报字段与 before 完全相同，三卡 kernel evaluation 逐字节相同。

### 13.6 ramp 之后的单调性与剩余缺口

按 QPS 分档（15 个 C500 case，signed error 均值；口径与 13.1 一致）：

| | qps1 | qps4 | qps16 | qps64 | 是否单调 |
|---|---:|---:|---:|---:|---|
| TTFT 前 → 后 | −24.83% → −22.17% | −19.97% → −18.04% | −17.27% → −16.12% | −3.79% → −2.05% | **仍是** |
| TPOT 前 → 后 | −23.59% → −18.12% | −12.77% → −7.39% | −10.34% → −8.40% | −3.39% → −0.95% | **已破** |

逐模型结论一致：TPOT 单调性在 qwen3-4b、qwen3-8b、llama-3.1-8b 上从单调变成已破（qwen3-32b 仍单调，qwen3-30b-a3b 本来就不单调），TTFT 五个模型一个都没变。**TPOT 基本就是每步 device 时间的累积，ramp 修的正是这一项，所以它的单调形态随之消失；TTFT 还额外承载排队与 prefill 路径，它的单调形态来源不同或不止一个。**残差对 batch 的形状也变了：llama 在零附近震荡（−1.02 到 +1.25 ms，无系统形态），qwen3 两个模型成 U 形（小 batch −18 至 −20%、中段 −2 至 −8%、大 batch −13 至 −18%）——「batch 无关常数 ÷ 增长分母」对剩余部分已不再成立。剩余 decode 缺口中 deployed GEMM 投影族占 **44%**（接线前 58%），GEMM 之外全部占 **56%**；qwen3-30b-a3b 形态不同（qps4/qps16 下 GEMM 只占 8–9%，decode 由 FUSED_MOE 主导、VECTOR 次之），针对 GEMM 的工作不会显著移动它，VECTOR 会。

**FLASH_ATTN 不能复用 ramp，这是机制类别边界**：ramp 描述连续的有限读流，其 `forbidden_use` 第一条就禁止外推到实测网格与每 CTA 长度域之外；而 KV 读是分页且带间接的，`src/kernel/predictors/flash_attn.py` 的 geometry 本身就携带 `block_table_lookup_count`（1228–1248 行），evaluation 侧 `paged_varlen` 是与 contiguous 并列的独立桶。把连续流的实测曲面套到分页间接访问上就是跨机制类别外推，gate 只能限制影响面，不能解决「测的不是同一个东西」；这个边界是结构性的，即使将来拿到部署版本 kernel 源码也不会消失。**VECTOR 的阻塞性质不同**：部署 vLLM 0.9.1 树不含 `csrc` 是取材阻塞，拿到源码即可解除，不要与机制类别边界混为一谈。

### 13.7 剩余缺口的落点：三卡同源的 helper/glue 发射未被 lowering 产生

现行实现补充（2026-10-30）：本节基于 2026-09-23 的发射计数快照。此后 `src/kernel/launch_config/metax_c500/flash_attn_launch.py` 已把 C500 prefill/mixed 的每层 `DeviceScanInitKernel` 与 `DeviceScanKernel` 作为两次 setup launch 送入 lowering，定向测试 `test_metax_split_forward_zero_fills_the_whole_call_once` 验证了该路径。因此下文的 `cub::` 发射欠账只描述当时快照，不能再作为现行模型的缺失发射数。现行 scan 用通用 VECTOR fill/copy 定价；其 canonical 程序身份与现有独立 Small 探针不一致，定价仍未闭合，见 `docs/decisions.md` 的 C500 DeviceScan 条目。

承载 TTFT 的是 mixed 步（一个 case 里 prefill 步恒为 1，mixed 步 14–124 个），误差随融合 token 数走：1–512 token 欠预测 −14% 到 −44%，513–1024 翻成过预测 +2% 到 +27%；mixed 步是一次融合 forward pass（`model_input_tokens == scheduled_tokens`），prefill 与 decode token 共用同一 kernel 实例，按分量归因不成立。逐类份额全部用实测值算（用预测占比会系统性低估被欠预测的类）：融合 GEMM 投影 0–27%（多数 <10%），FLASH_ATTN 5–11%（只能作下界——canonical 有 40 条 `FA_SERVING_*` 没有 provenance 块、测法未声明，若含 eager 间隙则真实份额更低），VECTOR **−2~−3%**（整体 measured/predicted 0.984，在抵消别人的欠预测而不是制造缺口），KV_CACHE 与 MEMCPY 合计 3–7%，**其余约 65–90% 没有被任何 kernel 计价**。口径上 C500 的 VECTOR 条目全部是 `paired_differential_events_around_cuda_graph_replay`，graph replay 与 serving 同执行模式、配对差分抵消固定发射开销，两侧可减；而「VECTOR 偏移可达六成」是在 **A800 的仪器对**上量的，不能未经检验套到这个设计（旁证：dataset 最短 VECTOR 条目 2080 ns，同 shape graph replay 实测 2270 ns）。

三卡发射计数证明方向是**模型少发，不是仪器少记**：canonical `alignment_steps` 全部 `coverage=complete`、签名带 kernel 名、数量均多于模型。欠账三卡都落在 glue 一格：骨架类逐项一致（每卡 `rms_norm 145 / rotary 36 / silu 36` 加各自 GEMM 与 attention），glue 为 A800 elementwise 115–226 + reduce_kernel 1–73、C500 elementwise 191–192 + `cub::` 0–72、910B aclnn 267–372；C500 的 `cub::` 前缀和 phase 特异性最强（prefill 72、mixed 72、decode 0），与「缺口集中在 mixed/prefill 而 decode 已被 ramp 修准」逐项对应。欠账比例 910B（23–35%）高于 C500（2–34%），直接否掉「C500 专有」——**判断是同源且 hardware-independent**，与 `wt-graph-padding-inversion` 的 `repair_status = blocked_pending_source_authored_helper_contract` 指向同一个缺口（两个入口、一个缺口），其 `forbidden_shortcut` 禁止用拟合或 case/model 常数闭合。**因此不在 C500 线里就地改**：修它会同时改三卡预测，验收需三卡同时前后对照，owner 是 lowering 而非 kernel 侧。A800 模型侧计数当时因 trajectory replay 抛 `Scheduler over-scheduled request output tokens` 取不到，已由 `6f6cb7d72` 修复（根因是 vLLM 版本差异：0.11.0 缺 `request.max_tokens` guard，0.24.0 与 0.18.0 都有），现在可补。

### 13.8 复算入口

Trajectory replay 单 case 入口见本文第 3 节。TP2 case 必须用模型专属 suite（`configs/evaluation/experiments/metax_c500_qwen3_32b_suite.json`、`metax_c500_qwen3_30b_a3b_suite.json`），默认的 `sharegpt_phase1.json` 只包含 TP1 cell。判断缺口是常数还是随工作量增长时，必须同时看跨 phase 与跨模型，只看跨 batch 会把 Kernel 定价误差误判成固定开销。

## 14. Goal 3 交付：三卡 helper/glue 调用点 0/1/N 发射计数谓词表

本节是 `wt-c500-batch-scaling` Goal 3 的交付物：从三份源码树（A800 vLLM 0.11.0、C500 vLLM 0.24.0+vllm_metax、910B vllm 0.18.0 core+vllm_ascend，位于 `/share_data/yangxinhao/chip_modeling/upstream/vllm/{v0.11.0_a800, v0.24.0_c500_sdist, v0.18.0_ascend, vllm_ascend_plugin}`）逐树导出的 helper/glue 调用点发射计数谓词，以及逐步逐 phase 的 canonical 验证结论。谓词与逐项证据在 `profiler_output/serving/helper_launch_20260923/{a800,c500,910b}_attention_predicates.txt`，验证脚本与完整报告在同目录 `g3_validation.py` / `g3_validation_report.txt`（8 case × 3 phase，全部 canonical `alignment_steps.json.zst`，只读）。910B 列以 helper-launch contract（main，59fbd1c1e；2026-10-12 起并入 `docs/serving_system.md` §11）的三类调用点为起点验证；该 contract 属 `wt-graph-padding-inversion`，验证中证伪的「每次 attention layer 调用恰好 1 次，无例外」断言已按其结论订正（三卡 canonical 实测：910B 全部 case 的 decode 为 0、910B 30b prefill 为 0、4b/8b prefill/mixed 为 2 而非 1）。

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

每个计数只来自三处零拟合来源，无一项来自 duration、MAPE、残差或「凑总数」：(a) 源码分支谓词（14.1「分支谓词」列，含 `expert_map is not None`、`ignore_invalid_experts=False`、qk_norm 结构谓词、attn 分支链）；(b) ATen/torch_npu 语义（view=0：dim-0 slice、`out=`；copy=1：slice-assign、跨布局 copy_；self-assignment copy_ src==dst=0——910B 外层 `forward` :1021-1024 即此，已单列）；(c) trace 测量只作验证，过滤条件是结构性的（kernel 名/几何/位置规则），在比较前固定，见 g3_validation.py。位置规则（preceded-by/followed-by、连续 4 串）只隔离同一调用点的发射，不改变计数语义；边界 singleton 的消解依据是位置结构而非数值凑合。

**G5 不可判清单**（全部是 python 源码之下厂商层——MACA torch / torch_npu aclnn / cudagraph 捕获 / C++ op——的发射律，不是源码判错）：

- C500 写回 elision：窗口内无任何 copy 族 kernel，机制在 MACA torch fork（maca3.8.2.5.torch2.10.0）编译路径，树内无源；已证伪 `maybe_contiguous`、无 `out=`、`copy_` import 三条归因。
- 910B 写回发射律：同一 python 位点 `:839` 按 case/phase 发 0/2L−1/2L−1/0，机制在 torch_npu/aclnn 派发；decode 走哪条 python 分支（paged 还是 fused-elided）静态不可判（`pa_shape_list` 成员与 cudagraph_mode 均非 case manifest 字段），计数两分支下都是 0——计数可判、机制不可判；trace 侧 910B 没有任何名字含 Paged 的 kernel，decode 跑的是 FusedInferAttentionScore。
- 910B 邻位 glue：InplaceCopy_Slice 1/层、cache 后 Contiguous 1/层（prefill/mixed）、v 物化 +1/层（prefill）——launcher 在 aclnn 算子边界内，python 不可见。
- A800 decode 写回 2L 与 class1b 3L：cudagraph 捕获内部发射，图外源码不可判。
- 三卡 `ops.moe_align_block_size` C++ 内部 9/8/7 launches/层：launcher 不可归因（无 csrc）。
- EP-off 分支：三卡均无非 EP MoE canonical case（30b-a3b 只有 tp2_ep2），mask-triple/index 的 0 分支不可从 trace 验证。

要把这些项也闭合，需要的不是更多 trace，而是厂商层源码（MACA torch、torch_npu aclnn dispatch）、0.18/0.11 的 moe_align csrc、manifest 记录 runtime 配置（cudagraph_mode、pa_shape_list）和一个非 EP MoE case；在此之前按「结构已知、计数实测、launcher 不可归因」进入实现侧 owner 的待办，不得用常数或拟合补上——这正是 `wt-graph-padding-inversion` `forbidden_shortcut` 禁止的路径。
