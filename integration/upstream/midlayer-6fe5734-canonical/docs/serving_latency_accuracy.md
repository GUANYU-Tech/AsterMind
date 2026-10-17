# Serving 延迟精度与 Host 时间归因

本文记录 Midlayer 在 ShareGPT Serving 数据集上的延迟精度、Host launch 与动态 Scheduler CPU 时间归因，以及仍未闭合的 Device/runtime 问题。2026-08-30 的原始诊断首先建立了 Ascend 910B Graph A/B 和 Kernel 覆盖证据；随后 `research/serving-host` 在相同请求和初始状态上分别运行 `device_only`、`device_plus_launch` 和 `device_plus_launch_plus_scheduler`，并为 A800 和 910B 各完成 15 个 case 的正式指标与逐 Scheduler step 对比。对应完整报告位于 `experiment_logs/serving/sharegpt_phase1/serving-host-a800-15case-three-mode-r2/` 和 `experiment_logs/serving/sharegpt_phase1/serving-host-910b-15case-three-mode-r2/`。

## 1. 当前结论

三种模式已经形成独立闭环，固定 launch 和动态 Scheduler CPU 的贡献也已分开量化。结果否定了“漏算 Host 时间是 910B 数倍低估的主因”：在 910B 的 15 个 case 上，加入两类 Host 时间只解释聚合 TTFT、TPOT 和 E2E 绝对误差的 1.37%、4.00% 和 4.84%；最终中位 signed error 仍为 -76.64%、-45.19% 和 -37.08%。A800 是方向相反的控制组，加入 Host 时间后聚合 TTFT、TPOT 和 E2E 绝对误差分别变化 -2.05%、-7.97% 和 -10.32%，最终中位 signed error 为 +11.00%、+21.75% 和 +17.24%。负解释量表示误差变大。

逐步工作量守恒在两套硬件、三种模式上均通过，但阶段时间直接暴露了不同机制。A800 最终模式相对实测的中位总 step 时间比为 1.086，Prefill、mixed 和 Decode 分别为 1.261、1.160 和 1.071，step 数比为 0.955；它整体已经处于相近量级且略慢。910B 的对应比值只有 0.414、0.456、0.282 和 0.444，而 step 数比反而为 1.144；模拟器做了更多 step，却在三个阶段都明显过快，mixed 最严重。因此 910B 的主要剩余误差不是漏掉请求、token 或 step，也不是固定 launch 或 Scheduler Python 时间，而是 Prefill/mixed 优先的 Device/runtime 路径与调度轨迹差异。

已有 910B、Qwen3-4B BF16、单卡 TP=PP=EP=1、QPS16 的同请求低到达偏差 A/B 给出了独立因果证据：实机原本只让纯 Decode 使用 Graph，Prefill 和 mixed 仍走逐算子路径；改为全阶段 Graph 且捕获上限为 512 后，平均 TTFT 从 145.123 ms 降到 74.566 ms，平均 TPOT 从 26.094 ms 降到 17.845 ms，相对模拟器预测分别消除了 57.45% 和 53.24% 的原始 gap。三模式结果说明不能把这部分效果等同为一个静态 enqueue 常数；它还包含 Graph runtime/device 空洞、batch、排队与调度反馈。

剩余主因当前可收敛到 910B 的 Device/runtime Graph 路径和低置信度 Kernel 覆盖，阶段上以 mixed、Prefill 为先，OP family 上以 GEMM 和 Attention 为先。目标 Serving 路径按预测时间的精确数据覆盖仅 1.674%，Prefill/mixed 仅 0.087%，其中 GEMM 和 Attention 都没有精确覆盖；关键 GEMM 维度 2,560、9,728 和 151,936 缺失。当前正式逐步数据保存了阶段、request/token 工作量和完成进度，但没有同一次运行的 kernel-level profiler 明细，因此不能把最终残差进一步精确分摊到单个 OP，不能用这份聚合结果声称某个具体 OP 的时延倍率。

## 2. 指标口径

- TTFT 是一个请求从实际发出到收到首 token 的时间。
- TPOT 来自同一个请求首 token 到末 token 的间隔，反映连续输出 token 的平均节奏。
- E2E 是单请求端到端时间。
- 整场完成时间同时受请求到达窗口和尾部积压清空影响，不能用来分解 TTFT 或 TPOT。
- model Event、profiler task 时间和设备 busy time用于定位运行机制，但只有与同一次请求轨迹和同一批 step 对齐后才能进入请求级误差分解。
- `device_only` 只使用设备执行预测；`device_plus_launch` 加入运行前由独立微基准确定的静态 eager/Graph replay enqueue 服务时间，并在 Host 提交与 Device 执行之间显式建模重叠；`device_plus_launch_plus_scheduler` 再线性加入本次独立仿真中围绕真实 Scheduler 逻辑逐 step 动态测得的 CPU 时间。三种模式分别重跑 Scheduler，不在一份结果上事后加减。

## 3. 已完成因果验证的配置

| 项目 | 配置 |
|---|---|
| 硬件 | Ascend 910B，device 0 |
| 模型 | Qwen3-4B，BF16 |
| 并行 | TP=1、PP=1、EP=1 |
| 请求 | 128 个 ShareGPT 请求，输入 28,159 tokens，输出 30,470 tokens |
| 负载 | QPS16，计划到达窗口 7.481 s |
| 实机 | vLLM 0.18.0，最大 8,192 batched tokens、256 sequences |
| 模拟器 | vLLM 0.11.0 Scheduler，当前 `device_only` Forward |
| 到达序列 | 实机与模拟器读取同一个 `requests.csv`，SHA256 `ea905f74...e5f9` |
| 原配置 | `CUDAGraphMode.FULL_DECODE_ONLY`，只有纯 Decode 使用 Graph |
| 对照配置 | `CUDAGraphMode.FULL`，捕获上限 512，符合条件的 Prefill、混合批次和 Decode 都使用 Graph |

三次正式 A/B 使用相同请求、相同计划到达序列、相同专用客户端和相同非 Graph 服务参数。实际请求发出相对计划时间的最大偏差分别为 7.588、4.813 和 5.944 ms，因此不同到达输入不是这个 case 发生误差的原因。

| 指标 | 模拟器预测 | 只有 Decode 使用 Graph | 原始误差 | 全阶段 Graph、捕获上限 512 | 消除的误差 | 占原始误差 | 仍未解释 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 平均 TTFT | 22.306 ms | 145.123 ms | 122.817 ms | 74.566 ms | 70.558 ms | 57.45% | 52.259 ms |
| 平均 TPOT | 10.599 ms | 26.094 ms | 15.495 ms | 17.845 ms | 8.249 ms | 53.24% | 7.246 ms |
| 平均 E2E | 2,500.191 ms | 5,276.136 ms | 2,775.945 ms | 4,107.509 ms | 1,168.627 ms | 42.10% | 1,607.318 ms |

表中的 A/B 消除量包含 step 变快后自然产生的 batch、排队和调度反馈，不是纯 CPU 时间，也不是 Kernel 执行时间。该比例只适用于这一配置，不能推广到其他模型、并行布局或硬件。

## 4. 三种硬件的证据边界

完整 ShareGPT 矩阵包含每种硬件 15 个 case，共 45 个 case。

| 硬件 | 请求指标现象 | 已完成的原因证据 | 当前可支持的结论 |
|---|---|---|---|
| Ascend 910B | 最终三模式中位 signed error 为 -76.64%/-45.19%/-37.08%；三个阶段 step 时间都明显过快，mixed 最严重 | 15/15 case 三模式和逐步工作守恒；Qwen3-4B TP1 QPS16 完成同到达 Graph A/B、NPU Event 和动态 profiler | Host 只解释 1.37%/4.00%/4.84% 聚合绝对误差；剩余主因在 Device/runtime Graph 路径和低覆盖 GEMM/Attention，不能再归咎于漏算 Host |
| MetaX C500 | 15/15 case 都低估；中位 signed error 为 -59.3%/-53.7%/-53.4% | 历史 eager profiler 的三个 step 同时存在 device/kernel 偏差和 launch 间空洞，但 profiler 运行比生产慢 2.63 倍 | 只能确认存在两类候选问题，不能给生产 TTFT/TPOT 分配比例 |
| NVIDIA A800 | 最终三模式中位 signed error 为 +11.00%/+21.75%/+17.24%；总 step 时间中位比为 1.086 | 15/15 case 三模式和逐步工作守恒，作为误差符号相反的控制组 | Host 模型没有改善总体误差，继续证明不能使用跨硬件统一倍率 |

910B 的数值比例不能套到 C500。C500 尚未完成同口径、无 profiler、同到达、带 request-to-step 的生产重测。A800 当前整体并非系统性低估，任何面向 C500/910B 的全局修正都必须用 A800 检查回归。

## 5. 对原始问题的回答

### 5.1 是否因为实机与模拟器使用了不同到达时间

对已验证的 910B case，不是。二者使用相同计划到达序列；正式 A/B 的实际发出偏差最大不超过 7.588 ms。历史 Scheduler sidecar 与 canonical metrics 不是同一次运行，因此其时间戳不能用于回答这个问题，只能用于工作量守恒检查。

### 5.2 是否是 Scheduler Python CPU 每轮串行导致

已经由三模式独立运行完成请求级影响归因。动态 Scheduler CPU 在对应 step 边界线性累计，且没有被 Host/Device `max` 隐藏；910B 上它只解释聚合 TTFT、TPOT 和 E2E 绝对误差的 1.26%、3.25% 和 4.04%，A800 上对应贡献为 -2.41%、-7.48% 和 -9.81%。因此 Scheduler Python 不是 910B 主要残差。

`execute_model(non_block=True)` 也不是纯 CPU submit 区间：它会同步进入 driver worker，混合了 Python、runtime 和设备工作。直接把这个墙钟区间当 CPU 开销会重复计时。

### 5.3 是否是推理过程的逐算子发射开销

对已验证 910B case 的 Prefill 和混合批次路径，答案是肯定的。相同逻辑工作从 eager 逐算子提交改为 Graph 后，六个完全匹配的混合调用加速 75.91%--81.28%；已经使用 Graph 的 Decode 对照基本不变。独立的便宜算子链微基准也观察到 eager host 提交与 Event 时间同步增长，而 Graph replay 的 host 提交保持很短。

这些证据证明机制存在，但当前微基准的 eager 单算子提交时间在两次运行间变化了 29%，而且只测试了 `torch.add`。因此它不能直接成为正式 Serving 参数。

### 5.4 是否是 Kernel 性能整体低估数倍

现有证据不支持“所有 Kernel 整体低估数倍”。910B 三模式逐步结果显示 Prefill、mixed 和 Decode 最终/实测中位时间比分别为 0.456、0.282 和 0.444，但这是整阶段结果，不是单 Kernel duration。历史三个精确 step 中，Prefill 和 mixed 的 overlap-safe task union低于预测，Decode 为预测的 1.34 倍。C500 三个历史样本的 device busy与预测之比分别为 1.33、1.48 和 2.06，但该 profiler 运行明显扰动生产墙钟时间。

这不等于 Kernel 模型没有问题。现有 Kernel 数据对该 Qwen3-4B Serving 路径的精确覆盖极低，仍不足以判断剩余误差由哪些算子贡献。

### 5.5 是否由动态调度放大

调度轨迹差异已经可以按 step 数、阶段工作量和累计完成进度定量比较。910B 最终模式的 step 数是实测的 1.144 倍，但总 step 时间只有 0.414 倍，说明“更多而更快的 step”确实改变 batch 和排队；三模式各自重跑 Scheduler 后的请求指标已经包含这种反馈。实机使用 vLLM 0.18.0、模拟器使用 vLLM 0.11.0，因此仍不能把该反馈继续拆成一个与 Device 时间独立的标量放大比例。

## 6. 为什么 Kernel 数据集误差不大，Serving 却严重低估

第一，Ascend Kernel 数据集的 golden 是 `msprof_task_duration`，测量单个 accelerator task 的执行时间。它不包含 eager 模式下 host/CANN 未及时提交后续算子而形成的设备空洞。Kernel 数据集与 Serving model call测量的不是同一个时间边界。

第二，数据集对目标 Serving 路径的精确覆盖很低。完全相同 kind、variant 和 shape 的记录只覆盖 85,432/715,140 个逻辑调用，按预测时间仅占 1.674%；shape 与 launch 同时匹配的覆盖率为 0%。Prefill 和混合阶段按预测时间的精确覆盖仅 0.087%，其中 GEMM 和 Attention 都没有精确覆盖。

第三，数据集缺少 Qwen3-4B 的关键 GEMM 维度 2,560、9,728 和 151,936。Vector token 点只有 1、2、4、8、16、64、256 和 1,024，而实机 Prefill/混合 step覆盖 10--1,418 tokens。

因此，18.0% MAPE 和 20.7% WMAE只回答已有孤立 task 的平均预测误差，不能约束当前 Serving 请求路径，更不能证明端到端误差应当接近 20%。

## 7. 调查过程与方法修正

1. 完成 45-case ShareGPT 矩阵，确认 C500/910B 系统性低估，而 A800 的整体误差方向不同。
2. 对历史 Scheduler sidecar 做请求、Prefill token、Decode token和完成数守恒检查，确认没有遗漏工作；同时确认 sidecar 与 canonical metrics 跨运行，拒绝使用其时间戳归因。
3. 在 910B EngineCore 边界增加轻量计时，发现 `execute_model(non_block=True)` 的语义不是纯提交时间，因此没有把它作为 CPU 修正量。
4. 用无 profiler NPU Event区分 eager 与 Graph model call，发现 eager Prefill/混合存在近似固定的 55--67 ms 区间，而 Graph Decode 随 batch变化。
5. 运行全阶段 Graph canary，用完全匹配的逻辑工作确认 Graph 的局部效果，同时发现单 asyncio event loop造成请求发送延迟。该次请求指标被否决。
6. 改为每个请求预启动独立 event loop，完成三组正式无 profiler A/B，把最大请求发送偏差降到 7.6 ms 以下，得到 TTFT/TPOT 的请求级因果比例。
7. 动态挂载 msprof，配对 1,392 个 Event 与设备 marker。识别出 `taskType=114` 的 `NOTIFY_WAIT` 是包裹 Graph 子 task 的嵌套区间，避免将其与内部 task重复计时。
8. 回查 Kernel 数据覆盖率和历史精确 step，确认 Kernel dataset 平均误差与 Serving 严重低估并不矛盾。
9. 尝试进行 C500 同口径重测时四张卡均被其他 owner占用，没有抢占设备，因此 C500 结论保持 partial。

本轮有三类方法被明确否决：用整场完成时间解释 TTFT/TPOT；把重叠 task duration直接求和；把跨运行 profiler、Scheduler sidecar和请求指标相减。以后相同诊断必须先确认运行身份、时间边界和 request-to-step 关系。

## 8. 当前可执行的修复

### 8.1 生产部署侧

对 Ascend 910B、Qwen3-4B、TP1，可以在显存和稳定性验证通过后启用 `CUDAGraphMode.FULL`，并把 `max_cudagraph_capture_size` 设为 512。该措施已经在同请求 A/B 中显著降低 TTFT 和 TPOT，但上游仍将全阶段 Graph 标为早期能力，不能直接推广到其他模型、TP/EP 布局或硬件。

### 8.2 模拟器侧

不能把 Scheduler Python wall time或一个固定微秒数直接加到每个 step。正确实现应把 host 提交和 device 执行表示为两个可以重叠的资源：host 顺序提交 eager 算子，device 在算子已提交且依赖满足后执行；Graph 区域只支付一次 replay 提交边界。

对 eager 逻辑算子 `i`，组合关系应表达为：

```text
host_done_i = max(host_done_(i-1), step_ready) + submit_service_i
device_done_i = max(device_done_(i-1), host_done_i, dependency_done_i) + predicted_kernel_i
```

每一种模式必须独立重新运行 Scheduler，因为 step 完成时间改变后续 batch、排队和完成顺序。当前实现已经输出三条独立轨迹；A800 和 910B 的 `submit_service_i` 来自版本化独立微基准，并绑定硬件、Host、driver/runtime、软件版本和 source SHA。C500 在完成相同测量前必须 fail loud，不能复用其他硬件参数。

生产 vLLM 0.18.0 与模拟器 vLLM 0.11.0 Scheduler还需要对齐或并排运行。不能在未验证语义变化时静默替换当前 Scheduler。

## 9. 下一次实验如何闭合 TTFT/TPOT

910B 和 A800 已经按正式 schema 重采全部 15 个 active case，并在同一次无 profiler 运行中保存实际请求、正式 metrics 和轻量 Scheduler step 轨迹；alignment 只承担顺序真值，不要求时间逐步对齐。后续若要把 910B 剩余阶段误差精确分摊到 OP，需要在一个代表 case 上追加同运行身份的窄 profiler 证据，至少记录：

- 每个请求的计划到达、实际发出和服务端接收时间；
- 每个 step 的开始、提交完成、设备完成和输出更新时间；
- 每个 step 中的 request ID；
- 每个 request 在该 step 是 Prefill 还是 Decode；
- 每个 request 在该 step 处理的 token 数；
- 每个 step 完成的 request ID；
- 每个请求首 token 和后续 token 的发出时间；
- Graph 模式、捕获桶、模型与 Scheduler 版本。

这些字段可以为每个请求重建四段路径：到达后的排队、Prefill 所在 step、连续 Decode step之间的等待、以及输出处理。一个 batch 中的所有请求都等待整个 step，不能把 step 时间除以请求数。

随后在相同配置上做一次窄 profiler运行，保留完全相同的 request-to-step 字段，并用观察到的 step描述运行不读取 duration truth 的 Forward。只有 prediction与 overlap-safe leaf task union来自同一轨迹，才能把剩余设备误差按 OP、shape、variant和 launch排序。

这一实验的验收条件是 128/128 请求完成、请求与 token严格守恒、低客户端发送偏差、Event 完整配对、未匹配 task显式报告，以及所有 task时间使用区间并集而非时长求和。

C500 的下一步是在设备真正空闲时完成同样的 Qwen3-4B TP1 QPS16 无 profiler request-to-step/Event实验；只有仍存在 device/dispatch歧义时才追加窄 profiler。A800作为控制组验证候选修复不会把当前相对正常的指标推坏。

## 10. Kernel 预测精度优化路线

### 10.1 先建立同轨迹残差账本

Kernel 优化的第一张表必须基于同一次 profiler轨迹，按 OP、shape、variant和 launch列出调用次数、预测总时间、overlap-safe leaf task union、signed residual和匹配失败原因。排序依据是 Serving 预测时间贡献和重复调用放大，而不是数据点数量或最大单点百分比。

### 10.2 优先补关键覆盖空洞

优先补 Qwen3-4B 的 2,560、9,728、151,936 GEMM 维度和实际 Prefill/混合 Attention shape。Vector 覆盖应扩展到 10--1,418 token范围内由 tile、core count和 reduction机制决定的边界，而不是根据 MAPE 反向挑选分段点。

已有 T=1 RMSNorm 2.75 倍和 T=1 SiLU 1.80 倍的异常，但只有在计算其 Serving critical-path总贡献后才能决定是否优先修复。大量低成本点不应压过少量但占时间主导的 GEMM 或 Attention 路径。

### 10.3 分离 launch、task 和 runtime

Launch observation只记录 algorithm、tile、grid、block、split和其他离散发射事实，不包含 duration。相同物理 lookup key出现不同 launch时必须失败，不能静默 fallback。

Profiler归因必须区分 leaf compute、copy/memset、同步 wait和 Graph wrapper，并对重叠区间求并集。model Event减 prediction不能直接命名为 Kernel error，因为 Event中可能包含 host提交造成的设备空洞和 Graph runtime。

### 10.4 保持 Forward 零拟合

公式和参数只能来自硬件规格、编译器或算子源码的静态语义，以及独立机制微基准。Serving duration只能用于 evaluation，不能用于拟合倍率、按 shape建立时延表、选择阈值或增加 residual correction。无法由独立物理证据解释的误差应保留为 limitation或 unsupported。

### 10.5 验证标准

Kernel 修改除 MAPE 外还应报告 WMAE、median APE、精确 shape和 launch覆盖率、与实际 Serving 调用分布加权后的误差，以及对 `device_only` Serving 时间线的预计影响。A800必须作为误差方向不同的控制组；C500和910B的全局倍率修正预期会伤害 A800，因此不允许采用。

## 11. 明确禁止的错误修复

- 不把 C500/910B 的 TTFT或 TPOT 比例乘到全部 Kernel。
- 不把整场完成时间、累计 Scheduler CPU时间或累计 model Event直接加到请求延迟。
- 不把模拟器自身 Python wall time当成目标硬件 CPU时间。
- 不把跨运行 profiler task union与另一运行的预测相减。
- 不把 Graph `NOTIFY_WAIT` 与其内部 task重复计时。
- 不把 910B 单 case 的 Graph改善比例推广到 C500、A800或其他模型和并行配置。
- 不在缺少 request-to-step关键路径时声称剩余误差已经归因。
- 不使用 Serving 真值拟合 Kernel Forward。

## 12. 当前交付状态

### 12.1 Qwen3-8B QPS64 长尾回归命名

`ascend_910b/qwen3-8b/tp1_qps64` 的 TTFT 回归命名为“超 512 token capture limit 后的全 eager critical-path 暴露 + source-authored per-step Host prepare 计费叠加”。该 cell 的逻辑 batch 超过 Ascend FULL capture 上限，Graph segment 消失，ModelRunner outer-prep、rotary cache update 以及其他 eager occurrence 在每个 step 重新进入关键路径；因此本轮新增的 occurrence charges 在最高 QPS、最大 batch 上具有累积放大效应。此前 60f4d0650 加入纯加性 outer-prep 后出现 Llama 预测 TTFT 从约 285 ms 降到约 244 ms，并不违反加法本身：每步成本改变了虚拟完成时间，随后 Scheduler 的到达竞争、batch 形成、排队和完成顺序也随之改变，新的闭环轨迹可以让请求级 TTFT 变小。该现象是 batching/scheduling feedback 的证据，不是把 additive cost 当作单调请求级修正的依据。现有证据支持 qps64 回归是过度计费嫌疑和可复现的回归原因，但没有独立证据允许删除或缩放保留的 outer-prep/rotary 参数；本轮不把 selected-step framework timing 接入 Forward，也不以该命名宣称已经修复。正式评估的 `formal_conclusion_allowed=false`（alignment sample mismatch）时，这个数值只能作为非正式 checkpoint 证据。

- A800 和 910B 共 30/30 个 active ShareGPT case 已按统一 schema 重采，并完成真值加三模式、Host 分解和逐 Scheduler step 图；三模式工作守恒全部通过。
- 910B 单 case 的同到达 Graph A/B、NPU Event、动态 profiler和复现 reducer已经保存。
- 910B 该 case 的 Graph 覆盖问题已有可执行的生产缓解措施。
- C500只有历史 reduction和冻结 instrumentation，生产请求级归因尚未完成。
- 正式三模式及其固定 launch、动态 Scheduler CPU、predicted Device、Host/Device overlap 和 step wall 分解均已可用；未知或缺失 Host profile 会 fail loud。
- 常规 ShareGPT evaluation 默认只运行一次 `device_plus_launch_plus_scheduler` 完整模式，并从该次实际调度轨迹的 Host timing manifest 输出 Device active、Launch 净关键路径增量和 Scheduler CPU 增量；独立三模式只保留为需要反事实调度对照时的专项诊断，不能把同轨迹贡献分解描述为三次独立仿真。
- C500 仍缺独立 Host launch profile、统一 schema 的 15-case Scheduler trace 和三模式评测；设备没有整张空卡时不得采集。
- 910B 剩余 TTFT/TPOT 误差已定位到 Prefill/mixed 优先的 Device/runtime Graph 路径和低覆盖 GEMM/Attention，但在缺少同运行 kernel-level 明细时不做更细的 per-OP 数值分摊。

完整三模式数值、图、逐步对比和 SHA256 见 A800 与 910B 的 `serving-host-*-15case-three-mode-r2` 报告目录；原始 Graph A/B、instrumentation 与 reducer 的 provenance 保留在对应诊断制品中。
