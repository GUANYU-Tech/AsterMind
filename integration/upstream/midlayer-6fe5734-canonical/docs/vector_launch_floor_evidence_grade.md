# VECTOR 时延：三卡误差差 2–3 倍，根因是 launch floor 的证据等级不同

## 实测（`1b6bf7206`，kernel evaluation 同一口径）

| 族 | A800 | C500 | 910B |
|---|---|---|---|
| VECTOR ALL | **14.52%** | 6.49% | **15.07%** |
| NORM | 18.31 | 5.66 | 19.95 |
| ELEMENTWISE | 15.49 | 5.99 | 11.96 |
| KV_CACHE | 15.46 | 6.41 | 15.53 |
| ACTIVATION | 7.87 | 4.04 | 13.93 |

A800 与 910B 的中位 p/m 在几乎每个子类型上都 < 1（0.844–0.946 / 0.879–0.932），C500 居中（0.960–1.021）。**两卡单向欠预测、第三卡不欠**——这排除了子类型公式本身，指向三卡共用路径上的一个缺失项。

## 按预测瓶颈分组，信号是单向的

| 卡 | 瓶颈 | n | MAPE | 中位 p/m | 欠预测数 |
|---|---|---|---|---|---|
| A800 | **launch** | 62 | 16.93 | **0.839** | **62 / 62** |
| A800 | pipeline | 109 | 17.69 | 0.898 | 78 |
| A800 | dependency | 124 | 10.62 | 0.943 | 81 |
| 910B | **pipeline** | 226 | 16.43 | **0.894** | 186 / 226 |
| 910B | launch | 65 | 11.21 | 0.914 | 49 |
| C500 | pipeline / launch / dependency | 156 / 86 / 44 | 7.43 / 6.04 / 4.55 | 1.012 / 1.001 / 0.972 | — |

A800 的 launch 组 **62/62 全部欠预测**，完全单向 —— 这是缺一个常数项的形状，不是噪声。

## 根因：floor 证据是在空任务上标定的

| 卡 | `launch_latency_model.model` | 非空任务 floor |
|---|---|---|
| A800 | `isolated_**empty**_task_scheduler_v1` | **无**（唯一探针 `add_rmsnorm_queued_wave_floor_sm80.json` 标记 `evidence_status: mechanism_rejected`、`production_predictor_consumption_allowed: false`） |
| C500 | `isolated_**nonempty**_task_scheduler_v1` | **有** —— `runtime_task_latency_models.pytorch_bf16_binary_tensor_iterator` |
| 910B | **无** launch 模型 | 无 |

`formula.py:8115-8125` 把这条差异写得很直白：有 `runtime_task_class` 时 `dispatch_s = runtime_task_class_duration_ns(...)`（C500 的实测非空任务 floor），否则 `dispatch_s = dispatch_overhead_ns`（A800/910B 的通用常数）。

**空 kernel 从不发出一次 load**，因此其 floor 在物理上不可能包含真实任务被拉起后必须串行支付的首次访存/流水填充；C500 的曲线是在非空任务上测的，已经含这笔成本。这就是跨卡不对称的来源。

量级自洽：A800 launch 组中位亏空约 0.53 µs（0.839 × ~3.3 µs），与独立实测的 A800 单 warp dependent-load 延迟 ~610 ns 同量级——两者都是既有的、可溯源的独立参数。

## 落地契约（路线 A，不需新采集）

**Gate 必须是离散可溯事实，不是误差门槛**：`hardware.launch_latency_model["model"]` 已是一等字段且在 `hardware_spec.py:744/752` 与 `formula.py:7786/7795` 上穷举分派。按 `isolated_empty_task_scheduler_v1` 分支加项，`isolated_nonempty_task_scheduler_v1` 分支不动 —— **C500 由构造逐位不变**，不是靠调参守住。

**改动**：empty-task 标定的卡上，整任务 floor 需串行追加该 kernel 读路径的首次请求延迟（`memory_diagnostics[].first_request_latency_s`，模型中已作为 `memory_latency:` service 存在），理由是 CTA 未被拉起前无法发出第一条 load，二者物理上不可重叠。

**适用域**：仅 `isolated_empty_task_scheduler_v1`（当前只有 A800）。910B 没有 launch 模型，走 `dispatch_overhead_ns`，需单独判断，不在本次改动内。

**预期影响**：A800 launch 组 62 case 由 0.839 抬向 1.0；VECTOR ALL 14.52% → 约 11–12%；C500 逐位不变；910B 不变。落地后须以同一口径给出实际变化，若与预期不符则以实测为准并说明。

## 未采纳的做法

按卡硬编码（`if hardware_id == "nvidia_a800"`）——它把一个物理事实伪装成设备特例，且在新卡上会静默失效。按 evaluation 误差挑候选同样被排除（`docs/flash_attention_accuracy.md` §6 已记录为错误做法）。

## 路线 A 第一次落地尝试：gate 成立，附着点不成立

**改动**：在 `formula.py` 的 dispatch floor 分支上，按 `launch_latency_model["model"] == "isolated_empty_task_scheduler_v1"` 这一离散事实 gate，串行追加该 kernel 读路径的首次请求延迟（`memory_diagnostics[].first_request_latency_s`，取并发读路径的最大值而非求和）。

**gate 按设计生效，逐位可验**：

| 卡 | 逐位相同的 case | 说明 |
|---|---|---|
| metax_c500 | **1027 / 1027** | 走 `runtime_task_class` 分支，非空任务 floor，未进入改动分支 |
| ascend_910b | **1027 / 1027** | 无 `launch_latency_model`，未进入改动分支 |
| nvidia_a800 | 54 / 1027 | 唯一 empty-task 标定卡 |

C500 与 910B 的不变是**由结构保证**的，不是靠调参守住的。

**A800 实测变化**（kernel evaluation 同一口径）：

| 子类型 | MAPE 前 → 后 | 中位 p/m 前 → 后 |
|---|---|---|
| KV_CACHE | 15.46 → **11.17** | 0.866 → 0.920 |
| NORM | 18.31 → **16.54** | 0.879 → 0.924 |
| ELEMENTWISE | 15.49 → **14.30** | 0.844 → 0.844 |
| ACTIVATION | 7.87 → **6.65** | 0.946 → 0.965 |
| ROTARY | 11.42 → **15.47** | 1.044 → **1.113** |
| GEMM/FA/MoE 合计 | 6.25 → 6.12 | 0.986 → 0.989 |
| VECTOR ALL | 14.52 → **13.54** | — |
| A800 ALL | 8.64 → **8.26** | 0.971 → 0.979 |

方向与预期一致但幅度小于预估的 11–12%；ROTARY 退步，因为它本来就是**过**预测（1.044），追加项把它推得更过——这是拆掉抵消，不是新错误。

**为什么没有合入**：`test_a800_block1024_add_rms_one_wave_program_never_shortens` 失败，且它是**物理不变量**——行数 1→108 单调增加时预测时长不得变短。`first_request_latency_s` 随行数**非单调**（小行数下读路径延迟受限、大行数下转为带宽受限），把它作为**加项**叠到总时长上，会让预测在某些行数上反而变短。这不是过期的期望值，是我选的附着点错了：首次访存延迟在物理上是**下界**（任务不可能快于「提交 + 首次读完成」），不是可以无条件累加到完整执行时间之上的一项。

**下一步**：改为下界形式（`duration ≥ launch_submit + first_read_latency`）而不是加项，使其在执行时间已超过该下界时不再生效，单调性自然成立。gate 与参数来源不变。

## 参数来源

| 量 | 值 | 来源 | 是否新增 |
|---|---|---|---|
| gate：floor 是否空任务标定 | `isolated_empty_task_scheduler_v1` | `configs/hardware/nvidia_a800.yaml:1948`；在 `hardware_spec.py:744/752`、`formula.py:7786/7795` 上穷举分派 | 否，既有一等字段 |
| 首次请求延迟 | `first_request_latency_s` | `model_contract.py:5911` 的 `MemoryDiagnostics`，由既有内存模型导出 | 否，既有导出量 |
| C500 非空任务 floor | 1956.48 ns | `runtime_task_latency_models`，`microbench/metax_c500/references/bf16_binary_tensor_iterator_task_floor_maca_3_7_2.json` | 否，未改动 |

无新增系数，无按 MAPE 选的参数、阈值或分段点。

## 为什么 `first_request_latency_s` 是错的量，以及正确的下一步

把改动后的预测按 token 数排序检查单调性，问题暴露得很清楚——追加项**随工作量减小**：

| `add_rmsnorm` H=2048 | T1 | T2 | T4 | T8 | T16 | T64 |
|---|---|---|---|---|---|---|
| 改动前预测 (µs) | 2.884 | 2.884 | 2.884 | 2.884 | 2.884 | 2.884 |
| 改动后预测 (µs) | 3.146 | 3.115 | 3.100 | 3.092 | 3.089 | 3.086 |

全数据集共 **52 对相邻 case 出现非单调**。一个真正的"首次访存延迟"是内存系统的属性，不该随 kernel 变大而变小；它会变小说明 `first_request_latency_s` 是个**被摊薄的量**，不是首次访存的裸延迟。

因此改成下界形式（`max` 而非加法）**也救不了**：下界本身随工作量下降，而这些 case 的执行时间是平的，取 max 后仍然非单调。**不是附着点的问题，是参数选错了。**

正确的参数应当是一个**常数**冷进入/依赖访存延迟。A800 确有相关证据 `microbench/nvidia_a800/references/add_rmsnorm_h4096_grid64_cold_entry_sm80.json`（`cold_minus_warm_source_chain_entry_v1`，cold−warm = 1437 cycles ≈ 1.0 µs），但它的 `validity_domain` 钉死在单一形状（bf16、H=4096、grid 64、1024 线程、512 活跃、2 常驻 CTA、1 source round）。把它推广到整个 VECTOR 族属于外推，不是取源。

**结论**：路线 A 所需的参数在当前仓库里**不存在可直接消费的通用形式**。最干净的落地路径是给 A800 补一条与 C500 同口径的**非空任务 floor 微基准**，然后走 C500 已验证的 `runtime_task_latency_models` 路径——那条路径不需要把延迟叠加到执行时间上，而是直接给出整任务下界，单调性与 gate 都由既有结构保证。该微基准属独立机制微基准，A800 本地可自行完成。

## A800 非空任务 floor 微基准：已完成，接线被 schema 的单卡 pin 挡住

按 C500 同口径完成（`microbench/nvidia_a800/runtime/task_floor/bf16_binary_task_floor_runner.py`，reference 为 `bf16_binary_tensor_iterator_task_floor_sm80.json`）：`torch.maximum` BF16 binary TensorIterator 任务入 CUDA graph 重放，配对差分去掉 replay setup 与 event 开销，取元素扫描上的最小差分作下界。

| 项 | 值 |
|---|---|
| 三次试验 | 1884.98 / 1885.80 / 1899.32 ns（跨试验离散 **0.76%**） |
| 试验 1–2 导出 floor | **1884.98 ns** |
| 试验 3 验证 | 1899.32 ns，**0 样本低于 floor** |
| C500 同口径参照 | 1956.48 ns |
| floor 区 / 带宽区 | 1、1024 元素平坦（1894 / 1885 ns）；16384 → 2278，1048576 → 4005 |

这是**常数**，不随工作量变化，因此不会重演 `first_request_latency_s` 的非单调问题。

**接线被挡住**：`hardware_spec.py:4164` 把该 task class 的 `software_revision` 钉死为 `torch_2.4.0_metax_3.7.2.0`（C500 的栈）。该 schema 当前**每个 task class 只允许一张卡**，A800 发布同一 task class 会 fail-closed 报错（48 errors）。

选择 task class 的代码本身**不是按卡硬编码**的——`vector.py:11557` 走 `hardware.runtime_task_class_floor_supports(...)`，即"这张卡是否为该 task class 发布了非空任务 floor"这一离散事实。所以一旦 schema 允许 per-hardware 的 `software_revision`，接线就是纯配置改动，gate 形式与既有结构一致。

**下一步**：把 `hardware_spec.py` 里那条 pin 从全局常量改成按硬件校验（保持 fail-closed：仍然必须匹配该卡自己记录的栈，而不是放宽成任意字符串）。这是核心 contract 文件，需在充足上下文下单独做。预期受益面：A800 **ELEMENTWISE**（96 case、15.49%、p/m 0.844）——它正是上一轮首次访存实验里唯一没有改善的一组，因为它受 launch floor 约束而非首次访存。

## 接线完成，但实测证伪了"缺非空任务 floor"这个假说

schema 的单卡 pin 已修好（`hardware_spec.py`：该 task class 的 `software_revision` 由全局常量改为按栈的**封闭集合**，未测过的栈仍然 fail-closed），A800 的 floor 已发布进 HWSpec。三卡 kernel evaluation：

| 卡 | 逐位相同 |
|---|---|
| nvidia_a800 | **1027 / 1027** |
| metax_c500 | **1027 / 1027** |
| ascend_910b | **1027 / 1027** |

**A800 也逐位不变——floor 一个 case 都没有约束到。** 两条独立原因：

1. **适用域外**：A800 的 96 个 ELEMENTWISE case，`element_count` 最小 2048，**没有一个 ≤ 1024**（floor 区上界）。
2. **即使纳入也不会生效**：A800 最小的预测时长是 **2080 ns**，已经**高于**实测 floor **1884.98 ns**。

第 2 条是决定性的：A800 现有模型给出的下界本来就比实测的非空任务 floor 更高，所以 **"A800 缺一个非空任务 floor"这个假说被它自己的测量证伪**。ELEMENTWISE 的欠预测（p/m 0.844）不在 floor 段，而在每元素/带宽段——A800 最小 ELEMENTWISE 实测约 2464 ns（2080 / 0.844），高出预测 380 ns，但这已经是 elements ≥ 2048 的带宽区，不是 floor 区。

另需注意：探针的 task class 是 `torch.maximum`，而 production 的 ELEMENTWISE 走 vLLM 自己的算子路径，两者是否同一 task class 未经证实——这也是这条 floor 不该被强行套用的理由。

**保留的价值**：schema 的单卡限制是真实缺陷，已修复且三卡逐位不变；A800 的非空任务 floor 是一条独立、可复核的机制证据（1884.98 ns，0.76% 离散，trial 3 验证 0 样本低于），未来若有落在 floor 区的 launch 仍可消费。但它**不是** VECTOR 时延误差的来源，这条线到此为止。

**下一个该查的地方**：A800 ELEMENTWISE 的欠预测在带宽区而非 floor 区，且 C500 同族准确——应当比较两卡在 elements ≥ 2048 区间的每元素吞吐模型，而不是继续找固定开销。

## A800 ELEMENTWISE 的真实形状：斜率错，不是偏移错

按元素数分桶看 p/m（三卡 evaluation 同一口径）：

| elements | ~2048 | 16384 | 131072 | 262144 | 1048576 | 2097152 | 16777216 |
|---|---|---|---|---|---|---|---|
| **A800** | 0.830 | 0.844 | **0.773** | **0.767** | 0.917 | **1.068** | **1.105** |
| **C500** | 0.908 | 0.900 | 0.996 | 0.897 | 0.932 | 0.937 | 1.012 |

A800 随规模**从欠预测穿越到过预测**（256K 处 0.767 → 16M 处 1.105，穿越点约 1M 元素）；C500 全程平在 0.90–1.06。

**这是斜率/形状错误，不是固定偏移。** 因此本分支前两轮找固定开销的方向（首次访存延迟、非空任务 floor）**在原理上就不可能修好它**——任何常数项都只会平移整条曲线，而这条曲线需要的是改变形状。这一点由上面的分桶数据直接判定，不依赖任何 MAPE 比较。

物理读法：中等规模欠预测意味着模型算得太快——那里 wave 数不足以把带宽跑满，而模型按饱和带宽计价；超大规模过预测意味着模型算得太慢——那里已真正带宽受限，假设的有效带宽偏低。两端方向相反，正是"占用度爬坡 → 带宽饱和"曲线被用一条常数带宽近似所产生的典型形状。

**下一步**：需要 A800 的"有效带宽 vs 并发/规模"曲线（独立机制微基准，A800 本地可做，不触碰 production 算子）。C500 之所以平，很可能是它的对应参数已覆盖这条爬坡；对比两卡该参数的形式是最短路径。

## 实测 A800 的有效带宽爬坡，机制坐实

用同一个 runner 加密元素扫描（`--elements 2048 … 33554432`），按 `3 × elements × 2 B`（读 a、读 b、写 out，bf16）折算达成带宽：

| elements | per-task ns | 达成 GB/s |
|---|---|---|
| 2,048 | 2171.9 | **5.7** |
| 16,384 | 2272.3 | **43.3** |
| 131,072 | 2518.0 | **312.3** |
| 262,144 | 2743.3 | **573.3** |
| 1,048,576 | 4025.3 | 1563.0 |
| 2,097,152 | 5633.0 | 2233.8 |
| 8,388,608 | 32941.6 | 1527.9 |
| 33,554,432 | 115391.0 | **1744.7** |

两点判读：

1. **欠预测区与低达成带宽区逐点重合**。A800 在 elements 2048–262144 上 p/m 为 0.767–0.844，而同一区间达成带宽只有 **5.7–573 GB/s**，远低于实测 DRAM 峰值约 1763 GB/s。模型按接近峰值的带宽计价，自然算得太快、时间太短——这就是欠预测。
2. **2M 元素点达成 2233.8 GB/s，超过 DRAM 峰值**，因为该工作集约 12 MB，装得进 A800 的 40 MB L2；顶端 8M/33M 回落到 1528–1745 GB/s，才是真正的 DRAM 受限区。模型在顶端 p/m 1.07–1.11 过预测，对应其有效带宽设得偏低。

两端方向相反、且与达成带宽曲线逐点对应，**"用一条常数带宽近似占用度爬坡"这个机制至此坐实**，不再是推测。

这条曲线来自独立机制微基准（合成张量上的 `torch.maximum`），无任何 production 算子或 duration 参与，也没有按误差选点。

**下一步**：把这条爬坡以可消费的形式接入 A800 的 vector 时延模型（与 C500 对应参数的形式对齐），而不是继续用常数带宽。raw 在 `profiler_output/microbench/nvidia_a800/bf16_binary_task_floor/bandwidth_ramp_trial_1.json`（本地产物，未被 git 跟踪）。

## 爬坡已参数化并发布，但消费链断在两处

**已完成**：A800 的爬坡以**实测表**形式发布（`bf16_binary_tensor_iterator_task_floor_curve_sm80.json`，15 个采样点，2048→33554432 元素，floor 2125.9→115137.0 ns）。派生规则沿用 C500 自己声明的 `minimum_all_binary_graph_replay_differentials`，按元素逐点取 5 个 trial（每点 10 样本）的最小值；trial 6 验证 30 个样本中 3 个低于 floor，最差 **−0.32%**，即该尺度的 run-to-run 噪声——如实记录，未加安全裕度去凑"0 个低于"。

查找规则是 **step-down（取不超过请求规模的最大采样点）**，因实测曲线单调非减，这是合法下界且**不需要选任何插值形状或分段点**。4194304→8388608 之间的台阶是 **A800 40 MB L2 的容量跨越**（三条 bf16 流在两点分别为 25 MB 与 50 MB），属硬件事实而非按误差选的断点。

schema 已支持该曲线模型（`hardware_spec.py` 新增 `..._curve_v1` 分支，含单调性校验），A800 配置已发布，**三卡 849 tests 绿、1027/1027 逐位不变**。

**消费链断在两处，都不是参数问题**：

1. **floor 模型没有消费者**。`runtime_task_class_is_launch_scheduler()` 只对 `warm_pytorch_custom_op_scheduler_v1` 返回 True，所以 `formula.py:8116-8121` 那条唯一读取 `runtime_task_class_duration_ns()` 的分支**永远不会读到 floor 类**。这解释了为什么 C500 的常数 floor 和 A800 的新曲线**都**是 no-op——该类参数自发布以来就没有被任何路径消费。
2. **`runtime_task_class` 对 production A800 elementwise 从未被设置**。`vector.py` 的 `source_bound_binary_iterator` 谓词要求 dtype bf16、reads==2、writes==1、operation ∈ {add, mul}、grid == ceil(elements/1024)、block == (256,1,1) 同时成立；实测 grid 条件满足（elements 2048 → grid 2），但整体谓词未命中，具体是哪一项未查清。

试过把 floor 接成 `execution_time_floor_ns` 直接绕过第 1 点——三卡仍逐位不变，因为第 2 点先行阻断。该改动未保留：在不知道谓词为何拒绝的情况下留一段不触发的代码没有意义。

**边界**：爬坡本身可在零拟合前提下参数化（已做到）；阻塞在消费侧的两道门，都需要先查清 `source_bound_binary_iterator` 的失配项，再决定是放宽谓词（需源码依据）还是给 floor 类补一条消费路径。

## 撤回：「C500 在 VECTOR 上准，是因为它有非空任务 floor」

这条前提**作废**，本文件前面几节都建立在它之上，一并订正。

实际情况：`runtime_task_class_is_launch_scheduler()` 只对 `warm_pytorch_custom_op_scheduler_v1` 返回 True，而 `formula.py:8116-8121` 是唯一读取 `runtime_task_class_duration_ns()` 的调用点。因此 **floor 类模型（C500 的常数 1956.48 ns 与 A800 新发布的曲线）从发布之日起就没有被任何路径消费过**——两者都是 inert 的。

所以 C500 在 VECTOR 族准确（4.04–6.49%，中位 p/m 0.960–1.021）的**真正原因目前未知**。它不是这条 floor。此前据此推出的"A800 缺非空任务 floor 所以欠预测"整条类比链随之失效。

**不受影响的**：A800 的爬坡机制是**独立坐实**的，不依赖对 C500 的任何类比——欠预测区（elements 2048–262144，p/m 0.767–0.844）与低达成带宽区（5.7–573 GB/s，对 DRAM 峰值约 1763）**逐点重合**，且两端误差符号相反，这由 A800 自身的实测曲线与 p/m 曲线共同判定。

## 消费侧阻塞点的进一步定位

在 `vector.py` 的谓词处插桩后运行 A800 最小的 ELEMENTWISE case，**插桩从未执行**。因此失配不在 `source_bound_binary_iterator` 的六个条件（dtype / reads / writes / operation / grid / block）中的某一项上，而在**更外层的守卫**：

```python
if (not graph_padding and not runtime_markers
        and kernel.variant.lower() == "elementwise"):
```

三者之一先行拒绝：`graph_padding` 为真、`runtime_markers` 非空（即 launch 参数里已带 task-class marker，走另一条分支）、或该 KernelSpec 的 `variant` 根本不是 `"elementwise"`。这是下一轮要分辨的三选一，比"某个条件失配"精确一级。

若最终是 `runtime_markers` 非空，则说明 marker 分支本应接管，问题回到第一处阻塞（floor 类无消费者）；若是 `variant` 不匹配，则要查 KernelSpec 的构造口径。两条都不需要新采集。

## 三选一已分辨：是 variant；再往下卡在 block 几何

逐层插桩的结果（A800 最小 ELEMENTWISE case）：

**外层守卫**：`graph_padding=False`、`runtime_markers=()`、`variant='elementwise_pytorch_vec4'`。三选一落在**第三项**——守卫要求 `variant == "elementwise"`，而 production 是 `elementwise_pytorch_vec4`。**不是 marker 分支接管**，所以问题不会退回"floor 类无消费者"那一条。

**任务类同一性有源码依据**：dataset 记录的 kernel 名就是 C++ 模板实例——
`at::native::vectorized_elementwise_kernel<4, CUDAFunctor_add<c10::BFloat16>, …>` 与
`…<4, BinaryFunctor<c10::BFloat16, …>, …>`。floor 探针用的 `torch.maximum`（连续 bf16）经 TensorIterator 分派到的正是同一个 `vectorized_elementwise_kernel<4, …>` 实例。两者是同一 task class，这是模板名而非名称猜测。

**放宽 variant 之后，其余条件逐项通过，只剩 block 不符**：

| 条件 | 实测 | 谓词要求 | |
|---|---|---|---|
| dtype | bf16 | bf16 | ✓ |
| reads / writes | 2 / 1 | 2 / 1 | ✓ |
| operation | add | add ∈ {add, mul} | ✓ |
| grid | (2,1,1) | ceil(2048/1024)=(2,1,1) | ✓ |
| floor 支持该规模 | True | — | ✓ |
| **block** | **(128,1,1)** | **(256,1,1)** | **✗** |

production A800 每 block 128 线程，谓词写死 256。

**这不是可以顺手放宽的一项**：block 几何决定每 CTA 的占用度，floor 只有在**发射几何相同**时才可搬运。所以下一步不是改谓词的 256，而是先确认 **floor 探针自身实际发射的 block 是多少**——PyTorch 为 `torch.maximum` 选的 block 未必是 256。runner 目前不记录该字段，补记一次即可判定（A800 本地、独立机制微基准、不需新授权）：

- 若探针也是 128：谓词里的 256 是陈旧值，改为按实测几何匹配有据可依；
- 若探针是 256：该 floor 对 production 的 128-thread 发射**不可直接消费**，需在 128 几何下重测，否则就是跨几何外推。

**另一条必须记下的约束**：仅把 variant 放宽到 `elementwise_pytorch_vec4`（不动其他）就已使 **910B 的 5 个 OTHER case 改变**（ALL 14.04 → 14.00），而 A800 毫无变化——因为 A800 仍卡在 block。这违反 910B 逐位不变的红线且无收益，故该改动**已回退**。真正落地时，放宽必须与 floor 消费一起、并由"该卡是否发布了该 task class 的 floor"gate 住，才不会波及未发布 floor 的卡。

## 落地：爬坡接通，A800 ELEMENTWISE 15.49 → 8.54

`block` 那个二选一已由实测判定。给探针补记发射几何后（读 CUPTI 的离散发射事实，不消费任何 duration）：

| elements | 探针 grid | 探针 block | production |
|---|---|---|---|
| 2048 | (2,1,1) | **(128,1,1)** | (2,1,1) / **(128,1,1)** |
| 16384 | (16,1,1) | (128,1,1) | — |
| 1048576 | (1024,1,1) | (128,1,1) | — |

**探针与 production 发射几何完全相同**，所以 floor 可搬运，不存在跨几何外推。谓词里写死的 `(256,1,1)` 是**陈旧值**，而非物理要求。

改法没有把 256 换成另一个写死的 128，而是把**探针实测几何记进 reference 与 HWSpec**（`probe_block_threads: 128`），由 `runtime_task_class_floor_supports()` 校验请求的 block 是否等于该 floor 被测量时的几何——几何不符就不可消费。

三处改动：
1. **谓词 variant** 放宽到 `elementwise_pytorch_vec4`（依据是模板名同一性），并由"该卡是否发布了该 task class 的 floor"gate 住；
2. **block 条件**改为按已发布几何校验，替掉写死的 256；
3. **floor 消费路径**：发布了 floor 类模型的卡，以该 floor 作整任务下界（`execution_time_floor_ns`）。这一条同时修好了前面记录的"floor 类无消费者"。

结果（对当前 main）：

| 卡 | 变化 |
|---|---|
| **nvidia_a800** | **ELEMENTWISE 15.49 → 8.54%**（中位 p/m 0.844 → 0.922）；ALL 8.64 → **7.99%** |
| metax_c500 | **1027/1027 逐位不变**（其 floor 是旧的 `_v1` 常数模型，无 `probe_block_threads`，且 variant gate 不放行） |
| ascend_910b | **逐位不变**（未发布该 task class 的 floor） |

⚠️ 一处需要说明的测量陷阱：中途 910B 显示 5 个 OTHER case 变化，一度被我当成红线违反。实为**基线过期**——期间 main 前进了。对**当前 main** 重跑后，我的改动在 910B 上是 0 个 case 不同。比较前必须确认基线与被测改动同源。

其余族（A800 的 NORM 18.31、KV_CACHE 15.46、ACTIVATION 7.87、ROTARY 11.42）未受影响：它们不是这个 task class，没有对应的已发布 floor。

## 待查：C500 在 VECTOR 上准确的真正原因（floor 解释已证伪）

**先复述已证伪的那条，避免后人重走**：曾认为"C500 准是因为它发布了非空任务 floor，A800 缺这条所以欠预测"。**错**。`runtime_task_class_is_launch_scheduler()` 只对 `warm_pytorch_custom_op_scheduler_v1` 返回 True，而 `formula.py` 是唯一读取 `runtime_task_class_duration_ns()` 的调用点，所以在本轮补上消费路径**之前**，floor 类模型（含 C500 的 1956.48 ns）从未被任何路径消费过——它是 inert 的，不可能是 C500 准的原因。

（本轮已为 A800 补上消费路径并生效；C500 的条目仍是旧的 `_v1` 常数模型、且 variant gate 不放行，故 C500 至今仍未消费它。）

**新发现：三卡的"实测"根本不是同一个物理量。**

| 卡 | `actual_source` | 粒度 |
|---|---|---|
| nvidia_a800 | `nsys_nvtx_kernel_median` | **kernel**（2224）+ op（106） |
| metax_c500 | `device_event_op_median` | op |
| ascend_910b | `msprof_timeline_op_median` | op |

三套仪器、两种粒度：nsys 的 kernel span、device event 围出的 op 区间、msprof timeline 的 op 区间，各自包含/排除的launch、排空、算子内多 kernel 间隙都不同。本仓已有同形先例（`device_execution span 不是 device busy`：同一字段名在三卡上有三种含义，跨硬件比 device 误差无意义）。

**因此"C500 5.66–6.49% vs A800 14.52%"这句话本身不是一个纯建模结论**——两者的分母不是同一个量。在把口径对齐之前，不应假设存在一个"C500 有而 A800 没有"的建模机制；差距里有多少来自仪器语义、多少来自模型，目前未分离。

**要判定它需要**：对同一张卡、同一批 shape，用两种粒度各取一次实测（例如 A800 同时取 nsys kernel span 与 device-event op span），量出同一预测相对两种口径的差。这不涉及 production duration 拟合，属口径核对；A800 本地可做。在此之前，把 A800/910B 的欠预测当成"缺某个 C500 已有的机制"来追，有走错方向的风险——本轮 floor 那条线就是这样被证伪的。

## 口径核对的结果：C500 的优势约三分之二是分母，不是机制

A800 本地实测，同一批 launch 同时取两种粒度（CUPTI kernel duration 与围住 op 的 CUDA event 区间；后者取背靠背摊薄形式，因为 capture 的 op median 是排队执行的，不该每次付冷提交）：

| elements | kernel span | op span (b2b) | 差 | 比 |
|---|---|---|---|---|
| 2,048 | 1920 | 6799 | **4879 ns** | 3.54 |
| 16,384 | 2432 | 6554 | **4122** | 2.70 |
| 262,144 | 3072 | 6426 | **3354** | 2.09 |
| 4,194,304 | 8608 | 9581 | 973 | 1.11 |
| 16,777,216 | 60479 | 61188 | 709 | 1.01 |

op 区间比 kernel span 多出 **0.7–4.9 µs** 的发射/算子间隙。而 A800 的 dataset 用 kernel span、C500 用 op span——**同样的绝对误差，除以 op span 得到的百分比小得多**。

把 A800 自己的误差换成 op-span 分母（分子不变，只换分母）：

| 族 | MAPE（kernel span） | MAPE（op-span 分母） | C500 实际 |
|---|---|---|---|
| ELEMENTWISE | 8.54 | **4.39** | 5.99 |
| ACTIVATION | 7.87 | **6.32** | 4.04 |
| KV_CACHE | 15.46 | **7.39** | 6.41 |
| ROTARY | 11.42 | **8.20** | 14.79 |
| **NORM** | 18.31 | **12.35** | **5.66** |
| **VECTOR ALL** | **12.27** | **7.91** | **6.49** |

A800 与 C500 的差距从 **1.89×（12.27 vs 6.49）收到 1.22×（7.91 vs 6.49）**。**约三分之二是仪器口径，不是建模差距。** ELEMENTWISE 与 ROTARY 换算后反而优于 C500，KV_CACHE 基本追平（7.39 vs 6.41）。

**所以"C500 有一个 A800 缺的机制"这个框架基本不成立**，剩下的真实差距集中在**一个桶**：

> **NORM：12.35 vs 5.66，仍差一倍以上。这是 A800 唯一值得继续追机制的 VECTOR 桶。**

**这条换算的局限必须一并记住**：偏移量测自 torch add/mul，套用到 NORM/ROTARY/KV_CACHE 时它们的 kernel 结构不同、间隙未必相同；且换算假设模型能精确预测那段间隙。因此它是"分母能解释多少"的**上界**，方向可靠、数值近似。它**没有也不会**改动 evaluation 口径，只用于判断该往哪儿investigate。

**下一步据此确定**：不再找"C500 有而 A800 没有的通用机制"，改为直接查 A800 NORM 这一个桶。KV_CACHE（换算后 7.39 vs 6.41）不值得单独投入。

## A800 NORM 的机制：单波内 CTA 数不计价

按 variant 拆开，A800 NORM 的误差几乎全在一侧：

| variant | n | MAPE | 中位 p/m |
|---|---|---|---|
| **`add_rmsnorm`** | 48 | **26.14** | **0.754** |
| `rmsnorm` | 48 | 10.48 | 0.950 |

`add_rmsnorm` 在 H=2048 上的形状很清楚——**预测在 T=1…64 上恒为 2.88 µs，实测却从 3.30 升到 4.13 µs**；到 T=256 预测 5.28 对实测 5.47（准），T=1024 预测 10.85 对 10.34（略过）。

发射几何解释了这个平台：`grid = T`、`block = 1024`，所以 T=1…64 是 **1 到 64 个 CTA 落在 108 个 SM 上，恰好一个波**。模型把"一个波"视为时长与 CTA 数无关——A800 的 launch 模型按 `cta_round_ns` 每 **ceil(grid/108) 个满片轮次**计价，波内增加 CTA 的代价是 **0**。

**实测说波内不是常数。** 同一张卡上已测的 floor 曲线（`bf16_binary_tensor_iterator_task_floor_curve_sm80.json`，grid = ceil(elements/1024)）在单波区间内单调上升：

| grid（CTA 数） | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 实测 ns | 2125.9 | 2241.5 | 2242.6 | 2267.1 | 2326.5 | **2379.3** |

**单波内 2→64 个 CTA，时长上升 12%。** 这也解释了上一轮为什么那条 floor 曲线能把 ELEMENTWISE 从 15.49 压到 8.54——它恰好把这段波内上升吃进去了；而 NORM 没有对应的已发布曲线，仍然吃 0 代价的平台。

这同时解释了 `rmsnorm`（0.950）为何比 `add_rmsnorm`（0.754）好：后者每行多读一条 residual、多写一次，波内争用更重，被低估得更多。

**下一步的边界要先说清楚**：直接测 production 的 `add_rmsnorm` kernel 取时长，就是用 production 算子 duration 做参数，红线禁止。可行的是**测通用机制**——单波内时长随 CTA 数的上升，用合成的、触碰内存的 CTA 在 grid 1…108 上扫一遍，得到与算子无关的波内 CTA 代价，再按 HWSpec gate 进 launch 模型。那是独立机制微基准，A800 本地可做，且同时覆盖 NORM 与其他吃同一平台的族，而不是给 NORM 单独配一条曲线。

## 波内 CTA 代价：实测存在且连续，不是波量化的

合成探针（`torch.maximum`，每 CTA 恒定 1024 元素，grid 1…216），用 graph-replay 配对差分去掉发射开销：

| grid | 1 | 2 | 4 | 16 | 32 | 64 | **108** | 128 | 216 |
|---|---|---|---|---|---|---|---|---|---|
| per-task ns | 1905.7 | 2114.6 | 2238.1 | 2278.4 | 2337.1 | 2399.9 | **2521.8** | 2512.2 | 2689.4 |
| 相对 grid 1 | — | +11.0% | +17.4% | +19.6% | +22.6% | +25.9% | **+32.3%** | +31.8% | +41.1% |

**每 CTA 工作量恒定，单波内（1→108）时长上升 32.3%。** 这正是 A800 launch 模型计价为 0 的那部分。总流量在 grid 108 处约 650 KB，远不足以带宽受限，所以量到的是发射与争用，不是数据搬运。

**两条形状事实**：

1. **上升是连续的，不是按波量化的**。grid 128（两个波）为 2512.2 ns，反而略低于 grid 108 的 2521.8。所以"再加一轮 `cta_round_ns`"不是正确的修法——真实代价随 CTA 数连续增长，而非每满 108 个跳一级。
2. 前段陡、后段缓（1→4 已吃掉 +17.4%，4→108 再加 ~15%），是饱和形状。

**一处探针设计教训**（避免后人重走）：先写的版本用背靠背 launch 直接测 kernel span，结果 grid 1→108 恒为 ~4750 ns、看不出任何上升，且跨 trial 离散 155%。原因是每次 launch 的间隙（~4.7 µs）远大于要测的效应并且恒定，把信号完全淹没。**必须用 graph-replay 配对差分**才能把发射开销消掉。该错误探针已删除，未留在仓库里。

## 落地前必须先解决的一处重叠

这条波内代价若直接加进 A800 的 launch 模型，会与已发布的 bf16 binary task floor 曲线**重叠计价**：那条曲线是同一支探针、同一个 op 测出来的，其随 elements 的上升**本身就包含**这段波内 CTA 代价（grid = ceil(elements/1024)）。ELEMENTWISE 已经通过 floor 吃到了它，再从 launch 模型加一次就是双记。

所以正确形态不是"两条都加"，而是二选一：
- 要么把波内 CTA 代价做成 launch 模型的通用项，**同时**把 bf16 binary 的 floor 曲线降级为只覆盖 grid=1 的单点（其余由通用项承担）；
- 要么维持 floor 曲线的形态，为 NORM 所属的 task class 另行发布一条同形曲线——但那需要一支**不触碰 production add_rmsnorm** 的探针来产生，而合成探针无法复现 add_rmsnorm 的每行 residual 读写结构。

前者覆盖面更广且不需要新探针，但要动 launch 模型这一三卡共用结构，必须 gate 到 C500/910B 逐位不变。这是下一轮要定的设计选择，尚未动手。

## A 落地的实测结果：ELEMENTWISE 未退步，但项的适用域超了

按 A 实施并实测：把波内 CTA 代价发布进 A800 HWSpec（`within_wave_cta_cost`，grid→ns 表，grid 1 为 0），在 launch 侧按 grid 查表串行计入，同时把 bf16 binary floor 曲线**降级为单 CTA 常数**（全部条目改为 1905.73 ns，不再随 elements 变化）。

**你要求同时验的那条：ELEMENTWISE 没有退步。**

| A800 族 | 改前 | 改后 |
|---|---|---|
| **ELEMENTWISE** | 8.54 | **8.31** |
| KV_CACHE | 15.46 | **10.18** |
| ACTIVATION | 7.87 | **6.57** |
| NORM | 18.31 | **15.64** |
| ROTARY | 11.42 | **14.41** |
| OTHER（GEMM/FA/MoE） | 6.25 | 6.07 |
| VECTOR 合计 | 12.27 | **11.12** |

| 卡 | 逐位相同 |
|---|---|
| metax_c500 | **1027 / 1027** |
| ascend_910b | **1027 / 1027** |

ELEMENTWISE 从 8.54 到 8.31 略有改善——**说明波内 CTA 代价项完整覆盖了原 floor 曲线所做的工作，A 是纯替换而非丢失**。C500 与 910B 逐位不变（两卡均未发布该表）。

**但这版没有合入，因为项的适用域超了。** 5 个 GEMM/MoE 契约测试失败，差值**恰好是 783.63 ns，即 grid 216 的表值**——说明该项也被加到了 GEMM 与 MoE 的 launch 上。探针是在 128 线程、每 CTA 1024 元素、触碰内存的 CTA 上测的；对一个每 CTA 运行 140 µs 的 GEMM，发射是与执行重叠的，不落在关键路径上，而对 2 µs 的 vector CTA 发射才是主导。**把 vector 形状的 CTA 发射代价无条件套到所有族，是超出探针有效域的外推。**

`OTHER` 从 6.25 改善到 6.07 **不能作为接受它的理由**——那正是"按 MAPE 接受一个物理外推"，本仓已多次记录为错误做法。因此本版**已回退**，`src/` 与 `configs/` 均还原，3926 passed。

**下一步需要的是一个有依据的有效域判据**，把该项限制在发射代价确实不被执行掩盖的 launch 上。可用的离散事实方向：CTA 的驻留时长相对发射间隔（需要一支测"发射速率 vs CTA 时长"的探针），或按 CTA 的工作量下界判定。**不得**用族名或按误差挑阈值。表本身（`within_wave_cta_cost_sm80.json`）是独立机制证据，已保留可复用。

## CUDA 探针的结论：判据不需要了，因为超域的不是"适用范围"而是"那张表混了两样东西"

写了一支能独立控制 grid / block / dynamic smem / 每 CTA 时长的 CUDA 探针（`microbench/nvidia_a800/runtime/cta_dispatch/cta_dispatch_cost.cu`，每 CTA 工作量是一条依赖 FMA 链、时长与 block 和 grid 无关；CUDA graph 配对差分去发射开销）。结果推翻了我自己上一轮的"足迹"假说。

**单波内的纯发射代价与 block、smem 都无关**（grid 108，chain 64）：

| block | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| 108 个 CTA 的绝对代价 | 230.4 ns | 220.8 | 219.2 | **238.4** |

16 KiB 动态 smem 对它的影响是 **0.984–1.001 倍**，即没有影响。

**它也不被执行掩盖**——它是**可加常数**，只是相对长 kernel 可忽略：

| 每 CTA 时长 | ~2 µs | ~5–16 µs | ~65–232 µs | ~1–3.7 ms |
|---|---|---|---|---|
| 绝对代价 | 220–238 ns | 226–266 | 253–499 | 877–6083 |
| 相对上升 | 10–15% | 1.5–5% | 0.2–0.5% | 0.1–0.2% |

纯发射代价随 grid 的曲线（波内 block 无关）：grid 1→0、4→54、16→46、32→79、64→150、**108→226 ns**，约 2.1 ns/CTA。grid 216 时才出现 block 依赖（300→1322 ns），那是每 SM 驻留数不同导致波数不同，**既有的 `cta_round_ns` 已经在建模这一段**。

## 由此重新判定上一版为什么超域

上一版用的是 **torch 探针导出的表**，它在 grid 108 处给 **616 ns**，而纯发射代价只有 **226 ns**——**2.7 倍**。差额来自 torch 探针把 `elements = grid × 1024` 一起放大，工作集随之增长，那部分是**内存系统争用**，不是发射代价。

所以上一版真正的错误不是"把发射代价用到了不该用的族"，而是**那张表里混进了一个随工作集增长的量，再被无条件套到工作集完全不同的 GEMM/MoE 上**。5 个契约测试的 783.63 ns 差值里，只有约 226 ns 是发射，其余是从 elementwise 形状外推过去的争用。

**因此"有效域判据"这条需求消解了**：纯发射代价对 block、smem 均无依赖，是**普适的**，可以合法地加到所有族；需要限制的不是它，而是那张混了争用的 torch 表——后者根本不该进 launch 模型，争用属于内存模型。

## 下一步（形态已确定，未动手）

1. 用 CUDA 探针的**纯发射代价表**（block/smem 无关，波内 ~2.1 ns/CTA）替换 torch 导出的 `within_wave_cta_cost`，作为 launch 模型的普适项；gate 仍按"卡是否发布该表"，C500/910B 无表即逐位不变。
2. bf16 binary floor 曲线**保持不动**——它含的争用部分对 ELEMENTWISE 是对的，且由 `probe_block_threads` 限定在测量几何内。
3. NORM（block 1024）由普适发射项覆盖，不再需要单独曲线。
4. 5 个 GEMM/MoE 契约测试届时会变化约 226 ns 而非 783.63 ns，需按物理复核其期望值而非直接改数。

`within_wave_cta_cost_sm80.json` 保留但**必须标注它混有争用**，不得当作纯发射代价引用。

## 纯发射项已接入：A800 ALL 7.99 → 7.60，C500/910B 逐位不变

`cta_dispatch_cost` 作为 HWSpec 的一等字段发布，launch 侧按 grid 查表、串行计入（CTA 未被发射前无法开始）。gate 是"该卡是否发布该表"这一离散事实——C500 与 910B 无表，`cta_dispatch_cost_ns()` 返回 0。

| 卡 | 逐位相同 |
|---|---|
| metax_c500 | **1027 / 1027** |
| ascend_910b | **1027 / 1027** |

| A800 族 | 改前 | 改后 |
|---|---|---|
| **ELEMENTWISE** | 8.54 | **5.67**（已优于 C500 的 5.99） |
| KV_CACHE | 15.46 | 14.22 |
| **NORM** | 18.31 | **17.61** |
| ACTIVATION | 7.87 | 7.67 |
| ROTARY | 11.42 | 11.94 |
| OTHER（GEMM/FA/MoE） | 6.25 | 6.20 |
| **ALL** | **7.99** | **7.60** |

参数表（trials 1–2 派生、trial 3 验证，跨试验离散 26.4 ns）：grid 1→0、4→46.8、16→54.0、32→75.6、64→157.2、**108→222.4 ns**。表止于满片宽度，因为超过一个波后代价转为驻留相关，那段已由 `cta_round_ns` 建模，更大的 grid 由 step-down 规则复用末项。

## NORM 只降 0.7pp——这条要如实说

预期 NORM 由普适发射项覆盖，实测只从 18.31 降到 17.61。原因清楚：纯发射代价满片才 222 ns，而 NORM 在 H=2048、T=1…64 平台段的缺口是 0.4–1.2 µs，**量级差一个数量级**。

上一版那张被污染的 torch 表把 NORM 降到 15.64，降得更多——**正因为它混进了争用**。所以 NORM 平台段的缺口主要不是发射，而是**波内 CTA 增多带来的内存争用**，那属于内存模型而非 launch 模型。**NORM 仍未解决，且方向已改：不要再往 launch 侧找。**

## 契约测试按物理复核，不是改数迁就

5 个 GEMM/MoE/floor 契约测试的差值**逐个等于该 launch 自己的 `cta_dispatch_cost_ns(grid)`**（grid≥108 为 222.4 ns、grid 4 为 46.8 ns），即该项按设计合成、未越界。期望值因此改写为**引用访问器**而非新常数。

另有一组 launch 语义契约需要更新，它们编码的是"A800 vector launch 发射时间为 0"——那只在 A800 尚无实测发射代价时成立。判据改为"该卡是否发布了发射代价表"这一离散事实。其中 `test_independent_additive_dispatch_is_not_a_second_floor` 证实了合成方式：**发射项对执行是可加的，不是第二个 floor**——空需求时总时长仍由 empty-task floor 决定，只有 `t_launch_s` 变化。

## 波内内存争用：已实测，**不是** NORM 平台段的原因

新增探针 `microbench/nvidia_a800/memory/wave_contention/wave_contention.cu`：每 CTA 流固定字节（4/8/16/32 KiB，覆盖 hidden 2048–16384 的 bf16 行宽），grid 1→108，block 128/256/1024，全程工作集在 40 MiB L2 内（与 NORM 自身所处区间一致），CUDA graph 配对差分计时。**争用 = t(grid) − t(1) − 已实测发射代价**。

block 1024（NORM 的几何）满片争用：

| KiB/CTA | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| 争用 @108 CTA | 129.6 ns | 128.0 | 441.6 | **1100.8** |

与 NORM 实际残差逐点比（`add_rmsnorm`、H=2048、发射项已在模型内）：

| T | 1 | 2 | **4** | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| 模型缺口 | 412 ns | 380 | **909** | 870 | 902 | 1087 |
| 实测争用 | 0 | 10 | **78** | 80 | 85 | 212 |

**量级差 4–10 倍，且形状不符**：缺口在 T=2→4 之间**跳变**（380→909 ns）之后basically持平，而争用是平滑增长的。**波内内存争用被排除。**

## 由此 NORM 的方向再次改变：不是 launch 侧，也不是波内争用

两条 launch 侧候选现在都已实测并排除：

| 候选 | 满片实测值 | NORM 所需 | 结论 |
|---|---|---|---|
| CTA 发射代价 | 222.4 ns | ~0.4–1.1 µs | 量级差一个数量级 → 已接入但不解释 NORM |
| 波内内存争用 | 130–442 ns（NORM 行宽处） | 同上 | 量级差 4–10 倍且形状不符 → 排除 |

指向别处的两条证据：

1. **T=2→4 的跳变**。grid = T，T=4 时并无波、占用或容量边界被跨过；发射与争用都不产生这种台阶。
2. **两个 variant 在 T=1 就差约 600 ns**：`rmsnorm` 被**过**预测 188 ns，`add_rmsnorm` 被**欠**预测 412 ns。T=1 只有一个 CTA，谈不上任何波内效应——这是**单 CTA 内的工作量定价**问题，即 `add_rmsnorm` 多出的 residual 读写这一条流被算少了。

所以 NORM 的主项应在 **per-CTA 工作量/流数的定价**上找，而不是 launch 侧或波内并发。下一步的取证对象是 add_rmsnorm 相对 rmsnorm 多出的那条 residual 流在模型里如何计入（源码静态语义可判，不需上机），以及 T=4 台阶对应源码里哪个分支。

两支探针都保留为独立机制证据：发射代价已入模型，争用表**未入模型**——它是真实的，但不是 NORM 的主项，若将来为别的族使用需另行论证。

## NORM 定位：residual 流没有漏计，错的是两个 variant 的时间定价

查 `add_rmsnorm` 相对 `rmsnorm` 多出的 residual 流如何计入（H=2048、T=1）：

| case | 模型 L2 | 模型 DRAM |
|---|---|---|
| `rmsnorm` | 12288 B = **3 条流**（x 读、y 写、gamma 读） | 4096 |
| `add_rmsnorm` | 20480 B = **5 条流**（多 residual 读 + residual 写） | 4096 |
| `add_rmsnorm` T=2 / T=4 | 40960 / 81920（随 T 线性） | **4096（恒定）** |

**residual 流没有被漏计**：两者之差恰为 8192 B = 2 × 行宽，即 residual 的读与写都在账上。

DRAM 恒为 4096 B（正好一条 gamma 行宽）且**不随 T 变化**，说明模型已把 x / residual / y 当作 L2 驻留的原地写回，只有 gamma 落 DRAM——这与 910B RoPE 那条"原地写回 L2-resident 不排到 DRAM"的结构是同一形态。A800 Vector 的 HBM 精度（2.47%）也支持这个流量判断大体成立。

**所以缺口不在流量，在时间定价**，而且两个 variant 错的方向相反（H=2048、T=1）：

| variant | 预测 | 实测 | 偏差 |
|---|---|---|---|
| `rmsnorm` | 3.77 µs | 3.58 µs | **过** 188 ns |
| `add_rmsnorm` | 2.88 µs | 3.30 µs | **欠** 412 ns |

注意一处反直觉：**`add_rmsnorm` 流量更大（20480 vs 12288）却被预测得更快（2.88 vs 3.77）**。实测两者的相对次序是一致的（3.30 < 3.58），所以次序不算错，但两条曲线各自偏了约 200–400 ns 且方向相反——单 CTA、无任何波内效应下就已如此。

**下一轮的取证对象**（源码静态语义可判，不需上机）：

1. 两个 variant 在 `vector.py` 里各自走哪条 pipeline/service 组装，为什么流量多的那个反而被定价得更快——这处不对称是主嫌。
2. T=2→4 的台阶对应源码里哪个分支（此前已排除发射与波内争用，两者都不产生台阶）。

**不得**用这 188/412 ns 去反推任何系数——它们是待解释的观测量，不是参数。

## 找到不对称的来源：两个 variant 产生的是**完全不同的 compute demand**

同为 pipeline-bound、同一组 attach 函数，但服务组装分叉——`rmsnorm` 走 `_has_typed_rmsnorm_scalar_round_services`，`add_rmsnorm` 走 `_add_rmsnorm_active_unit_service_weights` 与 `_add_rmsnorm_additional_round_widths`。落到 compute demand 上（H=2048、T=1）：

| variant | compute demands |
|---|---|
| `rmsnorm` | `compute_reduction` 1.0 + **`compute_fma` 8192.0** |
| `add_rmsnorm` | `compute_reduction` 1.0 + `compute_vector` 2.0 + **`warp_issue` 176.0** |

**8192 = 4 × H**，随 hidden 线性；而 `add_rmsnorm` 只记 176 次 warp issue 加 2 个 vector op，**既不随 H 线性、量级也小两个数量级**。

**这就是"流量更大却被预测更快"的直接原因**：两条路径用的是不同的计价物——一条按 FMA 计数，一条按 warp 发射计数——而不是同一物理量的两种取值。

## 由此收窄的问题

问题不再是"某个系数偏了 188/412 ns"，而是：**两个同族、同结构、仅差一条 residual 流的算子，为什么被安排在两套互不可比的 compute 计价路径上？**至少有一条是错的：

- 若 `rmsnorm` 的 4×H FMA 计数正确，则 `add_rmsnorm` 漏掉了自己那份归一化 FMA（它同样要做平方和与缩放），只记了 warp issue；
- 若 `add_rmsnorm` 的 warp-issue 计价正确（即该 kernel 实际受发射限制而非 FMA 限制），则 `rmsnorm` 的 8192 FMA 把它算成了计算受限，才会被**过**预测 188 ns。

两种可能都与实测方向自洽：`add_rmsnorm` 欠 412 ns（少记了计算）、`rmsnorm` 过 188 ns（多记了计算）。**但方向自洽不能作为判据**——必须回到两个 kernel 的源码，看各自真正执行什么、受什么限制，再决定哪一条计价物是对的。

下一轮的取证对象因此明确为：vLLM/CUDA 侧 `rms_norm` 与 `fused_add_rms_norm` 两个 kernel 的实际指令构成（源码静态语义，不需上机），以及 `vector.py` 中这两条服务组装各自的来源注释与依据。T=2→4 的台阶一并在那里查——它很可能同样出自这两条路径之一的分支条件，而非任何波内效应。

**不得**用 188/412 ns 反推任何系数；它们是用来验证解释的，不是用来生成参数的。

## 源码判定：`add_rmsnorm` 漏的是自己那份 H-线性归一化算术

`vllm/csrc/libtorch_stable/layernorm_kernels.cu` 两个 kernel 结构相同，都是两遍：

| 阶段 | `rms_norm_kernel` | `fused_add_rms_norm_kernel` |
|---|---|---|
| 第一遍（逐元素） | `variance += x * x` | `temp += residual_v[id]`；`variance += temp.sum_squares()`；`residual_v[id] = temp` |
| 归约 | `cub::BlockReduce` | 同 |
| 标量 | `rsqrtf(variance / hidden + eps)` | 同 |
| 第二遍（逐元素） | 读回、`x * s_variance * w`、写出 | 读回 residual、`x * s_variance * w`、写出 |

**`fused_add_rms_norm` 每元素做的算术严格多于 `rms_norm`**：它同样要平方累加与缩放，额外还有 residual 的加法与回写。两遍都随 `hidden_size` 线性。

因此前一节列出的两种可能里，**第二种被源码排除**：`add_rmsnorm` 不可能是"实际受发射限制、没有 H-线性算术"——它的算术比 `rmsnorm` 还多。

**判定：模型给 `add_rmsnorm` 记的 `warp_issue` 176 + `compute_vector` 2（均不随 H 变）漏掉了它 H-线性的归一化算术**，而 `rmsnorm` 的 `compute_fma = 4 × H` 至少在量纲上是对的。这与实测方向一致（`add_rmsnorm` 欠 412 ns），但结论来自源码结构，不是由该偏差反推。

`rmsnorm` 被**过**预测 188 ns 是另一回事，需单独判断 `4 × H` 这个系数与实际指令数是否相符——源码里第一遍每元素 1 次乘加、第二遍 1 次转换加 1–2 次乘，逐元素总数与 4 的关系要对着 `VEC_SIZE` 与 `_f16Vec` 的展开数清点，不在本轮范围。

**下一轮的实现方向**（仍须先对完指令数，不得按 188/412 ns 取值）：让 `add_rmsnorm` 与 `rmsnorm` 走同一套逐元素算术计价，再各自叠加自己特有的部分（residual 的加法与回写）。改动落在 `vector.py` 的服务组装，需 gate 到 C500 与 910B 逐位不变；契约测试期望值按指令数复核。

T=2→4 的台阶仍未解释，留待同一处排查。

## 逐元素指令清点：`rmsnorm` 的 4×H 对得上，`add_rmsnorm` 应为其上再加 residual 的两项

按 `layernorm_kernels.cu` 与 `type_convert.cuh` 逐元素清点（`HasWeight` 为真，即生产路径）：

| 阶段 | `rms_norm` 每元素 | `fused_add_rms_norm` 每元素 |
|---|---|---|
| 第一遍 | `x * x` 累加 = 1 乘 + 1 加 | `temp += residual` = **1 加**；`sum_squares()` 里 `z.x*z.x + z.y*z.y` 累加 = 1 乘 + 1 加 |
| 第一遍写回 | — | `residual_v[id] = temp`（**回写一条流**，已在 L2 账上） |
| 第二遍 | `x * s_variance * w` = 2 乘 | 同 = 2 乘 |
| **合计（算术）** | **4** | **5** |

**`rmsnorm` 的 `compute_fma = 4 × H` 与清点结果一致**——这解释了为什么它量纲正确；其 188 ns 的过预测不来自这个计数，需另找（候选是两遍之间 `cub::BlockReduce` 与 `__syncthreads` 的串行段被计价过重，属 pipeline 组装而非逐元素项）。

**`add_rmsnorm` 应为 `5 × H`**（多出的 1 是 residual 加法），外加已在流量账上的 residual 回写。当前模型给它的是 `warp_issue 176 + compute_vector 2`，**与 H 无关**，这正是漏项所在。

注意 5/4 = 1.25 只是算术条数比，**不可直接当作时间比使用**——两遍之间的归约与同步是串行段，且第二遍多读一条 residual 流。落地时必须按同一套逐元素计价重算两者，而不是给 `add_rmsnorm` 乘一个系数。

## 下一轮实现清单（不得按 188/412 ns 取值）

1. 让 `add_rmsnorm` 与 `rmsnorm` 共用同一条逐元素算术计价路径，前者按上表多计 1 次加法/元素；
2. residual 回写已在 L2 流量账上（5 条流 vs 3 条流，前节已验证），计价时不要重复计入；
3. gate 到 C500 与 910B 逐位不变——两卡各自的 NORM 路径不得进入新分支；
4. 契约测试期望值按上表的指令数复核；
5. T=2→4 的台阶：清点里没有任何随 T 变化的分支，**说明台阶不在逐元素算术上**，应查服务组装里按 grid/token 数选择的那部分（`_add_rmsnorm_additional_round_widths` 是首选嫌疑，名字即"额外的 round 宽度"）。

## 统一落地：逐元素算术改为两个 variant 共用一套计价（数值不变）

`vector.py` 的通用 NORM 路径此前用两个裸浮点 `arithmetic_per_element = 4.0 / 5.0`，相分解又在别处硬写 `2.0 * elements`。现在改为一套带来源注释的具名计价，两个 variant 共用：

```
_NORM_SQUARE_ACCUMULATE_OPERATIONS_PER_ELEMENT = 2.0     # 两个 variant 相同
_NORM_OUTPUT_MULTIPLY_OPERATIONS_PER_ELEMENT   = 2.0     # 两个 variant 相同
_NORM_RESIDUAL_ADD_OPERATIONS_PER_ELEMENT      = {"rmsnorm": 0.0, "add_rmsnorm": 1.0}
```

`add_rmsnorm` 不再是"另一个系数"，而是同一套计价**加上它自己那 1 次 residual 加法**；相分解的三项改为由同一组常数导出，并加了 fail-closed 断言：三项之和必须等于 `arithmetic_per_element × elements`，否则抛错。residual 回写仍只在字节账上，未重复计价。

**数值逐位不变**（4 = 0+2+2、5 = 1+2+2 与原值相同），因此三卡与 PIM 全部不动。契约测试 `tests/kernel/test_vector_norm_arithmetic_tally.py` 的期望值直接写成源码指令数表，并单独校验"除 residual 加法外两个 variant 每一项都相等"。

## 但前提被自己的数据推翻：逐元素算术不是 A800 NORM 的承重项

落地后必须报告一件事：**"add_rmsnorm 缺了 hidden-linear 的算术"这个前提，被两条独立证据推翻。**

### 证据一：算术只占 `rmsnorm` 预测时间的 1.5–3.6%

A800 `rmsnorm` 走通用路径，确实记了 `4 × H` 的 FMA。但它在预测里占的份额是：

| H | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| `fma_pipe` | 56.1 ns | 112.3 | 224.6 | 449.1 |
| 预测总时长 | 3771.7 | 4997.8 | 7450.0 | 12354.4 |
| 占比 | 1.5% | 2.2% | 3.0% | 3.6% |

承重项是 `memory:global_read`（H=2048 时 1170 ns、H=16384 时 9360 ns）与 `block_reduction_dependency`（864 ns 恒定）。**把 `add_rmsnorm` 的算术从 0 提到 `5 × H`，在 H=2048 只值约 70 ns，而缺口是 412 ns。** 逐元素算术在量级上就无法承担 17.61%。

### 证据二：真正拿到 `5 × H` 算术的那个 case，是最差的欠预测

`_has_typed_add_rms_round_services` 的精确宽度域是 H ∈ {2048, 2560, 4096, 5120, 8192}。**H=16384 落在域外，走通用路径，因此它本来就在按 `5 × H` 计价**——它是全部 `add_rmsnorm` T=1 里最差的一个：

| H（T=1） | 2048 | 2560 | 4096 | 5120 | 8192 | **16384（通用 5×H）** |
|---|---|---|---|---|---|---|
| 预测/实测 | 0.875 | 0.827 | 0.809 | 0.812 | 0.837 | **0.654** |

拿到完整 hidden-linear 算术的那一个不是更准，而是更差。所以"补上算术即可"在算术上和实测上都不成立。

### 为什么 typed 路径本来就不该再加算术

typed source-round 服务（`bf16_add_square_first_*_block1024` 等）是**对生产 kernel 同一段源码循环的独立 clock64 微基准**（`production_function_sass_sha256` 与生产函数一致），它测的是**整轮**——包含该轮的算术指令本身。在它之上再记 `5 × H` 次 FMA，等于把探针已经测过的指令再计一遍。`typed_norm_rounds` 把通用算术置零是正确的，本轮保留。

**结论（合法退出条款）**：统一到一套逐元素算术计价**已经落地且可被契约测试锁住**，但它**不能**改善 A800 NORM——不是因为做不到，而是因为这一项在物理上不承重。继续往这条线加系数只能靠 MAPE 反推，触红线。

## T=2→4 台阶：首选嫌疑 `_add_rmsnorm_additional_round_widths` 被排除

按 grid 逐点列 A800 Vector 全族实测（同一 launch 几何下的相邻 grid 差）：

| op | block | 最大单步 | 位置 |
|---|---|---|---|
| `rmsnorm` | 1024 | +480 … +640 ns | **g2→g4** |
| `add_rmsnorm`（H 2048–8192） | 1024 | +544 … +576 ns | **g2→g4** |
| `add_rmsnorm`（H 16384） | 1024 | +544 ns | g1→g2 |
| `rope` | 512 | +480 … +640 ns | g1→g2 |
| `kvcache` | 512 | +512 … +544 ns | g1→g2 |
| `silu` | 1024 | +384 … +512 ns | g1→g2 |
| `elementwise` | 128 | **+64 ns**（无台阶） | — |

台阶**同样出现在 `rmsnorm`**，而 `rmsnorm` 根本不调用 `_add_rmsnorm_additional_round_widths`（它走通用路径，`typed_rmsnorm_scalar_rounds` 在 A800 恒为 False）；也同样出现在 `rope`／`kvcache`／`silu` 这三条完全不同的服务组装上。**因此台阶不在 NORM 的 variant 服务组装里，首选嫌疑排除。**

台阶的形状：一次性、约 480–640 ns、**与 H 无关**（H 从 2048 到 16384 台阶幅度不变）、与 variant 无关，跨过之后到 g16 基本持平。这三条性质同时排除：

- **CTA 发射代价**：实测表 g4 = 46.8 ns、满片 108 才 222.4 ns，量级差一个数量级（已入模型）；
- **波内内存争用**：实测 g4 = 78 ns，且随 grid 平滑增长而非一次性跳变（已排除）；
- **时钟/DVFS**：若是降频则应是乘性的，但 H=2048 台阶 +480 对应 ×1.134、H=16384 台阶 +640 对应 ×1.059——**加性恒定而非乘性恒定**。

`elementwise`（block 128）全程无台阶，但它最小的实测 grid 已经是 2，无法区分"没有台阶"与"台阶在 g1→g2 已经发生"。

**这是一个跨族的、可复现的、尚无机制解释的离散事实。** 它不属于 NORM 的 variant 定价，本轮不入模型——没有独立机制微基准之前写进去就是按误差取值。下一轮若要追，正确的取证对象是"同一 kernel 几何下 grid 跨过最小值时是否存在一次性固定代价"，而现有两支探针（`cta_dispatch_cost`、`wave_contention`）都复现不出它，说明探针的几何还缺了生产 kernel 的某个特征。

## 本轮否定的候选清单

| 候选 | 判据 | 结论 |
|---|---|---|
| `add_rmsnorm` 漏记 hidden-linear 算术 | H=16384 本就走通用 `5×H`，却是最差欠预测（0.654） | **否定** |
| 补算术即可收敛 NORM | 算术占 `rmsnorm` 预测的 1.5–3.6%，H=2048 只值约 70 ns vs 412 ns 缺口 | **否定** |
| 在 typed 路径上叠加 `5×H` | typed 服务是同一段源码循环的整轮微基准，已含这些指令 | **否定（重复计价）** |
| `_add_rmsnorm_additional_round_widths` 造成 T=2→4 台阶 | 同幅台阶出现在不调用它的 `rmsnorm`／`rope`／`kvcache`／`silu` | **否定** |
| 台阶来自降频 | 台阶是加性恒定（+480…+640 ns）而非乘性恒定 | **否定** |
| 启用 A800 的 `rmsnorm_*_first_1024` typed 服务 | 按其费率反算 H=16384 得 16305 ns vs 实测 10848 ns；该表在 grid64 采集、含launch-wide 争用，现有 `return False` 正确 | **维持否定** |

## 下一轮的取证对象（已不在逐元素算术上）

承重项是 CTA 依赖链里的**每轮访存**。一个源码层面的量化观察，留给下一轮独立验证：标量 `rms_norm_kernel` 与 vec8 `fused_add_rms_norm_kernel` **每轮都只发射每线程一条全局 load 指令**，只是宽度 2 B 与 16 B 之别；而模型给标量轮记 292 ns/轮、给 vec8 轮记 196 ns/轮——即把 vec8 轮定价到了单次请求延迟之下。若成立，这解释了"标量多轮被过预测（`rmsnorm` H16384 比值 1.14）、vec8 少轮被欠预测（`add_rmsnorm` H16384 比值 0.65）"这一对相反方向。**但这需要独立的依赖 load 延迟微基准来判定，不得按上述两个比值反推。**

## 依赖 load 延迟微基准：每轮定价按字节计量被证伪

新增探针 `microbench/nvidia_a800/memory/lsu/per_thread_round_completion.cu`：一个 CTA、grid 1、每线程每轮发射 1 或 2 条全局 load，轮与轮之间严格串行（本轮 load 的值决定下一轮地址），轮内并行且合并。这正是源码循环体的执行单位，也正是模型里一个 per-round 访存 stage 要对应的物理量。窗口在计时前整块预热，CSV 输出 `prime_rounds` 与 `timed_wraps_a` 使驻留声明可审计。计时用 CTA 内 `%globaltimer`，不受 SM 动态频率影响；缓冲区全零，不消费任何生产 kernel、张量或时长。

**取指编码来自生产二进制的静态语义**，不是猜的。对 `/share_data/yangxinhao/chip_modeling/vllm_src_3ver/a800/vllm/_C.abi3.so` 反汇编：

| 生产函数 | LDG 编码 |
|---|---|
| `vllm::rms_norm_kernel<BFloat16>` | **3 × `LDG.E.U16.CONSTANT`** + 1 × `STG.E.U16` |
| `vllm::fused_add_rms_norm_kernel<BFloat16, 8>` | **2 × `LDG.E.128`（plain）** + 2 × `LDG.E.64` + 8 × `LDG.E.U16.CONSTANT` + 2 × `STG.E.128` |

`rms_norm` 的 input/weight 是 `const __restrict__`，走只读数据通路（`.CONSTANT`）；`fused_add` 原地更新 input/residual，两条主流是**普通 `LDG.E.128`**。探针复现的是 `.CONSTANT` 编码——对 `rms_norm` 逐位忠实；`plain` 编码需要轮内带上自己的 store 才能让 nvcc 不走只读通路，**声明在本探针域外**。另：`rms_norm` 的 3 条 `.CONSTANT` load 恰好对应"第一遍 1 条、第二遍 2 条（input 重读 + weight）"，与源码结构一致。

### 宽度律：8 倍字节只换来 1.3 倍时间

grid 1、1024 线程、L2 驻留、每线程每轮 1 条 load：

| 每线程宽度 | 2 B | 4 B | 8 B | 16 B |
|---|---|---|---|---|
| 轮完成时间 | **196 ns** | 198 | 226 | **258 ns** |
| 每轮字节 | 2 KiB | 4 KiB | 8 KiB | 16 KiB |

按字节计量预测 16 B 轮 = 8 × 2 B 轮；实测 **1.32×**。32–512 线程处更平（194→228，1.18×）。**按字节正比的每轮定价被证伪；轮是被请求延迟托底的。**

### 驻留律与流数律

1024 线程、`.CONSTANT` 编码：

| 每轮形态 | W=2 B | W=16 B |
|---|---|---|
| 1 条 load，L2 驻留 | 196 ns | 258 ns |
| 1 条 load，L1 驻留 | **90 ns** | 124 ns |
| 2 条 load，都 L2 | **456 ns** | 570 ns |
| 2 条 load，都 L1 | 136 ns | 234 ns |
| 2 条 load，一 L1 一 L2（`mixed`） | **224 ns** | 264 ns |

两条结论：

1. **每线程两次 L2 miss 会串行**（456 ≈ 2 × 196），不是重叠；
2. **一次 L1 命中与一次 L2 miss 完全重叠**（`mixed` 224 ≈ 单条 L2 196）。

即每轮的承重变量是**每线程未命中 L2 的流数**与**驻留等级**，既不是字节，也不是访问宽度。

### 判定：四个生产 stage 逐个对照

生产轮形态（由上表 SASS 与源码结构确定）对上探针同形态的实测值，对上模型当前的 stage 定价（H=16384、block 1024、grid 1 稳态轮）：

| 生产 stage | 每轮形态 | 模型 | 探针实测 | 模型/实测 |
|---|---|---|---|---|
| `rmsnorm` 第一遍 `load_reduce` | 1 条 W=2，L2 | 200.8 ns | 196 ns | **1.02 ✓** |
| `rmsnorm` 第二遍 `load_output` | 2 条 W=2，L1 重读 + L2 gamma | 384.2 ns | 224 ns | **1.72 ✗ 过** |
| `add_rmsnorm` 第一遍 `load_reduce` | 2 条 W=16，都 L2 | 261.9 ns | 570 ns | **0.46 ✗ 欠** |
| `add_rmsnorm` 第二遍 `load_output` | 2 条 W=16，L1 重读 + L2 gamma | 131.0 ns | 264 ns | **0.50 ✗ 欠** |

方向与两族误差一致（`rmsnorm` 过 1.14、`add_rmsnorm` 欠 0.65），但这张表**完全由探针与 SASS 得到，未使用那两个比值**。

### 我上一轮记下的假设，在其表述形式上被证伪

上一轮写的是"模型把 vec8 轮定价到了单次请求延迟之下"。**实测不成立**：单条 16 B 的 L2 轮下限是 258 ns，而模型给 `add_rmsnorm` 第一遍的正是 261.9 ns——**那一项恰好压在下限上，不在其下**。

真正的缺陷是另外两件事，两件都不是"宽度定价错了"：

1. **模型不为第二条 L2 流单独计时**。`fused_add` 每轮读 input 与 residual 两条都未命中 L2 的流，硬件上这两次 miss 串行（456 vs 196）；模型把两条流合成一份字节量后按同一个 path 时间分摊，只得到约一条流的时间。
2. **模型把 L1 命中的重读按 L2 字节份额计价**。`rms_norm` 第二遍重读的是自己刚写进 L1 的行（H=16384 时 32 KiB，在 L1 容量内），实测与 gamma 的 L2 miss 完全重叠、整轮 224 ns；模型收 384.2 ns。

### 可实现的模型改动（留给下一轮，不在本轮落地）

按**每线程未命中 L2 的流数 + 驻留等级**给 NORM 的 per-round 访存 stage 定价，取值来自上面这张已版本化的机制表；驻留等级由源码静态语义与静态容量判定（第一遍读的行是本 CTA 首触 → L2；第二遍重读同一行，行宽 ≤ CTA 的 L1 份额 → L1；gamma 首触 → L2），不是按误差挑的。gate 在"该卡是否发布了 `per_thread_round_completion` 机制表"这一离散事实上，C500 与 910B 不发布即逐位不变。

**本轮不落地的理由（须如实记录）**：把探针值直接代入，`rmsnorm` H=16384 的访存链从 9935 ns 降到 6720 ns（预测 12354→9139，实测 10848，由过 14% 变欠 16%），`add_rmsnorm` 从 786 升到 1668（预测 4669→5551，实测 7136，仍欠 22%）。两边都没闭合。原因是可辨识的结构问题而非取值问题：**探针轮只含访存，而模型把算术与 store 作为串行 stage 接在访存之后**；生产 CTA 里第 i+1 轮的 load 可以与第 i 轮的算术重叠，模型不允许。直接代入等于把重叠部分记两遍。这一层要么一并改轮内组装，要么就是按误差凑——后者触红线。

### 本轮否定的候选

| 候选 | 判据 | 结论 |
|---|---|---|
| 每轮按字节正比计价 | 8 倍字节只换 1.32 倍时间 | **否定** |
| "vec8 轮被定到单请求延迟之下" | 单条 16 B L2 轮下限 258 ns，模型 261.9 ns 正压在下限 | **否定（原表述）** |
| 宽度是承重变量 | 2 B 与 16 B 单流只差 1.32×，而流数从 1 到 2 差 2.33× | **否定** |
| 两条 L2 流会重叠 | 456 ≈ 2 × 196，32 线程处同样 2.04× | **否定** |
| 两流串行是 `STRONG.SM` 编码假象 | 弱编码 `LDG.E.CONSTANT` 重测，两版在 trial spread 内一致 | **否定** |
| 直接代入探针值即可收敛 | 两族分别变为欠 16% 与欠 22%，均未闭合；根因是轮内算术重叠被记两遍 | **否定** |

## 收益上限先算：NORM 不是可忽略的桶，值得做

按 A800 kernel evaluation 同一口径（`compute_metrics`：MAPE = 各 case APE 的**非加权均值**）分解 1027 个 case：

| op_type | N | MAPE | 对 ALL 的贡献 | 完全关闭后 ALL |
|---|---|---|---|---|
| FLASHATTN | 334 | 5.94% | 1.932pp | 5.667% |
| **NORM** | **96** | **17.61%** | **1.646pp** | **5.952%** |
| GEMM | 221 | 6.35% | 1.366pp | 6.233% |
| FUSEDMOE | 176 | 6.50% | 1.113pp | 6.485% |
| ELEMENTWISE | 96 | 5.67% | 0.530pp | 7.068% |
| ACTIVATION | 56 | 7.67% | 0.418pp | 7.180% |
| ROTARY | 32 | 11.94% | 0.372pp | 7.226% |
| KV_CACHE | 16 | 14.22% | 0.222pp | 7.377% |

**上限**（NORM 误差归零时）：

- VECTOR/ELTWISE **11.061% → 5.350%**，最多动 **5.711pp**（降 52%）
- ALL **7.598% → 5.952%**，最多动 **1.646pp**（降 22%）

NORM 的 MAPE 在全部桶里最高（17.61%，是 ALL 均值的 2.3 倍），绝对贡献排第二且仅次于 case 数多 3.5 倍的 FLASHATTN。**上限不可忽略，按主要矛盾推进。**

## 必须先更正：上一轮那四个比值不是"模型 vs 生产轮"

goal 里引用的 1.02 / 1.72 / 0.46 / 0.50 来自我上一轮把模型 stage 价格对上探针值。**这个对照是错位的**，我在本轮用与模型无关的方式查出来了。

只用实测（同 variant、同 block、只变 H，因此每轮形态不变）求每轮斜率：

| variant | H 区间 | 轮数 | 实测斜率（每"轮对"= 第一遍+第二遍） | 探针值之和 | 探针/生产 |
|---|---|---|---|---|---|
| `rmsnorm` | 2048→16384 | 2→16 | **518.9 ns** | 196+224 = 420 | **0.81** |
| `add_rmsnorm` | 8192→16384 | 1→2 | **2240.0 ns** | 570+264 = 834 | **0.37** |

**探针测的是下界，不是生产轮**，且欠缺比例按 variant 差一倍以上（0.81 vs 0.37）。所以那四个比值不能读成"模型错价 1.72×／0.46×"——它们把"模型 stage"和"一个缺项的下界"放在了一起。任何用单一系数去桥接 0.81 和 0.37 的做法都是按误差取值，触红线。

**缺的那一项已经定位：每轮的 store。** 生产轮形态（由 SASS 确定）：`rms_norm` 第二遍每元素写 2 B（`STG.E.U16`），`fused_add_rms_norm` 两遍各写 16 B（`STG.E.128`）。后者每轮对写 32 B/线程、前者只写 2 B/线程——正好是 0.37 与 0.81 这个方向的差异。

## 探针已加 store 轴，并因此拿到生产取指编码

`stores` 只对双流轮定义，写回的是**另一条流**的字，缓冲区保持全零，同时让 `ptxas` 无法证明"写回刚读的值"。副产品是编码自动对上了：一旦该流有 store，它的 load 就从 `LDG.E.*.CONSTANT` 掉回普通 `LDG.E`。`W=16 / 2 流 / 1 store` 实测发出 `LDG.E.128` + `LDG.E.128.CONSTANT` + `STG.E.128`，正是 `fused_add_rms_norm_kernel<bf16,8>` 的形态（生产的第二条主流也是 plain，因为它在另一遍被写）。

## 本轮被硬件占用阻塞，且这件事本身产出了一个 fail-closed 守卫

采集时另有 8 个进程把 8 张 A800 全部压到 100%。探针只占 1 个 CTA、1 个 SM：**L1 驻留 cell 是该 SM 私有的，与干净捕获逐点一致**（1024 线程处 90/90/104/124 完全复现）；**L2 驻留 cell 全部被污染**——同一个 `l2 / 1 load / W=2 / 1024 线程` cell，干净时 **196 ns**，占用下 **534 ns**，而**每 cell 的 trial spread 仍是 1.02**，还出现 8 B 轮比 4 B 轮更快这种物理上不可能的次序。

**持续性占用不会让数据变噪，而是让它整体偏移**，所以任何事后统计都不可靠地分辨它。因此新增 `per_thread_round_completion_runner.py`：采集前校验设备为空闲的 108-SM CC8.0 A800、无 compute 进程、保留显存 ≤8 MiB，否则拒绝运行（与 SiLU load-pair runner 同一先例）。这是设备前置条件，不是容差，不由任何测量或误差选出。实测该守卫在当前机器上正确拒绝。

`per_thread_round_completion_report.py` 负责发布：按 trial spread 扣留 cell，并记录 `width_monotonicity_flag`（同一行内更宽的轮不应更快，超出该行自身 trial spread 才记）。**这一项只标记不扣留**——干净捕获里 2 B 与 4 B 之间仍有 1–2% 的倒挂（两者在 1024 线程下物理上不可分辨），而被污染的捕获却可能整体通过。它是审计辅助，守卫是 runner。

**结论：store 轴的 cell 本轮不发布。** 生产 NORM 用的 1024 线程几何下，L2 与 mixed 的 cell 无一在干净条件下采到。`references/per_thread_round_completion_sm80.json` 仍是此前那次空闲捕获（无 store 轴）。被污染的捕获只作为失败模式证据留在 `profiler_output/`。

## 四个 stage 的关闭因此未落地，缺口是一次干净采集

实现方案不变（按流数+驻留等级定价、驻留由源码静态语义与静态容量判定、一并改轮内组装使 load 与算术不重复计），但它的取值必须来自**含 store 的、1024 线程、L2/mixed 驻留**的 cell。在拿到一次空闲设备的采集之前落地，就只能用被污染的值或自选系数，两者都触红线。

## 驻留判定：不需要新证据，落在仓库已发布的 L1 驻留域内

实现方案要求"驻留由源码静态语义与静态容量判定"。逐 case 核对后，**A800 全部 96 个 NORM case 的每 CTA 行足迹都 ≤ 32768 B**，正好是已发布机制 `rmsnorm_shared_gamma_l1_residency`（`source_shared_gamma_residency_tier_v2`）的 `maximum_evidenced_gamma_bytes` 证据上界：

| variant | H | 每 CTA 行字节 | ≤ 证据上界 |
|---|---|---|---|
| 两个 variant | 2048 / 2560 / 4096 / 5120 / 8192 / 16384 | 4096 / 5120 / 8192 / 10240 / 16384 / **32768** | 全部 **是** |

因此驻留分类可直接由源码静态语义给出，**不需要引入新的 L1 容量常数**：

- 第一遍读的行是本 CTA 首触 → **L2**
- 第二遍重读同一行（`rms_norm` 重读 input、`fused_add` 重读 residual）→ **L1**
- gamma 是本 CTA 首触 → **L2**

一处需要如实标注的外推：上述已发布的 tier 测的是**共享 gamma** 行，而这里用到 input/residual 的**私有**重读上。私有重读是更强的情形（同一 CTA 刚写进 L1、数百纳秒后重读、其间无其他流驱逐），所以"≤ 已证据的 32 KiB"是保守外推；探针的 `mixed` 驻留 cell（一流 L1、一流 L2）正是对它的直接验证，采集时一并确认。

## 采集目标已收敛到生产轮的确切宽度

把每个 case 的源码循环展开，列出生产轮**实际发射的活跃线程数**：

| variant | 每轮宽度 |
|---|---|
| `rmsnorm`（标量，W=2 B） | 512、1024 |
| `add_rmsnorm`（vec8，W=16 B） | 64、128、256、320、512、640、1024 |

vec8 的这组宽度与仓库已发布的 `add_rmsnorm_source_round_pipeline` 所测的 256/320/512/640/1024 一致，互为佐证。

探针原先只扫 32/128/256/512/1024，**缺 64、320、640**——那会迫使三个生产轮靠相邻 cell 插值。已把线程轴改为 **32、64、128、256、320、512、640、1024**，使每个生产轮都有一个逐点对应的实测 cell，不需要任何插值。

## 重大修正：`add_rmsnorm` 的轮定价**已经**是源码精确的，方案要换目标

在写新代码前先查了仓库已有证据，结果推翻了我自己上一轮的读法。

`microbench/nvidia_a800/references/add_rmsnorm_block1024_pipeline_sm80.json`（schema v6）**已经**是对生产 `fused_add_rms_norm_kernel` 两个源码轮的 clock64 实测：

- `production_function_sass_sha256` 与生产函数一致（`probe_and_production_resource_identity_exact: true`）；
- 服务就是 `add_square`（第一遍：2 load + store）与 `scale_multiply`（第二遍：2 load + store）——**完整的轮，含它自己的 store**；
- `production_cta_local_service: grid_one_intrinsic_round_only`，即取的是 **grid 1** 的 CTA-local 轮，更大 grid 的捕获只作验证（因为含 launch-wide 争用）；
- `composition_contract` 明确 `generic_complete_memory_task_replaced: true`；
- 捕获时设备空闲。

**所以对 5 个 typed H 值，`add_rmsnorm` 的每轮访存+算术已经由一个源码精确、空闲采集、含 store 的实测轮定价。**我上一轮读出的"`add_rmsnorm` 两个 stage 欠一半（0.46 / 0.50）"，对这些 case **不成立**——那是拿它和我自己那个**不含 store 的下界探针**比出来的，比错了对象。轮价是整个模型里证据最强的一项。

实际分解（T=1，全部 H-无关项已分离）：

| case | launch | 块归约 | 源码轮 | warp | 模型 | 实测 | 缺口 |
|---|---|---|---|---|---|---|---|
| H=2048 | 1105 | 864 | 876 | 39 | 2884 | 3296 | **412** |
| H=2560 | 1105 | 864 | 868 | 48 | 2886 | 3488 | **602** |
| H=4096 | 1105 | 864 | 1061 | 77 | 3107 | 3840 | **733** |
| H=5120 | 1105 | 864 | 1236 | 96 | 3302 | 4064 | **762** |
| H=8192 | 1105 | 864 | 1974 | 154 | 4098 | 4896 | **798** |

`launch` 与`块归约`与 H 无关，轮价源码精确，**缺口却随 H 增长（412→798）**，说明存在一个在"launch + 轮 + 归约"这套组装之外的 H-相关项。它不是任何单一轴的恒定比例：对轮价是 0.47/0.69/0.69/0.62/0.40，对元素数是 0.201/0.235/0.179/0.149/0.097 ns，对活跃线程是 1.61/1.88/1.43/1.19/0.78 ns。**本轮不对它取值**——没有机制就取值就是拟合。

## 真正的缺口在 `rmsnorm`，且修法是已被接受过的那一种

`rmsnorm` **没有** grid-1 的源码精确轮服务。A800 只发布了 `rmsnorm_input_square_first_1024` / `rmsnorm_output_scale_first_1024` / `rmsnorm_phase_entry_first_1024_active64`，而它们是 **grid 64** 采的联合包络；`_has_typed_rmsnorm_scalar_round_services` 因此对 A800 恒返回 False，注释写明理由是"grid-64 探针给的是 joint load/compute 源码循环包络，不能分离出可与归约、store 组合的 CTA-local 访存完成量"。所以 `rmsnorm` 至今仍走按字节分摊的通用链——**这才是那个结构性错价**。

修法不需要发明：**把 `fused_add_rms_norm` 已经走过并被接受的路，原样用在 `rms_norm` 上**——一支 grid-1、源码精确的轮探针，按它自己的宽度（标量 512 与 1024）分别测两个轮：

- `input_square`（第一遍）：1 条 load（2 B，L2 首触）
- `output_scale`（第二遍）：2 条 load（input 重读→L1；weight→L2 首触）+ 1 条 store（2 B，写 `out`，注意 `rms_norm` **不是**原地，输出是第三个张量）

先例、schema、组装契约、验收判据（grid-1 取值、更大 grid 仅作验证）全都现成。

## 对我自己那支通用探针的定位更正

`per_thread_round_completion.cu` 按（流数、驻留、宽度、线程）建表，实际上是在重新发明仓库已有并已被接受的"源码精确轮探针"方法。它测到的东西仍然成立且有价值——宽度律（8 倍字节只换 1.32 倍时间）、两条 L2 流串行、L1 命中与 L2 miss 完全重叠——**但它不是给这两个 stage 定价的正确工具**，源码精确轮探针才是。它继续作为机制证据保留，不进 Forward。

## 当前阻塞

两项采集都需要空闲 A800：`rms_norm` 的 grid-1 轮探针，以及我那支探针的 store 轴。**8 张 A800 全部被他人任务占到 100%**，已挂 monitor 等空闲，未抢占、未空等。

## 探针已就绪并通过端到端校验，剩下的唯一缺口是一台空闲 A800

`rmsnorm_scalar_round_service_1024.cu` 在被占满的设备上跑过一次**纯功能校验**（数值作废、不发布），结果证明 harness 本身是对的：

- 行数逐 cell 精确（trials × blocks）；
- SM 落位正确：`grid_1` 落 1 个 SM、`grid_4` 4 个、`grid_16` 16 个、`grid_108` **108 个互异 SM**；
- 运行时占用校验通过（resident CTA = 2，否则中止）；
- 四个「函数 × 宽度」特化全部产出；
- 两条物理次序都成立：`output_scale` 恒大于 `input_square`（2 load + store vs 1 load），1024 宽恒大于 512 宽。

同时它对共享资源状态**确实敏感**——被占用时各 cell 的 trial spread 升到 1.8–3.6（空闲先例里 `add_rmsnorm` 同类捕获是 ≤0.0125）。这既证明探针能反映 L2 状态，也再次说明这批数值不可用。

**结论：定价方法、轮形态、宽度集合、驻留判定、组装契约、探针与 harness 全部就绪并已提交；唯一未完成的是一次空闲设备上的采集。**

## 本轮状态：Blocked（外部资源）

按 AGENTS「不要靠等空窗，也不要抢占」，我没有抢占，也不再继续等窗口——已连续三个 30 分钟窗口 8 张 A800 全部 100% 占用。本轮按 Blocked 收尾：剩余路径只差一次采集，且该采集按仓库既有先例（`add_rmsnorm` 参考记录 `preflight_all_devices_idle: true`）必须在空闲设备上进行，我无法绕开也不应抢占。

采集命令（设备空闲后一条即可，runner 会自行校验并在不满足时拒绝）：

```
python microbench/nvidia_a800/memory/lsu/per_thread_round_completion_runner.py \
  --output profiler_output/microbench/nvidia_a800/<capture> --device 1 --owner <owner>
/usr/local/cuda-12.2/bin/nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo -maxrregcount=24 \
  -o rmsnorm_round microbench/nvidia_a800/compute/sfu/rmsnorm_scalar_round_service_1024.cu
CUDA_VISIBLE_DEVICES=1 ./rmsnorm_round > <capture>/raw.csv
```

## 占用不可校正：100 个 cell 的干净/占用逐点对照

在断言"被占用的数据不可用"之前，先按 AGENTS「用背靠背对照或相对比值」把两种校正都试了，**两种都被证伪**。拿已发布的干净捕获与占用下的重测逐 cell 相减（stores=0，100 个 cell）：

| 驻留 | n | delta 中位 | delta 范围 |
|---|---|---|---|
| `l1` | 40 | **0.0 ns** | −2 … +2 |
| `l2` | 40 | 139.0 ns | 60 … 5266 |
| `mixed` | 20 | 278.0 ns | 58 … 5278 |

- **L1 cell 逐点完全一致（40/40，±2 ns）**——它是该 SM 私有的，其他租户碰不到。这同时反证了探针本身是稳的。
- **L2 与 mixed 的 delta 跨两个数量级**（60→5278 ns），既不是恒定加性（否则可减去），也不是恒定乘性（比值 1.13–24.9）。**所以没有任何事后校正能把这些 cell 救回来**，相对比值法同样不成立（同一对 cell 的比值干净时 1.143、占用下 1.071）。

这条把"必须空闲设备"从一句谨慎话变成了实测结论。

## 生产 SASS 判定轮内组装：load 跨迭代独立，只有累加是串行链

组装该怎么改，不需要等采集——生产二进制直接给了答案。`vllm::rms_norm_kernel<c10::BFloat16>` 第一遍的循环体（`0xc0`–`0x170`）是：

```
0x0120  LDG.E.U16.CONSTANT R6, [R6.64]      ; 载入 x
0x0130  IADD3  R9, R9, c[0x0][0x0], RZ      ; idx += blockDim.x
0x0140  ISETP.GE.AND P0, PT, R9, c[..], PT  ; 循环条件
0x0160  FFMA   R3, R6, R6, R3               ; variance += x*x
0x0170  @!P0   BRA 0xc0
```

**地址寄存器由循环计数器 `R9` 重算，不来自上一轮载入的值**；跨迭代唯一的依赖是 `R3`（variance 累加器）。所以：

- **同一 warp 内**：`R6` 在 2 条指令后即被 FFMA 消费，该 warp 仍被 load 延迟卡住——循环未展开，每迭代只有一条 LDG；
- **CTA 层面**：1024 线程 = 32 个 warp，各自独立跑自己的 load→FFMA 链，彼此重叠。

因此模型现有的 `load_reduce_{i} ← input_fma_{i-1}` 这条边**在源码里不存在**，它把 32 条并行的 warp 链错当成一条全局串行链。这是轮内组装的结构性缺陷，判据来自生产 SASS 而非任何误差。

但**本轮不改它**，理由是红线而非保守：改完之后每轮该收多少时间，恰恰由那次还没做的采集决定（一个孤立轮、生产资源签名、grid-1）。先改结构再用现有的按字节分摊值去填，等于用一个没有机制的数去补一个刚刚被证明有机制的洞——那还是拟合。两件事必须一起落，顺序不能倒。

## 把收益上限再算深一层：本 goal 指定的那项工作，上限只有 0.211pp

上一节算的是"NORM 整桶归零"的上限（1.646pp）。但**指定的机制（按流数+驻留等级给轮定价）只能作用于 NORM 的一个切片**。按 variant × 波域分解那 96 个 case：

| variant | 波域 | N | MAPE | 占 NORM 误差 | 对 ALL |
|---|---|---|---|---|---|
| **`ADD_RMSNORM`** | **一波（grid ≤ 108）** | 36 | **26.05%** | **55.5%** | **0.913pp** |
| `ADD_RMSNORM` | 多波（T=256/1024） | 12 | 24.15% | 17.1% | 0.282pp |
| `RMSNORM` | 多波（T=256/1024） | 12 | 20.47% | 14.5% | 0.239pp |
| `RMSNORM` | 一波（grid ≤ 108） | 36 | **6.03%** | **12.8%** | **0.211pp** |

关键在于：**`add_rmsnorm` 的轮价已经是源码精确的**（前节已证），所以"重新给轮定价"这件事只能落在 **`RMSNORM` 一波**那 36 个 case 上——而它们当前 MAPE 只有 **6.03%**，即使**误差归零**，ALL 也只从 7.598% 动到 **7.387%（0.211pp）**。

按 goal 自己的规则（"上限可忽略就不要做，报我并转去主要矛盾"），**这项工作不该做**。goal 给的 1.646pp 是整桶上限，不是该机制的上限；把同一条纪律再往下用一层，结论就反过来了。

## 用现有干净数据实现的结果，也实测过了：会把模型做坏 4 倍

在下这个结论前，我把"用已发布的干净 cell 直接实现"跑成了反事实：`load_reduce_{i}` 取 `l2 / 1 条 load / W=2 / <该轮活跃线程>`，`load_output_{i}` 取 `mixed / 2 条 load / W=2 / <同>`（两个 cell 都在已发布的空闲捕获里，宽度 512 与 1024 也都覆盖），store 与算术保留模型现有的独立 stage，只替换临界路径上的访存 stage：

| | 当前 | 代入后 |
|---|---|---|
| `RMSNORM` 一波 36 case MAPE | **6.03%** | **23.74%** |
| 中位 预测/实测 | 0.997 | **0.751** |

**36 个 case 全部从 ~1.0 掉到 ~0.75**，均匀欠计约 25%——与"依赖轮探针只有生产轮对斜率的 81%"这条独立测量一致。这不是噪声缺口而是机制缺口（探针不含 store、不含循环开销、指令序列与生产不同），所以**正是需要源码精确单轮探针的理由**；而用任何系数去补上这 25% 就是按 MAPE 反推，触红线。

## 结论与转向

1. **不做**本 goal 指定的按流数+驻留轮定价：其上限 0.211pp，且用现有数据实现会把 `rmsnorm` 一波从 6.03% 做到 23.74%。
2. **NORM 内部的主要矛盾是 `ADD_RMSNORM` 一波（0.913pp，占 NORM 误差 55.5%）**，而它的轮价已经源码精确、launch 与块归约都与 H 无关，缺口却随 H 从 412 增到 798 ns——**残差在组装之外的某个 H-相关项上，不在轮价上**。这才是下一个该查的对象，且它**不需要新采集**。
3. 已建成的 `rmsnorm_scalar_round_service_1024.cu`（含 runner 守卫与端到端校验）保留：它是把 `RMSNORM` 一波那 0.211pp 做干净的正确工具，但按优先级排在主要矛盾之后。

## 缺口逐机制排查：两条结构性证伪，第三条卡在同一处空闲设备缺口

上一节把缺口钉在"launch + 轮 + 块归约 + warp 之外"，本节逐个检查模型里剩下的、原则上还能落进这条组装的机制，结果是没有一条能在不新增采集的前提下解释它。

**块归约不是候选**：上表里"块归约"对全部 5 个 H 都是常数 864——`block_reduce_latency.cu` 的占用感知公式 `T_reduce(G,B) = L_B + (min(N,R_B)-1)*S_res + max(0,N-R_B)*S_queue` 在 grid=1（N=1）时退化成单个与 H 无关的 `L_B`，这与实测的常数列完全一致，不可能是缺口来源。

**已有的 store 完成度机制不是候选**：`vector.py` 里 `_completion_dependent_dram_write_bytes` / `_resident_operand_dram_read_bytes` 按 `hardware.gpu_store_completion_scope`（`l2` 还是 `dram`）和 `l2_operand_residency_curve` 决定 store/load 字节要不要算穿透 L2 的完成成本——但它是带宽项（字节 ÷ 速率），不是固定延迟，且当前架构下所有内存资源组在每个 H 都已经严格小于 `pipeline`（取 max），所以无论这一项本身随 H 怎么变，它在现有的"取各资源组最大值"架构下永远浮不到 `pipeline` 之上。这条机制存在但结构性地够不着这个缺口。

**唯一还站得住的候选需要新采集，不满足本 goal 的前提**：T=1、grid=1 时 CTA 没有别的 warp/block 可以重叠，kernel 结束前最后一次 store 的完成延迟没有任何东西可以掩藏——这与 910B `[[910b-rope-l2-store-completion-fix]]` 是同一类机制（低并发单发射、原地写的终态完成暴露），方向未必相同但类别一致，是唯一还没被结构性排除的候选。但仓库里唯一测这类"轮完成时间"的探针 `microbench/nvidia_a800/memory/lsu/per_thread_round_completion.cu` 自己的头注释明确把生产相关的 store-carrying 配置（`fused_add_rms_norm_kernel` 实际发的 `LDG.E.128`）排除在"该探针声明的域"之外（第 105-109 行注释原文 "outside this probe's declared domain"）；对应的 `per_thread_round_completion_sm80.json` 逐 cell 核对，`stores=1` 的组合一条都没有被采集或发布过。这一候选因此无法在不做新采集的前提下定出参数，而这正好和本 goal 明确排除的"不需要任何采集"前提矛盾。

**结论：unsupported，本轮不取值**。412/602/733/762/798 ns 这条缺口，在"launch + 轮 + 块归约 + warp"四项组装之外，没有一条现有的零拟合参数来源能解释它；唯一物理上说得通的候选（终态 store 完成延迟暴露）需要一次新的空闲 A800 捕获才能定价，且必须和已经卡住的另外两项一起落地：`rmsnorm` 一波按流数+驻留定价的采集（上节，0.211pp）与生产 SASS 判定但尚未实现的轮内组装修正（`load_reduce_i ← input_fma_{i-1}` 这条不存在的边，本文件更早一节）。三者都在等同一张空闲设备，不该分批做——先落任意一个都会把还没定价的机制填上一个没有来源的数，等于拟合。

在 A800 空闲前，本目标按 unsupported 收尾：机制候选已穷尽到"需要采集"为止，代码不改，`per_thread_round_completion.cu` 与 `add_rmsnorm_block1024_pipeline_sm80.json` 均未改动。
