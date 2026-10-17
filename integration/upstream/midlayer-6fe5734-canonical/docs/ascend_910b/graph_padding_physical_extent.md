# 910B FULL-graph padding：padded VECTOR 按物理 extent 定价

本文记录 910B vLLM Ascend FULL-mode graph replay 下 padded VECTOR 的定价结论。相关原始诊断日志留在 `experiment_logs/` 不入版本控制；本文只保留结论与判据。

## 1. padding gate 必须留着，改的是 spec 不是 gate

`vector.py` 上三处 `not graph_padding` 守卫（`_TYPED_SWIGLU_PIPELINE_PARAMETERS` 完整性检查两处与 `cann_tiling_key` 检查一处）编码的规则是：**pinned per-request 调度表不得被原样复用到 masked 槽位上**。这条规则成立，不因本次改动被推翻。

改动的形状是**重绑**而非拆门：`_cann_full_graph_source_program_execution_kernel`（`vector.py`）在守卫求值之前，把 padded spec 的 `KernelSpec.shape` 改写为其 launch 记录本就描述的物理 extent。守卫之所以不再触发，是因为 spec 不再是 padded，而不是因为守卫被削弱。三处守卫逐字节未变。

由此确立的不变量：**`pipeline_plan` 不得静默变成 `None`**。padded 与 exact 两侧必须在同一模型深度上定价；两侧深度不同就不是对 padding 效应的有效测量，而是两个模型之差。

## 2. 判据来自 launch 记录自身，不来自按卡开关

重绑的触发条件是每一次发射的记录能否自证它跑在物理 extent 上，用两个来自 CANN 源码的恒等式：

| variant | 恒等式 | 来源 |
|---|---|---|
| `rmsnorm` / `add_rmsnorm` | 每个 AIV 拥有一段非空连续行区间，`blockFactor == ceil(physical_rows / blockDim)`，且 `blockDim <= physical_rows` | CANN NORM ABI |
| `activation` / `silu` | `total_tiles == ceil(rows / base_row_len) * ceil(columns / base_col_len)`，且 `blockDim <= physical_tiles` | `select_cann_swiglu_tiling`（`vector_launch.py`，`GluSingleTilingCalculator` 的直译，`cann/ops-nn@9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef`） |

两式在**物理 extent 上成立、在逻辑 extent 上被违反**，所以记录本身就能判定这次发射按哪个 extent 分派。记录证明不了物理 extent 分派时，spec 保持原有的逻辑 padded 契约，行为与改动前完全一致——即 fail-closed，不是 fail-open。

这条路径不引入任何 per-card flag。早前曾在 `ascend_910b.yaml` 上加过一个 `full_graph_source_program_physical_extent` 开关并随即回退，因为它的理由是循环的：用"这张卡按物理 extent 跑"去论证"按物理 extent 定价"。把判据放回发射记录上，从结构上消掉了这层循环。

## 3. masked 槽位计的是什么

padding 行与 padding tile 在**属于任何已调度请求之外**的 captured buffer 槽位上跑同一段保留程序：NORM block 仍经 MTE2 载入一行 hidden、在 AIV 上归约、经 MTE3 写回；SwiGLU tile 仍流入两列操作数、写出一列。重绑把这些计为**匿名的 AIV/MTE 占用**。

它不把这些槽位变成语义工作：重绑出的 spec 只在这一次定价调用内有效（`dataclasses.replace` 副本），Serving 的 token 记账仍用逻辑 token 数，masked 行不携带任何 per-request 身份——物理 extent 的调度分派的是匿名行与匿名 tile。

## 4. 锯齿：当前承重，且已实测

**此前"锯齿真实但不承重（0.0000 ms）"的结论已作废，不得继续引用。** 那个结论只在 padded silu 全部落在 plan-drop 路径上时成立；重绑把这些 spec 送回 typed 模型后，条件不再满足，该数字按"必须重测而非拼接"的要求重新测过。

在 `ascend_910b/qwen3-4b/tp1_qps16` 的 1382 步中 787 个 padded step 上，`VECTOR/silu` 有 215 个负向 step-kernel 合计 −12.7158 ms。按 CANN host tiler 在两个物理 extent 上的串行轮数分类：

| 轮数 exact → padded | 负向 step-kernel | 合计 |
|---|---:|---:|
| 2 → 1 | 104 | **−10.4919 ms** |
| 1 → 1 | 111 | −2.2239 ms |

即 **82.5% 的负向 silu 差额来自跨 pad 时串行轮数下降**，这是 tiler 的属性。典型形状（`columns` = 9728，48 个 AIV）：

| 物理行 exact → padded | `total_tiles` | `base_row_len` | 串行轮数 | n |
|---|---|---|---|---:|
| 25 → 32 | 50 → 48 | 1 → 2 | 2 → 1 | 8 |
| 28 → 32 | 56 → 48 | 1 → 2 | 2 → 1 | 15 |
| 33 → 40 | 51 → 48 | 2 → 5 | 2 → 1 | 31 |
| 43 → 48 | 50 → 48 | 9 → 1 | 2 → 1 | 6 |
| 49 → 56 | 49 → 48 | 1 → 10 | 2 → 1 | 3 |
| 34 → 40 | 48 → 48 | 3 → 5 | 1 → 1 | 32 |

> **这不是"padding 减少了工作"，不得如此表述。** 25 → 32 行时 padded 侧处理 311 296 个元素，exact 侧 243 200 个——**工作多出 28%**。它更便宜是因为 `GluSingleTilingCalculator` 在 padded extent 上选了另一个合法 base tile（1×9472 变成 2×4608），使 `total_tiles` 从 50 降到恰好 48，从而消去了一轮 48 个 AIV 中有 46 个空转的串行轮。工作上升，串行下降，在这些形状上串行占主导。另外 111 个两侧同为一轮、只有 tile 形状变化的 step-kernel（−2.2239 ms）是同一 tiler 事实的低幅版本。

## 5. 残差边界

−3.7728 ms 没有已验证机制，**按 G6 声明为 unsupported 并给出边界**，不硬塞机制、不加 floor、不 clamp：111 个 `VECTOR/silu`（−2.2239 ms，tile 形状变而轮数不变）、105 个 `VECTOR/add_rmsnorm`（−1.5272 ms）、105 个 `VECTOR/rmsnorm`（−0.0217 ms）。NORM 这一对上 padded grid 随 pad **上升**（例如 46 → 48 block）而 `blockFactor` 保持为 1，即更多的同样 per-block 工作被定价略低，幅度约为该 step-kernel 自身时间的 −1.5%。

给粗路径加 floor 仍然排除：那是误差补偿而非修复，与零拟合底线冲突。

## 6. 影响范围

**Kernel evaluation 在结构上不受影响。** `kernel_dataset/ascend_910b` 的 1027 条 entry 中携带 `graph_*` 属性的为 0 条，而 `graph_model_input_tokens` 只由 `src/serving/workload/build_workload.py` 与 `src/serving/kernel_lowering/lower_workload.py` 产出，kernel evaluation 不经过 `src/serving`，因此 padded spec 在该路径上不可能被构造。910B kernel evaluation 前后逐行相同（VECTOR/ELTWISE 296 例 MAPE 15.06、ALL 1027 例 13.99，完整语义产物 sha256 相同）。这是隔离证据，不是效果证据。

**只动 910B padded VECTOR。** A800 与 C500 的 ShareGPT QPS16 结果逐字段相同（各 126 行，除评估器自身 wall time 外零字段变化）。

**910B ShareGPT QPS16 效果**（per-request MAPE，同一口径；隔离基线为本改动的父提交 `77446bd9a` → 本改动 `73b38cfd8`，两次运行均在干净已提交树上）：

| case | ttft | tpot | e2e |
|---|---|---|---|
| llama-3.1-8b | 19.437 → 18.426 | 13.570 → 11.701 | 14.226 → 12.361 |
| qwen3-4b | 16.350 → 15.802 | 8.380 → 7.357 | 7.600 → 6.682 |
| qwen3-8b | 20.030 → 18.952 | 15.683 → 13.222 | 15.426 → 13.024 |
| 9 项均值 | | | **14.523 → 13.059** |

机制只允许预测值上移（masked 槽位现在有占用），9 个均值预测全部上移，方向由机制决定不是调出来的。父提交上这 9 个均值预测全部欠预测（例如 qwen3-4b tpot 预测 18.237 ms、观测 18.717 ms），因此上移在每个指标上都收窄误差；这不构成"改善即正确"的论据，方向判据仍只有机制本身。

> **此前表中的 `12.470 → 11.195` 与随之给出的"分裂结果"结论已作废，不得继续引用。** 那一对数字来自的两次运行（`gp-e5-before` / `gp-e5-after`）的 `source_commit` 同为 `c44cd2d58`，即两侧工作树都已带有 FIA 单任务覆盖补洞，且 appended-slot FA 重绑作为未提交改动同时在场。其 before/after 之差仍是对本改动的有效隔离，但两个绝对值描述的树状态在 `main` 上不存在。在那个条件下 qwen3-4b 的 tpot/e2e 已被抬到观测均值之上，上移才表现为变差；在 `main` 自身的基线上该分裂不出现。**分裂是条件依赖的，不是本改动的属性。**

## 7. FA/decode 负差额：模型机制被硬件证伪

配对诊断里 `FLASH_ATTN/decode` 有 124 个负向 step-kernel，合计 **−8.7162 ms**（kernel 级负差额 −23.2686 ms 的 37.5%），是四个负类中唯一既无机制也无边界声明的一个。本节给出它的最终归因：**模型内的省时机制在 910B 真机上不存在，负号是定价伪影，不是真实 shape 效应。**

### 7.1 模型内机制（成立且完备）

两侧走同一 typed 路径，limitations 逐字段相同、0 fail-open。launch 两侧都由 CANN `split_core.cpp` 的直译 `_cann_fia_split_core`（`src/kernel/launch_config/ascend_910b/flash_attn_launch.py`）解析推导；padded 侧因 `lower_workload.py` 的 `_flash_attn_selection_attributes` launch-selection 交换，从 61 条真实 + 3 条 dummy（q=1、kv=1、cost +384）推导，exact 侧从 61 条真实长度推导。kernel 时间 = 24 个并发 AIC 组各自 makespan 的最大值（`src/kernel/formula.py` 的 `analytical_main_execution_time = max(group_completion)`）。

追加 dummy 改变贪心输入（`core_limit = remaining_cost // (core_count - core_index)` 的整除余量），分割边界整体后移，真实请求被重新分组。以 step 349 为例：临界组 22 的任务数 58 → 42（req 51–53 的 16 个前置任务被移出），总 cost 25408 → 25792（**严格上升**）。全量扫描：124/124 个负步 padded 临界组 makespan < exact；124/124 的 `t_total × occurrence` 逐 step 精确复现观测 delta。注意这不是 §4 锯齿的同类机制候选——**它只在模型内成立**。

### 7.2 硬件证伪（预登记 A/B，910b-1 npu0，2026-09-23）

预测在运行前登记：padded 侧 device 时间更低（模型量级 61→64 为 −9.2%）。对照按 Goal 规定形态构造：同一批真实请求、同一 KV 分布，exact graph batch N 对 graph capture 于 N+k 槽后回放。eager 先测一轮被判 host-bound 作废（host 侧逐发射 tiling+enqueue 0.1545 ms/次 > kernel ~0.104 ms），改用 `torch.npu.NPUGraph` capture 冻结 tiling 后回放计时，200 次/轮 × 30 轮取中位数：

| arm | 模型预测 | 实测 delta |
|---|---|---|
| 61→64，dummy kv=1 | −9.2% | **+1.71%** |
| 61→64，dummy kv=0 | <0 | −0.10%（噪声内） |
| 28→32，dummy kv=1 | <0 | **+5.10%** |
| 28→32，dummy kv=0 | <0 | **+2.64%** |

模型对 exact 侧绝对时间的量级是对的（模型 0.10045 ms vs 实测 0.1038 ms）；错的是 delta 方向。真机上追加 dummy 槽后 device 时间持平或上升（工作变多），任何分布、任何 dummy 约定（kv=1 为 workload builder 假设，kv=0 为 vllm_ascend graph-pad 约定）下都不出现模型预测的反转。

### 7.3 结论与边界

**归因：定价缺陷而非 shape 效应。** 负号来源是模型按"每次发射用当前真实长度 + dummy 重推导 SplitCore 布局"定价，重推导在模型内恰好得到更轻的临界组；真实 kernel 不因 dummy 槽变快。代码位置即 §7.1 链路：`flash_attn_launch.py` 的推导输入 + `lower_workload.py` 的 `_flash_attn_selection_attributes` 交换 + `formula.py` 的 max-over-groups。修复方向（padded 侧 launch 应反映 capture 时刻输入而非运行时重推导）留给专门后续，本 Goal 内不动 FIA 定价与 fail-closed gate（E4），也不加 floor/clamp。该类的 −8.7162 ms 属于模型对 padding 收益的高估方向残差，按本节边界留档。

原始 A/B 产物与脚本留在 `experiment_logs/serving/g3-fa-attribution/`（设计上不入版本控制）；canonical reference 的发布由主 agent 决定。
