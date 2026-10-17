# 910B FULL-graph padding：appended-slot FA 按 captured 数组定价，并补齐 FIA single-task 覆盖洞

本文记录 910B vLLM Ascend FULL-mode graph replay 下 padded FLASH_ATTN decode 的定价结论，以及与之配套的 FIA single-task 覆盖洞补齐。VECTOR 一侧的同族结论见 `docs/ascend_910b/graph_padding_physical_extent.md`。原始诊断日志留在 `experiment_logs/` 不入版本控制；本文只保留结论与判据。

## 1. 两种 padding 形状，只有一种可以重绑

`GraphAttentionShape.from_token_shape`（`src/serving/workload/graph_padding.py`）恰好只有两种 padding 形状：

| 形状 | 触发条件 | 对 slot 数组的作用 |
|---|---|---|
| **appending** | 整批都是 uniform decode | 在尾部**追加** dummy 槽位，Q 与 KV 都等于 `uniform_decode_query_len`；每个真实请求的槽位逐字未动 |
| **extending** | 其余（prefill / mixed） | **拉长**最大的那个逻辑 segment，抬高某个真实请求的 KV 长度 |

`_has_graph_padding` 对两种形状一律拒绝 per-CTA source schedule，理由是：把 pinned 的一请求一槽位调度表原样复刻到被拉长的 segment 上，会把 masked 行算成语义 attention 工作。**这条理由只对 extending 成立**。appending 形状下 captured launch 确实按记录中的请求数逐槽位发射，把 source program 绑到 model-input 数组上定价的是**已被分派的工作**，并且严格高于它所超集的那个未 padded 工作量。

守卫本身未改；它不再触发，是因为 spec 不再是 padded——与 VECTOR 重绑同一形状。

## 2. 判据来自 captured 槽位数组自身，fail-closed

`_cann_fia_appended_slot_execution_kernel`（`src/kernel/predictors/flash_attn.py`）只在下列条件同时成立时重绑，且逐项来自发射记录而非按卡开关：

- `graph_runtime_mode == "FULL"` 且 launch 命中 CANN 标准 GQA / MLA decode 算法集合；
- `cann_source_task_count` 在 launch 参数中；
- model-input 数组的槽位数**不少于**逻辑槽位数（少于则 `ValueError`，padding 不可能缩减槽位）；
- 逻辑前缀逐槽位未被扰动；
- dummy 尾部每个槽位 `q_len == kv_len`。

任何一条不成立就原样返回，回到既有的 fail-safe 定价——fail-closed，不是 fail-open。重绑出的 spec 只在这一次定价调用内有效（`dataclasses.replace` 副本），Serving 的 token 记账仍用逻辑 token 数。

## 3. FIA single-task LUT 是稀疏精确匹配，缺格即 fail-open

`single_task_active_aic_groups` 是精确匹配表：active AIC-group 数不在表内时，FIA plan gate 直接放弃，该 kernel 退回按 launch 常数定价，而不是按实测 cross-domain program 定价。原表 13 格缺 11 个计数，本次补齐 10 个（group 2 的六轮跨轮 span 为 16.15%，超过 15% 参数 span 上限，withheld）。已发布的 13 格及其 report SHA 逐字节未动，新点进独立 provenance 的 `extended_single_task_points`，两串只在 HWSpec normalization 时合并。

在 `ascend_910b/qwen3-4b/tp1_qps16` 的 1278 个 FLASH_ATTN/decode step-kernel 上，退回 launch 常数（`cann_fia_fixpipe_service_evidence_unavailable`）的条数：

| 树 | padded 侧 | exact 侧 |
|---|---:|---:|
| main `73b38cfd8` | 724 | 12 |
| main + LUT | 717 | 0 |
| main + FA 重绑 | 25 | 12 |
| main + LUT + FA 重绑 | **0** | **0** |

即两处改动关闭的是**不同**的 fail-open：重绑把绝大多数 padded spec 带回被覆盖的 grid，LUT 补掉剩下的 25 条 padded 与 12 条 exact。只有两者同时在场时，这个 case 里没有任何 FA decode kernel 再按 launch 常数定价。

## 4. ladder：两个 rung 的符号只有在合并后才同时为正

padding 不可能减少被分派的工作，因此 `+Graph padding` rung 不得低于 `+Graph` rung。每个 rung 是独立重仿真（`independent_simulation_per_rung`）。`ascend_910b/qwen3-4b/tp1_qps16`：

| 树 | TTFT step (ms) | TPOT step (ms) |
|---|---:|---:|
| main `73b38cfd8` | +1.052021 | **−1.079305** |
| main + LUT `1761c11fd` | +0.245027 | **−1.192608** |
| main + FA 重绑 | **−0.455451** | +0.157415 |
| main + LUT + FA 重绑 `976585087` | **+0.214825** | **+0.211717** |

**main 今天在 TPOT rung 上已经是反的。** 单独合 LUT 把它压得更负；单独合 FA 重绑修好 TPOT 却把 TTFT 打成负；两者同时在场时两个 rung 才都为正。

## 5. paired step 分解（固定观测调度，无重仿真反馈）

同一 case 的 1382 步中 787 个 padded step，`graph_padding_paired_steps` 在同一条观测调度上按 exact 与 captured 两个 capture set 定价：

| 树 | `sum_delta_ms` | 负向 step | `FLASH_ATTN/decode` 净 | `VECTOR/silu` 净 |
|---|---:|---:|---:|---:|
| E5 之前 `77446bd9a` | −1247.946 | 776 | −1098.353 | −172.946 |
| main `73b38cfd8` | −945.711 | 734 | −1098.256 | +4.130 |
| main + LUT | −947.923 | 739 | −1100.468 | +4.130 |
| main + FA 重绑 | +184.350 | 81 | +31.804 | +4.130 |
| main + LUT + FA 重绑 | **+194.753** | **63** | **+42.208** | +4.130 |

两个份额是正交且可加的：已合入的 VECTOR 重绑关掉 VECTOR 那一份（silu −172.946 → +4.130），FA 重绑关掉 FA 那一份（−1098 → +31.8），LUT 在 FA 重绑之上再补 +10.4 ms。**FA/decode 的 −1098 ms 反转在 E5 之前就已经存在于 main**，不是 E5 引入的；E5 只是把 VECTOR 那一份移走之后让它单独暴露出来。

## 6. Serving 效果（ShareGPT QPS16，per-request MAPE）

对照 main `73b38cfd8` 与合并后 `976585087`：

| case | ttft | tpot | e2e |
|---|---|---|---|
| llama-3.1-8b | 18.426 → 17.911 | 11.701 → 7.208 | 12.361 → 8.168 |
| qwen3-4b | 15.802 → 14.827 | 7.357 → **9.527** | 6.682 → **8.602** |
| qwen3-8b | 18.952 → 18.611 | 13.222 → 8.065 | 13.024 → 7.840 |
| 9 项均值 | | | **13.059 → 11.195** |

机制只允许预测值上移（masked 槽位现在有占用，且不再按 launch 常数折价），方向由机制决定不是调出来的。qwen3-8b 与 llama-3.1-8b 的三个指标在 main 上都是欠预测（例如 qwen3-8b tpot 预测 22.207 ms、观测 25.325 ms），上移因而逐项收窄误差。qwen3-4b 不同：它在 main 上中位已贴近观测（tpot p50 预测 18.462 ms、观测 18.687 ms）而尾部已经过预测（p90 22.669 vs 21.934、p99 28.201 vs 24.629），本次上移把整条分布推过观测（p50 20.240、p90 24.368、p99 33.015），per-request MAPE 因而变差。**这是分裂结果而非整体改善**，4B 尾部的过预测另有来源，不得用本改动的其他指标去对冲，也不得据此给上移加任何上限。

只合 FA 重绑（不含 LUT）时 9 项均值为 11.251；LUT 在其上再贡献 −0.056 pp，但它真正承重的地方是 §3 的覆盖洞与 §4 的 TTFT rung 符号，不是这 0.056 pp。

## 7. 隔离证据（不是效果证据）

910B kernel evaluation 完全不受影响：1027 条 entry 的逐例语义结果在 main 与合并后**逐字节相同**（语义产物中只有 `hardware_config` 一项不同，因为 HWSpec 多了扩展 LUT 点），表格逐行相同（ALL 13.99、FLASHATTN/EAGERATTN 17.15、VECTOR/ELTWISE 15.06）。原因是 `graph_*` 属性只由 `src/serving` 产出，kernel evaluation 不走那条路径；而补齐的那些 active AIC-group 计数在 kernel dataset 的 grid 上一个都没被命中。

A800 与 C500 的 ShareGPT QPS16 结果逐字段相同（各 126 行，除评估器自身 wall time 外零字段变化）。

## 8. 残差边界

合并后仍有 63 个负向 padded step，负向合计 −23.269 ms，**按 G6 声明为 unsupported 并给出边界**，不硬塞机制、不加 floor、不 clamp：

- `FLASH_ATTN/decode` −8.7162 ms，124 个 step-kernel。其中 89 个（−5.5098 ms）两侧 grid 同为 `(24,1,1)`，即与覆盖洞和 grid 变化都无关，单条幅度约为该 step-kernel 自身时间的 0.4%–0.8%；28 个（−2.9944 ms）grid 由 23 升到 24。
- `VECTOR/silu` −12.7158 ms、`VECTOR/add_rmsnorm` −1.5272 ms、`VECTOR/rmsnorm` −0.0217 ms：与 VECTOR 重绑同一批，边界见 `docs/ascend_910b/graph_padding_physical_extent.md` §4 与 §5，本次未改变。
- `GEMM/gemm` −0.2878 ms：本次改动前后逐字节相同，属既有残差。
