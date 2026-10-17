# 跨卡 MAPE 对照不是同一分母：判定与影响范围

## 判定：dataset 级，不是 vector 专有

三卡的 "实测" duration 来自三套仪器，且**每张卡在全部族上取法一致**：

| 卡 | `actual_source` | 粒度 | 覆盖 |
|---|---|---|---|
| nvidia_a800 | `nsys_nvtx_kernel_median` | **kernel** | GEMM / FLASHATTN / FUSEDMOE / VECTOR 全部 |
| metax_c500 | `device_event_op_median` | **op** | 同上 |
| ascend_910b | `msprof_timeline_op_median` | **op** | 同上 |

所以**这个分母差异贯穿整个 kernel dataset，不局限于 VECTOR**。A800 比的是 kernel span，另两卡比的是 op 区间；op 区间额外包含发射与算子间隙。

## 影响的量级按 duration 分布，不是均匀的

A800 本地实测的 op span 与 kernel span 之差为 **0.7–4.9 µs**（见 `vector_launch_floor_evidence_grade.md`，探针在 `microbench/nvidia_a800/runtime/span_granularity/`）。相对影响 ≈ offset / duration：

| 族 | A800 实测中位 | offset 占比 | 跨卡百分比对照是否被实质扭曲 |
|---|---|---|---|
| **VECTOR** | **5.25 µs** | **可达 ~60%** | **是，且是主导项** |
| FLASHATTN | 103.10 µs | ≤5% | 轻微 |
| GEMM | 161.98 µs | ≤3% | 轻微 |
| FUSEDMOE | 2691.34 µs | ≤0.2% | 可忽略 |

换算关系近似 `MAPE_op ≈ MAPE_kernel × dur / (dur + offset)`。对 FLASHATTN（103 µs）约为 ×0.97，对 FUSEDMOE 几乎为 1。

**结论**：机制是 dataset 级的，但**实质扭曲集中在短 kernel**。GEMM / FLASHATTN / FUSEDMOE 的跨卡对照最多被抬/压几个百分点的相对量，不改变结论方向；**VECTOR 的跨卡对照则不可直接引用**——A800 12.27 对 C500 6.49 里约三分之二是分母。

## 附带发现：文档记的 C500 timing target 与 evaluation 实际消费的不一致

`docs/flash_attention_accuracy.md:23` 记三卡 timing target 为 A800 `nsys_nvtx_kernel_median`、C500 **strict Graph mcTracer kernel duration 之和**、910B `msprof_task_duration`。但 evaluation 对 C500 **全部族**报的是 `device_event_op_median`（op 粒度），不是 kernel duration 之和。

两者是不同的物理量（逐 kernel 时长求和不含算子间隙；op 区间含）。**已查清：陈旧的是文档那一侧。** canonical dataset 逐 entry 的 `op_info.evaluation.timing` 显示 C500 全部 1027 个 entry 都是 `op` 粒度的 `device_event_op_median`（vector 296、gemm 221、flash_attn 334、fused_moe 176），910B 同为 `op` 粒度的 `msprof_timeline_op_median`，A800 为 `kernel` 粒度的 `nsys_nvtx_kernel_median`（FA 的 334 行中 294 行如此，其余 40 个 FA_SERVING 行是 `op`）。`evaluation/kernel/dataset.py` 的 `_OP_EVALUATION_TIMING_SOURCE` 里那些 `("nvidia_a800", "NORM")` 等条目只是 op 粒度下的**允许值校验表**，不代表 A800 当前在用 op 粒度。`docs/flash_attention_accuracy.md` 已据此订正。本文件只记录事实，未改动任何 evaluation 口径。

## 使用约束

上面的换算用的偏移量测自 torch add/mul，套到 kernel 结构不同的族时间隙未必相同；且换算假设模型能精确预测那段间隙。**它是"分母能解释多少"的上界，方向可靠、数值近似**，只用于判断往哪儿查，不得当作校正后的精度值引用，也不得据此放宽任何验收门槛。
