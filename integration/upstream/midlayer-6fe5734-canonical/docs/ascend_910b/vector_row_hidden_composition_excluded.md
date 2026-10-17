# 910B VECTOR row/hidden 复合耗时候选：已排除（子问题级，非整体 Goal 判定）

本文记录 910B qps64 TTFT Processing 过计定位中，VECTOR-kind（`rmsnorm`/`add_rmsnorm`/`silu`）算子 per-row × hidden 复合耗时这一子问题的最终裁决。完整过程见 `experiment_logs/serving/910b_highqps_ttft_overpredict/localization_round2.md` §35-§38（该目录被 gitignore，本文保留需要长期留存的结论）。判定只覆盖这一个子问题；同一 Goal 下 GEMM per-prefill-token 斜率候选（§34）已被 910B 真机逐点实测证伪（方向相反，真值比 0.964），不受本判定影响；Leg A（decode 逐行代价随 batch 增长）按该 Goal 自身的非目标声明不在本判定范围内。

**最终结论（§38 更新，推翻本文早期版本的 G6 判定）：VECTOR row/hidden 复合过计候选已被排除，不是 unsupported。** 早期版本（基于 §35-37）用的 13 个 M 采样点全部 ≥63，只覆盖本 case 真实 qps64 调度轨迹里按发射次数加权质量的 18.61%；用真实占用质量分布重新加权采样（§38，27 点覆盖 60.04% 质量，低 M 密集）后，方向反转：VECTOR 在真实主导 M 区间（M≤63，占真实质量 83.76%）的净效应是模型**欠计** 13×-44×，不是过计。§35-37 发现的"斜率过陡"（1.875×-3.005×）在本数据集尾部独立复现，是真实现象，但只发生在稀有大 M 尾部（每点权重<0.5%），被真实质量加权稀释后不是聚合效应的主导方向，无法解释、也不能贡献于本 Goal 的过计症状（方向相反）。

## 1. 现象与量级（§35-37 原始发现，M≥63 子域内仍然成立）

固定 hidden/intermediate、扫 M∈[63,2942]：910B 真机 13 个 M 值 × 3 个 op（39 组，CV 0.05%-4.2%）显示模型预测斜率系统性比实测陡 1.875×-3.005×，预测 intercept 远小于实测（实测曲线接近 launch-floor 主导的近平坦曲线）。固定 M（32/256）、扫真实 H 分布：910B 真机 6 组只有 2.5%-23.0% 变化（多为非单调噪声），模型同 6 组显示 45.2%-293.7% 的明显、基本单调增长。两条正交实测轴当时收敛到"模型把本该记入 launch/dispatch 固定开销的耗时按 `rows × hidden` 展开摊进变动项"这一根因假设，汇总影响估计为模型 6.040 µs/pf-tok vs 实测 2.536 µs/pf-tok（比值 2.3812×）。**这一估计的问题在于 M≥63 完全不是真实分布的代表样本**——见第 2 节。

## 2. §38 真实 M 加权重测：方向反转

用本 case qps64 真实调度轨迹回放（`vector_traversal_qps64.json`，623 条真实 KernelSpec，含 `occurrence_count`）核算，rmsnorm 真实质量 M=1 占 26.64%，M≤10 占 40.24%，M≤63 占 83.76%——第 1 节的 M≥63 采样域只覆盖 18.61% 真实质量。按真实质量 CDF 重新选 27 个 M 点（低 M 密集，覆盖 60.04% 质量）、H 固定为本 case 真值、每点 3 run×30 samples 重测，与 `vector_traversal_qps64.json` 里模型自身真实 `predict()` 输出逐点 join：

| op | 加权测得 (ns) | 加权模型预测 (ns) | ratio(测得/预测) |
|---|---|---|---|
| rmsnorm | 89681.6 | 3254.1 | 27.56×（欠计） |
| add_rmsnorm | 83283.7 | 3033.4 | 27.46×（欠计） |
| silu | 70650.7 | 5424.2 | 13.03×（欠计） |

真实测得值在 M=1 到约 M=700 区间几乎是一条平线（rmsnorm 80,940-93,200ns），模型预测在此区间仅 1,600-33,300ns。只有 M>700-1200 的稀有尾部（每点权重<0.5%）模型才追上甚至超过真机（如 silu@M=2942 ratio=0.610，方向翻转为过计）——这正是第 1 节 §35-37 发现的现象，但仅局限在稀有尾部。

**测量伪影排查**：怀疑测得平坦值是 `measure_current_stream_events` 每 sample 强制 `device_module.synchronize()` 产生的 host-dispatch/queue-restart 伪影（该 Goal 明确警告的 FA-SyncAll 类陷阱：强制同步把发射与队列重启开销计入测得值）。用该函数已有的 `operation_repetitions_per_sample`（同一 Event 区间内背靠背执行 N 次不插入 sync）在 M=1 验证：rmsnorm per-call 中位数从 reps=1 的 95,210ns 降到 reps=200 的 40,677ns（降 57%），silu 从 78,760ns 降到 33,346ns（降 58%），确认约一半是排空重启伪影。**但即便按伪影校正后的保守 floor（约 33,000-41,000ns）计算，仍比模型在真实主导 M 处的预测（1,600-3,000ns）大一个数量级以上——方向结论不受伪影校正影响。**

## 2.5 §39 精确定位交叉点：silu 单算子确有过计，但净复合交叉点在枢轴之外

第 2 节的"欠计 13×-44×"是按真实发射次数加权的聚合比值，把 M=1（decode 步，占 26.64% 权重，与 TTFT-Processing/prefill 无关）与 M=729-2942（prefill 大篮子，各仅 0.09% 权重但正是 TTFT-Processing 实际覆盖的区间）混在同一平均里，掩盖了大 M 区间的真实方向。用同一份 §38 已实测数据（无新采集）改按篮子分段复查：局部斜率（模型/实测）在 M∈[1247,2942] 尾段为 add_rmsnorm 1.489×、rmsnorm 1.858×、silu 2.415×，方向与本 Goal 症状一致；silu 单算子的 pred/meas 比值在 M≈1000-1200 处由欠转过（M=729 时 0.854，M=1247 时 1.165），与本 Goal 已确立的枢轴 ≈1009 token 惊人接近。但这只是 silu 单算子。把 rmsnorm+add_rmsnorm+silu 在同一 M 逐点求和（物理对应同一 step 同一层的三次真实调用）后，净复合 pred/meas 为 M=729 时 0.505、M=1247 时 0.708、M=2013 时 0.933、M=2942 时 1.102，线性插值得净复合交叉点 M≈2338.7——比枢轴（≈1009）大一倍以上。**在枢轴实际所在的 M~700-2000 区间，VECTOR 净复合仍然欠计（0.5-0.93），方向与本 Goal 的过计症状相反**；只有远超枢轴的稀有超大篮子（M≳2339，真实轨迹中每点权重≈0.09%）才会反超转为过计。按收益上限估算：即便完全修复 silu 的斜率误差，在枢轴相关区间该修复只会让净 VECTOR 欠计"更欠"（因 silu 原本部分抵消了 rmsnorm/add_rmsnorm 的欠计），对本 Goal 没有正向收益。故 silu 的斜率过陡虽真实、已用真机独立验证（方法同 §34 GEMM 排查），但确认（细化而非推翻）本文第 4 节的排除判定；完整推导见 `localization_round2.md` §39。

## 3. 已排除的候选（§35-37，含证据，第 1 节子域内仍然成立但已不再是本 Goal 的相关候选）

- **row-group setup/invocation 摊薄错误**：`_type_cann_norm_rows`（`src/kernel/predictors/vector.py:5967`）在生产 `pipeline_across_rows=True` 路径下，setup/invocation 固定开销按 worker 收一次、不随 rows 增长，摊薄机制结构正确，不是根因。
- **7 个 typed intrinsic rate 常数量级错误**：`cast_bf16_to_fp32_serialized`/`cast_fp32_to_bf16_serialized`/`muls_serialized`/`adds_serialized`/`mul_serialized`（`vector_serialized_services_cann_8_5_1.json` 及其 h8192 扩展标定）在真实 H 范围内验证误差 0.05%-8.1%；`rmsnorm_half_interval_reduce`/`add_rmsnorm_reduce_sum_fp32`（各自的 source-exact reduce 子算法复刻微基准，标定/验证覆盖 2048-12288 全部真实 H 档位）验证误差 0.001%-6.2%。全部 7 个 rate 常数来源合规且数值准确；问题不在任何单个常数取值。
- **"generic" 聚合速率是主导误差源**：`generic_per_row = 7*hidden+2` 通过架构规格来源（`configs/hardware/ascend_910b.yaml:195`，128 op/clk × 1.8GHz/AIV）的 230.4 Gops/s 计价，手算验证其在 H=8192、M=256 case 上贡献约 1.49µs/worker，仅占模型总预测 16.94µs 的一小部分。
- **"无重叠/流水线建模"**：`_attach_cann_rms_row_group_pipeline`（`vector.py:6575-6830`）已含显式多行重叠语义，排除了"模型完全没建模流水线重叠"这一朴素假设。

## 4. 为什么这不再是本 Goal 的候选（而不是 unsupported）

§37 曾判定"MTE2/MTE3 按字节过度计费"这一复合层面的候选是 G6（bounded unsupported）——即怀疑模型在 M≥63 子域内存在真实但尚未定位到具体代码行的过计机制。§38 表明，即使该 M≥63 子域内的过计现象（斜率过陡 1.875×-3.005×）是真实的，它在真实占用质量分布下也只影响<0.5%权重的稀有大 M 尾部，聚合净效应被主导区间（M≤63，模型欠计 13×-44×）完全反转和淹没。因此，**VECTOR row/hidden 复合耗时对本 Goal（910B qps64 TTFT Processing 过计）的贡献候选被排除**：无论 M≥63 尾部的斜率过陡机制最终是否被定位到具体代码行，它都不能解释、也不能在方向上贡献于本 Goal 的过计症状。

这与 §34 GEMM 腿 B 的排除方式同构：都是"真实硬件实测方向与假设相反"的决定性负结果，不需要（也不应该）动用 G6 unsupported 模板——G6 适用于"机制真实存在、方向正确、但证据不足以安全落地代码修复"的情形，本候选连方向都不成立。

## 5. 遗留：M≥63 尾部斜率过陡、以及主导区间欠计 floor，都是真实但与本 Goal 无关的现象

两个现象本身仍然是真实的、值得记录但**明确超出本 Goal 授权范围**的发现，留给未来独立 Goal：

- **M>700-1200 尾部斜率过陡**（§35-37 原始发现，本文第 1 节）：真实但权重<0.5%/点，不影响本 Goal。收口仍需第 3 节列出的、§37 已详细论证的两个前提（覆盖真实 H 全档位的 MTE2/MTE3 完成时间专用标定微基准 + 通读共享时序合成引擎复合规则），本文不再重复。
- **M≤700 主导区间的大幅欠计 floor**（§38 新发现，伪影校正后约 33,000-41,000ns vs 模型 1,600-3,000ns）：方向与本 Goal（过计）相反，不在本 Goal 授权范围内。该 floor 是否在真实 serving（host 连续背靠背发射、非本诊断用的孤立单次调用+强制同步）下也会暴露，取决于生产环境是否 host-dispatch-rate-limited——本轮未验证，留给后续独立 Goal（如涉及）判断是否值得开专门调查。

在此之前，本条残差（VECTOR row/hidden 复合过计对本 Goal 的贡献）按已排除处理并如实报告；`src/kernel/predictors/vector.py` 全程零改动。
