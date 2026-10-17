# Kernel 精度优化研究指南

本文记录 Kernel 精度优化过程中已经得到、并且对后续工作仍有决策价值的结论。它不是某次 session 的 handoff，也不按经历多轮 rebase 后的 Git 提交顺序复述历史；它按 golden 口径、建模抽象、硬件机制、被否定方向和下一步证据缺口组织，供后续 agent 在不重复走弯路的前提下继续优化。

本文中的数值快照对应解析模型提交 `a5961bb3e0c2c26b6a1fabedc36c525d55df484e`。解析模型的通用设计与证据等级仍以 [Kernel 解析性能模型](kernel_performance_model.md) 和根 `AGENTS.md` 为准；涉及 MetaX C500 采集、聚合或 profiler 时，必须先完整阅读 [MetaX C500 kernel profiling](metax_c500/profiling.md)。

本文的定量结论覆盖三种生产硬件上的 Vector family。工作方法可以迁移到 GEMM、FlashAttention 和 FusedMoE，但本文没有为这些 family 给出新的精度裁决，不能把 Vector 的 cache、pipeline 或 floor 结论直接套到其他 kernel kind。

## 1. 当前结论摘要

当前 Vector 诊断结果已经接近既定目标，但不能报告为正式精度门禁通过：三种硬件的 canonical duration 缺少 runtime、source snapshot 和 runner/operator binary identity，自动报告因此把所有 duration comparison 标记为 provenance-unverified diagnostic。下面的 MAPE 和 APE 可以用于排序问题、比较同口径迭代和判断回归，不能替代正式 comparison coverage。

| Hardware | Vector diagnostic MAPE | Diagnostic max APE | Mean signed error | 数值目标状态 |
|---|---:|---:|---:|---|
| NVIDIA A800 | 10.77% | 35.59% | -1.8% | 所有 Vector 物理桶 MAPE ≤ 15%，所有 case APE ≤ 40% |
| MetaX C500 | 11.81% | 39.43% | -0.3% | 所有 Vector 物理桶 MAPE ≤ 15%，所有 case APE ≤ 40% |
| Ascend 910B | 12.44% | 40.08% | -6.5% | 仍有一个 RMSNorm 桶和一个 SiLU case 越界 |

`Mean signed error = (prediction - observation) / observation`，负值表示平均低估。A800 和 C500 的整体偏差已经接近无偏，但不同桶仍同时存在高估和低估，不能再用一个全局倍率修正。910B 仍总体低估，残差集中在短任务的固定入口控制而不是流量或主 Vector 数学工作。

最终诊断报告的 workspace-local 路径是 `experiment_logs/kernel/mechanism_validation/vector_accuracy_first_mte3_admission_20260829/`；核心数字已经冻结在本文中，不应假定未版本化的 `experiment_logs/` 会永久存在。

## 2. 精度目标和正确读法

当前优化目标是每种硬件、每个 Vector 物理桶的 MAPE 不超过 15%，且每个 case 的 APE 不超过 40%。物理桶由 hardware、variant、source/launch execution signature 等静态事实定义，不是按误差临时分组。

评估时必须同时看四个量：桶 MAPE、桶内最大 APE、signed error 和绝对时间贡献。只按最大百分比误差排序会让极短 case 长期占据首位；只按绝对时间排序又可能遗漏系统性机制错误。实际推进顺序应先按目标中越界最严重的桶排序，再在同一层级参考绝对时间贡献。

每轮只允许用同一份 observation 和同一套 bucket schema 比较 before/after。若更换 golden、runtime、source、launch facts 或评估口径，必须建立新 provenance 并单独报告，不能把两套结果串成“模型提升”。

## 3. Golden timing 的裁决

### 3.1 单 kernel 应使用什么时间

单-kernel Forward 的 golden 应使用未插桩 profiler/trace 给出的逐 kernel device duration：A800 使用对应 CUDA profiler 的 kernel duration，C500 使用 mcTracer 的 uninstrumented kernel duration，910B 使用经 provenance 对齐的逐 kernel duration。目标是测真实 kernel invocation，而不是 host 提交、graph replay 或整个 task 的 wall time。

`vector_device_time.json`、`gemm_device_time.json`、`flash_attn_device_time.json` 和 `fused_moe_device_time.json` 是独立的任务级 device-event companion，不含逐 kernel 分解，不能覆盖或替代原来的 `vector.json`、`gemm.json`、`flash_attn.json` 和 `fused_moe.json`。它们可以用于算子级总时间诊断，但不能成为单-kernel predictor 的训练或验收 target。

A800 的 `VECTOR_ELEMENTWISE_T8_H8192_B1_MUL` 做过明确裁决：原 golden 为 2.560 µs，同次重采 warm kernel duration 为 2.528 µs，误差约 1.3%；cold kernel duration 为 3.296 µs；cold Event 为 8.192–9.248 µs，warm Event 为 10.240–19.168 µs。Event 的大部分时间来自边界和设备排队空档，不是真实 kernel 执行，因此短 Vector case 应相信原逐 kernel duration，而不是一次调用外包 Event 的结果。

### 3.2 为什么差分 Event 仍不能直接替换 golden

C500 曾采用 `Event(N 次 graph replay)` 和 `Event(2N 次 graph replay)`，再计算 `(T2N - TN) / N`。该差分可以抵消固定 Event、graph launch 和部分提交开销，其中固定项可写为 `F`；它也能放大短 kernel，改善计时分辨率。若每次 replay 使用独立输入且工作集大于 L2，它还是有价值的 graph-level mechanism measurement。

但差分结果仍测量 graph replay 合同，而不是逐 kernel trace 合同；多 kernel task 无法从总差分中唯一恢复每个 kernel 的 duration，极短调用还可能受 graph queue、同步和 replay schedule 影响。因此它只能作为独立辅助证据，除非未来明确修改 dataset schema 和 evaluation contract，否则不能替换单-kernel canonical golden。

### 3.3 Profiler counter 时间不是 golden

C500 mcProfiler 会 replay 或暂停 kernel，`MetaX.during` 比正常执行放大几十至上千倍。它只能提供 pipe duty、traffic、cache、work count 等 observation，不能作为 duration target。C500 canonical duration 必须来自同 workload 的 mcTracer；mcTracer 和 mcProfiler 不是同一次 invocation，必须按完整 launch identity 和 ordinal 严格关联。

对其他硬件也适用同一原则：插桩 counter 的 elapsed cycles、分域 AIC/AIV 时间或 pipe time 不能与另一轮未插桩 duration 混除得到“利用率”，更不能把这种比值写回 production Forward。

## 4. 已确认的建模抽象

### 4.1 模型是硬件单元依赖图，不是 SASS 模拟器

Vector 的可复用抽象是 `Submit/Init → Load → typed compute/scalar/reduction → Store/Barrier` 的资源与依赖图。工作量来自算子语义、compiler/operator source 和离散 launch；服务能力来自 HWSpec、架构资料和独立机制微基准；阶段通过串行、并行、流水和 worker makespan 组合。

不应逐条解析 SASS 或把某个已存在硬件的 instruction trace 当成模型输入。那样既不能覆盖不存在或尚未实测的硬件，也容易把 compiler 偶然实现固化为 per-kernel lookup。若 source 能静态确定某类操作次数，应映射到有物理意义的 typed service，例如 Vector、SFU、FMA、scalar、MTE2、MTE3、queue control 或 barrier，而不是保存完整指令序列。

`2.03 µs` 之类的数值是依赖图求解结果，不是原因。若 Add、Mul、ReLU 或 SiLU 的误差不同，首先检查读写次数、typed math、source launch、阶段顺序和 overlap，不能因为把 operation 改成 `add` 后数值接近就把 operation name 当成修正参数。

### 4.2 工作量和依赖必须分开

单纯把 operations、bytes 或 instruction-like counts 加全仍可能严重低估，因为 `max(bytes/bandwidth, ops/throughput)` 会把真实串行链全部重叠。相反，把整张 tensor 的 read、compute、write 全部串行又会高估，因为不同 tile/CTA 可以流水。

正确做法是先守恒总工作量，再从 source 恢复最小必要阶段和依赖；对可流水的 tile/CTA 使用 worker program 或 max-plus pipeline，对真正的 scalar handoff、reduction barrier、queue ownership 和 first-request completion 保留串行边。

### 4.3 Floor 必须是机制，不是常数

“短任务 floor”只有在其物理机制和作用域明确时才合法。它可能随 grid、block、active worker、请求大小、queue depth、数据路径或 source round 数变化，不能用一个按硬件或算子名固定的常数覆盖全部 shape。

C500 的已验证 memory-dependency task measurement 描述同一 Event timer domain 中一个完整 task body 的非空增量。独立 CTA 之间没有 source-order edge，因此这个完整 task body 在 critical path 上只收费一次；grid throughput 和并发由资源模型承担，不能再按 CTA wave 数重复乘 floor。

910B 的 first-MTE completion 也是 exact mechanism：它按 worker admission domain 和 payload bytes 查表，只绑定 source 中第一个相应请求，不能扩展成所有 MTE 请求的统一 floor。

## 5. 跨硬件共同结论

1. 先从当前误差贡献最大的 Vector 桶开始，但不需要把一个桶一直做到目标才看其他桶；每轮应重新排序三硬件的最大残差，优先修复可复用、物理证据最强的机制。
2. 不要一开始就重采全量数据。先用现有 launch、source、traffic 和 counter 判断缺的是工作量、依赖、并发还是 golden 口径；只有缺少一个可命名的独立硬件机制时才设计微基准。
3. 三硬件可以复用 dependency/pipeline 抽象，不能强行复用具体 service。A800/C500 都是 GPU-style CTA/warp 路径，但 L1 和 source schedule 不同；910B 是 AIV/MTE/UB/TQue 路径，需要不同 HWSpec service。
4. Traffic 对齐不等于 duration 对齐。最终报告中 910B Vector 的 MTE2、MTE3、DRAM 和 L2 traffic diagnostic MAPE 分别约 0.6%、0.4%、2.8% 和 1.2%，但短 RMSNorm/SiLU 仍低估，说明剩余问题主要在控制与依赖。
5. Frequency 不是通用解释。C500 XCORE 在相关 replay 中稳定为 1.125 GHz；其 SiLU 残差来自低 active-AP、长 Average Wave Life、DNoC latency 和 VLS stall 所指向的依赖等待，而不是频率漂移。
6. HWSpec 的 cache 开关必须反映真实硬件路径。A800 `l1_cache_enabled: true`，模型允许 reduction 后的输入/残差在 L1 驻留；C500 `l1_cache_enabled: false`，重复 source request 必须继续到 L2，不能套用 A800 的 L1 reuse。

## 6. NVIDIA A800 结论

### 6.1 已接受机制

- 普通 Elementwise 使用 streaming dependency path：各 tile 的 global read、算术和 global write 可以流水，不能建模成整张 tensor 的三段全局 barrier。
- AddRMSNorm 使用 source CTA pipeline，并显式表达第一遍 add/residual store、reduction、第二遍 normalize/store。A800 的 L1 能保留 reduction 后需要复用的输入/残差，第二遍不能重复按 cold DRAM 读取。
- RMSNorm reduction 必须绑定 vLLM SM80 source 的真实 reduction pipeline；只算向量元素 FLOPs 会漏掉 reduction dependency，只加一个完整 kernel floor又会双算。
- SiLU 的 full CTA 和 tail CTA 具有不同的有效读并发；tail 不能按完整 CTA work 或完整并发能力处理。
- RoPE、KV cache 和 elementwise 的 source work、metadata bytes 与 launch geometry 必须进入同一 dependency model，不能用统一“Vector efficiency”。

### 6.2 最终诊断桶

| Variant | 最大桶 MAPE | 最大 case APE | 结论 |
|---|---:|---:|---|
| Elementwise | 13.96% | 27.65% | 通过数值目标，整体仍略低估 |
| RMSNorm | 9.94% | 35.42% | 通过；L1 reuse 与 reduction pipeline 都必要 |
| AddRMSNorm | 12.56% | 31.62% | 通过；不同 CTA class 不能合成一个经验常数 |
| RoPE | 12.91% | 35.59% | 通过；该桶贡献最终最大 APE |
| SiLU | 7.41% | 24.02% | 通过；tail read concurrency 是关键修复 |
| Paged KV cache | 7.08% | 19.94% | 通过 |

早期保留的同口径快照为 MAPE 13.2%、max APE 54.0%；最终为 10.8% 和 35.6%。主要收益来自 source streaming、Add/RMS pipeline、SiLU tail 和正确的 cache/dependency，而不是按 case 调参。

### 6.3 不要重复的方向

- 不要使用 device Event 的 8–19 µs 去“修复”约 2.5 µs 的短 kernel；A800 adjudication 已证明这是边界/排队污染。
- 不要把所有 Vector 算子统一加同一个 launch floor；不同 source pipeline 已能解释大部分差异。
- 不要把 resident CTA 数乘到物理 compute pipe 数；occupancy 提供 latency hiding，不复制执行单元。

## 7. MetaX C500 结论

### 7.1 环境和硬件边界

当前 source/launch 合同绑定已验证的 MACA 3.7.2。3.8.2 或其他 runtime 不能被视为等价环境；若 profiler、vLLM 和算子库不能组成同一已证明版本，必须停止并记录环境缺口，不能把另一版本结果混入评估。

C500 Vector L1 在当前架构合同中关闭。配置中的 nominal 32 KiB 只描述结构容量，production launch 会移除 L1 demand；RoPE 的重复 cos/sin、position request 以及 Norm 的重复读取会到 L2。A800 的 L1 机制不能直接复制到 C500。

### 7.2 已接受机制

- 通用 Vector path 使用 source-bound dependency task，而不是 `bytes / peak bandwidth` 加固定 overhead。
- packed BF16 SiLU 按 source loop 建模：每轮加载两个 128-bit operand，执行 BF16 conversion、Exp、reciprocal、multiply，再 store；不同 CTA 和同 CTA 内 round 分别处理并发和串行。
- RoPE 必须计入 Q/K loop、position/cos/sin 路径、BF16 round trip、integer div/rem 和 address multiply-add；L2 处理重复 source request，DRAM 只处理唯一驻留 footprint。
- AddRMSNorm 的 resident stages 应交错执行，不能把所有 CTA 的第一遍、reduction、第二遍全局串行。
- RMSNorm 在 L1 disabled 条件下需要重启 CTA request chain；不能沿用 A800 post-reduction L1 resident 假设。

C500 SiLU 的 source work 已被 profiler observation 支持：指令/工作量 diagnostic MAPE 约 0.000015%，global traffic MAPE 约 0.89%。早期仅按工作量除吞吐仍系统低估，是因为没有表达 dependency pipeline；mcProfiler 的 active AP、wave life、DNoC latency 和 VLS stall 只用于证明“存在等待”，不能直接回归成 production efficiency。

### 7.3 最终诊断桶

| Variant | 最大桶 MAPE | 最大 case APE | 结论 |
|---|---:|---:|---|
| Elementwise | 11.88% | 20.52% | 通过，signed bias 接近零 |
| RMSNorm | 12.13% | 39.43% | 通过但贴近 case 边界，优先防回归 |
| AddRMSNorm | 13.28% | 29.02% | 通过；存在高估和低估抵消，不能只看总体 bias |
| RoPE | 9.60% | 28.32% | 通过；conversion/address chain 是必要机制 |
| SiLU | 12.96% | 34.64% | 通过；短 T1 子桶约 1.87% |
| Paged KV cache | 11.74% | 26.32% | 通过 |

早期保留的同口径快照为 MAPE 14.6%、max APE 47.0%；最终为 11.8% 和 39.4%。C500 的整体 signed bias 已到 -0.3%，后续不应添加任何全局 correction。

### 7.4 不要重复的方向

- 不要把 mcProfiler `during` 当 duration，也不要把 instrumented cycles 除以另一轮 mcTracer 时间。
- 不要用固定频率误差解释 SiLU；相关 capture 的频率已经固定为 1.125 GHz。
- 不要给所有 grid/shape 添加相同 floor；task mechanism 的作用域是一次 complete non-empty task increment。
- 不要根据 kernel name、文件名或 duration 相似度关联 mcTracer 与 mcProfiler；必须按 PID、marker window、launch signature 和 ordinal。

## 8. Ascend 910B 结论

### 8.1 已闭合的主链

910B Vector 不能沿用 GPU CTA 模型。当前模型按 AIV source program 显式表示 GM→MTE2→UB/TQue→Vector/Scalar/Reduction→MTE3→GM，并在不同 CANN tiling key 下选择不同 worker、row-group 和 split-D 程序。

已闭合机制包括：typed tiling materialization、SwiGLU repeated tiling payload、Elementwise tile pipeline、RMSNorm/AddRMSNorm source row DAG、半区间 reduction、V→S/GetValue/rsqrt/SetValue/S→V scalar chain、queue/TBuf lifecycle、queue control cycle、RoPE local UB copy 与 scalar mask setup、KV slot lookup、Vector-to-MTE2 fence，以及一/二 worker 和 32/43/47 worker admission domain 的 first MTE completion。

RMSNorm key-0 当前已覆盖 `GET_TILING_DATA`、四个 TQue/三个 TBuf 初始化、完整四队列生命周期、BF16 half-interval reduce、local scalar dependency、MTE2/MTE3 completion。AddRMSNorm key-30 还覆盖第一遍 residual store、FP32 ReduceSum、local scalar handoff 和 Vector-to-MTE2 fence。不要再把这些 service 整体加一次。

### 8.2 First-MTE admission 的准确作用域

32、43 和 47 worker 的 first-MTE2/first-MTE3 service 是独立机制微基准，不消费 production/evaluation duration，并采用 exact-only payload lookup。当前发布值如下，单位均为 ns/request/AIV：

| Direction | Workers | 4096 B | 5120 B | 8192 B | 10240 B | 16384 B |
|---|---:|---:|---:|---:|---:|---:|
| MTE2→Vector | 32 | 430 | 440 | 440 | 440 | 460 |
| MTE2→Vector | 43 | 780 | 740 | 800 | 820 | 880 |
| MTE2→Vector | 47 | 890 | 920 | 930 | 1000 | 1060 |
| Vector→MTE3 | 32 | 未发布 | 240 | 200 | 250 | 260 |
| Vector→MTE3 | 43 | 700 | 700 | 680 | 690 | 780 |
| Vector→MTE3 | 47 | 880 | 800 | 810 | 860 | 890 |

32-worker、4096 B 的 MTE3 cell 因跨轮 median spread 为 23.08%，超过冻结的 20% publication limit，必须保持 unsupported，不能插值或拿邻近 payload 代替。

RMSNorm 只把 row-group 第一组 `row0_output_store` 绑定 first-MTE3；AddRMSNorm 只把第一组 `row0_residual_store` 绑定 first-MTE3。后续 row 和 terminal rstd store 不在这个 exact service 的作用域内。

96 B terminal rstd store follow-up 曾得到稳定诊断值：32 workers 三轮均约 180 ns，43 workers 约 610/620/640 ns，47 workers 约 700/690/660 ns。但它尚未形成合法的 production composition；把它附到每个 row group 会双算或改变 critical path，因此当前不接入 Forward。此前“给所有 terminal rstd store 加 completion”的实现已被完整撤销，不应恢复。

### 8.3 最终诊断桶

| Variant | 最大桶 MAPE | 最大 case APE | 状态 |
|---|---:|---:|---|
| Elementwise | 12.18% | 30.14% | 通过数值目标 |
| RMSNorm | 16.83% | 35.57% | 桶 MAPE 越界；问题集中在 key-0 短任务 |
| AddRMSNorm | 13.63% | 30.87% | 通过数值目标 |
| RoPE | 14.61% | 28.89% | 通过但接近桶边界 |
| SiLU | 11.03% | 40.08% | 桶通过；`VECTOR_SILU_T1_I6144_B1` 单 case 轻微越界 |
| Paged KV cache | 14.18% | 36.88% | 通过但接近桶边界 |

早期保留的同口径快照为 MAPE 22.6%、max APE 67.1%；在 source pipeline、tiling、scalar/reduction、queue 和 MTE admission 逐步接入后，最终为 12.4% 和 40.08%。最后一轮 first-MTE3 admission 将 910B Vector 从约 13.2%/52.2% 降到 12.4%/40.08%。

### 8.4 剩余 RMSNorm 根因候选

对短 RMSNorm 的 source audit 没有发现遗漏的 tiling、queue/TBuf、reduction、scalar rsqrt 或 MTE 主路径。唯一明确存在但尚未表达的是每个 AIV 的 `Init` 标量入口与行地址分配：tiling field load、`GetBlockIdx/GetBlockNum`、tail compare/select、`blockIdx * blockFactor * hidden` affine arithmetic，以及 x/gamma/y/rstd 的 `SetGlobalBuffer` descriptor construction。

这个机制对 T1/T2 不会被行循环摊薄，方向上能解释短任务低估，但当前没有独立的 u32 affine/address setup service，不能根据 20–35% 残差反推一个常数。实施前还必须在 910B CANN 8.5.1 容器复核 T1/T2 实际走 `rms_norm.h::KernelRmsNorm` 还是 `rms_norm_single_row.h` 的另一套 `Init`。

现有 `aiv_scalar_u64_divrem_dependency` 不能复用于 RMSNorm：它只对应 SwiGLU 的 64-bit quotient/remainder，而 RMS row allocation 是 32-bit affine mul-add、比较和 descriptor setup。可以新建可复用的 “AIV entry affine-address setup” primitive，但必须用独立微基准给出有单位的 service，再由 RMSNorm 和 SwiGLU source 分别静态计数。

### 8.5 剩余 SiLU 根因候选

SwiGLU 当前已经覆盖 typed tiling materialization、u64 div/rem、三个 queue/两个 TBuf lifecycle、逐 tile queue control、MTE2/Vector/MTE3 主路径。仍缺的最小静态机制是每 tile 的地址计算和 full/tail 参数选择：三个 GM offset、tail branch、`curCalLen` 以及 copy-parameter selection。

该缺口与 RMSNorm 的入口 affine-address setup 有可复用子集，但作用次数不同：RMS 主要是每 worker 一次入口，SwiGLU 还包含每 tile 的地址和 full/tail 控制。后续应先建立共享 scalar-control primitive，再由 source 静态计算 invocation count；不要创建 `silu_t1_floor` 或按 intermediate width 查时延。

### 8.6 Queue 顺序的边界

当前 RMSNorm 把完整 queue control chain 较粗地放在 gamma load 前，总工作量已覆盖，但真实源码中的 Allocate/EnQue/DeQue/FreeTensor 分散在多个阶段。重复叠加完整 queue service 一定双算；若以后需要改善顺序，只能把独立 queue operation 的机制微基准拆到真实依赖位置，而不是增加第二份 lifecycle。

## 9. 被否定或未获证据支持的方向

| 方向 | 结论 |
|---|---|
| 按目标 kernel duration 拟合固定 overhead、斜率、倍率或 shape floor | 违反零拟合原则；即使 MAPE 下降也不得进入 Forward |
| 为每个 Vector variant 保存经验 latency LUT | 不可迁移到新硬件，且等价于 per-kernel 拟合 |
| 解析 SASS 并逐指令模拟 | 不符合所需 analytical abstraction；compiler/hardware 不存在时无法工作 |
| 只增加 operations 或 bytes | 无法表达 reduction、scalar handoff、first-request 和 queue dependency，通常仍低估 |
| 把整张 tensor 的 Load/Compute/Store 全串行 | 丢失 tile/CTA overlap，通常高估 |
| 把一个 task floor 乘 CTA waves | complete-task mechanism 已含一次 task body，重复收费 |
| 用 C500 固定频率解释残差 | capture 已显示频率固定，等待来自依赖和并发 |
| 用 A800 L1 reuse 解释 C500 | C500 Vector L1 在当前合同中关闭 |
| 用 Event companion 替换逐 kernel golden | 短 kernel 已证明会混入 Event/queue 边界，且 task total 无法唯一分解 |
| 把 910B terminal rstd completion 加到每个 row group | service scope 不匹配并会双算，相关实现已撤销 |
| 对 32-worker 4096 B MTE3 做邻值插值 | 该 cell 稳定性门禁失败，必须保持 unsupported |
| 根据总体 signed bias 加全局 correction | A800/C500 已近无偏，桶间正负残差会被掩盖 |

## 10. 证据和代码导航

| 目的 | Canonical 位置 |
|---|---|
| 解析模型原则、公式和证据等级 | `docs/kernel_performance_model.md` |
| Vector workload → demand/dependency program | `src/kernel/predictors/vector.py` |
| 三硬件离散 launch selector | `src/kernel/launch_config/{nvidia_a800,metax_c500,ascend_910b}/vector_launch.py` |
| 硬件 service、cache 开关和证据引用 | `configs/hardware/{nvidia_a800,metax_c500,ascend_910b}.yaml` |
| 910B 可发布独立机制证据 | `microbench/ascend_910b/references/*.json` |
| A800 RMSNorm source/reduction 证据 | `microbench/nvidia_a800/references/vllm_rmsnorm_v0_11_0_sm80.json` |
| C500 profiler/golden/provenance 规则 | `docs/metax_c500/profiling.md` |
| dependency、exact service 和 worker composition 测试 | `tests/kernel/test_traffic.py`、`tests/kernel/test_hardware_spec.py`、`tests/kernel/test_launch_config.py` |

版本化 reference JSON 是机制发布入口；`profiler_output/` 是 raw evidence，`experiment_logs/` 是评估结果。后续 agent 不应从 raw 文件名或某次报告摘要直接增加 HWSpec 参数，必须先检查 reference 的 source snapshot、binary SHA、raw SHA、稳定性 gate、单位、作用域和 `production_or_evaluation_duration_consumed: false`。

## 11. 后续 Kernel 精度优化流程

1. 冻结精确 commit、canonical dataset SHA、hardware config SHA、launch data SHA 和 evaluation 命令；若 comparison provenance 不完整，先明确结果只具 diagnostic 资格。
2. 运行完整三硬件报告，按“越界桶 → 最大 case → 绝对时间贡献”排序。不要只打印 overall MAPE。
3. 对最高优先级残差先做 source/launch audit，列出现有模型已覆盖和未覆盖的阶段；没有明确缺失机制时不要启动硬件采集。
4. 检查 traffic、typed work、worker distribution 和 stage dependency 四个层次。若 traffic/work 已对齐，优先找 scalar、barrier、queue、first-request、tail 或 admission；不要继续堆 bytes/FLOPs。
5. 只有发现可命名且可跨 workload 复用的机制缺口，才设计最小微基准。微基准必须避免复刻完整 production operator，并输出有单位的硬件参数及稳定性范围。
6. Forward 中只消费 source/launch 静态计数和 workload-independent service。无法证明的 shape 保持 limitation/unsupported，不插值、不 fallback。
7. 每次修改后先跑受影响 contract 测试，再跑同一三硬件评估，报告目标桶、全 Vector 和其他硬件的 before/after；若一个桶改善却使另一个已通过桶越界，应回到机制作用域而不是继续调值。
8. 达到数值目标后仍要检查 formal comparison coverage、support coverage 和 provenance。Diagnostic pass 不能写成正式 goal complete。

对当前 910B Vector，下一步优先级已经足够明确：先复核 RMS T1/T2 source variant，并独立刻画 AIV entry affine-address setup；随后验证这个 primitive 是否能以 source-count 方式同时覆盖 RMSNorm 和 SwiGLU per-tile address/full-tail control。除此之外，当前没有证据支持再增加新的 Vector floor、倍率或 duration-derived parameter。
