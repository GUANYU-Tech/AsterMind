# Midlayer Feature 支持进度

> 最后更新：2026-09-04。本文是当前活动进度表，由项目维护者持续更新。设计边界见 `serving_system.md` 与 `kernel_performance_model.md`，强制规则只见根目录 `AGENTS.md`。

状态说明：

- **✅ 已完成**：该条目声明的基础范围已经进入主链并完成仓库内验证；不表示任意组合均支持。
- **🟡 部分闭合**：已有实现或验证切片，但仍有明确的语义、组合、证据或精度缺口。
- **⬜ 未完成**：尚未形成可用的生产主链能力。

具体 case 数、MAPE、commit 和一次性诊断不在本表维护，应由 evaluation report、issue 或 Git 历史保存。

## 第一部分：当前支持目标

### 计算算子

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| 公式重新统一设计和验证 | ✅ 已完成 | 统一解析框架、零拟合合同和基础仓库验证已完成；分算子精度验收单独跟踪 |
| 静态 GEMM 仿真 | ✅ 已完成 | 三种生产硬件均进入统一预测与 evaluation 链 |
| 静态 Attention 仿真 | ✅ 已完成 | 标准、paged/varlen、split-KV 和部分 MLA 形态进入统一链路 |
| 静态 MoE 仿真 | ✅ 已完成 | routing、expert ownership 和 tiling 进入统一链路 |
| 静态 Vector 仿真 | ✅ 已完成 | Norm、RoPE、Activation、Elementwise、KV-cache 分别建模 |
| Serving GEMM 仿真基础 | ✅ 已完成 | Serving lowering 复用统一 Kernel predictor |
| Serving Attention 仿真基础 | ✅ 已完成 | 实际 Q/KV 长度、causal、layout 与 split 进入 KernelSpec |
| Serving MoE 仿真基础 | ✅ 已完成 | routed rows、rank-local ownership 与 expert workload 可显式输入 |
| Serving Vector 仿真基础 | ✅ 已完成 | Serving 主链覆盖基础 Vector 类别 |
| GEMM 跨硬件精度验收 | 🟡 部分闭合 | 三硬件均可评估，部分 launch、split/reducer 和 worker/cache 证据仍未闭合 |
| Attention 跨硬件精度验收 | 🟡 部分闭合 | 数据语义已统一，部分真实 launch、pipeline、scratch residency 和硬件变体仍需验收 |
| MoE 跨硬件精度验收 | 🟡 部分闭合 | C500 cache/worker、910B transfer/credit 等机制仍缺完整证据 |
| Vector 跨硬件精度验收 | 🟡 部分闭合 | 部分静态 issue、transaction、worker 与短任务机制仍需补齐 |

### 通信算子

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| ProtoBridge 通信仿真器接入 | 🟡 部分闭合 | 已接入 SHA 锁定的上游 white-box surrogate 与 FullMesh；C500 FC2→FullMesh、C500 Tree→Tree、910B Mesh→FullMesh 已由 versioned selector 校验并接入 development-only profile。尚无作者提供的 A800/C500/910B production profile，因此结果仍不可用于精度、发布、硬件比较或 DSE 结论 |
| Collective 通信 | 🟡 部分闭合 | all-reduce、all-gather、all-to-all 已进入统一 workload/timeline；拓扑和硬件验证仍有限 |
| P2P 通信 | 🟡 部分闭合 | pipeline、activation、KV transfer 可表达为 P2P；跨硬件链路合同仍不完整 |
| 通信竞争与计算重叠 | 🟡 部分闭合 | 由依赖和资源身份决定重叠；更多 topology/fabric contention 仍需验证 |
| 通信算法选择 | 未完成 | 根据具体消息大小获得实际通信算法 |

### Offload

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| Ramulator Host DRAM 性能仿真 | ⬜ 未完成 | 尚未接入生产时间线 |
| Host DRAM KV cache offload | 🟡 部分闭合 | 已有 LOAD/EVICT、容量账本、PCIe interval、completion 与 isolated `L + bytes/B` Host DRAM service-demand backend；该 backend 尚未接入统一 ExecutionTimeline，仍缺原生 victim/LRU、prefetch、Host DRAM 排队/竞争和完整重叠 |

### Serving 框架与 Trace

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| Serving 接入框架优化重写 | ✅ 已完成 | 请求、Scheduler、logical workload、Kernel lowering、timeline 与 metrics 已按 owner 分层 |
| vLLM Scheduler 正式接入 | ✅ 已完成 | 锁定版本的 CPU-only Scheduler 已进入主链 |
| vLLM Scheduler 接入后调度结果对齐 | 🟡 部分闭合 | continuous batching、chunked prefill、KV、finish、preemption/recompute 有明确验证切片；不代表任意配置 |
| vLLM Scheduler CPU 开销建模 | ⬜ 未完成 | 当前真实 Scheduler 只决定 batch、KV 与请求状态，`schedule()` 及相关状态更新消耗的 CPU 时间尚未推进仿真虚拟时钟；后续需用独立、可复测的 Host runtime microbenchmark 建立有物理单位的机制参数，按 step 计入并与 forward 内部 Host 发射开销分开，Serving 实测时间只用于归因与 evaluation，不得用于反推参数 |
| Synthetic Trace | ✅ 已完成 | 可生成与回放合成请求流 |
| Production Trace 接入并根据信息合成 | 🟡 部分闭合 | allowlist reader、确定性 replay、request-safe aggregate 和 strict raw-to-compact converter 已接入；8 个 Production 请求 case 已发布到 Git LFS，因缺少模型与运行 setting 不发布 alignment，MoE/MTP 动态输入仍缺真实证据 |
| Prefix cache 不同命中率分析 | 🟡 部分闭合 | 支持逻辑命中长度与 block 语义；缺少 cache identity、真实淘汰和实例 affinity |
| MTP 支持 | 🟡 部分闭合 | propose/verify、acceptance replay、fixed/maximum-entropy assumption 与 handoff 有切片并绑定 provenance；真实 acceptance、完整模型执行和实机验证未闭合 |
| Arrival-rate sweep | ✅ 已完成 | Production Trace scale、指标和报告链已接入 |

### 多实例

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| 多 P 实例 | 🟡 部分闭合 | 多 P 路由、独立 Scheduler/KV owner 有明确切片；不是任意部署 |
| 多 D 实例 | 🟡 部分闭合 | 多 D 路由与 handoff 有明确切片；不是任意部署 |
| `least_queued_tokens` 最轻负载路由 | ✅ 已完成 | ingress→P 与 P→D 可按可见队列和 reservation 决策 |
| 多实例关键路径分析（P/D/KV 传输/排队等） | 🟡 部分闭合 | 已覆盖 Router、Scheduler、KV transfer 与 Kernel interval；复杂异构和共享资源仍需扩展 |

### 部署方式

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| PD 共置基础支持 | ✅ 已完成 | 基础 colocated 路径已进入主链 |
| PD 共置投稿级验证 | 🟡 部分闭合 | 有仓库内 gate，尚未完成完整生产实验验收 |
| PD 分离基础支持 | ✅ 已完成 | P→D handoff、单次 KV transfer 和状态连续性已闭合 |
| PD 分离投稿级验证 | 🟡 部分闭合 | 多 P/D 有窄切片，尚未完成完整生产实验验收 |
| AF 分离 | 🟡 部分闭合 | standalone AF 与 PD+AF 有明确切片；AF data plane 不是框架原生完整实现 |

### 并行方式

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| TP 基础支持 | ✅ 已完成 | shape/rank 切分与 collective 有明确切片 |
| TP 投稿级验证 | 🟡 部分闭合 | 已有仓库内验证，生产组合覆盖仍不足 |
| EP 基础支持 | ✅ 已完成 | expert ownership、routing 和 rank-local MoE workload 已接入 |
| EP 投稿级验证 | 🟡 部分闭合 | 仍需跨模型、跨硬件和生产 workload 验收 |
| PP 支持与投稿级验证 | 🟡 部分闭合 | PP 与 TP×PP 有明确切片；通用 pipeline bubble/overlap 和投稿级验证未完成 |

### 模型支持

当前模型接入目标为 GLM-5.2 与 Kimi-K2.6；后续模型不属于本阶段完成条件，统一记录在第二部分。

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| Dense 模型基础支持 | ✅ 已完成 | Dense decoder 主链已闭合 |
| Dense 模型投稿级验证 | 🟡 部分闭合 | 有仓库内切片，生产实验覆盖仍不足 |
| MoE 模型基础支持 | ✅ 已完成 | MoE、expert routing 与 EP 基础已闭合 |
| MoE 模型投稿级验证 | 🟡 部分闭合 | MTP、复杂 routing 和跨硬件生产实验仍需补齐 |
| GLM-5.2 Serving 支持 | 🟡 部分闭合 | 固定配置、vLLM 0.27.1 固定执行语义、A800 Transformers 5.10.2 tiny BF16 验证，以及 pinned-vLLM 0.11 CPU Scheduler→production trace→TP=EP=8/PP4 DSA/indexer/compressed-KV/FP8-index-cache/MoE/MTP proposer-verifier→KernelSpec→解析预测→时间线主链已闭合；MTP proposal depth、maximum-entropy mean acceptance rate、source ID 与 root seed 均显式配置，结果标记为 projected assumption；32-rank placement、DSA-safe `[18,20,20,20]` layer partition、stage-local 权重/KV/MTP residency、节点内 TP collective 与跨节点 8-lane PP P2P 均显式建模；严格相同的跨 rank KernelSpec 共享 lowering/predictor 结果，external trace 默认逐 step 消费并释放完整 IR；新增 kernel 仅有零拟合、未校准的 A800 projected analytical 结果，vLLM GPU execution 和三硬件实机精度仍 unsupported |
| Kimi-K2.6 Serving 支持 | 🟡 部分闭合 | 固定配置、vLLM 0.27.1 固定执行语义、A800 Transformers 5.10.2 tiny BF16 验证，以及 CPU Scheduler→production trace→TP=EP=8/PP2 MLA/压缩 KV/INT4 routed-MoE→KernelSpec→解析预测→时间线主链已闭合；16-rank placement、stage-local packed INT4+BF16 scale 权重/KV 容量、节点内 TP collective 与跨节点 8-lane PP P2P 均显式建模；严格相同的跨 rank KernelSpec 共享 lowering/predictor 结果，external trace 默认逐 step 消费并释放完整 IR；MLA、BF16 router 和 symmetric INT4 W4A16 Marlin 路径当前仅有零拟合、未校准的 A800 projected analytical 结果，vLLM GPU execution 和三硬件实机精度仍 unsupported |

### 精度支持

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| BF16 基础支持 | ✅ 已完成 | 三种生产硬件主路径支持 BF16 语义 |
| BF16 端到端验证 | 🟡 部分闭合 | 当前提交的三硬件单 Kernel 全矩阵通过；27 个 ShareGPT physical case 已完成数据与评估合同验证，但三硬件分别受 GEMM、Vector gather 和动态 prefill FlashAttention launch coverage 阻塞，尚无可报告的 Serving 精度值 |

### 异构部署

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| 不同实例使用不同硬件的分析（LPU 等） | ⬜ 未完成 | 当前实例与链路合同不足以支持通用异构部署结论 |

### 硬件支持

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| NVIDIA A800 production Kernel/Serving 基础 | ✅ 已完成 | 统一接口、canonical dataset 和受限 Serving 切片已接入 |
| NVIDIA A800 投稿级 Serving 验证 | 🟡 部分闭合 | 9 个 ShareGPT compact case 已进入统一 evaluator；动态 batch M 的 GEMM measured launch facts 不完整，当前 9 个均 fail loud |
| MetaX C500 production Kernel/Serving 基础 | ✅ 已完成 | 统一接口、canonical dataset 和受限 Serving 切片已接入 |
| MetaX C500 投稿级 Serving 验证 | 🟡 部分闭合 | 9 个 ShareGPT compact case 已进入统一 evaluator；MACA 3.7.2 `gather` Vector launch fact 不完整，当前 9 个均 fail loud |
| Ascend 910B production Kernel/Serving 基础 | ✅ 已完成 | 统一接口、canonical dataset 和受限 Serving 切片已接入 |
| Ascend 910B 投稿级 Serving 验证 | 🟡 部分闭合 | 9 个 ShareGPT compact case 已进入统一 evaluator；对应动态 prefill FlashAttention workload 缺 launch fact，当前 9 个均 fail loud |
| NVIDIA H200 production Kernel/Serving 基础 | 🟡 部分闭合 | canonical `kernel_dataset/nvidia_h200`（3728 case，vLLM 0.30.0 NSYS launch facts，无 Nsight Compute counter，故不进 traffic 表）与 `serving_dataset/sharegpt/nvidia_h200`（15 个 vLLM 0.30.0 ShareGPT case）已发布；Kernel Forward ALL 18.98%（计分 3298，显式 unsupported 430：424 个非 MLA FA3、6 个持久化 split-K GEMM）。已知缺口：GEMM tile 选择仍沿用 A800 launch 表而非 H200 实测 launch facts；split-K reducer 沿用 A800 的 partial 不落 DRAM 假设；vLLM 0.30 默认 tile 类别的 grouped MoE 以聚合路径定价并强制带 `moe_gather_service_class_not_separated`；device launch gap、4/8 卡 NCCL 协议表未实测（H200 已回收）；vLLM 0.30 的 request path、runtime prepare 与 Scheduler prefill 线借用 0.11 实测并在 reference 中以 `substitution` 注明（用户豁免）；vLLM 0.11.0 采集为非 canonical 候选 |
| DeepSeek-R1 H200 SGLang 对齐 | 🟡 部分闭合 | SGLang 0.5.12 Scheduler 镜像、flashinfer MLA、EAGLE V2、融合 AR+RMSNorm 与 SGLang DSV3 router launch 选择（`launch_data/nvidia_h200` schema v4）已接入，对照 InferenceX 8K/1K TP8；逐并发误差门 G1 由用户豁免，残差载体为服务中约 0.2 µs/node 的未定位图节点间隙 |
| NVIDIA B 系列硬件 | ⬜ 未完成 | 尚未作为 production hardware 接入 |
| AMD MI 系列硬件 | ⬜ 未完成 | 尚未作为 production hardware 接入 |

### 系统指标与分析

| Feature | 状态 | 当前范围与缺口 |
|---|---|---|
| TTFT 基础指标 | ✅ 已完成 | 从统一请求时间线计算 |
| TTFT CDF、P0～P100 与校验 | 🟡 部分闭合 | 分布与 percentile 已实现；完整生产校验未完成 |
| TPOT 基础指标 | ✅ 已完成 | 按 eligible request 计算 |
| TPOT CDF、P0～P100 与校验 | 🟡 部分闭合 | 分布与 percentile 已实现；完整生产校验未完成 |
| E2E latency 基础指标 | ✅ 已完成 | 从 arrival 到 completion 计算 |
| Request throughput 基础指标 | ✅ 已完成 | 已进入统一 metrics/report |
| Token throughput 基础指标 | ✅ 已完成 | 已进入统一 metrics/report |
| Request SLO goodput | ✅ 已完成 | 支持显式 SLO contract |
| Token SLO goodput | ✅ 已完成 | 支持显式 SLO contract |
| 不同实例的利用率分析 | 🟡 部分闭合 | 可按实例/资源统计；复杂共享硬件仍未覆盖 |
| Router queue 与 Scheduler queue | ✅ 已完成 | 两类等待分别进入时间线和报告 |
| 关键路径与瓶颈归因 | 🟡 部分闭合 | 已覆盖当前事件和资源类型；复杂异构资源仍需扩展 |
| Interactivity-TGS 分析 | ✅ 已完成 | 已接入 arrival-rate sweep 与版本化报告合同 |

## 第二部分：当前支持目标闭合后需要探究的问题

本节记录第一部分开发与投稿级验证完成后需要回答的研究问题。它们不是当前 Feature 完成条件，也不是可以立即扩大实现或采集范围的 backlog；启动具体研究前需要先固定研究假设、对照实验、目标硬件和证据边界。

### 后续模型支持

在当前 GLM-5.2 与 Kimi-K2.6 支持目标闭合后，模型接入顺序暂定为 `DeepSeek-V4-Flash → DeepSeek-V4-Pro → Kimi-K3`。

| 后续模型 | 预计需要新增或扩展的能力 |
|---|---|
| DeepSeek-V4-Flash | 优先支持 text-only、关闭 DSpark 的基础路径；在现有模型能力上新增 C4/C128 混合压缩 Attention、MHC、异构 KV cache，以及 FP8/MXFP4 MoE 的 workload、KernelSpec、lowering、容量和并行语义 |
| DeepSeek-V4-Pro | 复用 DeepSeek-V4-Flash 的算子与数据类型能力，重点扩展更大模型规模、最高 top-k 1024 routing，以及 DP/EP 容量、通信和并行执行验证 |
| Kimi-K3 | 在 Kimi-K2.6 基础上新增 KDA、Gated MLA、Attention Residual、多模态执行语义、复杂混合量化，以及超大规模并行与容量建模；因新增机制最多，放在上述 DeepSeek-V4 系列之后 |

### 异构执行与部署

| 研究问题 | 需要回答的核心内容 |
|---|---|
| 分层异构与跨层硬件映射 | 不同 Transformer layer 或同一模型不同阶段能否选择不同 Attention 实现、Kernel 路径或硬件；跨层迁移的 activation/KV 搬运、同步和流水线开销是否抵消异构收益；怎样根据 layer 特征形成可解释的放置策略 |
| Draft/MTP 异构 | Draft model、MTP propose/verify 与主模型能否部署在不同类型硬件上；小模型异构执行的算力收益如何与 token/hidden-state 传输、同步、接受率和资源空闲时间共同决定端到端收益 |

### 内存、搬运与容量约束

| 研究问题 | 需要回答的核心内容 |
|---|---|
| Host memory 与 device memory 搬运对 Serving 的影响 | 权重、KV cache 和中间状态在 Host DRAM 与显存/HBM 之间搬运时，带宽、固定延迟、排队、prefetch、计算重叠和 contention 分别怎样影响 TTFT、TPOT、吞吐与尾延迟 |
| 容量约束已经带来的系统影响 | 模型权重、KV cache、runtime workspace 和并发请求共同造成的容量限制如何改变 batch、并行方式、实例数量、路由、淘汰、offload 和可服务上下文长度；容量不足何时比单 Kernel 性能更早成为系统瓶颈 |

### Host 调度与 Forward CPU 开销

| 研究问题 | 需要回答的核心内容 |
|---|---|
| Model forward 外部的 Serving 调度开销 | Scheduler decision、batch formation、请求状态更新、worker coordination、model forward dispatch 和相邻 forward 之间的 device idle gap 应如何建模；Decode 每个 token step 都会重复承担该开销，需要量化它对 TPOT、吞吐和尾延迟的影响，并区分 CPU 饱和、同步等待与设备执行不足 |
| Model forward 内部的 CPU 开销是否需要进入生产模型 | 一次 forward 内部的 metadata/block-table 准备、MoE routing 整理、CPU operator、runtime bookkeeping 和逐 Kernel launch gap 是否足以影响端到端时间；Prefill 同时存在 forward 外部调度与 forward 内部 CPU 工作，不能预设主导项，需要通过清晰的 forward 边界和 CPU/device timeline 分别归因后再决定支持范围 |

这两个边界必须分别建模和验证：forward 外部开销属于 Serving Scheduler 与执行编排，forward 内部 CPU 开销属于一次 model execution 的 Host 侧工作，二者都不能被合并成一个按模型或 shape 拟合的经验常数。实测 Serving timeline 可以用于 attribution 和 evaluation；进入 production Forward 的机制参数仍需来自明确的 runtime 语义或独立、可复测的 Host microbenchmark。

### Serving 负载与 Attention 机制分析

| 研究问题 | 需要回答的核心内容 |
|---|---|
| 大 batch Serving 场景的仿真效果 | 验证高并发和大动态 batch 下的吞吐、排队、调度反馈、容量、Kernel shape、通信与资源竞争是否准确，明确误差随 batch、QPS、上下文长度和并行配置变化的规律 |
| 极致优化的低延迟推理 | 面向 single-batch 与 low-batch 场景分析首请求、短 Kernel、Host launch、Scheduler、同步和通信开销，评估 TTFT、TPOT 与端到端尾延迟的可优化下界及其硬件、runtime 和部署约束 |
| Attention 机制的系统性比较 | 覆盖 GQA、MLA、KDA、DSA、CSA、HCA 等机制，在统一模型与硬件口径下比较权重和 KV 存储、计算量、显存/HBM 访存压力、通信、并行映射及不同 prompt/decode token 长度下的瓶颈转移与适用边界 |

### 面向新硬件的系统设计探索

| 研究问题 | 需要回答的核心内容 |
|---|---|
| 面向 AI Systems 的新硬件设计验证 | 能否把 Midlayer 作为白盒设计空间探索工具，在目标硬件尚未量产或软件栈尚未完整时，根据可解释的 HWSpec、通信与内存机制预测 Serving 行为，比较不同硬件参数和系统设计，并用已有硬件实测建立可信度边界 |
| 超节点系统设计 | 超节点内 scale-up fabric、跨节点 scale-out 网络、共享或分层内存、加速器数量与拓扑怎样影响 TP/EP/PP、PD/AF 分离、KV 传输和多实例调度；哪些 workload 和并行方式真正能够利用超节点，哪些只会放大通信或容量浪费 |

这些问题的共同实验方法应是先以第一部分已经完成实测对齐的模型和三种硬件建立基线，再改变一个明确的部署、容量、拓扑或硬件机制，并同时报告性能收益、资源代价、关键路径变化和假设来源。目标是形成可验证的系统结论，而不是通过增加自由参数改善已有 evaluation case 的预测误差。

## 第三部分：长期方向（目前不作为支持目标）

本节只保留未来可能研究的方向，**不属于当前版本的支持承诺、backlog 或完成条件**。除非项目维护者明确把某项移入“当前支持目标”，否则不应据此自行开发、采集数据或扩大任务范围。

### 系统与存储

| 长期方向 | 当前基础 |
|---|---|
| SGLang Serving 支持 | 仅有受限 bridge，不是完整生产路径 |
| HBFrame FLASH/SSD 性能仿真 | 尚未接入 |
| SSD KV cache offload | 尚未接入 |
| Request Routing Cache 感知调度 | 当前只有非 cache-aware 路由 |
| CPU 侧发射开销（TileRT 等 MegaKernel 分析） | 当前只显式建模 Router dispatch，不覆盖该方向 |

### 并行方式

| 长期方向 | 当前基础 |
|---|---|
| DP 支持 | 尚未形成通用系统能力 |
| SP 支持 | 仅有算子内部 split 语义，不等于 Serving SP |
| CP 支持 | 尚未形成通用系统能力 |

### 低精度

| 长期方向 | 当前基础 |
|---|---|
| FP8 支持 | 尚未形成完整 Kernel/Serving 语义 |
| FP4 支持 | 尚未形成完整 Kernel/Serving 语义 |

### 训练

| 长期方向 | 当前基础 |
|---|---|
| 预训练支持 | 当前状态机和指标仅面向推理 |
| RL 训练支持 | 当前状态机和指标仅面向推理 |

### 更多硬件

| 长期方向 | 当前基础 |
|---|---|
| 其他新硬件支持与验证 | 需由维护者单独指定目标硬件、证据和验收范围 |
