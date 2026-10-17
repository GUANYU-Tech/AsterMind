# Midlayer Serving 系统设计

Midlayer 的 Serving 层回答一个系统问题：给定请求负载、模型结构、部署方式和硬件组合， 请求如何经过调度、算子执行、通信与数据迁移，最终形成端到端时延、吞吐和资源瓶颈。

本文只讨论从负载到算子执行时间线的设计。单个 kernel 如何从形状和硬件规格得到执行时间， 由 [Kernel 性能模型](kernel_performance_model.md) 负责。两层通过一个稳定边界连接：Serving 产生完整的算子及 launch 语义，Kernel 层返回一次调用的时间和物理瓶颈；任一层都不复制另一层的职责。

## 1. 目标与边界

Serving 仿真需要同时保持三类真实性：

1. **负载真实性**：请求到达、输入输出长度、prefix、MTP acceptance、专家路由等运行时输入必须来自明确的 workload、回放或显式假设，不能从性能结果倒推。
2. **控制面真实性**：准入、continuous batching、chunked prefill、抢占、KV block 管理等决策应由目标框架的调度语义产生，而不是由仿真器重新发明一个相似 scheduler。
3. **执行语义真实性**：每一步必须展开成带有形状、布局、dtype、rank、placement、依赖和数据量的逻辑算子，再由统一 Kernel 性能模型计时。

Midlayer 不执行模型权重和张量数据，不生成真实 token，也不试图仅凭请求 Trace 还原一个未被记录的生产部署。Trace 描述 workload；模型与部署配置描述“在哪些实例上如何运行”； Scheduler 描述控制面；仿真器负责把这些信息组合成可审计的事件时间线。

Production Forward 不读取 Trace 中的实测 latency、TTFT、throughput 或其他结果列。真实 observation 只在 evaluation 边界用于比较，不能反向影响调度策略、算子时间或系统参数。

## 2. 两层系统架构

```text
Production workload + Model + Deployment
                    │
                    ▼
          Request stream and runtime inputs
                    │
                    ▼
             Scheduling and routing
                    │
                    ▼
          Logical operator dependency graph
                    │
                    ▼
       Per-kernel analytical performance model
                    │
                    ▼
      Resource-constrained execution timeline
                    │
                    ▼
       Request metrics and bottleneck analysis
```

上半部分决定“运行什么”：请求状态、batch、token 工作量、模型层、并行 rank、通信与迁移。下半部分决定“需要多久”：单 kernel 预测、资源竞争、依赖等待和全局虚拟时间。两部分在完整的算子描述处相交，而不是通过 profiler kernel name 或经验 latency 表连接。

核心设计选择是：**给逻辑工作定价，而不是寻找一条历史 kernel trace 来匹配当前请求**。 Profiler 捕获的 kernel name 只描述某次实现和某次 launch；Serving 需要先从模型语义得到 Attention、GEMM、MoE、Vector、Collective 和数据迁移等逻辑工作，再为目标硬件选择 launch 并交给统一 Kernel 模型。这样改变 batch、并行方式或硬件时，工作量和时间仍能由同一条链解释。

完整闭环可以更具体地画成：

```text
生产请求 Trace          模型结构 / 并行 / 部署硬件规格与机制参数
      │                         │                              │
      ▼                         │                              │
  请求状态 ──▶ Router ──▶ Scheduler pool                      │
                                │                              │
                                ▼                              │
                    逻辑算子 / 通信 / 迁移图                   │
                                │                              │
                                ▼                              ▼
                       launch selection ──▶ Kernel 性能模型
                                                   │
                                                   ▼
                                      资源受限的全局事件时间线
                                                   │
                       完成、KV、token、queue 状态 ─┘
                                      反馈 Router / Scheduler
```

反馈箭头是 Serving 建模的关键：一次 kernel 的结束时间会改变下一次 Scheduler 决策，因此不能先离线固定所有 batch，再独立给它们计时。

系统由六个概念层组成：

- **输入层**：读取请求 workload、模型语义和部署配置。
- **控制面**：执行调度、路由、KV 容量管理和请求状态推进。
- **工作负载层**：把一次调度决策展开为逻辑算子及其依赖。
- **Kernel 层**：选择离散 launch，并预测单次调用的执行时间与瓶颈。
- **编排层**：根据依赖和物理资源构建全局事件时间线。
- **分析层**：从时间线计算请求指标、资源利用率和关键路径。

这种分层使框架语义、算子语义和硬件语义能够独立演进。更换 Scheduler 不应改变 kernel 公式；增加新硬件不应改写请求状态机；修正某个算子的 launch 也不应在 Serving 层增加时延修正项。

## 当前能力范围：支持哪些、如何支持

“代码能运行”不等于“该场景的性能已经被硬件验证”。一项能力必须从三个互相独立的维度描述：

| 维度 | 状态 | 含义 |
|---|---|---|
| 机制来源 | **Native** | 由目标框架原生状态机产生，例如 Scheduler 的 batching、抢占和 KV block 管理 |
| 机制来源 | **Captured** | 由允许的外部输入直接提供，例如请求到达和长度；仿真器只规范化，不重新生成 |
| 机制来源 | **Modeled** | 由 Midlayer 的 Router、工作负载展开或资源时间线产生 |
| 覆盖范围 | **Supported** | 语义和执行链在明确声明的组合内闭合 |
| 覆盖范围 | **Partial** | 只有列出的组合或机制闭合，不能推广到任意组合 |
| 覆盖范围 | **Unsupported** | 必要语义或物理参数缺失，必须显式失败或禁止用于性能结论 |
| 验证证据 | **Validated** | 已与独立 reference、人工守恒结果或硬件 observation 比较 |
| 验证证据 | **Projected** | 模型能够推演，但缺少目标场景的独立真值 |

这些状态可以组合。例如，“vLLM batching 是 Native+Validated”，“多实例路由是 Modeled+Validated/Projected”，“某种新硬件的 kernel 时间可能是 Modeled+Projected”。不能因为控制面是 Native，就把下游 timing 自动写成 Validated；也不能因为一项能力可运行，就省略其 Partial 范围。

当前主线能力如下。表中的“实现方式”描述它如何进入仿真，而不是罗列代码文件。

| 能力 | 当前范围 | 如何进入仿真 | 主要边界 |
|---|---|---|---|
| 生产请求 Trace | **Supported**：成功请求的 arrival、input/output token 数和 prefix hit 长度 | allowlist reader 将敏感 CSV 规范化成按时间排序的请求流 | 不读取 latency、TTFT、throughput 等结果列；不含失败、超时和真实路由 |
| 合成请求 Trace | **Supported**：可复现的 arrival 与请求形状生成 | 合成生成器产生与生产 Trace reader 相同的规范化请求流，后续共用同一 Scheduler、lowering 和时间线 | 只表达显式配置的分布和相关性，不能冒充生产 workload |
| vLLM 调度 | **Supported**：锁定版本下的 CPU-only continuous batching、chunked prefill、KV block 容量、finish、preemption/recompute | 只实例化真实 Scheduler 状态机；预测完成事件推进虚拟时间并反馈下一次调度 | 不加载权重、tokenizer 或 GPU executor；其他框架不是完整自动路径 |
| Prefix cache | **Supported/Partial**：零命中、部分命中和全命中的长度语义 | Trace 给出逻辑命中，adapter 按 block 与末 token 规则计算 Scheduler 可复用前缀 | Trace 没有 cache identity，不能恢复跨请求共享、真实淘汰或实例 affinity |
| 多实例路由 | **Supported**：多 P/多 D 的 `least_queued_tokens` | ingress→P 和 P→D 两级路由在决策时读取可见队列，并立即 reservation | 路由是给定部署下的模拟结果，不是 Trace 中的生产路由；暂不支持 cache-aware routing |
| 部署形式 | **Partial**：colocated、纯 PD、standalone AF、PD+AF 有闭环切片 | role 拆分改变算子 owner，并插入 KV 或 activation transfer 与依赖 | 每个实例独占自己的设备集合；不建模同卡多实例竞争 |
| TP/PP/EP | **Partial**：已有 TP、PP、TP×PP 与 MoE EP 的明确窄切片；目标 A800 路径固定为 Kimi-K2.6 TP=EP=8/PP2 与 GLM-5.2 TP=EP=8/PP4 | 先分片 shape/expert/rank，再产生 TP collective、8-lane PP P2P 和 pipeline dependency；对称 TP 计算保留一个 rank-local 代表调用，非对称 MoE 按 8 个 EP rank 展开，相同 KernelSpec 的预测仅在单次仿真内精确复用 | 不是任意 TP×PP×EP 组合；对称 TP 代表调用不提供逐 rank 利用率，SP、CP、通用 DP 与 pipeline bubble/overlap 尚未普遍闭合 |
| MTP | **Partial/Projected**：propose/verify、零/部分接受、短输出、handoff 与 replay contract | proposal 数由配置给出，接受数来自 observation replay 或显式假设，并逐 step 推进 token 状态 | 无 observation 时 fixed-count 只是假设；仿真不预测模型质量或真实 acceptance |
| 通信 | **Partial/Projected**：all-reduce、all-gather、all-to-all、pipeline/activation P2P | 通信作为带 rank、bytes、拓扑和依赖的一级工作负载进入统一时间线 | 当前解析后端可用；外部 ProtoBridge 尚未成为已验证的生产后端 |
| Host-KV offload | **Partial/Projected**：显式 LOAD/EVICT、容量账本、PCIe 区间和完成事件 | lifecycle 先产生迁移 intent，传输完成后才改变 KV residency | 尚不等于原生 vLLM offload；真实 victim/LRU、prefetch、Ramulator、SSD/Flash 和完整重叠未闭合 |
| Router CPU dispatch | **Partial/Projected**：多 pool 路由可使用一个串行 CPU dispatch 资源 | 每个 route decision 在进入目标 pool 前预约一段 CPU service interval | 当前只覆盖 Router dispatch；Scheduler、sampling 和 output processing 的通用 CPU 成本尚未闭合 |
| 模型 | **Partial**：Dense、MoE、MLA/MTP 共用主架构；Qwen3、Qwen3MoE、Llama、GLM4-MoE/MTP 有闭环切片；GLM-5.2 与 Kimi-K2.6 已有固定 A800 多卡 production trace 路径，GLM-5.2 另支持 projected MTP proposer/verifier 路径 | 固定模型语义生成层、attention、expert、MTP 和 KV workload；固定 vLLM 0.11 CPU Scheduler 负责控制面，0.27.1 固定源码负责目标模型执行语义；目标路径使用真实 TP/EP/PP placement 与解析 KV 容量 | 两个目标模型的新增 kernel 均只是 projected analytical，未做真机校准；vLLM GPU MTP/KV-offload execution 和跨硬件 target 路径仍未闭合，不能声称端到端硬件精度支持 |
| 硬件 | **Partial**：A800、C500、910B 可经过同一 Serving→Kernel 接口 | 每个实例绑定自己的硬件规格和 launch package，Kernel 层返回时间与 limitation | 当前只验证同类硬件实例组；跨硬件 P/D、LPU 等异构部署属于后续方向；结论继承 Kernel 层支持等级 |
| 系统指标 | **Supported**：TTFT、TPOT、E2E、CDF/percentile、throughput、SLO goodput、queue、资源利用率和关键路径 | 全部从同一条事件时间线与请求状态派生 | success-only Trace 的指标不能解释生产成功率或可用性；SLO 口径必须显式配置 |

这张表是后文的索引：后文分别解释每项能力为什么需要、状态如何推进、如何落成算子/传输和时间线。未列为 Supported 的组合不能通过“已有相邻 case”推断为支持。

### 2.1 Runtime Graph padding 的 shape domain

Serving Graph padding 必须区分 Scheduler logical token、Graph model-input tensor row 和算子恢复后的 active row。Production 入口接收 runtime 最终解析并实际捕获的 `capture_sizes`，在 contract 边界规范化为唯一升序集合，再对 logical token 数向上选择最近 bucket；canonical observation 的 `padded_total_tokens` 和 `capture_bucket_tokens` 只在 evaluation holdout 中核验，不能成为逐 step Forward 输入。未启用 Graph 时保持原有 eager/无 padding workload 逐值不变。

Graph dispatch 是与硬件和软件版本无关的数据驱动规则：配置由 runtime 原始 `configured_mode`、`uniform_decode_query_len`、唯一升序的 `capture_sizes` 和 PIECEWISE 编译器明确排除在 Graph 外的 `piecewise_eager_workload_kinds` 组成。统一 mode table 将 `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY` 和 `FULL_AND_PIECEWISE` 解析为 uniform/non-uniform runtime mode；每个 step 再根据所有 request 的 query length 是否等于 `uniform_decode_query_len` 选择对应 mode，并在存在不小于 logical token 数的 bucket 时令 model-input row 使用最小匹配 bucket。Attention/KV 的 active row 仍使用 logical token 数，lm_head 使用真实 logits selection；KV offload 继续消费 Scheduler 产生的 logical KV block range，不随 dummy model-input row 扩张。所选 mode 为 `NONE` 或超过最大 bucket时 runtime mode 解析为 `NONE`。Host timeline 的唯一 segment producer 按 expanded KernelPlan occurrence 和每个 pool/rank Host lane 生成描述：`FULL` 每 lane 覆盖全部 occurrence 的一次 replay，`PIECEWISE` 将配置的 eager workload kind 作为切分边界并把其间每段连续 occurrence 组成 replay，`NONE` 不生成 segment；source node、lane、repeat 或 coverage 不一致时必须 fail closed。不同运行栈的默认 bucket、configured mode 和 PIECEWISE splitting ops 属于 runtime 配置与 provenance，不进入 Serving predictor 的硬件或版本分支。CLI 使用 `--graph-mode <MODE> --graph-uniform-decode-query-len <N> --graph-capture-sizes <SIZE...> --graph-piecewise-eager-workload-kinds <KIND...>` 传入解析后的实际 profile。非零 padding 目前覆盖 standard-GQA、MTP disabled 的 colocated dense 单卡路径，以及经过 TP=EP=2 contract 验证的 MoE rank-local 路径；其他并行组合必须在 workload/lowering 边界 fail loud。

现有 canonical trace 对应的 resolved profile 是：A800 为 uniform decode `FULL`、non-uniform `PIECEWISE`，capture set 在 8 以上按 8 递增到 512；C500 为 `FULL`/`PIECEWISE`，capture set 在 256 以下按 8、从 256 起按 16 递增到 512；910B 为 `FULL`/`FULL`，capture set 与 C500 相同。这些是传给统一规则的数据，不是 production 分支。

Graph-specific source audit 如下；文件 hash 只锁定本次规则所依赖的静态边界，不把 runtime duration 引入 Forward。

| 硬件 | Runtime/source 身份 | 审计结论 | Production 状态 |
|---|---|---|---|
| Ascend 910B | canonical manifest：vLLM `0.18.0`，`vllm=0.18.0;core=b8a10b3d...a61439;executor=138fd673...e758d;vllm-ascend=5673759a...cdd3fd`；core/plugin snapshot SHA256 分别为 `b29f7ff4d721000f0ea169eb39b628227b647e8533e8ba37838c19f3291a5a49`、`daf28f173aab5f0231fbc697fed08082b5d0a145915b2af5cc8c402cdebca444` | `config/vllm.py`（SHA256 `cfe4308f67508bd091505614379394af2077852cfac2a12c837c6066271942d9`）解析 capture buckets，`v1/cudagraph_dispatcher.py`（`ad5769312ebb71adee0123db5239396779da6abac4fd51c27f37712df7d11454`）选择向上 bucket 或 NONE；plugin 使用 padded input、actual-token metadata、有效 KV slice 和 logits index compact | 通用 dispatch/shape-domain 规则 Supported；FA dummy-query 额外开销 Partial |
| NVIDIA A800 | 本地参考为 vLLM `0.11.0` | V1 config 将最终 capture set 预计算成 ceiling 映射；runner 以 `total_num_scheduled_tokens` 选择 padded model input，Attention metadata 保留 `num_actual_tokens`，slot tail 写 `-1`，lm_head 前按 `logits_indices` compact | 通用 dispatch/shape-domain 规则 Supported；实际 capture set 必须由对应 runtime 配置提供 |
| MetaX C500 | 本地参考为 vLLM `0.10.2` site-package snapshot | V1 config、ceiling 映射、runner actual-token metadata、slot sentinel 和 logits compact 与 A800 使用同一机制 | 通用 dispatch/shape-domain 规则 Supported；实际 capture set 必须由对应 runtime 配置提供，canonical 精度验证仍依赖绑定采集环境的 observation |

非零 padding node 必须完整携带 `graph_logical_tokens`、`graph_model_input_tokens` 和 `graph_runtime_mode`；lowering 后的 compute `KernelSpec` 另携带 `model_input_shape`。缺一个字段、`physical < logical`、`NONE` 中存在 padding、或 `model_input_shape` 不能逐维覆盖 logical shape 都会拒绝。没有 padding 的图保持既有字段和序列化结果，不以隐式默认值伪造 domain。FA 是例外：它使用完整的 logical `q_lens`/`kv_lens` 与 physical `model_input_q_lens`/`model_input_kv_lens`，两组长度和 runtime mode 缺一即拒绝。

下表是 production 的 operator × token-domain 审计矩阵；“预测”是用于有效 FLOP、KV、mask、routing 的 domain，“选择”是 launch grid、capture tensor、workspace 和临时 buffer 的 domain。

| Kernel family / 边界 | 预测 | 选择 / payload | 验证与边界 |
|---|---|---|---|
| Embedding/Gather | logical rows、logical index count | physical rows / captured input shape | Graph workload 保留 physical tensor；`KernelSpec` 将 elements/rows/index count 还原为 logical，并保留 `model_input_shape` |
| QKV/O/MLP GEMM、router | logical `m` | physical `m`、grid、workspace | launch selector 只见 physical shape；GEMM spec 的 `m` 只表达 active work |
| Vector（Norm、RoPE、activation、elementwise） | logical rows、seq、tok、elements | physical shape、grid、temporary buffers | source-worker program若假定每个 slot 都有效，则在 padding 下停用；保留 aggregate physical launch floor，不虚构 dummy semantic work；910B 的 fused Q/K RMSNorm+RoPE 同时保留 logical `seq_len` 和 physical `model_input_shape.seq_len` |
| FlashAttention main/combine | logical request `q_lens`/`kv_lens`、mask 和 KV work | physical batch、request lengths、grid、split workspace | `model_input_*_lens` 只用于 launch；padded slot 不进入 source per-CTA request schedule |
| KV cache write/offload | logical seq/KV block range | 无 padding 的实际 KV write；offload range 不变 | tail slot 是 sentinel，不能扩张 KV 或 offload bytes |
| FusedMoE / router | logical tokens、`tokens_per_expert`、routing provenance | physical tokens、recipe、grid、temporary buffers | routing histogram 必须为 logical tokens × top-k；每个 emitted MoE spec 保留 physical `model_input_shape` |
| TP/EP collective | 无额外有效 FLOP；routing/producer仍为 logical | captured activation bytes 为 physical | TP collective 与 EP output all-reduce 保留并校验同一 graph-domain；EP bytes 是 physical `B × H × dtype` |
| PP / AF activation transfer | 无额外有效 FLOP；producer仍为 logical | captured activation payload bytes 为 physical | P2P、目标 stage all-gather 和 AF hidden/residual 双 payload 都保留同一 graph-domain，并与直接 producer fail-closed 比较 |
| final norm / lm_head / sampling | final norm logical rows；lm_head 真实 logits rows | final norm physical captured tensor；lm_head 无 Graph row padding | `logits_indices` 只选择真实 request/verified token，Scheduler 生命周期保持 logical |

硬件 × runtime mode 的审计结论如下。C500 的标准 GQA 端到端 dense fixture 是 `head_dim=64`，而当前 source-backed analytical FA selector 明确只接受 `head_dim=128`；因此该精确组合是 fail-loud unsupported，而不是把 A800/910B launch 移植过去。

发射 provenance 也必须与 token domain 分开审计：`injected_launch` 只在已版本化的 source/observation launch entry 存在时消费离散 variant、tile、grid、split 等事实；`analytical_launch` 只可由算子静态语义、编译器/源码约束和 HWSpec 容量导出，尤其 A800/C500 GEMM 不是逐 shape 的源码实测 launch，不能标为 source-exact。910B 的 CANN tiler/program 和 source schedule 在可用时保留 typed source program；没有这样的 source-backed 组合必须 fail loud，不能借用相邻 shape 的 launch。Graph lowering 对 main、FA setup/copy/combine、GEMM split-K reducer、Norm materialization 和每一个 FusedMoE recipe helper 都以 physical ABI 调 selector；预测 spec 只把有效工作恢复到 logical domain。

| Hardware | FULL | PIECEWISE | NONE / eager |
|---|---|---|---|
| NVIDIA A800 | 66 logical → 72 physical 的 decode FA grid 为 `[1,72,4]`；FA predictor和 11→16 dense plan 都直接通过 | 11→16 dense plan 使用 logical work 与 physical launch；FA logical request lengths 保持不变 | 超出 capture 时 resolver 返回 `NONE`，不产生 generic graph-domain fields |
| MetaX C500 | 36 logical → 40 physical 的 FA main/combine grids 分别为 `[1,80,9]`、`[80,1,1]`，两者可直接预测 | generic token-domain contract 与 selector input 相同；上述 `head_dim=64` dense fixture fail-loud，C500 不对该缺少 source launch 的组合 fallback | 超出 capture 时 resolver 返回 `NONE`，不产生 padding |
| Ascend 910B | 11→16 dense 和 MoE plan 直接 lowering/predict；FA、GEMM、Vector、KV 和 MoE 保持上述分域 | 11→16 PIECEWISE lowering 保留 logical `m=11` 与 physical `model_input_shape.m=16` | 747-token、capture limit 512 的 resolver 为 `NONE`，不生成 padding |

代表 observation 是 `ascend_910b/llama-3.1-8b/tp1_qps4` 的 `vllm:120`：Scheduler 给出 1 个 decode token 与 10 个 prefill token，共 11 logical rows；显式 capture set 选择 FULL bucket 16，产生 5 个 padding rows。workload 与 launch shape 使用 16 rows，GEMM/Norm/RoPE/MLP 的 prediction spec 使用 11 logical rows并保留 physical `model_input_shape`，KV update 使用 11 rows，lm_head 使用 2 个真实 request rows；FA 保留真实 `(q_lens, kv_lens)=((1, 10), (33, 10))`，不把 5 个 dummy query 伪造成有效 KV。只读 holdout 对该 case 全部 2,067 个 step 重算 dispatch，其中 1,454 个非零 padding，runtime mode、physical token 与 capture bucket 逐项一致。

## 3. Workload、模型与部署

### 3.1 请求 workload

生产 workload 的最小输入是：

- 请求到达时间；
- 输入 token 数；
- 目标输出 token 数；
- 可选的 prefix 命中长度；
- 可选的、与运行时行为有关的显式输入。

原始 token 内容、用户文本和实测性能结果都不是仿真必需输入。敏感 workload 保持在本地， 只将允许的长度与时间字段规范化成请求流。

同一规范化请求流也可以由合成生成器产生。生产 Trace 与 Synthetic Trace 只在请求流的来源和证据强度上不同；进入系统后必须经过相同的 Scheduler、workload lowering 和全局时间线，不能为合成请求另建一套执行语义。

当前生产 Trace 表示成功请求路径。它可以回答“给定这些请求，系统的性能如何”，但不能据此声称生产请求成功率、超时率或可用性。原始到达时间回放表示一个已观察 workload；为了研究容量，应保持请求形状分布不变并扫描 arrival-rate，而不是把一次回放直接称为系统峰值。

ShareGPT canonical case 的 schema-v3 文件集合固定为 `requests.csv`、`metrics.json`、`scheduler_steps.jsonl.zst`、`alignment_steps.json.zst` 和 `manifest.json`。前三项逻辑内容必须来自同一次通过 observer-effect 门禁的无 Profiler trace-on Serving 运行：请求到达时间取实际 HTTP 提交时间并相对首个请求归零，metrics 使用该运行的 TTFT、TPOT、E2E 和 throughput，完整 Scheduler 序列记录逐 step wall 边界、scheduled request、prefill/decode token 和累计完成进度；两个 `.zst` 文件使用固定 level-1 zstd 表示，`manifest.json` 同时绑定物理压缩文件 SHA256 与解压后规范内容 SHA256、capture run ID、插桩版本以及同配置 trace-off/trace-on 四项指标均不超过 3% 的结果。`alignment_steps.json.zst` 解压后的 JSON 仍只保存 prefill、decode、mixed 三个结构代表 step，仅在 workload、operator 或 launch 顺序变化时重采，不承担时间对齐。

GLM-5.2 与 Kimi-K2.6 的 `debug_workload_1s` Production case 使用同一 `sharegpt_alignment_steps` kind 的 schema-v4 partial alignment：Prefill 与 Decode 分别保存固定真实上游源码身份、环境、实际 hook/tensor contract、代表性 production-size Midlayer operator 与完整 KernelSpec 对照，Mixed 未执行时必须显式标为 `not_observed`；每个 sample 必须声明 partial scope、phase、layer/rank、差异、遗漏与 unsupported 边界。该文件不得包含 duration 或伪称 launch capture，只能由 evaluation 消费，Production Forward 不读取；candidate 先写入 `profiler_output/target_model_alignment/` 并通过 case/manifest 校验，主 agent 审阅后才能发布到 `serving_dataset/`。

```bash
python scripts/evaluation/serving/build_production_candidates.py --target-model-validation-run experiment_logs/large_model_validation/a800_target_blocks_69461785.json
```

V1 async scheduling 会在上一 step 的 output update 完成前启动下一 step 的 schedule，因此相邻 step wall 区间允许重叠；四个边界各自必须按 step 单调递增，且单个 step 内必须满足 `step_start ≤ schedule_end ≤ update_start ≤ step_end`。该重叠是实际流水行为，不能在转换时串行化或裁剪。

Scheduler trace 通过进程内轻量 wrapper 在内存中追加紧凑记录，并在全部 canonical 请求完成后一次性写盘；采集不启用 Nsight、torch profiler、Ascend profiler 或其他外部 profiler。`experiment_logs/` 和 `profiler_output/` 只保存可删除的执行与候选产物，Forward 和 evaluation 的正式输入只能来自通过 manifest 校验的 `serving_dataset/` case。

需要区分“单步计时误差”和“预测时延改变后续 batching 的反馈误差”时，evaluation 层提供 **Scheduling Trajectory Replay**：`evaluation.serving.traces.scheduling_trajectory_replay.analyze_scheduling_trajectory_replay()`。它固定并重放 canonical `scheduler_steps.jsonl.zst` 解压后的真实 step 顺序、batch membership 和 token progress，用当前零拟合 Forward 重新预测每个 step，再在相同调度轨迹上重建 TTFT、TPOT、E2E 和 throughput。逐 step 的实测 serialized service 只在 Forward 完成后用于 comparison，结果中的 `forward_duration_observation_consumed=false` 明确记录这一边界。Dense case 不需要 expert routing 输入；MoE case 使用正式 Serving evaluation 相同的 canonical expert-frequency profile 仿真 routing，并在 `predictor_source.expert_routing` 中记录其身份与 provenance，不能将其解释为逐 step routing observation。

```bash
python scripts/evaluation/serving/scheduler/run_scheduling_trajectory_replay.py \
  --case ascend_910b/qwen3-4b/tp1_qps16 \
  --host-launch-profile configs/host_timing/host_launch_profiles/ascend_910b.json
```

令一个请求的完整 prompt 长度为 $P$、输出目标为 $O$、Trace 声明的逻辑 prefix hit 为 $H$。输入必须满足 $P>0$、$O\ge0$、$0\le H\le P$。$P-H$ 只是 Trace 语义下的逻辑 miss，尚不是 Scheduler 一定实际执行的 prompt token 数；block 对齐、末 token 和抢占会在后续产生 replay 或 recompute。`output_tokens=0` 是合法的成功请求：它保留完整 prompt 工作，但没有 TTFT/TPOT， 不能把不适用指标伪造为 0。

### 3.2 模型语义

模型配置提供稳定的结构事实，例如层数、hidden size、attention head、expert、top-k、MTP 结构和权重/计算 dtype。它决定每个调度 step 需要哪些逻辑算子，但不包含硬件时间。

Dense、MoE、标准 Attention 和 MLA 等模型族可以共享系统框架，但必须保留各自真实的数据流： MoE 的专家路由、MLA 的 KV 表示、MTP 的 proposer/verifier 工作都不能被压缩成一个平均算子。

#### Kimi-K2.6 分层支持矩阵

Kimi 的证据边界固定为 `moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb`、source config SHA256 `85825ca6e18cbe539eb83ee09eedfb3f4222265929f06e9f535a6d9364f55899`、官方 `configuration_kimi_k25.py` SHA256 `2bc5487efced291e0e6aa3b24aad1082bade4172aff94ea1b6f0ae20a1d6bb4f`、`modeling_deepseek.py` SHA256 `1fd8d198ff6ad69a5aec6fd85bf489d91ae2c432560b1e8ba7e34f710463c80a`，以及 vLLM 0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` 的 `DeepseekV2MLAAttention`、`MultiHeadLatentAttentionWrapper` 和 `MLAAttention`。只分析固定容器中的 `text_config`，不把视觉塔纳入 LLM Serving workload；生产尺寸 shape/branch 由固定源码与独立 manifest 验证，tiny CUDA smoke 使用 Transformers 5.10.2 的 `DeepseekV3ForCausalLM`。

CUDA smoke 使用已有隔离环境 `bullet`，不修改 `r2r_qwen3_next`。runner 会先校验 vLLM、Transformers 和两个模型 checkout 的 commit/hash，再执行两个 tiny block。

```bash
CUDA_VISIBLE_DEVICES=0 /home/yangxinhao/miniconda3/bin/conda run -n bullet python scripts/evaluation/serving/validate_target_model_blocks.py --vllm-root /share_data/yangxinhao/chip_modeling/upstream/vllm-v0.27.1 --transformers-root /share_data/yangxinhao/chip_modeling/upstream/transformers-v5.10.2 --glm-root /share_data/yangxinhao/chip_modeling/upstream/GLM-5.2 --kimi-root /share_data/yangxinhao/chip_modeling/upstream/Kimi-K2.6
```

| 层级 | 状态 | 已闭合语义 | Fail-loud 边界 |
|---|---|---|---|
| 固定配置解析 | Supported | 61 层、首层 Dense、其余 MoE、384 routed experts、1 shared expert、top-k 8、MLA geometry、BF16 compute、groupwise INT4 routed-expert weights | 修改配置或 provenance 后拒绝加载 |
| Scheduler step | Supported/Scheduler-only | 固定 vLLM 0.11 CPU Scheduler 从 production trace 生成 `ScheduleStep`，保留 batch、phase、token 与 KV length；A800 目标部署固定为 TP=EP=8/PP2，16 ranks、8 ranks/node | MTP、KV offload 或非 MLA KV layout 拒绝；这是控制面适配，不是 vLLM 0.27.1 GPU execution |
| 逻辑 workload | Supported/Static | fused Q/KV-A、Q-B、Q/KV norm、YaRN RoPE、576 元素压缩 KV、MLA attention、O projection、Dense/MoE/shared expert 与 LM head；prefill 执行 KV-B，decode 执行吸收后的 Q-down、MLA decode 和 V-up，mixed step 显式拆成两条分支；对称 TP 计算使用一个分片代表，MoE 按 8 rank 展开并在 rank-local routed/shared sum 后只做一次 TP all-reduce | 固定 TP=EP=8/PP2、DP=CP=SP=1；对称 TP 代表调用不提供逐 rank 利用率，KV offload 未接入 |
| 合法 KernelSpec | Supported/Structural | 每个逻辑工作项均 lowering；MLA attention、compressed KV update、decode batched GEMM、MLA RoPE、MoE router 与 INT4 routed experts 使用严格 typed contract；精确相同的 spec 只复用预测结果 | 只覆盖固定配置和目标并行合同，其他组合在 workload 或 lowering 边界 fail loud |
| HBM/KV 容量 | Supported/Projected-config | 每 rank 权重按 BF16 与 symmetric groupwise INT4 packed weight+BF16 scale 的真实 placement 计算，结合 90% A800 HBM、显式 2 GiB 非 KV reserve 和 pinned vLLM allocator 得到 64,745 blocks，即 1,035,920 tokens | reserve 是带 provenance 的 projected runtime configuration；production/canonical dataset 禁止以 debug block override 代替该计算 |
| A800 launch 与预测 | Partial/Projected | 既有 BF16 GEMM/Vector/Dense 路径继续使用 canonical selector；MLA prefill 使用 source-selected FA2 padding 语义，decode 使用 grouped MLA split 语义，BF16 router 与 symmetric INT4 W4A16 Marlin routed experts 均进入零拟合解析模型；prefill/decode/mixed production 执行可走完统一 predictor 与时间线 | 未消费 production duration，尚无生产权重 vLLM GPU execution 或真机性能校准；C500/910B 无对应 selector 时继续 fail loud，不从其他 attention/MoE variant 回退 |
| 实机验证 | Supported/Control-flow | A800 上以 Transformers 5.10.2 BF16 tiny Dense/MoE block 完成 3-token prefill 与 1-token decode，cache 长度增长到 4 | vLLM 0.27.1 要求 PyTorch 2.13，现有隔离环境为 2.7.1+cu126；这不是生产权重或 vLLM kernel 验证 |

#### GLM-5.2 分层支持矩阵

GLM provenance 固定为 `zai-org/GLM-5.2@b4734de4facf877f85769a911abafc5283eab3d9`、官方 config SHA256 `185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a`、Transformers v5.10.2 commit `0dad7b822255a0ae261ec45ae937371e859ffd1a` 和 vLLM 0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`。vLLM 0.16.0 commit `89a77b10846fd96273cce78d86d2556ea582d26e` 是审计到的首个注册版本，但尚未忠实实现跨层 shared-indexer pattern，因此逻辑语义固定到 0.27.1。Transformers 的 configuration/modeling SHA256 分别为 `3d72daa66b8a1bd703833b9b85a2b76e14fbd5158fe1238866d36e40b60225cc` 与 `0560406b772637b451cedfaccbb1d5a6c73fd002a0ae05807e95af89ced44da7`；vLLM 的 registry、`deepseek_v2.py`、MLA wrapper/base、sparse MLA/indexer 和 FlashMLA sparse backend SHA256 由 validation runner 固定。

隔离验证使用 `bullet` 环境的 Python 3.12.9、PyTorch 2.7.1+cu126 与 Transformers 5.10.2。安装的四个目标模型源码文件与固定 Transformers checkout 逐字一致；复现命令与 Kimi 相同。

```bash
CUDA_VISIBLE_DEVICES=0 /home/yangxinhao/miniconda3/bin/conda run -n bullet python scripts/evaluation/serving/validate_target_model_blocks.py --vllm-root /share_data/yangxinhao/chip_modeling/upstream/vllm-v0.27.1 --transformers-root /share_data/yangxinhao/chip_modeling/upstream/transformers-v5.10.2 --glm-root /share_data/yangxinhao/chip_modeling/upstream/GLM-5.2 --kimi-root /share_data/yangxinhao/chip_modeling/upstream/Kimi-K2.6
```

Transformers 隔离环境在 A800 上实际以 BF16 运行了 tiny `GlmMoeDsaForCausalLM` 和 `DeepseekV3ForCausalLM` 的 3-token prefill 与 1-token decode：两者 logits shape 分别为 `(1, 3, 32)` 与 `(1, 1, 32)`，cache sequence length 均为 4；GLM 同时验证了 full/shared indexer、Dense/MoE、shared expert 和独立 indexer cache 增长，Kimi 验证了 interleaved YaRN MLA、Dense/MoE 与 shared expert 控制流。Transformers 会先展开 K/V 再写 `DynamicCache`，因此 compressed MLA cache、GLM FP8 index cache和 vLLM decode absorption 仍由固定 vLLM 源码与生产尺寸 manifest 验证。vLLM 0.27.1 要求 PyTorch 2.13.0，而现有隔离环境为 2.7.1+cu126，因此没有伪称 vLLM CUDA execution。

| 层级 | 状态 | 已闭合语义 | Fail-loud 边界 |
|---|---|---|---|
| 固定配置解析 | Supported | 78 层、前三层 Dense、其余 MoE、256 routed experts、1 shared expert、top-k 8、MLA/DSA/indexer geometry、BF16 与 routed scale 2.5 | 修改配置或 provenance 后拒绝加载 |
| Scheduler step | Supported/Scheduler-only | 固定 vLLM 0.11 CPU Scheduler 从 production trace 生成类型完整的 `ScheduleStep`；DSA 的 BF16 compressed MLA cache 与 full-indexer FP8 cache 以 stage-local aggregate cache 参与 block accounting；A800 目标部署固定为 TP=EP=8/PP4，32 ranks、8 ranks/node；MTP depth、acceptance assumption、source ID 和 root seed 显式进入 Scheduler contract | KV offload 关闭；MTP acceptance 是 projected assumption，这是控制面适配而不是 vLLM 0.27.1 GPU execution |
| 逻辑 workload | Supported/Logical | prefill、decode 与真实 mixed graph；fused Q/KV-A、Q-B、compressed MLA cache、fused indexer K/weights、FP8 indexed K cache、full/shared top-k、phase-specific sparse attention、Dense/MoE/shared expert 与 LM head；PP 分层固定为 `[0,18)`、`[18,38)`、`[38,58)`、`[58,78)`，保证每个 stage 从 full indexer 开始且不伪造跨 stage index 传输 | 固定 TP=EP=8/PP4、DP=CP=SP=1；KV offload 未接入；workload 不能反向生成 Scheduler 决策 |
| MTP | Partial/Projected | 固定配置声明 1 个 next-token-prediction layer；CPU Scheduler 驱动可配置 proposal depth 和 acceptance，逻辑 workload 表达 embedding/hidden merge、DSA/index reuse、MoE/shared expert 与 shared head，并进入统一 lowering/predictor/timeline | acceptance rate 是显式 projected assumption；没有逐 step observation，且未执行生产权重 vLLM GPU proposer |
| 合法 KernelSpec | Supported/Structural | 每个逻辑工作项均 lowering；对称 TP 计算使用一个分片代表，DSA full/shared indexer、FP8 index cache、sparse attention、compressed MLA cache、decode batched GEMM、LayerNorm/RoPE、router、routed experts 与 GLM-5.2 MTP proposer 使用严格 typed contract；MoE 按 8 rank 展开并在 rank-local routed/shared sum 后只做一次 TP all-reduce | 对称 TP 代表调用不提供逐 rank 利用率；未有 launch 证据的 spec 使用 `launch=None` |
| HBM/KV 容量 | Supported/Projected-config | 每 rank 按 PP stage 计算 BF16 权重驻留，并为每层同时保留 compressed MLA cache 与仅 full-indexer layer 的 FP8 index cache；90% A800 HBM、显式 2 GiB reserve 和 pinned vLLM allocator 在 MTP disabled 时得到 65,586 blocks，启用 MTP 并计入末级 draft embedding/layer/head 权重时得到 57,296 blocks | reserve 是带 provenance 的 projected runtime configuration；production/canonical dataset 禁止 debug block override，异构 DSA cache 的 native allocator 适配仍是 CPU-only contract |
| A800 launch 与预测 | Partial/Projected | `batched_gemm`、FP32 `moe_router`、`layernorm`、`mla_rope`、DSA indexer/reuse、sparse attention、MLA cache 和 FP8 index cache 均有严格 launch 与源码工作量解析模型；prefill/decode/mixed production 执行可走完统一 predictor 与时间线 | 未消费 production duration；DSA/FlashMLA 的固定上游 CUDA 路径要求 SM90/SM100，因此 A800 只表示 SM80 projected analytical，不表示该 vLLM kernel 可在 A800 实机执行；C500/910B 无对应 selector 时继续 fail loud |
| 实机验证 | Supported/Control-flow | A800 Transformers tiny BF16 execution 验证 full/shared indexer、Dense/MoE、shared expert、prefill/decode 和 cache 增长；vLLM 0.27.1 源码已审计；TP=EP=8/PP4 Scheduler 与逻辑时间线有 CPU-only contract 测试 | 新增 kernel 尚无真机性能校准或生产权重 vLLM GPU execution；三硬件精度与 MTP 仍未验证，不能声称 GLM-5.2 端到端硬件精度支持 |

机器可读的 model × phase × layer-type × validation-level 矩阵由 `target_model_support_matrix()` 返回；production-shape 代表层 manifest 覆盖两模型 Dense/MoE 的 prefill、decode 和 mixed 共 12 个组合。

所有满足当前流式能力边界的 external canonical/production trace 默认使用 `--step-retention auto` 选择 `streaming_aggregate`，包括标准 ShareGPT case 与 Kimi-K2.6/GLM-5.2 production trace：每个 Scheduler step 仍完整执行 workload construction、KernelSpec lowering、统一 predictor 与时间线，但在生成有序 execution-contract hash 后立即释放完整 step IR，不再把所有 graph/plan/prediction/timeline 保留到 trace 结束；显式 `--step-retention full` 只用于需要逐 step artifact 的小规模审计。当前流式能力边界是单个 colocated pool、外部请求源且无 KV lifecycle replay；GLM-5.2 MTP 同时保留逐 step acceptance observation 以生成聚合 token/workload 守恒结果，其他拓扑在 `auto` 下继续使用 full，显式请求 `streaming_aggregate` 则 fail loud。最后一个请求到达后，accelerated 流式执行可通过 `--execution-workers` 并行投影相互独立的 workload、lowering、prediction、timeline 和 artifact contract；Scheduler 仍按原顺序逐 step 真实决策与完成，父进程以固定大小窗口按原顺序累加预测 duration 和有序 hash，并周期性回收 worker 限制长 trace 缓存占用，因此 schedule、指标、逻辑 kernel call 数与 schema 13 artifact contract 保持精确一致且 tail 内存不随 step 总数增长。PD transfer 不保留逐 transfer execution IR，而在 schema 13 streaming aggregate 中保留 trace transfer manifest digest、trace transfer count 与 simulation transfer count，并由 owner validator 穷举校验。流式模式保留 req/s、TTFT、TPOT、latency distribution、SLO goodput、总 step/kernel call 数、transfer counts、cache statistics、并行配置和阶段 wall time，但不物化逐 step artifact、system-resource 或 critical-path 明细，并通过独立 schema 13 aggregate contract 明确标记这一边界。

CPU-only vLLM Scheduler 的 length-only 请求不携带真实 token 内容且显式禁用 content hash，因此 adapter 用紧凑同值序列保存 dummy token IDs；长度、索引、append/extend 与 Scheduler 请求的逐 step 切片结果和物化列表完全一致，只在 vLLM 请求当前 step 的有限 token slice 时生成列表，从而避免长 production trace 为数十亿个无身份 dummy tokens 常驻分配内存。

锁定的 vLLM Scheduler 只允许 FCFS，单步最多接纳 `max_requests` 条等待请求；因此 adapter 对已经到达但位于该决策前缀之外的请求保留有序、run-local 延迟物化队列，并在每次真实 Scheduler 决策前补齐前缀。请求到达时间和顺序不变，每个 Scheduler step 仍由原生状态机动态执行，抢占请求仍由 vLLM 自己插回队首；这一机制只避免为当前决策不可见的长尾请求提前构造 native Request，不改变 batch、token、KV 或完成反馈。

#### Serving 独立验证 reference

`evaluation.serving.simulation.model_semantics_validation` 只比较事实，不复制 predictor 公式：请求流和 Scheduler step 用完整 SHA256 绑定，逻辑 operator 比较 kind/shape/dependency，KernelCall 比较来源节点、KernelSpec shape 与离散 launch，TTFT/TPOT/latency 只在 evaluation 侧按显式容差比较。`structural_lowering_manifest()` 可在不调用 predictor 的前提下输出 ScheduleStep hash、完整逻辑图、typed edge、cache side effect、逐节点 KernelCall 映射以及 supported/unsupported KernelSpec；Kernel 预测仍只能经 `lower_workload → predict_kernel_plan → kernel_base.predict()`，因此真实 TTFT、TPOT、kernel duration 与 launch observation 不进入 `ModelConfig`、production Forward 或 launch selector。

#### A800 7B/30B Serving 对照计划

首轮真实对照使用可部署 proxy，而不是把 Kimi/GLM 的静态分析误标为实机支持：Dense 使用 Qwen3-8B（7B class），MoE 使用 Qwen3-30B-A3B。两者必须复用与 C500/910B 相同的 `RequestStream → SchedulerBackend → ScheduleStep → build_workload → lower_workload → kernel_base.predict()` 代码路径；硬件 trace 只作为 evaluation reference。

| 维度 | Qwen3-8B | Qwen3-30B-A3B | 对照事实 |
|---|---|---|---|
| Prefill/decode | prompt 128/512/2048，固定输出 32 | prompt 128/512，固定输出 32 | Scheduler 每 step 的 phase、request、scheduled/computed token；operator 与 kernel 序列 |
| Batching | 同时到达 1/4/8 请求与限速到达 | 同时到达 1/2/4 请求与限速到达 | batch membership、mixed step、shape 与请求完成时间 |
| Prefix | 0、完整 block、非对齐、`prompt-1` 命中 | 0 与完整 block 命中 | trace hit、Scheduler effective hit、prefix replay 与 KV block 守恒 |
| MoE | 不适用 | TP=1 首轮；routing observation 仅进入 evaluation/replay | 每层 `tokens × topk` 守恒、routed/shared expert 节点与 kernel 序列 |
| Kernel sequence | Dense projection/attention/vector/LM head | 另含 router/fused MoE/shared expert | sidecar step 边界、ordered launch、KernelSpec shape/launch；duration 仅用于报告 |

硬件任务仍需主 agent 在执行前冻结如下记录：hardware=`nvidia_a800`；owner=`待确认的唯一 hardware executor`；device=`经 nvidia-smi 确认空闲后填写`；baseline=`本地已审计 commit`；session/PID=`启动后填写`；output=`experiment_logs/model_semantics_a800/<run-id>/`，若采集 nsys raw output 则写 `profiler_output/serving/nvidia_a800/<run-id>/`；预计时长为 8B 30–60 分钟、30B 60–120 分钟。批准后的入口形如 `/home/yangxinhao/miniconda3/bin/conda run -n r2r_qwen3_next python scripts/experiments/run_serving_experiments.py --manifest <approved-manifest> --run-dir experiment_logs/model_semantics_a800/<run-id> --real-run`；先运行 `nvidia-smi`，且 manifest 必须在执行前固定 model revision、请求矩阵、device 与输出身份。本阶段只提交计划，不创建 PID/session，也不占用设备。

### 3.3 部署语义

部署配置描述一次分析所假设的系统，而不是从 Trace 中猜测出来：

- 实例数量和角色；
- colocated、Prefill/Decode 分离或 Attention/FFN 分离；
- 每个实例使用的硬件与容量；
- TP、PP、EP 等并行方式；
- 实例之间的连接、通信和数据迁移路径；
- Scheduler 的 batch、token 和 KV 容量限制。

不同部署配置可以复用同一份 workload，从而比较架构选择。每个结果必须携带自己的部署和输入身份，不能把某个配置下的路由或调度结果当成 Trace 固有属性。

几种部署形式改变的是 owner 和依赖，而不是在总时延上乘一个比例：

- **Colocated**：prefill、decode、Attention 和 FFN 位于同一实例，KV 不需要跨实例 handoff；
- **PD 分离**：P 实例完成 prompt 阶段并产生可转移 KV，迁移完成后 D 实例才能继续 decode；
- **AF 分离**：Attention 与 FFN 分属不同 role，每个层边界都可能产生 activation transfer；
- **PD+AF**：同时具有请求阶段 handoff 和层内 role transfer，两类迁移必须分别建模。

当前 PD handoff 遵循 prompt-completion Scheduler 的状态语义。P 实例完成 prompt step 时，同时形成完整 prompt KV 和第一个 sampled output token；该 token 只承担 decode 状态推进，不要求仿真器恢复真实 token identity。随后：

1. P 停止处理该请求，不能继续执行后续 decode；
2. 完整 prompt KV 产生且仅产生一次 P→D transfer；
3. transfer 完成前，D 不能准入或执行该请求；
4. D 从“完整 prompt 已计算 + 第一个 token 已产生”的状态继续生成剩余 token，不能重算 prefill；
5. 全局 request ID、已生成 token 数和 KV token 数在 P、transfer、D 三段保持连续；
6. 若 P 阶段后已经没有剩余输出 token，请求直接完成，不产生无意义的 D handoff。

这组规则保证同一个请求不会被统计两次、KV 不会重复传输、P/D 也不会重复计算同一段 token。

并行方式也先改变工作负载，再影响时间：TP 切分 tensor shape 并插入 collective，PP 切分层并插入 stage send/recv，EP 重新分配 expert ownership 并产生对应的 gather/scatter 或 collective。 rank 数、shape、bytes 和依赖必须守恒；不能把“并行度”直接当作理想加速倍数。

## 4. 调度、路由与请求状态

### 4.1 Scheduler 的职责

Scheduler 决定请求何时进入 batch、每步处理多少 token、是否执行 chunked prefill、何时抢占以及如何分配和释放 KV block。仿真器向 Scheduler 提供虚拟完成反馈，使后续决策依赖前面步骤的预测结束时间。

主循环不是先生成一张静态 batch 表，而是在虚拟时间上反复执行：

1. 把当前时刻已经到达、迁移完成或重新可运行的请求交给对应实例；
2. 每个实例的真实 Scheduler 根据 sequence、token 和 KV block 预算做一次决策；
3. 将这次决策展开为模型算子、通信和数据迁移；
4. 由 Kernel 层预测每个工作项的时间；
5. 按依赖和资源竞争排入全局时间线；
6. 最早完成的事件推进虚拟时间，并更新 token、KV、queue 和请求完成状态；
7. 以新状态继续下一轮。

这种闭环与静态批处理不同：kernel 时间会改变下一次可调度时刻，新的 batch 又会改变后续 kernel 形状。因此 Serving 误差不仅影响单步时延，也可能改变后续调度轨迹。

命令行 runner 每完成 100 个 Scheduler step 会向 stderr 输出一次低频进度，包含累计 step、完成请求、调度 request rows/token、当前虚拟时间、pool 与 step ID；结果 JSON 和库调用默认行为不受日志影响。

#### Scheduler pipeline depth

`scheduler_pipeline_depth` 是每个 pool 可同时处于“已经作出 Scheduler 决策、尚未以 executor output 更新 Scheduler 状态”的 batch slot 数，不是 token budget、KV block 数或 PP stage 数。生产 trace 和 compact case 必须携带这一整数以及 Executor 的 `max_concurrent_batches` 原始证据；Forward 不为它设置默认值。单 pool 和每个 PD pool 都使用独立、显式的 slot ledger：提交一个 batch 占用一个 slot，只有该 batch 的完成 output 经 Scheduler update 后才释放该 slot；因此未完成 batch 不会被错误地当作可再次调度。

当前三套锁定 vLLM 源码均先由 Executor 暴露 `max_concurrent_batches`，再由 `EngineCore` 把它作为 `batch_queue_size`。队列未满时 `step_with_batch_queue` 先 `Scheduler.schedule()`，再经 `MultiprocExecutor.execute_model()` 向 worker message queue 提交 execution；每个 worker 的 `worker_busy_loop()` 串行 dequeue 并调用 `Worker.execute_model()`/`ModelRunner.execute_model()`，所以 host 可以保留多个未完成 future，但同一 pool 的 device step 仍按提交顺序执行。只有队满或无更多 request 可调度时，EngineCore 才从 FIFO 尾部取最早 future、等待 worker output，然后调用 `Scheduler.update_from_output()`；这个 update 完成后对应 batch tuple 已从 queue 移除，slot 才释放。当前 ShareGPT capture 都固定为 FCFS；`FCFSRequestQueue` 保持 waiting request 的插入顺序。

| Hardware | 锁定 source identity | Capacity 与 plan/submit | Worker dispatch 与 device serialization | Output/update 与 slot release |
|---|---|---|---|---|
| `nvidia_a800` | `vllm=0.11.0`; `core.py` SHA256 `4573563d926cafe0f1e11ebb25e862ce3bc722af9a24425dec791b9d43cfbedd`; `multiproc_executor.py` SHA256 `ad938d3411533330470ba2fdb63c6229c3dda06b193b9ea15ac9932792e0cd16` | `MultiprocExecutor.max_concurrent_batches` 返回 async=2、否则 PP size；`EngineCore.__init__` 建立 bounded deque；`EngineCore.step_with_batch_queue` 调用 `Scheduler.schedule` 和 non-blocking `execute_model` | `MultiprocExecutor.collective_rpc` → `rpc_broadcast_mq` → `WorkerProc.worker_busy_loop` → `Worker.execute_model` → `GPUModelRunner.execute_model`；单 worker dequeue/默认 CUDA stream 保持同 pool device 顺序 | `Future.result` → `Scheduler.update_from_output`；最早 batch 在 update 前从 FIFO pop，update 是请求状态推进和 slot 重用边界 |
| `metax_c500` | `vllm=0.24.0`; `core.py` SHA256 `d1219de0d526b2326a471fe8464b3b8a55730dcab63ea99eb9b15e8b90bb116b`; `multiproc_executor.py` SHA256 `5063ced047957ad64eba9774f24dbb18b341a57e9d1180750860084ecd3c784e` | `MultiprocExecutor.max_concurrent_batches` 返回 async=2、否则 PP size；`EngineCore.__init__` 建立 bounded deque；`EngineCore.step_with_batch_queue` 调用 `Scheduler.schedule`，executor 根据 capacity 选择 non-blocking RPC | `MultiprocExecutor.collective_rpc` → `rpc_broadcast_mq` → `WorkerProc.worker_busy_loop` → `Worker.execute_model` → `GPUModelRunner.execute_model`；单 worker dequeue/默认 device stream 保持同 pool device 顺序 | `Future.result` → `Scheduler.update_from_output`；最早 batch 在 update 前从 FIFO pop，update 是请求状态推进和 slot 重用边界 |
| `ascend_910b` | `vllm=0.18.0`; `core.py` SHA256 `b8a10b3dda030166495b3266f44c25449b63936a611985c98dfc2ea12da61439`; `multiproc_executor.py` SHA256 `138fd673505d1882c6f23b02ab6ba94884cb50342a371a9b62b9ffbf979e758d`; `vllm_ascend/worker.py` SHA256 `89661fbe301e7184893d17c47b4d69da0b95cafc8ea7992658a93e1f79bda081`; `model_runner_v1.py` SHA256 `8d35f26a9db8f63038158c139a43f92e3e970117bb4a2b43c82fd828e349b981` | `MultiprocExecutor.max_concurrent_batches` 返回 PP>1 时 PP size、否则 async=2/sync=1；`EngineCore.__init__` 建立 bounded deque；`step_with_batch_queue` 分别提交 non-blocking execute 与 sample future | `MultiprocExecutor.collective_rpc` → `rpc_broadcast_mq` → `WorkerProc.worker_busy_loop` → Ascend `NPUWorker.execute_model` → `NPUModelRunner.execute_model`/`sample_tokens`；worker dequeue 和 NPU stream/event dependency 保持同 pool execution 顺序 | sample `Future.result`（并验证 execute future）→ `_process_aborts_queue` → `Scheduler.update_from_output`；最早 tuple 在 update 前从 FIFO pop，update 完成才允许该 slot 被安全重用 |

Midlayer 的对应事件固定为 `schedule_start → schedule_end/slot acquire → device_queue_wait → device_start → device_end/completion → update_start → update_end/slot release`。`single_pool_engine` 和 `multi_pool_engine` 只把上述 queue capacity 参数化为正整数，分别维护每个 pool 的 slot ledger 和 device-ready time；reference fixtures逐项比较 acquire、FIFO device completion、update 和 release，而不是从 Midlayer 自身输出生成 golden。

这些 source state machine 对 PP=1 且 async scheduling 的可观察容量给出 depth 2。depth 1 是明确标记的反事实，作用是量化缺少第二个 in-flight slot 的影响；depth 3 和 depth 4 同样只表示在 PP=1 source state machine 上扩展容量的反事实敏感性，绝不是这些 runtime 已支持或已观测到的并发 batch 数。它们不能进入 canonical runtime evidence，也不能被写成硬件实测能力。

Scheduler 的容量由三个不能互相替代的约束共同决定：

- **KV block 容量**决定当前实例能够保存多少上下文；不足时请求等待、抢占或 recompute；
- **最大 sequence 数**决定一个 step 最多同时容纳多少请求，即使 token 很短也不能越过；
- **最大 batched-token 数**决定一个 step 最多处理多少 token，长 prefill 可能因此被切成多个 chunk。

增加 KV 容量不会自动放宽 sequence 或 token budget；反过来也一样。Production Forward 从所选 HWSpec 的物理 HBM 开始，应用带来源的 `gpu_memory_utilization`，扣除按模型与 TP/PP/EP placement 推导的 rank-local 静态权重，以及 runtime memory policy 中逐项、带单位和 provenance 的 activation、CUDA Graph、通信、allocator、fragmentation 与 framework reserve，再把每个 rank 的可用字节交给锁定版本 vLLM 的原生 KV allocator；最终 block 数取原生 allocator 的 Scheduler 结果。显式 block 数只允许走命名为 debug/conformance 的路径（包括固定输出的最小真实 Scheduler 一致性夹具），不能作为 production 默认值。真实 framework 报告的 available KV bytes 和最终 block 数仅由 evaluation 做精确身份绑定与整数相等校验，既不能覆盖 analytical 结果，也不能在缺失时回填。

### 4.2 Prefix 语义

Trace 中的 prefix hit 是逻辑命中长度；Scheduler 真正能够复用的 KV 还受 block 对齐、最后一个 prompt token 和框架规则约束。对当前锁定的语义，若 KV block 大小为 $B$，Scheduler 可直接复用：

$$
\boxed{
C=\left\lfloor
\frac{\min(H,P-1)}{B}
\right\rfloor B
}
$$

因此第一次执行的 prompt 工作量是：

$$
P-C=(P-H)+(H-C)
$$

第一项是 Trace 逻辑 miss，第二项是完整 block/末 token 规则产生的 prefix replay。系统需要继续区分：

- Trace 声明的逻辑命中；
- Scheduler 可直接复用的完整 KV 前缀；
- 因块边界产生的 prefix replay；
- 抢占后产生的 recompute。

这四者影响 prefill 工作量和 KV 容量，但含义不同。若 Trace 不提供 cache identity 或共享关系， 仿真只能把 $H$ 视为路由后仍然成立的系统级 oracle；不能虚构实例本地 cache affinity、跨请求共享、refcount 或淘汰历史。

### 4.3 多实例路由

路由策略只能使用决策时刻可见的状态，并在每次选择后立即预留负载，避免同一时刻到达的请求都看到过期队列。路由依据、负载口径和 tie-break 必须显式记录。

当前主路径使用基于未完成 token 工作量的 `least_queued_tokens`。PD 分离时，请求先经 ingress→P 路由；prefill 与必要的 KV transfer 完成后，再经 P→D 路由。每个决策都只看该时刻可见状态，并立即把新请求的工作量 reservation 加到目标实例；否则同一时间到达的一批请求会全部看到旧队列并错误地涌向同一实例。tie-break 必须稳定，路由依据和 reservation 口径必须进入结果 provenance。

Round-robin、cache-aware routing 或预测完成时间策略可以作为扩展，但必须说明读取了哪些状态， 并与生产 Trace 中不存在的真实路由信息区分开。

多 pool 路由还可以显式占用一个所有 route stage 共享的串行 CPU dispatch 资源。一次路由先等待该资源可用，再支付配置给定的 dispatch service time，随后请求才能进入目标 pool；因此高到达率下会形成独立于 Router queue 和设备 queue 的控制面等待。服务时间为 0 表示实验显式忽略这一机制，不代表 CPU 没有开销。非零值必须来自独立 CPU 机制测量或明确的 Projected 假设，不能由 Serving E2E residual 反解。

当前这一资源只覆盖 Router dispatch。Scheduler Python 执行、sampling、token/output processing 等 CPU 成本尚未形成统一生产模型，不能静默并入 dispatch service time；需要研究这些机制时， 应分别建立资源、依赖和证据。

### 4.4 运行时不可知量

部分工作量只有执行模型后才知道，例如 MoE 专家分布和 MTP 接受 token 数。无权重仿真不能凭空生成这些事实。它们必须来自以下之一：

- 与请求绑定的 observation replay；
- 独立、版本化的分布模型；
- 明确标记的确定性假设；
- 上下界或敏感性分析。

这些输入影响算子形状，但不能通过目标系统的实测时延进行校准。

MTP 是这类不可知量的典型例子。一次 step 的 proposal 数由部署配置和 drafter 约束决定，真正 accepted 的 token 数则必须来自逐 step observation replay、版本化分布，或明确的 fixed-count 假设。状态推进必须守恒：

```text
proposed = accepted + rejected
verified ≥ accepted
emitted ≤ accepted + terminal token
```

proposer、target verify、accepted prefix、剩余输出和 PD/AF handoff 都进入同一个请求状态机。无 observation 时，fixed-count 结果只能标记为 Projected；仿真器不能从 TPOT 或端到端 latency 反解 acceptance，也不能声称预测了模型质量。

## 5. 从调度决策到逻辑算子

一次调度 step 首先被解释为模型级工作：哪些请求执行 prefill、decode、MTP propose/verify， 哪些层和 rank 参与，以及 KV 状态如何变化。随后工作被展开成逻辑算子依赖图。

每个逻辑算子至少需要表达：

- 算子种类和数学形状；
- dtype、layout 与执行 variant；
- 所属实例、rank 和硬件；
- 输入输出数据量；
- 前驱依赖和同步语义；
- 与通信、KV 迁移或其他算子的关系。

逻辑算子以数学和部署语义命名，而不是以某次 profiler 捕获的 kernel 字符串命名。一个逻辑算子可以降低为一个或多个 kernel；不同硬件也可以选择不同实现。离散 variant、tile、grid、 block 和 split 由 Kernel 层的 launch selection 决定。

一次 decoder step 通常沿层顺序产生 norm、projection、Attention、residual、FFN/MoE 等节点； 并行配置再将这些节点复制或切分到 rank，并在数据依赖跨 rank/role 时插入通信。这个过程只回答 “哪些工作必须发生、数据从哪里流向哪里”，不在这里估算时延。随后每个计算节点降低为完整的 kernel 调用，每个通信/迁移节点保留 payload、participants、路径和资源身份，才进入统一时间线。

Serving 必须把算子补全到足以独立预测的程度，然后调用统一的单 kernel 模型。它不能根据 Serving case、runtime kernel name 或端到端误差选择另一套公式。

### 5.1 通信是一级工作负载

Collective 和点对点传输不是隐藏在 compute kernel 外的固定常数。通信描述需要包含参与 rank、每 rank 数据量、拓扑、算法、方向和依赖。其时间由对应物理后端给出，并与 compute 一样进入资源时间线。

TP、PP、EP 和 Attention/FFN 分离产生不同通信模式：all-reduce、all-gather、all-to-all、 pipeline send/recv 或 activation exchange。逻辑 bytes 和 rank 关系必须先守恒，再讨论通信性能和重叠。

通信节点至少包含 collective/P2P 类型、participants、每 rank payload、拓扑/链路、算法和前后依赖。解析通信后端可以根据这些事实计算链路工作；未来外部通信仿真器也必须消费同一份描述并返回同一计时域的结果。两种后端是替代关系，不能把外部结果再加到解析结果上。通信与计算是否重叠由资源身份和依赖决定，不使用全局 overlap ratio。

传输的**计时类型**与**互斥资源身份**是两个正交概念。前者说明使用 PCIe、片间互联、Host memory 或某种 collective 算法，以选择相应的物理计时模型；后者说明这些工作最终占用哪一组真实 fabric、endpoint 或 DMA engine，以决定竞争关系。两条 route 可以使用同一种计时模型但占用独立资源，也可以方向不同、逻辑路径不同却共享同一个 endpoint。只有后者不同，时间线才允许它们独立重叠；不能用 route 名称或传输方向自动推断资源互不冲突。

### 5.2 KV 与数据迁移

KV load/evict、Prefill 到 Decode 的 KV handoff、Host offload 和未来的 SSD/Flash 层都应表示为显式迁移工作，而不是在请求时延上增加经验常数。

迁移需要同时描述：

- 源和目的存储层；
- block 或请求粒度；
- owner rank；
- 容量预留与完成时刻；
- 使用的 PCIe、片间链路或 Host memory 资源；
- 与 compute 的依赖和重叠条件。

KV residency 只有在物理迁移完成后才改变。容量账本、迁移 intent 和实际 completion 分开， 才能避免尚未完成的 load 被错误地当成已驻留数据。

Host-KV 的状态变化可以概括为：

```text
GPU-resident KV
  ── EVICT intent → PCIe/Host interval → completion ──▶ Host-resident KV

Host-resident KV
  ── LOAD intent  → PCIe/Host interval → completion ──▶ GPU-resident KV
```

intent 创建时预留必要容量，但只有 completion 事件才能提交 residency 变化；被请求消费的 KV 还必须等待 LOAD 完成。这样 queue、PCIe busy time、容量压力和 compute overlap 都可以从事件时间线解释。当前机制仍是显式 lifecycle replay，不应写成已复现生产 vLLM 的 victim policy、 prefetch 或 Host DRAM 控制器排队。

## 6. 单 kernel 模型边界

Serving 向 Kernel 层提供完整的算子、launch 和硬件语义；Kernel 层返回：

- 预测执行时间；
- 主要资源时间和瓶颈；
- 使用的支持等级；
- 尚未闭合的物理证据。

概念接口为：

```text
Operator shape + Launch + Hardware
                 ↓
      Per-kernel performance model
                 ↓
Duration + Resource diagnostics + Limitations
```

Serving 只消费这一结果，不解释 Kernel 层内部的 compute、cache、transaction 或 pipeline 公式。Kernel 层也不读取请求 arrival、TTFT、Router queue 或生产 workload 的实测结果。

## 7. 全局虚拟时间与资源竞争

系统时间线同时满足两类约束：

1. **依赖约束**：一个工作只有在所有前驱完成后才能开始。
2. **资源约束**：同一物理资源上的工作按该资源的并发和仲裁语义执行。

因此，一个工作开始于“前驱全部完成”和“所需资源可用”两者中较晚的时刻。不同 rank、不同执行引擎或独立链路可以重叠；共享 GPU、通信 fabric、PCIe 或 Host memory 的工作必须竞争同一资源。

Batch kernel 的时间属于共享执行区间，而不是分别归属于每个请求。请求级 TTFT/E2E 路径应从共享区间中提取贡献，不能把同一 batch kernel 的完整时间重复计入每个请求的系统总工作量。

虚拟时间循环在一次 step 完成后把结果反馈给 Scheduler，随后产生下一步决策。多实例系统还要合并实例本地时间、Router queue、Scheduler queue、通信和跨实例迁移，形成统一系统 horizon。

## 8. 指标与瓶颈分析

从请求级 TTFT/TPOT 逐层定位到 Scheduler step、runtime interval 和单 Kernel 的统一操作方法见 [Serving 误差定位流程](serving_accuracy_diagnosis.md)。

### 8.1 请求体验

- **TTFT**：从请求到达到第一个对外可见 token 的时间。
- **TPOT**：输出阶段相邻 token 的平均间隔，需明确 eligible request。
- **E2E latency**：从请求到达到请求完成的时间。
- **Latency distribution**：经验 CDF 和明确的 percentile 算法，而非只报告均值。

### 8.2 系统产出

- request throughput；
- output-token 或 input+output-token throughput；
- 在版本化 TTFT/TPOT SLO 下的 request/token goodput；
- Interactivity 与 token-generation speed 的联合 trade-off。

Goodput 只统计满足给定 SLO 的成功请求，不等同于生产请求成功率。阈值、token 口径和仿真时间窗口必须属于实验配置，不能在报告代码中隐式固定。

### 8.3 资源与关键路径

资源分析应覆盖 compute pool、通信链路、PCIe/Host-KV、Router queue 和 Scheduler admission。关键路径将请求时间分解为：

- 调度或路由等待；
- kernel/通信/迁移执行；
- 共享 batch step；
- 依赖尚未满足的等待；
- 当前证据不能进一步解释的空隙。

无法证明原因的 gap 应保持未知，不能为了报告完整而强行归因。

## 9. 支持、验证与解释边界

系统功能和性能精度是两个独立维度：

- **Supported**：输入能够通过完整链路并产生结构合法的结果。
- **Validated**：关键语义或性能已经与独立 reference/硬件 observation 对齐。
- **Projected**：公式可执行，但缺少目标场景真值，只能作为推演。
- **Unsupported**：缺少必要语义、launch 或物理参数，必须显式失败。

一个 Serving case 可以在调度、shape、rank 和 bytes 上 Validated，同时在 kernel timing 上仍为 Projected 或 Unsupported。控制面对齐不能替代硬件时间验证，单 kernel 精度也不能证明完整 Scheduler 和系统时间线正确。

当前主要生产硬件范围是 NVIDIA A800、MetaX C500 和 Ascend 910B；主要算子范围是 GEMM、 FlashAttention、Fused-MoE 和 Vector。Dense/MoE、colocated、PD、AF、TP/PP/EP、MTP、通信和 KV offload 按独立纵向切片验证，不能从一个组合自动推广到所有组合。

每次运行还必须形成一份可复现的结果身份，至少绑定：

- workload 身份及允许读取的请求字段口径；
- 模型 revision、结构和 dtype；
- 部署中的实例、role、硬件、TP/PP/EP 和链路；
- Scheduler 版本及三项容量限制；
- Router 策略、稳定 tie-break 和 CPU dispatch service 假设；
- Kernel launch/hardware profile 的版本与支持等级；
- MTP acceptance、MoE expert 分布、Host-KV lifecycle 等外部 observation 或显式假设；
- 仿真代码版本、时间单位和指标/SLO 定义。

这份身份是结果的“说明书”：它不保存敏感请求内容，但必须让另一次运行能够重建相同的控制面、工作负载和物理参数。缺少上述任一项时，结果只能作为不可复现的临时观察，不能用于跨部署或跨硬件比较。

## 10. 演进方向

系统架构允许沿稳定边界扩展：

- 接入新的 Scheduler 或 Serving 框架；
- 增加新的模型结构和运行时输入；
- 增加 DP、SP、CP，以及跨硬件 P/D、LPU 等异构部署；
- 接入更精确的通信、Host memory 与 SSD/Flash 后端；
- 支持 FP8、FP4 等低精度；
- 扩展到新的硬件。

每种扩展都应先补全自己的语义和证据，再复用现有 Kernel 与时间线接口。训练和 RL 可以借用相同的算子/资源抽象，但具有不同的状态机、通信模式和目标指标，不属于当前推理 Serving 模型的隐式能力。

扩展时仍要保持层次边界：新的 Scheduler/backend 只负责产生调度 step、KV 状态事件和框架原生决策，不能直接构造硬件 kernel、调用性能公式或把调度耗时塞入 kernel 时间；新的模型结构负责产生守恒的逻辑算子和依赖；新的通信或存储后端只替换对应资源的计时实现。每项扩展都要同时验证本层语义守恒以及下游时间线结果，不能只以端到端数值能够运行作为验收。

## DSV3 H200 closed-loop 输入与框架对比

当前 Midlayer 覆盖六种 setting；完整的 framework × deployment × MTP 组合有八种，缺少 SGLang PD 的 MTP off/on 两种。SGLang 本身支持 PD，但 Midlayer 的 `sglang_mirror` 目前只实现 colocated 调度，PD 会显式报不支持，不能通过增加配置行启用。未来补齐这两种运行方式时仍可共享现有请求输入。CAL’26 图 5 的 PD 模拟使用 vLLM 后端，叠加的 InferenceX SGLang 实测点仅为 colocated。

`configs/evaluation/experiments/deepseek_v3_h200_closedloop.json` 固定完整 DeepSeek-V3 FP8 的 8K/1K closed-loop 输入：C=4/8/16/32/64，每客户端 3 个正式请求和 1 个单独执行、丢弃结果的预热请求，随机种子为 0。六种 setting 为 vLLM 的 colocated/PD × MTP off/on，以及 SGLang colocated × MTP off/on；colocated 使用 8 张 H200、TP8/PP1/EP1，PD 使用各 8 张 H200 的 Prefill/Decode 池。六种 setting 共 30 次运行，共享 5 组输入。`serving_dataset/synthetic/inferencex_style/dsv3_h200_8k1024_closedloop_tp8_ep1/` 按 cN 目录保存输入，每组包含 `manifest.json`、`requests.csv`、`warmup_requests.csv`，以及 colocated、pd_disagg 各一份内存配置；框架、PD 和 MTP setting 均引用同一对请求文件，全部由 Git LFS 跟踪；历史轮次不作为新的 setting 发布。

统一 Serving schema v5 的 `identity.source_kind="synthetic"` 分支由 `dataset_pipeline.serving_dataset.compact_case.load_compact_case` 校验。它记录生成算法、token 长度区间、请求数、闭环语义和输入文件校验值；metrics、alignment_steps、scheduler_steps 必须显式为 `not_observed`，不接受伪造的采集文件。关闭 prefix caching 下的零 cached-hit 是显式假设，不是实测零命中。正常运行只读已发布输入，禁止再次生成或覆盖；各框架的 Scheduler、graph、routing 和 MTP 运行配置由 suite 绑定。请求 CSV 保持统一四列和规范序列化，warmup 不进入正式统计。

vLLM setting 使用真实 vLLM Scheduler，SGLang setting 使用 `sglang_mirror` 的源码语义镜像而非真实 SGLang GPU 服务。此 suite 保留已存在的实验 recipe：vLLM 使用 `graph-mode NONE`，SGLang 使用 `FULL_DECODE_ONLY`，因此曲线对比包含执行配置差异，不能称为仅改变框架的严格控制实验。MTP on 固定 2 个 speculative tokens、maximum-entropy 接受策略、均值 0.673，来源为外部 InferenceX DeepSeek-R1/SGLang 日志的跨并发平均值，转用于本实验时仍是输入假设，不是这些请求上观测到的接受率。MoE 路由使用已有 DSR1 model-level profile，原始 source workload 未完整记录；PD 通信使用 development-only 理论参数，所有结果均为模拟投影。

```bash
# 只校验 5 组共享输入和 6 种 setting，不运行模拟、不写入数据集。
/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python scripts/evaluation/serving/run_inferencex_deepseek_v3_8k1024_closedloop_comparison.py --validate-only

# 并行运行六种 setting；output-dir 必须位于 experiment_logs 且为空。
/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python scripts/evaluation/serving/run_inferencex_deepseek_v3_8k1024_closedloop_comparison.py --output-dir experiment_logs/dsv3_h200_closedloop/run1 --max-workers 8

# 也可选择单个 setting 和并发点。
/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python scripts/evaluation/serving/run_inferencex_deepseek_v3_8k1024_closedloop_comparison.py --settings sglang_colocated_mtp_on --concurrencies 4 --output-dir experiment_logs/dsv3_h200_closedloop/smoke
```

输出含每次运行的精确命令、请求级模拟结果、summary 和两套 PNG/PDF 曲线。横轴固定为 `1 / P90 TPOT`；纵轴分别报告 output-only TGS 和历史脚本使用的 input+output throughput，不能混用。colocated 除以总计 8 张 GPU，PD 除以 8 张 decode GPU（总计 16 张），因此这个归一化也必须在比较时保留。该 InferenceX-style 定义与 `configs/evaluation/interactivity_tgs_h200.json` 中 P99/output-only 的独立评估合同不同，不能直接混合结果。1K/128、L32、req10 及 C=128 均不属于此 suite。8K/1K 旧 CLI 的生成参数已由 suite 替代，不再接受 `--data-subdir` 等生成选项。

论文绘图入口 `scripts/paper_plot/cal26/collect_data.py` 选择上述四种 vLLM setting、五档并发，写入 `<run-dir>/dsr1_sweep/vllm_<group>/cN.json`。其 `plot_figures.py` 直接读取模拟原始总吞吐，保持论文按全部 GPU 归一化的口径：colocated 除以 8，PD 除以 16；不使用 suite summary 中按 decode GPU 归一化的值。当前采集是已发布输入上的更新结果，原论文含 C=1/2 的七档结果须通过 `cal26-submit-review-ver` 复现。
