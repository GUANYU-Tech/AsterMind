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

SGLang Scheduler mirror可以在`add_request`显式提供完整`prompt_token_ids`和`output_token_ids` transcript，按chunked prefill及普通decode的调度行序导出`BackendStep.embedding_token_ids`，来源标为`sglang_explicit_request_token_ids`。它不执行模型或生成token值，长度型输入继续保留unknown，不能与带值输入在同一backend混用。EAGLE另需draft/tree状态，不从接受数恢复token值。调度计数与原长度镜像保持一致。

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

### 虚拟 forward 与自动存储生命周期

Qwen3的`cuda_eager_inplace`场景声明使用vllm_c CUDA RMSNorm：leading-axis有间隔的Q/K view直接读取原storage，仅末维非unit stride才复制；fresh输出在该内部复制之前分配。ROCm的flatten路径、IR native和batch-invariant仍是不同源码程序，不能互相替代，也不由该场景声明推断实际provider。

vLLM0.30 CUDA 场景可显式设置 `RuntimeInputs.norm_source_batch_invariant=False`，CLI 为 `--norm-source-batch-invariant disabled`。执行入口将该声明与当前每个 RMSNorm/add-RMSNorm 的真实 shape、layout 和源码版本交给既有 source-launch lowerer；其他算子及源存储图不变，逐步重复声明同一 Norm 会拒绝。未设置时仍为未知；True 分支尚未支持。False 来自上游 `envs.py` 的 `VLLM_BATCH_INVARIANT=0` 默认值，可用于声明默认场景，不证明历史采集的环境值或原生 provider；运行 provenance 单独保存该范围，不增加时间参数。

同一 CUDA 源码场景在 H200、BF16 analytical 执行中可显式设置 `RuntimeInputs.bf16_gemm_model="tiled_reference"`，CLI 为 `--bf16-gemm-model tiled_reference`。普通无 bias、同质 BF16 Linear/head 复用既有静态 tiler 与 GPU pipeline/resource 模型，表征有限 CTA、共享操作数和流水服务；真实 logical/physical Graph 行、源图身份、通信、依赖与容量保持不变。默认仍用现有数学路径；混合 FP32 输出 router、quantized 和 biased 程序沿各自既有模型。此选项声明固定解析参考程序，split=1、不声称 cuBLAS geometry、实际算法或 provider 已观测，相关假设独立写入 provenance；不得依据 production duration 或 MAPE 调整 tile 或逐 case 选择版本。

模型接入可以选用模型官方实现、Transformers 或目标推理框架的 forward，并绑定准确配置、源码版本及执行分支。适配层替换初始化、权重加载和基础算子接口；模型 forward 保留网络控制流。新增模型优先复用已有虚拟算子，只为新增结构补充接口，不另写一套依赖图或张量释放表。不同来源的融合、量化、KV 和采样行为需要分别声明，不能仅凭模型名称认定等价。

`src/serving/workload/traced_forward.py` 的 `ForwardTrace` 只创建 shape、dtype、stride 和 storage/view 元数据，不创建真实 CPU/GPU Tensor，不执行矩阵乘、排序或随机采样等数值运算。共享算子提交输出分配、alias 和原地写语义后，记录器自动建立读写依赖，并根据同步 CPython forward 的引用与作用域记录生命周期。输出复用既有 `WorkloadGraph`、`MemoryAllocation` 和 `TensorView`；同一源码程序可生成性能图与存储账本。涉及真实张量值的分支需要显式动态输入或条件化适用域。

动态 forward 用于模型接入和首次编译，仿真循环优先复用记录的图结构，绑定当前形状、KV 长度、专家路由和槽位。已验证的无 padding、每请求单 token 的双 rank 捕获程序使用只读绑定帧：静态依赖和节点元数据保留在模板，只有需要生成当前 KernelSpec 的代表节点才物化；显存统计与执行摘要直接读取同一帧的源信息。已验证的 vLLM 0.11、MACA 0.24 和 Ascend 0.18 双 rank、无 padding 普通 decode 热调用直接将当前参数送入冷编译的绑定 recipe，不再构造临时 prefix 图；其他执行结构和 padding 保持同一虚拟 Tensor source program 的既有编译与绑定方式。显式访问公共图时仍得到完整节点元组，旧快照保持不变。KernelPlan 在调用集合、硬件、资源和依赖不变时复用 DAG，只替换当前规格；源码、模板、绑定声明或执行分支变化时重新编译。未覆盖的结构继续使用同一虚拟 forward 的完整编译路径，不跳过工作或降低验收条件。

`fast` 模式保留逻辑工作图和源存储统计，但省略独立导出的逐调用操作数快照（`TraceMemoryContext`）。这不代表原生操作数端口已完整绑定；相关执行域的等价比较应覆盖工作图、KernelSpec、依赖与 placement、激活统计、逐调用预测及时间线，原生 provider、流量与 normal allocator 的验收仍单独说明。

既有 lowering memo 同时按精确物理 node contract 与 HWSpec 身份缓存 Norm、activation、RoPE 和 KV update 的不可变 source launch recipe，沿用原有有界 LRU。当前 source port、图及分支校验仍逐步执行；shape、layout 或运行开关变化会使用不同契约，非法分支不会继承暖缓存。缓存只减少重复几何构造，不缓存预测时长或删减模型调用。

H200 FA3 GQA 的完整 source-call factory 同样在既有 prediction-cache run scope 内复用不可变 KernelSpec recipe。键保留全部 shape、source attributes、控制参数及其类型，并绑定 HWSpec 身份与 SM 数；可变输入仍调用原 factory 校验和构造。LRU 的256项仅限制仿真器元数据，不是硬件容量或精度参数。该复用不删除任何层或调用，不改变 GPU 价格、Graph/Host 时序、激活统计或 provider 支持范围。

H200 GLM 的 vLLM 0.30 BF16/FP8 source 在 `fast` 模式、TP8/EP1、MTP-off、每请求单 token decode 域内记录完整 source，后续从真实一层前缀绑定当前 embedding、Attention/KV 字段和全部 MoE 路由。完整节点、通信及所有 rank 的存储账本继续保留；模板存储与事件为只读快照，绑定结果关闭后不能继续添加算子或分配。默认详细模式保留原有完整调用表示。该普通 source recipe 独立于双 rank native recipe 的适用域，prefill、mixed 及其他并行/推测配置继续按各自既有 source 路径执行。

该 GLM decode 域首次执行完整 lowering 后复用其调用 DAG；每步先验证所有当前 routing packet，再按精确有序 expert histogram、Attention/KV 几何和 embedding rank/shard 契约合并相同的动态物理类，仍由共同 ordinary lowering dispatcher 生成当前 KernelSpec。调用数量、variant、硬件或资源结构改变时重新编译。时间线仅在已验证不变的结构上复用调度元数据，当前预测时长仍逐步递推；这些缓存均限定在同一实验，不落盘预测或时延 LUT。该优化不增加 normal 容量、实际 provider 或正 prefix-hit replay 的支持承诺。

已关闭的原生绑定结果持有只读 storage 和事件序列；相同不可变输入复用生命周期统计，避免每步重建分配列表、验证全部事件和深度哈希。整个生命周期峰值与 capture 严格窗口峰值分别缓存，保持各自原有统计范围。原生执行内存投影首次完整校验后，只在运行内保留独立的标量与不可变元组事实；每帧仍绑定并验证所有 rank 的当前 KV、槽位和 backing。公开节点、边、存储和 per-call receipt 是独立导出，保持物理 allocation/view ID 与字段等价，不承诺跨快照的 Python 对象身份；修改旧导出不会改写内部模板，非法导出重新提交到公共输入边界仍须被拒绝。未由当帧绑定显式更新的可变嵌套元数据使用完整投影；显式动态元数据每帧读取并独立复制。动态分配、事件、窗口或模板改变时重新校验和计算；普通可变 trace 仍保留原有记录行为。

虚拟Tensor只替换模型建图，不另建性能预测入口。`ServingExecutionContext`将所选预测方法原样传给共同的`lower_workload → KernelPlan → predict_kernel_plan → predict(KernelSpec, HWSpec, method)`，已有普通算子继续复用原launch选择和predictor。方法支持由每个KernelSpec与kernel owner判断：显式无launch的数学程序可以只支持Roofline，Midlayer调用必须在该算子处明确拒绝，不能在Serving入口统一拒绝所有虚拟图，也不能静默换方法或重建另一套模型图。

同一`cuda_eager_inplace`场景可显式绑定`model_execution_version=vllm_0_11_0`与`vllm_0_11_plain_greedy_bf16`，复用普通A800 BF16 Qwen/Llama、colocated TP1/TP2、同步eager PP1/EP-off/MTP-off的完整forward。embedding、row-select、TP head及greedy均透传实际模型版本；0.11 TP embedding使用native mask→INT64 gather→原地mask→all-reduce，不能误用0.30的CUDA融合分支。方法仍传给共同kernel接口，具体无launch节点自行拒绝Midlayer。未迁移的默认source及其他旧模式不由该显式场景自动通过。

生产CLI在已实现的版本/模型/硬件及执行配置范围内自动选择共同虚拟forward和已声明greedy场景，包括A800普通0.11 dense、0.11 GLM4，以及0.30 dense/MoE、GLM、K2.6和V4的已有适用分支。调用方仍须提供routing、GLM workspace/indexer或V4物理分支等必要事实；自动选择不补未知字段，不把0.27.1模型源码与0.30 Scheduler身份混同。其他旧模式继续记录迁移缺口。

程序化Serving与typed静态probe也可以在H200、colocated、vLLM0.30、显式BF16 plain greedy和完整MLA构造参数下省略`symbolic_forward_source`：共同binder按GLM5.2 BF16、官方GLM5.2-FP8/5.3或准确K2.6配置选择已实现的同一个source程序。context/batch上限、routing、GLM indexer/workspace历史和逐step token输入仍必须由调用方提供；缺少GLM历史或不匹配模型/部署会拒绝，不回退旧builder，不选新的sampler或provider。该归一化不新增模型或MTP能力，各forward原有几何与分支校验继续执行，normal容量和真实后端验收独立保留。既有H200 PD配置也逐pool复用这些source：GLM TP8/EP1或EP8与K2.6 TP=EP8，PP/SP/CP/DP1、graph/MTP-off、角色同时包含Prefill和Decode，所有pool都须属于相同的已实现硬件/源码域。两侧的图、lowering、预测和同forward storage与相同pool的colocated执行一致；handoff继续由原request-scoped KV owner处理，不复制模型图或修改传输字节。CPU0.30 metadata bridge复用准确GLM TP8/EP1或EP8及K2.6 TP=EP8、MTP-off的原Request/完整KV group处理；GLM/Kimi的EP只划分FFN专家，DP1不触发MoE SP，逐层BF16 MLA及GLM indexer cache规格与EP-off相同。GLM仍在下一层input norm前归约MoE partial，Kimi保留其原FFN内late sum；共享producer-region及EP ownership/port校验，不复制一份建图或传输公式。process/bridge/allocator共用专家cache域校验，报告读取native CPU config的effective TP/EP/SP并与虚拟forward核对。其他并行、Flash native PD及实际connector仍按各自边界拒绝，不能由metadata请求完成升级GPU/provider或normal容量。

静态operator probe可显式传入逐phase的`RuntimeInputs`，直接复用同一`build_symbolic_workload`；phase集合、source与token IDs必须完整，拒绝与旧独立参数冲突，不补workspace历史或构造容量。probe逐call只通过公共`predict`判断Roofline计价；可选资源诊断缺少实现时仅记录limitation，不覆盖已成功的预测。旧source范围仍标记`legacy_unmigrated`，该接口不代表其余旧入口已迁移。V4静态CLI提供`--v4-source-max-batched-tokens`后，要求已有context cap、eager/modular routing、DeepGEMM分支、plain greedy及显式或已声明synthetic token IDs齐全，再通过同一typed入口运行；不能从scheduled rows推断实际构造容量。长scored indexer另需`--v4-indexer-selection-source declared_causal_topk`适用域。

Compact结构对齐可从捕获的首个decoder输入hidden shape/dtype调用同一个`Qwen3Model/LlamaModel.forward(inputs_embeds=...)`，继续复用layers、final norm、行选择和LM head，不需要生成token IDs或执行embedding/sampler。当前接线覆盖既有NVIDIA dense BF16 TP1/TP2、Qwen3MoE TP2/EP2、显式0.11/0.30来源，以及C500/Ascend BF16 dense TP1/TP2和Qwen3MoE TP2/EP2的0.24/0.18捕获对照已有0.30 CUDA-alike/Ascend数学源码场景；捕获与有效源码版本身份分别记录，不声称实际runtime等价。dense TP1/TP2捕获hidden若有padding，必须与显式graph配置解析出的物理行数一致；复用既有attention有效行与物理行分离程序，KV只写逻辑行、head只消费采样行，不执行真实capture或虚构allocator保留量。无padding时hidden行数等于逻辑调度行数；TP2从捕获的embedding all-reduce输入端口开始，校验每rank首collective及其完整payload，保留该collective后复用同一个inputs_embeds模型与TP head，不需要embedding之前的token ownership。MoE复用捕获histogram和已有专家factory；内部top-k节点保留在生产图中，仅在对应模块捕获投影中排除，0.30路由数学来源与捕获的模型版本分开。原生复合RoPE记录实际模块调用边界，结构投影折叠为一次rotary，内部读写与生命周期仍完整保留。配置中的graph bucket不能代替实际捕获port的行数。结构比较保留原有FP16/BF16存储宽度等价政策，同时分别记录captured dtype与model dtype，不声称数值精度等价。比较顺序来自虚拟forward已记录的各rank实际调用序列，collective使用合并后的canonical ID；不以任意拓扑排序替代调用顺序，也不添加虚假执行依赖。此结构对齐入口已删除旧builder分支及旧拓扑顺序投影；未绑定的model/version/hidden契约明确unsupported，不回退旧图。逐phase导出builder与source scope；结构匹配不证明launch几何，未绑定的launch在diagnostic中明确unsupported，不能用空几何冒充已绑定。

激活峰值为整个程序执行期间同时存活的独立 storage 字节数之和的最大值；view 和 Python alias 不重复计数，模型调用方保留的张量不能在子算子返回时提前释放。权重、持久缓冲和 KV 单独列账。普通模型调用与启动时模型/采样 profile 是不同的程序，后者应在同一生命周期中保留跨调用对象，而不是直接相加各组件峰值。编译后的释放、异步完成和未知库 workspace 需要相应来源，不能据源码最后一次读或日志差额补默认值。

Flash TP4与Pro TP8的启动存储可通过同一`report_activation_memory.py --source-only --program symbolic_startup`入口，显式选择`vllm_0_30_v4_sync_no_metadata_pytorch_sort_fp32_noise`。同时提供`--tensor-parallel-size`、`--max-model-len`、`--max-batched-tokens`、`--max-requests`、`--v4-source-use-deep-gemm`、`--source-indexer-max-logits-mb`、逐rank的`--source-workspace-initial-capacity-bytes`、routing identity/seed及H200配置。完整模型forward复用普通V4实现，attention仅适配源码无metadata的pad、workspace预留和zero-output分支；共同worker/sampler程序及通用storage扫描负责启动峰值，每rank输出激活峰值、profile范围总峰值、结束保留量和transient source headroom。无真实Tensor/GPU或数值计算；source-only storage程序不允许作为已完整计价的请求图，不能导出V4正常容量策略或冒充实际provider/日志验收。

虚拟模型入口通过 `--symbolic-forward-source cuda_eager_inplace` 选择 dense BF16 Qwen3或Llama、H200、PP1/EP-off/MTP-off、greedy 的源码场景；TP分片使用共同的rank-local投影与checked collective，不手拼依赖。Llama不添加Q/K Norm，其MLP保留源码覆盖局部变量的生命周期及未量化`maybe_fused_act_quant`分支。`vllm_0_30_v4_sync_fp8_ds_mla`选择明确声明eager metadata、modular路由和DeepGEMM分支的Flash TP4或Pro TP8场景，长C4还需显式输出cardinality适用域。二者复用真实vLLM0.30 CPU Scheduler及既有Roofline lowering。标准Serving报告只使用共同虚拟Tensor路径，已删除旧图选择开关；未迁移模型/硬件/模式明确拒绝，不自动回退。独立生产CLI尚有待迁移的旧模式，不代表旧builder已全部删除。新场景保留原reserve字节并降低其适用性证据等级，报告分别记录captured/effective配置和仿真/汇总snapshot。独立的启动存储入口为`scripts/evaluation/serving/report_activation_memory.py --source-only --program symbolic_startup`，需要明确token/request容量和source profile；缺少数学计价契约的storage-only节点会被性能lowering拒绝。这些入口不证明实际GPU/provider执行，也不自动将已有日志中的显存reserve替换为解析值；完整预算与可比日志检查仍需分别满足。

Qwen3-30B-A3B复用dense Qwen的GQA、residual、目标链和TP原语，MoE只增加BF16 GateLinear、显式softmax top-k与typed专家边界。准确config的norm_topk_prob必须是显式boolean；FP32权重、INT32 IDs和第三个INT32 token_expert_indices的包装局部生命周期分别记录。单独路由替换专家recipe里恰好一个旧top-k阶段，不能重复计或无价省略；源码选择服务未绑定时继续拒绝完整价格。普通TP1 embedding沿VocabParallelEmbedding.forward的input_.long()增加一次INT32到独立INT64的转换，TP融合embedding保留其原分支。

Kimi K3通过同一虚拟Tensor/层循环保留完整93层文本模型：69层KDA、24层NoPE MLA、Attention Residual、latent MoE、词表和普通greedy输出。source profile为`vllm_0_30_k3_mxfp4_sync`，保留BF16计算、MXFP4/group32专家权重、KDA BF16 conv/FP32 recurrent状态和MLA KV；普通CPU vLLM0.30 Scheduler桥负责请求准入、连续decode、完成和释放，动态KDA state indices与MLA slots按层/rank绑定。完整请求支持域是H200 colocated TP1 PP1/PP16、fresh prompt至少512 token、每请求1–8个输出token的首轮小规模验证，输入加输出不超过构造期`max_model_len=640`；TP1仅是有测试预算的数学主链，不是单卡容量部署。PP16使用前两个stage和末stage各5层、其余13个stage各6层的默认分区及相邻stage P2P。TP16另有按rank切分权重和混合缓存的容量报告，但尚不支持完整请求或source-memory执行。非64整数长度的KDA尾chunk按完整tile价格近似，masked lane精细服务和更长chunked prefill未验收；多模态、MTP、prefix-cache组合和实际GPU/provider不在范围内。

普通报告入口如下，环境变量必须指向已有独立vLLM0.30 SDK和源码，不下载模型权重；`--prediction-method`显式选择`midlayer`或`roofline`：

```bash
MIDLAYER_VLLM030_PYTHON=/path/to/.venv-vllm030/bin/python \
MIDLAYER_VLLM030_SOURCE=/path/to/upstream/vllm/v0.30.0 \
python scripts/evaluation/serving/report_kimi_k3_serving.py \
  --layout pp16 --prediction-method midlayer \
  --prompt-tokens 512 --output-tokens 4 \
  --output experiment_logs/serving/kimi_k3/pp16_minimal.json
```

统一Serving CLI通过`--symbolic-forward-source vllm_0_30_k3_mxfp4_sync`选择同一主链。TP1 PP1使用明确标记的数学byte budget，不能声明单卡容量；PP16必须绑定`configs/runtime_memory_policy/nvidia_h200/kimi-k3-pp16.json`并显式设置`--prefill-pp-size 16 --pipeline-ranks-per-node 8`，跨节点链路按实际placement计价。两种布局都要求H200、colocated、EP-off、MLA uniform、analytical launch和版本匹配的plain greedy，不支持静默回退。

重复`--prompt-tokens`可输入少量并发请求；`--layout tp16 --capacity-only`选择另一套容量配置，`--capacity-only`与`--source-memory-only`分别生成原生混合缓存容量计划和已知源码activation账本。所有KernelSpec仍经公共`predict`计价；任何关键价格缺失都会fail closed。完整请求报告只有在输出token齐全、未计价调用为0且时延为正时写出，逐步列出variant覆盖、关键路径贡献、解析FLOPs和仿真wall time；跨rank/调用的预测时间求和仅作诊断，不是阶段时延，source聚合价格未展开的typed logical bytes记为不可用而非零流量。路由是显式seeded assumption，反馈仅使用长度，不是真实数值生成。

PP16的解析容量逐项保留权重、原生混合状态arena、KDA/Marlin工作区、P2P、worker、allocator、fragmentation和framework。KDA按并发相位显式计算q/k、w/u/kg、v_new、chunk_h、FP32 final state/g、Aqk、beta及chunk metadata；varlen chunk数按`floor((tokens+63*requests)/64)`取上界，2048 tokens、两请求为33 chunks，KDA上界为538,706,472 B，KDA+Marlin为1,202,504,400 B。PP按stage分别核算：`worker.intermediate` persistent hidden/residual使用`ceil(start_layer/12)`个AttnRes块并归入framework，最严rank为264,241,152 B；模型临时block-residual使用`ceil(end_layer/12)`个块，PP transient按rank归入activation，最大同为264,241,152 B。容量校验从已知activation峰值扣除PP临时witness后加入其构造上界，避免重复，通信reserve保持128 MiB，无需增加总预留。只有完整绑定策略和全部分项均通过才接受。该结论仅是bounded source预算，不是实机容量或provider验收；H200实测精度与实际GPU运行仍未验证。非64尾chunk近似、TP16仅容量报告等限制不变。

V4的0.30缓存工厂按准确checkpoint匹配Flash/TP4或Pro/TP8，均限定PP/CP/DP1、EP/MTP-off、BF16 compute和fp8_ds_mla cache；从各自compress_ratios的target层建立SWA、compressed KV、indexer及FP32 state groups，不为draft-tail建立额外缓存。相同native allocator与placement校验接收逐rank预算，Scheduler桥接保留真实group/arena对象；错误TP、缺失policy和旧GLM-only入口继续拒绝。源码权重/缓冲与完整normal policy是独立条件，显式不完整测试policy不得升级为正常容量或provider验收。

Kimi K2.6的迁移复用同一虚拟张量和基础算子。`DeepseekV2MLAInput`沿vLLM0.30普通`MultiHeadLatentAttentionWrapper.forward`的unfused RMSNorm分支记录融合Q/KV-A投影、两个RMSNorm、Q-B与原地RoPE；context保留wrapper局部引用，`unsqueeze`只插入视图维度并保持原stride/offset，包括单行切片。准确checkpoint虽整体为mixed INT4/BF16，attention、dense/shared FFN的权重按真实module path排除量化，路由专家仍为INT4/group32，不根据全模型storage标签拒绝合法BF16模块或把专家改成BF16。`DeepseekV2MLP`继续复用同一forward body，构造时必须显式声明`reduce_results`：Kimi普通dense的down-projection通过公共`SymbolicRowParallelLinear`在自身frame内完成TP all-reduce；shared和GLM已声明分支保留partial。归约必须提供完整同rank组及对应collective配置，不能缺省跳过。Kimi的全decode注意力caller进一步通过`DeepseekV2MLADecodeAttention`接上BF16 cache append、token-major Q absorption、FlashMLA query concat、latent输出及FP32 LSE、V-up out=和O投影内TP归约；必须显式声明`vllm_0_30_flashmla_bf16`条件源码分支，不能推断实际provider。底层cache是声明的逻辑pool，源empty(0)依赖只表示控制顺序与零payload；实际页表、scheduler metadata、library scratch/stride和GPU launch未绑定。`bf16_mla_decode_softmax_v1`仅给共享latent KV的有用数学及参数payload，LSE沿上游test_flashmla.py的FP32 logsumexp参考计算；compiled指令不由此验收。`trace_kimi_k2_decode_forward`通过显式`vllm_0_30_k2_6_int4_sync_explicit_allreduce`接通全部61层all-decode与可选plain greedy目标链；共享已有residual decoder/model forward及GLM的typed target/路由身份helper，不另拼DAG或释放表。Kimi的BF16 gate和router factor1保持，INT4/group32专家输出再原地乘checkpoint routed_scaling_factor，随后shared-add和late TP all-reduce；GLM原先在router施加scale的分支不变。CLI使用该source时必须给准确K2.6配置、TP8、context/batch限制与显式routing假设，target链由同一`--target-sampling-source vllm_0_30_plain_greedy_bf16`选择。源码专家复用原Marlin定价，但不升级为0.30 native绑定；每个全模型case的480个Kimi route仍拒绝完整计价，整段时间null且CLI非零。prefill/mixed、Serving、完整source/worker内存/normal容量及实际CUDA provider仍未验收。

GLM BF16的`DeepseekV32AttentionInput`适配vLLM0.30 attention.forward到sparse/indexer backend调用之前的源码前段：fused A投影、full层独有的indexer投影、融合norm/RoPE/cache写、全部输入行的Q absorption及融合query准备。它是显式作用域context，调用方在作用域内执行后续backend，使原始投影和index Q等Python局部引用自动存活；不手写保留/释放表。`DeepseekV32Attention`在明确CP1、TP8、未padding单token全decode及`declared_causal_topk`适用域内连接indexer、arena query拼接、latent sparse数学调用、V absorption与未all-reduce的O投影。共享层复用同一个top-k，不重跑indexer或clear；所有层通过同一显式入口history的workspace提供64-head query backing。transposed view及out=自动保留原存储，latent actual-head view保留64-head输出backing。逻辑页转换、backend第三返回值和callee scratch仍是显式缺口；该attention组件本身不包括层循环、prefill/mixed、官方FP8量化、采样或Serving dispatch，不能单独作为完整模型或正常显存容量验收。

`trace_glm_dsa_decoder`进一步把准确GLM5.2 BF16配置的78层model.forward接到同一虚拟图，包含3层dense与75层MoE、最终norm及TP8归约；当前限定全decode、EP/MTP-off、PP/SP/CP/DP1和显式workspace入口。dense/shared FFN共用已有MLP forward body，按TP切分宽度，使用普通SiluAndMul；router为BF16 A/B与FP32 C，独立grouped route生成FP32权重和INT32 IDs。声明modular/non-overlap、shared不融合及普通all-reduce→RMSNorm源码分支，O投影和MLP均保留partial输出，归约由下一层input norm、当前post-attention norm及final norm持有。`SymbolicAllReduceRMSNorm`用自身Python frame保留reduced局部引用，不把它额外延长到后续attention/MLP；相同get_rope参数共用缓存对象。MoE内部workspace仍未完整表示，decoder范围也不包含词表/采样、prefill/mixed、官方FP8或Serving dispatch。

上述attention组件同时支持官方GLM5.2-FP8与GLM5.3的同一结构/量化配置，按准确module path解析FP8排除规则；indexer的wk/weights投影保持源码显式BF16，Q/KV-A、Q-B、indexer Q-B及O投影复用通用block-FP8虚拟Linear。未分片融合A投影的最后一个输出块可不满128，FP32 weight scale按ceil(N/128)计，不能补成BF16或删掉尾块。显式`block_fp8_fp32_scales_v1`数学GEMM只给FP8 A/B、BF16 C、FP32逻辑scale与2MNK算量的Roofline，前置量化复用独立无launch源码程序；两者均保持execution_support=unsupported，拒绝实际launch、FP32输出冲突与midlayer/native计价。虚拟临时量按quantizer→C分配的wrapper顺序存活，scale padding及library workspace未绑定；这不是完整worker峰值。官方FP8的完整静态decoder/target现由下述独立source profile选择；生产分派复用同一forward，但路由拒绝仍阻止完整请求执行，不把组件或静态图通过升级为正常容量与请求验收。

官方GLM5.2-FP8与GLM5.3通过静态`vllm_0_30_v32_fp8_sync_explicit_allreduce`共用同一78层decoder/target；source profile必须与checkpoint权重类型一致，不能以BF16 profile接FP8模型。dense/shared MLP保留同一forward body，仅按准确module path选择BF16或block-FP8基础算子；GateLinear仍为BF16 A/B、FP32 C。`SymbolicRoutedExperts`记录checkpoint权重类型、完整专家ID轴、FP32权重/INT32 IDs及partial输出；FP8按显式`block_fp8_swiglu_v1`有用数学程序计W13、普通SwiGLU、W2和加权top-k和，量化复用既有无launch程序。该数学程序不选择实际provider、tile/padding或workspace，也不借用只覆盖DeepSeek-V3几何的旧H200 FP8 launch表。两款官方配置只对显式routing assumption共用已有结构语义身份，原checkpoint/config及真实replay绑定各自保留。grouped route仍独立存在且未完整定价，整段时间继续为null；FP8 FFN内部临时区未知，不能把已表示峰值当正常容量。

同一GLM decoder/普通greedy目标链也支持H200 TP1：Embedding按源码先将INT32 IDs转INT64，无TP collective；attention/MLP复用相同模块，final直接RMSNorm，head保留完整词表，workspace历史必须恰好对应一个rank。官方FP8 family的target-off比较使用这一共同路径，显式记录每phase独立workspace、构造容量及合成token场景；原native-MTP静态比较也通过显式逐phase场景接入下述共同路径；CPU Scheduler的逐step场景供给与正常容量仍独立未验收。逻辑Roofline完整计价与native launch支持分别报告：仅完整共同目标链、已声明sampler且全部调用有价时标为逻辑priced，未知provider不因此获得native通过；旧partial source不能据此升级。

GLM三款的legacy eager padded MTP用`V32PaddedMTPInputs`声明完整请求顺序、两pass独立draft IDs、discard场景、逻辑block size及sequential/full-logits/greedy条件，复用同一78层目标模型和附加V32预测层。target与draft均保留spec2，indexer阈值为3：q2是decode，q8是prefill，mixed按两组各自形状建数学调用。目标raw-logits保留、greedy rejection及padded prepare与V3共用控制helper；pass0计算full top-k，按retained rows进行真实gather/copy compact，pass1只读shared旧选择，KV增长不增加旧count。V32预测层final all-reduce/RMSNorm后的同一hidden既供head也供下一pass，不再额外head norm；TP1沿fused embedding/EH，TP8沿分片embedding/all-reduce再fused EH。source typed ports记录每次分配、别名与写后依赖；无真实Tensor计算或GPU。`SimulationConfig.model_source_inputs_by_step`可冻结逐step的独立输入，执行时必须与当前请求顺序、token行及草稿深度一致；缺帧、重复owner或复用run-level旧值会拒绝。真实CPU0.30桥只接显式fixed-count/spec2/sequential/padded/full-logits/greedy GLM场景，使用真实SpeculativeConfig及目标78层加附加层78的latent/indexer cache owner，沿原Scheduler→Simulation.run执行完整请求。该接口不推导未知初始draft buffer内容，不把逻辑页坐标当native分配；显式测试KV预算、CPU调度和源码Roofline不证明正常容量、实际GPU/provider或最终Serving验收。

GLM三款的静态prefill/mixed可额外显式声明`--glm-prefill-attention-source force_sparse_mqa`，对应vLLM0.30已有的`attention_config.sparse_mla_force_mqa=True`；没有该声明仍拒绝多token prefill/mixed，不根据shape猜测默认dense/masked-MHA分支。模型输入复用同一vLLM0.30 decode-prefix packing程序，按token-row permutation重排IDs、positions和slots，目标logits取每个已打包请求的累计query长度减1；不增加源码没有的GPU置换copy。全部输入行仍执行Q absorption。强制MQA域继续用BF16 latent cache、共享top-k与64-head backing，按每query的因果长度计算有用indexer/sparse注意力工作；此前decode程序的适用域不扩大。prefill indexer与Flash共用FP8/UINT8 gather布局及同rank workspace管理器：GLM容量为runtime_max_model_len*40行，Flash C4再除4，scale保留零拷贝FP32视图。constructor保留的query backing、arena增长重叠及多层复用由虚拟引用自动记录；decode-only不请求该gather。可加`--upstream-vllm-root`，独立读取上游constructor/helper AST检查已有prefill caller的两个端口和scale alias，不代表完整调用清单验收。CP1 prefill的数据gather逻辑payload按每请求完整KV一次计价，query子块复用不重复计；候选数不超过top-k的行按源码不读score，但仍保留score producer、两份INT32 row-bound读取及完整输出capacity的索引/sentinel写。实际chunk划分、矩形score backing、非捷径selector与内部workspace仍未绑定，不能写成0或完整worker峰值。路由仍拒绝完整定价，整段时延仍为空；不接受为真实Scheduler请求、normal容量、默认vLLM分支或native/provider通过。

静态`scripts/evaluation/serving/report_model_operator_probe.py`已删除`--symbolic-forward-source`路径开关：显式版本化target sampler与模型类型/权重dtype共同选择已有GLM BF16/FP8、K2.6或DeepSeek-V3源码场景；不能仅凭模型配置推定实际backend。必须继续提供TP、上下文/batch限制、明确token输入，GLM还需`--glm-indexer-selection-source declared_causal_topk`和每rank workspace历史；native MTP及PP仍须满足各自完整输入契约。普通dense/MoE未声明sampler时保留原LM-head-only输出，不能通过完整forward gate。程序化逐phase RuntimeInputs也复用生产source绑定，不因缺路径标记回退旧builder。RuntimeInputs保留源码身份与constructor/cardinality/workspace事实；报告记录source_projection_only及runtime/provider未绑定。输入不完整或不匹配时拒绝，静态逻辑Roofline计价不等于Serving或正常容量通过。报告逐项记录未计价调用、variant和拒绝原因；任一调用被拒绝时`serial_roofline_seconds=null`且CLI返回非零，不用0成本补齐、也不返回部分和充当完整时间。`execution_scope`和`full_model_graph_complete`区分decoder与含词表/采样的完整模型；生产`predict_kernel_plan`继续fail-fast。正常容量、完整backend/worker存储、日志10%和native/provider仍独立未通过。

在该GLM静态source上显式加`--target-sampling-source vllm_0_30_plain_greedy_bf16`，或调用`trace_glm_dsa_forward`，才包含decoder之后的普通目标链：INT32 query_start_loc行索引→fresh BF16 sample_hidden_states→BF16 TP LM head→CUDA默认all-gather→词表padding切片→FP32 cast→INT64 argmax→INT32输出。`SymbolicTPLogitsProcessor`把网络rank-major backing和last-dimension API顺序分开：多行时对reshape/transpose后的view物化一次row-major输出，单行直接view；所有alias与内部局部引用仍由同一trace自动记录。重排前的backing在helper返回后释放，不能误保留到FP32 cast。lowering同时校验所有rank的head producer、行/分片宽度、collective字节和适用的rank-local重排。`full_model_graph_complete=True`只表明模型至采样的图端点齐全，不代表调用全部计价、源码全量验收或Serving通过；现有grouped route拒绝仍使整段时间为null。INT32 index的ATen内部转换、compiler/library scratch和metadata producer仍未绑定，不能把caller账本当完整worker峰值。

Qwen3 TP1与V4 Flash TP4/Pro TP8的普通目标链也使用相同`SymbolicLogitsRowSelect`和`SymbolicGreedy`，V4的TP head共用`SymbolicTPLogitsProcessor`。普通非MTP `query_start_loc`产生INT32行索引，不能沿用一般gather的INT64索引宽度；索引metadata在模型forward前进入同一source scope，identity选择也产生fresh输出。Qwen仍用已有未量化本地Linear head，V4的多请求输出按源码增加每rank一次gather重排，单请求不增加复制。worker启动`_dummy_run`的NumPy cumsum索引属于另一个明确的INT64源码程序，不能随普通请求路径一起改写；既有startup采样预算与其未验证边界保持。caller metadata、完整worker lifetimes和库临时区仍需单独闭合，不由这次共享迁移获得正常容量或native验收。

Serving执行入口通过symbolic trace observer在同一次forward完成后读取canonical allocation lifetimes；observer接收的WorkloadGraph必须是随后lowering的同一对象，不能为显存另跑模型。`ServingExecutionComponents`和`SimulationStep`的可选`source_activation_memory`保留各物理rank已表示storage的峰值及当时活跃对象。结果中的同名汇总字段保存逐步标量峰值、覆盖步数和每pool/rank的一个最大值见证；完整或streaming保留及CPU并行worker路径使用相同来源，不长期保留完整trace。未选择虚拟forward时字段缺失或为None，不能解释为显存零值。

外部canonical requests和Production Trace的持久化结果继续省略请求级内容；`source_activation_memory`保留覆盖情况、source profile和各pool/rank峰值，省略逐步series、step ID及allocation/event见证。返回的内存结果仍保留这些明细，生产容量验收标记不因持久化投影变化而升级。保存的JSON可直接用于Serving Markdown报告。

目前这些源程序快照都显式标记`complete=False`和`normal_capacity_accepted=False`。它们不包含尚未表示的workspace、compiler/异步runtime状态，也不代表真实物理KV或跨forward异步峰值；各rank的独立max不能相加充当单卡容量。新增统计不改变KernelSpec价格、时延验收或runtime memory policy，只有完整预算和可比启动日志验收满足后才能迁移正常容量。

Flash的Serving入口可用`--source-workspace-initial-capacity-bytes`声明每个TP rank在本次仿真入口的单lane、unlocked共享arena容量。未知入口仍为None；显式四个0表示调用方声明的cold source场景，不能解释为实际worker经过warmup后的状态。通用workspace在同一次forward记录最终容量；执行context只携带这些标量到下一步，保留扩容后的arena，避免小decode把容量重置成cold或忘掉先前大prefill。对这种跨步状态依赖，缓存计价仍可使用，但CPU tail使用顺序执行，独立fork worker不得分别重启入口历史。该声明不补齐其他workspace、锁定/异步lane或完整生产容量，也不增加实际provider验收。

长C4的indexer prefill也复用该manager，其FP8 gather及UINT8 scale port按源码constructor预留10倍runtime max_model_len行，FP32 scale只是同一port的typed view。存在空压缩prefill组时仍预留这些空间，但不添加不存在的payload读写；decode-only不发出prefill gather请求。这些port与main sparse及expert arena按生命周期共享，不按43层累加，不把预留字节转写成访存流量。外部score结果的backing和selector内部存储仍未绑定。

启动账本在 profile 开始和返回后记录测量边界，从同一组生命周期计算源对象总峰值、结束存活量及其差值。后来创建并保留的采样 cache 参与运行中的峰值，也参与结束存活量；不能在每个时刻一律排除它们。原生 Triton 采样的 buffer 增长、较小 view 保留完整 backing、共享查表和 split workspace 由虚拟对象及 cache 容器的引用自动记录，SM 数来自明确的 HWSpec，不读取 GPU。

`run_serving_simulation.py --activation-source-profile` 在真实 CPU vLLM0.30 allocator/Scheduler 入口调用同一生产 owner `resolve_symbolic_kv_capacity_plan`，将源码生命周期预算用于物理 KV 容量计算，不需要指定测试块数。当前可选的是 BF16 Qwen3/Llama、A800/H200、TP1或TP2/PP1、无 MTP 的 eager CUDA startup 源程序；CLI 与 policy 的模型、硬件、并行和全部 constructor limits 必须一致。结果 `kv_capacity.source_lifetime_projection` 保存既有 schema2 的 projected policy 和每 rank 字节账本。非激活预留仍保留外部来源，未知 workspace/runtime 开销及缺少可比启动日志会明确阻止正常容量验收；有正数 KV blocks 和请求完成不代表这些检查已通过。Llama复用公共worker/sampler与自己的forward frame，保留无QK norm、CUDA RoPE以及down projection前释放gate-up临时量的源码差异；既有framework预留若低于该source的已表示驻留下界会继续拒绝，不补差额。TP2对两个rank分别执行同一profile、记录独立worker与sampler cache；TP head经源码all-gather后每rank仍有完整词表sampler，不能把TP1峰值除2。CLI复用本pool已有collective配置；这只表示declared fresh-output storage，不补齐通信workspace或实际transport/provider。

SGLang0.5.19 的 DeepSeek-V3 FP8、H200 TP8 decode 通过 `build_symbolic_workload` 使用共同虚拟 Tensor forward。`sglang_0_5_19_deepseek_v3_decode_fp8` 声明 Cutlass FP8、Triton grouped top-k、FlashMLA BF16、BF16 absorption BMM 和不融合通信的源码分支；后缀 `_flashinfer_ar_norm` 选择已有 FlashInfer all-reduce/residual/RMSNorm 融合分支。分支参数保存在 source trace，不代表安装或执行了这些 GPU provider。decode profiles保留 INT64 input IDs、BF16 A/B→FP32 router logits、独立 FP32 路由权重/INT32 IDs、共享专家的额外槽和 BF16 logits→FP32 输出。`sglang_0_5_19_deepseek_v3_target_fp8`及其`_flashinfer_ar_norm`分支进一步声明ragged/eager/CP1/no-hook的fresh-MHA prefill，可在同次仿真继续普通decode。prefill的K concat与V cast复用共同算子，attention输出直接进入O投影；head前按源码INT32 cumsum→INT64减1产生末行索引并物化BF16选行，不能对全部prompt行执行head，也不添加vLLM专属output-copy。默认边界截至logits；target profile可显式配对`target_sampling_source=sglang_0_5_19_plain_greedy_fp32`，声明CUDA/all-greedy、无custom processor/logprob/grammar/return mask、NaN assert/sanitize及token-sync关闭的源码域，直接从FP32 logits产生fresh INT64 argmax输出，不增加vLLM专属INT32转换。该参数声明条件，不证明实际worker开关；token transcript仍是外部输入，仿真不执行数值argmax。普通target生产入口进一步声明默认prefix阈值8192：纯prefill且0<sum(prefix)<8192时复用同一decoder的absorbed MLA，保留ragged query长度，只产生latent O而不额外产生LSE。fresh prefill仍走MHA；sum(prefix)>=8192且sum(KV)<=声明chunk容量时，`SGLangPrefillInputs`提供显式combined-KV INT32索引，复用同一decoder执行cache写、分离latent/RoPE gather、全KV扩展和一次causal MHA，无公开LSE或merge。索引必须覆盖完整KV且suffix与当前slot一致，不推导实际req_to_token producer。超过chunk容量时，调用方须提供逐chunk的INT32 cache索引，source依次执行causal suffix和noncausal prefix chunks，再逐次物化O/LSE合并；不等长请求的空prefix行保留零BF16 O及负无穷FP32 LSE，合并仍读取两路完整状态。该数学域不绑定实际FlashInfer provider或compiler流量；其他未迁移mixed/EAGLE分支继续按各自入口边界拒绝。`SGLangSchedulerMirrorBackend` 可在构造时绑定有明确 source ID 的 `request_token_transcripts`，之后复用普通 `add_request(RequestSpec)` 引擎接口；没有 transcript 时保持 length-only，缺少 request 的值会拒绝，不能生成假 token 或冒充模型输出。

SGLang target profile另可接显式的单请求、topk1/steps1 EAGLE源码帧。`SGLangEagleChainState`记录当前bonus、上一轮draft候选、target argmax、tree/accept路径及独立draft槽；prefill执行target和NextN extend，verify执行两行target、tree verify/bonus和NextN extend。prefill carry是INT64 bonus，后续verify产生INT32 bonus；accept_lens包括bonus，Scheduler的mtp_accepted_tokens不包括。`SGLangSchedulerMirrorBackend(model_source_inputs_by_step=...)`按精确step ID绑定这些外部声明，检查请求顺序、连续候选、接受计数及输出transcript；`BackendStep.model_source_inputs`通过公共execution接口原样传到forward，不从长度或接受数恢复token值。所有状态都只作轻量元数据，不执行采样或模型数值运算。此域是显式合成/假设输入下的源码仿真，不是实际SGLang运行、正常容量或provider验收；更大batch、draft宽度和图捕获不随此通过。

`report_activation_memory.py` 的 source-only 模式可显式传入 `--base-runtime-memory-policy` 和 `--derived-runtime-memory-policy`，生成既有 schema2 的候选策略。生产 owner `project_symbolic_activation_policy` 从同次图按结束时存活的storage划分驻留与非驻留：framework覆盖结束驻留，activation取非驻留storage的同时峰值，arena不重复扣除；二者组成声明源码程序的预算界，不是已观测点估计或未知运行时分支的界。V4还必须传`--framework-source-epoch before_profile|after_profile`，明确base framework对应的时刻；before_profile验证已知初始对象后增加已表示驻留增长一次，after_profile验证结束下界后保持原值。Qwen沿用after_profile契约。其他外部预留及来源保留，候选策略始终标记为`unverified_analytical_assumption`，可由既有KV规划/Serving入口读取；profile最终workspace容量可显式传给请求source入口历史。此路径不改写基线配置或默认策略，不将缺失workspace、compiler/allocator、实际provider或其他峰值覆盖标记为完整；normal容量与日志10%仍需单独验收。

Qwen/Flash/Pro的启动图还共用`VLLM030WorkerStorage`，记录已知纯文本CP1/MTP-off的runner预分配设备缓冲与InputBatch常驻sampling数组；`vllm030_worker_storage_bytes`从相同descriptor给before-profile候选基线提供已知驻留字节，不按模型另写峰值公式。CpuGpuBuffer只计一次设备分配，CPU/pinned/NumPy侧不重复扣HBM。source-only CLI可加`--upstream-vllm-root`，独立解析上游constructor核对这组buffer的shape/dtype和分配转发；失败则拒绝导出候选策略。该分组不覆盖block table、可选多模态/投机状态或其他backend/runtime内存，结果仍有完整性限定。

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
| Prefix cache | **Supported/Partial**：默认 trace-hit 长度模式；显式 token-ID replay 支持原生内容缓存 | 长度模式注入命中量；token 模式由真实 vLLM Scheduler 根据完整块内容维护命中、共享与淘汰 | token 模式须提供真实 prompt 和输出 ID；同步 CPU0.11 已验证 TP/PP、MoE 与多实例 PD；MTP 还需逐步真实 draft；复杂缓存与外部 KV tier/lifecycle replay 不支持 |
| 多实例路由 | **Supported**：多 P/多 D 的 `least_queued_tokens` | ingress→P 和 P→D 两级路由在决策时读取可见队列，并立即 reservation | 路由是给定部署下的模拟结果，不是 Trace 中的生产路由；暂不支持 cache-aware routing |
| 部署形式 | **Partial**：colocated、纯 PD、standalone AF、PD+AF 有闭环切片 | role 拆分改变算子 owner，并插入 KV 或 activation transfer 与依赖 | 每个实例独占自己的设备集合；不建模同卡多实例竞争 |
| TP/PP/EP | **Partial**：已有 TP、PP、TP×PP 与 MoE EP 的明确窄切片；既有 Kimi-K2.6 固定 TP=EP=8/PP2；GLM-5.2 BF16 vLLM0.30 支持合法 TP、连续 PP、DP/EP，以及 PP1 MLA PCP 和源码 SP | 先分片 shape/expert/rank，再产生 TP collective、8-lane PP P2P 和 pipeline dependency；对称 TP 计算保留一个 rank-local 代表调用，非对称 MoE 按 8 个 EP rank 展开，相同 KernelSpec 的预测仅在单次仿真内精确复用 | 支持域依模型与版本；GLM PCP 要求 PP1；PP 的 sparse/decode 程序必须有完整 top-k 生产依赖，pure fresh dense-prefill 可绕过该读取；通信为显式解析场景，真实 provider、数值等价与 pipeline overlap 独立保留 |
| MTP | **Partial/Projected**：propose/verify、零/部分接受、短输出、handoff 与 replay contract | proposal 数由配置给出，接受数来自 observation replay 或显式假设，并逐 step 推进 token 状态 | 无 observation 时 fixed-count 只是假设；仿真不预测模型质量或真实 acceptance |
| 通信 | **Partial/Projected**：all-reduce、all-gather、all-to-all、pipeline/activation P2P | 通信作为带 rank、bytes、拓扑和依赖的一级工作负载进入统一时间线 | 当前解析后端可用；外部 ProtoBridge 尚未成为已验证的生产后端 |
| Host-KV offload | **Partial/Projected**：显式 LOAD/EVICT、容量账本、PCIe 区间和完成事件 | lifecycle 先产生迁移 intent，传输完成后才改变 KV residency | 尚不等于原生 vLLM offload；真实 victim/LRU、prefetch、Ramulator、SSD/Flash 和完整重叠未闭合 |
| Router CPU dispatch | **Partial/Projected**：多 pool 路由可使用一个串行 CPU dispatch 资源 | 每个 route decision 在进入目标 pool 前预约一段 CPU service interval | 当前只覆盖 Router dispatch；Scheduler、sampling 和 output processing 的通用 CPU 成本尚未闭合 |
| 模型 | **Partial**：Dense、MoE、MLA/MTP 共用主架构；Qwen3、Qwen3MoE、Llama、GLM4-MoE/MTP 有闭环切片；GLM-5.2 与 Kimi-K2.6 已有固定 A800 多卡 production trace 路径，GLM-5.2 另支持 projected MTP proposer/verifier 路径 | 固定模型语义生成层、attention、expert、MTP 和 KV workload；固定 vLLM 0.11 CPU Scheduler 负责控制面，0.27.1 固定源码负责目标模型执行语义；目标路径使用真实 TP/EP/PP placement 与解析 KV 容量 | 两个目标模型的新增 kernel 均只是 projected analytical，未做真机校准；vLLM GPU MTP/KV-offload execution 和跨硬件 target 路径仍未闭合，不能声称端到端硬件精度支持 |
| 硬件 | **Partial**：A800、C500、910B 可经过同一 Serving→Kernel 接口 | 每个实例绑定自己的硬件规格和 launch package，Kernel 层返回时间与 limitation | A800 Prefill→H200 Decode 已有 CPU 解析完整请求与独立容量入口；真实异构网络/provider、其他厂商组合与 LPU 仍待验证，结论继承 Kernel 层支持等级 |
| 系统指标 | **Supported**：TTFT、TPOT、E2E、CDF/percentile、throughput、SLO goodput、queue、资源利用率和关键路径 | 全部从同一条事件时间线与请求状态派生 | success-only Trace 的指标不能解释生产成功率或可用性；SLO 口径必须显式配置 |

这张表是后文的索引：后文分别解释每项能力为什么需要、状态如何推进、如何落成算子/传输和时间线。未列为 Supported 的组合不能通过“已有相邻 case”推断为支持。

### 2.1 Runtime Graph padding 的 shape domain

Serving Graph padding 必须区分 Scheduler logical token、Graph model-input tensor row 和算子恢复后的 active row。Production 入口接收 runtime 最终解析并实际捕获的 `capture_sizes`，在 contract 边界规范化为唯一升序集合，再对 logical token 数向上选择最近 bucket；canonical observation 的 `padded_total_tokens` 和 `capture_bucket_tokens` 只在 evaluation holdout 中核验，不能成为逐 step Forward 输入。未启用 Graph 时保持原有 eager/无 padding workload 逐值不变。

Graph dispatch 是与硬件和软件版本无关的数据驱动规则：配置由 runtime 原始 `configured_mode`、`uniform_decode_query_len`、唯一升序的 `capture_sizes` 和 PIECEWISE 编译器明确排除在 Graph 外的 `piecewise_eager_workload_kinds` 组成。统一 mode table 将 `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY` 和 `FULL_AND_PIECEWISE` 解析为 uniform/non-uniform runtime mode；每个 step 再根据所有 request 的 query length 是否等于 `uniform_decode_query_len` 选择对应 mode，并在存在不小于 logical token 数的 bucket 时令 model-input row 使用最小匹配 bucket。Attention/KV 的 active row 仍使用 logical token 数，lm_head 使用真实 logits selection；KV offload 继续消费 Scheduler 产生的 logical KV block range，不随 dummy model-input row 扩张。所选 mode 为 `NONE` 或超过最大 bucket时 runtime mode 解析为 `NONE`。Host timeline 的唯一 segment producer 按 expanded KernelPlan occurrence 和每个 pool/rank Host lane 生成描述：`FULL` 每 lane 覆盖全部 occurrence 的一次 replay，`PIECEWISE` 将配置的 eager workload kind 作为切分边界并把其间每段连续 occurrence 组成 replay，`NONE` 不生成 segment；source node、lane、repeat 或 coverage 不一致时必须 fail closed。不同运行栈的默认 bucket、configured mode 和 PIECEWISE splitting ops 属于 runtime 配置与 provenance，不进入 Serving predictor 的硬件或版本分支。CLI 使用 `--graph-mode <MODE> --graph-uniform-decode-query-len <N> --graph-capture-sizes <SIZE...> --graph-piecewise-eager-workload-kinds <KIND...>` 传入解析后的实际 profile。非零 padding 目前覆盖 standard-GQA、MTP disabled 的 colocated dense 单卡路径，以及经过 TP=EP=2 contract 验证的 MoE rank-local 路径；其他并行组合必须在 workload/lowering 边界 fail loud。

现有 canonical trace 对应的 resolved profile 是：A800 为 uniform decode `FULL`、non-uniform `PIECEWISE`，capture set 在 8 以上按 8 递增到 512；C500 为 `FULL`/`PIECEWISE`，capture set 在 256 以下按 8、从 256 起按 16 递增到 512；910B 为 `FULL`/`FULL`，capture set 与 C500 相同。这些是传给统一规则的数据，不是 production 分支。

Graph-specific source audit 只保留结论；hash 级 provenance 由 canonical manifest 与 git 历史维护，runtime duration 不进入 Forward。

| 硬件 | Runtime/source | 审计结论 | Production 状态 |
|---|---|---|---|
| Ascend 910B | vLLM `0.18.0` + vllm-ascend plugin snapshot | `config/vllm.py` 解析 capture buckets，`v1/cudagraph_dispatcher.py` 选择向上 bucket 或 NONE；plugin 使用 padded input、actual-token metadata、有效 KV slice 和 logits index compact | 通用 dispatch/shape-domain 规则 Supported；FA dummy-query 额外开销 Partial |
| NVIDIA A800 | vLLM `0.11.0` | V1 config 将最终 capture set 预计算成 ceiling 映射；runner 以 `total_num_scheduled_tokens` 选择 padded model input，Attention metadata 保留 `num_actual_tokens`，slot tail 写 `-1`，lm_head 前按 `logits_indices` compact | 通用 dispatch/shape-domain 规则 Supported；实际 capture set 必须由对应 runtime 配置提供 |
| MetaX C500 | vLLM `0.10.2` site-package snapshot | V1 config、ceiling 映射、runner actual-token metadata、slot sentinel 和 logits compact 与 A800 使用同一机制 | 通用 dispatch/shape-domain 规则 Supported；实际 capture set 必须由对应 runtime 配置提供，canonical 精度验证仍依赖绑定采集环境的 observation |

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
# 历史示例，该 run 已清理；命令形态仍适用于新的 target-model validation run。
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
CUDA_VISIBLE_DEVICES=0 /home/yangxinhao/miniconda3/bin/conda run -n bullet python scripts/evaluation/serving/validate_target_model_blocks.py --vllm-root /share_data/yangxinhao/chip_modeling/upstream/vllm/tags/v0.27.1 --transformers-root /share_data/yangxinhao/chip_modeling/upstream/transformers/v5.10.2 --glm-root /share_data/yangxinhao/chip_modeling/upstream/models/GLM-5.2 --kimi-root /share_data/yangxinhao/chip_modeling/upstream/models/Kimi-K2.6
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
CUDA_VISIBLE_DEVICES=0 /home/yangxinhao/miniconda3/bin/conda run -n bullet python scripts/evaluation/serving/validate_target_model_blocks.py --vllm-root /share_data/yangxinhao/chip_modeling/upstream/vllm/tags/v0.27.1 --transformers-root /share_data/yangxinhao/chip_modeling/upstream/transformers/v5.10.2 --glm-root /share_data/yangxinhao/chip_modeling/upstream/models/GLM-5.2 --kimi-root /share_data/yangxinhao/chip_modeling/upstream/models/Kimi-K2.6
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

显式调用 `Simulation` 或 standalone AF `MultiSimulation` 的 Host KV lifecycle 也可选择 `streaming_aggregate`。含 LOAD/EVICT 的步保留完整时间线直到物理完成事件校验和 tier 状态提交，其余步可使用紧凑时间线；完成后每步 IR 立即释放。流式 schema 16 在原有 ordered execution-contract chain 外记录 KV 完成记录哈希、搬运次数与字节数、最终 tier 状态。固定前缀 HBM/Host 比例是带整块取整规则的显式场景假设，Host 加载后两份有效拷贝均计入容量，到请求释放时一并回收；它不代表原生 vLLM 缓存淘汰或未来请求复用。通用 Serving CLI 的 `auto` 选择和外部 lifecycle replay 组合仍遵守上一段的范围约束。

自主 Host KV 的选定来源是本地锁定 vLLM `0.30.0` 的 `SimpleCPUOffloadConnector`，配置为 `kv_both`、`kv_buffer_device=cpu`、`kv_offload_backend=cpu`、`lazy_offload=false`、token-ID prefix caching，CPU 容量以每 rank 字节声明并乘 native world size 传入 connector。其 eager Scheduler 在后续调度步把已形成 hash 的完整 GPU 块提交 store；CPU 与 GPU 可同时保存副本，store 完成仅使 CPU hash 可查且释放临时 pin，不表示 GPU 块已淘汰。查询先看 GPU prefix，再查连续 CPU hash；CPU hit 给目标 GPU 块创建异步 load，完成反馈前请求保持等待状态。CPU 块容量不足时不提交相应 store，GPU/CPU 块由上游引用计数与 free queue 独立管理。Midlayer 已在普通 `Simulation.run()` 中接入真实 Scheduler 的 store metadata、跨请求 CPU prefix 命中、load-only 等待步、逐 rank D2H/H2D 时间线和完成反馈；CPU slot 重用先释放旧 Host 副本，请求结束后的 GPU 源页按原生 pin 状态延迟释放，纯传输尾步走同一 KV offload kernel/timeline。已运行的范围是显式 prompt/output token IDs、单组整块 KV、eager CPU、单 pool Qwen3 TP1/TP2 的宽松/紧 Host 容量短请求及零输出尾步；不据此声称一般容量或实际 GPU 验收。固定比例 Trace 回放不使用此自主策略。

固定比例 Production 入口 `evaluation.serving.experiments.production_trace_fixed_tier_request` 需要显式 `--hbm-percent`、容量、`--max-wall-s` 和数据路径；`--request-index`/`--request-end` 选择半开请求范围，`--arrival-start-s`/`--arrival-end-s` 可另选半开到达窗口。`--max-requests` 限制同时调度的请求数，默认结果用 `streaming_aggregate` 保存摘要；达到墙钟上限时写出已完成请求数、输出 token、步数、Host 命中放置字节及在已完成步边界采样的父进程和 Scheduler 子进程峰值 RSS，并标记 `checkpoint_wall_limit`。完整结果另给 H2D 逐 rank 服务时间与加载门槛跨度。默认保留原始请求长度、命中和到达时间；显式 `--synthetic-output-cap` 仅截短仿真输出长度，并用 synthetic 标签及独立 `canonical_requests` 保存原始输入，不得将其报告为原样 Trace 完成。
标准 Serving report 的 `--serving-execution-mode` 和单 case runner 的 `--execution-mode` 接受 `accelerated` 与 `fast`；默认 `accelerated` 保留详细解析路径，`reference` 仍是独立的对照实现，不与详细/快速模式混同。`fast` 使用同一 Scheduler、完整虚拟 forward、WorkloadGraph/KernelPlan、launch selection、`midlayer` predictor、时间线和逐 rank source storage report，但不为 prediction 构造 trace-memory snapshot、binding 或 memory-access program，而是将相同 KernelSpec 和 HWSpec 直接交给统一 `predict` 接口。该近似保留请求/token、主要计算/通信、逻辑调用计数和源存储身份/生命周期；逐调用 memory-access 诊断不再可用，需要这些诊断时必须使用详细模式。fast 不改用 Roofline，不读取 production duration，也不复用其他进程或历史运行的预测结果；memory source recipe 的实现签名在整个 run 入口/出口校验一次，所有当步动态 shape、routing 和 KV 参数仍重新绑定。

固定比例 Production 入口也支持显式 `--execution-mode accelerated|fast`，默认保持 `accelerated`。H200 TP8/EP1 的 Kimi K2.6/K2.7 单 token 全 decode 可在同次 fast 实验内复用完整 source：首次执行全部层和 rank，之后以真实第一层 forward 提供当前 Attention KV 长度、slot mapping、cache extent 与输入 metadata，逐 MoE 层重绑当前 routing。完整层、全部 rank 的存储生命周期、collective 依赖和目标链保留；静态节点、KernelCall 和 KernelSpec 对象尽量复用，之前步骤的图与存储记录保持不变。此机制与已有 GLM 同次 source 复用共享绑定路径；prefill/mixed、不同结构或不满足绑定条件时仍执行原始完整 source。缓存归本次执行作用域，不写预测或图 LUT，不读取历史实验缓存。

标准报告在单次调用内传递已解析的 Serving 结果对象，供校验、绘图和报告生成复用，减少重复 JSON 解析与对象构造；原有结果文件、校验和图表仍生成，调用结束后释放对象。外部 CLI wall time包含本次构造、仿真、聚合、写出与报告成本；逐实验离线报告也须计入其完整生成时间，不能用独立事后生成的耗时代替。

fast 的 pinned 0.11 CPU Scheduler 在当前进程内构造，复用同一 device guard、原生 plan/update、容量事件与 source 校验，省去逐步 RPC 和序列化，不减少 Scheduler 工作；`accelerated` 及其他构造入口继续使用隔离进程。`execution_mode=fast` 明确记录该执行选择，受信任的 `scheduler_runtime.bootstrap` 保持原合同，关闭时复查无 CUDA context 或设备句柄，并清理临时模型配置目录。

fast 的 `--execution-workers 1` 保持顺序执行；标准 report 另提供 `--serving-execution-workers`（1–16）。仅在已支持的 colocated Qwen3-MoE TP2/EP2 或 dense Qwen3/Llama TP1/TP2 PP1、depth-two、图捕获和完整 Host 计时域内，最后的请求到达后可并行执行相互独立的 decode step。父进程按原顺序重建 Host 时间、capture replay 历史和审计链；批量 drain 保留跨批 pending 输入历史，source namespace 按完整 run-local key 稳定生成，不把 worker 局部序号当作跨 worker 身份。worker 上限与 CPU affinity 构成资源上限，不增加机器或核数。

`model_profile_layer_mean_v1` 的 SHA256 加权无放回路由保留原概率、随机流、浮点求和顺序和每层 provenance。float profile 使用 `src/serving/workload/profiled_routing.c` 执行同一选择循环，Python 仍生成原始精确整数比例并验证守恒；本地 C 编译器只在临时目录产生执行文件，不形成跨运行预测缓存。其他权重类型继续走原 Python 实现，native 与原始实现的逐项等价由路由测试验证。

单 case 结果在 `simulation_wall_time_s` 之外保存 `simulation_started_monotonic_s` 和 `simulation_finished_monotonic_s`。标准矩阵的 `results.json` 以 `simulation_batch_timing` 记录最早 case 仿真开始至最晚 case 仿真结束的整批纯仿真边界、墙钟和逐 case 分项；外层 `evaluation_wall_time_seconds` 仍包含子进程及结果处理。早完成 case 的后处理可与后续 case 仿真交叠，不能用外层墙钟减去各 case 后处理累计时间来代替该边界。`scripts/evaluation/serving/compare_serving_execution_modes.py --baseline <results.json> --detailed <results.json> --fast <results.json> --output-dir <目录>` 只读取已完成结果，生成逐 case baseline/详细/快速指标、偏差、耗时和守恒摘要，不为加速重新执行或复用旧预测。可选 `--supplement <证据.json>` 合并 S4/S6/S7 的布尔 gate、证据路径及各项优化的工作消除量、收益上限和实际范围，形成完整验收表。

需要逐 rank 工作与状态证据时，单 case runner 的 `--state-audit-output <state_audit.json>` 和标准 report 的 `--serving-state-audit` 会生成独立 sidecar，不改变既有结果 schema。sidecar 汇总每 rank 的 KernelKind 逻辑调用、通信操作与字节、source allocation 的内存域、容量、分配/释放计数和状态链 SHA256，并保留跨 pool 迁移记录；不包含 duration。每个 step 必须提供 source 状态，缺 step、调用总数不符或状态缺失会 fail closed。启用后禁用 immediate projection；支持的 parallel-tail 由 worker 传递逐步 source-state 与逻辑调用快照，父进程按 Scheduler 顺序重建同一审计链，采集成本计入当次仿真。审计批次用于正确性证据，不与未启用审计的 S2 测速批次混用分母。

CPU-only vLLM Scheduler 的 length-only 请求不携带真实 token 内容且显式禁用 content hash，因此 adapter 用紧凑同值序列保存 dummy token IDs；长度、索引、append/extend 与 Scheduler 请求的逐 step 切片结果和物化列表完全一致，只在 vLLM 请求当前 step 的有限 token slice 时生成列表，从而避免长 production trace 为数十亿个无身份 dummy tokens 常驻分配内存。

锁定的 vLLM Scheduler 只允许 FCFS，单步最多接纳 `max_requests` 条等待请求；因此 adapter 对已经到达但位于该决策前缀之外的请求保留有序、run-local 延迟物化队列，并在每次真实 Scheduler 决策前补齐前缀。请求到达时间和顺序不变，每个 Scheduler step 仍由原生状态机动态执行，抢占请求仍由 vLLM 自己插回队首；这一机制只避免为当前决策不可见的长尾请求提前构造 native Request，不改变 batch、token、KV 或完成反馈。

#### DeepSeek-V4 Flash/Pro 的离线支持边界

`configs/model/deepseek-v4-flash.json` 与 `deepseek-v4-pro.json` 保存已有 H200 campaign 的真实模型配置。当前执行范围为 text-only、eager、H200 colocated、PP1/EP1，Flash TP4、Pro TP8；MTP、DSpark、多模态和其他部署组合显式拒绝。模型结构按 vLLM 0.30 DeepSeek-V4 源码表达，CPU 调度控制面使用锁定的 vLLM 0.11 Scheduler，因此不能把该投影称为真实 vLLM 0.30 GPU 服务。

Workload 显式展开 SWA、C4/C128 压缩与索引、MHC、FP8 投影和 MXFP4 routed experts，经统一 KernelSpec 与公共 `predict` 接入时间线。C128 选择全部已完成压缩的状态；压缩输出按每个请求的新旧长度整窗差计算，eager 下没有跨 C128 边界时不发射对应 compressor kernel。七类 cache lifecycle 分组区分窗口缓存、追加流与 FP32 部分状态；Scheduler 的页对齐投影保留逐 component 的 charged/raw bytes，尚不等价于真实 H200 allocator。

Flash/Pro 的最小 prefill/decode 验证使用显式 debug KV block 预算与带来源标记的 routing 假设，只证明结构和执行链闭合。H200 离线 Kernel evaluation 可读取已观测的离散 split plan；Serving 无该 observation 时由 H200 launch owner 推导具名解析 split 预算，两种路径的精度证据必须分开。没有 V4 Serving 硬件真值时，TTFT/TPOT 仅为分析投影；Attention 驻留与 Marlin 速率缺口见 `decisions.md`。

#### Serving 独立验证 reference

`evaluation.serving.simulation.model_semantics_validation` 只比较事实，不复制 predictor 公式：请求流和 Scheduler step 用完整 SHA256 绑定，逻辑 operator 比较 kind/shape/dependency，KernelCall 比较来源节点、KernelSpec shape 与离散 launch，TTFT/TPOT/latency 只在 evaluation 侧按显式容差比较。`structural_lowering_manifest()` 可在不调用 predictor 的前提下输出 ScheduleStep hash、完整逻辑图、typed edge、cache side effect、逐节点 KernelCall 映射以及 supported/unsupported KernelSpec；Kernel 预测仍只能经 `lower_workload → predict_kernel_plan → kernel_base.predict()`，因此真实 TTFT、TPOT、kernel duration 与 launch observation 不进入 `ModelConfig`、production Forward 或 launch selector。

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

Standalone AF 的行为验证覆盖 A800 上 Qwen3-4B、Qwen3-30B-A3B 的 Attention TP1 + FFN TP1，以及 Qwen3-32B 的 Attention TP2 + FFN TP2，使用完整 ShareGPT QPS16 请求；PD+AF 另有四池各 TP1 的组合验证。两侧按一对一互逆 activation route 配对，逐层传输 hidden state 与 residual；请求与 token 完成、rank/bytes、依赖和时间线均需守恒。非对称 TP、共享 FFN、多对多和异构 AF 仍不在公开支持范围。部分模型/TP 组合因现有 SiLU source-bound launch 域限制失败，不能由相邻成功组合推断支持。上述验证是结构与行为验证，通信时延仍带 ProtoBridge development projection 标记，没有 AF 硬件精度真值。

AF 的人工内存政策示例位于 `configs/runtime_memory_policy/af_disagg/`；运行入口为 `scripts/run_serving_simulation.py --deployment af_disagg`，需要显式提供 Attention/FFN TP、activation transfer 与内存政策。完整输出可经 `scripts/evaluation/serving/check_af_run_artifact.py --run-output <run.json> --model <model.json> --requests <requests.csv>` 检查模型绑定、请求和 token 守恒及 rank 对应的传输计数；该校验不证明时延精度。

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

Runtime memory policy 保留两种显式证据状态：`verified_uniform_across_pool_ranks` 与 `all_non_kv_reserves_for_bound_runtime` 配对表示 `verified_runtime_evidence`；`assumed_uniform_across_pool_ranks_unverified` 与 `reserves_inherited_unverified_not_complete_for_bound_runtime` 配对表示 `unverified_analytical_assumption`。两类字段混用或未知值立即失败，派生容量 manifest 保留 `evidence_status`。从 colocated 改绑至 AF 的 reserve 只能按未验证假设参与分析，不因仿真能完成请求而升级为 AF 真值；要求 verified 证据的 ShareGPT evaluation 拒绝该假设状态。

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

#### 真实 token ID 模式

显式设置 `--enable-prefix-caching --prefix-cache-mode token_ids`，并通过现有 `--production-trace` 输入传入 `--trace-prompt-token-ids-field` 和 `--trace-output-token-ids-field`。JSONL 字段值为整数数组；CSV 字段值为 JSON 数组文本。`RequestSpec` 对应可选字段为 `prompt_token_ids`、`output_token_ids`（不可变 tuple），长度必须分别等于 input/output token 数，ID 必须是非负整数且位于模型词表内。`output_tokens > 0` 时必须提供完整真实输出序列；仿真回放这些 ID，不生成文本，不以 dummy 输出建立共享缓存。零输出请求可省略输出数组。旧 `prefix_hit_tokens` 字段必须为 0，表示未注入 trace 命中量，不是观测到零命中。

原生 Scheduler 按当前驻留状态查询完整前缀块，并保留最后一个 prompt token 的重算规则；真实输出形成的完整 KV 块也能被后续请求复用。首次 admission 的命中单列为 `content_prefix_hit_tokens`，不填入 `trace_prefix_hit_tokens`。抢占后再次查到自己的已有计算块，不重复计作首次 prompt 命中。结果的 `scheduler_runtime` 披露 requested/effective prefix caching，`request_source.prefix_alignment.aggregate.content_prefix_hit_tokens` 汇总真实内容命中，`scheduler_progress.summary.prefill_tokens` 给出实际 prefill 工作量。请求内容只作输入，不写入 aggregate report。

当前 token 模式使用同步 vLLM CPU0.11 Scheduler 的单一 FullAttention 或 MLA KV group，TP/PP 不再因 token 模式单独拒绝；具体模型、并行和部署仍遵守各自已有 Forward 支持范围。已验证 dense TP2×PP2、标准 Qwen3-MoE、PD 1P1D/2P2D 的完整请求链，以及 GLM4-MoE MTP 的 target/proposer 源码链。多个 colocated 实例分别维护缓存；PD 交接保留完整 prompt/output 身份，等待传输完成才向 D admit，传入 KV 不计作 D 本地内容命中。普通 MLA 已验证 native0.11 Scheduler、官方 DeepSeek-V3 61 层、H200 TP1、colocated、单组 MLA 的真实 token → Forward → 全 kernel 计价链，覆盖重复前缀命中、不同前缀未命中和 decode；不代表其他 MLA 模型、硬件或并行配置已验收。该 API 显式声明 `symbolic_forward_source="vllm_0_11_deepseek_v3_sync_fp8"` 与 `target_sampling_source="vllm_0_11_plain_greedy_bf16"`，并提供源码 batch/context 上限；采用已有 source-math Roofline 与测试 KV 容量，不声称实际 GPU provider 或正常容量已验收。0.30/V4 混合 KV、滑窗、async scheduling、closed-loop admission、SGLang 及外部 KV tier/lifecycle replay 继续明确拒绝；独立能力通过不代表任意组合都支持。

MTP 使用 `VLLMSchedulerConfig.mtp_draft_token_ids_by_step` 声明每个原生 step 的真实 draft，CLI 对应 `--mtp-draft-token-ids-by-step drafts.json`。JSON 直接映射 step ID 到 request ID 与 token 数组，例如 `{"main:vllm:0":{"a":[2,60]},"main:vllm:1":{"a":[4,60]}}`，API 中数组使用 tuple。每个实际执行 proposer 的 step（包括执行 proposer 的 prefill/终止 step）必须覆盖完整调度 batch，序列长度必须等于 speculative depth，ID 必须在词表内；已接受 draft 必须与该请求真实输出 transcript 的相应前缀一致，缺失或不一致立即报错，不从接受数编造 ID。step ID 为 `<pool>:vllm:<index>`，PD 的 draft 属于 D 池，P 池仍关闭 MTP。GLM4 MTP 的完整 token 链目前用 API 显式测试容量验证；CLI 的 Production Trace 仍要求正常容量策略，而该模型缺少完整 MTP draft-weight residency，因此该 CLI 组合会明确拒绝，不能以 debug blocks 绕过。报告只披露 draft 输入来源与帧数，不输出 token payload。

硬件 Forward 仍按计算工作量预测时间，不加载权重或调用 GPU。传入 ID 但保持默认 `--prefix-cache-mode trace` 可以执行无内容缓存对照，报告保留实际关闭状态。默认 length-only 与 canonical 四列 CSV 不变。

下面的可复制示例使用三个间隔 1/16 秒的请求：前两个 prompt 完全相同，第三个不同。ID 数字仅用于演示；评估真实 workload 时应替换为同一模型 tokenizer 的真实 prompt/output ID。

```bash
mkdir -p experiment_logs/evaluation/prefix_cache
python - <<'PYEXAMPLE'
import json
from pathlib import Path
rows = [dict(id=str(i), time=i/16, input=32, output=2, hit=0,
             prompt_ids=[1 if i < 2 else 2]*32, output_ids=[3,4])
        for i in range(3)]
Path('experiment_logs/evaluation/prefix_cache/requests.jsonl').write_text(
    ''.join(json.dumps(row)+'\n' for row in rows))
PYEXAMPLE
/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python scripts/run_serving_simulation.py \
  --production-trace experiment_logs/evaluation/prefix_cache/requests.jsonl \
  --trace-format jsonl --trace-time-unit s --trace-arrival-field time \
  --trace-input-tokens-field input --trace-output-tokens-field output \
  --trace-prefix-hit-tokens-field hit --trace-request-id-field id \
  --trace-prompt-token-ids-field prompt_ids --trace-output-token-ids-field output_ids \
  --trace-success-filter-mode upstream_attested \
  --trace-success-provenance 'synthetic token identity example' --trace-replay-mode original \
  --model configs/model/qwen3-4b.json --hardware-config configs/hardware/nvidia_a800.yaml \
  --deployment colocated --backend vllm --prediction-method midlayer \
  --collective-backend analytical_roofline_v1 --launch-policy analytical_launch \
  --attention-kv-layout paged_varlen --attention-kv-distribution uniform \
  --prefill-tp-size 1 --scheduler-pipeline-depth 1 --max-requests 256 \
  --max-model-len 8192 --max-batched-tokens 8192 \
  --runtime-memory-policy configs/runtime_memory_policy/nvidia_a800/qwen3-4b.json \
  --enable-prefix-caching --prefix-cache-mode token_ids \
  --output experiment_logs/evaluation/prefix_cache/on.json
```

去掉 `--enable-prefix-caching --prefix-cache-mode token_ids` 并改输出路径即可运行关闭缓存的对照。此例完整块大小为 16，开启缓存后第二条请求命中 16 tokens，总 prefill 从 96 降至 80 tokens；不能将这个工作量节省比例等同于 TTFT 或端到端性能收益。

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

统一虚拟 Tensor 路径保留源码产生的完整依赖图、存储别名和生命周期。解析性能计价可以复用已验证的不可变几何与完整源程序；每步的动态输入、路由计数和逐层身份仍须独立绑定。几何模板仅共享形状、精度、variant 和 rank 等已验证字段，新节点 ID 和完整参数字典每次生成；不能用它缓存模型或模块的实际行为。

`KernelBlockRepeat.trace_node_ids` 将物化模板关联到全部逻辑来源节点。`KernelPlan.logical_calls()` 仅在显式请求时展开来源、参数与依赖，缺少来源映射的历史 repeat 会拒绝该查询；正常流式摘要不展开全部调用。来源映射对应的完整 artifact 使用 schema 15，流式 artifact 使用 schema 14 和 execution-contract chain v6。历史完整 schema 8、流式 schema 9/v1 与 13/v5 保留各自读取合同，旧结果不能重标为新版本。

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

异构 PD 的首个解析范围是 vLLM0.30、BF16 standard-GQA Qwen3、A800/H200 两端独立 TP/PP。普通 CLI 使用 `--decode-hardware-config`、`--prefill-tp-size`、`--prefill-pp-size`、`--decode-tp-size`、`--decode-pp-size`；跨硬件需显式 `--transfer-link infiniband --kv-transfer-backend analytical_roofline_v1`。每层/head 区间映射沿各端 physical rank placement，复制源只选择必要副本。P 的 native KV blocks 在传输期间保持额外引用，D 先预留自己的 blocks，全部传输完成后才准入 Decode；`pd_kv_lifecycle` 报告 pool、所有参与 ranks、请求身份、block groups 和关键时刻。传输为两端 HBM/PCIe/网络约束下的流式带宽下界，未知启动和传播成本未计价；继承的 runtime reserve 仍按自身证据等级报告，不由请求完成升级成已验证容量。Dense PD 的 DP 按真实 replica 展开并路由请求；Qwen3MoE 的 TP/EP 与 PP 可在两侧分别选择。额外的 vLLM0.30 `VLLMBackend.from_data_parallel_replicas` 接口协调独立 native CPU Scheduler/KV manager，将真实 `data_parallel_rank` 保留在每条 ScheduledRequest 中；DP>1 的 EP-on/off 路径均已接线，可按模型层区间组合 PP：EP-on 时 EP=TP×DP、SP=TP（TP1 时无额外SP搬运），EP-off 时专家中间维按 TP×DP 切分，通过 DP dispatch/combine 后再做本地 TP 归约。完整 TP2/EP4、TP2/EP-off、TP1/EP2 及 DP2/TP2/PP2/EP4 的 MoE 异构 PD 请求已测。PP 按 DP 副本分别发送 hidden/residual，经独立目的 backing、TP gather 和 runner copy 进入下一 stage；SP restore 的 padding backing 只发送真实行前缀，idle DP 不造 PP payload。SP 的 padded rows 只参与 MoE gate、dispatch、专家计算和 combine，不变成真实请求/KV；idle DP ranks 仍服务其他 replica 的专家。all_gatherv/reduce_scatterv 沿公共 COMMUNICATION、typed participant ports 和显式 pairwise 物理场景计价。普通 PD CLI 的独立 `--prefill-dp-size`/`--decode-dp-size`、TP/PP/EP 已接该协调路径：policy binding 可显式携带 data_parallel_size（省略规范化为1），容量先按全局 DP×PP×TP 推导专家权重和逐 rank 预算，再逐 DP 保留真实 native allocator receipt，并在启动 Scheduler 时严格核对。EP-off 专家 I 按 TP×DP 切分，router/attention/词表仍按本地 TP/PP 复制；KV 不除 DP/EP。source activation reconciliation 使用各 DP 的真实 native backing。0.30/EPLB-off 的 linear placement 支持 base/remainder 不均分和空 expert ranks；权重按真实 owned expert 数计，空 rank 省略零 extent 参数，以真实零贡献 output fill 保留 combine 依赖。各 DP 的 native block count 可不同，公共标量仅表示各 replica 最小值，实际 Scheduler 和 receipts 保留各自容量。完整 EP3/三请求正常容量 CLI 及 empty-expert 的 CPU E2E 已测；MLA PCP 与 GLM BF16 的扩展见下文，任意缓存格式转换仍未实现；MoE EP-off 的 DP 使用共享专家分片/通信，不能按 dense 独立专家副本处理。

PD 中 P 的 `RequestSpec.prefix_hit_tokens` 只描述 P 的 trace 命中。P 完成后先构造 D 端完整-KV请求，再用同一个 `prefix_hit_tokens=0` 的请求做目的预留和交接；保留 P 原请求及 trace/scheduler 命中证据。目的仍按完整 prompt 长度分配和接收 KV，不能减去 P 的命中长度；真实 prompt/output token IDs 与首个输出 seed 保持不变，所有物理 transfer 完成后才准入 D。原生 `token_ids` prefix 命中与 trace 命中分开，D 接收外部 KV 不重复登记成 P 的 cache hit。

GLM-5.2 BF16 的 vLLM0.30 源程序在 A800/H200、MTP-off、显式 sparse-MLA 场景复用上述异构 PD 入口；两侧 TP/PP/DP/EP/CP 独立选择，SP 由框架关系推导。CLI 增加 `--prefill-cp-size` / `--decode-cp-size` 和 `--prefill-pp-layer-partition` / `--decode-pp-layer-partition`（各 stage 的连续层数）。程序接口 `WorkerPoolConfig.pipeline_layer_partition`、`VLLMSchedulerConfig.pipeline_layer_partition` 与容量 policy binding 保留相同分区；pool 与全局显式分区冲突即失败。policy schema2 的可选 CP 与 partition 字段在边界规范化，旧 CP1/None 配置保持原往返语义。

PCP 仅适用于 MLA、DCP1、PP1：每个 prefill 按 2×CP ceil chunks 镜像分配 query，保留绝对 causal 位置；padding 不写 KV，逐层 gather normalized latent/RoPE 与 full-indexer K，再独立插入缓存，最终恢复原 hidden 行序。每个 TP/CP rank 保有完整压缩 cache；DSA full-indexer 层每 token 另有132字节，EP 不改变该 KV 大小。交接按真实 DP owner 和 PP 层区间选源 CP0/TP0 副本，并给每个所需 D TP/CP 副本计传输。CPU Scheduler/allocator 明确按每 DP 投影为真实 CP1 workers；完整 CP 物理副本、原配置/hash 与投影事实分别报告，不伪造 native PCP receipt。

GLM EP-off 的 routed I 按 TP×CP×DP 切分，先 gather/route 再按 DP/CP sum-scatter；idle DP 继续服务专家，不造请求、attention 或 PP payload。EP-on 且 DP>1、TP>1 时 SP=TP：PP1 执行模型级 SP，dense/shared 保留完整权重；PP>1 只有 MoE 内部 SP，dense 保持普通 TP、shared 完整权重，MoE TP gather 恢复真实行后保留 decoder late TP sum。TP1×CP>1×DP>1 的 EP-on 采用显式 `analytical_global_expert_projection` 闭合跨 CP 专家依赖；上游 naive provider 的 DP group 不能证明该全局交换，`native_provider_equivalence=False`，不能升级为 provider 验收。DP1 PCP 的 EP-on 源保留 Runner 与 decoder 两处 TP sum，数值/provider 等价未验证。

CP1 的 GLM PP 使用通用连续层分区和 stage-local TP/DP/EP groups。上游构造器可接受以 shared-indexer 层开头的 stage；PP payload 仍只有 hidden/residual。当前 attention 程序若读取 top-k（sparse、decode 或 mixed），必须有真实生产者，不能凭空补初始化或跨 stage 数据；该依赖不能推成整个 PP 配置非法。BF16 pure fresh dense-MHA prefill 的 q_lens=kv_lens、全部模型行均为 prefill、上下文不超过 checkpoint top-k 时不读取 top-k，现允许 shared-first 分区，例如78层39/39。CLI 在 CP1 省略 `--glm-prefill-attention-source` 可选这一严格按 shape 判断的既有源分支；显式 `force_sparse_mqa` 保留 sparse 依赖检查，PCP 仍须 force_sparse。以 full-indexer 层开头的26/24/28或38/40等分区仍适用原 sparse/decode 程序，数值只是例子。标准 TP-partial send、接收 gather、late sum 均按声明源保留；未新增 top-k 传输协议或声称数值等价。完整请求和正常 allocator 路径的 CPU 通过均不代表真实 GPU/provider、reserve 或精度验收；未知 launch 的源数学模型明确报告 unsupported，未用 production duration 拟合。

Host-KV 的状态变化可以概括为：

```text
GPU-resident KV
  ── EVICT intent → PCIe/Host interval → completion ──▶ Host-resident KV

Host-resident KV
  ── LOAD intent  → PCIe/Host interval → completion ──▶ GPU-resident KV
```

intent 创建时预留必要容量，但只有 completion 事件才能提交 residency 变化；被请求消费的 KV 还必须等待 LOAD 完成。这样 queue、PCIe busy time、容量压力和 compute overlap 都可以从事件时间线解释。当前机制仍是显式 lifecycle replay，不应写成已复现生产 vLLM 的 victim policy、 prefetch 或 Host DRAM 控制器排队。

### 虚拟 forward 的离散 launch 与 graph 入口

普通A800 BF16 Qwen3/Llama的eager共同入口支持`injected_launch`：同一次forward保持算子、storage和依赖，只把选择策略传到既有canonical selector。连续Linear保留row-major A/weight/C、transB=True的真实描述，单行LM head沿既有GEMM workload的gemv key；GQA复用已有main/setup/combine选择，不重复插入旧wrapper的填零和Q/output/LSE拷贝。缺离散key仍明确失败。embedding、logits选行和采样等source-only调用继续明示Roofline范围，不因为其他算子有launch就升级为整模型原生provider验收。

SGLang0.5.19普通`FULL_DECODE_ONLY`使用同一forward：完整decoder、head和FP32 logits在capture内，active logits切片及greedy在外。underfilled bucket必须有显式`SGLangGraphInputs`声明retained tail IDs，不能按未知值推导各TP rank的embedding行数。FlashMLA的dummy query使用seq_len1和slot0；源码保留64-token空页，声明逻辑arena把active范围移至该页之后，不把它当实际KV分配或正常容量。typed state可由mirror backend逐step传入，其他source的slot=-1/no-write规则不放宽。EAGLE与graph的组合仍需单独迁移。

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
- 扩大既有 DP、SP、MLA PCP 与跨硬件 P/D 的模型/版本和 provider 证据，接入 LPU 等部署；
- 接入更精确的通信、Host memory 与 SSD/Flash 后端；
- 支持 FP8、FP4 等低精度；
- 扩展到新的硬件。

每种扩展都应先补全自己的语义和证据，再复用现有 Kernel 与时间线接口。训练和 RL 可以借用相同的算子/资源抽象，但具有不同的状态机、通信模式和目标指标，不属于当前推理 Serving 模型的隐式能力。

扩展时仍要保持层次边界：新的 Scheduler/backend 只负责产生调度 step、KV 状态事件和框架原生决策，不能直接构造硬件 kernel、调用性能公式或把调度耗时塞入 kernel 时间；新的模型结构负责产生守恒的逻辑算子和依赖；新的通信或存储后端只替换对应资源的计时实现。每项扩展都要同时验证本层语义守恒以及下游时间线结果，不能只以端到端数值能够运行作为验收。

## 11. Lowering 的 helper-launch contract

区分两个缺失：launch 已进 `KernelPlan` 但 host 准备时间未计，归 `src/serving/runtime_prepare.py`；launch **从未进入** `KernelPlan`，无归属——既不被 device 定价也不被 host 准备，步里出现无人认领的时间。本节契约针对后者：lowering 在这些调用点必须声明发射 0 个、1 个还是 N 个 kernel，而不是给残差配参数。判据全部读自 vLLM / vllm_ascend 源码分支谓词，不含按卡开关；源码谓词与派发层实测分叉之处，以版本化离散发射 observation 为准。三硬件源码参考位于 `/share_data/yangxinhao/chip_modeling/upstream/vllm/{v0.11.0_a800, v0.24.0_c500_sdist, v0.18.0_ascend, vllm_ascend_plugin}`。契约已实现：调用点表与计数解析在 `src/serving/workload/helper_launch.py`，发射接线在 `src/serving/workload/build_workload.py`，版本化离散发射 observation 在 `configs/validation/serving/helper_emission_observation.yaml`（整数计数、不含 duration，加载时完整校验）。

三类调用点及其计数与实现：

1. **attention 输出写回（TensorMove）**。切片赋值分支（`vllm_ascend/attention/attention_v1.py` 的 `output[:num_tokens] = attn_output[:num_tokens]` 等）在 ATen 语义层是 1 次真实 device 拷贝；paged `out=` 分支（`forward_paged_attention` :851-859、capture 下 `full_graph_pa`）原位写出，ATen 层 0 次。实测发射律（910B canonical，每 rank 每 phase）：decode 全 case 0；qwen3-30b prefill 0；qwen3-4b/8b prefill/mixed 2L−1——偏差机制在 torch_npu/aclnn 派发层，python 不可判，消费这层计数只能以版本化离散发射 observation 为源，不得按 ATen 层计数或常数补齐。反例 `query = query[:num_tokens]`（:785）是 dim-0 切片**视图**，发射 0——形状、batch、卡全相同而计数相差 1，说明计数不能由形状推出。实现：FLASH_ATTN 节点经 `attention_output_writeback` 调用点用 observation 解析计数，并携带 `source_output_copy_plan`（logical/physical 行域，`build_workload.py:5743-5757`）；910B lowering 按该 plan 物化 TensorMove helper（`src/serving/kernel_lowering/lower_workload.py:1551-1589`），observation 记 0 时不发射。
2. **MoE routing mask**。EP 关闭时 0；EP 开启时每层连续 4 kernel 串 Index→NotEqual→Cast→Mul（`token_dispatcher.py:344-348`，源码算子数 3，Cast 来自 bool mask 与 topk_weights 的类型提升，逐 phase 实测 48/串/层）。`apply_router_weight_on_input` 为真时额外 1 次 Mul；selector 变体有条件发射（`masked_fill`、`torch.where`、renormalize 除法）。谓词已解析并挂在 ROUTER 节点属性上（`build_workload.py:1699-1718` 的 `topk_method` 等）。实现：`MOE_ROUTING_MASK_EMISSIONS` 按卡声明 workload 层发射哪些调用点——A800/H200 发射 expert-map-index，910B 发射完整 4 kernel 串，C500 由 FUSED_MOE recipe 内的 `moe_expert_map_lookup` 覆盖、workload 层不重复发射；发射循环在 MoE block 之后逐个调用点经 observation 解析计数并生成 `expert-map-index`/`router-mask-index` 等节点（`build_workload.py:5378` 起）。
3. **model 输入/输出边界**。每个产生采样的 step 恰好 1 次 gather：`sample_hidden_states = hidden_states[logits_indices]` 是 advanced indexing，必定物化；该调用点在各目标硬件的 vLLM 树里逐一存在（`gpu_model_runner.py` / `model_runner_v1.py`），是 hardware-independent lowering 事实。同边界的条件发射（`hidden_states[:-pad_size]`、`index_select`、`.copy_`）与反例（`hidden_states[:num_scheduled_tokens]` 视图发射 0）都必须声明。实现：主路径为 `final-norm` → `logits-boundary-gather`（WorkloadKind.GATHER，携带 `predicate`/`source_symbol`）→ `lm-head`（`build_workload.py:6230-6259`）；step 输入边界的 `token-embedding` gather 经 `token_embedding_gather` 调用点解析（`build_workload.py:5026-5047`），A800 prefill 的实测零由 observation 显式授权后不发射；MTP 分支的 GATHER/ELEMENTWISE 对同样已发射（`mtp-pass-N:token-embedding` / `position-zero-mask`，`build_workload.py:3197-3234`）。

契约形状（与 `runtime_prepare.py` 的 `RuntimePrepareRule` 同构，只声明**发生**，不携带 duration）：`source_symbol`（调用点源码符号）、`predicate`（可回指源码文件与行的谓词）、`count`（0/1/N，**`0` 必须显式授权**，不能是"未声明"的默认值——否则"静默记 0"与"没人写过"不可区分，契约退化成它本要替代的沉默）、`kind`/`variant`/形状来源（复用 `GATHER`、`ELEMENTWISE` 现有词汇，形状由 lowering 已有 token 数、hidden、topk 直接算出）。未覆盖分支 **fail-closed** 抛错；登记为必须由 observation 提供计数的调用点缺项即抛错，不回退源码谓词。验收是与 duration 无关的整数等式：声明的 per-step 算子计数必须与 trace 逐步精确相等。

参数化不能替代的理由：计数不是可观测量（shape/batch/模型/卡）的函数；要表达的是分支而非标量；用于拟合的量与被关闭的 closure gap 是同一个量，任何常数按构造都能关闭它、零证伪力；事实在 vLLM 层跨卡共享，按卡存常数会复制三份并随版本静默失配；typed 路径已存在，常数会丢掉形状依赖、把负载依赖抹平。

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

## 独立 LLMServingSim baseline

LLMServingSim 仅由 `scripts/evaluation/serving/run_llmservingsim.py` 显式启动，默认 Midlayer 流程不加载其 checkout、Python 环境、Chakra 或 ASTRA-Sim。版本绑定与五个原生 setting 由 `configs/baselines/llmservingsim/source.json` 唯一维护；这五例保留各自原始 meta、完整请求、TP/EP 和内存设置。它使用自己的 Scheduler、profile 插值/外推和真实 ASTRA-Sim analytical backend；其 duration 表不进入 Midlayer src、Forward 或 HWSpec。

源码原 checkout 只读。先在 `experiment_logs/evaluation/llmservingsim/runtime` 克隆指定源码，在副本中初始化 ASTRA-Sim、Chakra、fmt、spdlog、memory analytical 和 network analytical/yaml-cpp 子模块；analytical 运行不需要 NS3 或 GPU。使用独立 Python 3.10 环境安装 `numpy==1.23.5 pandas==1.5.3 pyyaml pyinstrument rich grpcio-tools setuptools-grpc networkx`，再从固定 Chakra 子模块执行 `uv pip install --python <独立环境>/bin/python --no-deps <Chakra源码路径>`。这仅安装 CPU converter 的依赖，不安装 vLLM、权重或可选 GPU/HTA 工具。首次实际运行的环境锁定文件位于任务结果目录的 `python-environment.txt` 与 `build-environment.txt`。

构建在另一个独立 Conda prefix 中安装 `protobuf=3.20 cmake<4 make`。用该 prefix 的 protoc 生成 Chakra 的 `et_def.pb.cc/.h`；Python protobuf 由 Chakra wheel 的 grpcio-tools 构建步骤生成，两者使用相同 schema。配置 `cmake -S <ASTRA>/build/astra_analytical -B <ASTRA>/build/astra_analytical/build -DBUILDTARGET=congestion_unaware -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DNDEBUG -DCMAKE_PREFIX_PATH=<构建prefix>`，再 `cmake --build <build目录> -j 4`，将 `<build目录>/AnalyticalAstra/bin/AnalyticalAstra` 链接到 `../../bin/AstraSim_Analytical_Congestion_Unaware`。显式 `NDEBUG` 使上游覆盖过的 release flags 与 Protobuf release 库一致；无需修改第三方源码。

```bash
# 已准备好隔离 runtime 与 Python 后，一条命令运行五例并生成 report.md。
# output 必须为全新的任务子目录；首例完成后其余四例并行。
/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python scripts/evaluation/serving/run_llmservingsim.py --bundled-suite --runtime experiment_logs/evaluation/llmservingsim/runtime --python experiment_logs/evaluation/llmservingsim/venv310/bin/python --output experiment_logs/evaluation/llmservingsim/run1

# 只刷新已有产物的比较报告，不重复仿真。
/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python scripts/evaluation/serving/run_llmservingsim.py --report --runtime experiment_logs/evaluation/llmservingsim/runtime --output experiment_logs/evaluation/llmservingsim/run1

# 只读解析四硬件 × 两模型的 canonical TP1 QPS16 输入并保留全部八例。
/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python scripts/evaluation/serving/run_llmservingsim.py --prepare-targets --runtime experiment_logs/evaluation/llmservingsim/runtime --output experiment_logs/evaluation/llmservingsim/run1

# 独立运行 active Midlayer case，延期项只保留历史；需要本地 build/lru_sim.so。
/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python scripts/evaluation/serving/run_llmservingsim.py --midlayer-targets --runtime experiment_logs/evaluation/llmservingsim/runtime --python /home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python --output experiment_logs/evaluation/llmservingsim/run1

# 无需目标时延表的 CPU 静态 KV 容量检查；不运行目标仿真、不改配置。
/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python scripts/evaluation/serving/run_llmservingsim.py --capacity-check --runtime experiment_logs/evaluation/llmservingsim/runtime --python experiment_logs/evaluation/llmservingsim/venv310/bin/python --output experiment_logs/evaluation/llmservingsim/run1
```

逐例 `run.json` 保存实际命令、配置、输入身份、后端二进制指纹与 exit code，`sim.csv` 是新执行生成的原始输出；`normalized.json` 按原 workload 行号恢复请求身份和顺序，校验完整率、到达时间、长度、实例和非负有限指标，并将纳秒换成秒。`comparison.json` 分开报告附带历史 sim 的复现差异和 vLLM 实测误差，包含均值、P50/P90/P99、请求级 MAE/MAPE 与完整分母。TPOT 使用 `output_tokens−1`，单输出 token 为零，native 纳秒取整值另行保留。吞吐误差使用双方各自的首次到达到最终完成窗口，另保留 Midlayer 的 workload 零时刻到完成窗口。vLLM 附带记录缺少逐 token ITL，不能报告 ITL 实测误差。

目标矩阵由 `configs/baselines/llmservingsim/targets.json` 选择，`hardware_status` 必须穷举历史 `hardware_ids`，值为 `active` 或 `deferred_by_user`。当前 active 为 A800/C500/910B 两模型六例，H200 两例由用户延期；报告分别展示当前六例分母与两例延期历史，保留其原始状态和已有 Midlayer 结果，延期不算成功。采集、native target、Midlayer target 和新的容量检查只执行 active 项；延期不能通过内部 Midlayer 执行入口绕过，也不触发 SSH 探测或入口问询。转换器保留历史八例 canonical 文件及缓存证据，输出完整请求长度、到达时间与 ID 映射；纳秒转换采用 nearest/half-even，原毫秒值与至多 0.5ns 的舍入差保存在映射文件。缺匹配硬件、模型、精度、TP 和实现语义的 layerwise bundle 时，active 项仍记录 `missing_profile` 并留在当前分母，禁止改名、缩放、注入 Midlayer 预测或用空表继续执行。

Llama 目标采集使用匹配的 Instruct config，不因 base 与 Instruct 结构相同而复用 RTX 表。canonical CSV 没有原 token IDs；合成 token identity 只实例化有结构化证据的零净前缀工作量代表值，并不复现原 prompt 或观测到的逻辑 hit。相同 memory-utilization 也不足以消除 native weight-only KV 预算与 Midlayer non-KV reserve 的差异，因此按绑定的 KV 字节预算换算并核对 block 大小与数量。报告保留每例输入身份、缓存代表值、collector/truth 软件版本、实际 block/context、实测坐标范围、uniform-only/eager/plugin 限制、原生外推及所有 warning；没有 per-lookup 计数时明确记录不可用，不声明精确 runtime 等价。

Midlayer 对照在任务目录内创建可再生源码副本，数据 LFS payload 只读物化到该副本；Host profile 所绑定的 serving/raw 证据通过 canonical checkout 的只读目录链接引用并由原 loader 校验。生产源码、HWSpec、canonical 数据和 main 分支均不修改。`midlayer-request-metrics/prediction.json` 仅在 evaluation 边界保留此次任务所需的请求级 timing；不会改写 canonical schema。

`--capacity-check` 从 active case 已有 allocation manifest 中仅提取静态容量字段，在独立 native Python 进程中构造真实 `MemoryModel`，比较 block 大小与数量。native utilization 的单位是 weights-plus-KV；由指定 KV 字节预算换算时使用 `(native weight bytes + available KV bytes) / physical HBM bytes`，不使用请求时延或误差。输出 `capacity-comparison.json` 保留原 policy、权重计数差异和换算结果；已有延期项只沿用其历史结果，不重新运行。这是显式配置单位检查，不会自动改动目标 config；即使 KV block 相同，也不能替代目标 profile、缓存代表值或完整资源等价性验证。

目标 profile 的采集入口为 `scripts/evaluation/serving/llmservingsim_profile_runner.py`，必须通过 `--owner <当前任务名>` 显式指定采集 owner（例如 `--owner wt-llmservingsim-20261025`），不能沿用其他任务的 owner；只运行任务执行副本中的原生 one-layer、dummy-weight、BF16/TP1 profiler，不下载权重。每次启动前检查所选设备空闲，raw `run.json` 登记 owner、device、collector baseline、实际源文件指纹、model config、命令和 PID；临时编译与 IPC 使用单次任务 `/tmp` 目录并在退出时清理。`--resume-capture` 只允许同任务、同硬件、同模型的已停止 capture，记录复用文件的内容地址，原生 sink 继续验证每组 measurement identity。`--attention-grid-factor` 是原生几何采样密度，改变密度不改变域上限；`--skip-skew` 必须在报告中保留 uniform-only 限制，不能声称已采 skew。

NVIDIA 采集入口认识 `nvidia_a800` 与 `nvidia_h200`，但 `deferred_by_user` 项在读取设备或创建 capture 前直接拒绝，当前 H200 不执行。active 项核对所选卡的 `nvidia-smi` 型号与 hardware ID；`--device-uuid <物理GPU UUID>` 绑定已分配设备，raw manifest 记录 UUID、物理 index、`CUDA_VISIBLE_DEVICES` 与进程内 device0 的映射。型号/UUID 不符、有设备进程或利用率非零时，在创建 raw capture 前失败关闭。

NVIDIA 的 `--stage hardware` 只按已核对的物理 UUID 查询 NVML 规格，将总显存、最大显存时钟和总线宽度留在 raw，并沿用原生 `probe_spec` 的 DDR 带宽公式转换为 hardware.yaml；此路径兼容缺少时钟/总线属性的 Torch 2.8。它不执行 kernel 或通信 benchmark，互联字段明确记录未测量，TP1 的显式配置不会消费互联占位值。旧 Torch 的 CPU Kineto operator 事件可能使用 resource0；`prepare_legacy_kineto` 只从带 linked correlation 的 CUPTI CPU 活动建立 logical/OS 线程映射，仍要求每个 CUDA 活动有唯一发射归属，设备区间及其并集口径不变。

采集适配实现由 `dataset_pipeline/serving_dataset/llmservingsim_profile.py` 生成：A800 使用 `--vllm-011` 对齐旧版本 request/KV API；MetaX catalog 只增加经源码证明的子类别名，并用实际 Kineto thread/launch correlation 映射设备活动；Ascend 使用 `prepare_ascend_profiler` 与 `prepare_ascend_catalog` 生成私有副本中的桥接代码。Ascend 的 `torch_to_npu` flow 关联 CPU module 与精确设备 task，时延只取设备活动区间的逐调用并集，runtime control 和 module 边界外的 runner 准备活动逐项列入 raw audit。NPU 高 token 区的固定二倍数采样是采集成本选择，原生插值/外推和请求 settings 保留；该差异不能作为原生默认 grid 描述。原生 kernel 实现、模型结构与采集软件版本记录在 bundle 中；不同版本或 eager 与 graph 执行差异属于比较限制。Ascend 适配在 engine 启动后、采样前校验实际 block size 是否与请求一致；若 backend 只支持 128 而目标要求 16，则记录 geometry mismatch 并失败关闭，不修改目标或声称同配置预测。

合法 capture 拉回本地后，`run_llmservingsim.py --target-case <hardware>/<model>/tp1_qps16 --profile-capture <raw模型capture> --hardware-capture <raw硬件capture> --runtime <执行副本> --python <独立CPU环境>/bin/python --output <已有结果根>` 先拒绝用户延期项，再校验成功 capture、硬件/模型/精度/TP、engine block/context、实测表与 acquisition identity，真实运行完整原 case。配置换算只保留已核对的 KV 字节预算；有结构化零净前缀工作量证据时，用独立合成 block-token identity 实例化相同 scheduler-equivalent representative，同时保持 prefix allocation 开启，不能标成 observed zero 或声称复现原 prompt。每例只写自己的结果目录，多个独立 case 可并行，`--report` 汇总当前分母并另列延期历史。报告分别列 LUT compute + analytical network 与 pure analytical compute unavailable；原生 `--network-backend` 只选择网络，不提供这些 dense Transformer 的纯解析计算模式。报告标为 checkpoint，最终 gate 验收由主 agent 承担；延期本身不使其它 gate 自动通过。

Ascend 的显式 `block16_capability=True` 采集适配允许在私有进程中保留目标 block16：它要求同一软件栈的原生 FIA 与 paged-attention 接口分别在 block16/block128 下通过 BF16 数值验证，并绑定原生 `refresh_block_size` 与支持列表函数的源码指纹。适配只替换将配置改成128的 policy 和 backend 支持声明，attention、cache device 算子及 metadata builder 保持原实现；配置范围限定为已验证的 BF16/TP1、32 query heads、8 KV heads、head_dim128 的 Qwen3/Llama。采集 CLI 通过 `--block16-evidence <capability.json>` 显式传入证据；实际缓存必须为16，仍须通过 engine geometry 和完整三阶段 coverage。TND mixed causal 数值控制与原始 block128 表单独保留，不能将 block128 时延改名为 block16。该适配差异必须在比较报告中披露，且不进入 Forward 或共享环境。

### 逻辑权重放置清单

`deployment_weight_placements(model, deployment)` 从 colocated 部署和模型配置生成启动期逻辑清单，不加载权重或分配设备内存。当前支持 unquantized BF16/FP16 Qwen3、Llama 和已有 residency 支持范围内的 Qwen3MoE；TP、受 residency 限制的 PP、Qwen3MoE EP 和多个 colocated 实例使用现有并行推导。TP 超过 KV head 数、跨 PP 的 tied embedding、自定义 PP 分层、其他压缩权重和拆分池等尚不支持，入口会拒绝。对象标识是每池每 rank 的逻辑身份，QKV、gate/up 与 expert 逻辑参数按已有 fused 参数标识对象，`object_view` 标识其中的逻辑视图；真实物理区间、地址、UUID 和物理布局未知。vocabulary 的 `logical_shape` 不含 padding，`storage_shape` 和 shard 区间采用 padded vocabulary；`storage_bytes` 不计同 rank 的 tied alias。总量仅为权重，不含 KV、activation、workspace 或 reserve。

可用 `python scripts/evaluation/serving/report_weight_placement.py --model configs/model/qwen3-8b.json --hardware nvidia_a800 --tp 1 --pp 2 --instance main:2,7 --output experiment_logs/weight_placement/qwen3-8b-pp2.json --table experiment_logs/weight_placement/qwen3-8b-pp2.tsv` 生成 JSON、逐分片 TSV 和每 rank 汇总表；物理 rank 是部署声明的编号而非设备 UUID。单池仿真和多实例仿真构造时分别可通过 `Simulation(..., weight_placement_consumer=callback)` 或 `MultiSimulation(..., weight_placement_consumer=callback)` 在构造校验通过后接收同一清单；不提供 callback 时不生成清单，参数和容量语义不变。

独立的 `dsv4_routed_mxfp4_placements(model, parallelism)` 描述 DeepSeek V4 EP-off routed expert 的 loader 阶段子清单：每个 fused 参数分别列出 MXFP4 packed 值与 UE8M0 scale，记录两段 gate/up 逻辑范围、存储形状、字节数及无 padding 的来源约束。它复用 `dsv4_routed_loader_payload_bytes` 的配置和对齐检查，不声明 post-load Marlin 物理布局，也不包含非 routed 参数、部署绑定或完整模型清单；因此尚不能用于完整容量验收。

`dsv4_postload_inventory_diagnostic(model, parallelism)` 复用既有 SM90 DeepGEMM/Marlin 来源定义，逐项核对完整 target-off 参数 post-load 字节；非 routed 参数保留来源 loader 局部形状，并依据 vLLM 0.30 构造标注复制或 TP 区间；scale 区间以 128×128 block 为单位，sink 另标局部 padding。同一部署清单入口可显式选择 `v4_weight_storage_profile="sm90_deepgemm_marlin"`，仅接受 H200、EP-off、PP1 的来源配置；这不表示运行时已选择该后端。CLI 使用 `--v4-weight-storage-profile sm90_deepgemm_marlin`，仿真回调使用同名参数。post-load 物理 shape 和布局未知；MXFP4 权重的 `local_shape` 是未压缩逻辑分片形状，`loader_shape` 是 packed loader 形状；UE8M0 scale 的 `logical_shape` 和 `local_shape` 表示 scale 网格，`represented_weight_shape` 与 `represented_weight_shard_intervals` 单独给出其覆盖的权重，不伪造 post-load shape。该来源配置的 FP8 scale 从 loader 的 UE8M0 转成 FP32；routed MXFP4 权重保留压缩载荷与 UE8M0 scale，Marlin post-load 物理布局未知。
## 离线交互执行报告

仿真 CLI `scripts/evaluation/serving/simulation_runner_entrypoint.py` 默认生成离线报告：有 `--output result.json` 时默认输出相邻 `result.html`；未指定结果文件时在 `experiment_logs/serving_reports/<独立运行>/` 输出。可用 `--interactive-report <path>` 指定位置。默认 `--interactive-view swimlanes --interactive-detail-mode summary`，保留全程请求/Forward 和已有资源工作量，不启用 `record_execution_timeline`，也不改 execution mode、workers、retention 或 source-state。没有真实活动区间时 busy/Host 为未知（投影为 null），不能用空数组或 Forward 包络冒充零占用/精确忙碌。支持 `full` 和 `streaming_aggregate` retention；后者通过既有 object-valued projector 包装原 execution digest 与 compact 投影，consumer 解包并保留 state audit snapshot，每步只累计一次。

显式 `--interactive-detail-mode sampled --interactive-max-events 100000` 保留有界逐算子样本；只有详细模式需要物化活动区间。详细 streaming 执行在消费步骤时物化区间，在事件预算内从整个运行过程保留完整步骤样本，优先覆盖不同请求和阶段，其次保留首尾、较大批次并用确定性抽样打破平局；超预算时替换重复覆盖的样本，不按预测时长选择样本，以及覆盖全部执行步骤的资源汇总。标准外部请求结果继续使用既有安全投影；可视化另存 `report.visualization.json`，不包含请求原文、token IDs 或 kernel attributes。该文件是已有执行产物的裁剪投影，不是新的 canonical dataset。

首页先显示模型/硬件及已记录的 TP/PP、完成请求数、吞吐、平均 TTFT/TPOT、全负载模拟跨度与 simulator wall time；全负载跨度不是平均请求 E2E，指标悬停说明该口径。Forward/明细计数在覆盖面板中展开查看。延迟均值只聚合已记录的有限数值，没有 eligible 值时显示不可用。模型主要组别按已保存的 predicted operation duration 累计，包含重叠，不当作端到端关键路径；最高资源需求取逐 rank 全程均值中的最大值，不把卡间需求相加。读写证据覆盖率与未计时 Scheduler 明示。完整配置与成本入口嵌入结果卡片，带宽方向、精确范围和图形设置按需展开；图例和全部展开/折叠控件放在图下。Forward、CPU/Device、资源曲线优先展示，请求/token 数曲线仍与同一时间轴对齐并排在该实例的资源曲线下方。多实例默认折叠、单实例多 rank 默认收起各 rank 细节，选择 Forward 会展开相关实例。选区摘要按完整相交 Forward 统计请求，不把它当作按时间切分后的 token 工作量；悬停读数注明当前 rank 范围。配置先显示关键项，其他参数渐进展开并可按名称搜索。

状态与成本区保留未知信息：未记录的 failure/unsupported 计数、HTML 构建耗时及阶段重叠关系不补零。renderer 返回实际 `page_generation_s` 与字节数，资源记录耗时直接显示在成本入口，完整报告成本仍缺失时明确注明。默认入口集成必须将记录、投影、序列化和 HTML 全部计入完整实验 wall time，不能用 HTML 生成耗时代表完整开销；不能直接累加可能重叠的阶段耗时。可选 `visualization_run_costs` 仅接收有来源的 `html_generation_s`（有限非负秒）和 `scope`；未提供时 HTML 中显示未记录，独立再生成本保存在调用方记录中。

内部测量可显式使用 `--no-interactive-report --no-report-reason "<原因>"`，结果保留 disabled receipt；普通用户无需开启 flag。报告和结果路径不得相同。manifest v9 的结果/设置摘要、标准 Serving 报告保留交互入口；失败或 unsupported 没有完整 timeline 时输出真实状态/配置概要，不能伪造资源曲线。Production fixed-tier 入口也默认生成 compact 报告；其显式 opt-out 同样要求原因。扫参点的报告路径独立。

`<report>.report.json` 保存记录/投影、压缩、HTML/再生 JSON 写入和 producer 总 wall 的实际测量，页面成本区可打开该记录。独立的成本元数据与最终 receipt 写入也单独计时，嵌入页面的总 wall 明确截至结果初次写入；最终 receipt/返回值与外层实验 wall 包括后续元数据更新。记录可能嵌在仿真内或并行发生，不直接相加。默认报告不能只用 HTML 构建约一秒来代表完整开销。compact 模式最多保留 1024 条交接元数据，不保留交接算子 IR；总交接数、净 payload 字节与元数据省略数独立保留。该上限只约束附加细节，全程请求/Forward 索引仍随负载增长。报告保留运行前读取的模型/硬件配置内容地址快照；缺少文件时列出未记录，不在报告再生时读取当前配置来冒充旧输入。

报告顶部的“Experiment configuration / 实验配置”面板展示保存的模型、部署/并行、调度器 requested/effective 值、输入身份、KV 容量、Host 模型、图执行模式与源码/配置来源；提供配置 JSON 下载。大规模 Host 逐步耗时不当作配置重复嵌入；请求原文、token ID 与凭据字段不导出。报告可保留经原始 SHA256 校验的模型/硬件配置文件内容，后续再生无需读取当前工作树文件；不匹配的当前文件不能冒充原运行快照。旧记录缺少的配置明确列出未记录。

交互报告默认英文，右上角按钮切换中文；资源大图和按需展开的 Forward 明细同步语言，切换保留时间范围、筛选与原始数据。UI 翻译在现有 reporting owner 中共享，源码字段、请求身份和原始证据不翻译。全程泳道放入独立的竖向滚动区，顶部时间刻度固定，左侧泳道名称在时间平移中保持定位；实例和 rank 标题可折叠，提供全部折叠/展开。全程导航条始终显示完整运行范围，拖选区平移、拖两端缩放；说明与模型假设收进可展开区域。

HTML 无 CDN、在线字体或服务依赖，可独立复制并用浏览器打开。数据在 HTML 内以 gzip + Base64 保存，打开时由浏览器自动解压，无需解压包或配套文件；需要支持 `DecompressionStream` 的现代浏览器，不支持或数据损坏时显示加载失败并禁用控件。压缩减少磁盘体积，不代表等比例降低浏览器内存。普通滚轮（含触控板横向滚动）左右平移，Alt＋滚轮围绕鼠标位置缩放；拖拽和平移滑条共用预测模拟时间轴。总览、资源大图和 Forward 明细采用同一交互；资源大图的 Shift＋拖动保留框选缩放，普通拖动用于平移；实例、rank、操作筛选及点击详情不重新运行仿真。明细的非连续采样区间、请求/批次覆盖数、事件预算及省略数量在页面显示。请求被覆盖仅表示它至少参与了一个保留明细的批次，不表示它的全部生成过程都被保留；事件预算不足时会明确显示未覆盖数量。请求选择器筛选该请求参与的已保留批次；批次中的操作仍由所有参与请求共享。资源表始终按实例/rank和时间窗口汇总全量工作，不随请求筛选变化。省略区间按步骤画聚合 envelope，包含空隙，不冒充连续忙碌；较早保存的投影没有操作分类时明确显示不可用。批次计算属于所有参与请求，不代表任一请求独占。Host 为被模拟系统的提交/运行时；Midlayer 仿真 wall time 单独显示。未开启 Scheduler Host 模型时，零值字段代表未计入该机制，不是实机观测为零。

默认视图放大已保留样本中请求数最多批次的首层，按实例与 rank 分行展示计算、通信、KV 更新/搬运及 Host 提交，用颜色区分 QKV、Attention、FFN、MoE、归一化及各类通信。分行是按操作类型的展示，不能推断不同类型使用独占硬件队列。HBM 需求行将该操作的逻辑字节均匀分摊到它的已记录区间，并非另有一个被观察到的 HBM 操作。模型瓶颈组行直接显示预测器的 `bottleneck_group`；Roofline 未细分或旧投影未保留该字段时保持未知。下界较大项行比较生产资源输入的计算量/对应峰值和逻辑字节/标称 HBM 带宽，只是独立下界的比较，不包含启动、同步，也不是物理流量预测或跨操作关键路径。

逐操作模型瓶颈组与资源服务时间来自原有预测结果，新生成的可选可视化记录在每步骤汇总所有操作的模型瓶颈组，即使明细超出预算仍保留整步的组别计数和预测操作时长。整步汇总不表示组别在步骤内的执行顺序；跨 rank 或重叠操作累计值可能重复，不能当成端到端瓶颈耗时。已经生成的旧投影若没有预测字段，可直接重绘并显示可用的工作量下界比较；模型瓶颈组标为未保留，不能从总时延反推。

可选记录还保存预测器提供的单算子内部关键路径、支持级别及限制，路径目录最多 4096 项，每步每瓶颈组最多汇总 32 条路径；未提供或超出预算的次数单独计数。路径是模型诊断，不是请求级关键路径，也不能把资源服务时间相加画成实际时间段。旧投影未保存时显示不可用，不从时延推断。

原 `breakdown` 视图的资源图随时间窗口变化，busy 为 Device 区间并集除以窗口，compute 为生产 predictor 的操作量除以对应资源标称容量与窗口，HBM 为 compulsory 逻辑字节需求除以 HBM 带宽容量的解析下界。该 HBM 数值不是物理 transaction 利用率；未提供资源输入的操作计入缺失计数，不补零。汇总模式对窗口内需求按步骤执行 envelope 均匀分摊，busy 仍按记录区间并集统计；这是时间分布近似，不代表瞬时硬件计数器。算力按 kernel dtype 和资源 role 区分，source 内部混合精度仍受原资源合同约束。超过 100% 的需求保留原值，图轴自动扩展。48 个等宽窗口用于资源趋势图，不是新的预测。

互联为生产 owner 输出的算法字节量，按具体 endpoint 与配置链路带宽统计，包含调用的起止、请求净 payload 和发送/接收端；同一物理调用只记录一次，不把多个 endpoint 相加后再除以单端带宽。算法字节不含协议开销或重传，非 ring/未建模 collective 显式不可用。容量分别展示 HWSpec HBM 上限、已配置 KV 预算及已知 source activation 峰值；保持 `complete=False`、`normal_capacity_accepted=False`，不插值成连续精确显存曲线。

默认报告只保存可视化需要的操作类别、层号、shape/dtype、绝对时间端点、预测总时间/启动时间、模型瓶颈组、非零资源服务时间，以及计算量/逻辑字节和模型路径证据。完整 `source_node_ids`、`t_per_group_s`、Host 依赖边和覆盖操作列表不再写入；Host 保留时间、类型、对应 call/graph segment、暴露/重叠时间及批次关联，不支持从此投影恢复完整 Host 依赖链。资源时间的零值仅按显式 sparse 编码约定省略，不表示缺失资源证据为零。未知资源输入及路径仍明确标记不可用。源节点只用于生成操作分类和层号，再丢弃；分类未扩展为硬件预测逻辑。

可视化 JSON 使用报告内的 `visualization_storage` 表去重算子描述、实例/rank/resource 及 call ID；它是可再生报告的存储编码，不改变 execution artifact 或 canonical schema。读取入口接受旧的未压缩投影及新的表引用投影，并拒绝非法引用。绝对端点保留原始数值，不通过相对时间加减重建；HTML 初始化时解析表引用。旁边的 `report.visualization.json` 用于离线再生成，浏览页面时只需 HTML。字段精简不会自动增加明细预算或恢复旧投影已省略的批次。

新增 `--interactive-view swimlanes` 选择端到端泳道：全部 forward 标记、CPU 调度/提交、每实例/rank 的 Device 活动及 Tensor、Vector、L2 BW、HBM BW、Scale-up BW、Scale-out BW 共用一个时间轴，缩放和平移同步。默认仍为原有 `breakdown` 视图。曲线依据全步骤汇总而不是已保存的采样明细；Tensor 将各精度工作量按对应标称容量归一化，Vector 取不同向量资源角色归一化需求的最大值，不把不同执行单元相加。L2/HBM 带宽图均使用 `PredictionInput.memory_paths` 中 `l2_cache` / `dram` 节点的读写字节，直接读取预测器已有缓存/LRU及写回处理后的内存路径，不用必要逻辑字节替代 HBM 流量，也不从资源服务时间倒推字节。H200 泳道图使用独立的显示峰值参考 `microbench/nvidia_h200/references/l2_peak_metadata_sm90.json`：L2 读 6016 B/clk、写 3008 B/clk，按 1.98 GHz 换算为 11.91168 / 5.95584 TB/s；HBM 读写共用公开标称 4.8 TB/s。L2 综合占比取窗口内读占比和写占比的较高值，HBM 综合占比取读写占比之和；读写合计 GB/s 始终是字节流量之和，不把不同分母的百分比相加，也不把 NCU tag-sector 合计峰值当成已证明的同时读写容量。下拉菜单和大图可切换读/写/综合，读数显示各方向百分比、GB/s、峰值分母及合计流量；超 100% 保留。显示参考由 src 导出，报告层只归一化；保存投影固定该参考供离线再生，不改变 HWSpec 或预测定价。未配置独立峰值的硬件继续明确标为模型配置带宽，不冒称标称值；混合硬件报告尚缺逐行硬件身份时不套用 H200 参考。必要逻辑字节只保留为单独参考量。

方向统计要求预测器导出读写方向账本；`MemoryNodeDemand` 只容纳正字节，完整方向账本中省略的下游节点表示已建模的零流量，例如 LRU 命中不再访问 DRAM、脏输出尚未回写。没有方向账本的操作保持未知并单独计数，不补零。统计只用聚合路径，不把 pipeline 内分阶段节点再累加。已有 LRU 在各预测器支持的发射/执行上下文内生效，报告不运行第二套 LRU，也不冒称已经证明了所有算子之间的全程连续缓存状态。通信按配置的链路及 endpoint 统计，单卡无该类链路时显示 N/A。窗口分箱宽度随缩放显示；超过 100% 的需求保留并扩展坐标，不截断。CPU 调度未启用耗时模型时，显示 schedule trace 中的调度/完成事件标记并注明未计时，零时长不画成非零活动；不为绘图自动改变原 case 的 timing mode 或引入额外 Host 定价参数。

Forward 选择器同时显示实例/后端轮次、Prefill/Decode/Mixed、本轮参与 request 数和调度 token 数（input_tokens + output_tokens，不含 padding）。总览增加每轮请求数与 token 数泳道，token 按 Prefill 输入和 Decode 输入堆叠；这些是本轮工作量而不是请求总长度或最终生成 token 总数。资源曲线默认使用可读高度和各自自动纵轴，也可切换固定 0–100% 或更高曲线。每行显示窗口均值、分箱峰值和纵轴刻度；不同自动刻度下不能用曲线高度跨资源比较。悬停给出浮动数值和确切分箱区间，点击固定读数。点击资源名、双击曲线或使用“单独放大”按钮打开大图；支持时间缩放、拖选区间和百分比/GB/s 读数，主视图时间轴同步更新。每点仍是窗口均值，步骤内部的均摊假设不会因放大而变成 kernel 级或硬件瞬时计数器。

`--interactive-detail-mode summary` 为默认，不保留逐算子明细；`sampled` 继续受 `--interactive-max-events` 限制；配合 `--interactive-view swimlanes --interactive-detail-mode all` 则完整记录每次 forward，忽略采样事件上限。全量模式每消费一个步骤就将请求安全投影独立压缩写入 `<report>.iterations/iteration-*.json.gz`，不常驻全程执行 IR；全程汇总和所有 iteration 的轻量索引继续保留。HTML 内嵌这些压缩块，打开只解压总览，点击某次 forward 才解压它，并复用原算子 breakdown 查看器；切换后替换上一轮明细。新视图需要同时支持 `CompressionStream` 和 `DecompressionStream` 的现代浏览器。无明细的采样 forward 明确显示“仅汇总”，不会用相邻样本补造事件。

泳道版的 `.visualization.json` 是带 chunk 文件 SHA256 和原始绝对时间平移量的再生索引，重新生成时必须保留它引用的 `.iterations/`，并校验路径、长度和内容哈希。输出使用新任务目录；已存在的 chunk 不静默覆盖。最终 HTML 独立可复制，浏览不需要旁边的 JSON 或目录。`report_interactive.py` 会识别该索引并重绘，无需重新仿真。源文件中的请求文本、token IDs、kernel attributes 不进入压缩块。泳道存储是现有可视化报告内部编码，canonical 和执行合同不变。

无需重新仿真可更新页面：

```bash
PYTHONPATH=. /home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python scripts/evaluation/serving/report_interactive.py experiment_logs/<run>/report.visualization.json experiment_logs/<run>/report.html
```

CPU-only 示例入口 `scripts/evaluation/serving/run_interactive_example.py` 提供 `single`、`tp2`、`pd`（完整 Qwen3-4B）和 `four`（既有四池小模型 fixture），传入 `--output`、`--sdk-python` 和 `--source-root`。SDK Python 保留虚拟环境路径。前三者使用真实 CPU vLLM 0.30、Roofline 和现有 H200 Host launch profile；四池使用既有 vLLM 0.11 token-prefix fixture，未启用 Host 定价。示例使用显式测试 KV 预算，不表示正常容量或实际 provider 验收。`--no-record` 用于同场景结果对照；`state.json` 保留请求完成、token、路由、传输和 activation 状态。真实请求集按有效 case 配置构造原仿真命令后增加报告参数，不根据 case 文件名推断运行时版本。

普通静态模型报告也默认生成独立离线概要；它展示已记录的图规模、串行预测时间、未计价项和配置，不生成 Serving 活动区间。历史 Serving 结果若没有保存可视化投影，报告仅生成 partial 概要，展示已有指标并明确活动区间不可用；缺失均值不会由分位数猜测。保存过的投影按其已声明视图重建，未知视图会报错。
