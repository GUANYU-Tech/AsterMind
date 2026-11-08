# 建模决策录

全部硬件的定价机制裁决与未闭合事项集中在此。每条带状态：`active`（生效中的机制结论/护栏）、`excluded`（已排除的候选）、`unsupported`（机制真实但无合法来源，残差保留并如实报告）、`open`（未闭合事项，带重开条件）。只保留结论、判据和不得做的事；诊断过程与效果数字见 git 历史原文（910B 四条原 `docs/ascend_910b/` 四份、C500 一条原 `docs/metax_c500/moe_hbm_unsupported.md`，2026-10-12 合并为本文）。

## 整卡 Roofline 容量（active）

仅有整卡算力和内存带宽的硬件可在同一个HWSpec/component schema中使用`features.kernel_memory_model=roofline`及`capacity_per_second`；不反解或虚构时钟、CU/SM数、缓存和launch。每个资源表示整卡SI容量，不能混用per-clock、variant、children或原生服务；普通HWSpec拒绝这种资源，公共predict接口只允许roofline方法。缺少FP32/FMA、SFU、整数或互联容量时保留缺失，正需求不得借Tensor吞吐计价。已有数学GEMM/BMM程序可复用，与native launch、模型后端绑定、正常容量及硬件精度验收分开。

BW1100配置中的容量来自用户提供的标称规格，未核实dense/sparse口径、GB/GiB口径或持续利用率；仅作为明确命名的参数场景，不升级为厂商或微基准真值。配置按十进制GB、TB/s表达，INT8使用TOP/s单位；不注册BW1101别名、不套用H200互联或执行参数。补充正式规格/独立机制证据后才能替换相应参数或开启原生能力，不根据production duration或对齐误差调参。

BW1100 RoCE带宽场景（active）：用户明确普通服务器每两卡共享400Gb/s RoCE、交换机可视为全互联；按Ethernet通常端口口径为每NIC每方向50GB/s，协议/PCIe损失未声明，不造效率折扣。显式participant_nic_ids绑定物理共享，单collective按最大NIC发送/接收字节负载计理想带宽时间，不能给两张并发卡各50GB/s，也不能将单活跃rank无条件减半。沿既有ring/pairwise字节语义，未知同NIC本地路径拒绝；不声称跨独立调用的争用、Serving/provider或完整时延已闭合。链路/启动延时仍open并保留null，不填0/100ns，不套scaleX40、NVLink或NCCL；参数与用户声明的来源口径见[硬件配置](../configs/hardware/hygon_bw1100.yaml)。

## Intel AGMF039R47用户设计（active / partial）

`intel_agmf039r47`配置按每片FPGA计，不按双芯片卡池化HBM/DDR；用户指定MM4/64/64@400MHz、相同低比特吞吐，不能因FP4位宽降低再乘峰值。MXFP4数学MM计完整几何波数、packed weights及group32 E8M0 scales；Vector沿用户声明的32 element/cycle整算子服务与35周期延时，不借GPU SFU速率。每片400Gb/s RoCE、20μs单向网络延时与DDR12.8GB/s读写合计分别处理；允许收算发流水重叠，不把非阻塞交换机独占化，也不放大共享NIC、MM或DDR容量。DDR4公开器件读时序仅作条件参考，不能替代未知控制器/DMA或完整收发延时。完整expert量化、异构Serving caller、GPU非矩阵服务与真实provider仍未闭合；源参数、设备粒度与未知项见[硬件配置](../configs/hardware/intel_agmf039r47.yaml)。

## 910B

### 1. FIA 写侧计价（active）

#### 被测量口径（三选一，换口径会换结论）

msprof 的 AIC 写必须固定用权威口径：**`write_main_memory_datas − UB_to_GM`**（前者是权威写计数，AIV 侧闭合于 `UB_to_GM`，相减得 AIC 侧）。`L0C_to_GM + L1_to_GM(estimate)` 不可用（`L1_to_GM` 是 msprof 推算，且带 `(estimate)` 的字段不得进入任何诊断构造）；仅 `L0C_to_GM` 也不可用（含到达内存前被丢弃的 padding）。引用任何一个写侧数字之前必须先说明口径。

#### 一笔 32.8–34.3 KiB/AIC 核的固定写：KFC workspace 清零（尚未计价）

权威口径下，实测 DRAM 写减源码环的 distinct extent 后，残差恒为 32.8–34.3 KiB/AIC 核，横跨两个 kernel、七个 layout×阶段组、核数 1–24 全域，对全部 shape 参数不敏感。归属是 AscendC 框架而非算子源码：CANN 8.5.1 `tikcfw/impl/dav_c220/kfc/kfc_comm.h:357` 的 `ClearWorkspaceImpl` 在每个 MIX kernel 入口按核清零 KFC 消息区，头文件常数算得每核 32896 B = 32.125 KiB（实测残差/该值 = 1.02–1.07）。它是 `copy_cbuf_to_gm`（L1→GM），不是 Fixpipe L0C→GM——这解释了权威计数为何比 `L0C_to_GM` 多约 33 KiB/核、算子源码里为何找不到写者、以及为何只随核数走。

**保持不计价。** 它既不是 MTE3（AIV）也不是 Fixpipe（AIC L0C），挂在现有任一条 path 上都会混淆两个引擎；正确做法是新增一条 path，且框架清零发生在任何 phase 之前，`_attach_cann_standard_decode_phase_ledger` 的逐 phase path 校验需要一并处理。launch 几何的机器可读版本见 `microbench/ascend_910b/references/fia_workspace_ring_geometry_cann_8_5_1.json`。

#### MLA 的两个环用两个不同的轮转键

GQA `mm1Res`/`mm2Res` 与 MLA `mm1Res` 按 `task_index % PRELOAD_NUM` 取槽；**MLA `mm2Res` 用 head-task 计数器 `bn2IdxInCurCore % preLoadNum`**（只在 `isFirstSInnerLoop` 时自增，`fia_kernel_nonquant_mla.h:498-561`）。launch 侧已按此改正。

#### MLA regime B 例外（未裁决）

MLA 以 **B≥8 且 KV≥2048** 为界：regime A（环不回绕）模型精确；regime B 实测说明回绕复用不是全额免费（预测写流 146.6 MiB、实测 23.4 MiB，84% 被吸收）。剩余候选是回绕间隔内的 KV 读把 slot 挤出 L2，判定需要带中间流量扫描的驻留微基准，未采集，保留上报。

#### prefill AIV：写流精确，被测量违反物理下界

SplitFuse prefill 四个 layout 的 AIV 写流被完全算对（pred/meas 1.000），误差全在 unique 那一步（DRAM pred/meas 1.20–1.28）。但 94 个 prefill case 有 54 个实测 unique extent 小于输出足迹——"写一次"的缓冲只有 70–94% 到达 DRAM，物理上不可能，偏差在被测量侧（profile 窗口关闭时仍脏在 L2 的部分窗口后才回写）。**在被测量修好之前，不对该项调参——按此残差的任何改动都是对一个违反物理下界的量拟合。**

### 2. FULL-graph padding：appended-slot FA 按 captured 数组定价

#### 两种 padding 形状，只有一种可重绑

`GraphAttentionShape.from_token_shape`（`src/serving/workload/graph_padding.py`）只有两种形状：**appending**（整批 uniform decode，尾部追加 dummy 槽位，真实请求槽位逐字未动）与 **extending**（prefill/mixed，拉长最大逻辑 segment）。`_has_graph_padding` 拒绝 per-CTA source schedule 的理由只对 extending 成立；appending 形状下按 model-input 数组定价的是已被分派的工作。

#### 重绑判据（fail-closed）

`_cann_fia_appended_slot_execution_kernel`（`flash_attn.py`）只在下列全部成立时重绑：`graph_runtime_mode == "FULL"` 且命中 CANN 标准 GQA/MLA decode 算法集合；`cann_source_task_count` 在 launch 参数中；model-input 槽位数不少于逻辑槽位（少于则 `ValueError`）；逻辑前缀逐槽位未扰动；dummy 尾部每槽 `q_len == kv_len`。任一不成立即回到 fail-safe 定价。重绑出的 spec 只在该次定价调用内有效，token 记账仍用逻辑 token 数。

#### FIA single-task LUT 是稀疏精确匹配，缺格即 fail-open

`single_task_active_aic_groups` 是精确匹配表：active AIC-group 数不在表内时 FIA plan gate 直接放弃，退回 launch 常数定价。覆盖洞已补 10 格（group 2 因跨轮 span 16.15% 超 15% 上限 withheld）；原 13 格逐字节未动，新点在独立 provenance 的 `extended_single_task_points`，两串只在 HWSpec normalization 时合并。FA 重绑与 LUT 补洞关闭的是不同的 fail-open，两者同时在场才没有 FA decode kernel 按 launch 常数定价。

#### 残差边界（G6 unsupported，不硬塞机制、不加 floor、不 clamp）

合并后仍有 63 个负向 padded step（合计 −23.269 ms）：`FLASH_ATTN/decode` −8.72 ms（124 个，其中 89 个两侧 grid 相同，幅度约自身 0.4%–0.8%）归因于"按运行时重推导 SplitCore 布局定价"的伪影——910B 真机预登记 A/B 证伪了模型的省时方向（实测持平或上升），修复方向（padded 侧 launch 应反映 capture 时刻输入）留作专门后续；`VECTOR/*` 负向见 §3 残差边界。

### 3. FULL-graph padding：padded VECTOR 按物理 extent 定价

#### 重绑而非拆门

`vector.py` 三处 `not graph_padding` 守卫逐字节未变：`_cann_full_graph_source_program_execution_kernel` 在守卫求值之前把 padded spec 的 shape 改写为 launch 记录描述的物理 extent，守卫不再触发是因为 spec 不再是 padded。不变量：**`pipeline_plan` 不得静默变成 `None`**——padded 与 exact 必须在同一模型深度上定价。

#### 判据来自 launch 记录自身（无按卡开关）

NORM 族恒等式：`blockFactor == ceil(physical_rows / blockDim)` 且 `blockDim <= physical_rows`（CANN NORM ABI）；SwiGLU 族：`total_tiles == ceil(rows/base_row_len) × ceil(columns/base_col_len)` 且 `blockDim <= physical_tiles`（`GluSingleTilingCalculator` 直译，CANN ops-nn@9f8a66e）。两式在物理 extent 成立、逻辑 extent 被违反，记录即可自证；证明不了就保持原有 padded 契约，fail-closed。早前在 HWSpec 加按卡开关的做法因理由循环（用"按物理 extent 跑"论证"按物理 extent 定价"）已回退，不得恢复。

#### masked 槽位计为匿名占用

padding 行/tile 在任何已调度请求之外的 captured buffer 槽位上跑保留程序：NORM 仍经 MTE2 载入、AIV 归约、MTE3 写回，SwiGLU 仍走完整数据流。重绑把这些计为匿名 AIV/MTE 占用，不变成语义工作，不携带 per-request 身份。

#### 锯齿与残差边界

负向 silu 差额的 82.5% 来自跨 pad 时 CANN host tiler 串行轮数下降（2→1 轮）：padded extent 上选了另一个合法 base tile 使 `total_tiles` 恰好降到 48 消去一轮空转——**工作上升、串行下降，不得表述为"padding 减少了工作"**。其余 −3.7728 ms 无已验证机制，按 G6 声明 unsupported（111 个 silu、105 个 add_rmsnorm、105 个 rmsnorm），给粗路径加 floor 仍排除（误差补偿，与零拟合底线冲突）。kernel evaluation 在结构上不受影响：`graph_*` 属性只由 `src/serving` 产出，kernel evaluation 不经过该路径。

### 4. VECTOR row/hidden 复合耗时候选：已排除（不是 unsupported）

**最终结论：该候选对本 Goal（910B qps64 TTFT Processing 过计）被排除。** 早期采样（13 个 M 点全部 ≥63）只覆盖真实 qps64 调度轨迹按发射次数加权质量的 18.61%；按真实质量分布重测后方向反转——真实主导区间（M≤63，占 83.76% 质量）模型**欠计** 13×–44×（伪影校正后仍成立：实测 floor 约 33–41 µs vs 模型预测 1.6–3.0 µs），M≥63 尾部的"斜率过陡"真实存在但每点权重 <0.5%，被稀释后不主导聚合方向。净复合（rmsnorm+add_rmsnorm+silu 逐点求和）交叉点 M≈2339，远超 TTFT 枢轴 ≈1009；枢轴相关区间净复合仍欠计，修复 silu 斜率只会让净欠计更欠——对本 Goal 零正向收益。

已排除的子候选：row-group setup 摊薄错误（结构正确）、7 个 typed intrinsic rate 常数量级错误（验证误差 0.001%–8.1%，全部合规准确）、generic 聚合速率主导（仅占 1.49/16.94 µs）、无流水线重叠建模（`_attach_cann_rms_row_group_pipeline` 已含显式重叠语义）。遗留两个真实但越界的现象给未来独立 Goal：M>700–1200 尾部斜率过陡；M≤700 主导区间的欠计 floor（是否在真实 serving 下暴露取决于生产是否 host-dispatch-rate-limited，未验证）。

## C500

### 5. MoE HBM 欠预测裁决（unsupported）

C500 `fused_moe_kernel*` DRAM 流量欠预测（MAPE 20.98%、176 case 中 59 越界、全部欠侧）的裁决。**边界是激活足迹相对 L2 替换容量（8 MiB）**：足迹在 L2 以内时模型精确，超过约 1× L2 后随足迹单调恶化（BM16/BM64 两种 tiling 签名与 shape 完全共变，不是自变量）。

**根因是执行顺序，不是 L2 本身。** GSM 组内相邻程序共享权重 tile，共享只在共享者于短时间窗内相继访问时成立；miss 核心按 104 个 CTA 一波的锁步顺序回放，共享全部成立；硬件上 CTA 进度失锁（共享者相差 30–80 个 k 步，超过 8 MiB L2 约 30 步的可保持复用距离），每个共享者重新取数，DRAM 读为锁步模型的 2–5 倍。只有 C500 显著是因为 L2 小（A800 40 MiB、910B 约 160 MiB 能容下同样的相位差）。实测逐 CTA 时间戳排出的访问顺序在普通 8 MiB LRU 上回放即复现计数器，锁步顺序低 2–5 倍——替换判定正确，缺的只有执行顺序。

**护栏：`wave_size = 104` 是锁相态的正确组宽（每 AP 4 槽 × 26 AP），不是 bug。** 它与锁步理想化两处偏差互相补偿，单独把 `wave_size` 改成物理常驻 CTA 数（416）实测反而更差（MAPE 20.98% → 22.01%）——**在锁步假设被替换之前，两处都不要动**。

**未修复原因与重开条件。** 失锁幅度由 AP 内部发射/调度微结构决定，对代码形态敏感（同几何 +2 个 ST 寄存器使干扰 0.42→0.26），探针数值不能移植到生产二进制；零拟合地预测需要周期级 AP 模型，超出 L2/DRAM 流量模型 owner；任何固定相位参数或按生产流量反推的常数都是拟合。重开需三条同时满足：(a) L2 set/slice 几何 + MSHR 容量的权威来源；(b) miss 核心升级为地址级仿真的架构授权；(c) 代码形态轴的物理解释或生产二进制逐 case 的 regime 判定手段。在此之前该残差按 unsupported 保留并如实报告。探针与测点见 `microbench/metax_c500/references/production_stream_phase_law.json` 与 `stream_sharing_interference.json`；完整过程记录见 git 历史中的 `docs/metax_c500/moe_hbm_unsupported.md`（2026-10-12 并入本文）。

## H200

### V4融合Q/KV RMSNorm的转换与算术（active）

`FusedQKVRMSNormKernel.kernel`分别处理Q/KV两行：BF16输入与gamma各转FP32，平方/最少归约加法/均值除法/epsilon/rsqrt及两次乘法均保持FP32，仅在最终store转回BF16。令E=T*(Q+KV)、任务数R=2T，则普通浮点3E+2R、归约加法E-R、rsqrt R；三处转换保留2E次BF16→FP32和E次FP32→BF16的typed计数，不使用INT32 ALU吞吐计价。已有逻辑输入/gamma/输出字节、launch和硬件速率不变；padding、gamma重读、compiler转换/归约服务及真实provider仍未验收，不根据duration选公式或参数。

融合Q/KV RMSNorm的epsilon须从模型配置沿workload/lowering传入KernelSpec；与普通RMSNorm共用正有限数或None的可选属性边界。旧generic调用未声明时只保留None，不补数值默认值；当前准确模型源码检查必须匹配其显式epsilon，不能以运算次数不变代替该语义参数。此字段不改变数学次数、字节、launch或硬件速率。

### V4 attention 的数学 Roofline 与 native 资源投影（active）

上游`tests/kernels/test_deepseek_v4_cpu_kernels.py:_flash_mla_eager`给出scaled QK、拼接每head一个learned sink、softmax及排除sink概率的PV；NVIDIA prefill/decode包装均传入同一层的scale和sink。第一阶段采用显式数学分解：每row的N个有效KV加1个sink，共N+1次shift/exp/normalize-divide、N次sum-add及N次浮点max比较。令P为query-head与有效KV的pair数、R为query-head行数，则普通浮点需求为4P+2R，exp为P+R，另加prefill共同有用的448值/row解量化乘法；浮点max仅保留typed计数，不进入INT32 ALU。QK/PV矩阵需求和既有逻辑访存投影保留。本条定义数学Roofline的稳定归一化指数程序，不选择FlashMLA的online/split算法、compiler issue或FP32求和顺序，不证明NaN/inf数值等价；native resource_input仍是原有粗投影，不能据此宣称已与源码指令对齐。不根据duration选择分解或调整速率，完整cache/metadata/scratch与provider验收继续分列。

### V4 inverse-RoPE/FP8 的数学资源分类（active）

SM90的fused_inv_rope_fp8_quant按有效旋转值计每值两乘一加/减，scale_raw的1/fp8_max是constexpr常量，随后每scale一乘；每个有效FP32值的x/scales_exp除法仍是一次普通浮点FLOP。令E=T*H*D、G=E/128、R为每head的RoPE宽度，则有用FMA类需求为3THR+G+E，log2/exp2为2G个SFU数学操作。浮点abs、amax/epsilon max、ceil、两侧clamp和FP32→FP8转换保留typed计数，不能计到INT32 compute_alu或伪造额外reciprocal SFU；实际shader服务尚未绑定。source中两个完整旋转候选的表达式站点数另列，不能据其SSA文本选择compiler CSE、masked issue或实际native指令数。FP32 scale的4行对齐零写和既有逻辑payload保留；本条不调整硬件速率、不按duration选机制，也不声明实际provider或数值相等。

### 6. DeepSeek-V4 驻留与 Marlin 残差（open / excluded）

**Attention 驻留缺口保持 open。** 64-head CTA pack 与 per-request split plan 是源码或离散 launch 事实，可用于计数；它们不能证明 gathered workspace 在重读间隔内驻留 L2。仅用单请求 workspace 小于 L2 容量推断全额命中，以及把 `microbench/nvidia_h200/references/l2_operand_residency_curve_sm90.json` 的 repeated two-read/one-write stream 曲线直接用于不同访问流，均为 excluded。该 reference 明确禁止超出 stream contract 外推；相同总 footprint 不能证明访问顺序、复用距离或 cache policy 相同。缺少合法驻留证据时保留显式未建模复用的 DRAM traffic 投影，不能从 production duration 较短推断 cache hit，也不能声称该投影给出端到端时延上界。

V4 C4 decode的每层wrapper调用compute_global_topk_indices_and_lens：读取并写出完整INT32 top-k行（含-1）、读取每token的INT32 request索引和bool有效标志、写INT32长度，随后attention消费global indices和长度。empty_like/empty为fresh输出，unsqueeze/view不添加copy；不能仅计有效候选索引读取而省略完整行producer。短上下文all-completed状态由源码确定非负索引数，另计每项4B block-table role read；scored路径仅有候选容量时不假定真实非负项数，其page-table读取保留参数化且未计价，原attention候选容量投影不升级为实际ID或点预测。is_valid_token仅把最终长度归零，不跳过索引转换和写出。C128的对应索引在metadata阶段准备，不能将该C4每层producer重复加到C128或SWA-only。整数/比较服务、真实page IDs、FlashMLA内部scratch及完整attention验收仍未关闭，本条不调整速率、cache驻留或native/provider结论。

V4短上下文fill复用model构造时一次torch.empty创建、逐层原样转交的INT32 topk_indices_buffer。每次只读INT64 positions并完整覆盖当前[T,index_topk]前缀（含-1），不读旧buffer、不逐层分配global temporary，Indexer返回三个None。decode读取前缀，prefill从num_decode_tokens起切片；二者均为view，不能添copy或把两者错用同一offset。source模型positions、每rank共享buffer身份、切片shape/stride/offset、cache就绪依赖、reader在fill之后及上一层读完再复用都须同时核对；只填声明不能通过。max_num_batched_tokens仍为容量参数且必须至少覆盖当前T，不以source对象账本接受实际allocator、正常容量、spill、GPU issue或完整attention callee。

C128 metadata的active view宽度取全batch max_seq_len，先对floor(max_seq_len/128)取至少128的next_power_of_2，再以构造期ceil(ceil(max_model_len/128)/128)*128容量截断。不能对每个prefill/decode子批单独取128-aligned ceiling；例如全batch长度32896、声明max_model_len=1048576时两阶段都用512列，即使prefill自身没有压缩状态，其combined indices仍fill640列。构造期max_model_len是独立运行输入，不能默认为HF max_position_embeddings：RuntimeInputs.v4_source_max_model_len只声明source场景，不证明真实runner。缺失该值时仅在所有合法构造容量给出相同active宽度的区间接受；有歧义则拒绝点预测。模板缓存身份包含该输入，避免跨容量重用旧宽度。只修正active view及其已知fill字节，不接受backing stride/峰值容量、实际allocator、metadata准备流量或native/provider绑定。

V4声明的modular pure-text top6路由返回两个独立fresh tensor：FP32 weights和INT32 IDs，直接供同rank的专家调用读取。CUDA hash包装另以empty分配INT32 token_expert_indices[T,6]，但FP32/hash/top6/renormalize快路径在launcher中提前返回，未将此指针传入所选kernel；该临时量未初始化、未返回、无payload读写，不能当成第三个输出或初始化流量。普通Triton包装只分配两个返回tensor；bias.data为参数view，hidden_states仅提供shape/device，INT32模型input_ids与INT32 hash表相同dtype时不产生全局cast。EP-off由上游配置校验排除EPLB；当前正常源程序使用默认无capture/replay、无融合shared列的返回路径，IDs同dtype不复制。验证须绑定实际logits producer、模型行顺序、typed outputs/consumer ports、lowered路由源码对象、FP32权重读入与INT32索引接口及依赖；不绑定真实checkpoint分配、外加capture/replay模式、compiled spill、实际provider或allocator容量。既有source算术、字节与价格不变。

V4 packed SWA融合insert的BF16 Q/KV输入只读，返回fresh BF16 padded Q；所有live/pad元素由kernel写出，empty不增加初始化流量。原KV不被原地RoPE改写，attention从既有uint8 cache读KV。该cache的每页数据区按576B/token排布、scale区从B*576开始按8B/token排布，不能用连续584B行地址代替；cache.stride(0)可以包含页padding，真实B/S和slot IDs仍为未绑定参数。prefill读取Q的decode-prefix之后的suffix，decode读取prefix并以unsqueeze做view；同rank视图与lowered就绪依赖必须同时核对。仅此源对象和地址程序可静态闭合，实际页归属、跨步覆盖/回收、reader内部scratch、编译spill及runtime/provider继续独立未验收；不改变既有数值工作量或速率。

V4状态/输出cache在generic device-pool、单head、indexer允许的BLHNC/BLNHC布局域中，按同一pool block的最大group字节P分块；各group从byte0重叠，同group内layer的页区间由spec前缀和page_size确定。同一非null活跃pool ID不能被独立分配给不同group，须保留读引用并只修改writable页；给每个group假设独立pool会使同ID的真实字节区间重叠。kernel-block拆分还要求manager页dense且无padding/其他layer间插，kernel ID为manager_id*R+subblock，stride变为manager_stride/R，不能直接混用两种ID。source条件程序和CPU合成几何反例只证明这些规则及其适用域；不证明实际layout/groups/page IDs、unpinned借读安全、GPU fence、正常容量或完整跨步cache验收，不调整Roofline点价格。

V4 TP CUDA fused embedding在一维contiguous INT32输入、BF16本地weight shard域中，直接转交原input_ids，不走TP1的全局long转换或非fused路径的mask/shift临时tensor。_custom_ops包装只empty分配一份contiguous BF16 partial输出，kernel对owned行copy、其余行写zero；即使rank没有owned行也必须完整写出。empty不添加初始化流量，输出不alias输入或weight，并须活到TP all-reduce读取。独立检查需同时绑定所有rank的typed partial ports、collective输入及lowered先写后读，不能仅凭fresh声明关闭数据流；collective内部scratch、编译spill、真实allocator/runtime另列，不改变原算术/字节Roofline价格。

V4最终RMSNorm的epsilon必须由config.rms_norm_eps经模型语义、workload、lowering传入KernelSpec并由独立源码比较核对。VECTOR/rmsnorm的optional epsilon仅接受正有限数或None；未声明时保持None，不能补成某个默认常数。此参数改变数值语义但不改变已有运算次数、逻辑字节、launch或价格，也不选择IR/vllm_c/batch-invariant/provider分支；分支内部临时物化与实际runtime继续未验收。

V4 modular Marlin在Standard contiguous BF16、EP-off/no-LoRA域中，workspace13与workspace2由同一个uint8 workspace的两个256B对齐请求区间提供，不是两份独立allocation。Marlin交换这两个workspace的用途：gate/up与加权专家输出复用workspace2同一offset，须在activation读完gate/up后才写W2；activation使用workspace13前缀。generic模块先建立的fused_out view在符合shape/dtype/device/contiguous条件时被调用方fresh empty_like输出替换，NoOP finalizer收到同一对象时不复制；不能把未用view当成写出或再加finalize copy。请求区间与view不等于实际workspace历史容量、traffic或峰值；FP32 GEMM scratch、routing缓冲、解量化临时区、lane并发与GPU fence继续单列，不据此关闭完整MoE或调整价格。

**MXFP4 routed-MoE 的 Marlin 残差保持 open，本轮不作经验补偿。** H200 launch owner 从 vLLM 0.30 `marlin_moe.py` 与已有 launch observation 推导 M tile、grid、block 和路由行数；这些证据不提供 WNA16 MMA、解量化与流水线组合的实际 issue rate。现有误差不能独立证明唯一根因就是 Marlin 吞吐率，禁止按时延反解有效 TFLOPS、缩放因子或 per-shape 修正。没有独立参数时保留原解析结果与残差，不把未闭合精度升级为已验证。

重开 Attention 驻留需与该访问流匹配的非 duration cache evidence、可验证的源码访问序列或独立机制微基准；重开 Marlin 定价需有单位、可复测、与 production workload 独立的 pipeline/issue/dequantization 机制参数，或等价架构与源码依据。当前离线约束不允许补采 H200。Kernel evaluation 的 observed split plan 与 Serving 的 `analytical_even_share_split_budget_v1` 分开报告；后者的非 duration agreement 仅验证已有 campaign 域，不构成通用 scheduler 等价证明。

V4 EP-off、BF16/MXFP4、clamped SILU 的第一阶段有用数学/role-use payload，按 vLLM 0.30 `MarlinExperts.apply → fused_marlin_moe → _fused_marlin_moe` 保留 W13→activation→W2（含 BF16 router-weight 乘法）→top-k reduce 的串行顺序，各阶段分别取理想计算/带宽最大项后求和。不能把四个 producer/consumer 阶段合并成一次 aggregate max；也不能把 launch 的 padded rows 当作有用数学 FLOPs。BF16 全局中间值读写按源码用途分别计，E2M1 packed weights 与每32值1B E8M0 scales 按已声明 histogram 的 active experts 绑定，histogram 不证明真实 checkpoint 路由。排序/alignment metadata、padding、FP32 GEMM scratch、解量化/编译服务仍单列未检查；该账本不是完整 MoE、实际 DRAM、时延界或正常容量验收，不改变 §6 的原生残差与零拟合裁决。其他 INT4/EP 分支保留既有定价，clamped native 继续拒绝。

Marlin alignment 的源码前序按独立声明的 valid INT32 top-k IDs、EP-off、无 token mask/LoRA 计逻辑 payload。M tile 由 host 的 `(8,16,32,48,64)` 与 `M*topk/E/block<0.9` 推导，sorted-ID capacity 为 `R+E*(block-1)`，`R<E` 时再取与 `R*block` 的最小值；容量不是实际写入的有效 routed 行数。所有 sorted-ID 槽位先写 sentinel，排序再覆盖 R 个有效位置，两次写分别计入；expert-ID capacity 全部写出。256 experts 的 Flash 即使只有一个 token，也走 general count/fill→sort 两个源码调用，不能因小 M 选中只允许 E≤64 的单调用分支。general cumsum 的 INT32 `[E+1]` 写出、唯一 prefix-boundary 逻辑读取，以及 sort 对每个有效 route 的 global atomic read/modify/write 与 sorted-ID 写入分开保留。整数、atomic 服务、shared-memory scan、GEMM 内的 metadata consumer CTA/split 重读服务仍未定价；逻辑字节除带宽不构成 atomic 时延界，也不证明实际 launch/provider 或 allocator 容量。

Marlin W13/W2 的 `use_fp32_reduce=True,use_atomic_add=False` 在每次 custom-op 内以 FP32 `new_empty` 请求 scratch：`min(N*sorted_capacity,SM*4*block_m*max_thread_n)`，`block_m==8` 再乘2，`max_thread_n=256` 绑定源码常量。SM 数仅取声明 HWSpec，不冒充运行时设备观测；两个请求的和不代表峰值 live capacity 或实际分配。`C_tmp` 只在 `slice_count>1` 时进入 global reduction，非 first slice 读取并做 FP32 加法，非 last slice 写回。BF16 程序的 fragment 行掩码与 m8 奇偶 fragment 掩码分别保留，按 valid rows 和候选 thread_n 给出参数化的 FP32 值数，不能以 useful BF16 行数替代 scratch 的 vector 域。未绑定实际 thread 配置、slice census/first/last 时，全调用 scratch bytes 与附加加法数为未知；不得把请求空间当流量、假设零读写或依据误差选 split/速率。该范围只补第一阶段 dtype/临时物化源码语义，不改变已有有用数学/字节基线的数值，也不解开 native 或正常容量验收。

无 LoRA 的 modular Marlin caller 使用默认 `workspace=None`；`_fused_marlin_moe` 在 alignment 完成后、W13 前调用 `marlin_make_workspace_new(device,4)`，默认 `existing=None`，源码以 `torch.zeros` 初始化 `SM*4` 个 INT32 锁槽位。第一阶段必须独立计这一次 `SM*16` B 写入，并保留 W13/W2 共用同一 workspace 的源码传递；不能因 GEMM kernel 会释放锁而省掉首次零初始化，也不能每个 GEMM 重复计一次。此已知写入与 FP32 `new_empty` scratch 区分：后者不自动产生初始化写。后续 lock acquire/release 的 source 服务与实际 allocator 仍未绑定，不从初始化写的价格推断 native launch 或锁延迟。

Marlin 的 W13/W2 分别消费同一份 INT32 sorted IDs、expert IDs 和 post-pad scalar。设声明 histogram 的有效 M blocks 为 `P=sum(ceil(count_e/block_m))`，每次 GEMM 的唯一 metadata 输入 payload 为 `4*(P*(block_m+1)+1)` B：完整有效 block 的 sorted-ID prefix（含 sentinel padding）、每 block 一个 expert ID，以及一个 post-pad scalar；不读取 allocation tail。W13 的输入行索引为 `sorted_route_id//topk`，W2 的 `top_k=1`，两者输出均 scatter 到原始 flat route ID，有效行掩码先排除 sentinel。基础 Roofline 补入这两份唯一输入 payload，且 padded rows 必须与声明 histogram 一致；不能把容量替代有效 prefix，或把该唯一 payload 当作 CTA/split 重读、DRAM、完整整数服务及真实 checkpoint 路由的证明。general atomic sort 的 expert 内顺序不绑定；任何允许的顺序都必须保留上述行映射。

V4 compressor 的 FP32 `torch.mm` 结果沿末维 split 成 KV/score views，两个 view 共享原投影 backing，行 stride 保留完整投影宽度，score view 从半行宽度开始；split 不产生 copy。save-partial 仅在 slot 非负时读取 projection/position/APE，按源码 slot 商/余数和显式 state strides 写 KV 与 `score+APE[position%ratio]`，score 半部偏移取 `state_cache.shape[-1]//2`。C4 的宽度系数为2，C128为1；main/indexer 及各 layer/rank 的持久状态对象不得混用。source 调用关闭 PDL，先 save-partial 再 finish，二者传同一 state tensor。逻辑对象身份与参数化地址表达式不证明实际 page alias、运行时 slots、跨步有效窗口或 allocator；这些缺口保持独立，不能据 views 声明删去真实 producer/consumer 读写，也不能为 split 添加不存在的复制。

save-partial 的 `score+APE` 两个操作数与结果均为FP32，每个有效score列产生一次浮点加法，必须进入FP32计算资源的FLOP需求，不能计入H200定义为INT32吞吐的compute_alu；保留每token一次APE phase整数操作，其他地址/guard服务仍未建模。typed状态side-effect输出与finish输入必须绑定同layer/rank/component的FP32持久对象；C128只有compress段读该状态，store段读fresh compressed_kv临时tensor，不重复声明原状态读取。此处修正工作类别，不改变硬件速率、不按时延调参；source payload/视图检查不能关闭实际页alias、历史值和跨步复用要求。

V4 finish 在有效 metadata 输入域中读取确定的 FP32 状态窗口：C4 是结束位置之前8行，负位置 masked；首个结束边界4只读窗口行4..7，所以选择状态 half1。半区由 `window_row//4` 决定，不能换成绝对位置取模。C128 读取完整128行、half0，以8个 warp 各16行及每warp两个8行状态页覆盖窗口。main/indexer 的 page lookup 与页内 offset 分别来自绝对位置除以/模以 state block size（C4为4，C128为8）；未知 page IDs 和 strides 作为参数，不能默认为线性页。声明步骤的 `[old,new)` 写入与每个完成窗口的 prior/current 子区间必须无遗漏地拼成整个有效窗口，且与 lowering 的完成边界、状态维度和有效行数一致。此 source 控制流证明不覆盖实际 state/output 共享页覆盖顺序、allocator 或真实 slot 映射，不能据此关闭完整缓存验收。

CP1、TOKEN_TO_KV_SLOT 路径的 slot constructor 在 allocation block size 为 kernel block size 整数倍时，普通/分块两条源码路径均化简为 `table_index=position//kernel_block_size`、`slot=int64(table[request,table_index])*kernel_block_size+position%kernel_block_size`；page IDs 仍为输入，不默认线性页。CompressorMetadataBuilder 直接转交 common slot tensor，且对同一 common block table 原地 `clamp_(min=0)`。只有有效读写所访问的页号在 clamp 前非负时，该转交保持 slot/page 一致；负的被访问页号是无条件等价的反例。未用 table padding 与 CUDA graph tail slots 的 -1 分开，不能把 table clamp 或 tail fill 的未知 extent 当成零开销。源码 allocator 为所有 cache groups 使用同一份 zeroed int8 backing，通过 byte offset/strides、dtype reinterpret 与 compressor squeeze views 绑定；同一 backing 可覆盖同一 bytes，活跃 physical page 的归属/复用与布局决定是否干扰。source view/slot 证明不选择真实 layout/pages，不验收真实 allocator、容量、元数据准备工作量或 runtime。

BlockPool 的可写条件是非 null、`ref_cnt==1` 且没有 cached hash；正引用计数或非负 slot 均不足以证明可写。新分配从 free queue 取引用计数为0的块并置为1，touch 增加共享引用，free 只有降至0才把非 null 块放回队列；prefix cache 身份还包含4B cache group ID。unpin 可将仍可读的内容以0引用放回队列，复用 callback 先于新 owner 接收，不构成 GPU reader fence 的证明。SlidingWindowManager 仅按旧的 committed prefix 释放完整 skipped blocks；对窗口W、状态页B、旧前缀C及非负额外保留量，最多可释放到 `floor(max(0,C-W+1)/B)*B`。该上界之后至C的 prior rows 保证保留，下一完成窗口所需的旧行均在此范围内；allocated-range cap 只会少释放。该界不等于实际 eviction、驻留容量或时延预测，实际 live page/alias、COW、异步 fence 和 metadata extent 仍单列。

V4 显式 modular MoE 源码场景的共享专家合并来自 `MoERunner.forward` 的 `result = shared_output + fused_output`：借用两路 producer tensor、只读输入、创建独立结果且没有额外源码 tensor temporary，然后交 combined-output TP all-reduce。mega-MoE 的 `final_hidden_states += shared_output` 是不同源码分支，不能移植其原地 alias 语义。该单操作 source tensor-object 契约须同时检查实际图的 producer/consumer 及 lowering 依赖和两读一写，不能只接受一个 verified 标志；模板复用须重写嵌套 producer/consumer ID。检查不接受 producer 内部 alias、实际 allocator 地址、库 scratch 或 GPU/provider/runtime，也不改变 Roofline 算术、字节或时延。

V4 target-off/all-greedy BF16 source场景的最终行选择是高级索引，恒产生独立hidden tensor，即使索引恰好覆盖全部行。词表slice共享gathered BF16 backing；普通BF16→FP32 cast产生独立logits，argmax返回独立INT64结果，view(-1)与随后的同dtype long()不分配；最终INT64→INT32 cast产生新tensor，返回的[requests,1] unsqueeze view共享该INT32 backing。TP实例化须把input producer与consumer绑定到同rank，all-gather保持group节点，最后输出不添加虚构consumer。上述source复制/视图关系不等于库内部scratch、实际allocator、GPU/provider或精细比较/索引指令定价验收。共享激活的forward_cuda和forward_native有不同的source临时物化程序，CustomOp启用、OOT注册及native编译配置未绑定时不得用一条路径的临时流量替代另一条。

vLLM 0.30 NVIDIA logits末维TP all-gather的CUDA/PyNccl与base communicator共享rank-major输出后的reshape→movedim→reshape程序。连续BF16 shard[R,N]先汇总为[P*R,N]，view为[P,R,N]，movedim后shape[R,P,N]、stride[N,R*N,1]；R>1、P>1且N>1时末次reshape[R,P*N]必须物化，每rank新增2*R*P*N B读与同量写。R=1、P=1或N=1仅view，不能虚构复制。V4显式greedy源码图按该边界新增独立layout复制，再由词表slice/FP32 cast消费；复制的基础Roofline仅计typed payload，native执行保持unsupported且无launch。不把原all-gather传输重复计入复制、不假定PyTorch kernel/grid、库scratch、物理DRAM、正常容量或runtime已验收。

V4 NVIDIA `_split_qkv_and_norm` 的QR/KV输入共享fused projection backing：末维宽度为`q_lora_rank+head_dim`，两段行stride均为完整宽度，KV从`q_lora_rank`列开始；split不分配。`fused_q_kv_rmsnorm`以两次torch.empty分别产生dense QR和KV输出，两者不复用输入或彼此的存储，FP32归一化及gamma乘法后一次BF16写出，qr_scale为None。主Q和适用长上下文indexer读取QR，融合Qnorm/RoPE/cache-insert直接读取KV；可达性中的QR投影路径不足以证明KV对象绑定，图和lowering须保留这条直接边。辅助输入投影及save-partial的join依赖是源码ready顺序，不作为norm额外数值输入。TargetModelSemantics从原始配置导出并校验正有限rms_norm_eps；不写固定默认值。对象、view与源码分配检查不接受实际GPU allocator、gamma checkpoint值或runtime；不据此修改预测速率、复制次数或Roofline价格。

V4 eager source model 输入须在同一物理行顺序中绑定 embedding IDs、positions、cache/compressor 请求窗口和 logits indices。vLLM 的 decode-prefix reorder 只按 region 排序错位槽，已经正确的槽保持原位，不能替换成全局稳定排序；普通采样 indices 为该 model batch 的 query-start 末行。源码投影以显式 caller ScheduleStep 顺序作为初始 batch 输入，导出请求/token permutation 和输出 request IDs，保留 scheduler invocation/routing histogram 身份，不另加 GPU permutation copy。fresh 请求是 reorder 的 region3，有 context 的长 query 是 region2；有 context 的单 token 请求以 done-prefilling 区分 region0/1，short extend 需要显式 v4_source_prompt_lengths，缺失则拒绝。随后 split_decodes_and_prefills(require_uniform=False,treat_short_extends_as_decodes=True) 取首个 query>1 之前的前缀作为 decode，因此 fresh 单token和short extend也可走decode，同一批内长query之后的单token仍属prefill后缀。Scheduler phase不能代替该分支；它不证明实际 persistent InputBatch 的初始顺序、真实模型输入、provider/runtime、元数据准备成本或数值执行。

### 7. FP8 量化的第一阶段算术/字节基线（source-only）

V4 packed SWA insert 的 BF16/512/64/584 源码程序中，Q `warpSum` 的 FP32 加法属于普通浮点工作，KV `fabsf/fminf/fmaxf/ceilf` 和 BF16/FP8/UINT8 转换作为 typed source actions 保留，不按 INT32 ALU 或逐元素 block-reduction 服务计价。NoPE 的28个活跃lane都执行 `exponent+127`，七个scale写lane才执行UINT8转换；不能把前者漏掉或与后者一并当整数操作。Q RMS mean/epsilon/rsqrt沿用逻辑每head一次的基础口径，额外lane复制、address/control及compiler/native服务单列；FP32 cache trig、BF16输入/输出、完整padding写和584B packed记录的逻辑字节不变。源码依据为vLLM0.30 `processDeepseekV4Slot`、`warpSum`、`warp4MaxAbs`；只修正数学与类型归属，不改硬件速率或声明实际provider/物理cache/正常容量通过。

声明`row_major_group128_fp32`的符号linear若由通用GEMM选择器拆出独立量化producer，仍须转交既有vLLM0.30条件FP32-scale源码契约，补齐原始rows/columns并保持launch/native unsupported；通用helper launch不证明该版本的CUDA/Triton、UE8M0或TMA分支。in-call量化不额外增加producer，独立producer必须作为matmul依赖，不能为了通过比较器将已知实现改名或放宽unbound source边界。第一阶段只在所有源码条件程序的共同数学/字节规则成立时计价，不据此接受实际provider。

vLLM 0.30 `Fp8BlockScaledMMLinearKernel.apply_weights` 的 `apply_input_quant=False` 路径直接别名转交BF16 input_2d；若input_scale为None，源码创建input_2d.new_empty(1)，否则复用原input_scale。FlashInfer subclass的apply_block_scaled_mm仅传A、B、Bs，不读取As。缺省As因此是未初始化的BF16单元素、2B逻辑存储对象，不是SDK内部量化的FP32 activation scales，也不产生source payload读写；不以逻辑对象字节推断实际allocator容量。外部量化模板则由独立producer创建FP8值和FP32 scales供matmul读取。输入view、同dtype输出to及最终view不添加复制。条件对象程序不绑定实际wrapper/provider或packed-scale格式，不替代全operator alias/temporary验收，不改已有算术、字节或价格；prequantized batched einsum不套用普通linear包装程序。

vLLM 0.30 的 `QuantFP8` / `per_token_group_quant_fp8` 保留 CUDA/Triton、FP32 scale 是否 ceil 到 UE8M0、column-major TMA 对齐条件。CUDA 的 `ComputeGroupScale` 先除448，再按条件对 scale 做 abs/max/log2/ceil/exp2；Triton 则乘1/448，取整时没有同一额外 scale 下界，不能宣称有限精度等价或仅凭 shape 选中一个真实 provider。第一阶段按显式定义对 FP32 加减乘除各计1个理想算术 FLOP，log2/exp2 使用现有通用 SFU 组；abs/min/max、ceil、类型转换和 lane 重复保留源码次数，不在缺少对应指令映射时塞入通用 integer ALU 速率。只有逐个源码条件分支都证明同一 payload 的带宽项主导时，才能返回共同的理想 Roofline 结果；否则拒绝给点预测，不选上下界代替。该 source KernelSpec 保持无 launch、native unsupported，不继承旧通用量化 resource-input 的计数。

重开 native 量化计价需要明确的源码条件绑定、编译器/ISA 的类型与管线映射或独立机制证据；不得从 production duration 反推或修改速率。本条不改变完整源码静态验收的分母，也不将未检查的数据流、实际 provider 或 native 工作转为通过。

V4普通FP8 linear的调用方对象须与实际rank-local producer、consumer及lowering form同时核对：BF16 dense输入view不复制，返回BF16 tensor与其最终view分开描述，外部量化的FP8值/FP32 scales存活至matmul返回，未读As保持其条件类型与未初始化状态。调用方核对通过不关闭callee临时物化要求。已知FlashInfer SM90 Python包装每次按runner.get_workspace_size分配未初始化UINT8 workspace，而runner由functools.cache保留；getWorkspaceSize取BF16/BF16、BF16/FP8、FP8/FP8三个runner请求的最大值，max_shape_m_4_align_保留历史最大M，expected_m_只在首次设置。缺少初始/前序runner状态时，不用当前M代替workspace请求，也不把allocation extent当流量；DeepGEMM内部temporary源码或零scratch证明缺失时同样保持未检查。这些callee义务逐投影单列，不因调用方图连接或基础计价通过而豁免，不接受实际provider/runtime或正常容量。

上述首次设置仅指binding调用的getWorkspaceSizeBase；另一个getWorkspaceSize方法会重设expected_m_，不能跨入口推广，而dense gemm不使用expected_m_。条件workspace程序以H=max(此前4行对齐最大M,本次4行对齐M)保留未知历史：A预留H*K字节，scale预留本次32行对齐M*(K/128)*4字节；binding的最大请求还含未使用的BF16/BF16 runner权重/scale预留。实际BF16/FP8 dense路径的FP8 A从offset0开始，FP32 scale从H*K开始，其列stride仍为本次4行对齐M而非H或32行对齐M。仅活跃M行由quantizer写入，预留padding及未用权重区不增加初始化或payload流量。独立核对参数化程序不证明H的实际值、matrix callee全部scratch、异步释放安全或正常容量，不改变价格、launch或速率。

SM90 normal-dense 的 BF16/FP8 SDK 矩阵阶段在启用 `ENABLE_FP8_BLOCK_SCALE` / `COMPILE_HOPPER_TMA_GEMMS` 且禁用 `PLACEHOLDER_KERNELS` 的源码条件下复用五个既有 tensor port：workspace 内的 FP8 A 和 FP32 As、借用的 FP8 B 和 FP32 Bs、调用方 BF16 D。legacy `Fp8Gemm::run`、JIT 的 M>=32 普通入口及 M<32 SwapAB 入口均未另建 source global tensor；TMA descriptors 是 host 值，normal scheduler 的 grouped-layout 指针为空，staging/barrier 位于 shared memory，FP32 accumulator 为 register fragment。这一条件源码程序须绑定各入口、device body 和 normal scheduler，并拒绝额外 pointer port 或显式 allocation；不能把它解释为 compiler spill、物理访存或实际 provider 的零开销证明。源码 host 的 workspace 大小计算可在 CPU 上用明确的合成冷/热初始状态验证；合成状态不绑定实际 rank 的历史，未知 H、外部 DeepGEMM 和完整 callee/容量义务仍保留，不改变现有字节价格或速率。

H 的源码历史可在新增完整状态连接后参数化证明，而不选择冷启动数值：`get_fp8_blockscale_gemm_runner_sm90()` 的无参数 cache 在同一 rank-process 源程序内跨层和 K/N 共享；每次请求更新三个 dtype runner。若独立源码清单与实际图同时证明该次模型调用的所有普通 FP8 linear 行数均为 M，则当前 SDK 调用之后恒有 `H=max(P,align4(M))`，对任意合法入口 P 和此前任意 in-call/external 分支组合均成立，依据 max 的幂等性。模型调用之间必须用同一 rank 的 exit/entry state port 连续连接，只有首步声明未知初始参数；不能逐层清零、按 shape 建 cache key 或把 None 偷换成0。此时可逐实例比较声明的 in-call 源码形态的完整 caller、workspace/矩阵 tensor 程序、顺序和源对象 lifetime；缺 SDK source、未绑定的外部 DeepGEMM 或非一致 M 的内部调用顺序仍为未检查。参数化 source 验收不证明真实 P、实际 process/provider、GPU 异步 allocator 安全、正常容量或物理流量，不能据此给出 workspace 容量点预测；既有 payload 定价与速率不变。

### 8. V4 完整 indexer 第一阶段有用数学/字节基线（source-only）

符号forward的长C4路径可接受显式`declared_causal_topk`适用域：调用方声明每行输出恰有`min(index_topk,floor((position+1)/4))`个不同、非负、因果有效的索引，其余为-1。这个声明是独立动态输入条件，不从有限logits/bias、shape或模型身份推导，也不是checkpoint/provider输出验收。该域内可计算每行有效索引payload和local-to-global helper的block-table读取role字节；实际page IDs、cache复用和物理流量仍未知。未声明的长C4符号路径保持拒绝，原`candidate_upper_bound`的scored mask人口与page-table读取保持symbolic，不能借新域补成已知。此扩展不使selector比较/refinement/scratch、score backing或native服务闭合，不调整速率或已有candidate价格。

通用符号workspace按vLLM0.30源码为每个typed port向上对齐256B，同次请求的ports共享单个UINT8 arena；扩容先释放manager引用，caller保留的views自动决定旧arena是否与新arena重叠。Marlin模块的common port同时provision workspace13和临时fused_out view容量，workspace2位于第二个port；CUDA实际输出由匹配的fresh caller output覆盖该临时view，不能把最终输出误记为arena alias。整个模型复用同一rank的manager。该存储投影必须显式提供每rank入口capacity及声明的lock状态；未知入口不偷换成cold0，静态报告中的cold/warm合成输入也不绑定实际进程历史。当前接入只覆盖明确的unlocked source projection及modular BF16 arena，alignment/条件FP32 scratch、其他workspace用户和真实allocator仍不完整；既有完整activation/normal容量拒绝保持，不能把arena字节或源码live peak当作物理流量或profile transient headroom。

V4 indexer 的 `pairs * ceil(log2(topk))` 候选轮数没有绑定到 vLLM 0.30 的 cooperative/persistent/per-row 或可选 top-k 程序，不能作为已知 ALU 指令量。第一阶段按官方数学参考保留 QK、ReLU、head 权重乘法、head 求和和正 KV scale 乘法；比较与整数选择指令另列，不以复杂度估计代替源码计数。CUDA wrapper 显式关闭 skip_topk_buffer_clear，完整 INT32 capacity 的 -1 写与选择写分别计入，清零、FP32 score 物化和消费按串行阶段求和，不用一次 max 重叠 producer/consumer。

该基线只描述有用数学/逻辑 role-use payload，不证明 FP32 求和顺序、实际 provider 或原生性能。显式 CP1 时，prefill 每个请求的完整压缩上下文只 gather 一次：请求组划分后，后续 query 子块设置 skip_kv_gather 并复用同一 workspace，因此数据拷贝行数为各请求 floor(kv_len/ratio) 之和，与 query 子块预算无关。FP8 values 与每行4B UINT8 scale backing 分别写出；scale 消费是 FP32 view，不添加全局转换拷贝。context_parallel_size 未绑定时不补 CP1 默认、不价格化该 gather；已知非CP1或非128-wide key不接受此源码模板。分组/峰值 workspace、矩形 backing/非活动区域、metadata/page tables、selector backend/值依赖 refinement/scratch 继续作为完整源码验收的未检查项；不得把这份部分账本当作完整 indexer 或正常容量验收。V4 indexer 的 Midlayer 精细预测拒绝，其他 DSA indexer 路径不受本条变更影响。重开完整 selector/native 计价需要源码条件、动态字段和编译/物理机制依据，不得根据 production duration 调轮数、速率或补偿系数。

CP1 prefill 的选择器固定来自 `sampler.cu:topKPerRowPrefill → topKPerRowJob`，没有 split-row template 参数；当该行 `rowLen <= topK` 时，返回自然顺序索引与 -1 padding，完全不读取 logits，NaN 和并列值不改变该分支。基线只对非捷径行保留有用 score 读取，并计入每行两个 INT32 bound 的逻辑读取；全捷径时 score 比较数及全局 refinement scratch 为零。score producer 仍按上游执行，不能因为选择器不消费它就省去 QK。非捷径行的额外扫描次数和 scratch 保持未知，不将单次有用读取冒充完整服务。此规则来自源码控制条件，不根据预测或实测时延挑选分支。

CP1的prefill组若完全没有已完成C4状态，`sparse_attn_indexer.py`先执行整段top-k sentinel clear，再逐query子块`fill_(-1)`；没有score、selector、Q/weights或row-bound payload读取，`new_empty([query_rows,0],fp32)`不分配payload。该明确域允许indexer及C4 attention的topk=0，不能推广到未知CP或普通scored masks；两次sentinel写各自保留，不能重用短上下文kernel伪装这条源码分支。自动图用write-only out及独立controls记录赋值/准备join，不为ready顺序添加假读取。

Sparse prefill的SWA/C4/C128 workspace由同一个显式入口history的rank manager提供。源码PREFILL_CHUNK_SIZE=4限制的是面积，允许一个chunk包含超过4个小请求；按caller context算compressed最大宽度和SWA gather最大宽度，并在面积限额下分组。BF16 gather、INT32 combined indices与lengths是三个typed ports；indices/lengths按max_num_batched_tokens分配，不能用active rows缩小allocation。最后一个小chunk保留manager的大arena；扩容时之前loop tuple的views跨get_simultaneous存活，旧/新arena可重叠。这里只增加source allocation/view/lifetime，既有gather/index traffic不重复计价。未知indexer manager用户/callee scratch等会影响历史与真实峰值，故这个子程序投影不是完整worker峰值或对它的已证明界，normal容量与日志验收仍保留。

V4 C4 indexer的CP1/FP8 prefill分支在读取chunks前向同一manager请求完整constructor容量：DeepseekV4Indexer.max_total_seq_len=get_max_prefill_buffer_size(config)//4，而该helper返回40*runtime_max_model_len；因此两个typed ports为[10*runtime_max_model_len,128] FP8与[10*runtime_max_model_len,4] UINT8，分别256B对齐。它们与main sparse及Marlin共享arena，不按active query/context缩小，也不按layer相加。scale的FP32 view保留UINT8 port的实际offset、stride4与同一backing，不新增转换分配。存在prefill即发出此请求，空压缩组同样预留但不写gather payload；decode-only不新增该prefill请求。wrapper的prefill局部引用跨后续decode存活，返回时释放；manager容量按已声明source历史跨步保留。该源码明确的新port不解决外部DeepGEMM score backing、query chunking、metadata、selector scratch或实际provider；allocation extent不计成全容量流量，原有数学/字节价格不变，完整activation/normal容量拒绝保持。

CP1 indexer 的评分输入 metadata 与选择器读取分开计：非空 prefill context 的每行两个 INT32 起止 bound 在 score 阶段产生 `8*query_rows` B 唯一输入 payload；普通未 padding 的单 token decode 为每行一个 INT32 effective context length。未知 CP、包含空 context 的 prefill 分组或多 token decode 不补此 metadata 默认值，相关字段保持未绑定。两个 score wrapper 都传 `clean_logits=False`，直接将 FP32 `logits` 交给 selector；矩形逻辑 shape 不证明 inactive cell 写入、连续 stride 或 backing allocation 字节。单请求 prefill 的 query 子块共享该请求完整压缩 context，因此全部逻辑输出面积为 `Q*floor(L/ratio)`，与 query 子块预算无关；多请求只给按请求分组的可证明面积界，不将界用作点预测、显存峰值或 traffic。decode 的输出列数由构造期 runtime indexer max_model_len 决定，不能替换为当前 active context 长度。物理 backing、未写区域、实际分块、page/schedule metadata 仍保持完整源码验收缺口。

V4在显式CUDA/H200 source、默认`KernelConfig.sparse_indexer_topk_backend="auto"`、CP1、单token decode且K属于512/1024/2048的域中，`indexer_topk.py:SparseIndexerTopk._resolve_auto`的可达分支只有cooperative或persistent；前者不满足约束时后者必可达。两者的forward都向同一rank manager请求`RADIX_TOPK_WORKSPACE_SIZE=1024*1024`个UINT8元素。允许记录这一共同的1MiB申请而不选择未知stride对应的backend；超过64行时只保留persistent域。该请求可扩容或复用arena，不逐层累加，不变成全容量traffic。源码构造期`runtime_max_model_len//4`和DeepGEMM公开返回契约另外限定FP32 score逻辑shape，不能将逻辑shape当连续backing申请；私有kernel scratch、refinement/比较量、实际provider与正常容量均继续未通过。此项仅补源码storage图，不改变Roofline定价、launch或硬件速率。

## 未闭合事项索引

| # | 事项 | 重开条件 | 详见 |
|---|---|---|---|
| 1 | FIA KFC 清零 32.1 KiB/核 计价 | 新增独立 memory path，并适配 `_attach_cann_standard_decode_phase_ledger` 的逐 phase path 校验 | §1 |
| 2 | MLA regime B 回绕复用未裁决 | 带中间流量扫描的驻留微基准（采集，未授权） | §1 |
| 3 | prefill AIV 被测量违反物理下界 | profile 窗口边界侧证据；修好前不得对 1.20–1.28 残差调参 | §1 |
| 4 | FA/decode 负差额定价伪影 | padded 侧 launch 反映 capture 时刻输入而非运行时重推导（专门后续） | §2 |
| 5 | 910B VECTOR M>700–1200 尾部斜率过陡 | 覆盖真实 H 全档位的 MTE2/MTE3 专用标定微基准 + 共享时序合成规则 | §4 |
| 6 | 910B VECTOR M≤700 主导区间欠计 floor | 验证真实 serving（host 连续发射）下是否暴露 | §4 |
| 7 | C500 MoE 失锁幅度建模 | 三条同时满足：L2 set/slice 几何权威来源、miss 核心地址级仿真授权、代码形态轴物理解释 | §5 |
| 8 | H200 V4 Attention 驻留与 tile/pipeline 精度 | 对应访问流的非 duration cache、源码调度或独立机制证据 | §6 |
| 9 | H200 V4 MXFP4 Marlin 定价残差 | 与 production workload 独立的 pipeline/issue/dequantization 参数或等价静态依据 | §6 |
| 10 | FP8 条件量化 native 服务 | 源码条件、编译器/ISA 类型管线映射或独立机制证据 | §7 |
| 11 | V4 indexer 完整源码/native 服务 | gather/chunk、score backing、metadata、selector backend/值依赖 scratch 的源码与输入绑定 | §8 |

V4 modular Marlin 的 top-k reduce 三种无mask源码形式都先把FP32 accumulator置零，再对每个BF16专家值转换后执行一次FP32加法，最终转回BF16。因此source级加法数为T*H*K，进入compute_fma；T*H*(K-1)仅保留为最少数学加法数，不能用它代替实际源码循环，也不能把这些浮点加法计入INT32 compute_alu。此处不绑定vector/scalar实际dispatch、编译器是否消除首个零加法或native指令次数；不改读写量、硬件速率及其他尚未闭合的scalar/转换服务。

V4共享/Marlin专家clamped SiLU共用同一source算术owner：每值2次FP32乘法、2次加法、1次除法和1次exp。第一阶段以5个普通FP32 FLOP和1个exp数学操作计基础需求，不把未绑定的除法lowering假定成额外1个SFU。三个FP32 fminf/fmaxf与符号取负保留为未定价的typed source动作，不能按INT32 ALU吞吐计价；源码还包含clamp下界的-limit，不能只计SiLU输入的负号。scalar输入取负的表达式类型为输入dtype，packed输入取负为FP32，两条路径另有FP32 limit取负；按每值两次记录，但不选择真实scalar/packed分支、不推断编译器hoist、转换或指令数。该调整只修正基本数学与未映射动作的边界，既有逻辑payload不变，native/完整算子服务继续不验收。

V4 finish的C4 main、C4 indexer和C128 compress/store第一阶段普通FP32加、减、乘、除（包含归约加法）按FLOP进入compute_fma；不能计入INT32 compute_alu，也不能把每个浮点加法当成一次block-reduction依赖服务。FP32 max/min、abs、ceil和取负分别保留typed action计数，不用整数吞吐或未绑定compiler指令给它们定价。原abs与两侧clamp反例保留为逐操作类型/次数检查，不再以INT32需求下界代替。exp/rsqrt数学次数、已有最少norm/quant树和显式C128 merge计数保留；lane复制、BF16/FP8转换、UE8M0位操作及具体归约/除法服务仍未计价。本条只校正基础数学资源类别，不改字节、硬件速率或声明实际provider/native性能通过。

V4 EP-off modular Marlin的W2 router weight路径先把FP32权重转成BF16，再用__hmul2乘BF16中间结果并保留BF16输出；每个routed输出值一次浮点乘法不能按INT32 compute_alu计价，也不能不经依据改成FP32乘法。生产typed program和W2阶段显式保留其BF16 FLOP计数、两lane打包及未定价服务；当前已有tensor/byte基础价格仍是partial，不包含该乘法的转换、舍入和实际pipe服务。普通BF16有限输入即可使先转BF16与保留FP32权重两种程序产生不同输出，不能用实数代数等价替代。此处不新增或调整HWSpec速率，不把未定价动作删去，不据此通过完整MoE、源码数据流或native/provider验收。

V4 mHC post的源码先以FP32 post coefficient乘layer output来初始化每个FP32残差输出，再执行hc_mult个FP32乘加。单独post的普通浮点工作因此为T*H*C*(1+2*C)，不能把初始化的一次乘法放入INT32 compute_alu。fused post/prenorm的每个projection tile都重做该初始化，次数为(n_out/tile_n)*T*H*C；residual_out及sqrsum仅由tile0输出的既有规则保持。该修正只迁移这一次可直接核对的FP32乘法到compute_fma，不改变其他数学次数、字节、launch或HWSpec速率；其余mHC/Sinkhorn/sigmoid及runtime分支不据此宣布闭合。

V4 mHC pre/norm与head的split求和、norm乘除及epsilon、gate scale/base、Sinkhorn行列求和/除法/epsilon均是FP32，进入普通compute_fma FLOP组；初始Sinkhorn行max的T*C*(C-1)次比较保留为未定价FP32 typed action，不能按INT32计价。sigmoid、exp、rsqrt分别保留源码函数调用数，既有SFU数学组不当成真实compiler指令次数或已绑定的函数pipe服务；不展开或假定具体sigmoid lowering。此处保留既有数学primitive总数、最少逻辑树口径、读写和分支条件，不改硬件速率；完整非线性/归约compiler服务及native/provider仍未验收。

V4 pure-text top6路由按源码区分类型：CUDA hash的32-lane五轮FP32加法和输出FP32除法进入普通FLOP，不能当block-reduction服务或SFU；Triton的六次selected-weight gather求和及八槽权重求和同为FP32加法，normalization divide也不是SFU。FP32 max、INT32 min和各类compare/select分别保留dtype/次数，不套用INT32峰值或block-reduction服务。hash NaN清零仅在isnan(weight)时赋值；source site extent与实际动态执行数分开，后者在缺logit值时为未知且有[0,6*T]范围，不假定为0或全执行。普通路由保留threshold softplus、NaN score处理及原始无bias权重，不断言与hash路径有限精度等价。本条仅修正数学类别和typed source program，不改变字节、硬件速率或source分支；predicate/selection、条件赋值、compiler和runtime服务仍未定价。

普通Triton top6的expert_offsets<NUM_EXPERTS在每个row的完整BLOCK_N范围产生INT32比较，与末尾八槽output_mask比较分别计数；BLOCK_N为源码next_power_of_2(num_experts)，不能遗漏初始mask或仅计六个最终有效ID。原始kernel可用无payload元数据寄存器执行六轮固定循环，独立记录FP32算术/SFU、typed比较/select/minmax和有效mask下的source读写；值依赖布尔分支拒绝，PDL分支仅在恰为两个无参数依赖同步void调用时从基础FLOPs/bytes程序排除。比较计数不是INT32吞吐需求，不据此改变Roofline价格或推断compiler/provider行为。

V4 mHC的非融合post→prenorm→norm可由显式、首次cached dispatch之前固定的VLLM_USE_DEEP_GEMM=False源码配置确定optional prenorm为TileLang单partial；True或未声明时，has_deep_gemm/platform支持仍未知，不得由H200或split形状反推。RuntimeInputs.v4_source_use_deep_gemm及block memo必须保留该声明，独立reference分别接收同一外部输入而不读candidate补全。首层broadcast显式use_tilelang_fallback=False，仍强制调用DeepGEMM，不能因env关闭而替换为TileLang/普通GEMM或接受实际provider。KernelSpec.source_prenorm_backend只声明prenorm源码程序，TileLang限定非broadcast且n_splits=1，排除不适用的条件SM90 TF32程序；native launch/support保持未绑定。此路径仅改变源码要求决定的FP32 partial写出/读入及split归约数学，不改HWSpec速率、全局模型身份或宣称完整源码/容量/Serving验收；FP8等其他调用方的独立条件程序与缺口保持其范围。

V4 modular BF16/MXFP4 group32的FP32 scratch可按条件源码配置给出完整访问census：实际group_blocks=2小于候选thread_k_blocks，不触发stripe分组舍入；先分离DP全K tiles，再对two-tile Stream-K尾部按CTA K区间求每tile partial数R。每tile按既有fragment有效行谓词取得V个FP32值，合计读取4*V*(R-1) B、写出同量、额外V*(R-1)次FP32加法，R=1时无scratch访问。声明histogram决定expert顺序的M-block有效行数，不等于真实checkpoint路由。自动thread候选与1..allow_count构成包含所有可选结果的superset；cudaFuncGetAttributes.numRegs、可用编译kernel与shared-memory筛选未绑定时仅发布各配置条件值及上下界，不能选某配置或把界写入点预测。allocation extent、初始化、访问总量和峰值分别保留；不把requested bytes当流量，不调整Roofline价格/HWSpec速率，也不据此接受完整MoE、native/provider或正常容量。

V4 compressor的寻址block size来自CompressorMetadataBuilder读取的resolved KVCacheSpec，经metadata同时传给save-partial与finish，不能把CompressorStateCache构造时的C4默认4行当成已绑定metadata值。page-size统一可按整数倍增加block size；C4源程序在参数B=state_metadata.block_size下查表position//B、取行position%B。C4账本保留constructor_default=4，但selected state_page_rows为空，并在lowered FP32输入保留同一参数程序；这不证明正常配置实际取B=8等值。既有C4保留区间上界floor(S/4)*4在B为4的正倍数时仍统一成立，因为floor(S/B)*B为不超过S的4倍数；因此不需要把旧的保留上界当实际释放位置，也不省略任何读取。当前CUDA主C128 split wrapper明确拒绝B!=8，独立保留这一适用域。正常allocator、实际页号/alias、layout及容量仍未验收；不改变数学需求、逻辑payload、launch、速率或点价格。

V4 packed uint8压缩KV的finish输出slot来自k_cache_metadata.slot_mapping，不能使用compressor内部state slot替代。main C4/C128与FP8 indexer按输出cache的B=shape[1]、字节stride S分别计算page=slot//B、row=slot%B，数据位于page*S+row*D，scale位于page*S+B*D+row*Q；不是逐token AoS。main的D=576/Q=8由448B E4M3、128B BF16 RoPE、7B UE8M0和显式零padding组成；indexer的D=128/Q=4保留FP32 scale。源码gather读取同一分离区域，page乘stride必须保留INT64语义。本条只声明有效completed boundary及适当对齐下的条件地址程序，不绑定实际metadata/page/grouping/reuse，不覆盖full BF16/per-tensor FP8 cache分支或FlashMLA内部；不改逻辑payload、数学工作量、价格或硬件速率。

V4 BF16/MXFP4 Marlin的源码M补齐可由显式expert histogram与block_size_m确定：P=block_size_m*sum(ceil(count_e/block_size_m))。原matmul循环的BF16 MMA对每个逻辑M/N/K tile执行2*M*N*K FLOP；M8通过交换操作数只发一条m16n8k16，M16发两条，因此M8按8行而非16行计算。原DP/SK控制将各M/N tile的K区间覆盖一次，不因split增加tensor FLOP。第一阶段W13/W2基础Roofline据此使用4*P*H*I、2*P*I*H；仍分别导出useful及padding FLOP，activation、有效输出写出、router乘法和top-k reduce只用live rows，不随P膨胀。该修正依据源码数学计数，不是Marlin误差/速率调参；仅对声明histogram成立，不将它称作真实checkpoint分配、实际编译provider或完整MoE验收。FP32 scratch条件访问、权重解量化、integer/atomic和compiler服务及硬件速率残差保持原边界。

V4 BF16/MXFP4 group32的Marlin寄存器解量化须保留源码两次BF16乘法：E2M1位放置后乘2^126偏置，再乘E8M0解码的BF16 scale；不能合并成一次FP32乘法或将其按INT32吞吐计价。每个matrix M-block对N*K个权重值分别执行两次乘法；源码scale解码site产生N*K/4个BF16值，区别于N*K/32个唯一逻辑group scale。权重/scale fragment是thread-local FragB/FragS，不创建全局BF16解量化tensor；host scale重排属于load-time工作，不每次Forward重复。E8M0字节e在该helper被放为BF16位模式e<<7，其中0映为zero、255映为positive infinity，不能无条件宣称与通用E8M0的2^-127/NaN相同；实际checkpoint字节未知时保留完整条件程序与数值等价未证明。BF16 intrinsic舍入/FTZ、compiler spill、repack到逻辑矩阵的完整映射及native issue服务保持独立缺口，不因此改速率或添加假全局读写。

V4 Marlin的checkpoint packed E2M1权重存储为UINT8，load-time经每expert的view(int32).T.contiguous与gptq_marlin_repack后，Forward持有INT32 packed tensor[E,K/16,2*N]，不能继续把prepared参数dtype写为UINT8；每word8个4-bit值，字节总量仍为E*N*K/2。N64/K16 repack tile的nibble坐标和scale的grouped64/single32→group4重排可按源码作独立整数映射验证；K32等边界必须选源码真实permutation，不能统一套64。各expert view属于同一fresh最终参数tensor且区间不交叠，逐expert的临时结果在load-time复制，不能每次Forward重复计复制或将参数source extent当正常峰值容量。当前声明的无额外维度padding程序要求N%64=0、K%32=0，已知不满足时拒绝而非截断尺寸；reference也拒绝非整除TP。该prep映射不证明随后MMA fragment reader、实际checkpoint值、CUDA编译/执行或provider；此前BF16数值/FTZ与完整MoE缺口保留，不改FLOPs、payload字节、速率或价格。

V4 BF16/E2M1/E8M0 Marlin的global→shared→寄存器权重坐标可按原始整数地址与同一PTX opcode的CUTLASS BF16 A/B layout traits核对；M>=16时权重为B operand，M8交换操作数并按b0[0],b1[0],b0[1],b1[1]成为A operand。CUTLASS在此只提供独立ISA坐标参考，不能据本地SDK存在宣称目标provider已绑定。每个有效matrix M-block会按K/N tiles重新发起N*K/2 B packed-weight及N*K/32 B scale读取；参数的active-expert唯一字节不能替代这些source-global访问。现有第一阶段memory投影因此按声明histogram的M-block数累计weight/scale访问，另保留distinct parameter bytes；实际L2/DRAM缓存结果、transaction放大、async ordering、activation ldsm、数值MMA、compiler spill与实际config仍未绑定，不将此投影称为物理流量真值或时延界。初始min(R,S-1)与后续max(R-S+1,0)次有效prefetch合计覆盖R个分配K tile，不把pred=false的fence计成读取。本条修复源码访存漏计，不调整任何速率、不按production duration选择缓存命中或回放系数；缺独立operand-layout源码时该字段保留unchecked，不能按候选自述通过。

V4 EP-off同步modular Marlin的调用对象必须从同rank的真实BF16 hidden、FP32 top-k weights及INT32 IDs端口连接到独立BF16输出，再由shared-add消费；不能仅以verified标记代替producer/consumer与终止依赖检查。source对象程序分别保留BF16 arena及view、INT32 alignment缓冲、两个GEMM的FP32 scratch请求、九个零长度占位对象和load-time参数映射；传入GEMM的C与返回C同对象，finalizer不增加复制。prepare的activation scale可为空或借用，W4A16不消费它，不能据此宣称总为空。源生命周期及同步调用顺序不证明实际allocator lane/capacity、GPU fence、provider或scratch访问量；未选定slice census时scratch流量继续参数化。本项不改变FLOPs/字节基线、速率或launch，不闭合其余MoE数学/流量/lowering及完整模型验收。

V4 save_partial的源码对象对齐要求同rank BF16 A/B→fresh FP32 C投影、C的两个行跨步split视图、FP32 APE参数、INT64 positions/slots，以及bind_kv_cache原始squeeze(1)继承的storage offset和stride。状态缓存按传入slot的page/row在原有FP32 storage中写入两半，不返回新tensor；不能把squeeze解释成复制、把state cache视为私有分配，或用verified标志代替实际输入端口、输出对象和lowered依赖。静态对象检查可使用显式的符号backing/stride程序，不为其选择实际页；该项与persistent_cache_and_cross_step_state分开，后者仍要求补足真实场景的metadata/page别名及复用关系。原源码在真实CPU tensor上的split/squeeze验证只证明view身份、stride、offset与修改传播，不接受GPU allocator、fence、provider、正常容量或时延；不改变FLOPs、字节、速率或launch。

V4 packed-FP8 finish的输入是同rank既有FP32 partial-state，gamma为BF16、rotary cache为FP32；输出原地写入k_cache_prefix指向的既有UINT8缓存，使用其独立destination slot mapping，不能复用state slot地址或声明缓存为私有新分配。C4主路径与FP8 indexer不创建全局压缩值tensor；CUDA C128创建fresh FP32[num_actual,512]，由compress写入并把同一对象交给store，仅完成边界对应的token行被写出/读入，其他torch.empty行保持未初始化且不得读取。临时区allocation extent不能缩成completed_states，亦不等于实际访问字节。独立对齐同时检查源参数/缓存owner、临时对象identity、lowered阶段与依赖、store program和consumer端口；short-context top-k的依赖只表示源码顺序，不代表它读取cache payload；主压缩缓存到indexer scoring的依赖同样只表示prepare并行任务join后的顺序，indexer只读取自己的index缓存，不能虚增主缓存payload读取。此为符号页/stride上的源码对象和副作用程序，不接受实际allocator复用、GPU fences、provider或正常容量；不更改既有数学量、字节、硬件速率或launch。

V4跨步partial-state来源按rank、layer、component和request_id记录逻辑prefix及entry/exit端口，不能按batch行号继承请求，也不能每步把未知初始内容重置为零。该来源记录只验证模型请求连续性，不证明物理内容保存：原始cache manager在partial hit时可用CoW私有block替换tail，并把(source,destination)入队；source hit-ref保留，destination多持有一个ref。普通extend不能被提升为所有分支都保持旧page ID；drain pending列表只转交描述，不表示worker copy或GPU同步完成。实际table转移、prefix-cache hit观测和copy完成信息未提供时保持None，persistent-cache要求继续unchecked；源码CPU metadata执行只能验证table/refcount/queue规则，不得宣称真实Scheduler或runtime验收。

V4 decode的FlashMLA调用还接收共享INT32 SWA indices/lengths；C4/C128另接收共享INT32 compressed indices/lengths。第一阶段沿既有all-valid causal active-prefix投影，每call的逻辑输入读取为4*(SWA pairs+query tokens) B，压缩层再加4*(compressed pairs+query tokens) B，均不乘padded query heads。C4 local-to-global materialization账本已包含压缩indices/lengths的consumer读取，合并两份账本时必须减去该重叠；C128即使当前compressed prefix为空，仍保留每query的INT32长度输入。producer的metadata构建/初始化不在每层consumer重复计价，FlashMLA实际masked/vector加载、重复读取、L2/DRAM、page IDs与内部临时区继续未绑定；scored路径沿既有候选容量投影，不把它升级为实际selected IDs或完整attention验收。本项仅补齐已知source输入payload，不修改FLOPs、split参数、硬件速率或launch。

vLLM 0.30 worker的CoW复制按(device,data_ptr)去重同view；若底层storage字节数等于num_blocks乘scheduler block stride，则以UINT8[num_blocks,row_bytes]复制整个底层storage，并再按(device,storage.data_ptr)去重，包含同一pool的其他layer region及padding。否则只复制cache.unflatten后的逻辑view，不能把storage级去重无条件应用到fallback。manager block含多个kernel block时，复制范围须按该整数倍展开。`blocks[dst] = blocks[src]`的INT64高级索引RHS创建fresh临时tensor，再scatter回原cache；每个实际复制unit的N对描述、B字节逻辑行对应source读、temporary写、temporary读、destination写各N*B，不能只计一次读写或当无临时。该条件源码程序不提供N、实际cache geometry或H2D/allocator峰值；事件缺失仍为None，不能定为零次复制。worker的Python提交顺序与scheduler retained-endpoint fence规则分别核对，不从queue drain或提交先后宣称GPU复制已完成；CPU原始copy core验证也不构成真实Scheduler、GPU/runtime或完整persistent-cache验收。

V4 BF16 LM head的第一阶段对象契约沿LogitsProcessor、UnquantizedEmbeddingMethod与真实ATen schema核对：同rank的row-select BF16[rows,H]和持久BF16[N,H]参数均只读，返回fresh BF16[rows,N]，TP gather必须连接所有rank输出且lowered依赖完整。默认auto dispatch的F.linear分支及CUDA batch-invariant的weight.t() view→matmul_persistent分支同时保留，不据shape选择实际backend或断言两者数值等价；weight transpose不创建Forward参数副本。调用方没有额外全局temporary，不代表原生library scratch为零，也不确定输出stride、allocator或provider。这些第二阶段边界不得冒充已通过，也不阻断第一阶段已知逻辑对象；collective内部alias/materialization独立保留。本项不更改FLOPs、字节、launch、硬件速率或Roofline价格。

V4 BF16/E2M1/E8M0 Marlin的条件源码账本另须区分逻辑A输入与source-global A读取：每个N tile重复读取live rows的K个BF16值，DP/Stream-K只划分K区间，不因此重复整K，合计2*live_rows*K*(N/TN) B。thread_block_reduce没有live-row guard；其G=threads/(TN/2)个K组对补齐M行归约，每个M/N slice有(G-1)*block_M*TN个FP32 addition site，slice数包含Stream-K额外partial。不能按live rows缩小此算术，也不能把global FP32 scratch流量替代shared归约计数。现有条件config census逐分支导出这些可执行计数，独立reference用源码归约循环重算；尚未选定的config继续保持未选定，不把某一分支或区间端点用于点预测，不把source-global读取称为物理DRAM。现有基础Roofline价格及完整MoE math/bytes未验收状态保持；这一步没有修改速率、duration、launch或生产点预测公式。

BF16最终RMSNorm的CUDA batch-invariant分支不能与IR reference或vllm_c合并为同一个源码rsqrt程序。原始_rms_norm_kernel以1024列分块，读取输入两遍，使用FP32 sqrt后再计算1/rms；FP32 gamma乘之前不作BF16 roundtrip。静态账本分别记录各块tl.sum的数学加法、从零累计chunk sum的加法、masked lane的source elementwise sites及逐row的BF16输入/weight/output payload。wrapper的input/weight contiguous复制各自保留条件，不能把未知contiguity定为零复制；kernel没有源码全局FP32中间tensor不代表compiler spill为零。原有unique-operand理想RMS baseline与这些条件source counts分别保留，不选实际backend、不宣称数值等价，不改速率、launch或点价格。本项补齐已知源码分支，不能作为完整norm alias、融合矩阵或模型验收通过。

V4 H200最终RMSNorm的第一阶段对象检查连接同rank hc_head的dense BF16[rows,H]、持久BF16[H] gamma、fresh BF16输出与greedy row-select。保留三条源码请求程序：IR eager reference的七个typed临时tensor、CUDA vllm_c的显式empty输出、batch-invariant的可选contiguous复制及reshape views；input和gamma均不被修改，最终输出不alias它们。CUDA默认IR优先级只给vllm_c/native，Oink显式要求SM100，当前H200 SM90排除；外部IR插件/override不在该source baseline内。原始IR函数体在真实CPU Torch上执行，观察到七个临时请求加fresh输出，末尾同dtype to返回同一输出对象；这不证明compiled allocator、peak capacity、compiler spill、实际provider或数值等价。检查同时约束producer/consumer端口、BF16/epsilon和lowered真实依赖，不以声明布尔值代替图绑定；价格和硬件参数不变。

虚拟forward声明的vllm_c CUDA RMSNorm支持rank2/3/4的leading-axis stride，不因整体non-contiguous就复制：kernels/vllm_c.py的flatten/reshape分支只在IS_ROCM时生效，libtorch_stable/layernorm_kernels.cu仅在input.stride(-1)!=1时materialize。Qwen3 packed QKV分出的Q/K head views末维连续，应直接读原backing，不添加每层两次复制。fresh dense output在Python wrapper进入C++之前已分配；确有非unit末维stride时，该输出与内部contiguous临时存储重叠。此处修正同一source profile的图与生命周期，不将CUDA规则推广到ROCm、IR native、batch-invariant或实际compiled/provider，也不按时间或日志误差选择复制条件。

V4 EP-off的专家ID轴必须在logical graph与lowered Marlin调用之间逐项保留。route总和、padding、FLOPs和参数足迹相同，不证明专家ID对应关系相同；反转histogram或owned_experts仍可改变实际使用的专家权重。生产target lowering现在先检查global/local counts一致、expert_rank=0及全局ID顺序的完整ownership，布尔值不能充当INT32 count/ID/rank，然后才构造KernelSpec。独立对象比较也约束这些输入到lowered字段的绑定；这是声明参数的传递不变量，不是从候选推导真实router输出，不接受seeded assumption为checkpoint观测。原始baseline函数的CPU反例已证明旧lowering会静默替换矛盾local counts，旧比较器会接受三类专家轴失配；新增拒绝检查保留。MoE完整math/bytes、真实路由和整个模型验收仍分别未闭合；不改变速率、launch或合法输入的价格。

### Qwen3 MoE softmax/top-k数学Roofline（active）

Qwen3 MoE 的 `moe_topk` 源程序只有显式 `mathematical_program=bf16_softmax_topk_flops_bytes_v1` 才能计算第一阶段普通 FLOP/逻辑公开参数 payload Roofline；旧无此声明的 source contract 和全部 native/midlayer 拒绝保持。范围为现有 BF16、2/128-expert、无 bias/padding 的0.30 `topk_softmax/topkGating`。先 stable softmax，再选择并按显式布尔参数归一化；数学 sum 用最少 n-1 次加法，减/乘/除各计一普通 FLOP、exp 计通用 SFU数学操作，不把 predicate/index 工作换算为 FMA 或 INT32 ALU。R行/E专家/K选择的普通算术为3RE+RK（不归一化）或3RE+2RK（归一化），exp为RE；保留源码最终乘法，即使调用方传入 factor1。sanitized selected sum 为零时分母取1，两条分支的这份算术数相同，不猜实际 logits 或分支频率。

0.11普通A800 Qwen3-MoE TP1/PP1/EP-off eager沿同一模型/算子owner，显式使用`vllm_0_11_fp32_softmax_topk`及`fp32_softmax_topk_flops_bytes_v1`，不冒用0.30 BF16输入程序。0.11的fused_topk先独立把BF16 gate输出物化为FP32，归一化时再由Python执行新weights/row-sum张量的sum/div；FP32 logits由outer fused_topk保留到归一化结束，raw weights、row sum与新weights的公共峰值重叠。源码C++直接存selected score，不含0.30最终scale乘法；归一化也无positive-sum兜底。复用相同stable softmax数学owner，普通算术分别为3RE和3RE+2RK-R，exp均RE；路由公开操作payload分别4RE+12RK和4RE+24RK+8R，前置cast另计6RE。此处字节为各公开操作的unique typed角色，归一化中raw weights的两次读取分别保留，不声明实际launch数量、DRAM、指令服务或数值等价。当前仅放行2/128专家且显式seeded routing或既有canonical profile routing/已知IDs；profile按原layer-mean或per-MoE-layer resolver保留histogram与provenance，不将其推断为当前checkpoint路由。其他TP/EP/async/graph和provider边界不扩；非power-of-two或E>256源码有额外softmax workspace，不按零处理。来源为本地0.11 Python fused_moe.py的vllm_topk_softmax/fused_topk与v0.11.0 csrc/moe/topk_softmax_kernels.cu，后者可从已有upstream Git对象读取，无需下载或GPU。

公开逻辑 payload 为 BF16 logits 2RE、FP32 weights 4RK、INT32 IDs及token_expert_indices各4RK B。2和128均为不超过256的2次幂，源码 `needs_workspace` 为False，`new_empty({0})`无正字节storage；不能推广到需workspace的expert数。源码 weight 的两次store/一次reload expression另列，unique public角色字节不代表这些site的issue、实际DRAM或编译器forwarding。max/top-k比较、NaN/Inf sanitation、index生成、tie-breaking、lane重复及padding仍单列未绑定，不因“零算术FLOP”而宣称零服务；native、数值等价、完整worker容量仍不通过。该程序仅补第一阶段已知解析工作，既有expert histogram保留独立声明身份，不由虚拟logits生成，不调整任何硬件参数或以duration选择公式。

Qwen3 MoE 的普通 DP/SP/CP/PP1、TP=EP=2 分支沿 `FusedMoEParallelConfig.make` 把 routed expert 的内部 TP 置1、EP置2；每rank保留完整输入行、全局128专家logits和top-k结果，线性分配64个完整中间维度专家，不能把每个专家中间维度再除TP2。全局histogram和每rank ownership/local counts分别保留，不能重命名或互换专家ID来伪造等价。`MoERunner.do_naive_dispatch_combine` 仅由DP>1或sequence parallel触发，此域没有新增all-to-all；本地专家贡献通过一次完整TP组all-reduce合并。虚拟group用原调用产生的同rank hidden/route输出和统一MoE invocation连接所有producer，允许独立通信顺序barrier，但data输入必须覆盖所有EP rank。旧无虚拟ports的手工图仍沿原router边验证，不能放宽其拒绝边界。

普通EP source和A800/C500/910B/H200共享同一模型body与typed routing/target primitives；Ascend继续显式使用其out-of-place norm与contiguous operand程序，C500/Ascend使用非融合mask/gather/fill embedding。不同硬件recipe各自只替换一次原routing stage（910B为`moe_gating_topk`，其余为`moe_topk`），保留原expert/init/finalize账本。非H200仅放行显式2/128-expert数学程序，旧无数学声明的CUDA source contract不扩域；不会据此绑定真实插件routing实现、NaN处理、数值等价或GPU provider。真实0.30 CPU allocator/Scheduler可保留该EP flag并投影普通GQA cache，完整config哈希、TP/EP、owner、页字节与placement仍逐项校验；EP只影响FFN，不能为attention虚构额外cache。原生模型/正常容量不随CPU调度通过；正式Serving case保持原expert-frequency profile，不改成seeded假设以过关。

### 符号forward的BF16 BMM数学Roofline（active）

BF16输入、BF16未量化权重、BF16输出的batched GEMM允许在无launch且execution_support=unsupported时给出理想数学Roofline：B个独立矩阵乘的算量为2BMNK，A/B/C各计一次逻辑payload，依次为2BMK、2BKN、2BMN字节。统一虚拟Tensor的transpose只交换shape/stride，out=保留调用方已有backing；canonical strided字段在该无launch抽象中只表示逻辑footprint，不证明转置或token/head交错view满足实际packed ABI，也不增加未出现的packing copy。原K3 NoPE源程序复用相同数学计数，其来源/适用域不变。FP8及其scale/量化、mixed-output、library workspace、compiler spill、真实流量/缓存/launch和native/midlayer不由此获得支持；未知内部对象不能用已表示storage峰值代替完整容量。该路径不引入硬件速率、拟合项或实际GPU执行。

### 符号forward的DSA decode数学Roofline（active）

GLM的V32符号decode可显式声明`dsa_weighted_relu_topk_v1`：限定CP1、未padding的单token decode、128-wide FP8 indexer Q/K及FP32 score/weights，源码已在fused_norm_rope清空top-k时使用`skip_topk_buffer_clear=True`，不得再次计clear或重复cache insertion。该程序复用V4的weighted-ReLU数学定义，按score物化→selector有用payload分阶段求Roofline；ReLU/选择比较、值依赖refinement/scratch和额外扫描保持未计价，不以`ceil(log2(K))`代替。它的非空声明必须对应无launch且execution_support=unsupported的KernelSpec，不能给已有analytical launch加上声明来取得native支持；旧DSA contract不隐式选择新程序。

BF16 576/512 latent sparse decode的无launch数学路径把同一512维latent同时用作NoPE key和value，RoPE为另外64维；同一cache输入不拆成独立K/V分配。每query所选行的cache role payload在head间共享，跨query按role-use累计，不推断实际DRAM/L2或选中ID并集。对P个query/key/head对、R个query/head行，QK/PV为2P(576+512) FLOPs，显式数学softmax为4P-R标量FLOPs、P次exp、R次reciprocal，max比较单列；不据此计价真实在线softmax、padding heads或split服务。`declared_causal_topk`仅为显式输出cardinality适用域，不由finite logits、配置身份或seeded routing推导，不接受为真实checkpoint输出证据。

V32注意力caller的BF16 query拼接写入既有workspace的actual-head前缀，保留64-head token stride；未写padding不补初始化或额外copy。公开latent输出保留64-head backing，actual-head view供V absorption，后者用out=写入forward早先分配的输出。logical-to-physical top-k转换、backend被丢弃的第三返回值、score backing/非活动区域及内部scratch仍未绑定，组件storage峰值不是完整worker峰值或其证明界。全模型源码、正常容量、Serving和native继续分别验收；该数学扩展不改硬件速率、不拟合时延。

GLM BF16 modular/EP-off符号FFN沿vLLM0.30 V32 decoder的reduce_results=False及apply_routed_scale_to_output=False分支：router权重包含routed_scaling_factor，Runner收到1.0，不另加输出scale或FFN内部TP all-reduce；shared输出与routed输出相加后，由下一input/final norm归约。独立FP32路由节点替换canonical expert recipe中的一次旧moe_topk，不能重复或删除路由；原完整路由服务未定价的拒绝保持。BF16 expert lowering要求global/local histogram与完整expert-ID顺序一致、输出未归约、权重BF16/group0、EP1及声明的routing stage，不按置换等价类重命名源码专家轴，矛盾输入fail-closed。该分支只复用已有专家定价，不补未知workspace，不依据性能选路由或backend。明确源码fallback的all-reduce/norm helper持有自己的reduced局部引用，返回后由caller引用决定生存期；不得因适配时内联helper而跨attention额外保留临时量，也不得把声明fallback当实际FlashInfer/provider选择证据。

### 符号forward的TP目标logits对象（active）

GLM全decode普通目标链沿GPUModelRunner的INT32 query_start_loc[1:]-1输入索引保留materialized row-select，不能把identity选择当alias或无依据改写成INT64。caller的BF16源行、INT32 index和fresh BF16输出使用已有无launch逻辑role-byte契约；ATen内部索引转换/temporary未绑定，不接受为native或完整显存证明。H200 CUDA继承Platform.use_all_gather=True，BF16 ParallelLMHead各rank产生完整词表分片；base/PyNccl源码的all-gather suffix把raw [TP,rows,shard]转为[rows,TP,shard]后reshape。对rows>1且shard>1，这一步fresh物化[rows,padded_vocab]并读写完整BF16 payload；rows=1只view，词表切片也不复制。独立helper保留raw backing至重排结束、返回后释放，不延长到Sampler的FP32 cast。lowering只接受完整、等宽、等行数的所有rank head producer，匹配collective per-rank字节；重排必须是同rank、同rows/TP及正确padding范围的已声明source节点，不能借一般vector节点绕过原all-gather依赖检查。此处不推断实际通信provider、allocator、library scratch、编译launch或argmax指令服务，不改参数/速率；原路由拒绝和全模型时间缺失保持。

### 符号forward的GQA数学Roofline（active）

统一虚拟张量图可显式声明FLASH_ATTN的mathematical_program=gqa_softmax_v1，仅提供理想数学FLOPs/唯一参数payload的Roofline，不调用真实tensor库、不选择实际FA2/FA3/provider或物理split。令P为所有有效query/key/head对，R为query/head行数、D为head维，QK和PV合计4*D*P FLOPs；显式scale→subtract-max→exp→sum→reciprocal→normalize数学程序计4*P-R个标量FLOPs、P个exp和R个reciprocal。max的P-R次比较单列未计精细指令，不把它们塞入INT32 ALU。KV按GQA实际head数共享，输入Q/K/V及输出各计一次唯一payload，cache update另作producer；不虚构score矩阵、在线softmax重标定、LSE/split scratch或编译spill。该抽象不证明与实际浮点在线算法数值等价，也不是实际DRAM或硬件时延界。KernelSpec必须无launch、execution_support=unsupported，拒绝physical ksteps/效率修正、graph-padding及未知program；midlayer/native继续拒绝。原有FA2 combine超出已支持源码范围的拒绝保留，不通过更改split、调整速率或按误差选择算法规避。未知library workspace、allocator及正常容量继续独立验收。

共用 dense BF16 Qwen3/Llama 的 `cuda_eager_inplace` 源码场景适用于已有 A800 与 H200 目标：模型 forward、vLLM0.30 的 CUDA TP embedding、INT32 target row-select、TP logits 重排与 BF16→FP32 cast 使用相同逻辑程序，不以 H200 身份代替适用域判定。上述新增 A800 入口仍使用各自 HWSpec 及硬件 launch owner，显式数学 attention 和 target payload 仍无 launch/unsupported；不继承 H200 的 SM90 专用 V4/GLM/Kimi 分支，不把 CUDA 源码场景冒充 C500/Ascend provider。报告保留 captured 旧版本与 effective0.30/eager 的区别及未验证 reserve，不能据此宣称真实 runtime、正常容量或硬件精度验收通过。不更改速率或既有算术公式。

C500 的 BF16 dense TP1 也可执行该显式 CUDA-alike/eager 源码场景：本地 MetaX0.24 插件的 `MacaPlatformBase.is_cuda_alike()` 为 True，`MacaRMSNorm.forward_oot` 与 `MacaRotaryEmbedding.forward_oot` 转发到 CUDA 实现；对应 base 的 `vllm_c` RMSNorm 使用 unit-last-stride、fresh norm output 和 inplace residual-add 程序，与共用元数据算子一致。单卡 embedding 的 INT32→INT64、materialized target row-select 及 FP32 logits cast 保留各自 payload。模型 forward 复用独立核对的 Qwen/Llama主体；0.24 Llama 直接调用 activation 与0.30通过未量化 helper 返回 activation 的语法差别不自动标 AST 匹配。每个 case 保留有效0.30源码场景与捕获旧版本之间的区别；这不证明实际 C500 IR/provider 选择、normal 容量或 runtime0.30。C500 TP embedding/logits 的后续扩展见下一条；MoE 以及 Ascend 的 out-of-place norm/rotary 路径仍未迁移，不借此放行；不改 C500 速率、launch 或既有残差裁决。

### CUDA-alike 非融合 TP embedding 与 MetaXLink（active）

C500 的普通 TP 只扩展已有 dense BF16/EP-off 源码域，不能借用 NVIDIA 融合 embedding：vLLM0.30 的 `use_fused_embedding` 要求 `current_platform.is_cuda()`，Maca 的 OOT 平台走 `get_masked_input_and_mask→masked_input.long→embedding→masked_fill_→all_reduce`。共同 TP primitive 显式保留 mask helper 的 fresh INT64 IDs 和 bool mask、独立 BF16 gather、同一 BF16 backing 的原地输出 mask，再由完整 TP collective 消费。bool Tensor 与 Python int 的乘法把 offset 提升为 INT64，随后 INT32 input 减该 offset 的结果也为 INT64；因此 `.long()` 不另分配。mask helper 按已编译 pointwise 函数的公开 I/O 计13T B，输出 mask 按 BF16 `[T,H]` 一读一写加每行1B bool 参数计 `4TH+T` B；gather 与 collective 分开计价。typed 计数仅为源码表达式站点，不是 compiler 指令；整数/选择服务与内部暂存仍未绑定。该无launch/unsupported程序仅提供现有数学 Roofline 的 FLOP/逻辑参数 payload 基线，native 继续拒绝，不据此接受完整物理流量、正常容量或实际 provider。

现有 `analytical_roofline_v1` 的 ring/all-gather 网络字节公式可用于显式 MetaXLink；使用 C500 HWSpec 已保留的独立 TransferBench P2P 容量，schema 旧字段 `nvlink_unidir_bw_GB_s` 在该硬件的原说明中明确代表 MetaXLink，数值和来源均不修改。KernelSpec 必须绑定 `metax_c500/metaxlink`，不能用 NVLink 身份绕过，也不调用 NVIDIA IPC custom-all-reduce 服务。该结果仍是给定 P2P capacity 的逻辑通信 Roofline，拓扑瓶颈容量不是实际 selected pair、collective provider 或同步时延的观测；不新增模型/shape 时延参数、不拟合 duration，原有 ProtoBridge 分支保留。C500 TP2 后续验收继续使用其真实链路身份。

vLLM0.11 A800 CUDA的`VocabParallelEmbedding.forward_cuda`仍调用`forward_native`，TP输入走相同的mask/gather/fill程序。以独立`vllm_0_11_symbolic_masked_embedding`来源和`mask_program=vllm_0_11`标记，复用既有无launch逻辑payload与typed表达式计数；不增加公式、速率或native绑定。来源身份和mask程序不一致必须拒绝。对应head gather/reorder及greedy保留显式0.11身份，既有0.27.1模型来源与0.30 Scheduler驱动的区分保持。

### 虚拟图的 OOT native RoPE 源码分派（active）

普通 dense adapter 的 RoPE 来源须复用现有 `vllm_rope_program` 的精确类名注册规则，不能从 Norm 使用 CUDA-alike 分派推出所有 RoPE 类都融合。C500 未注册 `Llama3RotaryEmbedding`，因此其 Llama3 路径沿 `RotaryEmbedding.forward_native/forward_static` 和 `ApplyRotaryEmb.forward_static` 执行 cache row-select、Q/K 各四次乘法及减/加、rotate concat 和 full-width empty-suffix concat；原 Q/K 保持只读，返回 fresh BF16 结果。虚拟算子用同一 trace 记录真实输入/输出 view 和 Python helper frame，使 query_rot 活到 K 分支结束；不调用旧建图器，也不手填释放表。`aten::cat(Tensor[] tensors, int dim=0) -> Tensor` 的非 alias 返回契约使两个 outer concat 仍各有 fresh output，即使 suffix 没有 payload；第一阶段按其逻辑读写保留，不能只因旧 kernel census 未列出调用而当 alias。普通 BF16/neox/full-head eager 场景共17个逻辑程序节点，不把它称作17次已观测 GPU launch；ATen 内部 copy/provider、allocator 与编译优化仍未绑定，不改硬件速率或按 duration 选机制。已有融合 CUDA 场景结果不能自动继承为该 C500 native 分支验收。

### Ascend dense 虚拟 forward 的存储程序（active）

显式 `vllm_ascend_eager` 场景复用 vLLM0.30 普通 BF16 Qwen/Llama forward 主体并组合本地 Ascend 插件源码语义，不能标成 CUDA inplace 场景或已安装的0.30 NPU runtime。`AscendRMSNorm.forward_oot` 的 `npu_rms_norm` 返回 fresh y/FP32 rstd；`npu_add_rms_norm` 返回 fresh y/rstd/residual，输入与原 residual 保持只读。CANN 的非连续输入物化由共同虚拟 Gather 表示；每逻辑 row 的4B rstd来自现有 CANN source launch 契约，在同一算子输出中自动记录其作用域，不重复增加一次独立写。常规无 bias BF16 域不推断量化 anti-method、定制 norm bias 或权重预取。

Ascend default full-head RoPE 的 Triton wrapper 与 torch_npu wrapper 都先 contiguous 再原地更新 Q/K；这一共同 storage 事实不选择实际 provider。未注册的 Llama3 类使用 native composite，共用已有 `aclnn_materialized_operands` 来源：每次乘法为非连续 Q/K half 和 cos/sin half 产生独立 contiguous 输入，单行连续 cos/sin 不复制。外层 empty-suffix cat 保留 fresh 返回 storage 和逻辑读写；旧物理 kernel census 的 elision 不能作为 alias 证据。依赖和释放事件由同一虚拟 forward 推导；实际编译优化、库 workspace 与物理 launch 保持独立未验收。TP1 的 embedding widening、materialized logits row-select、FP32 cast 和 greedy 只使用现有数学 FLOP/逻辑 payload Roofline，不借此放行 native。普通 TP 扩展见下文，EP/PP/MTP 等模式仍须分别迁移；不改 HWSpec 参数、不拟合 production duration。

Ascend 普通 TP/DP1/SP1 的 `AscendVocabParallelEmbedding._forward_origin` 保留 mask→INT64 gather→inplace fill→all-reduce，限定 `embedding_tp`/`lmhead_tp` 关闭且无 added vocabulary。其 `_get_masked_input_and_mask` 的无新增词表分支只有两次 INT32 比较、一次 bool and/not、两次 INT64 乘及一次减；与普通 vLLM 的四比较程序用 `mask_program` 分开，公开13T B 的 ID/mask I/O 和 BF16输出 mask payload不变。`AscendLogitsProcessor._get_logits_normal` 沿继承的 `use_all_gather=True` 进入普通全词表 gather/reorder，与共享 primitive 使用同一 producer 和每 rank 字节校验；不推断独立 embedding/lmhead TP 组、SP reduce-scatter 或实际通信 backend。

HCCS 的数学通信沿已有 ring/all-gather 网络字节公式，使用 910B HWSpec 已保留的独立两 rank、64MiB HCCL broadcast 带宽作为显式理想链路容量；保留 raw 来源和旧 schema 字段，不修改数值。KernelSpec 必须绑定 `ascend_910b/hccs`，不能以 NVLink 或 MetaXLink 身份借速率，也不能调用 NVIDIA custom-all-reduce service。该容量不证明多 rank 拓扑、实际 HCCL collective、同步时延或原生 provider；这些要求不因数学 Roofline 全调用可计价而通过。

### 符号forward的block-FP8数学Linear（active）

GEMM可显式声明mathematical_program=block_fp8_fp32_scales_v1，仅限b=1、BF16输出、K整128组及无launch/unsupported；FP8 A/B、BF16 C的逻辑payload分别为MK、NK、2MN B，FP32 activation/weight scale分别为4M(K/128)、4ceil(N/128)(K/128) B，矩阵算量2MNK按HWSpec既有FP8峰值计算。该账本与原block-FP8模板共享payload实现，但不继承vLLM0.24 FlashInfer/DeepGEMM launch或实际vLLM0.30 provider。前置BF16量化复用独立无launch的0.30条件source程序，不在matmul重复计量化。fp8_utils的create_fp8_scale_parameter按ceil生成scale，validate_fp8_block_shape允许未TP切分merged投影的最后一块为tail；准确GLM融合A的2048+576输出因此保留2624行及21个scale输出块，不补齐C或改为BF16权重。FP32输出、bias/observation声明与实际launch不得借此契约通过。量化/scale的物理layout padding、完整非算术服务、库scratch与完整worker峰值仍未绑定；只验收逻辑数学与payload，不接受为实际流量、原生计价或正常容量，且不调整硬件速率或拟合时延。

### 符号forward的block-FP8专家数学程序（active）

`routed_experts`可显式声明`mathematical_program=block_fp8_swiglu_v1`，限定BF16 compute、FP8权重、完整128组、EP-off、无swiglu clamp、无physical tile/graph padding及无launch/unsupported。以显式histogram的R=T*topk个有效assignment执行W13→普通SwiGLU→W2→加权和；逐active expert复用block-FP8 Linear的逻辑operand/FP32 scale账本，矩阵有用算量合计6RHI，不把未选择的tile padding加入。W13输入T*H与activation后的R*I量化分别复用已有条件程序，仅在其全部source分支的理想Roofline相同时给点值。普通CUDA SiLU/mul与clamped路径共用唯一算术owner：保留runtime alpha/beta的乘/加、分母加法/除法及最终乘法，合计5RI普通FP32 FLOP与RI个exp；输入取负单列typed action，不计FLOP，也不把除法虚构成额外reciprocal SFU；加权输出计RH乘法和T(topk-1)H加法，router中已有的routed_scaling_factor不在FFN再次施加。BF16 W13/activation/W2逻辑payload及FP32 routing weight/INT32 ID role保留，不推断实际metadata排序、scale/cast/address指令服务、Triton/DeepGEMM provider、DRAM/cache、workspace或compiled数值等价。该程序是有用数学基线，不是物理流量或实际时延界；完整route服务及正常容量仍独立未闭合，不能用它把旧DeepSeek-V3 FP8 launch census套到GLM6144-wide几何，也不能把未表示临时区写成0。

### GLM显式force-MQA的prefill/mixed数学域（active）

vLLM0.30的attention_config.sparse_mla_force_mqa=True会使_use_sparse_mha最终返回False，允许CP1、BF16 latent cache、EP/MTP-off的prefill/mixed沿forward_mqa，而不推断默认dense/masked-MHA/provider选择。统一虚拟图只在显式source声明下使用该分支；普通decode与原dsa_weighted_relu_topk_v1契约保持原域。新增dsa_weighted_relu_force_mqa_v1与bf16_latent_force_mqa_v1均无launch/unsupported，按每request的query/最终KV长度计算每query因果候选及min(K,causal_length)的已声明有效cardinality。数学score/selector与latent注意力的FLOP、逻辑role payload复用既有owner，不虚构log2(K)选择指令、actual IDs或runtime。模型输入复用vLLM0.30的四region decode-prefix重排；request末行索引来自累计query长度减1，alias与所有T行Q absorption保持。prefill indexer共享gather容量沿get_max_prefill_buffer_size绑定为40*runtime_max_model_len行，FP8[capacity,128]和UINT8[capacity,4]共用同rank管理器；scale的FP32[capacity]视图保持相同storage/offset，不另分配。Flash的C4路径复用该布局实现并保留容量除4。仅prefill请求gather端口，decode-only不请求；多层复用及旧arena被constructor的q_concat引用保留时的重叠由通用生命周期记录，不手填释放表。独立reference从upstream helper/constructor AST求期望，仅检查已出现caller的两个端口和scale alias；完整chunk分组、score backing及内部workspace仍未绑定，不能据部分数学价或部分storage通过完整source、容量或端到端验收；已闭合的unique gather/selector逻辑payload见下一段。完整路由定价拒绝仍保留，不据时间或误差选择force标志或调整速率。

GLM显式force-MQA的CP1 prefill/mixed复用V4已有的gather/selector源码逻辑owner：每prefill请求的完整未压缩KV只gather一次，FP8 values及4B UINT8 scale backing的读取加写出合计264*sum(prefill_kv_len) B；query子块通过skip_kv_gather复用，不需假定logits分块预算。decode组不添加该gather。prefill的score与selector分别读取每行两个INT32 bound；decode score仍读取每行一个INT32 effective length。sampler.cu的unsplit rowLen<=topK分支不读logits，仍写出全capacity的自然索引和-1 padding，前置fused clear和score producer均不删除；多请求shortcut统计用长度闭式计算，不遍历所代表的score元素。mixed中的decode服务保持原有用payload基线，不把prefill捷径推广到未知decode backend。该规则只改变已知数学/逻辑payload账本，没有实际分块、矩形backing、非捷径扫描/selection scratch、物理流量、launch或时延界结论；完整路由拒绝及native/normal gates保持。DSA显式数学program拒绝decode query_len>1，旧未声明该program的MTP/verify契约不扩大或收紧。

### 符号forward的BF16 dense MLA decode数学Roofline（active）

FLASH_ATTN/mla_decode可显式声明mathematical_program=bf16_mla_decode_softmax_v1，限定BF16、未padding、causal、QK576/V512/latent512、uniform MLA cache、CUDA graph关闭且无实际provider/launch；原单token结果与字节不变。多query只接纳显式vLLM0.27.1 FlashMLA BF16源码分支、VLLM_BATCH_INVARIANT=False和uniform 1<=Q<=128，保留其UNIFORM/threshold128条件，不套用causal Triton的threshold1或推断installed provider。源码reshape_query_for_spec_decode将[T,H,576]无copy地view为[B,Q,H,576]，backend的BF16 O[B,Q,H,512]返回后view为[T,H,512]，FP32 LSE保留[B,H,Q]实际布局。MLACommon.forward_impl的Q absorption及V-up覆盖全部T行，各用独立BF16 BMM，cache append和query concat另作producer；数学attention不重复这些工作。P为heads*sum(Q*K-Q*(Q-1)/2)个因果有效query/key/head对、R为query/head行数，QK/PV合计2*(576+512)*P FLOP；理想softmax复用GQA的4P-R个普通算术、P个exp与R个reciprocal，max比较单列。上游tests/kernels/attention/test_flashmla.py的scaled_dot_product_attention/ref_mla明确以logsumexp为FP32 LSE；复用理想softmax的max和sum再计R个log与R个加法，普通算术合计4P，LSE写出4R B。latent cache的前512通道同时作为V，每request只计一次576-wide cache payload，不按Q重复或虚构第二份V读取；query、latent输出及LSE各计一次逻辑payload。该抽象不证明FlashMLA内部exp/log实现、浮点顺序、padding/split、page table/scheduler metadata、library scratch/物理stride、GPU时延界或正常容量；源empty(0)依赖只作cache写后控制顺序与零payload，不冒充实际allocator对象绑定。KernelSpec及caller对未知program、非uniform或Q>128、未知/启用的batch-invariant分支、graph padding、FP8 cache/quant、provider或实际launch声明继续fail-closed，midlayer/native拒绝，不改硬件速率或按误差选机制。

### Fresh BF16 MLA prefill的数学Roofline（active）

GLM的fresh短上下文沿`_use_dense_mha_prefill`的源码条件：HiSparse关闭、force-MQA关闭且prefill最大长度不超过index_topk；当前统一forward从fresh prefill域推导该分支，允许单token decode前缀，不推断长上下文masked backend或chunked-context实现。V32前置全T Q absorption/fused_q仍执行，随后fresh Q concat进入MLACommon.forward_impl；KV展开按checkpoint模块量化策略，K的NoPE/RoPE拷贝与backend fresh输出至既有output的copy保持独立。该body与K2.6共用`MLAFreshPrefillBackend`，BF16 attention数学program扩展到GLM的QK256/V256/latent512，算术及payload公式不变。indexer的`dsa_weighted_relu_dense_mha_v1`只接收fresh且prefill最大KV<=topk的prefill或mixed，decode上下文不受该prefill阈值限制，复用原score/gather/selector逻辑owner，不复制公式或继承force-MQA分派。mixed沿MLACommon.forward_impl先执行prefill，再对decode前缀从fresh full-head query拆分并第二次Q absorption，不能复用V32外层先前对全T计算的吸收结果来删运算。该吸收写token-major fresh backing，FlashMLA sparse concat写既有padded-head workspace，decode V-up写同一输出backing的prefix，prefill copy写suffix；typed视图及存储依赖保留两个写入，不构造额外整批输出。没有实际provider、内部scratch或launch结论；600个完整路由服务拒绝及normal/Serving门槛保持独立，不能因分支迁移宣称完整模型时延通过。

`FLASH_ATTN/mla_prefill`可显式声明`mathematical_program=bf16_mla_prefill_softmax_v1`，限定BF16 QK192/V128或QK256/V256、latent512的fresh causal序列、显式可选FP32 LSE、无graph padding、provider或launch。KV expansion与cache append已由caller独立计价，此处读取prepared Q/K/V并写O各一次，不再读取latent cache。有效query/key/head对为P，query/head行为R，矩阵算量复用共有softmax数学owner的2P(QK_dim+V_dim)，普通算术4P-R、exp P次、reciprocal R次；max比较继续单列。逻辑payload为各参数唯一用途字节，不代表DRAM、在线算法、指令issue、库scratch或GPU时延界。K3 fresh wrapper必须显式声明backend supports_out条件，直接写caller分配的输出；未知能力不能省略可能的物化copy。该程序只解除因缺离散launch而错误阻断的数学Roofline，midlayer/native仍拒绝；不更改硬件速率、不使用时延拟合，chunk/route/SiTU/专家及normal容量缺口不随此通过。

### K2.6 BF16 gate与INT4专家输出缩放（active）

vLLM0.30 DeepseekV2MoE在Kimi配置缺moe_router_dtype时，GateLinear保持out_dtype=None，BF16 A/B产生BF16 C；独立kimi_moe_route保留BF16 logits、FP32 correction bias/weights与INT32 IDs，不借GLM的FP32 gate替代。NVIDIA的apply_routed_scale_to_output=True使Router/专家侧factor为1，Runner对BF16 routed输出原地乘准确checkpoint factor2.827，再shared_output+fused_output并做一次late TP all-reduce。声明modular/no-overlap、shared分离、TP8/PP/SP/CP/DP1、EP/MTP-off；共享MLP partial与routed partial不提前各自归约，普通dense MLP和attention O投影仍在其RowParallel Linear内部归约。源SharedExperts.output读取后清空模块缓存，符号Runner用局部返回保留同一生命周期，不为每层建立手写释放表。INT4/group32专家复用既有Marlin物理/数学定价owner及完整专家ID/histogram轴；这不证明0.30实际backend/packed ABI、metadata/scratch、物理traffic或normal容量。独立route节点不删除、不重复旧moe_topk；完整非线性/packed选择服务继续拒绝，不能因整61层/target图齐全就给整段时延或actual runtime通过。新增ScalarMultiply复用已有elementwise unary语义计一次标量乘法及原地读写，不改HWSpec速率、不拟合duration。

### V4启动无metadata与普通attention的函数作用域（active）

vLLM0.30 attention.py.forward保留输入投影结果、qr/kv和o_padded，_prepare_and_attn局部持有padded Q及indexer输出；helper返回后这些局部storage在同步源码程序中释放，不能因将helper内联到符号forward而错误延长至inverse-RoPE/输出投影。恢复真实helper作用域，由通用引用记录器导出释放，普通请求的数学、流量与kernel价格不变；异步retirement与编译图仍不由该源码生命周期推断。

V4启动profile是独立于带metadata请求的source程序，仍复用同一完整模型forward、decoder、MHC/MoE、TP logits及共享worker/sampler scope。显式无metadata时compressor在split后返回，不发save/finish；Q不作norm/RoPE/cache insert，仅在local heads不足时F.pad；C4仍投影/量化indexer Q和weights，不走metadata驱动的短上下文捷径。sparse_attn_indexer.py的profile分支向同一workspace manager请求FP8[10*runtime_max_model_len,128]、UINT8[10*runtime_max_model_len,4]和1MiB radix空间，并在函数内持有UINT8[indexer_max_logits_mb*2^20] dummy；大小是显式env条件，不能从时延或启动峰值选择。nvidia/flashmla.py无metadata分支按N=ceil(runtime_max_model_len/ratio)（SWA-only为0）、M=N+window+max_batched，预留BF16[4,M,512]和INT32[max_batched,round_up(topk+window,128)]/INT32[max_batched]后zero输出，跳过gather/topk/attention。纯文本max_image_tokens=0，TP4 Flash/TP8 Pro、EP/SP/CP/MTP-off；source workspace初始容量逐rank明确声明，不猜实际历史或provider。

同一rank的profile总storage峰值减profile结束保留量得到已表示源对象的transient headroom；新建共享arena的驻留增长单列，不能从每个时刻都排除它。每个物理rank独立扫描，不相加rank峰值、不相加模型与sampler组件峰值，也不按layer累加复用workspace。storage-only节点无完整数学计价contract，production lowering继续拒绝；该程序不升级normal容量策略、不证明完整worker内存或与启动日志10%对齐。尚未表示的库/通信内部workspace、compiler/allocator及其他runtime状态继续列为缺口，未知值不补零；V4不得直接套用旧Qwen按persistent标签排除对象的activation预算，候选策略须满足下述同次图的驻留划分契约。

### 启动source驻留与临时存储的统一容量划分（active）

V4完整启动图已给出结束后仍被workspace manager保留的arena，旧raw activation peak中也包含该arena；将该值与覆盖结束状态的framework预留相加会重复扣除同一storage。重开候选策略导出的依据是这一可枚举的同次source allocation/lifetime证据，不是启动log差额或时延误差。通用扫描以profile结束时仍存活的allocation ID集合R划分驻留与非驻留；F_source=sum(size(R))，A_source=max_t sum(size(live(t) minus R))。F_source+A_source覆盖这段声明源码程序的每个时刻，旧arena被view保留时仍进入非驻留峰值。该预算界与用于日志比较的headroom=P_source-F_source不同，二者分别导出；不能把预算界冒充已观测点值或未知warmup/steady/provider程序的界。

既有schema2的V4候选策略必须明确base framework覆盖before_profile还是after_profile。前者先验证所有已表示初始对象和既有模型buffer下界，再增加同次图的已表示驻留增长一次；后者验证结束驻留下界并保持framework字节。activation统一来自非驻留同时峰值，其他reserve及其来源继承并整体降为unverified_analytical_assumption。缺少epoch、原reserve低于对应source状态、未知source/并行/量化分支均拒绝，不猜base是否已包含arena；Qwen维持原after_profile契约并共用同一划分。模型first-layer FP32 MHC broadcast buffer补入图的持久对象；V4 Pro目标层无SWA-only层，MTP-off不得因draft尾部ratio1额外计SWA RoPE。构造图与model buffer账本逐rank核对这一差异，避免通过放宽测试掩盖它。

启动候选预算可交既有解析KV规划和真实CPU Scheduler验证功能，同时将同次profile最终workspace容量显式传给请求source历史。完整non-KV账本、其他启动/运行时分支的峰值覆盖、日志10%、actual provider及normal容量门槛均不因此通过。缺项仍需源码/执行配置或外部事实，不能将继承的零预留标成observed zero或完整预算。

BF16纯文本、CP1/MTP-off的vLLM0.30启动图共用VLLM030WorkerStorage记录GPUModelRunner的16个无条件设备缓冲与InputBatch的6个常驻sampling数组。inputs_embeds即使在纯文本profile中未使用也已由constructor预分配，必须计入驻留；InputBatch的INT32 top_k不是dummy sampler的临时INT64 top_k，不能合并。CpuGpuBuffer的cpu/pinned/NumPy侧不重复计入HBM，其gpu是同shape/dtype的独立一次分配。所有对象在profile开始前创建、结束后仍由worker owner保留，故增加驻留floor但不改变同次profile headroom或非驻留峰值；before-profile基线使用同一组descriptor求已知worker字节，不在调用方手算总峰值。独立AST参考只求这22项的上游shape/dtype及单设备分配绑定，不执行Torch源码；缺项、额外项或分配转发变化不缩分母。block tables、可选M-RoPE/投机/多模态状态、其他backend metadata与库/runtime仍在未闭合范围，22项通过不能标记完整worker或normal容量。

### V4普通路由包装的未使用临时存储（active）

vLLM普通modular纯文本hash top6包装先申请FP32权重、INT32 ID和INT32[rows,6]的token_expert_indices；CUDA hash fast path不把第三块传给选定kernel，包装也不返回它。虚拟Forward用同一个call_outputs的workspace生命周期记录24*rows B请求，使它在route返回事件之前分配、之后释放，并与输出同时存活；不添加kernel节点、初始化或读写流量。普通Triton路径在该包装分配前已返回，只有两个输出，不重复增加第三块。原始Python包装与CUDA launcher fast-path检查共同约束这一分支；逻辑分配请求和同步eager scope不证明实际allocator、compiled spill、正常容量、路由数值或provider。

### K2.6 fresh prefill和decode-first mixed的共同虚拟forward（active）

K2.6沿vLLM0.30 `layers/attention/mla_attention.py` 的 `forward_impl/forward_mha` 补全fresh prefill与decode-first/fresh mixed；继续限定准确INT4/BF16 checkpoint、H200 TP8、CP/SP/PP/DP1、EP/MTP-off、BF16 cache/输出及显式plain greedy。输入投影与cache append覆盖全T一次，query absorption、latent attention与V-up仅覆盖decode前缀Td；fresh后缀做BF16 KV expansion和causal attention，不虚构Q absorption或V-up。local heads为8，源码优化concat的128-head条件不满足，因此同一新K backing的NoPE/广播RoPE两个切片分别copy；广播只有零stride view，不分配expanded payload。BF16输出未启用fused quant，源码向prefill backend传out=None并把fresh返回值copy到外层output，此copy和两份重叠storage均保留，不能借K3的supports_out分支省掉。完整输出只做一次O投影/all-reduce。fresh数学attention复用既有BF16 192/128/latent512 softmax程序，不新增公式、速率或实际launch；source参数payload和同步生命周期不证明库scratch、DRAM、provider或normal容量。context/chunked prefill仍拒绝；原Kimi route未计价拒绝不变，完整层图及其他调用可计价不代表整模型时延或Serving通过。

### 虚拟forward的PD复用与CPU0.30控制面（active）

既有Qwen3 TP1/TP2及Qwen3MoE TP=EP=2的PD范围复用每pool的同一模型forward、typed storage和lowering，不另写PD模型图。原多pool geometry/角色/同硬件及KV head分片契约进入deployment owner，模型builder和Serving共用；KV handoff仍由既有request-scoped transfer及rank/head重叠规则计价，不为匹配旧图变更字节或硬件速率。CPU0.30 bridge的role从caller保留到独立SDK进程，只允许colocated/prefill/decode；其他模型的非colocated bridge继续拒绝。prefill pool发首token，decode pool沿既有metadata PD adapter向真实Request注入已传入的prompt computed count和首token，由真实KV manager分配/account，再仅调度decode。该状态是显式仿真connector及synthetic length-only feedback，native.kv_transfer_config仍为None，不能声称实际vLLM PD connector、KV内容、模型采样或GPU/provider验收。容量策略按每pool原绑定校验，继承的未知预留不因PD请求跑通升级为normal容量。PP/AF/Host KV和其他未迁移范围不由此通过。

既有H200 GLM TP8/EP-off及准确K2.6 TP=EP8的PD逐pool模型程序复用原colocated forward、typed storage与lowering，调用方仍须提供完整constructor、routing、workspace/indexer和逐step token事实；缺失或任一pool不属于同一已实现source域时拒绝，不能退回手拼图。Kimi EP8的完整专家ownership与同rank归约校验不变，只允许PD单次图属于一个独立Prefill或Decode pool。源码模型和算子不复制，PD传输仍消费原owner的prompt KV字节与rank配对。

GLM CPU0.30 metadata PD局部重开依据为：通用 `_materialize_request` 的Decode admission仅设置真实Request的prompt computed count和首token，交给真实Scheduler/KV manager管理，未使用Qwen专属attention值；已有prepared bridge逐rank校验GLM完整MLAAttentionSpec与indexer spec、worker配置、Scheduler代表配置及aggregate geometry。因而准确GLM TP8/PP1、EP/MTP-off可以复用该metadata状态程序。它仍是显式仿真connector、synthetic length-only反馈和source-projected cache specs，native.kv_transfer_config不改为实际connector，也不证明KV内容、GPU/provider、性能精度或normal容量；Kimi PD bridge和其他模型/并行边界仍拒绝。

K2.6 TP=EP8的CPU0.30 metadata PD复用同一cache-placement校验：`MLAAttention.get_kv_cache_spec`固定单KV head，EP只影响DeepseekV2MoE的本地专家；vLLM0.30 `ParallelConfig.use_sequence_parallel_moe`在DP1为false。因此普通BF16 MLA、PP/CP/DP1、MTP-off时，缓存spec、完整worker校验和逐rank字节与EP-off相同。process、prepared bridge及allocator callback共用该精确适用域，不再各自复制EP2-only判断；V3 EP、其他TP或MTP仍拒绝；GLM EP8的独立source闭环见下段。两侧真实CPU Scheduler的effective TP/EP/SP字段从其native config导出，须与共同虚拟forward的TP8/EP8/SP-off一致；不能把关闭EP的控制面写成EP8已验收。metadata connector及反馈仍为显式仿真，normal/provider拒绝与逐request原KV传输字节不改变。

GLM BF16/官方FP8的H200 TP=EP8继续使用同一DeepseekV32Model/Decoder/MoE，只把routed intermediate保留为完整宽度并按32个连续专家分片，shared intermediate仍为TP8 shard。v32 `reduce_results=False`保持：当前层shared-add的MoE partial在下一层input all-reduce/RMSNorm前归约，末层在final norm前归约，不能复制Kimi的FFN内归约位置。collective携带被归约的前一MoE invocation身份；source末尾的同shape reshape只创建alias，公共producer-region校验把传输端口绑定回相同allocation/shape/stride/offset/dtype的唯一完整producer，拒绝partial view，不增加copy或放宽跨rank依赖。

`block_fp8_replicated_ep_swiglu_v1`是GLM FP8 EP8的独立声明数学程序，保留旧EP1程序的拒绝边界。依据vLLM0.30 `experts/triton_moe.py` 的完整T*K workspace、默认不丢弃非本地expert的assignment，以及 `fused_moe.py::fused_moe_kernel` 中off_experts=-1的zero-store分支：两次matmul只计本地有效行及活跃本地权重，C的非本地行计零写；激活及后续quantizer遍历完整T*K行，最后sum读完整T*K输出并做T*(K-1)*H加法。本地W2按实际本地行计权重乘法，不由histogram推断未知per-token局部K或只计本地激活；输入quantizer仍读完整T行。该条件程序复用既有FP8矩阵、SiLU、group128量化及DRAM解析资源；不声明实际provider选了Triton/unfused、issued MMA、fusion流量、alignment/scan/metadata、workspace或normal容量，不用时延拟合。EP8仅允许H200准确GLM几何/FP8/无clamp/无graph，native/Midlayer拒绝保持。

### 共同Qwen forward的完整张量PP与0.30逐stage cache（active）

既有A800、BF16、untied dense Qwen3、TP1/均匀PP>=2的范围沿vLLM Qwen2Model.forward的first/last stage分支：仅首stage embedding，仅末stage final norm、logits和sampler；所有stage复用同一decoder/基础算子。IntermediateTensors的hidden_states/residual分别经相邻rank发送，接收创建独立storage，由真实Tensor引用驱动各rank生命周期；sender释放不释放receiver。通信operator记录两个数据port及源码字典发送顺序，lowering核对端点、完整字节、最终MLP/残差producer和下一stage首norm的真实输入，不套用旧手工图仅保留一条跨stage边的布局。传输价格仍调用原P2P predictor/HWSpec，未新增时延参数、拟合或实际GPU launch。TP2的slice/all-gather、EP、非均匀PP、其他模型及Host KV保持各自原边界，不借完整张量程序替代。

0.30 cache owner按共享pipeline_layer_ranges将完整Attention owner map分配到各worker，实际get_kv_cache_configs接收逐rank map，validate/normalize和prepared bridge逐rank核对；同一TP stage的group仍必须一致，所有rank必须共享真实allocator协调后的block数。真实Scheduler代表仍由上游generate_scheduler_kv_cache_config产生；不把首stage的layer名字复制成其他stage的真实owner。CPU native ParallelConfig保持声明的PP数，没有改成PP1绕过库约束；此为真实CPU控制面和源码cache投影，不启动GPU/model worker，不代表actual provider、异步retirement、通信元数据服务或正常容量通过。Qwen3Config明确可省略的rms_norm_eps=1e-6及hidden_act=silu进入已有模型schema并在边界归一化，原始config/身份不改；显式值（包括null）不被覆盖，真实HF配置还须逐字段相等，否则拒绝。继承reserve的容量策略仍标unverified，启动日志10%和完整worker账本不随请求成功通过。

### 既有双专家AF fixture的共享路由数学范围（active）

vLLM0.30 `csrc/libtorch_stable/moe/topk_softmax_kernels.cu` 的switch明确含 `case 2: LAUNCH_TOPK(2, ...)`；wrapper以 `!is_pow_2 || num_experts > 256` 决定workspace，E=2和既有E=128均无正字节softmax workspace。为迁移既有双层/双专家AF+EP2配置，同一显式BF16数学程序接纳E=2、1<=K<=2，复用相同最少运算和公开FP32 weights/INT32 IDs角色，不新增速率、阈值、拟合或实际launch声明。E=64等未迁移范围维持拒绝；源码存在其他case不使其自动通过。该变更只闭合逻辑Roofline，完整比较/索引/编译服务、midlayer/native provider、实际checkpoint路由和normal容量继续独立未通过。

### GLM4原有小配置的共同虚拟forward（active）

vLLM0.11的`glm4_moe.py`以FP32 `nn.Linear`读取显式cast后的hidden，A/B/C均FP32，不能借BF16 A/B加FP32 C替代。新增显式无launch的FP32普通linear数学程序仅支持b=1、无bias、同型FP32输出：2MNK普通FLOPs、4(MK+KN+MN)公开operand字节，复用原HWSpec普通FMA/DRAM资源，不选cuBLAS/TF32/compiler。GLM4单group sigmoid路由的独立数学程序使用实数表达式1/(1+exp(-x))，bias只用于选ID，权重来自未加bias的sigmoid，按config声明是否除以所选权重和；router侧factor为1，模型随后先对routed输出做fresh标量乘，再与shared输出相加。普通非modular `UnquantizedFusedMoEMethod`显式`inplace=True`，故专家输出保留输入backing；shared必须先读完，图自动记录读写依赖。公开路由只有FP32 weights/INT32 IDs两份输出，没有Qwen的第三份token_expert_indices。此为useful数学/role字节Roofline，比较、ID构造、冗余group处理、tensor程序临时流量、sigmoid实际lowering及CUDA/torch有限精度差异不冒充已绑定；原GLM V32/Kimi完整路由拒绝不变。

GLM4配置schema按其HF/attention源码归一化省略的RMSNorm epsilon=1e-5、SiLU、single-group、norm_topk_prob=True和partial_rotary_factor=0.5，原始config身份保留。RoPE只读写active prefix，完整head与未旋转suffix仍在同一view/backing关系内。0.11 `config/speculative.py`把GLM4 draft的num_hidden_layers改为0，原first_k_dense_replace=1的MTP层因此为dense；不能换成0.30 appended-layer MoE。MTP输入的`positions==0`产生bool数组，随后索引赋零原地修改embedding；逻辑Roofline分别记录INT64读/bool写和bool读/仅选中BF16行写，实际nonzero/indexing临时区未闭合。MTP模型体复用同一cache并保留norm/concat/merge/decoder/residual/head顺序；proposer/rejection调用方未迁移时继续拒绝。该范围仍是原有两层/两专家fixture和A800 TP1显式eager source；完整Scheduler/MTP调用、normal容量、实际provider和全迁移完成均不能由模型体或普通请求阶段通过来替代。

### DSV3的0.27.1 grouped routing数学投影（active）

已有DSV3 FP8/BF16计算配置在H200 TP8、EP-off的虚拟forward可显式使用`VECTOR/moe_topk`的`vllm_0_27_1_bf16_sigmoid_bias_grouped_topk` source program及`bf16_sigmoid_bias_grouped_topk_flops_bytes_v1`数学程序。保留BF16 gate logits、FP32 correction bias及FP32 weights/INT32 IDs，限定256专家、top8、8组取4组；不能借GLM/Kimi的单组路由替代。`grouped_topk_router.py::grouped_topk`有bias时每组top2相加，除每个score的sigmoid和bias加法外，另计每行8次普通加法；归一化按显式布尔声明计selected sum/divide，只有router实际scale不等于1才在此计乘法。source Runner后置scale由独立producer表示，不能重复。取负、组/专家选择、mask及ID构造单列非FLOP工作；公共role字节为BF16 logits读、FP32 bias读、FP32 weight写及INT32 ID写，不冒充重复中间Tensor流量、实际DRAM、有限精度等价或已绑定选择指令服务。无launch/unsupported及native拒绝保持，不改HWSpec物理速率；该投影不证明0.30执行、实际provider、库scratch、normal容量或完整迁移通过。

SGLang0.5.19的独立source profile保留BF16 A/B→FP32 gate C，复用同一个grouped sigmoid/top2-group数学owner并按FP32 logits计4RE B；声明Triton grouped gate、8 routed加1 fused-shared槽时，公开FP32 weights和INT32 IDs均为R(K+1)项，shared weight除法、全部9槽的归一化与声明的router缩放按源码表达式计普通算术，不重复专家输出scale。专家仍复用既有FP8 fused-shared recipe和独立声明的histogram；INT64 fused embedding复用原ownership/role字节计算并保留8R B IDs，不借vLLM CUDA kernel的离散launch事实。unfused与FlashInfer AR/residual/RMSNorm为两个显式source profile；后者读取hidden、residual和gamma，返回两份fresh BF16输出，保留2048行源码适用域并复用既有communication implementation。上述均不新增硬件速率、duration拟合或选择服务机制，不断言有限精度等价、实际provider、内部workspace、正常容量或EAGLE验收。

### GLM4的0.11 padded-MTP调用链与typed payload（active）

GLM4的A800 TP1 colocated/eager/all-greedy程序沿0.11 `GPUModelRunner._calc_spec_decode_metadata/_sample`与`EagleProposer.prepare_next_token_ids_padded/prepare_inputs_padded/propose`保留实际materialization：target选取每请求最后verified+1行，bonus用普通BF16→FP32 greedy，rejection的`compute_probs(all_greedy=True)`直接返回独立BF16 target logits；不能套用0.30的额外target FP32 cast/clone。rejection先完整填INT32[batch,max_verified+1]的-1，再按声明的接受前缀读取draft INT32/target argmax INT64并写出，全部接受才读取bonus。该接受前缀是Scheduler scenario/replay输入，不执行模型或推断真实匹配；未知接受信息不能补默认值。

padded proposer的首pass始终包含完整scheduled query；采样位置为query_end减去rejected行，不按accepted数量缩小MTP首pass矩阵。0.11 bool.sum返回INT64 valid count，后续减法/where使rejected与token_indices_to_sample提升为INT64，普通无spec query-end仍INT32。arange全行advanced indexing仍产生fresh ID/position/hidden/slot输出；shift-copy、scatter、proposer持久buffer copy、首pass两份hidden选择、逐passINT64 argmax→INT32输入、position/sequence更新、block gather和slot port、最终stack全部显式连接。采样/准备helper保留各自Python frame释放边界；KV reader使用当前真实虚拟INT64 slot Tensor port，不能只写坐标声明而遗漏producer依赖。声明的逻辑page arena按原block size对齐，每请求不重叠，但不冒充实际native page ID/内容绑定。source max_model_len/max_batched_tokens/max_requests必须显式提供；Eagle无条件inputs_embeds、ID/position/hidden/arange/backup驻留由这些constructor容量生成。cap-exceeded、discarded/incomplete行、tree、PP/EP及其他未迁移域继续拒绝；AF范围见下述独立服务契约。

`VECTOR/vllm_mtp_eager`只定价这些source操作的typed公开role字节，所有整数/比较/转换/ID选择的浮点算量为0；这不表示其instruction、library、临时区或真实DRAM工作为0。native执行与未绑定服务保持拒绝，不借旧generic vector时间补齐。Host backup/index H2D复用既有MEMCPY/PCIe payload；DMA Roofline沿D2D相同的include_latency=False语义只计bytes/bandwidth，不依赖或补造未知PCIe启动latency，midlayer仍要求真实独立latency。原0.11桥接从实际Scheduler Request的非spec token前缀和scheduled_spec_decode_tokens后缀导出完整输入ID；这是既有length-only反馈程序的数据，不是模型采样结果、假SDK或GPU验收。原v32/Kimi路由拒绝、normal容量与最终全迁移门槛均不随该小配置请求闭环而通过。


### 既有GLM4 AF服务中的共同target/MTP forward（active）

原A800 TP1、EP-off的GLM4两层fixture沿既有Attention/FFN服务契约运行同一个虚拟target和dense MTP模型。每次decoder调用把完整BF16 hidden_states与residual发往FFN pool，由该pool执行post-attention norm及dense/shared/routed FFN，再分别返回两份输出；首个padded MTP pass保留全query行，后续pass仅保留每请求一行。重复调用同一MTP层使用不同传输节点，保持同一模块/cache backing与实际Python生命周期，不复制另一套模型图。相同布局、offset、dtype与backing的返回view允许绑定唯一对应的producer输出region；切片或其他region不得借此绕过typed lowering检查，也不引入虚构contiguous复制。

AF CLI按显式collective_backend选择现有analytical_roofline_v1或ProtoBridge point-to-point服务，并如实登记所消费的profile；没有选中的ProtoBridge参数不写成Forward依赖。该接线不证明vanilla vLLM提供实际AF runtime。KV block几何从同次resolved Scheduler capacity plan传入虚拟proposer；显式测试block数仍不是normal容量。合成token/接受反馈、完整worker峰值、provider、性能精度和全迁移验收继续分别未通过；Host KV/preemption等组合不由普通AF迁移自动通过。


### GLM/K2.6显式路由数学Roofline与请求链（active）

第一阶段的`biased_sigmoid_topk_flops_bytes_v1`只表示GroupedTopKRouter的有用实数数学：每个logit一次exp、一次分母加法、一次除法及一次选择bias加法；每请求row对所选K项做K-1次求和、K次归一化除法和K次权重缩放。取负和比较/ID选择分别保留为typed action，不记作FP32 FLOP或INT32峰值需求；exp和除法不是compiler指令/pipe服务绑定。GLM保留FP32 logits，K2.6保留BF16 logits，bias与输出weights均FP32、输出IDs为INT32。公开role读写字节复用各自既有source账本，bias按row的role读取不等于不同参数backing或已观测DRAM交易。该新程序仅在显式声明时返回理想数学Roofline；未声明程序仍拒绝，不删除原完整选择服务的拒绝边界，不断言CUDA block/warp/torch有限精度等价，不借局部CUDA12.2编译当H200 provider或精度验收。该程序与既有Qwen/GLM4 useful-math路径同层，来源为数学/源码，无duration拟合或硬件速率调整；历史未计价结果保留原snapshot。

普通CP1、PP1、EP-off、MTP-off的K2.6缓存图按KimiK25.language_model的DeepseekV2Model及MultiHeadLatentAttentionWrapper/MLAAttention源码生成：每层`language_model.model.layers.<i>.self_attn.attn`持有一个num_kv_heads=1、head_size=kv_lora_rank+qk_rope_head_dim的BF16 MLA spec。INT4权重不改变KV dtype；无indexer、window或draft缓存。真实CPU0.30 allocator按每rank显式预算分配并核验所有source owner、native offsets与block数，不创建模型层或真实模型Tensor；这不证明权重/reserve下的normal容量。GLM/Kimi scheduled embedding ID从真实Scheduler Request及显式length-only反馈导出，缺行直接失败，不补占位值。其他并行/量化KV/推测分支继续拒绝。

### BF16 MLA单块历史上下文与注意力状态合并的数学Roofline（active）

普通MLA forward_mha的历史上下文分支保留causal suffix O/FP32 LSE、缓存latent/RoPE读取与kv_b展开、noncausal context O/FP32 LSE，以及最终merge_attn_states。现有mla_prefill的bf16_mla_context_softmax_v1只描述非空矩形query/key数学工作，P=H*sum(q_i*k_i)，不把q<=k作为非因果注意力要求；prepared Q/K/V/O各计一次，FP32 LSE另计4R B。fresh causal程序仍要求q_lens==kv_lens，显式输出LSE时复用softmax max/sum另计R次log和R次加法。新VECTOR/vllm_attention_state_merge只接显式vllm_0_30_nonempty_lse_merge_flops_bytes_v1：两路BF16 O与FP32 LSE读、BF16 O写及可选FP32 LSE写，常规实数工作为每query/head行2exp、2减/1加/2除，每输出元素2乘/1加；输出LSE另加log与加法。max和dtype转换不冒充浮点算术，空状态分支、CUDA逐pack重复工作、有限精度异常值、scratch及实际provider/launch保持未绑定，native拒绝。caller可声明全部请求均有正历史上下文且总量能装入源码workspace的单chunk；Kimi另可显式声明单请求、block_size整除65536且source最大长度已使workspace饱和的多chunk分支；连续位置属于明确的logical cache arena，不是实际block-table或物理page结论。不得由该图或数学计价升级normal容量、编译器服务或GPU精度验收，不改HWSpec，不拟合production duration。

### 虚拟dense FULL decode的逻辑行与捕获backing（active）

单bucket FULL_DECODE_ONLY的源码投影区分有效L行与模型输入P行：同一虚拟forward的embedding、norm、投影与残差按P表示，vLLM0.30 FlashAttention后端按num_actual_tokens切出L行query/output，KV只追加有效行，logits/greedy只选实际请求。GQA数学程序在FULL单token decode可保留model_input_q_lens/model_input_kv_lens来描述捕获backing，注意力有用FLOPs与角色payload仍只来自有效q_lens/kv_lens；不据此声明实际FA launch或读写padding。KernelSpec继续拒绝实际launch、tiling、PIECEWISE和多token数学decode混用。source侧静态port身份、capture范围与Python生命周期可由同图表达，不把全部中间存储标成永久驻留，不把动态图峰值等同CUDA graph私有池reserve、compiler内存复用或异步retirement。未知pad值只在TP1已知全部词表归属的逻辑域表示，不伪造token ID或扩TP ownership；缺失真实Host capture切分的组合保持拒绝。

### Kimi TP=EP8 Marlin 的完整行激活与零本地路由（active）

显式 `routed_experts/mathematical_program=kimi_ep_marlin_swiglu_v1` 接 A800/H200、BF16 输入、INT4/group32、384 专家/top8/H7168/I2048、TP=EP8 的无 launch/unsupported 源码程序。MoE 内部 TP1，每 rank 保留完整输入行，只拥有连续48专家，global histogram 与384项 local mask 分开验证；共享专家仍是 I/8。vLLM0.30 `experts/marlin_moe.py` 的激活缓冲按 T*topk 行分配，普通 `activation.py` SiLU/mul 不读取 expert_map，因此即使本 rank 无 assignment，仍计完整 T*topk*I 激活，不能删 rank、改 routing seed 或伪造零 grid。矩阵 useful FLOPs/INT4 权重与 BF16 group scales 只随本地 assignment/active expert 计；map-aware sum 按 expert_map 跳过非本地 intermediate 读取，零本地 rank 仍写完整 BF16 输出。ID/map 字节是公开 role-use payload，不是逐 warp/vector 发射读取量；activation 复用共有 SiLU 源码算术，矩阵/activation/weighted sum 逐阶段求理想 Roofline。排序、整数/atomic、packed 解量化及转换舍入、scratch/allocator、实际 launch/provider 和 normal 容量仍独立未绑定，midlayer/native 继续拒绝，不改 HWSpec 速率，不使用 duration 拟合。

既有A800 K2.6 TP8/PP2/EP8按31/30层分配复用同一DeepseekV2KimiModel、MoE和MLA；各stage独立8-rank专家组，full-row激活、local assignment矩阵、map-aware sum的数学工作量与H200相同，只读各自HWSpec速率。Kimi的late all-reduce在FFN内部完成，所以PP发送已归约hidden，接收后不插入GLM的stage-input all-reduce。两模型共用vLLM0.30的flattened lane收发、fresh gather、worker intermediate copy及storage生命周期程序，各自的源码分支与严格producer端口校验保留；每个请求步骤仍由共同Serving执行引擎lower、predict和建时间线，不调用旧builder或memo。该迁移不把明确给定的token/路由与constructor状态升级为真实Scheduler反馈、native provider、数值结果或normal容量验收。


### MLA 多 chunk accumulator 与 FP32 copy 的逻辑 Roofline（active）

SGLang0.5.19不等长prefix的后续chunk允许部分request的KV长度为0，但整个chunk仍有正KV。局部重开原空状态拒绝的依据为：本地FlashInfer0.6.11.post1源码 `flashinfer/data/include/flashinfer/attention/prefill.cuh::init_states` 将O置0、m置负无穷、d置1，ragged kernel保留kv_len_safe并写O/LSE；相邻 `state.cuh`也明确同一初始状态。SGLang0.5.19的 `models/deepseek_common/attention_forward_methods/forward_mha.py::_chunked_prefix_attn_mha` 在每个chunk后调用 `merge_state_v2`；`python/sglang/kernels/aot/csrc/attention/merge_attn_states.cu` 将无穷LSE归一为负无穷，仍读取两路完整O/LSE并执行同一合并。此为声明的数学源码域，不证明安装FlashInfer就是该SGLang实际provider或已执行的版本绑定。

新 `bf16_mla_context_empty_rows_softmax_v1`仅开放H200的192/128/512 BF16非因果context＋FP32 LSE：q_lens正、kv_lens非负、至少一request为0且最大KV为正。空行仍物化零O与负无穷LSE，逻辑payload包含全部Q/O/LSE，只有非空行计softmax/reciprocal/log；K/V和矩阵算量按有效pair计。新 `sglang_0_5_19_empty_prefix_lse_merge_flops_bytes_v1`须显式声明空prefix query行数，suffix始终非空，保留两路O/LSE读与fresh输出，不把空prefix当作省略merge的理由。原nonempty程序继续拒绝零KV，其他空状态、both-empty、真实DRAM/反复pack算术、compiler/launch与native仍未绑定；不改HWSpec速率、不用production duration。此接线闭合既有SGLang分块prefix的逻辑Roofline子域，不代表所有SGLang模式、旧路径删除、normal容量或最终Serving验收通过。

Kimi 单请求65537行历史在已声明source block_size=64、65536行饱和workspace条件下按65536＋1分块，复用同一BF16 gather workspace的有效切片；每chunk独立gather/kv_b/noncausal attention，首chunk将O/LSE复制到独立BF16/FP32 accumulator，后续chunk原位merge并写回FP32 LSE，最后才与causal suffix合并。原位merge只允许精确prefix O/LSE aliases，final输出仍须独立；自动图记录写后依赖，不能改成逐chunk对suffix合并或漏掉初始copy。VECTOR/gather显式mathematical_program=materialized_fp32_copy_bytes_v1仅接FP32、factor1、无index/scatter且materialized_output=True、无launch/unsupported，一次读和一次写合计8E B，无浮点算术；这解除因缺物理copy launch而阻断的逻辑Roofline，不绑定实际DRAM、copy compiler、页/allocator、workspace分配或native provider。旧未声明program的BF16/INT32 gather及native拒绝边界保持，不造INT32数据通路支持，不调整速率。

### SGLang fresh target 的共享逻辑 Roofline（active）

SGLang0.5.19的显式target source profile限定ragged/eager/CP1、无layer hook、fresh prefix0的MHA prefill及后续普通decode，复用已有attention数学、block128 GEMM、source typed字节与通信owner。prepare/core按源码独立concat K、cast V和cache写，O直接入row-parallel投影，不套用vLLM caller输出copy。LogitsProcessor的INT32 extend lengths经cumsum得到INT64并减1，advanced-index生成fresh BF16请求末行；逻辑typed owner分别计一次读/写，不虚构Triton launch或对全部prompt行执行head。该源码分支不绑定安装provider、未知prefix/chunk分组、EAGLE或sampler，不构成完整worker内存、normal容量或GPU时延验收，不改HWSpec、不使用production duration。

SGLang target可显式配对`sglang_0_5_19_plain_greedy_fp32`：CUDA/all-greedy、无custom processor/logprob/grammar/return mask、NaN assert/sanitize及token-sync关闭。采样复用既有argmax逻辑角色字节owner，FP32 logits读与fresh INT64结果写，不插BF16转换、INT32 copy或long/unsqueeze。比较/索引选择不计作FLOP；source返回的INT64元数据不代表执行了数值采样，后续输入token transcript仍须外部明确提供。只给传统FLOPs/bytes Roofline，不绑定PyTorch实际reduction launch、compiler比较指令、GPU时延、native/provider或normal容量；不改速率、不拟合。


### SGLang 短 prefix 的 absorbed MLA 逻辑输出（active）

SGLang0.5.19的显式ragged/eager/CP1/no-hook、BF16 MLA source在纯prefill且0<sum(prefix)<8192时沿FlashInferMLA parent EXTEND，只返回latent O，不增加FlashMLA decode的公开FP32 LSE、log及输出加法。`mla_absorbed_prefill/bf16_mla_absorbed_prefill_softmax_v1`复用原absorbed MLA causal score-pair及softmax数学owner，支持不齐query长度，不借用FlashMLA uniform-query<=128约束；prepared Q、shared latent/RoPE cache和latent output各按公开role计一次。阈值来自源码默认分派，属于声明适用域，不按时延选择。实际page planning、scratch、物理traffic、launch及native provider仍未绑定，midlayer继续拒绝；无新速率、KernelKind或production duration拟合。

### V32 两 token MTP 的 decode 元数据与复用选择（active）

vLLM0.30的legacy drafter保留目标的speculative_config；DeepseekV32IndexerMetadataBuilder以next_n=1+num_speculative_tokens构造decode_threshold，所以显式spec2使用阈值3，q2仍是decode、q8是prefill。旧单token数学程序保持原域；新的specdecode程序携带num_speculative_tokens与decode_threshold，mixed按两组各自query/context形状建数学调用，共享其声明输出backing，不把max-query当成同一物理kernel。算量与公开role字节复用既有causal pair/softmax/indexer owner，不新增速率或推断draft实际backend。pass0后的shared indexer只复用已compact的旧top-k：显式reused_selection_kv_lens绑定旧query的causal KV长度，selected=min(K,old)，新query的KV增长不增加旧选择，短4→5仍计4；只有调用方提供完整数据流时才允许这条复用声明。实际selected IDs、物理page/metadata转换、scratch、provider/launch与normal容量仍分别未验收。

### SGLang 长 prefix 的 one-shot MHA 逻辑路径（active）

SGLang0.5.19的attention_backend_handler以sum(prefix)>=8192进入KV扩展路径，在sum(KV)<=声明的SGLANG_MAX_KV_CHUNK_CAPACITY时选择ONE_SHOT（源码默认131072）。该分支先写suffix cache，再按显式INT32 combined-KV索引读取分离的BF16 latent与RoPE backing，对全部KV做kv_b扩展和一次causal MHA；不加公开LSE或chunk merge。新的bf16_mla_one_shot_prefill_softmax_v1复用已有causal pair与softmax owner，旧fresh程序仍拒绝Q<KV。共同入口通过SGLangPrefillInputs传入request顺序、索引和capacity，不推导实际req_to_token映射或索引producer；来源只接受声明场景，不能冒充观测。超capacity的chunked分支、actual provider/launch及normal容量独立保留未闭合，不新增速率或拟合。

### 虚拟 dense source 的既有 launch 选择与 0.11 capture（active）

A800 BF16 Qwen3/Llama共同forward可把injected_launch透传到现有canonical selector，不能只解除入口拒绝后继续静默生成数学attention。连续Linear沿已有row-major A/weight/C、transA=False/transB=True及epilogue描述，单行LM head使用既有gemv观察key；不得为命中lookup任意改转置或补新速率。source管理的GQA不重复注入旧wrapper的输出填零与三次Q/output/LSE拷贝，main/setup/combine仍走同一selector，缺key原样拒绝。embedding、选行、采样等source-only程序不升级为native，整图midlayer仍按未绑定调用拒绝。

vLLM0.11 GPUModelRunner在model.forward之后选hidden rows并compute_logits，FULL graph尾部slot mapping明确填-1，与共同dense capture的边界一致。因此同一dense源支持已维护的uniform-query1 FULL_DECODE_ONLY及既有PIECEWISE组合，传递0.11 model/sampler身份；不借用SGLang的slot0/seq1语义。正常KV allocate/free事件不应触发仅针对Host KV传输的capture拒绝，LOAD/EVICT继续拒绝。调用只验证声明源码场景，未知compiler、workspace、graph pool与normal容量仍分别未闭合，无新性能参数或GPU执行。

0.11 A800 TP2的非融合embedding对完整物理输入执行mask/gather/fill，padding ID可保持未知，不能借此放宽0.30融合分支。0.30 NVIDIA dense TP2不足bucket须逐step提供`VLLMDenseGraphInputs`，明确request顺序、声明来源和保留的padding IDs；有效行IDs仍使用原scheduled port，两者拼接后才决定各rank的词表权重读取。源码`gpu_model_runner._preprocess`将`input_ids[:num_input_tokens]`交给model，positions尾部清零，KV slot尾部仍按原-1语义处理；head/sampler只消费有效请求。缺失、过期、长度或词表不合法的尾部不得补值。该状态沿既有`BackendStep.model_source_inputs`或同一immutable per-step binding传递，eager step可显式声明None，缺step key仍失败；不建立第二条模型forward。lowering memo必须保留source embedding的rank，因为词表shard与`owned_token_rows`由其deployment位置决定，不能跨rank复用第一张卡的KernelSpec；其余与placement无关的算子继续复用。声明的padding输入和逻辑图不证明真实Scheduler捕获了这些值、provider、allocator或normal容量，已有结果保持各自snapshot，不能自动继承缓存修正后的价格验收。


### GLM TP8/PP2 共同虚拟 forward 保留原 TP-partial 传输（active）

GLM 78层的既有H200 TP8/PP2、EP/SP/CP/DP1、MTP-off source projection复用同一decoder/基础算子与deployment的38/40层分配，各stage首层必须有full indexer。vLLM0.30原程序发送最终MoE的TP-partial hidden与residual的flattened lane，接收stage各做fresh all-gather，再copy到worker预分配intermediate backing，随后按原decoder执行input all-reduce和residual norm；不能补发送前归约、发送完整tensor或改变原P2P字节数来宣称数值等价。共同lowering核对全部lane的独立receive backing、offset、producer端口、gather、worker copy、接收stage输入归约与norm依赖，沿用既有P2P/collective与算子Roofline价格，不增加速率或launch假设。此模式保留source_projection_only与numerical_tp_pp_equivalence_bound=False；源码投影全调用可计价不解除TP-partial传输数值一致性、实际provider或正常容量的独立门槛。

既有A800 BF16 TP8/PP4/EP8的18/20/20/20层分配也由相同DeepseekV32Model执行；PP1/PP2/PP4只改变每stage的连续layer range、首stage embedding和末stage norm/logits范围。每层routing绑定其所属stage的8个物理expert rank，不能把32个worker作为同一个EP组。三个PP边界分别发送两种tensor的八个flattened lane，保留原TP-partial语义，receiver各自gather和worker copy；同图逐rank storage由通用生命周期owner导出。A800的普通Prefill/Decode逻辑Roofline继续复用既有DSA、BF16专家和BF16 A/B＋FP32 C数学工作量，使用A800自身速率；未绑定launch的融合norm/RoPE与mixed GEMM只接解析Roofline，不借用H200 launch或FP8矩阵能力。lowering逐边界验证全部lane、worker copy及接收stage输入归约，EP按每次invocation所属stage选8个专家rank；三个stage末尾的partial仅允许进入其已完整校验的PP消息，其他层及末stage仍须精确all-reduce收敛。后续stage的in-place residual可指向worker大buffer的有效prefix，发送字节仍只含有效行，backing容量与生命周期不缩成payload。该源码数学路径不解除TP-partial数值等价、normal/provider与完整Serving验收门槛。


### SGLang topk1 EAGLE batch 与两步草稿共用虚拟 NextN（active）

SGLang0.5.19既有topk1 EAGLE一/两步和多请求复用同一虚拟NextN body；prefill逐请求shift/copy并按实际ragged长度生成positions。verify保留每请求完整W行，使用显式接受前缀，完整draft-extend后按row*W+accept_len-1选择carry；两步草稿在target之前仅运行num_steps-1次draft-decode，本范围为一次。初始/后续draft IDs、tree与slots来自typed scenario，不能由虚拟模型计算或任意生成。split8192 partial FP32/INT32缓冲及final probability/INT64/position/可选列写沿准确本地kernels/ops/speculative/topk1.py；tree只发布已声明链的useful unique role地址投影，不代表CUDA issued load、branch、DRAM或内部workspace。源码分支以eagle_worker_v2.py、eagle_worker_common.py、eagle_utils.py、spec_tree.py的公开端口和models/deepseek_nextn.py为准；不把本地旧0.5.12 CUDA实现当成0.5.19 provider证据。lowering保留逐请求query_lens和可选draft列写的显式字段；沿用原Roofline资源价格，不改物理速率或launch，实际provider与完整native验收继续拒绝。

普通all-prefill、graph-off且0<sum(prefix)<8192时，target与NextN均复用ragged absorbed MLA；分支由batch prefix总量选择，不能把其中prefix0的请求单独改成fresh MHA。新短prefix的SGLangEagleChainState必须显式prefill_tail_source=final_suffix_no_prompt_override，声明这是完整最终suffix且不使用chunked_req_next_prompt_token覆写bonus。仅bonus等于argmax或发出一个token不能证明该条件；未声明时拒绝，verify不接受此字段。历史fresh JSON缺失此Optional字段只归一为None，其余required字段保持严格。cache历史、slots、token/bonus来源仍是caller声明；该静态源码迁移不解除Mirror/CLI prefix注入缺口，不接受非最终chunk、长prefix、mixed或prefix graph，不新增attention公式或provider。


### 普通dense未知输入值与原生异步placeholder（active）

普通BF16 Qwen3/Llama单pool colocated、TP1/PP1、EP/路由/MTP/graph关闭的common source中，SymbolicEmbedding只消费INT32 input的shape/dtype，全词表归同一rank，具体ID不影响逻辑读写或后续算量。vLLM0.11 A800的TP2 unfused mask/gather/fill程序同样对全部输入行执行公开typed读写，再做all-reduce；其逻辑payload与ID所属shard无关，允许同一普通eager范围保留未知ID。两种范围均允许embedding_token_ids=None，仍创建虚拟INT32输入端口，不能补零或顺序ID冒充真实输入；positions/slots继续完整校验。0.30 fused embedding的owner-only权重读取仍依赖ID，其余TP分片/版本及模型/值依赖范围继续拒绝。普通KV ALLOCATE/RELEASE不消费token值，允许其既有生命周期事件；prefix-cache及Host传输保留独立契约。上述全行payload不证明编译器条件写、cache/DRAM流量或实际provider。

已有Qwen3 FP16配置和Serving范围复用同一dense forward，embedding、hidden、KV、LM head与logits row-select/reorder保持FP16；仅原Sampler.forward显式转FP32后执行argmax并输出INT32。0.11/0.30分别使用`*_plain_sampler_fp16`与`*_plain_greedy_fp16`声明，不能把FP16输入标成BF16来源，普通BF16行为保持。所有16位operand沿现有FLOP与字节表达式，不增加性能参数或native绑定。0.30 C500既有TP2 masked embedding与0.11 A800一样逐完整输入行mask/gather/fill，其公开logical payload可保留未知IDs，不适用于CUDA融合分支或其他TP规模。默认Serving及H200静态probe保留model/config dtype和原输出范围，不因迁移自动补sampler。C500现有vector/GEMM launch owner只接受其绑定的BF16 artifact，FP16源图不能据此生成虚构launch、转BF16或声称完整价格；无launch indexed gather和其他通用FP16数学契约仍待实现，native/provider独立未通过。原worker启动sampler storage程序仍只绑定BF16，FP16启动/normal容量不自动继承该账本，必须明确拒绝。基础配置与实际checkpoint身份沿原owner保留，配置替代项不新增模型或解除最终验收。

### C500 homogeneous FP16 的无 launch 数学路径

共同虚拟Tensor图的C500 FP16 eager节点可显式声明`unquantized_fp16_flops_bytes_v1`（无bias、无量化、b=1的同dtype Linear）或`dense_fp16_source_flops_bytes_v1`（materialized gather、RMSNorm/add-RMSNorm、普通SiLU/mul、full-head RoPE及有效slot的auto paged KV写入）。GEMM与原BF16 BMM复用2BMNK算量和typed唯一A/B/C payload；vector/KV复用现有数学表达式，索引/position/slot保持INT64，norm的H-1次求和及mean/epsilon逐项计数，gamma只计一份。Roofline仅使用既有HWSpec资源容量，逻辑payload不代表实际DRAM流量、浮点数值等价或完整指令服务；无launch/unsupported必须同时成立，native/midlayer、resource-input、graph padding、clamp、量化和异dtype输出继续拒绝。lowering在物理launch选择前确定该显式数学域，不执行量化器或split-K reducer推导，不以BF16 launch代替FP16。该路径解除已有FP16源码数学图的计价缺口，不证明实际provider、工作区、启动账本、正常容量或最终Serving验收。

Qwen3的同FP16输入/权重可复用显式PyTorch-sort worker启动storage程序，限定无head_dtype override的模型dtype head。vLLM0.11/0.30的Sampler.forward均先转FP32，分别保留一次/两次warmup及caller-retained logits生命周期；worker inputs_embeds和head/logits保持FP16，sort/noise临时区保持FP32且indices为INT64。原FP16全拒绝边界仅在这两个source版本的TP1/TP2显式sort范围解除；FP16 native-Triton、FP16 Llama及未知profile继续拒绝，不将条件source账本当实际backend、完整worker reserve或启动log10%验收。

vLLM0.11 AsyncScheduler可按已知num_tokens加num_output_placeholders调度尚未返回的输出，Request.all_token_ids仅含已解析prefix。仅在0.11、async开启、pipeline depth大于1且非spec的明确分支，按调度前placeholder数校验query末端上界，合法未决输入显式导出None且token source为None；已知ID仍校验类型、prefix长度和普通缺失，不能伪造反馈、改调度顺序或时序来消除缺值。shape-only dense TP1及上述0.11 A800 TP2 masked程序可继续运行，其余需要具体ID的source在上层拒绝。conformance Roofline及原生CPU Scheduler async/depth均复用同一forward/预测/Host时序；该事实不证明并发worker全局activation峰值、normal容量或GPU/provider验收。


### GLM no-metadata 启动程序共用 decoder/NextN 生命周期（active）

vLLM0.30 profile_run使用最大token/request构造域执行target _dummy_run、两次完整batch的drafter dummy_run，再执行target sampler；这不是接受/拒绝驱动的小请求程序。GLMStartupAttentionFactory只接受明确no-metadata启动配置，复用原投影、Q absorption及source indexer工作区/dummy logits，attention返回零输出，不创建请求KV cache或slot。target hidden保留到两次draft均返回，draft中间值按各自Python frame释放，sampler复用公共worker profile storage程序；同图逐rank生命周期给出该条件源码程序的已知峰值，不能把各rank独立max相加当单卡峰值。BF16与官方FP8 GLM共用结构及原量化linear owner，无真实tensor计算。该程序仍缺全部routing下的内部workspace上界、index/spec元数据、通信/库/compiler/runtime/allocator状态；已知启动峰值不能升级为完整worker headroom或normal KV容量，官方FP8驻留guard不解除，不改硬件速率或拟合时延。


### SGLang EAGLE 三段exact source capture与公共端口（active）

SGLang0.5.19 topk1/steps2的verify迭代保留draft-decode B行、target-verify 3B行、draft-extend 3B行三段FULL source capture。draft图包含首列token copy、同一NextN、split topk1和position reset，返回parent_list/top_scores_index/draft_tokens；target/extend分别返回完整FP32 logits与BF16 hidden。target采样/tree verify及extend后的选择/greedy均在对应图外。公共capture_sizes保持model-input token-row单位，必须显式包含B及3B；source_graph_batch_size另记实际请求batch B，不把字段重解释为request桶，不以普通query1图代替三段图。prefill仍走已有eager程序，underfilled或其他depth/图模式不由此通过。

ForwardTrace保存source_captures序列并验证分段输入/完整named outputs，未结束图拒绝发布；source_capture仅兼容零或一段，不能静默丢掉多段。执行报告按stage保留端口、逻辑replay与source瞬时峰值；同一allocation_id的静态port容量只计一次，输入与输出同名端口保持分开。该记录不表示物理backing跨bucket复用、compiler partition、CUDA pool reserve、实际replay或provider。完整normal容量继续拒绝，未增加launch参数、速率或时延拟合。

Host时序可将source捕获段映射到既有RuntimeGraphSegmentDescriptor，图外head/sampler保持eager，不把整step套成一次replay。source_runtime_mode保留原捕获FULL，step_host_coverage_mode按整step实际覆盖记为PIECEWISE；不更改用户GraphExecutionConfig或猜compiler partition。跨捕获边界的lowering fusion、缺失canonical source顺序、未展开的repeat occurrence及非连续Host段继续拒绝；价格只来自原HostLaunchProfile，不新增速率或独立时延参数。

同一ForwardTrace的同步source stream在capture入/出及PIECEWISE段之间保留程序barrier，避免图外独立准备被data-DAG拓扑提前排入未完成的捕获区域。相邻区域的sink通过后一区域first source op连接其余独立root，边数保持线性，区域内部其余DAG并发不变；普通无capture图的原依赖不变。barrier不是新增kernel或tensor访问，不推断compiler内部stream或CUDA pool；多stream声明仍须自身完整contract，不能通过全rank串行化掩盖。

### 静态dense探针与完整请求的采样边界（active）

历史H200 BF16 Qwen3/Llama TP1静态探针可复用同一虚拟forward截止于LM head，不因迁移自动添加greedy。该域以既有target_sampling_source=None表达，限0.30 CUDA/eager、无TP/PP扩展、MoE/MTP/graph或Host传输；报告标明lm_head_only_no_sampler及条件source，不表示checkpoint实际provider。完整逻辑请求gate同时校验报告scope和每phase的RuntimeInputs，不能删scope字段后把静态图冒充完整；Serving执行边界在建图前拒绝无sampler的静态配置，公共CLI仍要求明确匹配的采样源。直接trace API默认完整greedy行为保持，模型和算子实现不复制。


历史H200 BF16 Qwen3-MoE TP1静态探针也复用共同forward，沿用原显式seeded routing；显式canonical profile仍由原resolver携带histogram/provenance。TP1 embedding只消费INT32 shape，routing独立消费声明的histogram与schedule，不读token值，故普通0.30 eager/PP1/EP-off/MTP-off/graph-off可保留未知IDs。此例外不推广到hash router、GLM/Kimi/V4或TP shard ownership。旧target_sampling_source=None仍截至LM head，未新增greedy；显式greedy才包含sampler，Serving完整请求继续拒绝无sampler配置。默认probe四phase经同一新路径，未知/已知IDs的图、spec、预测及storage保持一致，无新公式、launch或provider绑定。


### Dense启动存储按vLLM版本复用source程序（active）

普通BF16 Qwen3/Llama的启动账本沿同一虚拟model forward与live-storage扫描，显式model_execution_version区分0.11和0.30。0.11 worker构造保留INT64 discard_request_indices与num_accepted_tokens，省去0.30才有的数组；dummy sampler只调用一次，PyTorch top-p scatter物化fresh输出，输入logits按outer frame保留。0.30的原constructor顺序、in-place scatter和二次warmup保持。TP2的0.11 embedding采用原mask/gather/fill程序，head/gather保留同版本来源。0.11暂只支持声明的PyTorch-sort/no-logprobs/FP32-noise源，不将实际FlashInfer或编译分支推断为该程序。source report和候选policy保留版本，未知库/compiler/runtime/allocator继续uncovered且normal=false；原生KV resolver、外部reserve与日志10%验收不因这次兼容改变。


H200 vLLM0.30普通BF16 Qwen3/Llama的TP2静态probe，在caller逐phase显式提供有效token IDs时也复用同一parallel forward；保留fused embedding shard ownership、row selection、分片LM head、all-gather及必要的fresh vocab重排。target_sampling_source=None只关闭共同forward末端sampler，保留所有head/boundary工作，完整请求仍拒绝该静态scope。unknown IDs不得据模型配置补造、均分或回退旧builder；TP4、PP/EP、MTP/graph及其他静态组合不由此放行，不新增collective或算子计价公式。


A800普通BF16 Qwen3/Llama的0.11 CLI可把显式PyTorch-sort启动程序直接接到同一source policy projection与既有native KV allocator，再交真实CPU Scheduler执行；model_execution_version作为显式输入和report字段传递，不能在0.11容量中默用0.30的worker/sampler布局。policy的hardware/model/config/TP/构造容量/page尺寸必须与CLI逐项相等，debug blocks、graph、MTP、MoE及其他backend/source组合继续拒绝。activation从同一图的非驻留同时峰值导出，外部reserve仍继承且unverified；成功完成请求不把其normal_capacity_accepted、启动日志10%或GPU/provider门槛升级。原0.30路径保持相同allocator与source声明。


A800 vLLM0.11 Qwen3-MoE的既有validated TP=EP2域复用同一EP专家模块与collective组：内部expert TP=1、线性半区ownership、完整输入行和全局histogram、本地专家结果仅做TP all-reduce，DP1没有额外naive dispatch/all-to-all。0.11仍保留其独立FP32 routing cast和FP32 top-k数学程序，embedding选原0.11 mask/gather/fill；TP1旧域不变，非EP TP2与其他未维护组合仍拒绝。EP lowering按版本严格连接gate→cast→route→experts，校验cast的rank/pool、BF16输入、完整元素数、typed读写次数和唯一公共输入输出，不能只放宽route dtype或跳过gate hidden绑定。CLI默认选择也只追加既有validated2/128-expert EP2，真实CPU Scheduler输入保持其length-only身份；完整请求成功不升级normal容量、真实模型输出、实际launch/provider或计时准确性。


A8000.11普通dense默认CLI的source选择不再按prediction_method区分analytical与injected入口；沿已支持的同一injected source contract保持用户原Midlayer方法和launch policy，不为继续运行而换成Roofline或旧builder。embedding/选行/采样helper缺少实际launch时继续由KernelSpec的unsupported契约明确拒绝，入口已统一不表示Midlayer已完整计价，原正常容量和完整报告门槛保持。


### vLLM0.11 prefix 请求标志迁移保留长度回放边界（active）

既有CPU0.11 backend始终关闭content-hash prefix caching，使用trace_length_private_nonshared_v1：canonical输入独立提供命中长度H，按block对齐后回放已计算prefix，KV仍为private/nonshared backing。共同dense forward按剩余query行和完整KV上下文执行，不从length-only token值推断命中；CLI可保留requested_prefix_caching标志而effective_prefix_caching仍为false。conformance不由此获得hashing或prefix语义，其他版本/source的原边界不变。完整小请求校验H17/block16得到computed16、query16、KV32和正常allocator生命周期；这不是共享缓存页、实际content cache、normal容量或provider验收。

同一长度回放契约也适用于既有A800 TP1 GLM4 target/MTP source。默认/显式GLM4入口保留requested prefix标志，实际内容hash缓存仍由原CPU backend关闭；source仅消费Scheduler传出的computed/query/KV与接受场景，不制造命中长度。既有AF两请求抢占/恢复继续受原32→2/H0与32→4/H32、batch64、六块HBM、depth2/fixed接受1的完整Simulation scope限制；共同模型不复制抢占专用建图。完整小请求API检查和CLI解析覆盖分别报告，不以程序化fixture替代标准Serving report或正常容量验收。


H2000.30 denseTP2默认静态探针与显式输入探针使用同一normalization；省略整份token映射与逐phase传None等价，均保留未知值并由共同forward拒绝缺少shard ownership输入。不能仅因映射缺失就转入legacy模型图，后者不提供所缺事实。显式有效IDs的原LM-head-only/greedy两种输出范围保持，TP4及其他未迁移组合不由此放行。


### DeepSeek-V3 静态探针复用同一版本化 forward（active）

既有H200 TP8/PP1/EP-off DeepSeek-V3算子探针可显式选择已实现的vLLM0.27.1共同forward，复用同一投影、FP8、MLA、collective和greedy基础算子，不复制模型图或计价公式。caller须提供逐phase token IDs、最大context和batch构造容量，以及匹配的0.27.1 greedy源；0.30 sampler、GLM workspace/indexer和native字段不能混用。四phase静态fixture的mixed按query-length-one请求在前声明model-input顺序，外部IDs须按该顺序打包；这不是Scheduler/InputBatch重排观测，也不计算模型输出。报告保留0.27.1身份、完整三阶段greedy链和每调用逻辑Roofline计价，不能升级为0.30 runtime、native provider、normal容量或完整Serving验收。原GLM/Kimi/dense解析及旧native拒绝范围保持。


GLM4-MoE既有两层Scheduler fixture的PD+AF默认/显式CPU0.11入口复用已经支持PD+AF的共同AF forward。每步仅执行当前P或D的Attention/FFN pair，target与两次完整drafter均通过同一模块和AF transport；不复制FFN/采样或引入新的传输模型。CLI保留原1P×1D、TP/PP/EP1、max_requests1、batch128、context128、spec2/fixed_count和Midlayer-only MTP范围，不能为迁移成功而改用Roofline。直接共同图可独立检查其逻辑Roofline工作量，但该检查不是Midlayer请求完成或normal容量验收；helper launch缺口仍拒绝，原PD+AF的完整Simulation scope和prefix限制不放宽。


### 普通dense AF共用版本化 forward 与 kernel-only launch 标记（active）

既有A800 BF16 Qwen3/Llama standalone AF的TP1/匹配TP2、PP1/EP-off、MTP-off、同步eager路径复用同一0.11/0.30 forward。AF wrapper显式传递model_execution_version，0.11沿原masked TP embedding、版本化logits边界和greedy，不因放置Attention/FFN到两个pool而使用0.30默认值。FFN与完整hidden/residual双向服务继续用原AF contract、collective和KernelSpec；同输入的非通信计算程序及预测与同版本colocated严格相同。CLI默认/显式选择保持原Roofline/Midlayer方法及analytical/injected策略，不默换方法、补token值或新增launch选择。0.11 MoE AF、graph/async/prefix请求标志及其他未迁移组合保留原边界。

ForwardTrace的launch_policy只绑定计算WorkloadKind；collective、P2P、PD KV、AF activation、memcpy和offload保留其transport owner的属性契约，不携带kernel launch selector字段。未知WorkloadKind明确拒绝。该限制由实际AF injected请求先触发extra launch_policy拒绝后修正验证，不通过放宽AF transfer schema消除错误。完成CPU Scheduler请求、全调用逻辑计价及source storage报告仍不证明正常KV容量、native helper launch或硬件精度；debug块数继续显式标为测试预算。


CPU0.11普通dense的既有A800 PD TP1/TP2 pools与PD+AF TP1也复用同一版本化forward。PD保留原request-scoped KV handoff和按KV-head分片的wire payload，PD+AF保留每步active Attention/FFN pair及同版本colocated计算；未复制模型或改传输字节。默认/显式入口共用版本化greedy及known-ID条件，保留原analytical/injected策略与Roofline/Midlayer方法，所有TP、PP、EP、框架和request配置仍经既有Simulation/route边界校验。CPU0.11 MoE disaggregation、graph/async/prefix请求标志及非A800目标不由此放行；不为未迁移域猜测0.30 provider。完整小请求验证了两个真实CPU Scheduler及PD交接、每调用逻辑价格和逐pool source memory；debug容量与development-only timing标志保留，不能升级normal容量、native helper launch或最终报告验收。


### Qwen PP共用版本化模型和接收生命周期（active）

既有A800 untied BF16 Qwen3的uniform TP1 pipeline及TP2/PP2通过同一Qwen3Model、source算子、ForwardPipelineGroup与lowering执行0.11/0.30版本。模型、embedding、logits边界和greedy显式传递版本；plain-greedy helper保留0.30默认，旧GLM/Kimi调用不改变。0.11 TP2采用其原masked embedding，lowering只在第一stage接受对应embedding，logits/sampler仍只在最后stage。CLI默认/显式选择保留既有pipeline scope、预测方法与analytical/injected选择，不增加模型、并行模式、launch事实、速率或性能公式；graph/async/prefix请求标志及MTP不由此放行。

本地0.11 GroupCoordinator.recv_tensor_dict的CUDA/Torch-distributed分支按hidden_states、residual逐role分配完整backing、接收TP lane、立即fresh all-gather并reshape；旧完整backing在下一个role分配前释放。CPU custom send/recv分支不作为CUDA source。0.30原irecv/postprocess程序则保留两个旧backing直到两次gather完成。共享PP wrapper仅以版本分支表达该源码差异，wire字节、TP lane offset、输出端口和原传输服务不变。直接lifetime检查保留两种释放顺序、完整hidden/residual和非法lane拒绝。三个现有小型Scheduler fixture的完整CPU0.11请求只证明共同逻辑图可执行与全调用计价，不证明真实CUDA/NCCL transport、checkpoint数值、完整worker常驻内存、normal KV容量或硬件精度。


### CPU0.11 MoE AF/PD复用版本化FFN（active）

既有A800 Qwen3-MoE AF wrapper向同一FFN factory传递model_execution_version，0.11保留BF16 gate、独立FP32 cast及FP32 softmax/top-k程序，0.30保留原BF16 route；专家ownership、all-reduce和AF服务由同一模块/contract表达。TP1、以及原validated AF+EP2的静态源图均检查router与专家在FFN pool上；不复制routing/专家公式。默认/显式CLI只迁移既有AF TP1/validated AF+EP2和PD TP1范围，保留显式路由、known IDs、原预测方法、MTP/graph/async/prefix限制和AF+EP2的0.11 Midlayer-only门槛。

PD单rank GQA pool沿既有Simulation单rank边界，不误套仅为TP1/TP2跨rank ownership建立的Qwen/EP2表。此分支限A800、标准GQA、Qwen3/Qwen3-MoE/Llama、P/D两phase、每pool单rank/worker、rank-local且EP-off；多rank仍须原validated表。MoE PD+EP2的完整CPU0.11 Scheduler被原_validate_native_scheduler_scope拒绝，不能把静态ownership支持当成真实Scheduler能力；本轮不改该backend，不为其放宽scope或增加完整请求case。AF TP1与PD TP1完整小请求保留debug预算和development-only timing，正常容量/native/helper launch/完整最终报告仍独立未通过。


### CPU0.11共同forward中的Host KV与容量事件（active）

既有A800 TP1 GLM4 AF＋MTP沿同一target/rejection/proposer和外部KVCacheTierManager执行Host换入换出，不复制模型或DMA公式。CLI保留原受限请求形状、固定接受数、三块HBM/一块Host及hash绑定replay；不放宽PD＋AF、prefix请求标志或Midlayer门槛。proposer几何绑定须保留已有Host层和lifecycle owner，已有HBM的block数、block tokens或每token字节与resolved Scheduler不一致时拒绝，不能覆盖旧cache掩盖冲突。

0.11普通单cache group的block ID被多层共同使用，FullAttention/MLA spec的page_size_bytes是单层页；DSA aggregate spec则已含跨层字节。ALLOCATE/RELEASE统一从该次native KVCacheConfig的全部worker kv_cache_tensors.size之和除以num_blocks取得物理行字节，不能直接使用单层spec页，也不能再无条件乘层数。该账本只消费源码构造的tensor元数据，V4的既有独立生命周期groups和0.30的prepared worker几何保持原分支；不创建模型Tensor，不据此证明normal容量、实际DMA provider或性能精度。


### 同profile实例及既有MoE EP2探针复用共同模型（active）

CPU0.11同A800配置的多colocated实例按每pool复用版本化forward，原router、rank placement、collective、路由输入与预测方法不变。dense Qwen TP1/TP2、Llama TP1和Qwen3-MoE TP1/validated TP=EP2沿各自已有边界；MoE仍要求analytical launch及显式seeded/profile路由，不由此启用injected、混硬件、graph/async/MTP或其他EP配置。每个实例的实际执行都由同一个source accumulator报告storage生命周期，不能用配置了多个pool冒充每pool已执行请求。

H2000.30静态Qwen3-MoE TP=EP2探针也复用已有分片模型、专家ownership和TP logits processor；LM-head-only与显式greedy分别保留，sampler-off只停止在公共logits输出，不暗加采样。该旧probe范围原来把标准GQA默认传成MLA而无法lower，迁移统一使用source的paged-varlen元数据，不把旧失败结果当等价基准。fused embedding必须由caller提供每phase token IDs，未知值明确拒绝，不转回legacy或造ID；TP2 EP-off、其他TP和native/MTP等未迁移域不由此放行。静态所有调用的逻辑Roofline计价不是完整Serving、normal容量、实际H200 provider或性能精度验收。


### 同profile多实例的0.11未决输入端口（active）

既有A800 dense多colocated实例的Scheduler pipeline/depth与async模式复用同一per-pool模型；CLI保留原depth、async、预测方法与router，不复制异步建图。0.11全词表TP1或unfused mask/gather/fill TP2的公开逻辑payload与具体ID无关，允许各实例未决INT32输入继续为None；每个pool的物理rank、graph、KernelSpec、预测与storage生命周期由同一forward产生。源码等价检查覆盖第二个实例的非零物理rank，未知ID与显式ID的上述结果相同；不填dummy或推断模型输出。该许可限同源A800、colocated、rank-local、PP1/EP-off、普通dense、MTP/graph/Host关闭的既有源程序；0.30 fused embedding及MoE异步/其他placement仍保留原拒绝。异步已提交但丢弃输出的终端物理步骤仍执行并计价，不能因不增加逻辑输出token而省掉。正常容量、native/provider和最终Serving矩阵验收不由小请求通过升级。

### A800 TP1 embedding 的 INT32→INT64 转换（active）

新增本地 Torch 2.8 sm80 二进制证据闭合了 vLLM0.11 `forward_cuda→forward_native→input_.long()` 的主机分派和实际 dynamic-cast device entry，见 `microbench/nvidia_a800/references/embedding_id_widen_torch_2_8.json`。单次 contiguous iterator 的适用域为 1≤N≤2²⁸，launch 为 ceil(N/512) 个128线程CTA、每线程4元素、32寄存器、零shared/local/stack；每个有效ID读4B、一次整数SHF符号扩展、写8B。该域可调用既有通用GPU内存、整数ALU与launch解析服务，超出域仍拒绝单launch绑定；不借用同dtype copy的字节数、vt2或BF16/FP32 task-duration floor，不改变HWSpec速率。

该静态绑定只覆盖 A800／vLLM0.11，不能推广到 H2000.30、历史capture二进制身份或实际GPU/provider验收；uniform分派与iterator地址指令未计入，缓存流量沿共同解析模型而非实测hit率。其他版本仍保留无launch逻辑Roofline。该转换计价不等于后续 embedding gather、TP mask、完整请求或normal容量已通过，旧源码拒绝仅在本条明确范围内重开。

同一二进制进一步闭合 BF16 embedding 的 `embedding_symint→index_select(dim0)` 分支：N≤16 且32位地址安全时调用small-index，grid=min(ceil(H/128),SM×8)、block=min(H,128)、24寄存器；N>16使用 IDs.view(N,1).expand(N,H)→gather_out。`91aa3078` 只确认了该分支的generic fallback，遗漏aligned fastpath，不能作为已对齐的large launch证据；新事实由 `IndexKernelUtils.h::fast_gather_kernel_eligible<16>` 和host实际调用共同重开。对连续16B对齐的表、输出、row stride和slice，实际调用 `vectorized_gather_kernel<16,long>`，grid=(N,ceil(2H/4096),1)、block=min(256,32ceil(ceil(2H/16)/32))、24寄存器；原128线程/vt8 generic仅为未满足该eligibility的fallback，当前不开放。共同虚拟embedding保留真实64行对齐的词表行数；当前开放域仍为A8000.11 BF16、H为128倍数且不超过108×8×128、2VH/2NH/8N均小于2³¹的单launch域。

该域small程序每row/thread读取低32位ID一次，每线程恰复制一个payload；large fast程序在payload tail判断之前每个launched thread读取一个8B ID，故索引指令参数字节为8N×gridY×blockX，包括末tile中不复制payload的线程；有效线程各读写一个16B pack，数据读写仍各2NH B。Forward以该load程序调用共同aggregate内存服务，不臆测broadcast或cache命中，也不借MoE冲突修正；整数选择沿原gather抽象计数，完整寻址／uniform指令未建模。Roofline单独保留4NH＋8N的逻辑物化payload，不能将重复load程序当唯一backing容量。该能力仍不是GPU correctness、provider或时延精度验收；TP mask、其他版本和其他materialized gather不因此放行。

### A8000.11 的高级行索引及共享 cast（active）

安装的 `TensorAdvancedIndexingUtils.h::make_info` 明确把Int索引物化为Long；vLLM0.11的 `hidden_states[logits_indices]` 因此保留公开INT32输入、独立INT64临时量和fresh选中hidden输出。共同虚拟forward显式发出通用 `elementwise/cast_int32_int64` producer，复用上述embedding ID转换的同一launch与解析服务，不能冒用embedding来源身份。producer和consumer通过真实虚拟Tensor端口／data依赖相连；lowering拒绝缺少转换、错误操作或非对应输出端口。`query_start_loc_int32_to_int64` 只在0.11分支表示这一已知内部storage，其余版本的原公开source／unbound边界不变，不从本地Torch2.8推断0.30的runtime。

BF16高级行索引的dtype分派、index_stub、gpu_index_kernel和fastpath host调用已核实；其最终调用与上述aligned embedding相同的 `vectorized_gather_kernel<16,long>`，只把allow_neg_indices设为true；当前显式indices仍限in-bounds increasing正索引，原有边界不放宽。共同helper统一launch，复用当前aggregate内存／整数选择服务和限制，不新增速率、测量或性能机制。自动storage生命周期包含新增8N B的Long临时量，不能只补时延而继续读原INT32端口；库workspace、实际allocator和完整normal容量仍独立未验收。

后续BF16→FP32 logits cast的静态launch／2N读＋4N写已查明，但实际sm80指令为PRMT位拼接，而非F2F/SFU。现有独立资料尚未给出其typed issue/pipe服务；不得借普通SHF整数吞吐或FP32→BF16 round service补价。保留Midlayer拒绝，逻辑Roofline不受此缺口阻断；解除条件为适用架构／编译器文档或既有独立机制证据证明对应服务，不要求真实模型执行，也不以production duration拟合。

### H200 Qwen3-MoE TP2 EP-off 的共同静态入口（active）

已有普通0.30 BF16 Qwen3-MoE TP2／EP-off／PP1模型body、TP logits processor和greedy程序由同一forward表达，静态probe默认入口也复用它。sampler=None仍只输出到LM-head／共同logits边界，显式plain greedy才追加FP32 cast和采样；不因迁移补sampler或更改原seeded/profile routing。该LM-head-only模式使用与greedy相同的模型prefix，仅停止在相同source output位置，TP=EP2的既有ownership分支不变。两种输出范围均须显式每phase IDs以决定fused embedding shard ownership，缺失时在共同路径拒绝，不能回退旧builder或造ID。默认及显式phase RuntimeInputs的归一化算子、shape、字节与预测一致；本条不增加模型、硬件、性能机制、normal容量或native/runtime验收。

相同完整事实的Serving API也直接绑定这个既有source，而不只由静态CLI绑定。运行时版本仍限0.30、硬件H200、BF16同dtype权重、普通PP1 eager及显式greedy/seeded或profile routing；逐step IDs继续由执行方提供，缺失时共同forward拒绝。原0.11和TP=EP2窗口不扩展，不选择额外sampler、routing或provider。默认与显式source的图、kernel plan、预测及同forward storage须一致；这只消除入口迁移漏接，不代表真实Scheduler、完整normal容量或最终Serving验收。

### vLLM0.11 MoE TP1 既有部署的共同 API（active）

已有A800 BF16 Qwen3-MoE TP1/EP-off/PP1的单pool colocated、PD和AF两个pool模式已由CLI执行共同forward；程序化API在相同完整版本、greedy和seeded/profile routing下绑定同一个source。角色、route、step IDs和active pools继续由已有forward验证；缺少routing或sampler时拒绝，不能回退旧模型图。AF的router/expert仍只在FFN pool，PD仍只执行当前phase pool；不增加PD+AF、TP2或其他版本范围，不更改通信、kernel计价或容量公式。默认与显式source在实际active pools上的图、kernel plan、价格和storage需相等，normal/provider/真实Scheduler仍分别验收。

AF residual的embedding producer校验同时识别0.11源码的真实padded `source_rows`输入维度，仅在typed端口和`index_select_source=vllm_0_11_embedding`下接受该扩展shape；transfer的输出行数、hidden、dtype、字节、rank和producer端口校验不变。旧无typed-source图仍要求原shape；不能删除真实词表维度去迎合旧校验或把参数输入表大小计入输出payload。

## 用户Flash GPU/FPGA MoE数学边界（active / partial）

2026-10-22用户明确接受GPU向量只计访存的Roofline近似。仅在BW1100的显式`features.gpu_vector_memory_only: true`场景中，GPU向量/标量/SFU工作以已有逻辑读写字节除以HBM带宽计价；混合算子仍保留已有矩阵算量和对应tensor峰值，各数学阶段按原依赖相加。FPGA继续使用用户声明的MM、Vector吞吐/固定延时和HBM/DDR/网络服务，不继承此近似。默认strict配置仍拒绝未知GPU向量服务，不添加或修改硬件速率。完整图的预测必须全调用计价并经共同资源/依赖timeline得出，报告显式标记该近似及未知软件/协议启动开销；该结果不证明native/provider、真实数值正确性、硬件性能精度或normal容量。此授权局部重开下面的GPU服务拒绝，不降低其他门槛。

异构部署在同一虚拟Tensor图中表达GPU DP8与FPGA chip EP16/32，保留逐token整数路由、物理共享NIC及显式DDR读写。FPGA FP8×MXFP4/BF16输出MM使用原4/64/64几何、group128逻辑FP32 activation scales与group32 E8M0 weight scales；独立量化采用用户声明的整操作Vector服务，不能在MM内重复计量化。BW1100普通BF16 A/B、FP32 C投影与已量化FP8矩阵仅消费既有矩阵峰值及typed逻辑字节；显式BF16 gather允许不带launch的逻辑字节Roofline。均不绑定实际compiler、provider、精度等价或原生packed launch。普通GPU向量程序/FP32-SFU速率、控制器启动和协议流量未闭合时，报告具体未计价项与完整时延null，不把部分服务和当完整时延，不把矩阵峰值借给向量算子。buffer仅为显式未绑定容量的片上对象，不能证明normal容量。重开条件为相应源码程序、用户明确的硬件服务规格或独立机制证据，不按production duration/误差选参数。硬件参数见[FPGA配置](../configs/hardware/intel_agmf039r47.yaml)，评估入口为 [evaluate_heterogeneous_flash_moe.py](../scripts/evaluate_heterogeneous_flash_moe.py)。

异构Flash模型source图沿既有DeepseekV4DecoderLayer/Attention，替换Decode的FFN callback以复用同一FPGA专家body；Prefill沿同一decoder及GPU DeepseekV4MoE，TP8/EP-off。二者当前是独立部署图，GPU rank0..7不能被默认为同一物理PD placement。所有层先构造、模型owner保留各层cache到head之后；可选state_observer只暴露实际层对象供后续交接，不是KV transport或完整请求。无H200/CUDA硬件事实时，mHC wrapper保留明确logical_math_only及未知scratch，不选择CU/splits；V4 sparse数学program的decode split_counts必须为None，拒绝native launch声明。纯数学sparse attention复用既有softmax、gather和index ledgers，按逻辑operand角色计字节，不推断split/combine临时、head-pack重复DRAM访问或583/584B source分支的实际选择；packed cache按584B record footprint保留。KV与indexer同样直接复用现有source ledger，缺FP32/SFU/整数容量时据实拒绝，不因缺native launch而拒绝已可表达的数学工作。Grouped FP8数学输出投影按各组计FP8 A/B、group128 FP32 scales与BF16 C，仍不绑定provider或scale packing ABI。以上source图与公式范围不升级完整时延、PD、Scheduler、normal容量或native/runtime验收。

TP8 Prefill的模拟路线显式配置交错NIC ring0/2/4/6/1/3/5/7，保留原TP shard顺序及每两卡共享400Gb/s的NIC负载约束；不再把默认rank ring中未知的同NIC本地传输当作唯一可行路径。该scope固定此analytical route，拒绝改写，其他collective部署不继承；这不是实际library ring/provider观测。BW1100未声明网络传播/软件启动仍缺，公共通信返回的带宽component不能升级成完整collective时延。

TP8 GPU routed expert的`dsv4_mxfp4_w4a16_swiglu_v1`仅复用既有Marlin source ledger中W13→clamp10/SwiGLU→W2/router-weight→topk sum的四个有用数学阶段；每rank intermediate256、group32 E8M0 scales及packed E2M1保留，按声明整数histogram的活跃专家数实例化权重角色字节。矩阵计算是BF16 W4A16逻辑乘积，不用FP8峰值替代；无launch时不制造Marlin tile、SM数、alignment、锁或FP32 scratch。它是保留source的条件数学投影，不是海光实际provider或数值等价结论，typed dequantization/router-weight及完整compiler服务继续未计价。缺FP32/SFU容量时拒绝并列明缺项，不能以部分矩阵/带宽价形成完整专家或模型时延。

异构Flash功能请求采用显式full_logical_arena_replica_copy_v1：TP8中选择一个MQA cache/state副本，完整逻辑arena原始字节复制到独立DP目的storage，保留未使用容量的未知初始化状态；不是推测native页集合、最小传输或实际connector。跨步复用原storage/token对象，物理Prefill/Decode/FPGA ranks分离。现有真实CPU0.30 colocated TP4 Scheduler仅作为请求长度控制驱动，物理TP8 Prefill及DP8/FPGA EP图分派独立记录；不放开原Flash native PD bridge的拒绝。固定功能时钟不能作为零延迟，完整时延/QPS在缺GPU服务等事实时保持null。仅对应功能请求/source依赖完成，不升级完整Roofline计时、正常容量、native/provider或最终gates；请求调度实现及范围见[异构请求评估](../evaluation/serving/metrics/heterogeneous_flash_request_sweep.py)，命令参数由 [CLI](../scripts/evaluate_heterogeneous_flash_moe.py) 的 `--help` 提供。

## Flash异构有效利用率场景（active / user-declared）

用户要求与100%基线同setting对比GPU算力/HBM66%、FPGA算力/HBM/DDR80%。通过aggregate Roofline HWSpec的显式有效速率副本表达，不修改canonical物理峰值，不按总时延缩放。FPGA保留400MHz、MM4/64/64与Vector32元素的有限波数，波数的吞吐服务按有效compute容量延长；35cycle Vector启动及250ns HBM延时不变。网络400Gb/s、20μs、共享NIC拓扑与所有容量不变。两个场景独立重算共同依赖/资源时间线，禁止跨利用率复用结果；历史无utilization字段的原runner输出仅代表原100%基线。该用户假设不属于硬件真值、native/provider或normal容量验收。

### V3批量padded MTP沿同一虚拟forward（active / source projection）

vLLM0.27.1的既有two-draft调用链保留每请求完整verify+bonus行，首draft仍执行全部target行，递归draft每请求一行；target/rejection、buffer preparation、首pass与递归pass均共用原模块。`llm_base_proposer.set_inputs_first_pass`对packed IDs整体左移后按每请求accepted位置scatter，不改成逐请求shift，不丢弃拒绝行或未知stale行；draft输入ID与接受前缀仍为显式独立scenario。每请求逻辑KV arena同时容纳target与递归位置，全接受时先保留新增行再安排下一请求；递归沿同一cache backing/slot producer，不能覆盖下一请求context。普通causal Triton MLA的threshold保持1（其源码仅non-causal DSpark分支提高threshold），多query使用已有context-prefix/suffix/merge；FlashMLA仍要求uniform queries<=128及显式BATCH_INVARIANT=False。批量context仅接已有单chunk域，超过可证明workspace、混合未知context或不完整metadata继续明确拒绝，不推断native页面、provider、实际采样值或normal容量。此接线解除原B1 caller限制，不升级真实Scheduler请求流、所有模式迁移或最终验收。

V3普通target沿相同context backend接通多请求非空prefix及decode-prefix＋context-prefill组合；逐请求context起点由显式logical slot推导并校验完整不重叠arena。原causal suffix、BF16 prefix gather/扩展、prefix attention及state merge均保留；packed logits只选择各请求末行并沿0.27.1 greedy source采样。不把多请求context伪装成fresh prefill，单chunk workspace边界和混合未知/空context拒绝保持；该source迁移不证明真实Scheduler、provider或normal容量。

V3的0.27.1 padded caller可复用既有`V32PaddedMTPInputs`载荷和共同Serving逐step绑定（历史类型名保留，不新增第二套schema）。current frame与原静态draft-ID tuple互斥；typed frame按当前request顺序、首pass全部query行、后续每请求一行、两pass深度及同一cache block size校验。帧映射不可变，缺当前step、未知IDs、重复owner或过期shift输入均拒绝；每step仍须显式传当前target IDs，不把run初始帧当持续的真实草稿。V3与GLM继续保留各自0.27/0.30 target/draft采样源码，复用的只是输入载荷/调度接线。连续两次共同engine执行证明逐步输入被消费、全调用计价及同forward存储导出，旧builder/memo设置tripwire；不作为真实Scheduler MTP请求流或normal/provider验收。

### SGLang三段padding的causal空query数学域（active / partial）

0.5.19 `FlashMLABackend._apply_decode_target_verify_metadata`给target-verify的seq_lens加固定3-token窗口，graph dummy原seq_len=1因而为K=4；draft-extend graph则直接使用fill后的K=1和q=3，不能统一假定成fresh q3/K3。`eagle_draft_cuda_graph_runner.replay`在padding时清零topk_index/hidden/positions；extend runner只覆盖active input/hidden前缀，padding token IDs需保留显式source事实，不能借draft的清零语义补值。三段的完整buffer/capture接线尚待实现，不以此数学域通过整链迁移。

`bf16_mla_causal_empty_rows_softmax_v1`是同一MLA variant的显式无provider数学程序：bottom-right causal mask令q>K的前q-K行没有可见key；有限行数min(q,K)，有效pair为min(q,K)*K-min(q,K)*(min(q,K)-1)/2。只对有限行计reciprocal/log，保留全部query读、O写和FP32 LSE写；理想空行状态仅作zero O/negative-infinity LSE数学定义，不作为native输出/指令或checkpoint证明。原普通MLA程序仍拒绝q>K；虚拟SG graph例外仅在显式source、dummy请求域、reserved64 cache页和q1/seq1或q3/seq1/4范围生效，真实请求仍须q<=K。kernel/provider、masked tile冗余计算、normal容量与完整三段padding仍分别未通过，不按误差/时延选参数。

### SGLang underfilled EAGLE复用完整三段forward（active / source projection）

topk1/steps2保留同一target与NextN，仅对显式配对token-row catalogue选择满足B及3B的最小bucket。既有`SGLangEagleChainState`新增可选target_graph_inputs/extend_graph_inputs，复用`SGLangGraphInputs`的retained-ID载荷；两段均须显式提供，不从draft的zeroed topk_index推断另外两段tail。exact bucket拒绝多余tail状态，未填满bucket缺任一段状态、长度不符、请求顺序或词表不符均拒绝。CLI递归解析同一dataclass字段，不增加第二套JSON schema。

同一source执行draft B'、target3B'、extend3B'，其中B'为capture request batch；采样/tree verify和后续greedy只消费真实B行或3B行，公共capture_sizes仍为model-input token-row单位。各runner的静态输入buffer、active prefix copy、源码明确的zero/fill及unknown retained tail分别表示；BF16 hidden zero-fill仅在steps2的typed materialization程序开放，仍只计公开写字节。target/NextN各自预留64行匿名dummy page，active逻辑arena整体平移，dummy slot0不与真实history重叠；target dummy q3/K4与extend q3/K1使用上一裁决的数学域。三段各有完整named output与source stream barrier，Host分别replay，各图外准备/采样保持eager；不合并成一个graph dispatch。

完整61层TP8 B2→capture B4的共同engine、Host分段和B1→capture B2公共CLI小请求均已直接执行并全调用计价，旧builder/memo以tripwire拒绝调用。此为source-mirrored Scheduler与显式synthetic token/acceptance/retained输入，容量仍为debug块数；不能升级实际SG CUDA graph catalogue/provider、数值checkpoint、完整startup/normal容量、硬件时延或最终Serving验收。


### H200复用声明的0.11 dense模型源，不借用sm80 launch（active）

原标准report从prediction配置传入Scheduler0.11，即使H200 capture选项为0.30，也不改变该执行合同。既有BF16 Qwen3/Llama colocated TP1/TP2、PP1、EP-off可复用同一0.11 CUDA模型、masked TP embedding、capture、logits和greedy源；普通API的pending声明也允许已支持的decode/full-and-piecewise capture，仍由deployment和共同forward校验硬件及模式。保留原midlayer/injected、prefix请求标志、async和pipeline depth，不把capture身份冒充模型版本或实际runtime。C500、Ascend、H200 MoE和H200 FP16/PP扩展不由此放行。

TP1 embedding的Long materialization与BF16 selected-row读写，以及高级行索引的INT32→INT64 producer，具有独立的typed storage。H200只计源码已确定的逻辑字节：cast读4B写8B/元素；gather读写各一份BF16 selected payload并读一份INT64 indices，capture按物理padding行数执行。高级索引的`aten_advanced_indexing_long`身份仅表示逻辑程序，没有sm80二进制含义；lowering保留其producer/data依赖，KernelSpec必须无launch且unsupported。相同显式unbound contract可走Roofline，Midlayer、resource-input与伪造launch继续拒绝；A800原Torch2.8 launch及精确几何护栏不变。未修改HWSpec物理速率、未新增duration参数；完整原60case、normal容量及硬件/provider门槛不由本条通过。


### C500 dense复用共享模型并独立绑定vLLM0.24源码（active）

既有C500 BF16 Qwen3/Llama colocated TP1/TP2、PP1/EP-off/MTP-off复用同一虚拟模型和基础算子，声明`vllm_0_24_0`及对应sampler/embedding/logits身份，保留原prediction的CPU0.11 Scheduler、depth/async、prefix请求标志、FULL_AND_PIECEWISE和analytical/injected策略。源码参考复用rope_dispatch的v0.24.0_c500_sdist与vllm_metax_0.24.0既有owner：vocab_parallel_embedding.py:472–491和mask helper:168–193；gpu_model_runner.py:4366–4367的高级选行；logits_processor.py:75–104全量all-gather；sampler.py:95–109,138–146,240–271的FP32 cast、INT64 argmax及fresh INT32输出。模型source0.24不声称实际设备runtime，也不把Scheduler0.11伪称为C500设备源码。

TP2保留mask→INT64 gather→in-place fill→AR，不使用NVIDIA fused embedding；未知token值与capture tail保持未知，其公开payload由shape确定。高级选行保留INT32→INT64 producer与typed依赖，逻辑ATen cast无launch并可计12B/元素的Roofline，不能借用A800 sm80二进制。新0.24 KernelSpec身份只接受C500；原其他版本/硬件及MoE、FP16、PP、retained fused padding不由此放行。模型体进入capture，选行/head/采样在图外；TP1捕获也必须传递按硬件和rope type选出的真实程序，C500 Llama3的native composite不得退化成默认fused RoPE。

同输入默认/显式共同forward、TP1/TP2图与storage、完整短CPU请求及原injected方法的拒绝边界分别验收；源码可确定的逻辑工作不因native launch缺失而改成旧图，但原Midlayer/normal/provider门槛仍保持。没有修改任何物理速率、网络、duration参数或误差阈值。

### Ascend0.18 FULL FIA的显式padding数学域（active / partial）

共同图迁移不得改变已有且适用的计价分派。injected launch下，FULL FIA的decode物理query逐项为1时，复用既有CANN regular FIA source selector和原预测模型；正KV域保持原接口，一个独立zero-KV尾行使用显式`ascend_0_18_full_fia_decode` source contract。该合同只允许保留真实正KV请求前缀与最多一个Q1/K0尾行，必须绑定910B BF16 CANN typed source program；不能把零改成正数。其他非均匀query、fresh-prefill及未绑定ABI仍保留显式数学域，不借用Q=1发射。类似地，supported gather必须继续执行其既有launch/Forward模型，materialized_output=True仅描述写回语义，不能作为切换到纯字节价格的条件；无launch的unsupported物化继续保留原显式解析服务。此修正不新增速率、duration或误差参数，不把legacy命中等同于实际provider验收。

CANN8.5.1既有FIA源码身份沿`microbench/ascend_910b/references/fused_infer_attention_score_cann_8_5_1.json`唯一owner复用。`fia_kernel_nonquant.h:283–339,462`在带mask的TND/NTD路径先初始化完整Q输出域；0.18 attention metadata始终提供split-fuse causal mask。zero-KV尾行在静态split-core中无score/value task，仅保留初始化写出；真实正KV行随后写出计算结果。初始化字节覆盖全部physical rows，计算写出只覆盖positive-KV rows，DRAM distinct extent取二者的并集。不能要求两种写出都分别覆盖全buffer，也不能把初始化专有尾行从unique写出里扣掉。源任务的Q/K/V读取仍只覆盖真实请求前缀，尾行无虚构输入读取；该独立受限域的padding已被source program完整表示，不再因padding统一禁用原生pipeline。未知workspace、实际provider和其他多行尾部仍独立未通过。

多行zero-KV尾部以及fresh/nonuniform FULL TND仍为native unsupported，显式数学程序只能说明其解析工作量，不能当作原生FIA执行服务已绑定。重开原生点预测需要匹配版本的host tiling源码或不含duration的真实tiling payload，证明actual-Q/KV数组、TND与KV存储布局对应的runtime key、M/S2 tile、worker endpoints及输出初始化规则；不能把仅用于Q1的tile规则直接用于更大的physical max-Q。既有tiling_reflection_runner仅反射字段布局，不能生成所缺的选型事实；tiling_capture_runner的live分支以及GEMM tiling query helper均会选择设备和分配设备内存，不得冒称CPU-only替代路径。缺少上述事实时保留解析价格及其精度限制，不用旧shape价格、虚假正KV或实测时延残差补齐。

安装SDK还提供真实CPU编译入口`tbe.common.utils.op_tiling.do_op_tiling`，最终调用`libregister.DoOpTilingForCompile`；`host_tiling_query_runner.py`仅构造描述符/host长度常量并调用此入口，不初始化设备、不分配设备内存或执行kernel。因此不能仅因安装树缺少host C++文件而判定没有可行动的分块查询路径。SDK prototype的inner_precise默认1与已安装Torch NPU `.out` schema默认0不同，必须显式保持调用方参数身份；CPU查询产出的离散key/分块/任务参数是compiler candidate事实，仍需核对compile与OpAPI调用条件并接入相应source程序，不自动当作原生Serving/provider已通过。

本地vllm_ascend_plugin的worker/model_runner_v1.py::_pad_query_start_loc_for_fia在configured FULL下保留真实请求边界，非uniform模型行数追加一个独立dummy请求；exact bucket非uniform时仍追加q0/K0。attention/attention_v1.py的PrefillNoCache分支使用query边界，因此fresh dummy为q=K=padding；cached分支使用seq_lens，因此dummy为q=padding/K0。不能把padding加入最大真实query或改成多个q1请求；真实KV producer只写有效行，fresh FIA直接读全物理Q/K/V并保留前置cache更新的控制依赖，cached FIA读cache。

显式`ascend_0_18_full_fia_softmax_v1`只在910B/BF16/causal/FULL域保留该有限softmax数学及全物理Q/O payload，q>K空行不制造exp/reciprocal，原CUDA数学域的正长度及q<=K约束不放宽。full_graph_fia使用调用者的主output和独立BF16[1] softmax_lse作为同一.out调用的两个写目标，不新增主output拷贝或计算kernel；标量placeholder只计2B公开输出，不推断逐行LSE数学。未知library workspace保留`symbolic_workspace_complete=False`，无launch/unsupported继续拒绝Midlayer，不代表0.18模型入口、完整请求、normal容量或provider验收。未改HWSpec速率或用duration选公式。

普通BF16 Qwen3/Llama colocated TP1/TP2、PP1、EP/MTP-off沿同一虚拟模型绑定`vllm_0_18_0`、`vllm_ascend_eager`及0.18 greedy/embedding/logits身份；原CPU Scheduler0.11、async/depth、prefix标志和method/launch策略独立保留。FULL body包含真实模型padding域，选行/head/采样仍在图外；NONE复用原eager模块。TP1 embedding.long分配INT64，TP2无新增词表mask保留既有Ascend表达式；目标选行保留公开INT32 indices，不据CUDA ATen源码推断NPU内部Long物化，未知CANN workspace不记成零。ordinary NPU communicator实际继承base_device_communicator.py的dist.all_reduce(input_)并返回同一input，embedding、attention o-proj和MLP down-proj均显式沿该原地storage程序；不采用parallel_state注释中的out-of-place说法，也不推断compiler functionalization。all-gather仍为fresh rank-major及必要多row reorder，FP32 cast→INT64 argmax→fresh INT32输出与0.18 sampler一致。新增0.18 public KernelSpec只接受Ascend、BF16对应边界及无launch/unsupported，不放行其他版本、FP16/MoE/PP或provider。模型身份不冒充已安装NPU runtime，Roofline请求及直接contract不代替原60case Midlayer、normal容量或硬件验收。

### A8000.11 Qwen3-MoE capture复用共同forward（active / partial）

既有validated TP=EP2、PP/SP/CP/DP1、BF16/MTP-off域将FULL/PIECEWISE的physical model rows传入同一Qwen3MoE、FP32 routing和expert模块；每层histogram必须守恒physical_rows×topk，FA/KV与图外logits选择仍使用logical请求边界。v0.11.0_a800的qwen3_moe.py:162–180不在SP1下裁剪hidden；forward_context.py:352–357在DP1不创建dp_metadata，fused_moe/config.py:603–604及layer.py:2001–2025的all-to-all要求DP>1。因此该域各EP rank接全部physical行、计算owned experts后规约，不凭padding添加EP all-to-all或active-token裁剪。layer.py:603–623和fused_moe.py:1586–1589的普通expert结果原地写入输入，此alias在所有0.11共享Qwen MoE source保留；未知workspace仍独立未闭合。

捕获节点和collective ID沿同一公共段映射，FULL覆盖完整模型，PIECEWISE仅FA/KV eager，采样在图外；超过最大bucket按原eager规则执行。0.11 masked TP embedding的shape-only语义允许未知ID，不虚构retained padding值；声明的routing profile/seeded分布按physical行数实现，不升级为逐步checkpoint观测。CLI新增范围只覆盖原A800标准MoE case的device_only及depth1/2、async和既有analytical/injected模式；这不是其他Host模式或更深流水被源码排除的结论。其他硬件、版本和原Midlayer/normal/provider验收仍须独立闭合，不修改速率或借用时延。

原H200标准prediction同样声明CPU0.11 Scheduler；其BF16 Qwen3-MoE TP=EP2/PP1/SP/DP1沿相同0.11 CUDA模型、masked embedding、FP32 routing和原地expert程序，不把采集0.30身份当作执行0.30。上述logical/physical行、histogram、capture段和图外target边界共同复用；只将显式FP32数学topk合同扩到H200，仍无launch/unsupported，不借A800编译器二进制或更改公式、速率。NONE及超过bucket的eager执行同一shape-only masked输入程序，配置未启用capture不成为绕开injected selector或省略未知ID工作的理由。CLI仍只接既有colocated TP=EP2请求域，不放开H2000.11 TP1/EP-off MoE或其他vendor源码；原normal/Midlayer/provider门槛独立保留。

### C5000.24 MoE公开数学与native内部保持分界（active / partial）

`v0.24.0_c500_sdist`的fused_topk_router.py:17–32、69–111和_custom_ops.py:2310–2328证明BF16 gate直接进入topk_softmax，公开返回FP32 weights及两个INT32矩阵，renormalize由融合op参数传递；没有Python FP32 logits物化或0.11的外部sum/divide张量。当前sdist及已有Git对象没有对应native C++实现，不能据此继承0.30的R×K尾乘、每行倒数、positive-sum分支、expression sites或零workspace。`vllm_0_24_bf16_softmax_topk`仅配`bf16_softmax_topk_public_math_v1`：复用stable softmax数学owner，选中归一化按每row K−1次数学sum与K次数学divide表示，不绑定native除法指令或恒1乘法。公开unique typed读写为2RE+12RK；native尾部、比较/清理指令和workspace显式未知，不能以数学调用计价通过升级native/provider或normal容量。

普通BF16 Qwen3-MoE validated TP=EP2、PP/SP/CP/DP1、MTP-off复用同一Qwen forward/physical capture与routing resolver。MetaX customized/layers/unquantized_fused_moe_method.py:65–98进入core modular_kernel.py:1396的empty_like，因此expert结果fresh；Runner:434–453的末尾AR及Maca继承的CUDA communicator:299–311也为fresh，与0.11的expert alias明确分开。DP1没有naive dispatch裁剪（Runner:719–721），全部physical行参与gate/route/expert/AR；FA/KV/采样仍按共同logical边界。保留model0.24、CPU Scheduler0.11与原profile/async/depth/capture/injected身份，不以源码数学构图替代原60case Midlayer或容量验收。


### Ascend0.18 Qwen3-MoE共享数学body和custom dispatcher（active / partial）

原BF16 Qwen3-MoE validated TP=EP2、PP/SP/CP/DP1、MTP-off沿共同Qwen3模型/physical capture，配置FULL复用0.18 FIA的dummy边界；0.18本地core缺qwen3_moe.py，因此这是已适配共同数学body与已核实plugin算子的组合，不宣称完整原生模型逐行绑定。模型/plugin0.18与真实CPU Scheduler0.11分开。experts_selector.py的out_flag=False路径写BF16 weights与INT32 IDs；第三FP32 score窗口不写，但host allocation未知，不能替换成CUDA第三INT32 token-index矩阵或记成已知空分配。该路径用显式公开softmax/topk数学合同，缺native比较/调度/内部workspace继续未绑定。

token_dispatcher.py的AllGather dispatcher先做INT32 expert_map[ids]、!=−1的BOOL结果及fresh BF16 weights乘mask；这些依赖和临时输出独立记录，只计一次。静态linear map由0.18 core determine_expert_map生成INT32[E]；EP2每rank仍进入全部physical rows，由active_expert_range过滤本地专家。custom init的逻辑active_num=T×K窗口与INT64 local-expert count分别计typed payload；count长度E/EP，.to(int64)为alias。它不调用仅按E/H选择的full-range V2+count-cast recipe。W1/activation/W2继续复用同一expert owner；init、abs与unpermute显式无launch/unsupported的数学payload，BF16概率读为2TK，K项加权和不凭zero mask猜测native skipping；逻辑窗口不等于host完整分配或真实DRAM。expert结果fresh，ordinary NPU末尾AR原地修改这个结果。未知扩展buffer/workspace与第三score分配继续阻止normal/native验收，不以公开算术计价替代原60case Midlayer或容量通过。


### 既有A8000.11 Qwen PP捕获的阶段端口（active / partial）

原untied BF16 Qwen3 TP1/PP>=2或TP2/PP2、EP/MTP-off沿同一stage forward执行FULL单token decode、PIECEWISE prefill及超过bucket的eager；CLI保留原sync/depth1/prefix-off和method/launch约束。gpu_model_runner.py的_preprocess/sync_and_slice_intermediate_tensors（0.11:1768–1791,2033–2036）证明接收hidden/residual后、模型执行前，两者分别复制到worker-owned active buffers；旧共享PP图遗漏这两次copy，现0.11与0.30普通dense PP均补齐。stage0捕获INT32 input IDs、INT64 positions/slots，后续stage直接捕获BF16 hidden/residual与positions/slots；中间stage同时保留hidden/residual输出，最终stage只保留hidden。通信、接收gather和runner copies在模型捕获外，目标选行/head/采样同样在外，不伪造后续stage embedding。body和传输按physical rows，KV写入及目标选择按logical请求，TP lane byte数和接收端copy依赖由同一tensor views推出。工作buffer只描述active prefix，未知max_num_tokens完整backing/allocator reservation保持未绑定；不据此通过normal或native replay。

A8000.11 embedding的静态编译launch只覆盖已有对齐/寻址域，复用hardware launch owner的同一predicate。域外的合法BF16/INT64 index_select仍有确定的公开payload：每selected BF16 row读/写一次及Long IDs；以显式unsupported/no-launch保留Roofline，不强选该编译launch。域内仍强制exact原launch，域外Midlayer仍拒绝，不放宽实际provider合同，不改速率/参数。纯数学支持与实际launch证据分别验收。

### GLM0.30既有顺序MTP深度复用（active / partial）

`llm_base_proposer.py:615–640,684–775`先执行首轮完整输入；只有checkpoint的`index_share_for_mtp_iteration=True`才沿`:584,612`的guard压缩共享top-k并在后续pass跳过indexer。False时不产生该gather/copy，后续每轮重新执行完整indexer与选择，并消费当轮KV长度。深度1在采样后直接view返回，不读取/物化后续位置与反馈hidden，不回滚seq-lens，不生成推进cast/slot更新或stack。多轮只在进入循环前回滚一次seq-lens，随后每轮推进位置、slot和seq-lens，回填前轮draft IDs及反馈hidden，再复用同一NextN权重；共享index取决于上述明确flag。虚拟source以显式独立draft-ID元组长度绑定深度，并要求每request的padded proposer深度一致；不从配置或时延猜测token值、接受率及constructor事实。decode-prefix和DSA分界统一沿`1+num_speculative_tokens`推导，原uniform decode query限定保持。逻辑cache为所有明确后继写入预留足够页，constructor cap不容纳时仍拒绝；深度1不虚构下一行。深度1/2/3的数学source不代表真实Scheduler可达性、GPU/provider、normal容量或任意混合proposer subset均已验收；旧mixed disabled/proposer1组合仍须独立核实。

### 既有AF服务按worker分段捕获（active / partial）

A8000.11 BF16 dense Qwen/Llama的standalone AF匹配TP1/2及PD+AF TP1复用同一模型forward；Attention与FFN按本地计算段记录`midlayer_af`边界，完整physical hidden/residual消息在捕获外，KV更新与目标采样仍只消费逻辑请求行。该分段是仓库既有AF服务的解析执行契约，上游vLLM不提供该AF runtime；不把跨pool服务包装成单GPU图或声称native capture。各rank source调用顺序复用同一`operator_order_by_rank`接口，TP按collective group的canonical IDs归一，Host KV外层DMA保持capture外且不偏移原trace调用索引。Host replay区域从实际节点绑定worker pool，不能把FFN记到Attention Scheduler的pool；barrier控制边不冒充两份activation payload。CUDA FULL只适用于uniform单token decode，mixed/prefill沿PIECEWISE；完整消息字节及active KV范围保持，Host KV DMA保留原block范围。CLI保留sync、depth1、prefix-off、原method/launch和所有容量门槛。该source迁移不提升normal、实际GPU/native provider或原60case验收。

### K2.6单rank共享forward及0.30 MLA捕获行域（active / partial）

既有K2.6 BF16/mixed-int4的TP/PP/SP/CP/DP1、EP-off在A800/H200复用同一Kimi模型、typed MLA与专家模块。A800仅声明`source_projection_only`及专用Kimi scope；数学Roofline不赋予无launch路由native执行能力。FULL/PIECEWISE及overflow eager共用模型体，选行/head/采样在capture外。GLM的未支持capture仍在其模型边界拒绝，不能因共同MLA输入合同扩大而自动放行。

0.30 `gpu_model_runner.py:4335,4399,2459–2465`在FULL将attention metadata的`num_actual_tokens`本身设置为padded rows；`mla_attention.py:938–950`的slice因而不裁掉FULL尾行。FULL的Q absorption、concat、latent公开输出/LSE backing和V-up保持physical rows；PIECEWISE的attention eager metadata仍用logical rows，Q/V absorption在该域裁剪。O-projection、decoder/router/expert始终沿physical模型输入，KV写入和采样沿logical请求。FULL dummy seq-lens为零且slot为−1，数学attention只含真实有效cache窗口；公开storage明确保留尾行但其值、native空行指令/流量/服务未绑定，不以零填充或虚构历史补足。V3独立构造器显式初始化无capture状态，保持其0.27.1路径。未知allocator/workspace及provider不由上述数学source验收消除。

### 静态模型探针的输出范围复用（active / partial）

已有target-off探针的`target_sampling_source=None`只要求选行与LM head，不能迁移时暗加greedy。GLM/K2/V4的共同生产入口明确保留该范围，GLM独立decoder API仍不包含head；V3沿既有0.27.1 logits-only路径。旧`vllm_0_30_plain_sampler_bf16`范围仅额外物化FP32 logits，不含argmax或完整随机采样，复用同一greedy primitive的cast阶段，不虚构概率、噪声、token值或runtime。完整plain greedy仍执行cast、INT64 argmax和INT32输出。上述无采样范围只接受显式eager/non-MTP source facts，不使不完整Serving请求或native provider通过；探针fixture的constructor/workspace/IDs必须明确是独立合成场景。

V4原显式`vllm_0_30_triton_top6_pure_text`声明只绑定nonhash层，旧图对hash层留下无source的片段；不能将调用方声明改写成modular。共同forward保留该输入声明，按`deepseek_v4/nvidia/model.py:846–857,995–1010,1022–1029`的checkpoint hash层和modular模型分支，复用已支持的hash-table路由与normal Triton路由，各node分别标明实际源码数学身份。新增已核实的hash数学依赖不意味着原triton片段完成了native/provider验收。

### Kimi FlashMLA的uniform分界及共享context后缀（active / partial）

0.30 `flashmla.py:113`的reorder threshold为128，`MLACommonMetadataBuilder`沿`split_decodes_and_prefills(require_uniform=True)`处理连续相同query长度的decode前缀。因此Kimi源码profile不再复用DSA的threshold1/nonuniform分界；显式backend阈值与uniform要求由同一输入重排helper表达，不伪装成127轮投机。原caller `(q4,q1)`进入4行absorbed decode及1行context-prefill；后者分别展开当前suffix和已有prefix再merge，不能继续声称全5行decode或只读suffix。GLM仍沿其既有DSA阈值，普通GLM/Kimi的short-extend需调用方显式`symbolic_source_prompt_lengths`，不能按Scheduler phase猜prompt。

context cache窗口按实际slot映射恢复arena起点，按区间验证不重叠，不能把重排后的KV长度累加当作原arena地址。小query可能属于uniform decode，真正prefill/chunk直接域使用query>128；两种场景分别保留。新增0.30多query数学profile不宣称实际BATCH_INVARIANT选择、launch、provider或normal容量；TP1、既有TP8/EP8及pipeline仅放行对应明确source投影。

### V3 fresh批量MTP复用（active / partial）

0.27.1 `gpu_model_runner.py:5209–5223`在fresh prompt的`spec_metadata=None`只采样每request末行，不构造verification rejection/prefix；`llm_base_proposer.py:843`及后继loop仍对全部target行运行第一pass，后继每request一行。共同V3 source将fresh和verify目标边界分开，复用相同NextN循环、cache和RoPE，独立draft IDs仍必须显式声明。后继位置由完成上下文减rejected推导，fresh q4不错误退回位置1；raw0保持0。FlashMLA与Triton是各自显式数学source，局部完整图计价不代表native或真实请求流通过。

### 旧建图入口退休及保留的静态scope（active / partial）

Serving与模型探针统一调用共同虚拟forward，旧手拼builder、BlockConstructionMemo及两个fallback删除；执行缓存仅保留kernel_lowering/kernel_prediction，reference模式两者为None。完整Serving需已有绑定的source和明确运行输入，未绑定输入不再执行缺失来源的另一套图。构图scope、真实Scheduler和normal容量仍分别验收；不把静态fixture的最小constructor或source全价冒充实际请求。

原探针合法的H200 V3 TP1复用同一0.27.1模型和完整维度，无collective；embedding的无条件`.long()`复用既有ATen逻辑cast，无其他版本A800 launch。GLM4 TP1静态范围复用准确0.11 source，纠正旧探针混用0.30/MLA的身份；None、FP32 logits preparation和greedy分别保留，0.30采样声明不改写成0.11。新增H200 GLM4路由仅为既有数学source的软件admission，HWSpec物理速率不变、launch仍未知。其他原本非法TP/EP组合不因退休而放行。

V4旧attach helper的synthetic frame不等于自动trace的生产对象；相关反例保留为独立comparator/parser合同，真实FP32 view、persistent状态和C128两阶段临时另以生产primitive对独立reference核查。APE/gamma/destination-slot/active-row/cross-step lineage若未导出，继续明确未绑定，不把synthetic声明升级为生产或normal验收。

### GLM mixed单轮proposer的完整物理输入（active / partial）

0.30 `gpu_model_runner.py:5252–5288`的padded proposer保留全部incoming请求；`llm_base_proposer.py:849–870`对完整target rows做shift并scatter每request的保留行，K1在`:640–652`完成全batch采样后返回。fresh或显式完整preemption-replay请求的`mtp_proposer_mode=disabled`只表示不消费其proposal，不允许从首轮forward删除它的模型行；多轮`:675–769`仍保持完整batch，每后继pass每request一行。共同source要求完整typed incoming request IDs和独立draft输入；raw输出计数0保持0，不解释成未知、不补1，也不把物理候选当成已消费proposal。原旧builder的proposer-consumer subset因此少计的drafter工作属于有源码依据的差异。同步源的正常非recompute decode只包含一个pending sampled token和明确draft suffix，按`scheduler.py:656–660,813–823`保留原terminal行守恒；不将该同步限定推广到通用ScheduleStep或其他框架。既有非uniform speculative attention域仍拒绝，源码投影不宣称真实Scheduler已生成该组合。

### C500 index_select 的观测行数与源码索引域（active）

同版本 Torch 二进制只读复核见 `microbench/metax_c500/references/embedding_index_select_torch_2_4_maca_3_7_2.json`。BF16 caller 的三次 canUse32BitIndexMath 检查与16个索引分界决定分派；8192只是既有launch记录覆盖上限，不是源码阈值。显式连续source_rows、BF16表/输出和INT64索引的字节extent均小于2^31时，使用已存在128线程、small/large每CU32/64 CTA封顶公式；该字节条件是比源码元素offset更严格的充分域。缺source extent仍保留旧观测域，宽索引和strided-source扩展不放行。native RoPE由同一虚拟forward携带真实连续cache行数，不能以selected rows代替source extent。未改速率、工作量或正常容量；本项不是GPU/provider或完整请求验收，不消费production duration。

实际原C500 Llama QPS16复跑进一步定位到 rows=23072、hidden=128、index_count=0、scattered=False：该首个拒绝是native RoPE的cat物化，不是indexed gather。共同forward保留cat与空suffix-cat的明确source_operation，只有C500、完整旋转宽度、injected launch的既有native帧绑定materialization_source；cat复用现有MACA_CatArrayBatchedCopyNoPartialWrite的128线程、每线程8值发射规则，index_select复用上面的同版本规则。KernelSpec区分两类typed BF16物化，predictor通过公共selector重新核对exact launch，伪造grid和其他硬件仍拒绝。图的调用、storage和逻辑读写不变；不把cat当index_select，不给其他materialized gather或analytical launch批量放行。直接完整native RoPE帧17/17调用已可Midlayer计价；这不是原128请求case或normal容量通过。

### C500 小向量 INT32→INT64 物化（active / partial）

同一既有Torch2.4+MACA3.7.2库的host copy分派、CUDALoops_copy.cuh、xcore1000代码与ELF元数据见 `microbench/metax_c500/references/embedding_id_widen_torch_2_4_maca_3_7_2.json`。连续INT32→fresh INT64且1≤N<2048时，不依赖可选优化开关，选择256线程、每线程4slot、ceil(N/1024)个CTA、22个thread寄存器且零private/shared的generic unrolled程序；每有效元素读4B、ASHR_I32一次符号扩展、写8B。普通0.24 embedding及同类型公开选行转换复用既有整数ALU/访存模型，不引入速率或算子duration；寻址、uniform和dtype分派指令仍排除，不宣称完整ISA计数。N≥2048有源码明确的优化分支及环境开关，不能凭generic符号存在就外推；缺分派事实的大N域、strided/split和其他版本仍拒绝。编译库声明不冒充captured0.24 provider身份，完整请求、正常容量和GPU时延仍分别验收。提取工具曾把ELF副本写到输入库旁，该副本已清理，原库哈希复核未变；没有GPU执行。

### C500 TP1 token embedding 复用 index_select（active / partial）

0.24 `vocab_parallel_embedding.py:77–78,472–491` 的普通 unquantized TP1 分支先 `.long()` 再 `F.embedding`；二维连续词表按 dim0 index_select 物化，不是 runner 的高级索引。共同 SymbolicEmbedding 显式携带 embedding_source/source_operation 及真实64行对齐的词表 extent；仅 C500、injected launch 与该明确 source 身份复用既有 `pytorch_2_4_maca_index_select` 服务及保守32位寻址域。读写、storage 和速率不变；不把其他版本、TP分片mask、缺词表extent或高级索引一起放行。直接验证同版本合法图可Midlayer计价，其他版本未声明的物化仍拒绝，原native RoPE服务保持可用；这不代替完整请求、normal或实际provider验收。

### C500 单行 logits 高级索引（active / partial）

同版本头文件、host分派和bundle95编译元数据见 `microbench/metax_c500/references/logits_single_row_index_torch_2_4_maca_3_7_2.json`。连续BF16 hidden与单个Long index使TensorIterator折叠到一维，选择index_elementwise_kernel_1_1<128,4>，block128/gridceil(H/512)、14thread寄存器、零private/shared；该域与广播优化环境开关无关，不要求未证明的向量指针对齐。每个有效scalar在tail predicate之后读取一个8B index及2B值并写2B，逻辑唯一payload为4H+8B；源级重复index读不推断广播cache减免。数据通路仅拷贝BF16，无浮点算术/转换；地址和uniform控制指令服务仍排除，不能将INT64比较套INT32速率。lowering保留独立INT32→INT64 producer与依赖，KernelSpec和predictor重用公共selector验证exact launch。R>1有另一个广播/向量化分支，仍保持unsupported；不是index_select，也不是GPU/provider、完整请求或normal验收。

### C500 条件 BF16→FP32 Copy 定价（active / partial）

同版本Copy源码/编译ISA/元数据见 `microbench/metax_c500/references/logits_bf16_fp32_copy_torch_2_4_maca_3_7_2.json`。仅明确的连续logits、N>=2048、输入/输出实际指针至少4B对齐且优化未禁用时，采用256线程/vt4/gridceil(N/1024)；整块10个thread寄存器，尾块14个，不能混用。BF16x4经两次SHL_B32和两次AND_B32位扩展，scalar尾部逐值SHL，合计N个整数32位操作，读取2N并写4N B；复用原整数ALU和访存资源，不当F2F/SFU、不新增速率。尾块另有倒数/转换用于索引计算，仍属公开排除的地址与控制服务，不宣称完整ISA时延。KernelSpec及公共SymbolicGreedy的可选copy_dispatch承载明确flag及两个pointer alignment，lowering/predictor共享selector校验exact launch；缺字段不补默认，不从fresh/contiguous推断对齐。PYTORCH_DISABLE_UNROLL_FLOAT_OPT首次读取后缓存，未设置时允许优化，任何已设置字符串（包括0/空串）均禁用。原Serving case没有这些分支事实，继续unbound；本项只能证明显式条件源码的计价，不解除原完整请求、normal或actual provider门槛。

### C500 小向量 greedy 输出 INT64→INT32（active / partial）

同一Copy源码和精确Long输入/Int输出编译分支见 `microbench/metax_c500/references/greedy_int64_int32_copy_torch_2_4_maca_3_7_2.json`。0.24 greedy最终to(int32)在1<=N<2048时固定选择256线程/vt4、15thread寄存器的generic unrolled entry，不依赖优化开关；N>=2048仍拒绝。输入地址按8B前进，但四个展开槽均仅LDG_B32读取Int64低32位，cast_and_store<int> helper以STX_B32 devc写4B，未执行数据cvt或符号扩展。逻辑role保持8N读/4N写，源指令issued payload为4N读/4N写，二者不冒充DRAM事务或cache命中。Midlayer复用已有访存资源，地址、dtype分派和helper call服务仍排除；不借Int32速率计Int64比较。输出cast通过公共selector严格验证并接共同forward，不改变argmax的unsupported边界；完整链仍拒绝未计价argmax，不能以单个输出cast通过升级完整采样或原请求验收。

### Masked TP embedding 的真实 table extent（active / partial）

普通0.11/0.24、无added vocabulary的TP分片先把global vocabulary按64行padding再除TP；129行/TP2得到真实96行table，不能把96再次padding成128。共同SymbolicEmbedding可接明确source_rows，masked调用从同一TP布局传入它；INT64 mask输出直接进入F.embedding，不增加不存在的long copy。A800与C500复用各自已有同版本index_select规则及寻址域，不新增launch或速率；C500 gather的内部source身份使用unquantized_index_select，不把TP2标为TP1。mask_ids和mask_output依然分别unsupported，本条不放行整条embedding或完整请求。

0.11 mask helper的torch.compile使用dynamic=True、平台inductor backend但默认fullgraph=False；注释称单kernel不证明实际单launch。BOOL乘Python-int的dtype规则使offset及masked IDs成为INT64，返回mask为BOOL；随后long()为alias。生成的mask wrapper/kernel及独立masked_fill_的实际dispatch/资源仍须各自绑定，不能用cast、gather或BF16 add/mul launch冒充。

### C500 显式 Copy 事实的大尺寸 ID 程序（active / partial）

同一已保留host分派的新适用域核对证明，`0x16df749`在缓存的disable flag为True时直接跳到`0x16e1869` generic entry，与N及指针对齐无关。明确提供`copy_dispatch.disable_unroll_float_opt=True`且连续INT32→INT64、8N<2³¹时，N>=2048可复用原256线程/vt4/22寄存器程序及每元素一次ASHR_I32；不是仅凭generic符号存在外推。进一步复用既有bundle51反汇编与MsgPack ELF note，精确Long输出functor的optimized full/tail entry分别为0x542700/0x542800，两者均10个thread寄存器、零private/shared；源码Long/Int分支以256线程、vt2、ceil(N/512)发射，N%512决定entry。启用优化且明确两指针至少4B对齐时使用该vt2程序；packed LDG_B64/STG_B128每次处理两个值，奇数tail逐值补齐，仍是每元素4B读/8B写和一次ASHR_I32。普通embedding与公开整数cast共享selector和校验，不新增速率；共同SymbolicEmbedding显式传递同一Copy事实，真实INT64中间storage仍保留。没有flag的原配置、启用优化但未证明4B对齐、split/strided仍拒绝；不得读取仿真进程环境代替被仿真运行状态。原小N域无需flag，BF16→FP32条件分支和argmax拒绝不变。这只补齐可证明的源码域，不通过原完整请求、normal或provider验收。

### A800 TP mask 的既有编译缓存身份（active / partial）

`microbench/nvidia_a800/references/embedding_mask_torch_2_8_inductor.json`绑定仅用pickletools静态解析的真实FX cache，installed helper与upstream0.11完整函数AST（含decorator）相同，annotated graph保留函数源码行，cache source_code与wrapper全文一致，编译身份为Torch2.8/Triton3.4。N=1实际wrapper为两个kernel；动态N的size_hint128/8192均有2<=N<=INT32_MAX的重叠guard，五个vocab/padding标量均未被专门化。离散best_config将两个8192-hint源码绑定到XBLOCK256、4warps及16寄存器sm80 cubin，但不能按当前N任选hint、把候选cubin当已选择配置或外推为原Serving通用launch。完整外层Dynamo guards、指针对齐、原请求实际compile context和ISETP/SEL/PLOP/carry等typed服务仍未闭合。此事实替代“找不到helper编译来源”的旧缺口，不替代native、正常容量或原60case验收；不再重复同一cache全盘检索。

### C500 通用 BF16→FP32 Copy 与图内 layout 事实（active / partial）

复用同一bundle51和MsgPack ELF note，Float输出的generic LoadWithCast entry为0x614b00，256线程/vt4/ceil(N/1024)、17个thread寄存器、零private/shared。BF16输入tag15的四个slot均LDG_U16后SHL_B32移16位；Float输出tag6进入cast_and_store<float>，0x393be0的STX_B32原样写出，数据路径没有F2F/SFU或值依赖转换。连续、4N<2³¹的N<2048由源码固定使用该entry；更大N须明确cached disable flag为True。启用优化且已证明指针对齐的大N继续复用原10/14寄存器full/tail程序；仅有低对齐保证不能反向证明实际未对齐而选择generic。三个程序统一selector和每元素一次整数位扩展、2B读/4B写，未新增速率；地址、dtype dispatch、uniform、helper call与tail控制仍沿既有排除项，不宣称完整ISA时延。

共同SymbolicGreedy从虚拟Tensor真实stride计算copy_input_contiguous，C500 lowering只在明确连续的小N域自动绑定无flag generic；strided词表slice保持无launch，不能按shape猜连续或因fresh FP32输出推断输入对齐。大N的contiguous=True不提供优化开关或指针对齐，原C500完整case的FP32首拒绝因此继续保留；采样argmax、完整normal/provider和原60case均未通过。可选layout事实不是实际指针对齐观测，不恢复旧图、不省略采样或制造provider。

### C500 显式禁用广播的多行高级索引（active / partial）

复用 `microbench/metax_c500/references/logits_single_row_index_torch_2_4_maca_3_7_2.json` 的同版本IndexKernelMaca头文件及既有bundle95 ELF note。连续BF16[S,H]和Long[R]的二维row-select在明确`disable_index_broadcast=True`时，源码直接选index_elementwise_kernel_1_2<128,4>，无需向量指针对齐；block128/gridceil(RH/512)、22thread寄存器、零private/shared，不能沿用一维entry的14寄存器。每个有效元素读取8B索引及2B值、写2B；逻辑唯一payload为4RH+8R B，不推断缓存广播减免。R或H为1去掉unit iterator维后复用一维entry。整数寻址、div/mod、uniform与控制服务沿既有排除项，不宣称完整ISA时延。

共同SymbolicLogitsRowSelect、lowering、KernelSpec及predictor共享可选boolean事实并重新核对exact launch；该事实仅属C5000.24源。内部capability统一为pytorch_2_4_maca_scalar_index_launch。缺flag或明确启用广播的多行二维选择仍unsupported，不读取仿真进程环境补默认；原完整请求未因此获得缺失的Copy/argmax事实。未新增速率、不消费production duration、不改变normal/provider门槛。

### C500 argmax 的 MACA 专用分派与 typed pair（active / partial）

`microbench/metax_c500/references/argmax_fp32_int64_torch_2_4_maca_3_7_2.json`保留无GPU、无耗时的同版本host jump table和安装头文件。Float tag6在0x4912c38的第6个signed相对slot落到0x26e579b，并调用Float/Long ArgMaxOps；随后Reduce.cuh在generic前可分派MACA continuous/no-struct程序，不能据generic符号存在直接选128x4/256x2。对R1/V128256、32bit、4B对齐且continuous未禁用的明确连续场景，源码可推出block512/grid1、pair16B时shared8200B、无跨CTA scratch；该条件不证明原Serving已选择这个程序。SDK的FP32和long/longlong shuffle源码分别1和2个32bit bpermute调用，仍不是已选ISA opcode数量或吞吐证明。NaN/相等时的小INT64索引语义保留；typed比较/选择/permute服务、实际entry/ISA和原分派事实仍缺，argmax继续拒绝Midlayer，不将浮点sum或泛整数速率借作完成。

### C500 argmax 条件入口的真实 ISA 与静态 SMEM（active / partial）

沿上条同版本源码，已从bundle124的xcore1000代码对象绑定Float/Long、vec4、false/false的InputPerOutputContinuousReduceKernel，入口0x722e00/size0x3a08。ELF note证明24个thread寄存器、58个scalar寄存器、最大block512、零private及4B静态shared。此前8200B只是该明确源码场景的动态shared量，两者原始shared需求之和为8204B；runtime对齐/分配粒度和实际CTA admission footprint仍未绑定，不能把原始和当已观测的硬件占用。该SMEM不是HBM activation，也不证明原Serving选择了此入口。

实际ISA保留cmp_o_f32/cmp_eq_f32/cmp_gt_f32、原生cmp_lt_i64/cmp_le_i64、csel_b32以及sm_bperm_b32；不能根据vocab小于2^31把INT64比较当INT32，也不能假定NaN分支被编译器消除。整个entry有21个sm_bperm_b32静态site，形成七组triple，但包含循环及分支，不能把静态site数当每lane动态指令数或按N直接相乘。NaN/mask控制、实际运行分派和上述typed服务仍缺；不新增production速率/launch，不消费production duration。证据统一扩展既有argmax reference，没有重开已通过的Copy/gather或Kernel回归。

### H200 ID Copy 的既有 SM90 资产与离线工具边界（active / blocked source step）

`microbench/nvidia_h200/references/embedding_id_widen_torch_2_8.json`引用已有唯一Torch2.8+cu128库身份及host Copy证明，已重新核对库哈希；未import Torch或加载该库。CUDA12.2 cuobjdump能列出该库的SM90 ELF对象，所以不能把H200 helper缺失写成本地没有SM90资产；但指定Long输出Copy entry的SASS＋资源和仅资源读取均以SIGSEGV退出。现有/usr/local/cuda别名仍指同一12.2工具；此边界仅说明这次静态工具调用失败，不推断根因或源码不可实现。

没有获得SM90寄存器、指令或选择事实，不把A800/sm80结果外推到H200。重新开启条件是提供可用的本地兼容静态解析工具，或同一库/entry的已解码SM90 cubin与元数据；此步骤不需要GPU授权。请求已明确到本地工具路径，不反复全盘检索或重跑同一崩溃。原配置/生产launch/物理速率均未修改，完整Serving与normal容量门槛继续未通过。

### H200 ID Copy 的独立 cubin 恢复及生产绑定（active / partial）

上条整库cuobjdump崩溃已在本步骤绕过：流式提取同一固定库中的完整SM90 ELF，保留包含精确Long输出Copy entry的libtorch_cuda.151.sm_90.cubin，及时删除无关临时对象；本次临时峰值约41MB，任务目录已清理。独立nvdisasm可解析symbol467，故该entry不再以“缺兼容工具”阻塞，不需要GPU或新增下载。来源仍由 `microbench/nvidia_h200/references/embedding_id_widen_torch_2_8.json` 维护，不能把旧崩溃视为整个离线路径不可行。

同一host Copy动态cast分派在SM90的独立compiled entry证明block128/vt4/gridceil(N/512)、32寄存器、零shared/frame/minimum stack。INT32 tag3的四槽分别LDG.E4B、SHF.R.S32.HI符号扩展，Long tag4分别STG.E.648B。普通vLLM0.11 TP1 embedding与显式pytorch_2_8_sm90整数物化接共同launch primitive和原整数/访存资源；source迭代器域为连续、未拆分且N<=2^28。A800保留独立sm80身份，两个hardware package均调用共同几何构造，不互相import。KernelSpec重新验证exact launch，wrong arch/source或伪造geometry/寄存器仍拒绝。地址、dtype dispatch及uniform/control服务沿原公开排除项；不增加物理速率、不当actual provider验收。

H200 embedding gather、masked embedding、FP32 cast和argmax没有因此放行。下一步按保留原method/config/normal策略的原H200请求复查首拒绝，不能将ID单项通过当作完整请求或原60case通过。

### H200 embedding 及行选择的独立 SM90 gather（active / partial）

从同一固定Torch2.8库流式提取397个SM90对象，仅保留精确vectorized_gather_kernel<16,long>的cubin198及BF16索引程序所在cubin199，其余临时对象已清理，提取临时峰值约359MB、本worktree始终低于5GB。独立nvdisasm及NVINFO证明vectorized入口和SmallIndex<BF16,long,uint,2,2,-2>均24寄存器、零shared/frame/minimum stack。Long的host分支0x1f8ff23进入0x1f4f110，二维连续case在0x1f5015f选择对应hostentry0x1df3610，不混用INT32 index template。证据统一扩展既有H200 embedding reference，不重复维护Torch全库身份。

连续BF16表/输出、Long IDs、H为128倍数及32bit字节extent下，R<=16小分支用block128/gridH/128（H<=132*8*128），每thread/row低32位读取Long索引、BF16 scalar读写；R>16沿源码vectorized快路径，grid(R,ceil(2H/4096))、block min(256,32*ceil(2H/512))。vectorized程序在payload-tail前读8B索引，活跃线程各搬16B，不能把重复索引读取当唯一索引payload或推断cache减免。普通0.11 logits高级行选择复用该vectorized entry并保留独立INT32→INT64 producer。

A800/H200使用共同几何与domain helper及既有analytical gather模型，保留各自compiled身份与HWSpec资源；没有新增速率、production duration或完整ISA时延声明，寻址/分派/uniform控制的既有排除项不变。KernelSpec核对物理padding shape和exact launch，宽索引/非对齐slice/错误source及假造资源仍拒绝，TPmask/FP32cast/argmax/normal/provider未放行。下一步仅复跑已因该项失败的原H200请求和直接依赖。

### H200 BF16→FP32 的 SM90 IMAD 位展开（active / partial）

`microbench/nvidia_h200/references/logits_bf16_fp32_copy_torch_2_8_sm90.json`从已保留同版本cubin151独立解析Float-output entry（symbol446）：30寄存器、零shared/frame/minimum stack，block128/vt4/gridceil(N/512)。BF16 input tag15明确进入四个LDG.E.U16＋IMAD.U32(value,65536,RZ)槽；Float output tag6以STG.E原样写32位。每元素2B读、一个普通INT32位展开乘法、4B写；数据路径不使用PRMT、F2F或SFU。与A800 sm80的PRMT实现不同，不能把A800缺PRMT服务的结论外推为H200同样缺失，也不能反过来给A800放行。

H200只绑定普通vLLM0.11 BF16 sampler、由同一虚拟Tensor stride证明连续、1<=N<=2^29的未拆分iterator；无需MACA Copy开关或额外向量对齐事实。复用已有H200 INT32算术及访存资源，IMAD加数为零，不计浮点FMA或虚构转换速率。KernelSpec与predictor重用exact selector；strided词表slice、FP16、宽索引/其他版本和伪造30寄存器程序仍拒绝。dtype dispatch中的PRMT及其他地址/uniform/control服务保留已有排除项，不声称完整SASS时延、actual provider或完整采样通过。argmax与最终INT64→INT32输出仍分别验收；不消费production duration，不修改物理速率。

### H200 greedy 输出 INT64→INT32 的低字 Copy（active / partial）

`microbench/nvidia_h200/references/greedy_int64_int32_copy_torch_2_8_sm90.json`复用同一Torch2.8库和已保留SM90 cubin151，独立解析Int-output dynamic-cast entry（symbol474）：30寄存器、零shared/frame/minimum stack、block128/vt4/gridceil(N/512)。host的Int输出tag3经jump table落在0x16a7eab，并在连续动态输入cast域选择对应unrolled entry。Long输入tag4的四个slot均只LDG.E读取8B步长地址的低32位，Int输出tag3原样STG.E写4B；没有数据转换或浮点算术。逻辑storage保持8N读/4N写，已发出payload为4N读/4N写，不能混作实际DRAM事务或cache命中。

仅绑定普通vLLM0.11 BF16 all-greedy链的fresh contiguous INT64输出到INT32、1<=N<=2^28的未拆分iterator，随后unsqueeze仍是view。H200与既有C500各自验证exact launch后复用共同访存模型，不新增速率、不借INT32吞吐定价INT64比较。地址、dtype-dispatch、uniform和control服务沿既有Copy排除项；strided/split、其他版本/硬件不外推，argmax、actual provider、正常容量与原Serving请求仍未通过。没有GPU执行或production duration消费。

### H200 argmax 的独立 SM90 程序与未定价服务（active / partial）

`microbench/nvidia_h200/references/argmax_fp32_int64_torch_2_8_sm90.json`从同库cubin227保留三个Float/ArgMaxOps模板；output_vec_size为1/2/4的编译资源分别为32/54/83寄存器，均零frame/minimum stack且1040B静态shared。它们不是按block宽度命名的三个shape候选：连续末维归约的output_vec_size=1应使用512/1模板。仅在已选择该generic程序、连续FP32[1,128256]及未拆分iterator的条件下，Reduce.cuh可推出block512/grid1、8192B动态shared且无global partial/semaphore；加上1040B静态shared的原始需求为9232B，不证明实际launch或runtime对齐/admission。

symbol223的0x1930 NaN predicate在0x1970保留实际条件跳转，numeric/equality及NaN分支有不同FP32 predicate、双字INT64排序和select程序。不得把源码最少N−1次pair merge当compiled工作、默认无NaN或用FP32 FLOP/单INT32速率冒充typed服务。固定库argmax_kernel_cuda的Float tag6在jump table中落到0x23a5e10，再进入0x23a4cb0的Float/ArgMaxOps；其中setReduceConfig调用为0x23a4f0e，output_vec_size=1的entry加载与launch为0x23a5a5a/0x23a5a81。该离线产物解除“没有SM90 argmax编译代码/host Float分派”的缺口；全请求shape的配置/scratch、typed服务及值依赖域仍未闭合，生产argmax继续unsupported，不据metadata通过宣称原Serving或normal验收。既有独立register probe只作准备，不重复构造或执行未获授权GPU实验。

H200生产selector现已接入该source的完整连续32bit适用域：rows>0、vocab>128且FP32输入/INT64输出extent均小于2³¹。geometry由Reduce.cuh原公式及显式HWSpec SM数/线程上限决定，采用同一output_vec_size=1的32寄存器模板；block.y参与归约时不分摊输出，grid.x为rows，不能错写成ceil(rows/block.y)。V128256所有常规rows无global split；R1在V130560→130561时values_per_thread由255跨到256，按源码开始16个CTA，创建256B partial与4B semaphore并先清零semaphore；H200132SM/2048线程下R64/V151936则为grid64×9、partial9216B和semaphore256B。两个workspace的源码extent和局部scope通过logical_work暴露，不能当已追踪graph lifetime、实际allocator或normal通过。

KernelSpec只为这一精确H200 argmax允许已知launch同时保持unsupported，reason为typed_service_unavailable；其他unsupported仍须launchless。必须携带完整的两个正整数hardware source字段、exact geometry/寄存器/shared/scratch，并禁止提升为supported。Midlayer继续拒绝，不增加任何速率、不改变原数学FLOPs/bytes Roofline口径；typed predicate/64bit排序/select/shuffle服务及值依赖域仍是点预测的独立缺口。旧的launch_selector_unavailable不能用于包装一个known launch，元数据通过不替代完整采样或Serving验收。

现有FA2独立机制服务不能补作argmax速率：`microbench/nvidia_h200/operators/flash_attn/fa2_scalar_dependency.cu`的mode1是128线程CTA内的`max.f32`重复链及两轮quad4 shuffle，mode6是INT32 column>=maximum_column后写-inf的掩码链；`fa2_scalar_dependency_sm90.json`及HWSpec中的cycles/128-thread-CTA-stage是这些完整stage的单位，不能除以源码比较站点数得到纯FP32 predicate、双字INT64排序、pair选择或单shuffle速率。`compute/typed_register_primitives.md`的六类探针也只有离线准备，compare+select bundle即使将来测得也不自动成为纯predicate或完整pair比较服务。排除的是跨单位/跨程序借用，不否定原FA2服务；重开条件为独立物理文档或经授权的机制raw，及到当前SM90指令程序和依赖/值域的可验证映射。不从production duration选系数、不查询或执行未授权GPU；normal账本和缺失compiled-profile分配图另行保留，不用source下界冒充日志10%验收。

### A800 精确 PRMT 机制服务与 BF16 logits Copy（active / partial）

用户授权后在空闲A8007执行独立寄存器机制：16个编译程序静态检查通过，32个smoke和三轮288个matrix cell全部正确，source为369e4b190。唯一绑定owner为 `microbench/nvidia_a800/references/logits_bf16_fp32_prmt_torch_2_8_sm80.json`，引用既有Torch2.8同库sm80 BF16→FP32 compiled程序；block128/vt4、30寄存器，每元素一条 `PRMT Rdst,RZ,0x5410,Rsrc`。固定预声明chains4/threads256/grid432/outer511，先每trial对实际同SM区间work/span取中位，再取三trial中位，得到1.9672056077033155 warp PRMT/SMcycle；全部遥测1140MHz，乘32lane和1140MHz、除4subcore得到每物理subcore每秒17940915142.25424次lane PRMT。它是独立机制的持续服务观测，含循环/调度/计时开销，不是production Copy duration，不按Serving误差选择。具名compute_alu服务保留物理单位，不能借作SHF、纯predicate或argmax。

生产仅绑定普通0.11 BF16 sampler、已由同虚拟Tensor stride证明连续、未拆分32bit iterator的1<=N<=2^29；2N读/4N写及allocation不变。现有共同lowering/KernelSpec复核exact launch，固定SI服务仅在HWSpec nominal1140MHz允许，其他clock拒绝。依赖链的BF16展开probe两步后归零，非归零rotation对照在同几何下速率一致；这不证明任意输入timing完全无关，限制随预测保留。寻址/dtype-dispatch/uniform控制沿既有Copy排除项，完整argmax、actual provider、normal容量与原60请求仍须独立验收。收益上限仅为移除原9/60首拒绝，不能当15%的端到端完成承诺。

C500228个机制cell亦已通过，但当前compare+select bundle不能拆为纯predicate；probe每shuffle一个arrive与production三shuffle后一个arrive不同，不能将probe时延直接乘三。910B独立INT32→INT64真实dispatch正确不允许消费整个Cast duration；UB count-mode Cast另测。局部机制证据不自动接受其source/provider绑定或Serving。

### Performance-only analytical helper pricing (active; user decision 2026-10-23)

The user explicitly scoped this task to performance simulation and approved modeling argmax by one complete input read. Forward may therefore price an explicitly defined analytical helper even when its native launch is unavailable. Plain greedy argmax reads FP32 rows×vocab once and writes one INT64 index per row; latency is these bytes divided by the configured effective DRAM bandwidth. The output INT64→INT32 materialization uses its logical read/write bytes. Existing typed-byte models cover ID widening, logits conversion/row selection, masked embedding and declared ordinary embedding indexing. No model values, NaN branches, exact reduction tree, ISA matching or accelerator experiment are prerequisites for these performance estimates.

This is a deterministic analytical policy, not an exception triggered by an observed duration or a swallowed failure. Native support/launch metadata stays separate, unknown analytical work still fails, and Oracle/PipeWeave do not inherit the policy. No physical rate is changed. Existing finite/nonnegative validation of simulator timings and breakdowns remains mandatory. Old native-provider conclusions retain their original scopes; they no longer block the approved analytical performance path.

The same policy applies to the Vector owner's existing explicit mathematical programs, including source softmax/top-k routing and TP logits gather/reorder. Midlayer reuses those logical FLOP/typed-byte programs and labels their predictions `analytical_logical_work`; it does not infer native support or select a provider. Missing mathematical programs, unknown input storage widths and conditional branches without a defined analytical projection still reject. Source-launch validation remains at the KernelSpec boundary (with hardware input when required), so generic predictors no longer import hardware launch selectors. This removes repeated native prerequisites without changing any hardware service rate or supported native formula.

The declared legacy GLM draft argmax uses the same performance policy with its BF16 input width: one full logits read and one INT64 output per row. Both Roofline and Midlayer consume that existing typed-role ledger. Unknown draft bookkeeping metadata modes remain unsupported; this policy does not guess their branches or materializations.

### Exact source-layer repeats in accelerated device-only execution

The full virtual source forward remains the authority for dependencies, aliases and storage lifetimes. Summary execution may lower adjacent equal dense layers through the existing WorkloadBlockRepeat/KernelBlockRepeat contract. Equality uses the lowering owner's existing KernelSpec projection and the complete incoming/internal DAG, rather than elapsed time or a model-name assumption. Every original operator occurrence remains in the repeat's trace mapping, and logical node, edge and kernel-call counts are preserved. Nonlocal feedback, early external consumers, unequal projections, unrepresented nodes or a changed logical edge count keep the full graph. Detailed timelines, Host timing and MoE retain their existing execution representation. No hardware service, formula, request or source storage accounting is removed or changed by this optimization.

### Physical prediction class of the pinned FA2 split reducer

The `flash_attn_splitkv_combine` program consumes FP32 partial output and LSE and writes final output/LSE. For an unchanged launch, split count, query rows, head geometry and complete remaining contract, its compute, traffic and dependency program is independent of the upstream KV window. Prediction memo keys may canonicalize only `shape.kv_len` and the equal logical/model-input KV-length arrays for this pinned reducer. Masked graph slots, main attention and every other reducer keep their exact contracts. Public KernelSpecs, resource inputs, call counts, dependencies, predictions and hardware rates are unchanged; this does not canonicalize the resource-input API's historical Q/K/V footprint field. The rule follows the existing `_combine_traffic`, `_combine_flops` and `_attach_gpu_fa2_combine_pipeline` source algebra, not production durations or an error target.

### 共享图必须保留0.11的版本化buffer程序（active）

vLLM0.11的`_custom_ops.py::rms_norm`先无条件执行`input.contiguous()`，Qwen3从packed QKV分出的Q/K在多token时有leading-axis gap，因此该版本每层确有两份materialization；不能将0.30 C++支持unit-final-stride的规则应用到0.11。共同RMSNorm通过`vllm_c_cuda_0_11` storage program保留这段Python wrapper，并将实际分组stride交现有TensorIterator copy selector，不新增速率或duration。0.30程序仍保持unit-final-stride输入不复制。

vLLM0.11 `attention/layer.py::Attention.forward`的use_output分支为`torch.zeros`，不是0.30的empty分配；版本化source必须计完整physical query输出buffer的一次fill。FA2 decode的既有三个TensorIterator wrapper copy也属于该原生调用，不能仅因虚拟forward不展开C++内部而删除。A800准确case的canonical alignment decode launch sample同时保留每层Q/K两次copy、FA2前一次和后两次copy及一次BF16 fill；这些非duration事实支持复用已有launch/model，不能把它们描述为“旧图多余操作”。共同attention node显式携带source版本及physical output rows，PIECEWISE填零按完整output backing行数计，不能用batch×max_query的矩形近似冒充实际行数。

PIECEWISE下模型的Linear/activation等body执行padded token extent，eager attention/KV只消费active tokens：0.11 model runner按num_input_tokens传model，FlashAttention backend用query/output[:num_actual_tokens]及实际cu_seqlens调用kernel。新图保留该区别；旧图把body改按logical rows计、又给eager attention附加padding metadata时的差异不应为数值相同而恢复。完整请求误差仍须独立比较，源码正确改变不保证旧预测的偶然误差抵消继续存在。

### MetaX0.24 Attention输出copy与前端buffer程序独立（active）

`vllm_metax_0.24.0/v1/attention/backends/flash_attn.py`的split-forward在decode调用`flash_attn_with_kvcache`后，将返回结果赋给`output[:num_decode_tokens]`；prefill的fresh结果同样赋给output后缀。版本化source必须分别复用既有decode输出copy selector及prefill partition输出copy，不能因前端`Attention.forward`已从zeros改为empty就删除backend输出copy。0.24前端output仍不计BF16 zero-fill；RMSNorm wrapper也没有0.11的显式input.contiguous，不恢复旧Q/K副本。既有native copy服务、参数和物理字节定义不变，未绑定的其他source不得继承该copy程序；此修正不代表完整Serving请求误差或实际provider已验收。

### Ascend0.18 FULL Q1按真实inner_precise默认绑定SplitFuse（active / partial）

910B约定容器的Torch NPU `npu_fused_infer_attention_score.out` dispatcher默认inner_precise=0；vLLM0.18 full_graph_fia不覆盖该值。真实CPU DoOpTilingForCompile在BF16/TND/H32/KV8/D128/paged128/sparse3的无补齐Q1输入选5000000000010200203/24AIC/128B，而不是regular FIA103系列。源接口没有Serving的prefill/decode分类参数，不能只凭decode标签选regular。FULL cached、全正KV且没有额外dummy的Q1现沿显式ascend_0_18_full_fia_split_fuse合同复用既有SplitFuse 128行/512栈MMAD、AIV和pipeline实现；不改算力、带宽或任何duration参数。普通独立Kernel及其他来源的原selector保持原合同，不据此重写旧采集身份。

同一CPU查询的Q5pad8及Q7pad8也选择SplitFuse，其flash_attention_regular.h在kvSLoopNumTotal=0时调用block_epilogue_init_outputs.hpp的局部零输出epilogue，并仍调用loadQGM；这不是regular fia_kernel_nonquant.h的全输出InitOutput/SyncAll程序。因此撤销Serving单Q1/K0尾行借用regular native的分派，暂保留显式解析工作，不能沿用d94的regular全输出初始化结论。多行/零行dummy、fresh/nonuniform以及实际OpAPI/provider仍独立待闭合；CPU编译记录是离散compiler事实，不是设备执行或时延验收。原raw位于已登记fia_host_tiling_cpu_20261026；第三方来源身份沿现有FIA owner维护，结果报告保留实际运行snapshot。

同源核对还发现SplitFuse公式内AIV初始时钟额外加了regular FIA的SyncAll，虽然其launch envelope已经正确排除该程序；flash_attention_regular.h正常路径没有这次同步。删除这项重复借用，保留regular FIA真正的InitOutput/SyncAll以及独立冷填充机制不变。独立机制值为3.1159us/launch，对36层完整step的可去除上限为0.1121724ms；这个数仅用于影响估算，没有用误差选择机制或参数。直接SplitFuse source不变量检查必须同时排除launch envelope和pipeline时钟内的regular初始化同步。

FULL cached Q1的独立zero-KV尾部沿同一SplitFuse source程序计价：真实请求仍进入既有QK→softmax→PV→normalize工作计划；尾部AIC执行loadQGM，AIV只执行block_epilogue_init_outputs.hpp的Duplicate及逐head输出写，无QK/PV、Exp/Div、workspace写、全输出初始化或跨域handoff。typed FiaSplitFuseWorkPlan新增非负empty_kv_tail_query_length字段（默认0，限定paged KV），其任务从真实任务序号继续按24AIC轮转，但不增加正KVhandoff次数。局部Duplicate跨head复用，不能按每个head重新生成整块零值；源subblock在head bundle=1时分Q行，否则分head。内存与Vector服务守恒同时覆盖真实前缀和独立尾部，输出只写一次。正KV prefix与源dummy保持严格验证，不制造KV1或拆成多个虚假请求，fresh/nonuniform源调用仍独立保留。

合成Q=(1,127)/KV=(513,0)的额外CPU查询仍返回5000000000010200203/24blocks/128B，真实反射布局解码totalTaskNum=40、firstBatchTaskNum=8，等于8个真实任务和32个空KV输出任务；raw追加至已登记fia_host_tiling_cpu_20261026目录。仅用compiler离散事实核对大尾部的head拆分，不消费任何duration或声明实际provider通过。默认空尾部保持既有Kernel价格不变；完整Serving按原请求重跑，不将小批次估算当成TTFT改善保证。

同一FULL FIA合同现覆盖已有BF16 GQA prefill/mixed：_get_fia_params在PrefillNoCache时无block table、直接TND K/V并使用累计Q边界作为KV边界，因此lowering显式绑定contiguous KV；cache-hit/mixed保留paged KV与逐请求KV长度。两者共享既有SplitFuse程序，fresh尾部q=KV=padding，cached尾部KV0，q0尾部不产生任务。KernelSpec要求显式ascend_fia_prefill_no_cache布尔值并与logical Q/KV、物理prefix、独立dummy、layout和对应500...key一致；不把fresh来源假称paged decode，也不把零行变成Q1。source contract注册名统一为vllm_0_18_full_fia_split_fuse，未改HWSpec物理参数。

匹配真实.out默认值的三项CPU查询并行完成：cached Q=(1,7,0)/KV=(513,7,0)返回paged500.../24blocks/totalTaskNum16；fresh Q=KV=(7,1)返回contiguous500.../24blocks/16任务；fresh Q=KV=(8,0)返回同contiguous key/24blocks/8任务。原始记录追加至已登记fia_host_tiling_cpu_20261026，runner9fdf1226c仅扩展描述符布局和零行边界，无设备初始化或duration。由这些新事实彻底退休ascend_0_18_full_fia_decode旧source标签；KernelSpec/selector拒绝它，不再保留可误绑0.18 regular FIA的入口。regular FIA的正KV独立Kernel及其真实初始化服务仍保留，实际provider与eager调用路径仍不由此证明。

### Ascend0.18 eager FIA输出物化与fresh KV布局（active / source performance）

attention_v1.py:822–840的非捕获forward_fused_infer_attention先接收新attn_output，再执行output[:num_tokens]赋值；FULL .out直接写目标，没有这份copy。共同虚拟forward为明确走FIA的eager prefill/mixed（包含fresh Q1）恢复前端输出buffer、FIA返回buffer及显式copy节点和数据依赖，目标buffer在KV更新前分配，两个输出buffer同时存活到copy。copy复用已有CANN ELEMENTWISE operation=copy、reads=writes=1的性能模型，不新增速率或用完整算子duration拟合，也不把该抽象Vector发射声称为已观测的Torch/CANN TensorMove二进制。纯cached Q1的非FULL调用可能另走PA，未据此扩大其source绑定。

PrefillNoCache同_get_fia_params读取direct contiguous TND K/V；此前eager fresh沿runtime cache layout选择paged SplitFuse，错误引入paged-address服务，现按明确源码改为contiguous source selector。eager mixed继续paged。安装libtorch_npu.so非.out dispatcher schema也已只读提取，实际inner_precise默认0，与FULL .out一致，raw存已登记fia_host_tiling_cpu_20261026/torch_non_out_schema.json；未import Torch或初始化设备。这些修正处理性能工作及storage，不代表实际provider、全部源码workspace或normal容量已验收。

### CPU0.11 Scheduler 的 colocated GQA block128（active / control-plane geometry）

在既有16-token baseline之外，仅开放Qwen3 BF16标准GQA、colocated、TP1/PP1、无EP/MTP的128-token block。既有构造直接把显式block size传给原生CacheConfig和FullAttentionSpec，并在构造后校验effective几何；不改Scheduler版本、不启动真实worker、不替换0.18 attention或编译模式。该控制面域用于消费显式正常容量策略；策略的unverified证据等级保留，不能据此宣布0.18 provider、PIECEWISE性能等价或原16-token case的normal容量通过。其他block/role/并行/MTP组合继续拒绝。

### Ascend0.18 PIECEWISE attention与启动生命周期（active / source performance and capacity）

Ascend插件platform.py归一FULL_AND_PIECEWISE为PIECEWISE，全局NONE则关闭npugraph_ex；启动的segment-reference生命周期政策不能借给FULL或全局NONE请求。明确Qwen3 BF16 TP1/PP1、attention/KV eager的PIECEWISE配置保持物理body行数，attention_v1.py按实际Q/K/V前缀执行FIA。using_paged_attention只在FULL_DECODE_ONLY成立，因此PIECEWISE的cached decode也保留fresh FIA输出及向调用者output前缀的copy；超过捕获bucket而落到runtime NONE时仍保持该配置分支。copy在unified_attention_with_output内部，与attention一起归eager边界，不能因其ELEMENTWISE kind而算入编译段；其他ELEMENTWISE与既有FULL分支不变。

解析容量入口显式消费既有Ascend head-only startup source、source_lifetime_policy及完整compilation options，按source retained/nonretained生命周期分区并调用既有CPU0.11 allocator。外部reserves、capture/warmup及provider证据等级独立保留；native128、PIECEWISE策略不能替代原FULL/block16的容量验收。这里复用既有FIA、KV与copy性能模型，没有新增硬件速率、duration参数或KernelKind。

### Ascend0.18 TP2 masked embedding 的 Host dispatch（active / unsupported）

完整 Host 计时路径尚未覆盖 TP masked embedding 的 Host dispatch 分类：`runtime_prepare._dense_node_dispatch_events` 遇到 `(VECTOR, vllm_masked_embedding)` 的 `embedding:mask-ids` 节点明确拒绝。该边界影响当前标准报告的 Qwen3-32B TP2 EP1 和 Qwen3-30B-A3B TP2 EP2、各 QPS1/4/16，共六项；不能把失败项当作已完成或零误差。设备侧已有部分 typed 数学投影，不代表同版本 TP wrapper 的 Host 成本与编译边界已被描述。

现有 `Tensor.copy_` 独立测量不能替代包含比较、布尔、乘减及 masked_fill 的完整 helper；普通 embedding/gather 的 Host 类别也不能凭邻近关系借用。不得删去 Host prepare、虚构零成本或切换 device-only 使原 fullHost case 通过。重开条件为绑定同版本 Ascend TP wrapper 的实际 dispatch/编译边界，提供可复测的独立 Host 机制归类或微基准证据，并验证各 rank 的调用次数及 capture/eager 覆盖守恒。设备 native mask、正常容量和完整请求精度的独立边界仍需分别验收；本次本地集成未新增硬件采集。

### Source Host 重复层的等价展开（active / execution reuse）

此前“Exact source-layer repeats in accelerated device-only execution”条目中的 Host 限定已由现有 timeline 的逐 occurrence 调度补齐：Host 事件使用 `(template_call_id, occurrence_index)` 与 `KernelBlockRepeat.trace_node_ids` 关联真实 source 节点，保持每次 graph/eager 边界及依赖顺序，同时复用等价层的预测与内存绑定工作。异构层继续独立，无法对应的边界明确失败。该变更不修改硬件速率、Host 成本参数、请求、storage 生命周期或逻辑调用数量；完整128请求对照保留逐请求预测及内存统计，只有表示的图哈希与浮点舍入级 Host 汇总差异。具体验证与剩余运行速度回退见本次集成交付记录。
