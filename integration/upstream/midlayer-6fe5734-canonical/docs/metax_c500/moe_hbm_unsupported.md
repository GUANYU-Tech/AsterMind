# C500 MoE HBM 欠预测：判定 unsupported，以及一处不得单独修复的正确性问题

本文记录 C500 `fused_moe_kernel*` DRAM 流量欠预测的有界裁决。完整证据在 `experiment_logs/kernel/c500_dram_reuse/g3_bounded_verdict.md`（该目录被 gitignore，本文保留需要长期留存的结论）。判定只覆盖这一个子问题：同期 C500 其余各族 HBM 为 GEMM 1.29%、Vector 4.25%、FA 非 MLA 1.66%，C500 MoE 的 L2 为 1.69% 且越界 0，A800 与 910B 逐位不变。

现状：C500 MoE HBM MAPE 20.98%、中位 pred/meas 0.981、176 个 case 中 59 个越界，全部落在欠预测一侧。

## 1. 边界不是 tiling 签名，而是激活足迹相对 L2 替换容量

全部 176 个 MoE case 只有两种 tiling 签名（采集 runner 按 `M <= E` 二选一，镜像 vLLM 0.10.2）：`BM=16 BN=32 BK=64 GSM=1` 84 个 case，中位 1.000、越界 0；`BM=64 BN=64 BK=32 GSM=8` 92 个 case，中位 0.590、越界 59。

签名与 shape 完全共变，所以签名不是自变量。真正的自变量是源激活缓冲足迹 `TOK × H × dtype_bytes` 相对 L2 替换容量（8 MiB）：足迹在 L2 以内时模型精确，超过约 1× L2 后开始欠预测并随足迹单调恶化，中位 pred/meas 从 1.000 一路降到 0.267。BM64 组上 log-log 相关为激活足迹 r = −0.759，权重足迹仅 −0.318。

**订正**：BM16 组之所以准，是因为它的 case 全部落在小 TOK、激活足迹从未跨过 L2，**不是**因为 tiling 参数本身。此前把 BM16/BM64 对照当作某个 tiling 参数判据的结论都应按此收紧。

## 2. `wave_size` 物理上是错的，但不得单独修复（已由 §8 订正：104 是锁相态的正确组宽）

共享 miss 核心使用 `wave_size = min(m_tiles × n_tiles, num_compute_units) = 104`。这是 AP 数而不是常驻 CTA 数：ISA 给出 shared=16384，即 4 block/AP，物理真值是 416，现值小 4 倍。这个量纯由 launch 事实加硬件规格推出，零拟合。

换成物理真值实测反而更差：MAPE 20.98% → 22.01%，越界 59 → 61，BM64 中位 0.590 → 0.548（BM16 组两种设置下都是 1.000，不受影响）。

原因明确：核心的 `for wave: for step: for index in wave` 让同一波内所有程序处于同一 k 步，于是每个同波共享者都免费命中，**波越宽过信越多**。现值 104 看起来没那么糟，只是因为它恰好把波取窄了 4 倍，部分抵消了锁步理想化带来的高估。

**两处错误互相补偿。在锁步被替换掉之前，两处都不要动。** 单独把 `wave_size` 改成物理真值会让 MoE 退步，而这不是修复。

## 3. 为什么没有被允许的来源能关闭它

要移动这条残差，唯一已知的结构改动是把"同一波内所有程序同处一个 k 步"换成真实稳态相位分布。该相位量没有合法来源：

- **规格与源码**不含"天花板"这个量。
- **微基准族已被证伪**：轴 D 实测硬件自身的共享不依赖每 k 步的算术量（0→4096 FMA、compute 指令 47×、周期 8.2×，共享只从 0.080 变到 0.072），而生产对激活足迹 r = −0.759；探针在 24→184 MiB 上完全平坦。微基准标定不了它。
- **ISA 不含相位关系**。反汇编 device 镜像给出的正面事实是完整的：3 级软件流水，每级恰好 2 条 `ldg_b128`（A tile 64 行 × 64 B、B tile 32 行 × 128 B，每 block 每 k 步 8192 B = `BK×(BM+BN)×2`，独立确认了核心 replay 的每步字节数），载入在前两级的 MMA 块中间发射，每 block 最多 24 KiB 在飞，无任何缓存或驱逐修饰符。但指令流里没有任何把一个 block 的进度与另一个联系起来的构造：原子 0、信号量 0、全局计时器 0、自旋 0；全部同步只有 6 条 `barrier`，其中 3 条在循环体内对应 `num_stages=3` 的双缓冲，且这些 barrier 必为 workgroup 范围——canonical grid 48960 对驻留 416，device 范围的 barrier 会死锁。

稳态相位分布由发射速率、block 启动错开、访存延迟方差与替换策略共同决定，四者皆不在 ISA、不在 kernel 源码、不在任何可用规格文档。把它写进 Forward，无论包装成核心重构还是参数修正，都是拟合。

## 4. 已否决的捷径

从生产流量反推可以得到一个覆盖全部受控点的描述：`realised_adjacent = min(sharers, C)`、`realised_spaced = s(spacing)`，只用"最大间距处 spaced 项触底于 1"一条结构约束即可定出 `C ≈ 11.8`。但 `C` 与 `s` 只能从生产流量解出，而微基准直接否证了 `C` 的存在——相邻共享到 16 个共享者精确，90 个给 59.93，没有任何 11.8 的封顶。这是按误差反推，不落地。

把探针实测值原样搬进 Forward 同样不行：会让 DRAM 欠预测约 4 倍，方向更坏。对应 reference 的 `forbidden_use` 已写明。

## 5. 重开此题的条件

只有当稳态相位漂移量获得一个被允许的来源时才应重开：新的硬件规格文档披露发射或替换策略、厂商提供 block 启动错开的定义、或者设计出一个能在不复刻 production 算子的前提下直接测量相位漂移的独立机制微基准。在那之前，本条残差按 unsupported 保留并如实报告。

## 6. 2026-09-24 重开轮：相位律已实测，边界收紧为「幅度不是 launch 物理量的函数」

按 §5 条件（3）执行了计数器机制微基准重开（wt-c500-moe-hbm，checkpoint 收口：src 零改动，MoE HBM 维持 20.98%/59 全欠侧）。探针源码 `microbench/metax_c500/memory/l2/{production_stream_phase_law,concurrent_phase_sharing_law}.cpp`、81 计数器测点 reference `microbench/metax_c500/references/production_stream_phase_law.json`（raw/二进制 sha256、复测命令齐全，duration 零消费）；完整裁决报告 `experiment_logs/kernel/c500_moe_phase_law/findings.md`；主 agent 验收集成 `162885dbc`，run_tests 3986 全绿。

实测结论分三层。第一层，单流律已钉死且 miss 核心已正确表达：B 流（列块权重）单独恰好 2.00 遍（对深度、布局、驻留数、算术、stride pad 全部不变），A 流（gather-dup 激活）单独按驻留距离律（K1440/2880/5760 对应 3.0/3.5/8.3 遍）——不存在可修的单流误差。第二层，合流干扰真实且生产相关：两流共驻时 DRAM 读 ≈6× 单流之和，被任一开关（b_contig 0.103、a_scatter_nodup 0.106、a_contiguous 0.136、no_c_write 0.229、stages3 驻留 208 的 0.102、work2048 0.103）打回无干扰总量；L2 读请求流严格按流可加（a_only 281.85M + b_only 132.71M = base 414.56M）——干扰完全落在 L2 命中结局层，不在请求生成层。第三层是本轮阻断性新事实：**合流干扰幅度不是 launch 物理量的函数**——同一二进制内 90/91/连续三种行算术给 0.26/0.39/0.10；同 launch 几何、同机同日，仅 +2 个 ST 寄存器的重编把 base 从 0.41–0.43 移到 0.246–0.287（occ_query 证实占用率由 smem 决定，寄存器差不经驻留数起作用）；fully-associative LRU 对完整访问序在 tile/line 两粒度、锁步与稳态相位展开全 interleave 的复放全部 ≤ 锁步答案（0.03–0.11）。

方向判据成立：生产 176 case 按 rows/expert 分桶 median 0.9997/0.762/0.377、五分位单调、≥256 行 43/43 全欠，与实测律同向（`experiment_logs/kernel/c500_moe_phase_law/g2_case_direction.json`）。幅度耦合被证据合法阻断：probe2b（52 ST 寄存器）与 probe3b（54 ST）之间的数值差没有 launch 物理来源，按生产聚合在两者间挑选即 MAPE 驱动选择；`wave_size`→416 单独落地维持 §2 已证伪先例；把 set/映射建模需要本核心不消费的地址级信息。

重开条件由 §5 收紧为三条，缺一即拟合：(a) L2 set/slice 几何 + MSHR 容量的权威来源（硬件文档或可反演几何的专门探针）；(b) miss 核心升级为地址级/集合级仿真的架构授权（静态 set 模型也无法解释代码形态轴——那是 MSHR/时序层现象）；(c) 代码形态轴的物理解释，或生产二进制逐 case 的 regime 成员判定手段。

## 7. 2026-09-26 地址/集合级重开轮：探针轮次概要

本轮在 (a)(b) 授权下重开（wt-l2-set-miss-model，baseline `6e26ef05c`），结论与未修复原因统一见 §8，这里只列探针与各自回答的问题；全部测点、raw sha256 与复测命令在 `microbench/metax_c500/references/stream_sharing_interference.json`。

- `stream_sharing_interference.cpp`（计数器）：保留权重流、只换激活流。1 MiB 热区 A + B 精确可加（2.008 = 1.978 + 0.001），2.65 亿次顺序冷 miss + B 精确可加（33.636 = 31.641 + 1.978）——L2 请求流量与在飞 miss 压力不承载干扰；随机 gather（任意重复度、12–189 MiB 区间）+ B 出现干扰，§6 的 dup/nodup 判据在同一二进制内不成立。
- `stream_sharing_phase_stamps.cpp` / `stream_sharing_step_stamps.cpp`（时间戳孪生）：给出逐 CTA 执行顺序；按实测顺序在 8 MiB LRU 上回放逐点复现上述计数器，锁步顺序只复现锁相构型；两态 CTA 都经过 4 个 AP 档位，差别在档位轮转是否跨 AP 对齐。
- `cta_dispatch_order.cpp`、`l2_recency_protection.cpp`、`l2_random_pool_eviction.cpp`、`l2_stride_conflict.cpp`（计时/计数器）：分派为全局动态队列；替换按新近度、非复用保护；L2 全芯片共享、随机池逐出为平滑斜坡；2 的幂步长无集合集中。
- `routed_stream_phase.cpp` + `routed_stream_phase_routing.py`：上述探针的激活是随机洗牌、权重误用 [E, K, N] 列块；本探针按 canonical runner（c8cf3f318）精确复原路由并按 [E, N, K] 读权重，在生产形态上确认失锁（§8 的数字即来自此探针）。复刻探针与同 case 生产 kernel 量级不同（E32 W1 11.06 对 7.75 GiB），与代码形态依赖一致。

## 8. 收尾结论（2026-09-26，用户决定不继续）

**根因。** C500 BM64/GSM8 grouped MoE 的 DRAM 欠预测来自执行顺序，不来自 L2 本身。GSM 组内相邻程序共享权重 tile、同一 M tile 的激活被多个 N 程序共享，共享只在共享者于短时间窗内相继访问时成立。miss 核心按 104 个 CTA 一波的锁步顺序回放，共享者几乎同时访问，共享全部成立；硬件上 CTA 进度失锁，共享者相差 30–80 个 k 步，超过 8 MiB L2 约 30 步的可保持复用距离，每个共享者重新取数，DRAM 读为锁步模型的 2–5 倍。只有 C500 显著是因为 L2 小（A800 40 MiB、910B 有效约 160 MiB 能容下同样的相位差）。

**已确定的事实。** (1) 用实测逐 CTA 时间戳排出的访问顺序在普通 8 MiB 随机 hash LRU 上回放即复现计数器（合成 18 构型逐点对上；按 runner c8cf3f318 精确复原路由、[E, N, K] 权重的生产地址复刻 6 条流中 4 条在 3% 内），锁步顺序回放低 2–5 倍——L2 替换判定正确，缺的只有执行顺序。(2) 锁相态下每 AP 4 槽按最老优先经过 4 个推进档位、全芯片槽位轮转对齐，`wave_size = 104` 是该态的正确组宽（订正 §2）。(3) 失锁态下随机 miss 使 CTA 寿命离散（p10/p90 698/1218 µs），对齐衰减，相邻 pid 落在不同档位的 AP 上。

**已实测否定的候选。** 集合几何/相联冲突、2 的幂步长冲突、复用保护型替换、L2 请求流量、MSHR/在飞 miss 压力、未决 miss 合并、按 pid 绑定 DPC 队列、VL1、固定相位偏移（等于拟合），以及由已测物理量搭的 CTA 进度动态模型（优先级共享、实测 4 档速率、AP 级队头阻塞、DRAM 延迟均值与指数尾部抖动）——后者全部被共享数据耦合拉回锁相，无法产生失锁。

**为什么没有修复。** 失锁幅度由 AP 内部发射/调度微结构决定：驱动失锁的扰动源尚未被任何探针或文档识别，且幅度对代码形态敏感（同几何 +2 个寄存器使干扰 0.42→0.26），探针数值不能移植到生产二进制。零拟合地预测它需要 AP 发射/调度的周期级模型并按生产二进制指令组合驱动，属于 kernel 时延层，超出 L2/DRAM 流量模型 owner；任何固定相位参数或按生产流量挑选的常数都是拟合。此外 Serving 没有 token 级路由，激活行号序列无法在共用 Forward 路径上复原。用户评估授权周期级建模的风险与改动面后决定不继续，本残差按 unsupported 保留：C500 MoE HBM MAPE 20.98%、59/176 越界全欠侧不变。

**开放子问题。** FUSEDMOE_TOK4096_E8_TOPK2_H4096_INT14336 的 W1 在复刻探针中单流即远超实测顺序回放（B only 20.23 对 4.81 GiB，A only 2.37 对 0.066），步长冲突、DRAM 饱和、未决 miss 合并已排除，原因未定。

**重开条件。** 一个由独立探针参数驱动、零拟合的 AP 周期级模型，能复现 `microbench/metax_c500/references/stream_sharing_interference.json` 的锁相/失锁分区与共享者相位差分布，并能由生产二进制静态指令组合推出档位速率；三卡各有独立来源；Serving 路由缺口有明确处理。证据：探针源码 `microbench/metax_c500/memory/l2/{stream_sharing_*,routed_stream_phase*,cta_dispatch_order,l2_recency_protection,l2_random_pool_eviction,l2_stride_conflict}`，完整报告 `experiment_logs/kernel/l2_set_miss_model/report.md`（gitignore，canonical checkout 同路径）。
