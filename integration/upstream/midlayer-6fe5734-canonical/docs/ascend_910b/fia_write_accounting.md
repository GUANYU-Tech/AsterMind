# 910B FusedInferAttentionScore 写侧计价：环几何、被测量口径与未归属常数

本文记录 910B FIA 写侧 DRAM 计价的机制结论与**被测量的正确口径**。launch 几何的机器可读版本在 `microbench/ascend_910b/references/fia_workspace_ring_geometry_cann_8_5_1.json`，含各 buffer 的引擎、轮转键、extent 与源码 SHA。

## 被测量口径（三选一，只有一个是对的）

msprof 的 AIC 写有三种取法，它们给出三套不同的结论，必须固定用权威那一套：

| 构造 | 是否可用 | 理由 |
|---|---|---|
| `L0C_to_GM + L1_to_GM(estimate)` | **不可用** | `L1_to_GM` 带 `(estimate)` 后缀，是 msprof 推算；FIA cube 块只有 GM→L1（`common/CopyInL1.h:61`、`fia_block_cube_nonquant_gqa.h:783`），**没有任何 L1→GM 写路径**。 |
| `L0C_to_GM` | 不可用 | 管道侧计数，含到达内存前被丢弃的 padding；group=1 时它比权威写计数还大。 |
| **`write_main_memory_datas − UB_to_GM`** | **权威** | `write_main_memory_datas` 是权威写计数；AIV 侧闭合于 `UB_to_GM`（差 0.006–0.10 MiB），故相减得 AIC 侧。 |

三套口径在 decode 上给出的环/实测分别是 0.50、0.81、0.52——**换口径会换结论**。本分支曾把第一种用在诊断里并据此读出机制，两轮后撤回；已判定不可入账的字段，必须在**任何**诊断构造里排除，不只是在入模型那一步排除。

## 口径换了会换出什么：一次撤回的完整记录

这一节保留过程，因为结论本身依赖于口径，而口径错了会读出一个看似很强的机制。

第一轮诊断用 `L0C_to_GM + L1_to_GM(estimate)` 作 AIC 写流，得到"两个 kernel 共用一笔 **35.59 KiB/核** 固定写"，并据此把 `workspaceParams` 四个字段按尺寸不变性全部排除。剔除那个不可入账字段后：

| 组 | 含 L1(estimate) | 仅 L0C | **权威口径** |
|---|---|---|---|
| decode contiguous | 35.12 | 7.68 | **32.89** |
| decode paged | 35.09 | 10.47 | **32.90** |
| decode paged_varlen | 36.85 | 4.47 | **32.75** |
| prefill chunked_prefill | 36.19 | 31.08 | **34.32** |
| prefill contiguous | 35.81 | 31.76 | **34.06** |
| prefill paged | 35.27 | 32.34 | **33.95** |
| prefill paged_varlen | 36.04 | 32.18 | **34.18** |

仅 L0C 那一列 decode 与 prefill 差三倍（4–10 vs 31–32），**对照组**——24 个 `L1_to_GM(estimate) == 0` 的 decode contiguous case——给出 87.65 KiB/核（p25 53.9 / p75 115.1），与任何一个数都不符。所以在那一列上"跨路径共享常数"确实立不住，当时的撤回是正确的。

换到权威口径后它重新成立，且数值从 35.59 收到 **32.8–34.3**，与早前用 `w_req − L0C_to_GM` 在 4 个 decode case 上独立得到的 33 KiB/核 吻合。**撤回没有白做：它逼出了正确口径，并把这个数从两个错误构造的巧合变成了一个可复核的量。**

对应的环/实测也随口径变（同一 decode 环）：含 L1 为 0.495–0.529、仅 L0C 为 0.813–0.914、权威为 0.517–0.559。引用任何一个数之前必须先说明口径。

## MLA 的两个环用两个不同的轮转键

`fia_kernel_nonquant_mla.h:498-561` 的 `CalcParams` 分开维护两个计数器：`info.loop` 每 **(task, s2 分块)** 递增一次；`bn2IdxInCurCore` 只在 `isFirstSInnerLoop` 时自增（l.557-561），即每 **(bN2, gS1) head-task** 递增一次。据此：

| buffer | 轮转键 | 出处 |
|---|---|---|
| GQA `mm1Res` / `mm2Res` | `info.loop % PRELOAD_NUM` | `fia_block_cube_nonquant_gqa.h:936` / `:1066` |
| MLA `mm1Res` | `info.loop % preLoadNum` | `fia_block_cube_nonquant_mla.h:787` |
| **MLA `mm2Res`** | **`info.bn2IdxInCurCore % preLoadNum`** | 同上 `:999` |

launch 侧 `tasks_by_worker` 的元素本就是 `(b_n2, g_s1, s2)` 三元组，所以循环里的 `task_index` **就是** `info.loop`。因此 GQA 两个环与 MLA 的 mm1 按 `task_index % PRELOAD_NUM` 取槽是对的；**只有 MLA 的 mm2 用的是 head-task 键**。

已按源码改正：launch 侧对 MLA 的 mm2 改用在 `is_first` 处自增的 head-task 计数器取槽。当时预计"数值影响为零"，实测而非假定的结果是：910B 聚合 8.5742% → 8.5681%、L2 四位小数不变、MLA 流量 MAPE 10.41% → 10.04%。影响确实极小（两个键都被 `PRELOAD_NUM = 2` 截断，且 MLA 的 mm2 每 head-task 只写一次），但环现在在两个键上都跟着源码走。

## 环模型与残差（权威口径）

预测 = 源码环的 distinct extent；残差 = 权威实测 DRAM 写 − 预测环，按 AIC 核归一：

| 组 | n | 环/实测 | 残差 每核 KiB（中位，p25–p75） |
|---|---|---|---|
| decode contiguous | 100 | 0.517 | 32.89（32.50–33.14） |
| decode paged | 28 | 0.521 | 32.90（32.49–33.03） |
| decode paged_varlen | 56 | 0.559 | 32.75（32.17–34.40） |
| prefill chunked_prefill | 26 | 0.965 | 34.32（34.10–34.45） |
| prefill contiguous | 31 | 0.965 | 34.06（33.74–34.30） |
| prefill paged | 28 | 0.950 | 33.95（33.69–34.39） |
| prefill paged_varlen | 9 | 0.965 | 34.18（34.16–34.51） |
| decode mla（regime A） | 6 | 0.885–0.938 | 33.0–33.9 |

**一笔 32.8–34.3 KiB/AIC 核 的固定写，横跨两个不同 kernel、七个 layout×阶段组、核数 1–24、S/KV/B/head_dim/group_size 全域**，四分位极窄。这与早前用完全不同的构造（`w_req − L0C_to_GM`，4 个 decode case）独立得到的 33 KiB/核 一致。

它不是环：环的 slot 是 `group_size × q_len × (s2_base + head_dim) × 4`，每 task 随 group 放大 18→47.5→225 KiB（group 4/8/64），而残差恒为 33.88/33.38/33.36 KiB/核。

**保持不计价。** 它对所有 shape 参数不敏感，因此 `workspaceParams` 的四个字段（全部随 shape 变）按尺寸不变性全部排除；它同时出现在 workspace 布局不同的两个 kernel 上，因此不属于任一算子的 workspace，只能来自二者共享的基础设施。抽取的 CANN 树只含算子源码、不含运行时头文件，无法证实，**不作猜测性归属**。

## 唯一例外：MLA regime B

MLA 以 **B≥8 且 KV≥2048** 为界分两段：

| 段 | case | 环/实测 | 残差 每核 |
|---|---|---|---|
| A（B=1 任意 KV；B8/16/32 KV=512） | 6 | 0.885–0.938 | 33.0–33.9 KiB |
| B（B≥8 且 KV≥2048） | 8 | 0.449–0.703 | 162–550 KiB |

界线的物理含义是**每个 core-task 的 s2 分块数**：KV=512 只有一个分块；B=1 时 FlashDecode 把 KV 切到各核上，每核也只剩约一个分块。两种情况下环从不回绕，"distinct extent = 写入字节"平凡成立，所以模型精确。只有 regime B 真正发生回绕，而实测说明回绕的复用**不是全额免费**——但也远非全额计费（预测写流 146.6 MiB、实测 DRAM 23.4 MiB，仍有 84% 被吸收）。

同段 MLA 的 L2 也同向欠（0.906–0.960）。两个口径同向欠指向源码账本身而非被测量。已排除：轮转键（换 `bn2IdxInCurCore` 仍封顶 2 槽）；L2 容量（该段环总足迹约 12 MiB，远小于 L2）。

**剩余候选**：回绕间隔内的 KV 流量把 slot 挤出 L2，使部分重写落盘。`gm_write_back_residency` 探针测的是无中间流量的重写，得到 `write_hit = 1 − slots/passes` 精确；production 的回绕之间夹着 2048–8192 个位置的 KV 读。判定它需要一条带中间流量扫描的驻留微基准，属采集，本轮未授权，故保留并上报。

## prefill AIV 侧：写流精确，被测量违反物理下界

SplitFuse prefill 四个 layout 对 canonical 口径同向过 1.20–1.23，是 G1 剩余误差的主要来源（68 个非 chunked case，MAPE 28–42%）。把它按 stream 与 unique 拆开：

| layout | n | MTE3 写流 pred/meas | DRAM pred/meas | 预测 unique/stream | 实测 unique/stream |
|---|---|---|---|---|---|
| chunked_prefill | 26 | **1.000** | 1.219 | 0.144 | 0.119 |
| contiguous | 31 | **1.000** | 1.200 | 0.140 | 0.116 |
| paged | 28 | **1.000** | 1.278 | 0.186 | 0.145 |
| paged_varlen | 9 | **1.000** | 1.247 | 0.167 | 0.134 |

**AIV 写流被完全算对（1.000，四个 layout）**，误差全在 unique 那一步。

但被测量这一侧无法作为判据：`gO` 至少被写一次，其足迹是实测 distinct extent 的**硬下界**，而

- **94 个 prefill case 里有 54 个违反该下界**（实测 unique < 输出足迹）；
- 逐 layout：contiguous 21/31（最差 0.733）、paged 22/28（最差 0.704）、paged_varlen 9/9（最差 0.741）、chunked 2/26（最差 0.939）。

最差处实测说一个"写一次"的缓冲只有 70% 到达 DRAM，这在物理上不可能，所以偏差在推导侧而非模型侧。

进一步排除了一个候选：曾怀疑 AIV write 计数器只覆盖 `UB_to_GM` 流的 29%，若采样偏向被反复重写的环（高 hit）会把 unique 推低。若如此，环占比越大偏差应越大；实测相反——环占比**低**的一半 `meas/out` 中位 0.850（违反下界），环占比**高**的一半 1.108（不违反）。**该候选证伪。**

残留候选是 profile 窗口：输出在 kernel 末尾写出，窗口关闭时仍脏在 L2 的部分在窗口之后才回写。输出占流比例越高的 case 越晚写完、违反越重，与观测方向一致。判定它需要窗口边界侧的证据，不在本 Goal 的模型侧范围内。

**结论：prefill AIV 的 1.20–1.28 不能全部记作模型误差，其中至多 30% 是被测量在写一次缓冲上的亏空。** 在被测量这一侧修好之前，不对该项继续调参——任何按此残差的改动都是对一个违反物理下界的量拟合。

## 33 KiB/核 常数的归属：AscendC 框架的 KFC workspace 清零

前面把这笔固定写标为"源码无归属、保持不计价"，依据是 FIA 算子源码里找不到写者。**它确实不在算子源码里——它在 AscendC 框架里。**

`/usr/local/Ascend/cann-8.5.1/aarch64-linux/tikcpp/tikcfw/impl/dav_c220/kfc/kfc_comm.h:357` 的 `ClearWorkspaceImpl`：每个 MIX kernel 入口，**每个核清零自己的 KFC 消息区**

```
msgStartAddr = workspace + (size + msgOffset11) * GetBlockIdxImpl()
size = BIDIRECTION_NUM * MAX_MSG_COUNT * AlignTo32(sizeof(KfcMsg)) * MIX_NUM
block = size / 2048           // 16 次 copy_cbuf_to_gm，每次 2048 B
```

代入头文件常数（`MAX_MSG_COUNT = 64`，`kernel_macros.h:88`；`BIDIRECTION_NUM = 2`；`MIX_NUM = 2`；`sizeof(KfcMsg) = 4+4+120 = 128`；`sizeof(MsgUBAvalied) = 64`）：

- `size = 2 × 64 × 128 × 2 = 32768 B`（`block = 16` 整除，佐证 128）
- `sizeUbmsg = 2 × 64 = 128 B`
- **每核 32896 B = 32.125 KiB**

与实测残差（权威口径，实测 AIC DRAM 写减预测环，按 AIC 核归一）对比：

| 组 | n | 残差/核 | 残差 / 32896 B |
|---|---|---|---|
| decode contiguous | 100 | 32.89 KiB | **1.024** |
| decode paged | 28 | 32.90 KiB | **1.024** |
| decode paged_varlen | 56 | 32.75 KiB | **1.020** |
| prefill chunked_prefill | 26 | 34.32 KiB | 1.068 |
| prefill contiguous | 31 | 34.06 KiB | 1.060 |
| prefill paged | 28 | 33.95 KiB | 1.057 |
| prefill paged_varlen | 9 | 34.18 KiB | 1.064 |
| **全部非 MLA** | **278** | **33.20 KiB** | **1.033** |

**它是 `copy_cbuf_to_gm`，即 L1→GM，不是 Fixpipe 的 L0C→GM。** 这一条同时解释了之前三件对不上的事：

1. 为什么权威写计数比 `L0C_to_GM` 多出约 33 KiB/核——这笔写根本不走 L0C。
2. 为什么 FIA 算子源码里找不到写者——它由框架在 kernel 入口发出，不属于任何算子的 `workspaceParams`。
3. 为什么它对 S/KV/B/head_dim/group_size/layout 全不敏感而只随核数走——常数全部来自头文件宏，与 shape 无关。

同时订正一处早前的推理：曾以"FIA cube 块没有任何 L1→GM 写路径"为由判定 `L1_to_GM_datas` 不可入账。**前半句对算子源码成立，但对框架不成立**——`ClearWorkspaceImpl` 正是一条 L1→GM 写。该字段仍因带 `(estimate)` 且量级与本项对不上（prefill 每核 116–332 KiB）而不宜直接使用，但"没有 L1→GM 写路径"这个理由本身是错的，此处订正。

**尚未计价。** 依据完全来自头文件常数、与误差无关，满足零拟合，但它既不是 MTE3（AIV）也不是 Fixpipe（AIC L0C），挂在现有任一条 path 上都会把两个引擎混为一谈；正确做法是新增一条 path，而 `_attach_cann_standard_decode_phase_ledger` 会逐 phase 校验 path 节点集合，框架清零发生在任何 phase 之前，需要一并处理。留作下一步。
