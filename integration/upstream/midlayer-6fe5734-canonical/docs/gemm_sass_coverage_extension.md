# GEMM SASS Launch-Coverage 扩展：常量推导方法论边界

本文记录 A800 GEMM predictor（`src/kernel/predictors/gemm.py`）SASS-pinned fixed-issue 常量族的一次扩展尝试：目标 launch 组合的 provenance 已经确认，但常量本身未落地，因为交叉验证暴露出批准该路线时的前提有误。通用 Kernel 模型和证据规则见 [Kernel 解析性能模型](kernel_performance_model.md)；A800 VECTOR/NORM track 见 [`vector_launch_floor_evidence_grade.md`](vector_launch_floor_evidence_grade.md)。

全程 `src/kernel/predictors/gemm.py` 只读，未做任何代码改动。

## 1. 背景：ceiling 定位与目标 launch 组合

按 A800 kernel evaluation 同一口径分桶，GEMM 桶 221 个 case、MAPE 6.35%，对 ALL 贡献 1.366pp（完整表见 `vector_launch_floor_evidence_grade.md` "收益上限先算"一节）。对 `_sm80_depth5_fixed_non_hmma_warp_instructions` / `_sm80_depth6_fixed_non_hmma_warp_instructions` 这一 SASS-pinned fixed-issue 常量族的 guard 组合做穷尽核对，定位到未覆盖组合中 case 数最大的一个：`tile=(64,64,64)`、`depth=6`、`split_k=1`。组合外的其余情况保持现有（未改动）行为，不做近似或静默 fallback。

## 2. Provenance 单位更正：四元组，不是"同一份 cubin"

批准这条路线时的前提是"纯静态反汇编、不吃卡"且"新常量与既有常量共用同一份 cubin"，两者都被本轮证据推翻（后者见本节，前者见 §5–§6）。

`libcublasLt.so.12` 是 fatbinary：`cuobjdump -xelf all` 一次性提取出 1942 个内嵌 sm_80/sm_86/sm_90 等 ELF/cubin object。已确认现有常量本就跨两份不同 cubin（索引 1905 的 SHA `b3ca549...` 下 64x64 符号数为 0，说明现有 64x64 常量不可能来自 1905；它们实际来自索引 1906）。"所有常量共用一份 cubin"这个提法因此不成立。

更正后的 provenance 单位是四元组：**(.so SHA256, cubin 文件名, cubin SHA256, 精确符号名)**。本轮验证用的源 `.so`：

```
/usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublasLt.so.12.2.5.6
SHA256: fbfbc8a1faf681fa5f3ed3ab86d456e196058e9a6b601c930d200dcd2822b1ae
```

对该文件重跑 `cuobjdump -xelf all` 复核：共提取 1942 个 object，索引 1906/1907 的 cubin SHA256 与下方 §3–§4 引用的完全一致——本节的四元组是本轮新鲜验证过的，不是转述早前记忆。

## 3. 登记项更正：depth5/64x64 常量的 provenance 缺口不存在

现有 `_SM80_64X64_K64_DEPTH5_FIXED_NON_HMMA_WARP_INSTRUCTIONS` 的 docstring 引用符号 `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`，此前一度怀疑该符号的 non-relu 版本"找不到"、构成 provenance 缺口。核实后**该缺口不存在**：只是此前只看过索引 1905/1906 中的 1905，符号实际在 1906 里。

四元组：

```
.so:    /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublasLt.so.12.2.5.6
.so SHA256: fbfbc8a1faf681fa5f3ed3ab86d456e196058e9a6b601c930d200dcd2822b1ae
cubin:  libcublasLt.so.12.2.5.1906.sm_80.cubin
cubin SHA256: 06008e27431fd0cef94019bd3a00b552eedd883a689a619310e8ddae706cb226
symbol: ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn
```

已按要求额外核实：索引 1907（cubin SHA `d2e5e607...`）中**不存在**该 non-relu tn 符号——1907 只有该符号的 `_stages_64x6_{tn,tt}`（depth6）版本，没有 `_stages_64x5_tn`（depth5）版本。登记结论保持不变：**常量本身没问题，此前记录里只是漏了它属于哪份 cubin。**

## 4. 目标常量的 provenance（已确认，常量本身未落地）

502 个已确认的 sm_80 cubin 穷尽扫描，目标符号只存在于唯一一份：

```
.so:    /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcublasLt.so.12.2.5.6
.so SHA256: fbfbc8a1faf681fa5f3ed3ab86d456e196058e9a6b601c930d200dcd2822b1ae
cubin:  libcublasLt.so.12.2.5.1907.sm_80.cubin
cubin SHA256: d2e5e607399c521a4cdb592cb6c442e25822cc85dd725df71e94c6d7b0a2dd5e
symbol: ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn
```

周边 cubin 分布（同一次扫描）：1905（`b3ca549...`）0 个 64x64 符号；1906（`06008e27...`）只有该 depth6/sliced1x2 家族的 fp16 non-relu `nn`/`nt`，缺 `tn`/`tt`；1934/1935 是 bf16（`5069e4cb...`/`805a88ae...`），dtype 与目标 fp16 不符。`tn` 变体只存在于 1907。

该函数的 K64-tile 循环地址范围为 `0x1500`–`0x1a40`，与既有 depth5/64x64 常量引用的循环地址范围完全相同（同一份汇编器输出布局）。

## 5. 方法论交叉验证：循环体计数精确复现

计数方法：循环体 = 标号（`.L_x_N:`）到其反向分支（含 `BRA` 与 `BRA.U` 两种助记符）之间的全部指令；总数减去 `HMMA` 助记符行数即为 non-HMMA 数；乘以 CTA 内活跃 warp 数（`block=(128,1,1)` = 4 warp）得到 docstring 所称的"per K64 tile across four warps"。

用该方法对**已有、已发布**的 depth5/64x64 常量做复现：循环体（cubin 1906，`0x1500`–`0x1a40`）85 条指令/迭代 = 32 HMMA + 53 non-HMMA；`53 × 4 = 212`，与该常量 docstring 记载的"212 non-HMMA warp instructions per K64 tile"**精确一致**。方法论在循环体这一层被验证为可复现。

对目标 depth6/64x64（cubin 1907）应用同一方法：循环体同样是 `0x1500`–`0x1a40`，85 条指令/迭代，**同样是 32 HMMA + 53 non-HMMA**，即同样得到 212 non-HMMA warp-instructions/K64-tile。

## 6. 方法论交叉验证：真正要落地的"fixed"常量不可复现

既有函数实际编码进公式的不是循环体那个 per-K64-tile 数字（那只是 docstring 的叙述，代码里没有单独乘算），而是循环体之外、prologue+epilogue 的 **fixed** 常量（depth5/64x64 为 3076，depth6/128x64 为 4880）。对这个数字尝试同一套"静态计数 × 4 warp"方法：

- depth5/64x64 全函数：768 条指令，32 HMMA，736 non-HMMA。
- 减去循环体单次出现的 53 条 non-HMMA，得到循环外静态 683 条 non-HMMA；`683 × 4 = 2732`。
- 已发布/已编码的常量是 **3076**，与静态方法算出的 2732 相差 **344 条（约 11%）**，且这 344 条差值方向无法用"少算了某类 warp"简单解释（静态方法本身即偏低）。

深度对照 depth6/64x64 目标函数，发现它的全函数静态结构与 depth5/64x64 **逐指令完全相同**：768 条指令、32 HMMA、736 non-HMMA，连内部分支/标号拓扑也一致（例如 depth5 里 `.L_x_2916`、depth6 里对应位置 `.L_x_109` 都是同一种自引用回边标号，指示 epilogue 内部存在一个静态计数看不见真实执行次数的子循环）。这意味着同一个 344 条的未解释缺口会对称地出现在 depth6/64x64 的 fixed 常量上——单靠反汇编无法区分"这个子循环真实执行了几次"。

## 7. 结论：批准前提被推翻，这条路线是 A800-blocked

两个既有 docstring 都写明各自的 fixed 常量是对着一个真实的 **canonical non-duration `inst_executed` 计数器**收口的（depth5/64x64："closes exactly at 24,836 instructions per CTA"；depth6/128x64："canonical non-duration counters close exactly at both 128 and 448 K64 tiles"）。这是硬件 profiler 计数器交叉验证，不是纯静态反汇编——当初批准这条路线时"纯静态、不吃卡"的前提本身就不成立。静态方法与已发布常量之间那 344 条的差，正是这一步交叉验证要解决、而静态反汇编本身看不见的动态部分：warp 特化的分叉路径、epilogue 内部子循环的真实 trip count。

**不做更深的纯静态结构推导（不带计数器验证）**：即使把 epilogue 的分支拓扑和子循环全部人工追出来，结果也无法对着真实计数器自验；更关键的是它有一个明显的失败模式——最终数值很容易收敛到 3076 附近（因为只有那样才与既有常量"自洽"），而这正是"照抄既有常量的 gap 去凑新常量，而不是独立推导"，与零拟合红线冲突。因此这一步不尝试。

这条路线因此与本 worktree 此前已 parked 的三项工作（NORM 0.211pp round-pricing 上限、SASS-derived assembly 组装修复、`add_rmsnorm` H-dependent residual）性质相同：**结论已经查清、下一步动作已经明确，但需要一台空闲 A800 才能继续**。重开条件：拿到空闲 A800，跑与既有 3076、4880 两个常量同样的 `inst_executed` 非-duration 计数器交叉验证。

## 8. 给下一次继续者的起点

- §2–§4 的四元组 provenance 已经确认，重新拿起这项工作时不需要重做 502/502 扫描或重新核实 1905/1906/1907/1934/1935 的符号分布。
- §5 的循环体计数方法（标号到反向分支、`BRA`/`BRA.U` 都要匹配、绝对行号而非 `sed` 子区间相对行号）已交叉验证，可直接复用；目标函数循环体的 212 non-HMMA warp-instructions/K64-tile 数字可信。
- §6 的静态全函数计数（768/32/736，depth5 与 depth6/64x64 完全相同）已经算好，是下一次做计数器交叉验证时的起点，不必重新反汇编。
- 真正要拿到的数字是目标函数的 fixed 常量本身，只能通过对一个已知 `(b,m,n,k)` 形状跑 ncu/nsys 拿到 `inst_executed` 非-duration 计数器、按 §5–§6 的分解方式解出来——不能假定它等于 3076 或 3076+344，那两者都是"照抄既有数再凑"。

## 9. 状态

GEMM SASS-coverage 扩展本轮转入 parked 队列。A800 上现有四项工作（NORM round-pricing 上限、SASS-derived assembly 修复、`add_rmsnorm` H-dependent residual、本 GEMM 常量扩展）现在全部等待同一台空闲 A800。三卡 before/after 未运行——没有代码改动可供比较。
