# Ascend 910B microbenchmarks

本目录使用独立微基准测量 Ascend 910B 的单一硬件机制，为 `HWSpec`、launch selection 和 Kernel 模型提供可复测证据。微基准不得运行完整 production 算子，也不得从 production/evaluation kernel duration 反解参数。

通用零拟合、机器访问和硬件任务规则见仓库根目录 `AGENTS.md`；参数能否进入 Forward 见 [`docs/kernel_performance_model.md`](../../docs/kernel_performance_model.md) 第 10 节。

## 目录

| 路径 | 测量内容 |
|---|---|
| `compute/vector_pipe/` | UB-resident Vector opcode/request service |
| `memory/dram/bandwidth/` | GM/HBM streaming bandwidth sweep |
| `memory/dram/latency/` | dependent GM/L2 pointer chase |
| `memory/dram/mte2_tile/` | 基础 GM→UB MTE2 tile service |
| `memory/dram/mte2_tile/{finite_drain,gemm_source_order,grouped_pair,l1_program,nd2nz}/` | MTE2 finite-drain、GEMM source-order、operand-pair、L1 program 与 ND2NZ 机制 |
| `memory/dram/mte3_tile/` | UB→GM MTE3 tile service |
| `memory/dram/aiv_workspace_transfer/` | CANN FIA Vec1/Vec2 短 workspace transfer 原语、有序链及 primary-AIV completion |
| `memory/fixpipe/` | FixPipe 独立 service candidate |
| `operators/gemm/single_core_split_k_tile/` | MatMulV3 key-65632 resident MTE1/Cube/source-control tile service |
| `operators/flash_attn/` | FlashAttention tiling/metadata launch facts |
| `operators/fused_moe/` | Fused-MoE tiling 与 scatter/MTE2 机制 |
| `operators/norm/` | ReduceSum 与 Norm reduction 机制 |
| `operators/swiglu/` | SwiGLU tiling facts |
| `operators/vector/softmax/` | UB softmax-vector 机制实验 |
| `operators/vector/elementwise_control_lifecycle/` | 48-AIV elementwise DAG/TBuf/mutex source-control lifecycle 候选 |
| `operators/vector/elementwise_single_tile_program/` | 47-AIV 单 tile elementwise 2R1W memory/control 完整 worker span |
| `runtime/cross_domain/` | AIC↔AIV cross-domain handoff |
| `runtime/fia_edge_handoff/` | FIA 三条 AIC↔AIV CrossCore 依赖边的独立 completion latency |
| `runtime/fia_program_span/` | CANN FIA 非空三槽 controller 与三次 handoff 的完整 program-span 候选 |
| `runtime/launch/` | AIC/AIV/mixed empty task 与 minimal non-empty AIV task |
| `references/` | 可由仓库复核的紧凑 reference 与硬件资料 |

`runtime/launch/fallback_dispatch_runner.py` 是独立的 fallback-dispatch 微基准：它只测量 synthetic NPU tensor 上重复 `aten.reshape` metadata 节点的 Host wall service，以 chain-length sweep 拟合每节点成本并保留 512-node holdout；不读取 Serving duration，也不复刻 production graph。当前 TP2 两个物理 lane 的结果与 source occurrence contract 见 `references/compiled_fx_reshape_metadata_cann_8_5_1.json`，生产只在该 reference 声明的 Ascend Qwen3 MoE TP2 图结构上消费。


`runtime/launch/compiled_fx_metadata_runner.py` 的同一固定 sweep 还采集了 dense fallback 图中 source-authored `aten.view` 与 `aten.empty.memory_format` metadata 节点；两 lane 的独立结果与 Qwen3/Llama occurrence contract 见 `references/compiled_fx_fallback_metadata_cann_8_5_1.json`。dense contract 只在 layer-0 加重复 layer-1 且 attention split children 总数守恒的图上生效，MoE 图不消费该机制。

`runtime/launch/compiled_fx_executor_runner.py` 测量 synthetic opaque compiled-FX backend entries 的 Host service；设备 0/1 的独立结果与 Llama native-composite RoPE helper occurrence contract 见 `references/compiled_fx_rotary_fallback_dispatch_cann_8_5_1.json`。该 contract 对每个 source `RotaryEmbedding.forward_native` group 只收费一个 `rotary-cos-sin-gather` anchor，`rotary-*` 子调用仍由普通 eager submit 机制负责，已注册或 fused RoPE 路径不产生该 fallback entry。

`runtime/launch/model_runner_outer_prepare_runner.py` 测量 `NPUModelRunner` 外层 input、attention-metadata binding 和 logits-index preparation 的 host-only source service；它不执行模型或 NPU operator。源码快照、ranked/unranked token-embedding/LM-head anchor contract 与独立容器产物见 `references/model_runner_outer_prepare_cann_8_5_1.json`，生产事件在图 replay 中保持 graph-external，并在缺失任一配对 anchor 时 fail-closed。

`runtime/launch/graph_replay_sync_runner.py` 测量独立 synthetic NPU work 后的 `torch.npu.current_stream().synchronize()` API 基线，并验证 pending-device barrier 行为；设备 0/1 结果与 vLLM-Ascend `ACLGraphWrapper` 的一 replay 一同步 occurrence contract 见 `references/graph_replay_sync_cann_8_5_1.json`。Forward 只消费源码声明的 barrier 语义：等待已排队 device predecessor，再计现有 graph-replay enqueue service；不会把 serving residual 或 pending-work duration 拟合成常数。

每个可编译 leaf 自己维护 `CMakeLists.txt`；runner 和 report 的参数以各脚本 `--help` 为准。构建产物必须放在 leaf 的 ignored `build/` 或外部临时目录，不能提交到仓库。

## 当前进入 Production Forward 的证据

只有下表中的已发布机制会被 production 路径消费。具体数值、单位、适用范围和 provenance 由 reference/HWSpec 保存，README 不复制第二份。

| 机制 | Canonical evidence | 消费位置 |
|---|---|---|
| NORM serialized Sqrt/Duplicate/Div request service | `references/vector_norm_request_cann_8_5_1.json` | `configs/hardware/ascend_910b.yaml` 的 typed Vector service |
| Cast/Muls/Adds/Mul serialized Vector service | `references/vector_serialized_services_extended_cann_8_5_1.json` | `configs/hardware/ascend_910b.yaml` 的 Vector affine service |
| AddRMSNorm key-30 ReduceSumFP32 service | `references/add_rmsnorm_reduce_sum_fp32_cann_8_5_1.json` | `configs/hardware/ascend_910b.yaml` 的 Vector affine service |
| AddRMSNorm key-30 32-AIV H2048/H2560 input-pair completion | `references/aiv_add_rmsnorm_row_group_input_pair_completion_32_workers_cann_8_5_1.json` | key-30 每行 `input_load` 与同阶段 byte capacity 取最大值 |
| Vector-to-MTE2 fence service | `references/vector_to_mte2_fence_cann_8_5_1.json` | CANN AddRMSNorm key-30 每行 source DAG |
| RMSNorm 首个 DataCopyPad MTE2→Vector completion | `references/aiv_datacopypad_mte2_to_vector_completion_{32,43,47}_workers_cann_8_5_1.json` | CANN key-0 row-group 首次 gamma load 的精确 AIV admission 域 |
| AIC/AIV/mixed empty-task floor | `runtime/launch/` retained capture | `configs/hardware/ascend_910b.yaml` 的 scheduling LUT |
| minimal non-empty AIV task floor | `runtime/launch/` retained capture/report | `aiv_nonempty_task_latency_model` |
| ACL graph replay pre-replay synchronization barrier | `references/graph_replay_sync_cann_8_5_1.json` | Ascend `RuntimeGraphSegmentDescriptor.sync_before_replay` device-timeline barrier |
| FIA three-handoff program span | `references/fia_cross_domain_three_handoff_cann_8_5_1.json` 与 retained report | `fia_cross_domain_three_handoff` |
| Vector ReduceSum request latency | `operators/norm/` retained capture | `vector_reduce_sum_latency_lut` |
| CANN tiling/metadata facts | `references/*tiling*.json` 与对应 capture runner | Ascend launch package |
| AIC serialized global scalar read | `references/aic_scalar_global_get_value_cann_8_5_1.json` | paged SplitFuse QK/PV descriptor address dependency |
| GroupedMatmul K2880 base-M128 per-output-tile completion | `references/grouped_matmul_k2880_output_tile_completion_cann_8_5_1.json` | CANN MoE 每次 `SetSingleShape`/`IterateAll` 的有限流下界 |

DRAM latency/bandwidth、MTE2/MTE3、FixPipe、softmax 和其他 operator experiments 默认只是候选或诊断工具。未被当前 reference/HWSpec 明确引用，就不能认为已发布，也不能被 predictor 静默读取。

## 采集合同

一次可发布采集必须满足：

1. 使用 canonical checkout 的明确 commit 或经审计 source snapshot；
2. 通过 `ssh 910b-1` 进入宿主机，在容器 `yxh_midlayer_serving_910b1` 的约定工程副本执行，并在开始前确认目标 NPU 空闲；
3. benchmark 只隔离一个物理机制，timed region 不混入无关初始化、host 同步或完整算子；
4. 保留 source、binary/compiler/CANN 身份、命令、device telemetry、逐 trial raw、数值校验和退出状态；
5. 使用预先固定的 sweep、trial/order 和 publish gate，不能根据结果反选点、阈值或公式；
6. 失败必须明确输出不可发布，不能挑选部分 cell、建立 per-shape LUT 或退回旧参数；
7. benchmark 结束后确认子进程退出且设备状态恢复。

原始产物统一写入：

```text
profiler_output/microbench/ascend_910b/<run-id>/
```

该目录保存完整 raw provenance，不由 Git 跟踪。需要进入 production 的紧凑证据放入 `references/` 或 `configs/hardware/ascend_910b.yaml`，并明确绑定 raw report identity。

## 发布流程

```text
source/contract freeze
        ↓
isolated hardware capture
        ↓
report validates raw + provenance
        ↓
candidate reference
        ↓
consumer/source audit
        ↓
HWSpec or launch package publication
        ↓
repository tests and evaluation
```

硬件任务只产生 raw 和 candidate；candidate 不会自动进入 Forward。发布前必须确认 consumer 的 dtype、operation count、task/worker mapping、timer domain 和适用范围与 benchmark 完全一致。无法证明映射时，参数保持 unavailable，并由 predictor 报告 limitation 或 fail loud。

历史 run、失败结论、checksum 和一次性排障不在本 README 维护；它们由 retained raw、Git 历史和对应 report 保存，避免把某次实验状态误当成长期设计合同。

## FIA nonempty controller program span

`runtime/fia_program_span/` 复刻固定 CANN 8.5.1 regular-GQA FIA 的三槽 `RunInfo` task cache、`CreateTask`/`ShouldExecuteTask`/`ExecuteTask` controller 和 MM1→Vec1→MM2→Vec2 三次 cross-domain handoff。计时边界从 task-cache 初始化之前开始，到最后一个有效 Vec2 actor-local body 返回结束；四个 stage 只执行会被 checksum 保留的最小 scalar work，使 source branch 非空但不复刻完整 Attention 算术、tensor traffic 或 production kernel。冻结训练 sweep 是 active AIC groups `{1,8,16,24}` 与每组 task count `{1,2,3,10,11}` 的笛卡尔积；预注册 holdout 只使用 22-group 的 single-task、`20×3+2×2` 和 `11×11+11×10` tail topology。每个 measured cell 前固定执行一次 untimed 24-group/11-task scheduler preamble 和两次 exact-cell warmup，以消除 host-global mixed-task scheduler 的前序 grid 状态；三次独立 run 均使用六轮 forward/reverse position-balanced 顺序，每个 cell 的 cross-run span 与 position-order error 都必须不超过 20%，否则候选不得进入 Forward。

该 probe 的结果单位为 `ns_per_complete_aic_group_program_span`，只能作为 source-bound main DAG 的 end-to-end 下界来替换旧 empty-ring span；不得与现有 task DAG 或 mixed-task outer envelope 相加。report 预先冻结 local/shared depth-survival 公式及 exact active-group/task-count domain，并独立检查 intermediate-count 和 22-group tail holdout；即使机制 gate 通过，也必须先证明目标 decode bucket 改善且其他 FlashAttention case 不回归，才能发布到 HWSpec。

三轮 CANN 8.5.1 nonempty campaign 的 cross-run 与 validation gate 通过，但两个 position-balance cell 超过冻结的 20% 上限，因此该 span 明确不可发布、不可进入 Forward，且不得挑选稳定 cell 输出 service、参数或 LUT；`references/fia_nonempty_program_span_negative_diagnostic_cann_8_5_1.json` 只保存可复核的 negative diagnostic identity 与 gate 结论。当前已发布的 empty source-bound FIA three-handoff service 不含 nonempty actor body，不得外推为本 nonempty controller span。

## FIA cross-domain edge completion

`runtime/fia_edge_handoff/` 在 `KERNEL_TYPE_MIX_AIC_1_2` 中按 CANN 8.5.1 固定次序执行 C1→V1、V1→C2、C2→V2 三条 CrossCore 依赖边，并以 device-global 50 MHz `GetSystemCycle` 分别记录 producer 与 consumer 时间戳之差。计时区不执行 Attention 算术、tensor 搬运或 production operator；每个 actor 仅在全部依赖边结束后写回时间戳和完整 ring elapsed。v2 冻结 campaign 覆盖 active AIC groups `{1,8,11,12,16,18,19,21,22,23,24}` 与 task count `{4,8,16,32,64,92,128}`，并直接保留前三次 completion 与后续 steady completion，三次独立 capture 各使用六轮 forward/reverse position-balanced 顺序。只有 provenance、跨 run、跨 task count、顺序、held-out run 和完整 ring 下界 gate 全部通过，结果才可作为逐边 completion service；它不能作为完整 FIA task 或 production kernel duration。

首轮 v1 capture 的稳态跨 run span 不超过 2.6%，但把三槽 ring fill 压缩成单一 startup 的 reduction 在 holdout `B24/N4/C2→V2` 上达到 38.5% APE，且 V1→C2 双 sender 并不满足错误预设的 lane 对称性，因此整项 v1 不可发布；identity 与失败 gate 固化在 `references/fia_cross_domain_edge_completion_v1_negative_diagnostic_cann_8_5_1.json`，不得从中择取子集进入 Forward。

v2 直接保留前三次 completion 后，完整 ring 下界 gate 通过，但单边 latency 仍随同一 active-group 下的 task count 和执行顺序显著变化：最大 task-count span 为 162.9%，平衡顺序 P95 为 72.0%，独立 validation 最大 APE 为 116.2%。因此单边 handoff 不是可脱离 mixed-core scheduler 状态复用的稳定 service；`references/fia_cross_domain_edge_completion_v2_negative_diagnostic_cann_8_5_1.json` 固化该反证，禁止把任意 topology cell 或稳定子集发布到 Forward。

## MatMulV3 large-M diagnostic boundary

`operators/gemm/tiling_query_runner.py` 的四次 launch-only capture 与 CANN 8.5.1 静态源码共同确认 large-effective-M case 分属两条 legacy 机制：key 65632 是 AIC:AIV=1:2 的 single-core Split-K GM-to-L1 mixed kernel，key 65536 是 AIC-only base kernel。capture 使用的 `Mc2MatmulV3TilingData` 在 50-word `TCubeTiling` 后依次编码 L2 tile 和 run-info，不包含 arch35 Stream-K tail；M4096 capture 确认 key 65632 的 `singleCore=(512,2816,384)`、`base=(128,128,128)`、depth `(9,6)`、step `(3,1)`、L2 tile `(8,8,4,4,calOrder=0)` 和 `4*M*N + 20,972,032 bytes` workspace。对应 kernel source 显示每个 AIC 独占完整 M×N 宏块并按 K=384 串行累加 FP32 workspace，两个配对 AIV 在每个 inner-M stripe 完成后执行 cast/store；不存在跨 AIC 的 Stream-K tail reduction。aligned stripe 路径只有 AIC `NotifyEvent<PIPE_FIX>(5)` 到 AIV `WaitFlagDevLocal(5)` 的单向边，没有逐 stripe AIV→AIC completion；workspace offset 由 M macro tile、inner-M stripe 与 N macro tile 唯一决定，连续 stripe 不覆盖同一区域。因此 `reverse_aiv_to_aic_completion_per_stripe` 与 `workspace_tile_overwrite_requires_release` 两个候选均被源码否定，不能据此增加 completion wait；该簇剩余误差保留为 mechanism-level WONTFIX。key 65536 的 source-order generic LRU 仍不能复现 retained non-duration traffic/cache counters，因此其 L2 residency correction 保持 partial unsupported。完整 capture identity、source/binary SHA256、payload ABI、候选判定与 LRU/counter gap 见 `references/matmul_v3_large_m_diagnostic_cann_8_5_1.json`。

`operators/gemm/single_core_split_k_tile/` 进一步隔离 key-65632 每个 `AMatMulB` 的 resident `128×128×{384,256}` 本地完整 K tile/tail 程序：输入在计时前进入 L1，计时区保留 `SetSingleShape`、A1/B1 tensor binding、`Iterate`、M-to-FixPipe handoff、单个 FP32 workspace tile store、`End` 与 ping/pong L0C 生命周期。它不包含 GM→L1、AIV、production 外层循环或完整 GEMM；冻结 sweep 为 24 AIC 上每个 K extent 的 `{64,256,1024,4096}` 次迭代，只有三轮正反序结果通过正差分和 10% 跨 run 稳定门禁后才可发布。

`memory/dram/mte2_tile/gemm_source_order/aic_mte2_gemm_region_runner.py` 的 `long`、`k5120-long` 和 `k13824-long` profiles 分别重放 key-65536 large-region `256×112, K8192`、`256×54, K5120` 和 `256×20, K13824` 的完整离散 worker schedule，并独立测量 MTE2、MTE2→MTE1 与 MTE2→MTE1→Cube spans。MTE2-only spans 未超过对应统一解析 pipeline；后两种 probe 会主动 drain producer/consumer queues，不能充当 production MatmulImpl 的 completion floor。精确 program、三轮结果、raw/source/binary identity 与否定边界见 `references/matmul_v3_large_m_diagnostic_cann_8_5_1.json`。

同目录的 `k1536_m16_n224` profile 隔离 23 AIC、A1 `16×1536` 单 descriptor 与 B1 `224×256` 六 descriptor 的非对称有限 source program，并保留 MTE1/Cube consumer。MTE2-only span 不构成缺失 service，drained MTE1/Cube completion 不代表 production overlap；该 cell 仅作为 validation-only 否定证据，见 `references/aic_matmul_v2_k1536_narrow_pipeline_completion_cann_8_5_1.json`。

## AddRMSNorm Split-D launch probe

`operators/norm/add_rmsnorm_launch_{application,interposer,report}.*` 对 BF16、H=16384 的 T=1/64/256/1024 执行四个固定的 launch-only probe。interposer 在 `aclnnAddRmsNorm` execute 边界捕获 `aclrtLaunchKernelWithHostArgs` 的 host args、placeholder map、runtime kernel name 和 `block_dim`，并返回成功而不提交目标 device kernel；report 使用 `AddRmsNorm` runtime reflection 严格解码 80-byte `AddRMSNormTilingData`，拒绝非 key-31 kernel、字段布局变化和重复执行不一致。该 probe 不读取 duration、counter、利用率或 evaluation error。

执行前按根合同确认设备空闲并显式绑定 physical device。先用 `operators/flash_attn/tiling_reflection_runner.cpp` 对 `libophost_nn.so` 的 `AddRmsNorm` registration key 生成 reflection TSV，再编译 interposer，以 `ASCEND_RT_VISIBLE_DEVICES=<physical-id>` 和 `LD_PRELOAD=<interposer>` 运行 application，最后调用 report；命令行参数以 application/report 的 `--help` 和 interposer 源码常量为准。原始产物必须写入 `profiler_output/microbench/ascend_910b/<run-id>/`，紧凑的已验证离散事实见 `references/add_rmsnorm_splitd_launch_cann_8_5_1.json`。

## AddRMSNorm SingleN input-pair completion probe

`operators/norm/add_rmsnorm_input_pair_{kernel,main,runner,report}.*` 隔离 CANN 8.5.1 AddRmsNorm SingleN BF16 路径开头的两个等长 GM→UB `DataCopyPad` 请求及其两个 `MTE2_V` completion event。冻结 sweep 对每个 AIV worker 发出两个 `payload_bytes` 相同的请求，只允许 `(block_dim,payload_bytes)` 精确匹配 `{1,2} × {4096,5120,8192,10240,16384}`；结果单位是一次双请求 launch completion 的 ns，不能拆成两个独立请求相加。probe 不执行完整 AddRMSNorm，也不消费 production/evaluation duration；production 在两个 wait 之间还有首路 BF16→FP32 Cast，因此本结果仅是 pair-completion 机制候选，尚未接入 predictor 或 HWSpec。

可发布 evidence 必须来自三个不同 run ID，每轮 120 个 position-balanced trial；report 对每个 trial/worker 使用 `request_interval(pair_order=1)-control_interval(pair_order=0)`，再取 launch 内最慢 worker，并要求 within-run parity replication 和对应 cell 的 cross-run median spread 不超过 20%。已通过门禁的 CANN 8.5.1 reference、raw/source/binary SHA256、精确单位与 lookup 范围见 `references/aiv_add_rmsnorm_single_n_input_pair_completion_cann_8_5_1.json`。

`operators/norm/add_rmsnorm_row_group_input_pair/report.py` 只发布 key-30、32 AIV、BF16 H2048/H2560 的 4096/5120-byte-per-input exact cells。它必须从原始 10-cell position-balanced CSV 审计这些 cell，拒绝预过滤 CSV，并对三次独立 run 分别执行两轮完整 schedule 的 parity gate；同一 campaign 中未通过复现门禁的其他 payload 不进入 reference 或 Forward。

## FIA AIV workspace transfer

`memory/dram/aiv_workspace_transfer/` 隔离 CANN 8.5.1 `FiaBlockVecNonQuant::DealBmm1ResBaseBlock` 与 `DealBmm2ResBaseBlock` 的短 workspace 搬运机制。Vec1 cell 分别测 4096 B GM→UB、2048 B UB→GM 和二者的有序链；Vec2 cell 分别测 256 B 状态读、256 B 状态写和带 `MTE3_MTE2` event/wait 的有序读写链。所有 cell 保持真实 `KERNEL_TYPE_MIX_AIC_1_2` 拓扑；针对 D64、M=1 源码分工，primary AIV 执行 Vec1/Vec2，secondary AIV 保持 `vecDealM=0` 且由 host 校验没有执行主数据链。它不执行 attention、MatMul 或 production workload。

先在经审计的 910B source snapshot 中记录 runner 计算出的 snapshot SHA256：

```bash
python microbench/ascend_910b/memory/dram/aiv_workspace_transfer/aiv_workspace_transfer_runner.py snapshot
```

按根 `AGENTS.md` 从低编号向高编号扫描并确认空闲卡后，显式传入实际 physical/logical device ID、完整 source commit 和 snapshot；输出目录必须位于 `profiler_output/.../ascend_910b/` 且预先不存在：

```bash
python microbench/ascend_910b/memory/dram/aiv_workspace_transfer/aiv_workspace_transfer_runner.py run --run-id <run-id> --source-commit <40-hex-commit> --expected-snapshot-sha256 <64-hex-sha256> --physical-device-id <physical-id> --output-dir profiler_output/microbench/ascend_910b/<run-id>
python microbench/ascend_910b/memory/dram/aiv_workspace_transfer/aiv_workspace_transfer_report.py --capture-dir profiler_output/microbench/ascend_910b/<run-id>
```

Runner 只从已冻结且校验通过的 source snapshot 在 capture 目录内 configure/build，不接受外部 binary；它把指定 physical card 屏蔽为容器内 logical device 0，并记录 910B2 model、board ID 和 PCI identity。Runner 生成 `manifest.json` 与 `raw.csv`；report 生成 `result.json` 与 detached `candidate.json`。候选值的单位是 `ns_per_actor_task_completion`，只允许作为独立短搬运链的 completion 下界，与聚合带宽时间取 `max`；不得跨独立 AIV actor 相加，也不会自动进入 Forward。

## RoPE first-MTE completion probe

`memory/dram/latency/first_mte_completion_runner.py --campaign rope_key24_q_slice_edges` 隔离 key-24 的 4096 B Q-slice 请求完成边：direction 2 是 GM→UB `DataCopy` 后的 `MTE2_V`，direction 3 是 UB→GM `DataCopy` 后的 `MTE3_V`。冻结 sweep 只包含 `(block_dim,tile_bytes)` 为 `(1,4096)` 的 unloaded control 和 `(32,4096)`、`(43,4096)`、`(47,4096)` 三个代表并发 geometry，并对每项交换 request/control 顺序；不执行 production RoPE。key-22 的完整 8192 B Q slice 在本 campaign 中明确为 partial，后续应以独立的 `(1,8192)` control 与 `(16,8192)` representative probe 补齐；256 B direction 3 不是 production representative。

`--campaign rope_key22_first_q_slice_edges` 独立隔离 key-22 单 UB 程序的首个完整 Q slice：CANN 8.5.1 ATB 源码把 BF16 slice 固定为 4096 elements，即 8192 B；基础 sweep 包含 `(1,8192)` unloaded control 和 key-22 评估域最高并发 `(16,8192)`，`rope_key22_active4_replication_edges` 与 `rope_key22_active8_replication_edges` 分别补充精确的 4-worker 和 8-worker admission geometry，方向均为 2/3，并对每项交换 request/control 顺序。该 evidence 只允许用于每个 worker 的首个 Q slice；不得外推到后续 Q/K slice、未测量的 2-worker geometry，也不得借用 key-24 ping-pong 语义。

稳定性验证必须使用三个不同 run ID 分别启动 runner，并将 stdout 保存为三个独立 CSV；一个 run 内增加 schedule cycle 不能替代独立复测：

```bash
python microbench/ascend_910b/memory/dram/latency/first_mte_completion_runner.py --binary <absolute-first_mte_completion-binary> --device-id <physical-device-id> --run-id rope-key24-q-slice-01 --campaign rope_key24_q_slice_edges > profiler_output/microbench/ascend_910b/rope-key24-q-slice-01/raw.csv 2> profiler_output/microbench/ascend_910b/rope-key24-q-slice-01/runner.log
python microbench/ascend_910b/memory/dram/latency/first_mte_completion_runner.py --binary <absolute-first_mte_completion-binary> --device-id <physical-device-id> --run-id rope-key24-q-slice-02 --campaign rope_key24_q_slice_edges > profiler_output/microbench/ascend_910b/rope-key24-q-slice-02/raw.csv 2> profiler_output/microbench/ascend_910b/rope-key24-q-slice-02/runner.log
python microbench/ascend_910b/memory/dram/latency/first_mte_completion_runner.py --binary <absolute-first_mte_completion-binary> --device-id <physical-device-id> --run-id rope-key24-q-slice-03 --campaign rope_key24_q_slice_edges > profiler_output/microbench/ascend_910b/rope-key24-q-slice-03/raw.csv 2> profiler_output/microbench/ascend_910b/rope-key24-q-slice-03/runner.log
```

key-22 campaign 同样要求三个不同 run ID；每轮至少一个完整 balanced schedule cycle，并使用 `first_mte_completion_report.py --campaign <campaign> <run1.csv> <run2.csv> <run3.csv>` 执行独立复测门禁。通过门禁的 CANN 8.5.1 结果、raw/source/binary SHA256 和精确 lookup 边界记录在对应的 `references/aiv_rope_key22_first_q_slice_*_cann_8_5_1.json`。

## SwiGLU T1024 queue-edge probe

`memory/dram/latency/first_mte_completion_runner.py --campaign swiglu_t1024_queue_edges` 隔离 CANN 8.5.1 BF16 SwiGLU 的两个 depth-one TQue completion 边：direction 4 是 GM→UB `DataCopy` 后的 `MTE2_V`，direction 5 是 Vector→MTE3 handoff 后的 UB→GM `DataCopy`。静态 tiler 给出的 dominant T1024 base tile 是 `19 × 512 × 2 = 19456 B`，因此冻结 sweep 仅包含 `(block_dim,tile_bytes)` 为 `(1,19456)` 的 unloaded control、`(24,19456)`、`(48,19456)` 和 source-valid 单缓冲 payload control `(48,26624)`，并对每项交换 request/control 顺序。该 campaign 不执行完整 SwiGLU，也不读取任何 production duration。

可发布 evidence 必须来自三个不同 run ID，每个 run 至少一个完整 balanced schedule cycle；每个 run 的 schema、cell/order balance、pair-order replication 和正 completion gate 必须通过，所有对应 cell 的 cross-run median spread 必须不超过 20%。该 reference 是独立机制候选，不会被 production pipeline 自动消费。

当前通过全部 gate 的三-run CANN 8.5.1 结果及 raw/source/binary SHA256 见 `references/aiv_swiglu_t1024_queue_edge_completion_cann_8_5_1.json`；其中 lookup 仅允许精确匹配 worker count、direction 和 payload，production 尚未接线。

## ReshapeAndCache key-200 equal-chain probe

`operators/kv_cache/single_tbuf_lifecycle/single_tbuf_lifecycle_runner.py` isolates one complete `TPipe` scope with one `VECCALC` `TBuf` initialization. The frozen sweep covers worker counts `{1,2,4,8,16,48}` and exact 1024/2048-byte CANN key-200 token buffers plus the 110080-byte CANN BF16 elementwise key-210020000 two-tile allocation (`ubFormer=5504`, compile-time DAG footprint 20 bytes/element). Each interval repeats 64 independent scopes to resolve the 20 ns device counter; `single_tbuf_lifecycle_report.py` uses the long-minus-short interval and requires three independent runs plus payload and concurrency invariance. The published service is a workload-independent per-AIV lifecycle and excludes slot mapping, copies, token loops, cache layout, arithmetic and complete production duration.

`operators/vector/gather_v3_two_tque_lifecycle_kernel.cpp` replaces only the device kernel in the preceding frozen harness to isolate the exact CANN 8.5.1 GatherV3 key-10330 queue initialization: one depth-two 512-byte `TQueBind<VECIN,VECOUT,1>` plus one depth-two 8-byte `TQue<VECIN,1>` in a single `TPipe` scope. The unchanged payload field is only an invariance label because both allocations are compile-time constants. Three independent runs must cover worker counts `{1,2,4,8,16,48}` and all three labels, and the existing report must pass before the exact one-lifecycle service can be used. The probe excludes queue access, copies, scalar index decode, Gather arithmetic and complete operator duration; the published evidence and exact source/snapshot identity are in `references/aiv_gather_v3_two_tque_lifecycle_cann_8_5_1.json`.

`operators/kv_cache/reshape_cache_equal_chain_runner.py` 隔离 CANN 8.5.1 `ReshapeAndCache` key 200 的 equal-head copy chain：每个 AIV 通过同一个 `TBuf` 顺序执行一次 K 和一次 V 的 GM→UB→GM 搬运，每条链保留 `MTE2_MTE3` 与 `MTE3_MTE2` hard completion edge。probe 不包含 slot mapping、token loop、host tiler、cache layout 或完整 production 算子时延。冻结 sweep 仅包含 block1 unloaded control 与最高误差贡献的 block48 source representative，payload 仅为 1024 B 和 2048 B，并交换 request/control 顺序。

可发布 evidence 必须来自三个不同 run ID；每个 run 的 schema、64-trial balanced cell order、pair-order replication 和正 completion gate 必须通过，对应 cell 的 cross-run median spread 必须不超过 20%。运行时先确认物理卡空闲并显式传入 device ID，stdout 和 stderr 只能保存到 `profiler_output/`：

```bash
python microbench/ascend_910b/operators/kv_cache/reshape_cache_equal_chain_runner.py --binary <absolute-reshape-cache-equal-chain-binary> --device-id <physical-device-id> --run-id <unique-run-id> > profiler_output/microbench/ascend_910b/<run-id>/raw.csv 2> profiler_output/microbench/ascend_910b/<run-id>/runner.log
python microbench/ascend_910b/operators/kv_cache/reshape_cache_equal_chain_report.py <run-1.csv> <run-2.csv> <run-3.csv>
```

当前通过全部 gate 的三-run 结果及 raw/source/binary SHA256 见 `references/reshape_cache_equal_chain_completion_cann_8_5_1.json`；production 只在 key 200 且 worker count 为 1 或 48、单个 K/V payload 为 1024 B 或 2048 B 时消费对应 exact service，其他 domain 保持 partial 且不插值。

`equal_midworkers_warm` campaign 在同一机制 buffer 上先完成固定 16 次 launch，再测量第 17 次 launch 内的两-copy hard-event chain device interval，用于隔离 cache-steady runtime service；预热 launch 不进入计时区间，probe 仍不包含 TPipe/TBuf 初始化、slot mapping、token loop、cache layout 或完整 production 时延。冻结 domain 仅包含 worker count `{2,4,8,16}`、payload `{1024,2048}` 和每 worker 一个 token；Forward 只消费 exact worker/payload cell，不插值、不外推，也不用于 repeated-token worker。

```bash
python microbench/ascend_910b/operators/kv_cache/reshape_cache_equal_chain_runner.py --binary <absolute-reshape-cache-equal-chain-binary> --device-id <physical-device-id> --run-id <unique-run-id> --campaign equal_midworkers_warm --warmup-launches 16 > profiler_output/microbench/ascend_910b/<run-id>/raw.csv 2> profiler_output/microbench/ascend_910b/<run-id>/runner.log
python microbench/ascend_910b/operators/kv_cache/reshape_cache_equal_chain_report.py --campaign equal_midworkers_warm --warmup-launches 16 <run-1.csv> <run-2.csv> <run-3.csv>
```

三次独立正式 run 的 exact service、稳定性 gate、source/binary identity 与 raw SHA256 见 `references/reshape_cache_equal_chain_warm_completion_cann_8_5_1.json`。

## Elementwise source-control lifecycle probe

`operators/vector/elementwise_control_lifecycle/` 在固定 48 AIV 上隔离 CANN 8.5.1 BF16 Add/Mul 公共 scheduler 的控制开销，不执行 Add/Mul tensor payload。冻结 variant 为公共 tile loop 与 terminal `PIPE_ALL` fence 的 `control_baseline`、增加单个 384 B `VECCALC` TBuf、七节点 traversal 和每 tile 12 次 buffer-ID lookup 的 `dag_tbuf`，以及再增加每 tile `PIPE_MTE2/V/MTE3 = 2/9/1` 次 mutex lock/unlock 的 `full_lifecycle`；tile-count class 固定为 `{18,29,30,36,58,59}`，其中 `{18,36,59}` 只用于 affine calibration，`{29,30,58}` 是 capture 前冻结的 holdout。

`operators/vector/elementwise_single_tile_program/` 保留 CANN BF16 Add/Mul 公共 47-AIV 单 tile memory/control program：46 个 worker 各处理 1408 elements，尾 worker 处理 768 elements，执行一对 GM→UB copy、三次 cast、以 `Duplicate` 替代 production Add/Mul、一次 UB→GM copy 与 terminal `PIPE_ALL` fence。Forward 仅将三轮最大 worker body 中位数作为该精确 source domain 的竞争下界，production arithmetic 仍由独立 Vector demand 守恒；不使用完整 production 或 evaluation duration。

采集必须使用三个独立 run ID，每 run 六轮 forward/reverse position-balanced 顺序。report 要求每 cell cross-run median spread 和 position-balance error 均不超过 20%，holdout MAPE 不超过 15%、max APE 不超过 25%，且所有 paired delta 与 `startup_ns + tile_count * per_tile_ns` 参数均为正；任一 gate 失败时参数保持 `null` 且 `mechanism_publishable=false`。通过 gate 的候选单位为 `ns_per_48_aiv_source_control_program_completion`，后续只能按 `max(existing_48_aiv_tile_pipeline_span, source_control_candidate)` 组合，禁止相加或建立 per-shape LUT；本目录不会生成 canonical reference，也不会自动接入 Forward。
