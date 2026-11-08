# MetaX C500 kernel profiling

本文是 MetaX C500 kernel dataset 采集与排障的长期参考。活动实现位于 `dataset_pipeline/kernel_dataset/metax_c500.py`，profiler 直接执行的硬件 runner 位于 `scripts/kernel_dataset/metax_c500/`。本文只记录已经由 raw artifact、当前代码或重复采集验证的行为；容器名、个人目录、一次性诊断命令和未复现的环境搭建过程不属于仓库 contract。

芯片结构与术语参考见 `docs/metax_c500/architecture.md`；mcTracer/mcProfiler 的字段语义、采集、timing 与 provenance 规则均以本文为准。

## 适用范围

本文中的工具行为、串行限制、attach 时间、文件格式、fallback 条件和排障结论只适用于当前 MetaX C500 的 mcTracer + mcProfiler 采集链。它们不是通用 profiler 规范，也不得外推到 NVIDIA A800、Ascend 910B 或其他硬件。跨硬件共享的只有 workload schema；各硬件的 profiler provenance、runner lowering 和采集约束仍由各自 pipeline 定义。

## 数据边界

MetaX C500 采集链同时使用两个工具，它们提供不同且不可互相替代的证据：

| 证据 | 来源 |
|---|---|
| kernel name、grid、block、register/thread、static/dynamic shared memory | mcTracer |
| uninstrumented kernel duration | mcTracer |
| instrumented `during`、pipe duty、memory bytes、cache hit rate、waves、roofline counters | mcProfiler |

当前 C500 mcProfiler 的 `--custom --per-kernel` JSON 不包含 grid、block、register/thread 或 shared-memory allocation。`WORKGROUPS` 和 `WAVES` 是计数器，不能还原三维 launch geometry。Canonical JSON 中的 `Launch Statistics` 是 aggregate 将 mcTracer launch metadata 与 mcProfiler counters 按 kernel identity 连接后的兼容表示，不代表这些字段由 mcProfiler 原生提供。

Layer-1 counter 采集会 replay 或暂停目标 kernel，因此 mcProfiler 的 `during` 是插桩执行时间，已观察到比正常 launch 放大几十至上千倍。Canonical `Duration [nsecond]` 和由它换算的 `Elapsed Cycles` 必须始终来自 mcTracer；唯一评分合同 `op_info.evaluation.timing` 的逐 kernel duration 也必须复制同一批 mcTracer wall time， 其 `total_duration_ns` 必须等于逐 kernel duration 之和。`MetaX.during` 只作为诊断 provenance 保留，不能作为 evaluation target。若 mcProfiler 的 AP active cycles 与 mcTracer wall time 不相容，Oracle 必须将依赖这些 cycles 的 eta 和 per-pipe time 标记为 unavailable，不能借 replay 后的 counter 伪造真实执行时间。

mcTracer 与 mcProfiler 会用同一个 runner 和 workload 参数分别执行。它们不是同一次 kernel invocation，因此 aggregate 必须：

1. 用 CPU `mcLaunchKernel`/`mcModuleLaunchKernel` 的 `co_id` 关联 GPU event，并以最后两次 `mcDeviceSynchronize` 之间发起的 launch 作为唯一 marker call；
2. 对同一 shape 的所有 mcTracer shard 要求完整、有序的 kernel-name/launch-signature 序列一致，只逐 ordinal 合并 duration；
3. 按编号文件的全局 launch ordinal 将 `_allk` invocation 与 trace 一对一连接，同时要求规范化 kernel identity 在该位置相符；
4. 校验 counter 侧 work count：普通 direct launch、MLA persistent worker 和现有 tile16 queue 的 `WORKGROUPS` 必须等于 mcTracer 物理 grid 总数；mcBLAS GEMM 的工具口径随 launch 形式变化：direct/padded grid 记录物理 worker 数，persistent/swizzled grid 记录 `ceil(B*M/tile_m) * ceil(N/tile_n) * split_k` 个非空逻辑 tile，aggregate 只接受这两个由 trace 和 workload 独立推导的值；已验证的 C500 tile64×64、block256 chunked-prefill packed queue 必须等于 `B * ceil(Q_LEN / 64) * HQ * SP_REQUESTED`，因为该 counter 记录 padded logical jobs， 而 mcTracer grid 记录较小的 worker pool；
5. 对缺失、歧义、数量不等、顺序不等或未消费的 invocation 立即失败，不用 duration 相似度、最大 DRAM bytes、按名字聚合或无编号目录顺序猜测匹配关系。

当前 layer-1 mcProfiler canonical pipeline 每次 profiler invocation 只承载一个 shape。旧的多-shape batch collector 依赖 range 数量和文件顺序拆分 `_allk`，无法满足上述唯一 provenance contract，不得作为 canonical dataset 来源。Graph-only mcTracer candidate 可以在一次 mcTracer invocation 中串行执行多个 shape，但必须由 `kernel_golden_graph_runner.py --trace-batch-manifest` 为每个 exec 后的真实 runner child 记录唯一 PID、case ordinal 和预期 Graph replay 顺序，并由 `_parse_graph_trace_batch` 同时按 child PID、CPU `mcGraphLaunch.co_id`、同步边界和完整 per-call launch sequence 拆分；该例外不适用于 mcProfiler `_allk`，也不允许按文件名、duration 或仅按 kernel name 猜测归属。

## 环境预检

采集必须在安装了 MetaX torch、目标算子库、mcTracer 和 mcProfiler 的容器内进行。具体容器名不是接口；每次采集前应记录实际软件版本，并至少检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
assert "metax" in torch.__version__.lower()
PY

command -v mcTracer
command -v mcProfiler
test -d "${MACA_PATH:?MACA_PATH is required}"
test -d "${MACA_ETC_PATH:?MACA_ETC_PATH is required}"
```

安装其他 Python 包可能连带替换 MetaX torch。典型结果是 flash-attn ABI import 失败， 或 mcProfiler 只产生 `event_batch` 而没有 report。遇到这种情况应修复环境，不能把空 report 当成采集成功。

已验证的采集环境使用 MACA 3.7.2；曾使用的 3.2.1 runtime 不包含同一套 profiler config redirect 行为，会得到 `Profile is not available`。具体容器名不是 contract， 但换环境后必须重新验证 runtime、算子 ABI 和 counter service，不能只确认 GPU 可见。

当前 MACA 3.7.2 的 `mx-smi --show-hbm-bandwidth` 在 GPU engine utilization 为 0 且无设备进程时仍可能显示 1 MBytes/s；采集 preflight 将 0–1 MBytes/s 视为该工具的空闲量化底噪，同时继续严格要求 0% engine utilization、无设备进程，并拒绝负数或高于该范围的 HBM 读数。

当前 MACA 3.7.2 的 `mcTracer --odname` 参数实际按“相对于 mcTracer 进程工作目录的文件夹名”处理，即使传入以 `/` 开头的字符串也会继续拼接当前目录。若同时传绝对 `--odname`，日志会出现 `Failed to create <cwd>/<absolute-path>`；工具仍可能返回 0，但不生成 `tracer_out*.json`。活动 pipeline 因此从 kernel trace 根目录启动 mcTracer，并只把 case basename 传给 `--odname`。采集成功必须以目标 case 目录中出现 trace JSON 为准； “进程返回 0”不构成成功。无 trace 时必须保留并报告 mcTracer stdout/stderr，不能继续 mcProfiler 或把旧 trace 当作本次结果。

`MACA_ETC_PATH` 必须存在于 mcProfiler server 进程环境中，并指向包含真实 counter 定义的目录。它不得被显式写进被 profile 的 `--cmdline`；C500 的已验证失败模式是目标子进程因此跳过 profiler target 配置检查，最终报 `Profile is not available`。使用空目录也可能让服务注册成功但产生错误 counter magnitude。`MACA_PATH` 不能带尾部分隔符。活动 pipeline 在 server env 中显式传递这两个变量，只把 `MACA_PATH` 和 runtime library path 写进 child cmdline。已验证日志中的关键成功证据是 config redirect 后仍出现 `mctxTPImpl.cpp:300 ... profiler target ENV file ... exist`；只有 redirect、随后却出现 `:432 Profile is not available`，通常表示 child cmdline 或路径仍被污染。此时应先检查这两个环境变量，不能把空 report 当作硬件计数为零。

## 采集入口与串行约束

Canonical raw workspace 是：

```text
profiler_output/kernel_dataset/metax_c500/
```

从仓库根目录串行运行：

```bash
python scripts/kernel_dataset/metax_c500/build_dataset.py \
  --config configs/dataset/metax_c500.yaml \
  --kernel gemm,flash_attn,fused_moe,vector \
  --stage profile

python scripts/kernel_dataset/metax_c500/build_dataset.py \
  --config configs/dataset/metax_c500.yaml \
  --kernel gemm,flash_attn,fused_moe,vector \
  --stage aggregate --candidate-only
```

审计阶段必须使用 `--candidate-only`，输出位于 `profiler_output/kernel_dataset/metax_c500/candidate/`。确认完整性、provenance、 SHA-256 与评估结果后，才可在获得 canonical dataset 重写授权的任务中去掉该参数发布。

当前 C500 mcProfiler layer-1 per-kernel 模式是主机级单例。不同 GPU、不同端口仍会竞争同一个设备侧 RPC/profile service，因此不能并行运行多个 `perf_exec --per-kernel`。`--start-idx` 和 `--end-idx` 只用于串行分段、恢复或交接，不能作为并行 worker 切分依据。mcTracer-only 任务可独立规划，但不能与活动的 layer-1 mcProfiler job 混写同一个 shape 目录。

异常退出后不得使用宽泛的进程清理命令。先确认目标 PID/进程组属于当前采集任务，再由 pipeline 的 process-group cleanup 或精确 PID 终止。

mcProfiler 异常退出还可能在临时目录留下较大的 PyInstaller `_MEI*` extraction。长时间采集应监控临时盘容量；清理前必须逐个确认目录的 owner、创建时间和关联进程，不能执行无归属校验的通配符删除。

## Marker window 与 attach 等待

Layer-1 runner 用 `torch.cuda.profiler.start()/stop()` 只包围一次真实 profiled call。 `--custom --per-kernel` 因而排除 tensor 初始化、warmup 和 autotune setup，只保留 serving 算子调用窗口内的 kernel。

C500 layer-1 pass 一次采集 marker window 内的全部 kernel，不附加 `--kernelnames`；按名字重复启动 runner 会反复支付初始化成本，并增加不同执行之间 dispatch 漂移的机会。同名 kernel 可以在一个真实调用中出现多次，必须作为不同 ordinal 保留，不能平均成一条记录。 `--single-pass` 只用于 whole-shape 辅助报告，不能替代产生 numbered invocation 文件的 `--custom --per-kernel`。

当前 C500 mcProfiler 通过 RPC attach 到 runner。runner 执行过快时，设备端可能在 client 连接前退出，日志会出现 `cannot connect to server`，且没有编号的 isolated 文件。活动代码的 attach wait 基线为 GEMM 30 秒、FlashAttention 15 秒、FusedMoE 5 秒、Vector 5 秒，并对 layer-1 失败做有限重试。GEMM 曾使用 2 秒基线，但真实采集出现 runner 已执行、RPC port 仍未被 mcProfiler client 发现且最终没有 report 的失败；30 秒是对该已复现 attach 机制的等待，不是 kernel duration 参数。若该基线仍失败，应先保留日志并检查 RPC/service 状态， 不能退化为零 counters 或跨 kernel report。

### Graph-only mcTracer batch

面向 graph-covered serving 的 mcTracer candidate 可以把多个 selected case 放进同一个 mcTracer 进程，以摊销工具启动成本；case 在父 runner 内必须串行，不能通过多卡同时启动多个 mcTracer。典型入口如下，其中 `--odname` 仍是相对于当前 trace root 的目录名，Event case 和表仍写 `experiment_logs/kernel_golden/`，PID/顺序 manifest 与 trace 只写 `profiler_output/kernel_dataset/metax_c500/`：

```bash
cd profiler_output/kernel_dataset/metax_c500
mcTracer --odname graph_batch_000 \
  python /absolute/repo/scripts/kernel_dataset/metax_c500/kernel_golden_graph_runner.py \
  --config /absolute/repo/configs/dataset/metax_c500.yaml \
  --output-dir /absolute/repo/experiment_logs/kernel_golden/graph_batch_000 \
  --baseline-commit COMMIT --source-snapshot-sha256 SNAPSHOT_SHA256 \
  --timer-sanity-sha256 TIMER_SHA256 --family vector \
  --case-id CASE_A --case-id CASE_B --case-id CASE_C \
  --warmup 5 --samples 10 --graph-repetitions 100 500 \
  --trace-batch-manifest /absolute/repo/profiler_output/kernel_dataset/metax_c500/graph_batch_000/manifest.json
```

mcTracer 返回并生成全部 `tracer_out*.json` 后，用 dataset owner 的显式 stage 严格拆分并写 candidate；该 stage 不发布或覆盖 canonical JSON：

```bash
python scripts/kernel_dataset/metax_c500/build_dataset.py \
  --config configs/dataset/metax_c500.yaml --kernel vector \
  --stage graph-trace-batch \
  --graph-trace-dir profiler_output/kernel_dataset/metax_c500/graph_batch_000 \
  --graph-trace-manifest profiler_output/kernel_dataset/metax_c500/graph_batch_000/manifest.json \
  --graph-measured-repetition-policy per-case-high \
  --graph-candidate-output profiler_output/kernel_dataset/metax_c500/candidate/graph_batches/graph_batch_000.json
```

正式 mixed-duration batch 使用 `--graph-measured-repetition-policy per-case-high`：parser 对每个 case 独立选择其 `measured_sample` 中最大的 replay count。该 count 只从已经与 Event artifact 逐字段核对的 manifest `graph_launch_plan` 推导，不读取 Event 或 mcTracer duration；batch candidate 的 `measurement_selection` 记录 policy 和 derivation，每个 case 记录实际选中的 replay count。原有 `--graph-measured-repetition-count N` 保留为显式固定 count 模式，并与 policy 互斥。Family merge 要求所有 batch candidate 使用完全相同的 selection policy，并继续从各自 raw evidence 重新校验。

正式运行前先在相同参数后加 `--dry-run`；dry-run 只验证并打印 case 顺序、repetition 和 buffer contract，不创建 manifest、Event artifact 或 profiler raw。正式 batch 禁止 `--resume`，因为跳过的 case 会引用上一次 mcTracer invocation 的 PID。manifest 在启动前以 `complete=false` 原子写入，每完成一个 child 就追加其 exec 后 PID、Event artifact SHA-256 和从已校验 measurement 推导的 Graph launch plan，全部完成后才改为 `complete=true`；异常退出留下的不完整 manifest 必须拒绝解析。

解析时必须同时满足：manifest case/PID 唯一且顺序连续；manifest 的 baseline/source snapshot 以及逐 case canonical-attributes/runner/Event-artifact SHA-256 与 Event artifact 内容逐字段一致，artifact measurement 推导的 Graph launch plan 也与 manifest 完全一致；所有相关 runtime launch/synchronize event 都显式携带 `args.process_id`；所有 trace runtime PID 恰好属于 manifest，声明 PID 无缺失且不能跨多个 trace shard；每个 CPU `mcGraphLaunch.co_id` 在 child PID 与 trace 作用域内唯一关联非空 GPU kernel 序列，允许不同 child 的独立 trace 重用数值相同的 `co_id`；每次 Graph launch 后、下一次 launch 前存在同 PID 的 `mcEventSynchronize` 或 `mcDeviceSynchronize` 边界；实际 Graph launch 数量、repetition count、case 执行顺序和完整 per-call kernel-name/launch-signature 序列均与 manifest 一致。GPU kernel 顺序按设备时间戳和稳定 event ordinal 确定，不能依赖 JSON 数组排列。所有 trace、manifest、Event artifact 和 candidate 路径均按 realpath 校验 owner 边界，candidate 不得覆盖 manifest 或任一 trace。任一额外 PID 都视为 cross-attach 污染并整批失败。已观察到的失败模式是三个并行 mcTracer 各自产生另外两个 target PID 的 trace；即使每个目录也包含自己的 PID，该批次仍不可使用。

## mcProfiler 输出语义

`mcprofiler_out/_allk/` 中常见文件：

- `N_<kernel>.txt.json`：该 kernel 的 per-invocation isolated counters，是正常消费者输入；
- `report.txt.json`：整个 marker window 的 aggregate，不是普通的 per-kernel observation， 只作 raw 归档，不写入 canonical kernel entry；
- `N_period*.txt.json`：region rollup，跨 kernel 汇总，必须排除；
- `event_batch_*.json`：底层 counter batch 配置/原始产物，不提供 launch geometry。

日志中的 `PerfDrop ... fallback to alloc memory` 只表示工具改走较慢的分配路径，本身不能证明 counter 被丢弃。是否成功必须以 numbered invocation、必需 metric 和 provenance gate 为准，不能仅凭这条日志修改 profiler flag。

mcProfiler 可能把非必需的 `Constant SL1 Hit Rate` 输出为精确错误串 `can not find export value @sl1_hit_rate`，并把依赖它的 `Memory Data Flow Chart` 输出为 `cannot draw from data: can not find export value @sl1_hit_rate`。Aggregate 只把这两组已知的 metric/sentinel 规范化为 `MetaX.all_scalars` 中的 `null`；所有必需 metric、其他字段或其他错误串仍然立即失败。原始 numbered report 必须保留，不能把 unavailable 伪造成 0。

`l2c_memacs` 与 `all_memacs` 不是同粒度的逐 byte 上下游计数。当前 canonical evidence 中， 前者描述 L2C 侧请求流量，后者逐条等于 `Global Memory Read bytes + Global Memory Write bytes`，并与 `Dnoc Read Req`/`Dnoc Write Req` 呈约 128 byte/request 的下游 cache-line 关系。小粒度或非对齐请求可能让 global-memory line traffic 大于 L2C request bytes；因此不能用 `all_memacs > l2c_memacs` 判定报告损坏，也不能用 `L2C Hit Rate` 强行校验两者的 byte ratio。后者是 request/event 口径，不是 byte-weighted miss ratio。

Oracle 的 C500 integrity gate 分别校验 read/write 的 global bytes 与 DNoC request count； 当前保守上限是每个 request 两个硬件 cache line。若任一方向存在非零 bytes 但零 request， 或超过该上限，则 L2/DRAM formula pair 必须一起回退为 analytical；独立通过 L2C instruction-volume gate 的 raw L2C bytes 仍可作为 xbar evidence。有效 pair 的 L2 service envelope 使用 `max(l2c_memacs, all_memacs)`，以包含下游完整 cache-line fetch 的放大。

即使 shape 经 trace 证明恰好包含一个 kernel，`report.txt.json` 仍是 device-wide rollup， 不能作为 numbered per-invocation counter 的 fallback。缺少编号文件必须重采 layer-1； 不得把 whole-shape report 伪装成 canonical per-kernel 数据。

部分 sub-microsecond auxiliary kernel 会触发 mcProfiler roofline 的除零错误。该情况必须在输出中记录 `dur_source` 和 `tool_limitation`；不能吞异常、填零或把其他 kernel 的 duration/counters 混入。无论 roofline 是否成功，`dur_source` 都是 mcTracer；成功时的 mcProfiler `during` 仍只保留作插桩诊断。

## Kernel identity 与文件名截断

当前 C500 mcProfiler 会截断较长的输出文件名，截断后的 Itanium symbol 可能无法由 `c++filt` 完整 demangle。活动 aggregate 依次尝试：

1. canonical demangled name 精确匹配；
2. truncated mangled source-name 子序列匹配；
3. truncated demangled prefix 匹配。

后两级都带唯一性守卫。存在多个 identity 候选时必须失败；identity 匹配后还必须与编号文件的全局 launch ordinal 一致，不能按 substring 第一个命中、counter 数值或同名文件平均选择。`period` rollup 不参与 identity matching。

Case 目录、mcProfiler `--casename` 和 aggregate lookup 必须统一使用 workload expander 生成的 `shape["filename"]`。C500 旧实现曾另造包含 `_SP` 的 `trace_name`，导致 profile 写入一个目录、aggregate 却读取另一个目录；活动 pipeline 不再保留第二套 case identity。

旧 Vector expander 还曾用 `op2` 生成 `_ADD`/`_MUL` 文件名，却没有把 `op2` 写进 shape Attr；runner writer 随后用默认值 `add`，导致所有历史 `_MUL` artifact 实际执行 `torch.add`。活动 workload contract、script args 和 runner 都要求显式 `op2`，审计时还必须核对生成 runner 与 counter kernel identity，不能只相信 case 文件名。

mcTracer event 也不得按 kernel-name 子串黑名单过滤。C++ demangled name 的模板参数可能包含另一个函数名；曾经用 `"silu_kernel" in name` 过滤 setup kernel，结果同时误删了模板参数引用该符号的真实 SiLU 主 kernel。当前 C500 pipeline 保留 marker call 内所有具有有效 launch metadata 的 event；若 `_allk` 数量或 ordinal identity 无法一一对应则要求重采，不把任何 trace event 静默标成 out-of-scope。

## Trace 生命周期与 provenance

同一个 shape 目录中的所有 `tracer_out*.json` 会被视为同一 observation 的 shard。不同 runner 版本、不同软件环境或不同采集批次的 trace 不得混放。重采应使用新的 checkpoint 目录，或先把旧 artifact 移入带日期、软件版本和替换原因的归档目录；不要只覆盖其中一个文件。

同一 shape 的多个非空 trace 必须具有完全相同的 kernel-name/launch-signature 序列；它们只能提供 duration 的重复观测，不能按 kernel name 取并集。已确认的失败模式是两个旧 execution 分别 dispatch 不同 FlashAttention variant，按名字并集后会伪造出一个从未真实执行过的双-main observation，并把同一份 layer-1 counters 重复绑定给两个 variant。

存在 trace 不等于有效。空 trace、截断 JSON、没有真实 kernel event 的 capture 或同名 kernel launch signature 冲突都必须在 aggregate 前失败。一个已确认案例中，同名 mcBLAS kernel 的旧 shards 同时出现 `(64,112,1)` 与 `(16,32,1)`；旧的“取第一个 shard”策略会产生错误 provenance，新采集和当前 union gate 要求唯一 signature。

不能用“寻找周期并折叠 replay”推断真实调用边界。MoE 的一个真实调用本来就可能包含多个同名 fill 或 projection kernel，周期折叠会改写顺序、漏掉 invocation 并把不同 ordinal 的 duration/counter 平均。活动 parser 只接受由 CPU launch `co_id` 和最后两次 synchronize 界定的最终 marker call；多个 shard 必须逐 ordinal 一致后，才对同一 ordinal 的 duration 取中位数。

## Workload contract 陷阱

FlashAttention 的实际输入由 canonical `Q_LENS`/`KV_LENS` 决定，不从 `kv_dist` 标签重新生成。Paged-varlen prefill 的 `Q_LENS` 可等于逐请求 `KV_LENS`，所以 `sum(Q_LENS)` 通常不等于 `B * Q_LEN`；runner 必须按逐请求长度构造 query tensor 和 `cu_seqlens`。Chunked-prefill 则将 requested Q 解释为 batch token budget，并要求每个请求 `Q_LENS <= KV_LENS`。

Vector `element_count` 表示 canonical logical elements，读写或 fused-operation multiplier 由 predictor 根据 variant 计算：

- SiLU：`tokens * intermediate_size`，不是输入/输出合计；
- KV-cache：`tokens * num_kv_heads * head_dim`，不是 K/V 两份流量之和；
- RoPE：同时记录 query heads 和 KV heads，predictor 再计算实际旋转元素数。

把 traffic multiplier 预乘进 `element_count` 会让不同硬件或 variant 重复计数。

## 源码版本归属与 canonical 数据的对应关系

vLLM 源码参考树（`/share_data/yangxinhao/chip_modeling/upstream/vllm/`，仓外只读副本）下有两棵 C500 源码树，它们都不能默认当作 canonical serving 数据的版本，**必须用 launch signature 逐个核对**。

`v0.10.2_c500_legacy/` 是 2026-07 从 `yxh_vllm_native` 拉的 vLLM 0.10.2 MetaX fork，只含 Python 树、没有 `csrc/`，不能用来推导任何 device program。`v0.24.0_c500_sdist/` 是 vLLM 0.24.0 sdist，配合同级目录的容器 plugin 副本 `vllm_metax_0.24.0/`（0.24.0+g436274.d20260813），含完整 `csrc/libtorch_stable/`。

**但 0.24.0 树并不等于 canonical serving 数据的版本。** 已用 launch signature 证实两处不一致：canonical 的 rope 是 `rotary_embedding_kernel<c10::BFloat16, true>`，两个模板参数、十一个运行时参数，有 head stride 而无 `rope_dim_offset`/`inverse`，即 ABI11；0.24.0 的同名 kernel 是三个模板参数、十三个运行时参数的 ABI13。canonical 的 MoE routing 是 `topkGatingSoftmax<4,128,4,16,32,int>`，而 0.24.0 树里该 kernel 名为 `topkGating`、模板参数有八个，`topkGatingSoftmax` 这个符号在 0.24.0 树中根本不存在。函数参数个数不随 demangle 丢失，因此这两条都是判定性证据而非推测。

由此得到本仓库的硬性做法：为 canonical serving 数据建模时，**program identity 以 canonical 自身的 launch signature 为准**，源码树只用于解释该 signature 已经确定的程序的内部结构；跨版本借用签名不同的 kernel 属于未经验证的断言。容器 `yxh_vllm_serving_c500` 当前安装的 plugin 同样与 canonical 不一致（它跑 `topkGatingSoftmaxDecode`，该 kernel 在 0.24.0 开源 `csrc` 中不存在），所以在该容器内重采也复现不出 canonical 的 routing 路径。canonical serving 数据究竟采自哪个 plugin build，是一个待澄清的 provenance 问题。

## 发布与归档

Raw mcTracer/mcProfiler artifact、日志、PNG/PDF report 和 isolation report 留在 `profiler_output/` 或机器侧归档，不提交 Git。Canonical payload 只有：

```text
kernel_dataset/metax_c500/gemm.json.zst
kernel_dataset/metax_c500/flash_attn.json.zst
kernel_dataset/metax_c500/fused_moe.json.zst
kernel_dataset/metax_c500/vector.json.zst
```

这四个 JSON 由 Git LFS 管理。发布前后记录 tar 与各 JSON 的 SHA-256，确认 entry count、空 kernel、非正 duration、invalid grid、incomplete shape 和 unisolated kernel 均为预期值， 再运行仓库门禁。Dataset 的实测 raw launch 可以超出当前 predictor 支持范围；这种情况应保留数据并让 evaluation fail loud，不能为了通过预测而改写 measured metadata。
