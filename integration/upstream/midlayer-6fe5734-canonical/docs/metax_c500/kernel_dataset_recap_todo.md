# MetaX C500 数据集重采交接

## 当前分支

- 工作分支为 `research/c500-recapture`，不得直接操作 `main` 或 push。
- Serving instrumentation 与预热修复位于本分支已有提交；C500 15-case Serving 数据更新单独提交为 `30d02090`。

## Kernel Dataset 状态

- `kernel_dataset/metax_c500/{vector,flash_attn,gemm,fused_moe}.json.zst` 共 987 个 case，类别数分别为 296、294、221、176。
- 原有 494 个 Event timing 保持不变；提交 `9806a710` 已更新 346 个 r2/r5/r6 结果；本轮 candidate 再更新 147 个结果，分布为 FlashAttention 79、FusedMoE 2、Vector 66、GEMM 0。
- 147 个结果中 139 个通过冻结的统一稳定性阈值。其余 8 个 Vector case 的 CV 和跨 repetition 差异均通过，仅最大单样本相对中位数偏差为 5.051% 至 7.710%；用户已明确接受这 8 个结果。
- 8 个例外为 `VECTOR_ELEMENTWISE_T1024_H16384_B1_ADD`、`VECTOR_ELEMENTWISE_T1024_H16384_B1_MUL`、`VECTOR_ELEMENTWISE_T1024_H5120_B1_ADD`、`VECTOR_ELEMENTWISE_T1024_H5120_B1_MUL`、`VECTOR_ELEMENTWISE_T1024_H8192_B1_ADD`、`VECTOR_ELEMENTWISE_T1024_H8192_B1_MUL`、`VECTOR_ELEMENTWISE_T256_H16384_B1_ADD`、`VECTOR_ELEMENTWISE_T256_H16384_B1_MUL`。
- 987/987 timing 均为完整 OP 的无 profiler Device Event，`granularity=op`、`source=device_event_op_median`、`profiler=none`；没有使用 mcTracer duration 或 kernel duration 求和。
- 指定的 84 个 `fused_moe_missing` case 已由 4 个 20-case batch 和 1 个 4-case canary 完成 mcTracer Graph trace；raw 重新解析通过，84/84 launch sequence 与 canonical kernel observation 一致，因此无需改写既有 kernel/counter payload，mcTracer duration 未进入 golden。
- 最终 candidate 与逐 case 选择、哈希、稳定性和 trace 绑定记录位于 `profiler_output/kernel_dataset/metax_c500/candidate/final_987_20260902/source_manifest.json`；147 个新 Event artifact tree SHA-256 为 `fe426417c0985f4ba723d3ddc65e0252fbc1a3f260a9c90f1e7a98e4a13fa845`。

## Serving Dataset 状态

- `serving_dataset/sharegpt/metax_c500` 的 15/15 case 已更新为本轮 BF16、prefix cache enabled、offload disabled、async scheduling enabled、pipeline depth 2 的结果。
- 每例 128 个请求，schema-v3 loader 全部通过；运行时 Graph mode、padding、fallback 和 in-flight step 证据齐全，15/15 的最大 in-flight step count 均为 2，JIT warning 总数为 0。
- `requests.csv` 与此前 canonical 逐字一致；`alignment_steps.json.zst` 的解压内容未重采算子，仅将 `raw_source_set_sha256` 重绑到本轮 ground-truth 与 scheduler raw；`manifest.json`、`metrics.json` 和 `scheduler_steps.jsonl.zst` 的解压内容来自本轮同一 case 运行。
- 本轮 Serving candidate 报告位于 `profiler_output/serving/metax_c500/c500_async_v4_818e4533_20260901_candidate_workspace/suite_report.json`。

## Raw 产物位置

- Serving 有效 raw 与 candidate 已完整同步到本 worktree 的 `profiler_output/serving/metax_c500/c500_async_v4_818e4533_20260901_*`，另包含 observer-effect canary `profiler_output/serving/metax_c500/c500_async_v2_38e24a9e_20260901_qwen3-8b_tp1_qps4_trace_on_canary`；校验清单共 167 个文件且哈希一致。
- Kernel Event 完整远端 raw 位于 `muxi:/mnt/disk2/yxh/execution_snapshots/c500_gemm_bf16_41d29da9_r2/experiment_logs/kernel_golden/c500_full_event_formal_41d29da9_20260901_r{8,10..19}`，总计约 8.5 GiB。体积主要来自 `process_isolation/` 高频快照，未整包回传；本地已保留所有 JSON、CSV、log、stdout、stderr 与 `process_isolation_manifest.json`，路径为 `experiment_logs/kernel_golden/c500_full_event_formal_41d29da9_20260901_r*`。
- 84-case mcTracer raw 已完整同步到本地 `profiler_output/kernel_dataset/metax_c500/c500_fused_moe_trace*_20260901_r1*`，约 293 MiB；严格解析后的 batch candidate 位于 `profiler_output/kernel_dataset/metax_c500/candidate/graph_batches/`。

## Main 集成前检查

1. 独立审阅 Kernel 数据提交，确认接受上述 8 个最大单样本偏差例外；不要把它们表述为通过冻结稳定阈值。
2. 独立审阅并合入 Serving 数据提交 `30d02090`；确认 Git LFS 对象可用。
3. 合入后分别加载 C500 987 个 Kernel case 与 15 个 Serving case，并确认文件哈希与本交接记录一致。
4. 本 worker 未操作 `main`、未 push；远端 8.5 GiB 完整 Kernel raw 在主 agent 完成验收前不得删除。
