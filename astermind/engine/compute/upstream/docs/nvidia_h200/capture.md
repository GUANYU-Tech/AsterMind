# NVIDIA H200 ShareGPT serving capture

本文是 H200 ShareGPT serving 真值采集的操作参考：它说明在 H200 上如何复现 A800/C500/910B `serving_dataset/sharegpt/<hw>/<model>/<case>/` 背后的全部 raw artifact，以及 vLLM 0.11.0 与 vLLM 0.30.0 两个 runtime 在采集链上的差异。活动实现位于 `scripts/experiments/run_nvidia_h200_sharegpt_capture.py`（单 lane 编排）、`scripts/experiments/nvidia_h200_serving_alignment_runner.py`（eager 算子 + Scheduler + Nsight 采集）、`evaluation/serving/experiments/sharegpt_ground_truth.py`（无 profiler 真值与 Scheduler trace）、`dataset_pipeline/serving_dataset/vllm_operator_trace.py`、`dataset_pipeline/serving_dataset/vllm_scheduler_trace.py` 与 `scripts/evaluation/serving/build_nvidia_h200_sharegpt_candidate.py`（转换）。

## 请求与 case

H200 的 15 个 canonical ShareGPT case 已发布到 `serving_dataset/sharegpt/nvidia_h200/`（vLLM 0.30.0 runtime 采集；vLLM 0.11.0 采集仍为非 canonical 候选）。每个 H200 case 以 `--target-hardware-id nvidia_h200 --runtime-version <v>` 重放同名 A800 canonical case 的 `requests.csv`（同一请求文件、同一到达时刻、同一 ShareGPT 采样 seed 42，runner 逐请求校验 prompt/output 长度），结果记录 `workload.request_template`（A800 manifest 路径与 SHA）；发布后的 canonical `requests.csv` 与 A800 同 case 逐字节相同。15 个 case：Qwen3-4B、Llama-3.1-8B、Qwen3-8B 的 TP1 QPS 4/16/64，Qwen3-32B TP2 与 Qwen3-30B-A3B TP2+EP2 的 QPS 1/4/16；suite manifest 与 A800 相同，模型 `config.json` SHA 与 suite 一致。

## 每个 case 的 raw artifact

输出根为 `/mnt/public/yangxinhao/serving_capture/<runtime_version>/nvidia_h200/<model_id>/<case_id>/`，只追加不覆盖：

| 路径 | 内容 | A800 对应物 |
|---|---|---|
| `host_request_path_probe_before.json` / `_after.json` | delivery-region probe（host only，fanouts 1..64，40 rounds，3 repeats），失败时写 `probe_failed` 记录 | `run_sharegpt_ground_truth.py` 的 bracket probe |
| `trace_off/ground_truth.json` + `logs/` | 无插桩 production run（graph + async scheduling + prefix caching） | `trace_off_gpu1_observer/ground_truth.json` |
| `trace_on/ground_truth.json` + `logs/` | 同配置加 Scheduler trace | `trace_on_batch1/ground_truth.json` |
| `trace_on/scheduler_steps.raw.jsonl` 及 `.runtime.json`、`.clock.json`、`.warmup.json`、`.bootstrap.jsonl`、0.11 的 `.rank-<r>.jsonl`/`.compilation-counter.rank-<r>.jsonl` | Scheduler trace 与 runtime depth evidence | `trace_on_batch1/scheduler_steps.raw.jsonl*` |
| `trace_on/scheduler_trace_observer_effect.json` | trace-off/trace-on 3% gate，本 case 实测 | `trace_on_publication_wait/scheduler_trace_observer_effect.json`（A800 为代表 case 继承） |
| `raw/ShareGPT-<model>_<TP1/TP2/EP2>_n128_qps<q>/` | `meta.json`、`arrival_schedule.json`、`serving_metrics.json`、`plugin_trace.json_provenance.json`、`plugin_trace.json_scheduler.jsonl`、`plugin_trace.json_operators.jsonl.rank<r>`、`nsys_profile.nsys-rep`、`nsys_profile.sqlite` | `serve_exp/nvidia_a800/sharegpt/ShareGPT-*` |
| `clocks.jsonl` | 每个 GPU stage 前后的 pstate、SM/mem clock、温度、功耗（H200 不能锁频，只记录） | A800 为锁频断言 |
| `case_capture.json`、`logs/*.log` | 命令、返回码、起止时间、GPU、snapshot commit | — |

A800 manifest 中的 `external_scheduler_depth/<sha>/ground_truth.json` 是 depth 元数据迁移时单独补采的 executor 证据；H200 的 trace-on run 本身写出 `.runtime.json`（live executor/vllm_config 的 `max_concurrent_batches`），v5 转换器直接使用它，因此不单独采集。

## 转换

`build_nvidia_h200_sharegpt_candidate.py --case-dir <case> --work-root <dir> --converter-source-commit <sha>` 先用 `sharegpt_raw_adapter`（`nvidia_h200` 与 A800 同走 arrival schedule、显式 `req_N` 映射和 nsys 分支）生成带 `alignment_steps` 的 compact case，再用 `midlayer.scheduler_capture_candidate.v5` 绑定 trace-on 真值、Scheduler trace 与实测 observer-effect，产出 `requests.csv`、`metrics.json`、`alignment_steps.json.zst`、`scheduler_steps.jsonl.zst`、`manifest.json`。候选只写 `<work-root>/{alignment_stage,candidate}/profiler_output/serving_dataset_compaction/`，从不写 `serving_dataset/`；canonical 发布由主 agent 决定。

trace-on 默认取 `<case>/trace_on/`。`run_nvidia_h200_sharegpt_capture.py rerun-trace-on --tag <tag>` 重采的 trace-on 写在 `<case>/trace_on_<tag>/`（`ground_truth.json`、`scheduler_steps.raw.jsonl*`、`scheduler_trace_observer_effect.json`），记录为 `<case>/case_capture.trace_on_<tag>.json`，原 `trace_on/` 保留；转换时用 `--trace-on-dir-name trace_on_<tag>` 显式选择，代表 case 的 gate 目录用 `--observer-representative-trace-on-dir-name` 独立选择（默认 `trace_on`，只能与 `--observer-representative-case-dir` 同用）。CLI 在 stage 1 之前校验：目录名须为 `trace_on` 或 `trace_on_<字母/数字/下划线>`，目录及 `ground_truth.json`、`scheduler_steps.raw.jsonl`（用本 case gate 时另需 `scheduler_trace_observer_effect.json`；代表 case 只需 gate 文件）必须存在，tagged 目录还须有 `case_capture.trace_on_<tag>.json` 且其 `case`、`hardware_id`、`runtime_version`、`rerun_of=trace_on`、`rerun_dir` 与本 case 一致并已写 `ended`。选中的目录名由 v5 writer 写入 `manifest.json` 的 `capture.run_id`、`scheduler_steps.provenance.capture_run_id` 与各 accepted `raw_sources[].relative_path`（如 `trace_on_fix1/ground_truth.json`），代表 gate 另由 `observer_effect_validation.validation_case_id` 标明来源 case，打印的 summary 同时列出 `capture_run_id` 与 `accepted_source_relative_paths`。本 case gate 的 `trace_on.run_id` 必须等于所选目录下的 `ground_truth.json`，否则 loader 以 `measured observer-effect instrumented run must be the accepted run` 拒绝。选哪个 trace-on 作为 canonical 由主 agent 决定，转换本身不做选择。

```bash
# 本 case 的 rerun 及其自身 gate（以 0.30 qwen3-4b/tp1_qps4 的 trace_on_fix1 为例）
python scripts/evaluation/serving/build_nvidia_h200_sharegpt_candidate.py \
  --case-dir /mnt/public/yangxinhao/serving_capture/0.30.0/nvidia_h200/qwen3-4b/tp1_qps4 \
  --trace-on-dir-name trace_on_fix1 \
  --work-root <scratch>/0.30.0/qwen3-4b/tp1_qps4.trace_on_fix1/work \
  --converter-source-commit <sha>
# 自身 gate 未通过时改为继承同 runtime 代表 case 的 gate
#   --observer-representative-case-dir /mnt/public/yangxinhao/serving_capture/0.30.0/nvidia_h200/qwen3-8b/tp1_qps4
#   [--observer-representative-trace-on-dir-name trace_on_<tag>]
```

H200 的 kernel 到 Scheduler step 归属不用 A800 的 sampler-tail `direct_copy` 启发式：算子插桩在每个正式 forward 外推 `midlayer-op-step:<iter>` NVTX range（Model Runner V2 的 sampler 另推 `midlayer-op-sample:<iter>`），转换器按“发起该 kernel 的 runtime API 调用落在哪个 range（同线程、同进程 correlation ID）”归属，再按每步首个 kernel 起点切连续 device window。

## vLLM 0.11.0 与 0.30.0 的差异

| 方面 | 0.11.0（与 A800 相同） | 0.30.0 | 采集侧处理 |
|---|---|---|---|
| Model runner | V1 `vllm.v1.worker.gpu_model_runner.GPUModelRunner`，采样在 `execute_model` 内 | 默认 Model Runner V2 `vllm.v1.worker.gpu.model_runner.GPUModelRunner`，`execute_model` 与 `sample_tokens` 分离；`gpu/warmup.py` 用合成 SchedulerOutput 调 `execute_model`，profile 用 `dummy_run=True` | 算子 trace 以 Scheduler 输出上的 iteration 戳（`_midlayer_operator_trace_iter_id`）而非 runner 调用计数定 step；V2 logits 在 `sample_tokens` 内计算，step 在 sampler 结束才关闭 |
| Async scheduling | 需 `--async-scheduling`，executor 暴露 `max_concurrent_batches=2`，日志打印 `Batch queue is enabled with size 2` | 默认开启；`max_concurrent_batches` 只在 `vllm_config`（V2 为 `pp+1=2`），`Batch queue` 日志降为 debug | depth evidence 走 `runtime_vllm_config_field`；A800 async 日志 marker 校验只用于 0.11 |
| 每步 CUDA Graph 证据 | 无 runtime 统计，A800 路径包装 V1 worker 的 `cudagraph_dispatcher.dispatch`，写每 rank observation（instrumentation v4，`rank_observations`） | `--cudagraph-metrics` 使 `ModelRunnerOutput.cudagraph_stats`（`CUDAGraphStat`）逐步给出 unpadded/padded/runtime mode；V2 不再经 V1 dispatcher | 0.11 用 v4（与 A800 同 schema），0.30 用 v2（与 C500 0.24、910B 0.18 同 schema） |
| Compilation config | `{"level":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}` | `level` 改为 `mode`：`{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}` | 按版本生成 |
| 日志 marker | `torch.compile takes` | `torch.compile took` | 按版本校验 |
| UniProcExecutor | `vllm.v1.executor.abstract` | `vllm.v1.executor.uniproc_executor` | 按版本 import |
| MoE 层与路由 | `FusedMoE`，`FusedMoE.select_experts` staticmethod | `FusedMoEFactory` 构造 `MoERunner`，路由在 `FusedMoERouter.select_experts` | 按版本 patch；`tok_per_expert` 宽度取全局 expert 数 |
| 请求 ID | 进程内 `req_N` 原样 | `InputProcessor.assign_request_id` 追加 `-<8 hex>` | eager 采集设 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`；HTTP 真值的 `cmpl-request-N-0-<hex>` 由既有正则解析 |
| benchmark client | `vllm.benchmarks.datasets` 模块 | `vllm.benchmarks.datasets` 包，`ShareGPTDataset`、`RequestFuncInput.request_id`、共享 session 均保留 | 无需改动 |

eager 对齐采集沿用 A800 raw 的 engine 参数（eager、chunked prefill、`max_num_batched_tokens=8192`、`max_model_len=8192`、`gpu_memory_utilization=0.8`），并显式 `async_scheduling=False`：0.11 默认如此，0.30 默认开启，显式关闭使两版本的 eager step 与 Scheduler iteration 一一对应。

## 环境与 GPU 纪律

h200-tmp 无外网；采集依赖根由 `MIDLAYER_H200_PUBLIC_ROOT` 或等价的 `--public-root` 显式指定，下文记为 `<public-root>`。vLLM 0.30.0 与 0.11.0 分别位于 `<public-root>/envs/vllm030`、`vllm011`（由 wheelhouse 离线安装），仓库 import 但 vLLM 不带的 `zstandard` 位于 `<public-root>/envs/midlayer_pydeps_cp312` 并追加到 `PYTHONPATH`。执行副本由 `scripts/experiments/deploy_nvidia_h200_serving_capture.sh` 从干净 commit 导出到 `<public-root>/src/midlayer_h200_serving_capture/<sha>/`（含 `SOURCE_COMMIT` 与现编 `build/lru_sim.so`）。每张 GPU 同一时刻只允许一个进程：每个 GPU stage 以升序对所占每张卡的 `<public-root>/h200_microbench/kcampaign/runs/gpu<N>.proc.lock` 取 `flock -x`，与 kernel campaign 共用同一把锁；GPU 0 属于其他实例，编排器拒绝使用。`run_nvidia_h200_sharegpt_capture.py plan --runtime-version <v> --public-root <public-root> --gpus <list> --pairs <n>` 打印按 lane 分配的 detached 启动命令（TP2/EP2 case 走 GPU 对，TP1 case 走单卡）。

## Smoke 结果（2026-09-26，GPU 3，qwen3-4b/tp1_qps4）

| 项 | vLLM 0.30.0 | vLLM 0.11.0 |
|---|---|---|
| probe before/after | 312 s / 313 s，成功 | 222 s / 222 s，成功 |
| trace_off / trace_on stage wall（含等锁） | 483 s / 325 s；首次冷编译 readiness 166 s，热缓存 46–50 s | 126 s / 102 s；readiness 76 s / 44 s |
| alignment（nsys eager）/ export | 111 s / 9 s；sqlite 186 MB，operators 143 MB | 93 s / 12 s |
| Scheduler trace | 8074 行，depth 2，FULL 7947 / PIECEWISE 109 / NONE 18（`above_capture_limit`），instrumentation v2 | 8066 行，depth 2，instrumentation v4（`rank_observations`），source identity 与 A800 canonical 的 `core=4573563d...cfbedd` 一致 |
| 每请求 scheduled tokens −（input+output−1） | 全为 0 | 全为 1（与 A800 canonical 相同：0.11 async scheduling 多调度一个 decode token） |
| observer-effect（3% gate） | 未通过：TTFT +6.28%（重复一次 +9.52%，均值 +0.95/+1.47 ms，15 ms 量级）；TPOT/E2E/throughput 均 <0.8% | 通过：TTFT 0.13%，TPOT 0.29%，E2E 0.27%，throughput 0.02% |
| alignment_steps | prefill/decode/mixed 均 available，每 phase 398 个算子（0.30 TP1 不再调用 all-reduce） | 均 available，399 个算子（含 TP1 all-reduce，与 A800 相同） |
| 最终 candidate | stage 1 完成；v5 被 observer gate 拒绝（writer 同样要求 `passed=true`） | 完整生成：`alignment_steps.json.zst`、`scheduler_steps.jsonl.zst`、`requests.csv`、`metrics.json`、`manifest.json`（schema 5，converter v5） |

单个 TP1 case 串行约 15–18 min（两次 probe 约 8–10 min 为纯 CPU 时间），首个 case 另加冷编译约 2 min；TP2 case 预计 20–30 min（权重加载与 nsys 数据量更大，未实测）。

## 已知限制

- vLLM 0.30.0 的 observer-effect gate 在 qwen3-4b/tp1_qps4 上三对 A/B 均未通过：TTFT +6.28%、+9.52%，把 Scheduler trace 的请求 ID 解析改为每请求只解析一次（`after_schedule` CPU 从 95 µs/step 降到 38 µs/step，输出不变）后为 +3.36%；TPOT/E2E/throughput 始终 <0.8%。同一 case 四次 trace-off 的 TTFT 均值为 15.06/15.38/15.58/16.04 ms（另一次为 bootstrap 已加载但未插桩），三次 trace-on 为 16.01/16.85/16.58 ms：插桩效应约 +0.5–1 ms，而 trace-off 自身的 run-to-run 波动约 1 ms，15 ms 量级 TTFT 上的单对 3% gate 处于噪声底。0.11.0 同 case 一次通过（TTFT +0.13%）。未通过的 case 不能以 `measured_for_this_case` 生成 v5 candidate；若同 runtime 的其他 H200 case 实测通过，可用 `--observer-representative-case-dir` 以 `inherited_from_representative_case` 转换（A800/C500/910B canonical 的 gate 均为代表 case 继承），否则该 runtime 的 raw 只能保留、不能出 candidate。所有 A/B 原始结果（`trace_off_r2/3`、`trace_on_r2/3`、`trace_off_bootstrap_loaded`）保留在 smoke 目录。
- 0.30.0 生产 run 的 observer-effect 定位（qwen3-4b tp1_qps4/qps16、qwen3-32b tp2_qps1 及 smoke 三对）：(1) 所有 0.30 trace-on run 的 request 0 TTFT 恒定 +18–20 ms（24.5–27 → 44–46 ms；0.11 为 +1.8 ms），时间落在首个 formal step 的 executor 窗口（37 ms，对照同规模 prefill 约 12 ms），该窗口内唯一的插桩代码是首次 depth-evidence 解析（GPFS 上 `Path.resolve`，冷态实测 4.6 ms），另有两次 bootstrap marker 的 GPFS append 落在首 step 调度/输出路径上；(2) 最后一次 `update_from_output` 内同步序列化+写 trace，使最后一个请求的输出晚到 88–260 ms，吞吐 delta（0.2–1.06%）全部来自这里；(3) 稳态每 step hook 开销约 20–40 µs（本地 microbench；TPOT +0.3%/+1.4%）；(4) 6 次 trace-on 中 2 次出现 56–105 ms 的 engine 停顿（off 5 次未见），原因未定位；(5) trace-off 自身 5 次 TTFT 均值 15.06–17.29 ms，生产 qps4 的 +9.5% 实为 trace-on 更快。`_uses_deferred_trace_io` 仅对 H200+vLLM 0.30.0 把 marker append、最终 trace 发布移到单独的 trace I/O 线程（sidecar 先于主 trace 原子发布），depth evidence 在首个 warmup submission 时解析并缓存；记录内容与同步版本逐字段一致（`tests/dataset_pipeline/serving/test_vllm_scheduler_trace_deferred_io.py` 以修复前代码生成的 golden 校验），其他硬件/版本行为不变。
- 0.30.0 无 V1 `cudagraph_dispatcher.dispatch` 路径，A800 v4 的每 rank dispatcher observation 与 compilation-counter 快照不可得，由 `cudagraph_stats`（v2）代替；0.30.0 的 `Batch queue is enabled` 为 debug 日志，A800 async 日志 marker 校验不适用。
- 0.11.0 TP2/EP2 trace-on（qwen3-30b-a3b/tp2_ep2_qps1，bf5f5e349）在 step vllm:1055 以 `A800 worker observation is missing the Scheduler step: rank=1` 杀死 EngineCore。根因：0.11 `MultiprocExecutor.execute_model` 以 `unique_reply_rank=output_rank` 只等 rank 0 的回复，rank 1 在其 `execute_model` 返回后才 append 的 `.rank-1.jsonl` 行与 `update_from_output` 无 happens-before（A800 同一路径，只是未触发）。修复后 rank observation 在到达时关联到已记录的 step（只消费带换行的完整行），非末 step 必在下一 formal update 前落盘（rank r 须完成 step N 才能参与 N+1 的 TP/EP collective）；封存 trace 时对末 step 最多等 30 s，超时 fail-loud。记录内容与按序到达时逐字节一致。
- 同一机制的后处理面：qwen3-30b-a3b/tp2_ep2_qps16 trace-on 服务完整结束，但 `_a800_scheduler_trace_summary` 以 `A800 worker execution envelope is outside executor envelope` 失败——1598 行中 6 行（vllm:144/226/230/305/308/324）的 rank 1 `worker_execution_end_ns` 比 executor RPC envelope 终点晚 157–278 µs。executor envelope 只以 reply rank 的回复收口，非 reply rank 的 `Worker.execute_model` 返回与之无先后关系。现在 reply rank 由 `.runtime.json` 的 executor class/版本/PP 与 rank 数按 vLLM 0.11 `_get_output_rank`（`world_size - TP`，本 runner PP=1 且无 KV connector，故为 rank 0）导出；reply rank 仍须完整落在 envelope 内，所有 rank 仍须 start 在 envelope 内且 start ≤ end，非 reply rank 的 end 越界不再报错，而是逐条记录在 summary 的 `worker_execution_envelope`（step、rank、越界 ns、计数与最大值），raw 行不改。只影响 worker-dispatch（`rank_observations`）路径；下游消费者（compact/replay/selected-step）只用各 rank 自身 start/end 或做 NSYS 时钟对齐，均不假定 rank 被 envelope 包含。qwen3-32b tp2_ep1_qps1/qps4 重算 summary 除新增字段外与已写出的 ground_truth.json 一致（越界 0）。
- TP2/EP2 未在 smoke 中实测（只有 GPU 3 可用）。0.30.0 MoE 路由在 `FusedMoERouter.select_experts` 捕获；若 runtime 选择 monolithic MoE kernel（不经 `select_experts`），eager 采集会 fail-closed 报错。
- A800 manifest 中单独补采的 `external_scheduler_depth/` 不再采集，depth evidence 由 trace-on 的 `.runtime.json` 提供。
- H200 不能锁频，`clocks.jsonl` 只记录每个 GPU stage 前后的 pstate 与 SM/mem clock；probe 与其他 lane 的 GPU stage 同时运行时会共享 host CPU。
