# EngineCore host-round cadence probe (ascend_910b)

独立机制微基准：在 910B 容器内用真实 vLLM 0.18 组件复刻 EngineCore busy-loop 的一轮 host 节拍，测量 admission 循环里没有任何 Host profile 覆盖的 host 段。产物只提供 host 循环节拍参数（单位秒），设备 stub 时长只作校准证据，绝不进入 Forward。

## 机制

`engine_core_host_round_runner.py` 在容器 `yxh_midlayer_serving_910b1` 内构造：真实 `AsyncScheduler`（`async_scheduling=True`，与 910B serving 配置一致）、`step_with_batch_queue` 双深度批量队列镜像、非阻塞 `collective_rpc`、单线程 `ThreadPoolExecutor` 的 WorkerAsyncOutput 镜像、`synchronize_input_prep` 事件屏障、ACLGraphWrapper pre-replay `current_stream().synchronize()` 壁垒语义，以及 copy-stream D2H + tolist 的输出唤醒链。设备负载是固定的 mm 链 stub（graph lane 为捕获的 NPUGraph 回放，eager lane 为逐算子 mm 链）：stub 时长与请求无关，因此轮节拍对 R 的任何依赖按构造属于 host。每轮按段计时（`_RoundTimings`），段单位纳秒。

graph lane 的 submit 链（引擎专属，无任何既有 profile 覆盖）= `prep_sync + h2d + pre_replay_sync + sample_submit + thread_submit + poll + queue_decision`；wake 链 = `wake_block + d2h_event_sync + tolist + put`。eager lane 的 submit 链同上去掉 `pre_replay_sync`（eager 无 pre-replay 同步）。`plan`/`update` 由 SchedulerHostProfile 定价，`state_host`（≈ model_runner_input_prepare）由 host_runtime_prepare profile 定价，`submit_lane`（逐算子 enqueue ≈ 9.8µs/op，与 HostLaunchProfile eager 9.35µs/op 一致）由 HostLaunchProfile 定价——这四类段被排除在本探针参数之外，避免双计。

## 可复测命令

在 `910b-1` 的 `yxh_midlayer_serving_910b1` 容器内、仓库根 `/mnt/gpfs/midlayer_yxh`：

`/usr/local/python3.11.14/bin/python microbench/ascend_910b/runtime/engine_core_host_round/engine_core_host_round_runner.py --output-dir profiler_output/microbench/ascend_910b/engine_core_host_round/<run_dir> --device-id 1 --baseline-commit <full-commit-sha> --expected-source-sha256 <runner-sha256> --container-name yxh_midlayer_serving_910b1 --model-path /mnt/gpfs/models/Qwen3-8B`

矩阵：graph lane R∈{1,8,32,64,120,128}×stub{small(2-op),large(64-op)}；eager lane R∈{1,32,120,128}@ops700 与 ops∈{64,256}@R=32；每 cell 30 warmup + 200 测量轮。manifest 自校验 runner sha256 与 status fail-closed。

## 参数（run4，引擎 profile 唯一来源）

`configs/host_timing/engine_core_host_round_profiles/ascend_910b.json`。函数形式由机制与段级证据先验固定（submit 各组成段在 R 上测平、跨 run 拟合斜率变号即噪声；wake 斜率三次 run 全为正，对应逐请求 output put/tolist）：graph submit = 526.898µs 截距常数；graph wake = 9.9499µs + 10.928ns/req；eager submit = 438.20675µs 截距常数；eager wake = 8.9607µs + 18.721ns/req。lane 分派：纯 decode → graph，prefill/mixed → eager（qps64 观测 1071/1098 步 FULL 且纯 decode；6 步纯 decode 落 NONE 的误分类上界 ≈ 6×80µs，可忽略）。

## 复测记录

- run2（2026-09-23，修正 fsync 伪影后；runner sha256 `4e3b3cc…`，R≤120）：engine 链与 run3/run4 见 `references/run2_*`。
- run3（2026-09-24 复测，同源同命令）：graph submit 中位数 551µs vs run2 529µs（+4%），wake 10.6 vs 9.8µs，eager submit 487 vs 446µs，各段中位差均在 host 抖动内；斜率符号一致。
- run4（2026-09-24，runner sha256 `7b6ad67…`，矩阵扩到 R=128）：参数来源；R≤120 中位数与 run2 一致（graph submit 526.9 vs 525.9µs）。

## references/

`run{2,3,4}_{summary,manifest,trials}` 为三次运行的原始产物副本（sha256 见各自 manifest 与引擎 profile JSON 的 artifacts）。
