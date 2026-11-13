# Midlayer

Midlayer 是面向 LLM 推理的轻量级解析性能工具，包含白盒 Kernel 性能模型与 Serving 仿真器。它把系统问题分成两个边界清晰的层次：Serving 层将请求、模型和部署转换为逻辑算子与资源时间线；Kernel 层根据算子、离散 launch 和硬件资源预测一次调用的执行时间。

```text
Production workload + Model + Deployment
                    │
                    ▼
             Scheduler / Router
                    │
                    ▼
       Logical operators and transfers
                    │
                    ▼
          Complete KernelSpec + HWSpec
                    │
                    ▼
        Unified per-kernel prediction
                    │
                    ▼
     Resource timeline and system metrics
```

Production Forward 是零训练数据的解析模型。目标算子、kernel 和 Serving workload 的执行时间只用于 evaluation，不能用于拟合 Forward 参数或 residual correction。仓库底线、按需流程与默认工作策略见 [AGENTS.md](AGENTS.md)，设计思想分别见 [Serving 系统设计](docs/serving_system.md) 和 [Kernel 解析性能模型](docs/kernel_performance_model.md)。

## 当前范围

生产硬件是 NVIDIA A800、MetaX C500、Ascend 910B 和 NVIDIA H200，其中 H200 的 production Kernel/Serving 基础尚为部分闭合（已知缺口见 [开发进度](docs/dev_progress.md)）；Samsung HBM-PIM、UCB Gemmini 与 Groq LPU 是研究参考平台。所有硬件共用同一 Kernel predictor 接口，硬件差异由 HWSpec、launch package、能力和证据表达。

“能够运行”“机制完整”和“经过真值验证”是不同状态。Serving 组合的 Supported/Partial/Projected 边界见 [Serving 系统设计](docs/serving_system.md#当前能力范围支持哪些如何支持)；Kernel 预测的 `complete`、`partial`、`aggregate_only` 和 `compatibility` 由 prediction diagnostics 报告，不能从相邻 case 推断为支持。

当前进度与尚未闭合的方向由 [开发进度](docs/dev_progress.md) 维护。

`main` 保留已验收的稳定版本，`dev` 汇集开发功能；新 feature 在独立分支和 worktree 中推进，再审阅集成到 `dev`。Agent 权限及集成要求以 [AGENTS.md](AGENTS.md#git-与操作权限) 为准。

## 仓库入口

| 路径 | 职责 |
|---|---|
| `src/kernel/` | Kernel contract、硬件规格、launch selection 与 predictor |
| `src/serving/` | 请求、部署、Scheduler、workload lowering、时间线与指标 |
| `evaluation/` | Case、observation、comparison、gate 与 report |
| `dataset_pipeline/` | Profiler 采集、转换、聚合与一致性校验 |
| `scripts/` | 薄工作流入口和 profiler 必须直接执行的 runner（全量索引见 `scripts/README.md`） |
| `configs/` | 人工维护的硬件、dataset、evaluation 与 baseline 配置 |
| `kernel_dataset/` | Git LFS 跟踪的 canonical 单 kernel ground truth |
| `launch_data/` | Git LFS 跟踪的 canonical GEMM launch selection；四种硬件统一使用确定性 CSV 和绑定版本、证据、domain 与内容哈希的 manifest |
| `serving_dataset/` | Git LFS 跟踪的 canonical Serving 请求、动态输入与具备完整 setting 的实测对齐数据；包含 `sharegpt/` 与经严格转换的 `production/`，目标模型 debug Production case 可发布经审阅的 partial alignment |
| `profiler_output/` | 本地 profiler 原始与中间产物；campaign 登记与保留级别见 `docs/profiler_output_registry.md` |
| `experiment_logs/` | 预测、评估和报告等可再生运行结果；Profiler 采集及直接处理产物归 `profiler_output/`，见 `docs/profiler_output_registry.md` |
| `microbench/` | 按硬件和物理机制组织的独立微基准 |
| `tests/` | 按 owner 镜像的常驻回归测试，统一经 `scripts/run_tests.py` 运行 |
| `docs/` | 设计说明、精度与机制记录、硬件专项文档与诊断裁决 |
| `archive/` | 已退出活动路径的历史实现，仅供追溯 |

目录 owner、依赖、数据边界和 Git LFS 规则只由 [AGENTS.md](AGENTS.md) 定义。

每个 `launch_data/<hardware_id>/` 目录由压缩的确定性 GEMM launch CSV 和绑定证据、内容哈希的 manifest 构成；lookup 只含物理发射字段、不消费 duration，具体 schema 与绑定项由各目录 manifest 定义。

## 快速开始

仓库直接从根目录运行，不需要 `pip install -e .`。克隆后必须初始化 `third_party/protobridge` submodule（ProtoBridge 上游 surrogate 的唯一来源，`src/kernel/predictors/protobridge/upstream.py` 会对其做内容寻址校验）：

```bash
git clone --recurse-submodules <repository-url>
# 已有克隆执行：
git submodule update --init --recursive
```

固定 Conda `r2r_qwen3_next` 环境必须提供 `zstandard==0.25.0`，用于读取 canonical Serving 与 launch 压缩载荷；测试统一通过仓库入口切换到该环境，避免 shell 当前的系统 Python 或 Conda base 污染测试。硬件 profiler、vLLM、SGLang 和外部模拟器按任务单独准备。

```bash
conda activate r2r_qwen3_next
mkdir -p build
cc -O3 -shared -fPIC -o build/lru_sim.so src/kernel/csrc/lru_sim.c
python scripts/run_tests.py
```

无参数的测试命令运行全部受维护测试，默认 96 个 worker 用小批次按需派发待运行任务，减少少数长队列拖尾；用于代码 Goal 收尾和 main 集成/发布候选验收；日常修改采用直接相关的定向检查，把 nodeid 追加到 `python scripts/run_tests.py` 后即可。常驻套件只保留稳定物理机制、公共 contract、fail-closed 边界和少量端到端 smoke；独立微基准 runner/parser、一次性实验报告、固定 SHA/快照和重复参数穷举不作为长期回归测试维护。具体触发条件见 [验证与交付](AGENTS.md#验证与交付)。后续 Python CLI 也在上述 Conda 环境中运行。

验证 vLLM 0.30 `SimpleCPUOffloadConnector` 的受控 CPU store、跨请求 Host 命中与 load，可运行 `python -m evaluation.serving.experiments.simple_cpu_short_case --help`；固定比例 Production Trace 窗口使用 `python -m evaluation.serving.experiments.production_trace_fixed_tier_request --help`。两者都需要显式传入隔离 SDK Python 与锁定源码路径；后者还要求容量和墙钟上限。已验证范围、运行结果与剩余边界见 [SimpleCPU 短请求记录](experiment_logs/dev_history_cleanup_20261103/archived_records/kv_offload_simple_cpu_short_20261029.md) 和 [Production 窗口记录](experiment_logs/dev_history_cleanup_20261103/archived_records/kv_offload_production_windows_20261029.md)。

运行 DeepSeek-V4 Flash 的静态 Roofline 仿真（H200 规格、TP4/PP1、EP/MTP 关闭、greedy）：

```bash
CUDA_VISIBLE_DEVICES='' conda run -n r2r_qwen3_next python scripts/evaluation/serving/report_model_operator_probe.py \
  --model configs/model/deepseek-v4-flash.json --tensor-parallel-size 4 \
  --mtp-mode off --prefill-tokens 8 --v4-eager-metadata --v4-source-max-model-len 4096 \
  --v4-source-max-batched-tokens 4096 --no-v4-source-use-deep-gemm \
  --v4-indexer-selection-source declared_causal_topk \
  --v4-routing-source vllm_0_30_modular_cuda_top6_pure_text \
  --target-sampling-source vllm_0_30_plain_greedy_bf16 \
  --synthetic-input-token-ids --summary-only \
  --output experiment_logs/evaluation/operator_support/v4_flash_simulation/short.json
```

输出 `steps[].latency_ms` 为完整模型单步时间线预测，覆盖 prefill、连续两步 decode 和独立 mixed batch；不会把所有 TP rank 的时间相加。将 `--prefill-tokens` 改为 `127`、更换输出路径可运行 C128 边界。输入 token IDs 由显式 synthetic 选项生成，专家分布使用声明的 seeded assumption；也可用 `--embedding-token-ids <JSON>` 指定四阶段输入。此入口模拟工作量和时间，不运行模型或生成真实 token；它是 FLOPs/字节基线，完整源码对齐、正常容量、真实 Scheduler/QPS16 和硬件精度仍分别未验收。存在未计价调用时摘要模式报错；去掉 `--summary-only` 可输出完整算子明细。

以下为按需使用的评估命令，验收范围遵循 [AGENTS.md](AGENTS.md#验证与交付)，不要求每次修改全部执行。单硬件 Kernel evaluation：

```bash
python scripts/evaluation/kernel/run_kernel_evaluation.py \
  --hardware-id nvidia_a800 \
  --mode forward
```

不传 `--mode` 时默认为 `standard`（oracle + forward + roofline），不会加载外部 PipeWeave checkout；仅在显式指定 `--mode pipeweave` 或 `--mode all` 时运行 PipeWeave。

运行默认完整 Kernel matrix：

```bash
python scripts/evaluation/run_full_evaluation.py --part kernel
```

运行最小 Serving conformance gate（按 ShareGPT 配置检查执行链路）：

```bash
python scripts/evaluation/run_full_evaluation.py \
  --hardware-id nvidia_a800 \
  --serving-suite sharegpt \
  --part serving
```

此 gate 为每个配置构造固定单请求（16 个输入 token、2 个输出 token），使用一致性检查后端，不回放真实 ShareGPT 请求流，因此不能替代 ShareGPT QPS16 性能精度验收。

生成 Kernel 与 Serving 标准报告使用统一入口：`--all` 同时更新两份报告，`--kernel-only` 或 `--serving-only` 只更新对应报告并复用另一部分最近一次已验证产物；输出写入 `experiment_logs/evaluation/standard_report/`（`kernel_report.md` 与 `serving_report.md`）。Serving 统一执行虚拟 Tensor 路径，须显式提供现有 SDK Python 及 vLLM 源码路径；未迁移 case 保留 unsupported 且命令非零。硬件覆盖与具体参数以 `--help` 为准。

下面的 `--all` 和 `--serving-only` 示例未传 `--case`，会运行全部所选 ShareGPT cases，仅在用户明确要求全 case 评估时使用。日常 Serving 验收用 `--case` 显式选择本轮相关的 QPS16 case；`--all` 表示更新两份报告，不等于已限定 Serving case 范围。

Serving 报告默认按选中 case 数启动 worker；不填 `--jobs` 或传 `0` 都表示每个 case 一个 worker，传正整数可手动限制并发。

```bash
python scripts/evaluation/run_standard_evaluation_report.py --all --vllm-sdk-python "$VLLM_SDK_PYTHON" --vllm-source-root "$VLLM_SOURCE_ROOT"
python scripts/evaluation/run_standard_evaluation_report.py --kernel-only
python scripts/evaluation/run_standard_evaluation_report.py --serving-only --vllm-sdk-python "$VLLM_SDK_PYTHON" --vllm-source-root "$VLLM_SOURCE_ROOT"
```

报告分别记录采集配置、实际仿真配置与各自 snapshot；同步eager的Roofline结果不能冒充CUDA graph/实际provider验收。`--case`可选择具体case，`--reuse-virtual-run <既有运行目录>`只重建与当前命令完全一致的已完成虚拟路径结果的汇总，不重跑仿真。CLI保留匿名延迟分布，报告不将其还原为逐请求误差。 标准入口已删除路径选择开关；`--reuse-serving-results`、`--nvidia-h200-dataset-root` 和 H200 vLLM 0.11.0 属于独立的 [Scheduler 诊断入口](scripts/evaluation/serving/diagnostics/run_scheduler_depth_breakdown.py)（`--full-serving-report-only`，历史结果使用 `--reuse-results`），标准入口会明确拒绝。`--kernel-only` 可继续读取历史 state，保留其原始 Serving 身份。

需要固定并重放真实观测到的 Scheduler step 顺序、batch membership 和 token progress，以区分单步计时误差与动态 batching 反馈误差时，使用 **Scheduling Trajectory Replay**：`evaluation.serving.traces.scheduling_trajectory_replay.analyze_scheduling_trajectory_replay()` 或以下 CLI。输出包含逐 step、分 phase 的 TTFT、TPOT、E2E、吞吐预测误差；真实 duration 只用于 evaluation comparison，不进入 Forward。

```bash
python scripts/evaluation/serving/scheduler/run_scheduling_trajectory_replay.py \
  --case ascend_910b/qwen3-4b/tp1_qps16 \
  --host-launch-profile configs/host_timing/host_launch_profiles/ascend_910b.json
```

ShareGPT raw 的审计和 compact candidate 构建入口是 `scripts/evaluation/serving/build_sharegpt_candidates.py`：raw 输入根为 `profiler_output/serving/`，输出先进入发布链硬编码工作区 `profiler_output/serving_dataset_compaction/`，不能直接覆盖 canonical dataset。

运行三硬件 Qwen3-4B QPS16 的公平 colocated 与 1P×1D PD 对照实验：

```bash
python scripts/evaluation/serving/run_pd_disaggregation_study.py \
  --run-id qwen3-4b-qps16-1p1d \
  --jobs 3
```

结果和图表数据写入 `experiment_logs/serving/pd_disaggregation/<run-id>/`；可重复传入 `--config`、`--case` 和 `--topology` 组合既有 ShareGPT suite case，具体参数以 `--help` 为准。

实验 manifest v9 可从同一配置预览并运行 CPU Serving 仿真。以下示例使用两个合成短请求、每请求 4 个输出 token 和测试 KV 块数，要求测试环境中可用的 pinned vLLM 0.11 CPU Scheduler；它用于验证入口和 Graph 配置传递，其 A800 analytical-launch 计时不支持性能或精度结论。

```bash
python scripts/experiments/run_serving_experiments.py \
  --manifest configs/evaluation/experiments/cpu_qwen3_short_example.json \
  --run-dir experiment_logs/serving/cpu_example --dry-run
python scripts/experiments/run_serving_experiments.py \
  --manifest configs/evaluation/experiments/cpu_qwen3_short_example.json \
  --run-dir experiment_logs/serving/cpu_example --cpu-run
```

`run_summary.json` 的预览列出原始选项、仿真 CLI 归一化选项和待执行命令；执行后每个 cell 的 `command_manifest.json`、`result.json` 和 `settings.md` 分别记录运行、机器结果和实际设置。v9 的 `simulation` 字段使用仿真 CLI 的长选项名（去掉 `--`）；无值开关使用 JSON `null`，列表用于接受多个值的选项，默认值由仿真 CLI 负责。修改示例中的 workload、模型或容量时，必须符合底层仿真边界，不得把测试容量标为正常设备容量。失败、取消或中断的 cell 可用相同 manifest 加 `--resume --cpu-run` 重试，旧 attempt 保留；信号终止记录信号及未知原因，不假定是用户取消。成功结果可用 `--resume --cpu-run` 复用：入口检查配置、输入、代码、结果、Python 标准库、site-packages 及Python 安装前缀 `lib/` 中的共享库、动态链接器登记的系统库及其所在目录内的共享库文件内容；设置了 `LD_LIBRARY_PATH`、`LD_PRELOAD`、`LD_AUDIT` 或用户环境中的 `PYTHONPATH` 时拒绝复用；存在未指纹化的 editable Python 源码或其他已有 Python 搜索路径时也拒绝复用。检查仍不覆盖其他路径的系统库或运行中瞬时变化。校验会读取 Python 库文件，大环境需要额外时间。v1 的 `--real-run` 可能启动真实 GPU 服务，v2–8 与旧专用脚本仍沿用各自入口。

要用同一 manifest 重新执行已成功的 v9 cell，可加 `--resume --rerun-success --cpu-run`。这会创建新的 attempt、记录原 attempt 路径并实际重跑；它不复用旧结果，也不证明两次运行的依赖相同。

自动复用以运行期间不修改代码、输入和依赖为前提，并比较运行前、运行后与复用时的身份；不监控修改后又恢复的瞬时行为。首批单实例和 1P1D CPU 场景分别记录启动器及所选 SDK Python 的搜索路径；已有路径必须位于本仓库代码、已检查的 Python 库或显式校验的 vLLM 源码范围内，否则拒绝自动复用。依赖内容无法读取、目录链接超出检查范围、存在未指纹化 editable 源码或使用自定义搜索路径时也拒绝；可使用 `--resume --rerun-success --cpu-run` 显式重跑。运行记录中主要包的安装清单摘要本身不代表逐文件内容校验，SDK 从源码加载 vLLM 时源码身份单独记录。

首批场景的已有运行证据还包括 `/etc/gitconfig`、`/etc/ssl/openssl.cnf` 和动态链接器缓存；入口记录前两个文件的内容或缺失状态，并检查缓存。其他未被上述身份检查覆盖的依赖环境不支持自动复用，应显式重跑；已有场景的文件跟踪不能证明任意系统路径均已覆盖。

PipeWeave 的源码和模型等依赖不在上述身份检查范围内，成功结果不支持自动复用；可用 `--resume --rerun-success --cpu-run` 显式重跑。

如果入口被强制终止，记录可能保留为 `running`。对于已记录入口进程和子进程组身份的新记录，`--resume --cpu-run` 只有在同一启动周期和 PID 命名空间内确认入口及进程组均不存在时才创建新 attempt；旧记录保持原样，原因标为未知。进程仍在运行、无法检查或旧记录缺少身份时会拒绝重试，不会自动终止其他进程。

同一入口也可运行已有的 1P1D CPU 仿真路径：将上面两条命令的 `--manifest` 改为 `configs/evaluation/experiments/cpu_qwen3_pd_example.json`，并使用新的 `--run-dir`。该示例同样使用合成短请求和测试 KV 块数；P/D 的 Scheduler、逻辑 rank 与 KV handoff 分别记录，通信计时使用开发用 ProtoBridge 投影，不构成实机或精度验收。

v9 CPU 执行要求可验证的 Git 工作树；仅包含 `SOURCE_COMMIT` 声明的执行副本不能证明文件内容，入口会拒绝执行。

更具体的 Serving 参数以对应 CLI 的 `--help` 和版本化 experiment manifest 为准，避免 README 复制一份易漂移的长命令。硬件采集前阅读 [AGENTS.md](AGENTS.md) 的硬件与授权规则；C500 profiler 任务还须按 [C500 profiling contract](docs/metax_c500/profiling.md) 的相关章节执行。

## 文档

**设计**

- [Serving 系统设计](docs/serving_system.md)：从请求、调度和部署到算子时间线与系统指标；§11 是 helper-launch 发射 contract。
- [Kernel 解析性能模型](docs/kernel_performance_model.md)：从算子、launch 和硬件资源到执行时间。
- [开发进度](docs/dev_progress.md)：当前能力状态和仍未闭合的工作。

**开发与执行**

- [AGENTS.md](AGENTS.md)：仓库底线、流程触发条件和默认策略。
- [Agent 操作流程](docs/agent_workflow.md)：按需查阅分工、worktree、集成、存储、硬件收尾与验证步骤。
- [Goal 模板](docs/goal_template.md)：独立 worker Goal 的唯一模板；短小只读咨询无需建立独立 Goal。
- [Modal 使用说明](docs/modal_readme.md)：客户端环境、获授权后的云端执行、结果回传和清理。

**硬件专项**

- [MetaX C500 profiling](docs/metax_c500/profiling.md)：C500 专属采集规则与工具限制。
- [MetaX C500 架构](docs/metax_c500/architecture.md)：C500 架构编程抽象、执行过程与微基准参考值。
- [NVIDIA H200 serving 采集手册](docs/nvidia_h200/capture.md)：0.11.0 与 0.30.0 两个 runtime 的采集链差异。

**数据**

- [profiler_output 登记](docs/profiler_output_registry.md)：`profiler_output/` campaign 登记与保留级别、`experiment_logs/` 组织约定、压缩归档的还原方法。

**诊断裁决**

- [Serving 误差定位流程](docs/serving_accuracy_diagnosis.md)：从 paired TTFT/TPOT 到固定 Scheduler 轨迹、step、runtime 与单 Kernel 硬件闭环的标准诊断方法。
- [建模决策录](docs/decisions.md)：各硬件定价机制裁决（护栏 / 排除 / unsupported）与带重开条件的未闭合事项索引。

普通 Kernel 评估现在默认生成独立离线 HTML 概要，并在终端输出路径：

```bash
python scripts/evaluation/kernel/run_kernel_evaluation.py --hardware-id nvidia_h200 --mode standard
# 可指定 --report-output experiment_logs/evaluation/kernel/example.html
# 内部配对计时可显式 --no-report --no-report-reason "paired measurement"
```

概要复用同一次评估的 semantic results，展示真实覆盖/缺失、方法误差、时延对比与完整参数/硬件快照；不额外运行预测。默认路径为独立 run 目录，标准 Kernel Markdown 和跨硬件矩阵链接到对应概要，Serving 报告独立保留。pure-Forward timing 仍排除报告开销；报告 sidecar 记录生成和写盘成本，完整 CLI 墙钟还包含进程启动及 sidecar 写入，不能用内部纯预测计时替代。
