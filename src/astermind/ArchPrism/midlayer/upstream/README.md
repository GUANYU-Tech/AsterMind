# Midlayer

Midlayer 是面向 LLM 推理的白盒 Kernel 性能模型与 Serving 仿真器。它把系统问题分成两个边界清晰的层次：Serving 层将请求、模型和部署转换为逻辑算子与资源时间线；Kernel 层根据算子、离散 launch 和硬件资源预测一次调用的执行时间。

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

Production Forward 是零训练数据的解析模型。目标算子、kernel 和 Serving workload 的执行时间只用于 evaluation，不能用于拟合 Forward 参数或 residual correction。完整强制规则见 [AGENTS.md](AGENTS.md)，设计思想分别见 [Serving 系统设计](docs/serving_system.md) 和 [Kernel 解析性能模型](docs/kernel_performance_model.md)。

## 当前范围

生产硬件是 NVIDIA A800、MetaX C500、Ascend 910B 和 NVIDIA H200，其中 H200 的 production Kernel/Serving 基础尚为部分闭合（已知缺口见 [开发进度](docs/dev_progress.md)）；Samsung HBM-PIM、UCB Gemmini 与 Groq LPU 是研究参考平台。所有硬件共用同一 Kernel predictor 接口，硬件差异由 HWSpec、launch package、能力和证据表达。

“能够运行”“机制完整”和“经过真值验证”是不同状态。Serving 组合的 Supported/Partial/Projected 边界见 [Serving 系统设计](docs/serving_system.md#当前能力范围支持哪些如何支持)；Kernel 预测的 `complete`、`partial`、`aggregate_only` 和 `compatibility` 由 prediction diagnostics 报告，不能从相邻 case 推断为支持。

当前进度与尚未闭合的方向由 [开发进度](docs/dev_progress.md) 维护。

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
| `experiment_logs/` | 代码运行的产出结果（评估报告、对比产物），全部可再生、无例外，见 `docs/profiler_output_registry.md` |
| `microbench/` | 按硬件和物理机制组织的独立微基准 |
| `tests/` | 按 owner 镜像的常驻回归测试，统一经 `scripts/run_tests.py` 运行 |
| `docs/` | 设计说明、精度与机制记录、硬件专项文档与诊断裁决 |
| `archive/` | 已退出活动路径的历史实现，仅供追溯 |

目录 owner、依赖、数据边界和 Git LFS 规则只由 [AGENTS.md](AGENTS.md) 定义。

每个 `launch_data/<hardware_id>/` 目录由压缩的确定性 GEMM launch CSV 和绑定证据、内容哈希的 manifest 构成；lookup 只含物理发射字段、不消费 duration，具体 schema 与绑定项由各目录 manifest 定义。

## 快速开始

仓库直接从根目录运行，不需要 `pip install -e .`。固定 Conda `r2r_qwen3_next` 环境必须提供 `zstandard==0.25.0`，用于读取 canonical Serving 与 launch 压缩载荷；测试统一通过仓库入口切换到该环境，避免 shell 当前的系统 Python 或 Conda base 污染测试。硬件 profiler、vLLM、SGLang 和外部模拟器按任务单独准备。

```bash
mkdir -p build
cc -O3 -shared -fPIC -o build/lru_sim.so src/kernel/csrc/lru_sim.c
python scripts/run_tests.py
```

默认门禁运行全部受维护测试。常驻套件只保留稳定物理机制、公共 contract、fail-closed 边界和少量端到端 smoke；独立微基准 runner/parser、一次性实验报告、固定 SHA/快照和重复参数穷举不作为长期回归测试维护。定向测试把 nodeid 直接追加到 `python scripts/run_tests.py` 后即可。

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

运行单硬件 Kernel evaluation：

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

运行仓库内 Serving gate：

```bash
python scripts/evaluation/run_full_evaluation.py \
  --hardware-id nvidia_a800 \
  --serving-suite sharegpt \
  --part serving
```

生成 Kernel 与 Serving 标准报告使用统一入口：`--all` 同时更新，`--kernel-only` 或 `--serving-only` 只更新对应报告并复用另一部分最近一次已验证产物；输出写入 `experiment_logs/evaluation/standard_report/`（`kernel_report.md` 与 `serving_report.md`）。Serving 统一执行虚拟 Tensor 路径，须显式提供现有 SDK Python 及 vLLM 源码路径；未迁移 case 保留 unsupported 且命令非零。硬件覆盖与具体参数以`--help`为准。

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

更具体的 Serving 参数以对应 CLI 的 `--help` 和版本化 experiment manifest 为准，避免 README 复制一份易漂移的长命令。硬件采集前必须阅读 [AGENTS.md](AGENTS.md) 的机器与安全规则；MetaX C500 还必须完整阅读 [C500 profiling contract](docs/metax_c500/profiling.md)。

## 文档

**设计**

- [Serving 系统设计](docs/serving_system.md)：从请求、调度和部署到算子时间线与系统指标；§11 是 helper-launch 发射 contract。
- [Kernel 解析性能模型](docs/kernel_performance_model.md)：从算子、launch 和硬件资源到执行时间。
- [开发进度](docs/dev_progress.md)：当前能力状态和仍未闭合的工作。
- [Goal 模板](docs/goal_template.md)：独立 worker Goal 的唯一统一模板。

**硬件专项**

- [MetaX C500 profiling](docs/metax_c500/profiling.md)：C500 专属采集规则与工具限制。
- [MetaX C500 架构](docs/metax_c500/architecture.md)：C500 架构编程抽象、执行过程与微基准参考值。
- [NVIDIA H200 serving 采集手册](docs/nvidia_h200/capture.md)：0.11.0 与 0.30.0 两个 runtime 的采集链差异。

**数据**

- [profiler_output 登记](docs/profiler_output_registry.md)：`profiler_output/` campaign 登记与保留级别、`experiment_logs/` 组织约定、压缩归档的还原方法。

**诊断裁决**

- [Serving 误差定位流程](docs/serving_accuracy_diagnosis.md)：从 paired TTFT/TPOT 到固定 Scheduler 轨迹、step、runtime 与单 Kernel 硬件闭环的标准诊断方法。
- [建模决策录](docs/decisions.md)：各硬件定价机制裁决（护栏 / 排除 / unsupported）与带重开条件的未闭合事项索引。

- [AGENTS.md](AGENTS.md)：唯一强制仓库规范。
