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
| `profiler_output/` | 本地 profiler 原始与中间产物 |
| `experiment_logs/` | 本地 evaluation 与实验记录 |
| `microbench/` | 按硬件和物理机制组织的独立微基准 |
| `tests/` | 按 owner 镜像的常驻回归测试，统一经 `scripts/run_tests.py` 运行 |
| `docs/` | 设计说明、精度与机制记录、硬件专项文档与诊断裁决 |
| `archive/` | 已退出活动路径的历史实现，仅供追溯 |

目录 owner、依赖、数据边界和 Git LFS 规则只由 [AGENTS.md](AGENTS.md) 定义。

每个 `launch_data/<hardware_id>/` 目录由 zstd 压缩的 `gemm_launch_selection.csv.zst` 和 `gemm_launch_selection.manifest.json` 构成。解压后的 CSV 只以 dtype、variant、shape、layout、transpose、epilogue 和 bias 等物理输入选择 tile、pipeline depth、split-K 与 reduction；模型位置和 TP/EP 等语义上下文只保留在 domain/provenance，不参与 Forward lookup。manifest 同时绑定压缩文件与解压内容 SHA256，以及由 scheduling 资源上限及 L1/L2 容量构成的 launch-only 硬件 contract、软件版本、证据、capture baseline、domain 和字段 schema；频率、带宽、机制 service、注释和 YAML 排版不改变 launch provenance。loader 不兼容未压缩 CSV 或旧 JSON，也不消费 duration。

## 快速开始

仓库直接从根目录运行，不需要 `pip install -e .`。固定 Conda `r2r_qwen3_next` 环境必须提供 `zstandard==0.25.0`，用于读取 canonical Serving 与 launch 压缩载荷；测试统一通过仓库入口切换到该环境，避免 shell 当前的系统 Python 或 Conda base 污染测试。硬件 profiler、vLLM、SGLang 和外部模拟器按任务单独准备。

```bash
mkdir -p build
cc -O3 -shared -fPIC -o build/lru_sim.so src/kernel/csrc/lru_sim.c
python scripts/run_tests.py
```

默认门禁以 48 workers 运行全部受维护测试。常驻套件只保留稳定物理机制、公共 contract、fail-closed 边界和少量端到端 smoke；独立微基准 runner/parser、一次性实验报告、固定 SHA/快照和重复参数穷举不作为长期回归测试维护。定向测试把 nodeid 直接追加到 `python scripts/run_tests.py` 后即可。

运行单硬件 Kernel evaluation：

```bash
python scripts/evaluation/kernel/run_kernel_evaluation.py \
  --hardware-id nvidia_a800 \
  --mode forward
```

不传 `--mode` 时默认运行 `Forward → Oracle → Roofline`，不会加载外部 PipeWeave checkout；仅在显式指定 `--mode pipeweave` 或 `--mode all` 时运行 PipeWeave。

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

生成 Kernel 与 full-only Serving 可视化报告时，使用标准报告入口。Kernel 报告按 NVIDIA A800、NVIDIA H200、MetaX C500、Ascend 910B 四硬件渲染，其中 H200 只报告、不计入必需硬件；Serving 默认覆盖四种 canonical 评估硬件，H200 使用 vLLM 0.30.0 的 15 个 canonical case，指定 `--nvidia-h200-vllm-version none` 可排除 H200。`--all` 更新 `experiment_logs/evaluation/standard_report/kernel_report.md` 和 `experiment_logs/evaluation/standard_report/serving_report.md`，`--kernel-only` 或 `--serving-only` 只更新对应报告并复用另一部分最近一次已验证产物。Serving 默认读取 canonical `serving_dataset/sharegpt`，使用 45 个并行 worker，且每个 case 只运行一次完整 `device_plus_launch_plus_scheduler` 模式；显式指定 `--nvidia-h200-dataset-root` 时可评估 H200 candidate，并在报告中标明数据来源，正式 canonical 报告不得使用该覆盖项。

```bash
python scripts/evaluation/run_standard_evaluation_report.py --all
python scripts/evaluation/run_standard_evaluation_report.py --kernel-only
python scripts/evaluation/run_standard_evaluation_report.py --serving-only
```

需要固定并重放真实观测到的 Scheduler step 顺序、batch membership 和 token progress，以区分单步计时误差与动态 batching 反馈误差时，使用 **Scheduling Trajectory Replay**：`evaluation.serving.traces.scheduling_trajectory_replay.analyze_scheduling_trajectory_replay()` 或以下 CLI。输出包含逐 step、分 phase 以及调度轨迹重放下 TTFT、TPOT、E2E、吞吐的预测误差；真实 duration 只用于 evaluation comparison，不进入 Forward。Dense case 不需要 expert routing 输入；MoE case 使用正式 Serving evaluation 相同的 canonical expert-frequency profile 仿真 routing，并在 `predictor_source.expert_routing` 中记录其身份与 provenance，不能将其解释为逐 step routing observation。

```bash
python scripts/evaluation/serving/scheduler/run_scheduling_trajectory_replay.py \
  --case ascend_910b/qwen3-4b/tp1_qps16 \
  --host-launch-profile configs/host_timing/host_launch_profiles/ascend_910b.json
```

ShareGPT raw 的审计和 compact candidate 构建入口是 `scripts/evaluation/serving/build_sharegpt_candidates.py`；具体输入、provenance 和发布流程见 [Serving 投稿实验计划](docs/serving_experiment_plan.md)。输出先进入 `profiler_output/serving/`，不能直接覆盖 canonical dataset。

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

- [Serving 系统设计](docs/serving_system.md)：从请求、调度和部署到算子时间线与系统指标。
- [Kernel 解析性能模型](docs/kernel_performance_model.md)：从算子、launch 和硬件资源到执行时间。
- [Serving helper-launch contract](docs/serving_helper_launch_contract.md)：Serving lowering 缺失的 helper-launch contract 的形状、判据与不能参数化的原因。
- [Serving 投稿实验计划](docs/serving_experiment_plan.md)：固定模型、三硬件实测矩阵、Production Trace、projected 大模型与多 Agent 执行顺序。
- [开发进度](docs/dev_progress.md)：当前能力状态和仍未闭合的工作。
- [Goal 模板](docs/goal_template.md)：独立 worker Goal 的唯一统一模板。

**精度记录**

- [Kernel 精度优化研究指南](docs/kernel_accuracy_optimization.md)：golden 口径、建模抽象、硬件机制、被否定方向与下一步证据缺口。
- [FlashAttention 精度与机制记录](docs/flash_attention_accuracy.md)：FlashAttention 解析模型已验证的物理结论、未闭合问题与已否定的建模方式。
- [Kernel golden 计时与数据结论](docs/kernel_golden_timing.md)：三硬件 OP 计时口径、重采结论、新旧 golden 变化和后续精度评估规则。
- [Serving 延迟精度调查快照](docs/serving_latency_accuracy.md)：2026-08-30 ShareGPT 误差调查的 Git 留档，包含已验证根因、证据边界及建议的后续路线。

**硬件专项**

- [MetaX C500 profiling](docs/metax_c500/profiling.md)：C500 专属采集规则与工具限制。
- [MetaX C500 架构](docs/metax_c500/architecture.md)：C500 架构编程抽象、执行过程与微基准参考值。
- [MetaX C500 数据集重采交接](docs/metax_c500/kernel_dataset_recap_todo.md)：C500 kernel dataset 重采的交接记录。
- [NVIDIA H200 serving capture](docs/nvidia_h200/serving_capture.md)：H200 ShareGPT serving 真值采集的操作参考，以及 vLLM 0.11.0 与 0.30.0 两个 runtime 的采集链差异。

**诊断裁决**

- [Serving 误差定位流程](docs/serving_accuracy_diagnosis.md)：从 paired TTFT/TPOT 到固定 Scheduler 轨迹、step、runtime 与单 Kernel 硬件闭环的标准诊断方法。
- [跨卡 MAPE 分母语义](docs/cross_card_span_semantics.md)：三卡实测 duration 取法不同、跨卡 MAPE 对照不是同一分母的判定与影响范围。
- [VECTOR launch floor 证据等级](docs/vector_launch_floor_evidence_grade.md)：三卡 VECTOR 误差相差 2–3 倍与 launch floor 证据等级不同的归因。
- [GEMM SASS launch-coverage 扩展](docs/gemm_sass_coverage_extension.md)：A800 GEMM SASS-pinned fixed-issue 常量扩展尝试暴露的方法论边界。
- [910B FIA 写侧计价](docs/ascend_910b/fia_write_accounting.md)：910B FusedInferAttentionScore 写侧 DRAM 计价的机制结论与被测量口径。
- [910B graph padding：appended-slot FA](docs/ascend_910b/graph_padding_appended_slots.md)：appended-slot FA 按 captured 数组定价与 FIA single-task 覆盖洞补齐。
- [910B graph padding：VECTOR 物理 extent](docs/ascend_910b/graph_padding_physical_extent.md)：padded VECTOR 按物理 extent 定价的结论与判据。
- [910B VECTOR row/hidden 候选排除](docs/ascend_910b/vector_row_hidden_composition_excluded.md)：910B VECTOR row/hidden 复合耗时候选的排除裁决与证据边界。
- [C500 MoE HBM unsupported 裁决](docs/metax_c500/moe_hbm_unsupported.md)：C500 MoE DRAM 流量欠预测的 unsupported 判定与边界。

- [AGENTS.md](AGENTS.md)：唯一强制仓库规范。
