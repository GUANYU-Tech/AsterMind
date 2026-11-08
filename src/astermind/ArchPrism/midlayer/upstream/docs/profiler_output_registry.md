# 数据登记：profiler_output 与 experiment_logs

AGENTS.md「Contract 与数据」节是规则本体，本文是其配套登记与操作手册。

## profiler_output 目录登记

Profiler 原始产物唯一存放处。**新建任何 campaign 目录时必须在此添加一行**，写清用途、对应 canonical 版本和保留级别。profiler_output 内禁止存放文档与代码副本（微基准自带的探针源码除外）。

保留级别：

- `pinned-raw`：被 canonical manifest（SHA256 / source snapshot 绑定）或 committed 配置钉住的 raw 源文件。**只压不删**；派生文件删除前必须确认可从源文件再生。
- `regenerable`：运行脚本即可再生的中间产物或工作区，可随时清空。
- `closed`：对应诊断/任务已闭环，无 committed 引用方；删除或压缩前需全仓 grep 确认零引用。

| 目录 | 用途 | 喂给哪个 canonical / 配置 | 保留级别 |
|---|---|---|---|
| `serving/` | 各硬件 serving 采集 raw | `serving_dataset/`、evaluation alignment | pinned-raw；子目录登记见下表 |
| `kernel_dataset/` | kernel 采集 raw：C500 `<family>_traces/`（987 个 mctracer json）、A800 golden `profile_runs/`（ncu-rep）、910B `msprof_*.db.zst`、`fa_serving_coverage_20260915/` | `kernel_dataset/`、`launch_data/` | pinned-raw |
| `microbench/` | 独立机制微基准原始输出 | HWSpec 机制参数 | pinned-raw |
| `microbench/nvidia_a800/operator_support_typed_20261023/` | 用户授权空卡上的寄存器字节重排/比较/选择独立机制及非耗时launch事实；先smoke、单卡单owner | 当前operator-support迁移缺口的candidate物理证据；不自动发布canonical或绑定Forward | pinned-raw |
| `microbench/metax_c500/operator_support_typed_20261023/` | 用户授权约定容器内空卡上的typed独立机制、Copy环境/离散launch事实；不消费production duration | C500原Serving helper缺口的candidate物理证据；不自动发布canonical或绑定Forward | pinned-raw |
| `microbench/ascend_910b/operator_support_typed_20261023/` | 用户授权约定容器内空卡上的typed向量独立机制及转换/索引非耗时launch事实 | 910B原Serving helper缺口的candidate物理证据；不自动发布canonical或绑定Forward | pinned-raw |
| `microbench/ascend_910b/fia_host_tiling_cpu_20261026/` | 安装CANN8.5.1的真实CPU host tiler对合成TND描述符的离散分块查询；不初始化设备、不分配设备内存、不执行kernel、不记录duration | 910B FULL FIA多行dummy的candidate选型事实；不自动发布canonical或绑定Forward | regenerable |
| `microbench/metax_c500/embedding_launch_static_20261023/` | 约定 profiler 容器中既有 Torch 二进制的只读 host 分派/发射反汇编；不执行 GPU、不消费 duration | C500 embedding/index_select/cat、整数物化、条件BF16→FP32转换及单行高级索引的同版本静态证明；不发布 canonical | pinned-raw |
| `microbench/metax_c500/argmax_launch_static_20261023/` | 同一固定 Torch2.4/MACA3.7.2 库的 ArgMax host 分派与 reduction 源码只读记录；不执行/查询 GPU、不消费 duration | 原C500 greedy采样的launch/typed服务边界分析；不发布 canonical，不当完整采样计价或provider通过 | pinned-raw |
| `microbench/nvidia_a800/argmax_launch_static_20261023/` | 固定Torch2.8库的精确SM80 Float/Long ArgMax entry与资源、host/source分派和值依赖分支的离线静态记录；不执行/查询GPU、不消费duration | 原A800完整Serving采样首拒绝的source/适用域证据；不发布canonical、不把上界当点预测 | pinned-raw |
| `microbench/nvidia_h200/embedding_launch_static_20261023/` | 既有固定 Torch2.8 CUDA 库的 SM90 Copy/embedding/argmax 编译产物静态读取；不执行/查询 GPU、不消费 duration | 原H200 embedding与采样链的架构对应证明；不发布canonical，不当actual provider验收 | pinned-raw |
| `microbench/nvidia_a800/embedding_mask_static_20261023/` | 既有 Inductor FX cache 的原始压缩条目、仅用 pickletools/AST 的源码身份核对及无耗时离散配置/SM80反汇编记录；不加载缓存或执行 GPU | TP embedding mask 编译来源与重叠 guard 的静态边界；不发布 canonical，不当原Serving launch或定价通过 | pinned-raw |
| `microbench/metax_c500/memory_unification_mla_ir_20261017.tar.zst` | 固定MACA FlashMLA设备bitcode、IR与source manifest；保留原campaign内相对路径，无运行耗时定价 | memory_metax_mla_controls及MLA源码描述；不替换canonical数据 | pinned-raw；只压不删 |
| `microbench/nvidia_a800/memory_unification_grouped_mla_20261016.tar.zst` | grouped MLA静态源码和编译接口事实；保留原campaign内相对路径，无运行耗时定价 | MLA输入、分组及共享存储描述；不替换canonical数据 | pinned-raw；只压不删 |
| `microbench/nvidia_a800/memory_unification_alignment_source_20261018.tar.zst` | MoE alignment静态源码与manifest；保留原campaign内相对路径，无运行耗时定价 | MoE alignment逻辑依赖描述；不替换canonical数据 | pinned-raw；只压不删 |
| `microbench/ascend_910b/memory_unification_cann_tnd_20261016.tar.zst` | CANN TND静态源码与IR记录；保留原campaign内相对路径，无运行耗时定价 | CANN attention逻辑数据流描述；不替换canonical数据 | pinned-raw；只压不删 |
| `microbench/ascend_910b/memory_unification_cann_decode_20261016.tar.zst` | CANN decode静态源码与接口记录；保留原campaign内相对路径，无运行耗时定价 | CANN decode逻辑数据流描述；不替换canonical数据 | pinned-raw；只压不删 |
| `microbench/ascend_910b/memory_unification_rotary_source_20261018.tar.zst` | CANN rotary静态源码及manifest；保留原campaign内相对路径，无运行耗时定价 | CANN rotary输入与变换描述；不替换canonical数据 | pinned-raw；只压不删 |
| `microbench/ascend_910b/memory_unification_cann_routing_20261016.tar.zst` | CANN routing静态源码及manifest；保留原campaign内相对路径，无运行耗时定价 | CANN routing逻辑依赖描述；不替换canonical数据 | pinned-raw；只压不删 |
| `collective/` | 集合通信采集 + 选择报告 | `scripts/collective_selection/` | pinned-raw |
| `scheduler_host/` | host scheduler 服务时延采集 | `configs/host_timing/` | pinned-raw |
| `runtime_memory_policy/` | 运行时显存策略采集（910B + A800） | `configs/runtime_memory_policy/` | pinned-raw |
| `runtime_memory_policy/ascend_910b/operator_support_scope_20261024/` | 约定910B容器中既有vLLM0.18模型局部作用域的只读AST原始输出；不import Torch、不调用NPU、不写远端 | 统一虚拟图startup容量的candidate源码身份/生命周期证据；不发布canonical、不以数值误差选择分支 | pinned-raw |
| `serving/nvidia_h200/simulator_cpu_full_profile_20261026/` | H200 dense TP2 QPS1配置的CPU仿真器16步准备后32 FULL步cProfile归因；未调用GPU，不发布或供给硬件真值/Forward速率 | 不对应canonical发布；派生报告位于operator_support analytical_performance_closure campaign | regenerable |
| `serving/nvidia_a800/startup_native_memory_20261026/` | 本地空闲A800上Qwen3-8B BF16/TP1 vLLM0.11正常启动的原生显存计数候选；0推理请求，不消费duration定价 | 原60项block16配置的normal容量候选验证；不自动发布canonical | pinned-raw |
| `serving/nvidia_a800/serving_reuse_cpu_20261027/` | A800 Qwen3-4B真实decode计划的CPU预测复用计数/cProfile；不调用GPU、不供给Forward参数 | 无canonical发布；Serving复用修复诊断 | regenerable；闭环压缩或清理 |
| `serving/nvidia_a800/serving_group_reuse_cpu_20261027/` | A800真实decode计划冷预测数值复用CPU profile；不运行GPU或变更公式 | 无canonical发布；第二轮Serving复用诊断 | regenerable；闭环压缩或清理 |
| `serving_dataset_compaction/` | 发布链硬编码工作区（不得改名） | — | regenerable |

### `serving/` 子目录登记

ShareGPT raw 统一布局：`serving/<hardware_id>/sharegpt_raw/<ShareGPT-模型_并行_n128_qpsX>/`（loader 的 canonical 解析路径，四硬件已统一）。

| 目录 | 内容 | 保留级别 |
|---|---|---|
| `<hardware_id>/sharegpt_raw/` | canonical ShareGPT 15 case 采集 raw | pinned-raw（只压不删） |
| `nvidia_h200/sharegpt_raw_vllm011/` | H200 vLLM 0.11.0 同布局采集（非 canonical；canonical 为 0.30.0） | pinned-raw |
| `nvidia_h200/sharegpt_raw_smoke/` | H200 采集链 smoke 的 observer-effect A/B 证据（ground_truth 与日志） | closed |
| `ascend_910b/startup_native_memory_20261026/` | 约定容器实际安装版本的源码绑定与单卡 Qwen3-8B 正常启动 allocated/reserved/free 原始计数；8192 tokens / 256 sequences，无推理请求，不用于性能定价 | pinned-raw；normal 容量候选证据，不自动发布 canonical |
| `nvidia_h200/sglang_dsr1*/`、`nvidia_h200/host_profiles/` | SGLang DeepSeek-R1 TP8 采集与 host profile | pinned-raw |
| `metax_c500/c500_async_v2_38e24a9e_*`、`metax_c500/c500_async_v4_818e4533_*` | C500 async scheduling 战役现场，被 `serving_dataset/sharegpt/metax_c500` manifest 引用 | pinned-raw |
| `metax_c500/c500_batch_scaling_20260922/` | C500 batch scaling 诊断证据（`docs/serving_accuracy_diagnosis.md` 引用） | pinned-raw |
| `helper_launch_20260923/` | 三硬件 helper/glue 发射计数谓词证据（`configs/validation/serving/helper_emission_observation.yaml` 引用） | pinned-raw |
| `inferencex_alignment/raw/` | InferenceX DeepSeek-R1 外部实测原始点（论文脚本引用） | pinned-raw |
| `ascend_910b/ascend_910b_serving_full_depth2_20260901/`、`ascend_910b/host-timeline-910b-*`、`ascend_910b_pd_disaggregated_1p1d_depth2_20260903/` | 910B serving depth2 / host timeline / mooncake PD 采集（910B 远程已清，字节唯一） | closed（删前需用户确认） |

| `llmservingsim-20261016/` | LLMServingSim 独立 LUT baseline 的原生/vendor layerwise GPU/NPU profile、执行登记和 bundle 证据；不发布 Midlayer canonical、不进入 Forward | pinned-raw（本任务 baseline bundle 证据） |
| `metax_c500/startup_native_memory_20261026/` | 约定 `yxh_vllm_serving_c500` 容器和 C500 设备占用、静态软件版本及 canonical Qwen3-8B 已知模型路径的只读 preflight；供0.24零请求启动采bytes的准备，不执行GPU或发布canonical | pinned-raw；关联 `serving_dataset/sharegpt/metax_c500/qwen3-8b/tp1_qps16` 和0.24 source预算候选，native10%/normal未验收 |

## experiment_logs 组织约定

只存**代码运行的产出结果**，全部必须可重复生成、随时可删重跑，不允许例外；不可再生的证据或外部输入一律放 `profiler_output/` 并登记。

- `evaluation/`：报告工作区。`standard_report/` 只存报告本体 + `state.json` / `hashes.json`；中间产物按侧分放 `kernel_runs/<stamp>/`、`serving_runs/<run_id>/`；`kernel/` 与 `logs/` 是 full evaluation runner 的输出位；`diagnostics/` 为诊断脚本输出。
- `kernel_golden/`、`synthetic_publication/`：kernel 数据发布/验收 runner 的写死输出位，重跑即再生。
- `inferencex_alignment/`：InferenceX 对比图输出位（原始输入在 `profiler_output/serving/inferencex_alignment/raw/`）。
- `inferencex_style_dsv3_8k1024_closedloop_tp8_ep1/`：DeepSeek-V3 闭环仿真 run，重跑脚本即再生。

## 压缩与还原

- `.zst` 一律 `zstd -d --long=27 <路径>.zst` 还原。已压缩项：A800 `plugin_trace.json`、910B `msprof_*.db`（重解析前先还原，sha 与 manifest 一致）、C500 `nsys_kernels_full.json`、C500 sharegpt_raw 45 个大 json（`mctracer-443.json`、`plugin_trace*.jsonl.rank<r>`、`.mctracer_sync_ends_*.json`）。全部经 sha256 往返校验。
- NVIDIA `nsys_profile.sqlite` 已全删（派生物），需要时在有 Nsight Systems 的机器上重导：`nsys export --type sqlite --force-overwrite true -o nsys_profile.sqlite nsys_profile.nsys-rep`。重导字节与原文件不同，manifest 的 sqlite pin 无法再校验，内容等价。
- 910B `mindstudio_profiler_output/msprof_*.json` 中间导出物已删，可在 910B 上用 msprof 从 db 重导。
- kernel A800 golden 的 3225 个 `*.sqlite` 已删（同法重导）；`.ncu-rep` 完整保留。
- `fa_serving_coverage_20260915/` 的窗口级现场已删，保留 `capture_manifest.json`、124 个 timing/counter json 与 9 个 canonical sha 绑定的 `source_snapshot.tar`，重发布能力完整。

## 既有缺口（pin 校验对不上时先查这里）

- 全部硬件 × 15 case 的 `production-ground-truth`（`ground_truth.json`）：pin 仅指纹，字节从未存在。
- `ascend_910b` 多卡 6 case pin 的 12 个 msprof db 及双 rank operators trace：字节已随 910B 远程清理丢失，canonical 数据集完整，仅失去从 raw 重推导能力，补救需重新上卡。
- C500 57 条 canonical FA 条目绑定的 tar（sha `54967e3c…`）：字节不存在于任何机器，pin 仅指纹。
- 各硬件 `nsys_profile.sqlite`：见上文压缩节，派生物已删。
- `configs/runtime_memory_policy/metax_c500/*.json` 的 `source_reference` 指向已删除的 `experiment_logs/serving/production_ground_truth/20260824_c500_v0_graph_matrix_serial_7b5a1fd/`，数值失去字节级溯源，补救需在 muxi 重采。
