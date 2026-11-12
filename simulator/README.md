# 原生 Cell 接入与运行

此入口将 `Merge_ML_Version` 的原生 Cell、vLLM Scheduler、算子依赖图、TP/PP/EP 通信与 KV 存储，接到当前分支的动态 Runtime。`python -m simulator` 和标准实验 CLI 使用同一实现。

## 本地依赖

使用 Linux/WSL、Python 3.12、C/C++ 编译器和项目已有的物理仿真环境。第三方源码、模型配置和校准资料只放本地，不随本次代码提交。

- 完整 Mid-Layer 快照：`6fe573479c37a931744a6442e7a4b3503eec02e6`。默认目录为 `third_party/midlayer-6fe5734-canonical`；也可传 `--midlayer-root` 或设置 `ORBITSERVE_MIDLAYER_ROOT`。
- vLLM 0.31.0 官方 wheel 的 Python 文件身份由 `integration/runtime/vllm_0_31_identity.json` 校验，不使用替代 Scheduler。
- `integration/requirements-scheduler-latest.lock` 是 CPU Scheduler 验证环境的冻结清单，不是 GPU 推理服务的完整依赖解算结果。在隔离环境中按原清单安装；不要据此宣称完整 GPU/HTTP 依赖检查通过。

```bash
# 在隔离的 CPU Scheduler 环境安装冻结清单；核心物理依赖按根 README 安装。
python -m pip install --no-deps -r integration/requirements-scheduler-latest.lock \
  --extra-index-url https://download.pytorch.org/whl/cpu
python -m simulator.build_midlayer --midlayer-root /path/to/midlayer

# 外部 FlowSim 必须带共享 Runtime API；构建输出仅保存在 local/。
python -m simulator.build_flowsim \
  --source-root third_party/astermind/third_party/flowsim --output-dir local/native
export PYTHONPATH="$PWD/local/native:$PYTHONPATH"
```

FlowSim 构建器保留源 checkout，必要时只在本地构建副本中加入与原生 `run()` 一致的浮点前进保护，防止极小剩余数据导致 `advance_until()` 停滞；输出记录源文件和二进制 SHA-256。ns-3 使用根 README 的原生依赖安装方式。

## 运行

```bash
python -m orbitserve.experiments.cli simulator/examples/dynamic_tp2_pp2.json \
  --midlayer-root /path/to/midlayer --out results/cells/cross.json
python -m orbitserve.experiments.cli simulator/examples/dynamic_migration_tp2.json \
  --midlayer-root /path/to/midlayer --out results/cells/migration.json
```

标准 CLI 默认保存摘要、完整算子/传输审计和物理数据目录。已有完成的物理输出不会被覆盖。可将场景中的 `physical_network` 替换为 `physical_dataset`，指向已生成的 `*.artifacts/physical`，供其他后端或请求配置复用。显式数据集会校验物理版本、身份和文件哈希，损坏时不会自动重算。`--full` 输出完整 JSON；`--artifact-dir` 指定物理数据及审计目录。

旧的静态例子仍可通过 `python -m simulator --scenario ... --output ...` 运行。`--validate` 只检查配置，不验证执行。该入口的运行结果可用 `python -m simulator.audit REPORT --output AUDIT` 单独审计。

## 物理资源与计算资源的对应

- `physical` 明确配置 owner、GPU、交换机、终端和端口连接；`cells[].placements` 明确配置逻辑 rank 到 GPU 的映射。同星多 GPU 与跨星 TP/PP 都使用真实设备节点，不把 GPU 合并为一个卫星别名。
- GPU 的输入、输出端口分离，不能作为转发节点。交换机/终端的容量门控制所有经过它的流量。地面用户不能作为星间中继。
- 星间、星地设备链路必须与规范物理数据的候选链路一一对应，可用 `physical_link_bindings` 消除歧义；不能复制一条物理链路来倍增容量。
- 外部链路的距离、可见性、建链、方向预算、RF 共享池和故障恢复均来自现有 `NetworkState`；设备端口额定带宽再限制实际服务能力。内部链路使用显式配置的带宽和时延。
- 全部传输与计算回调使用 `_EventEngine` 的事件队列。原生 COMM 预测只作参考，物理网络承担实际通信耗时，避免重复计费。ns-3 的内部容量门不产生额外 PPP 数据包。
- 原生算子 DAG 的时间是整数 ns：实际物理完成后向上取整到下一个 ns 通知算子；`network.flows` 和网络日志保留物理连续时间。不会提前释放依赖。

`orbit`、`link_parameters` 已从 SimulationConfig 删除，输入这些字段会被拒绝；设备 `link_policy` 同样不接受。轨道与链路预算统一使用规范 `physical_network`/`physical_dataset`。新例子的 SGP4 平均根数明确标为合成输入，链路容量为显式固定预算加精确时间降速，用于功能正确性检验。

## 显存与迁移

原生 vLLM 决定 admission、分批、KV 分配和前缀复用。默认 `num_kv_blocks` 是显式测试池，结果标为 `debug_explicit_block_count`，不能解释为真实 HBM 上限。提供与硬件哈希、模型哈希、并行度和运行参数严格匹配的 `memory_policy_file` 后，按每个 rank 的权重驻留、非 KV 预留和最紧约束计算容量。

该 Mid-Layer 快照附带的 A800/Qwen3-4B 显存策略与当前硬件文件指纹不同，会被拒绝。不能只改指纹继续套用旧测量。本次容量测试另建了明确标注的合成策略，验证扣减与分配逻辑；不代表完成硬件校准。

同布局 KV copy/move 逐 rank 读取和发送，等待全部分片及目标写入完成后提交；失败则回滚目标保留并释放源保护。move 只在提交后删除源缓存。不同 TP/PP 布局的重分片、PD/AF、运行中计算节点恢复及原生算子能耗尚不支持；能耗输出为 `not_evaluated`。同一 Cell 使用同一种 GPU 模型，可在不同 Cell 配置不同硬件。

## 验证

```bash
# 必须使用真实依赖；缺少依赖直接失败，不跳过。
python -m unittest discover -s tests/external/native_cells -v
# 项目原有四个测试套件，包含 slow。
python -m unittest discover -s tests -t .
```

原生专项覆盖：跨星 TP2/PP2 故障等待与恢复、同星 TP2 加跨星双分片迁移（FlowSim/ns-3）、迁移失败的原生分配器回滚、A800/C500 异构 Cell 与显存容量账目、MoE TP2/EP2。审计核查算子耗时、依赖、通信阶段、输入分发、字节量与提交顺序。这些结果证明接入语义，不等于硬件性能预测已通过实测校准。

## 调度与算子前端回放

从完整的 `native_result.json` 或 `python -m simulator` 报告导出，无需再次运行模型：

```bash
python -m simulator.replay REPORT1.json REPORT2.json --out local/results/native-cells-viewer
```

本机工作台入口为 `http://127.0.0.1:8766/native-cells/`，主页有“原生调度与算子回放”链接。导出目录也可由已有静态 Web 服务托管；页面不加载外部 CDN，不上传报告。

视图包括 GPU/交换机/终端连接、按时间的实际逐跳服务、按 rank 的算子条带、原生批次 token 与 KV 事件、局部依赖图、COMM 分阶段传输以及 KV 提交/回滚。COMM 在所有参与 GPU 的条带上显示，仍只计为一个原生操作。选择算子、依赖、批次或迁移分片可追踪相关通信；滚轮缩放时间线，时间标记定位降速、故障、恢复和首 token。

核查面板从实际记录重新检查依赖时序、原生耗时、通信字节与屏障、输入分发、物理完成到算子通知的 ns 取整，以及迁移终态；违规会显示 JSON 证据位置。报告 SHA-256 标明所读版本。成功回滚会显示“传输失败、回滚一致”，不会伪装成迁移交付成功。显存数字是明确标记的最终原生快照，不是滑块时刻的实时显存。设备连接示意不代替物理轨道数据，也不证明 GPU 预测误差或算子能耗已校准。

```bash
python -m unittest tests.unit.runtime.test_native_replay
```

此套件刻意破坏依赖、原生耗时、字节量、容量、交付通知时刻及迁移保留状态，确认核查不会始终显示通过。
