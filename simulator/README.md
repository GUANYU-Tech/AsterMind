# 计算与通信仿真框架：Mid-Layer 接入

运行入口为 `python -m simulator`，计算只使用 Mid-Layer。先前的 FlexET 路径、静态图导出、复制源码、测试和示例已移除。上游 M-L 的模型、并行规则、kernel 与本地访存模型不在本目录另写一套。

## 版本与分工

当前默认使用 `src/astermind/ArchPrism/midlayer/upstream/` 中的本地 Mid-Layer 开发快照。源码、配置与已展开的校准数据直接随仓库提供。三个执行部分和快照来源见 [AsterMind](../src/astermind/README.md)。`--midlayer-root` 可在新进程中选择接口兼容的快照。

旧 `midlayer_hw_model-main` 可用同一入口运行。已验证 GLM-5.2 共置、A800 32 卡、TP8 × PP4／EP8。计算沿用旧 main 的模型、算子和 CPU Scheduler，星间通信仍由 OrbitServe 运行。旧目录使用其 README 的 LRU 构建和 vLLM 0.11 CPU Scheduler 环境。

```bash
python -m simulator --midlayer-root /path/to/midlayer_hw_model-main --scenario simulator/examples/glm52_single_cell_32ranks.json --gc-gen2-threshold 400 --output simulator/runs/glm52_legacy.json
```

`--gc-gen2-threshold 400` 调整本次 CLI 的完整垃圾回收频率，自动回收继续开启。该值在本机 CPython 3.12 的六请求场景中验证。

模型、合法并行组合和 kernel 正确性由 M-L 开发线持续建设；本框架负责卫星部署、请求与多个 Cell、计算通信协同、轨道与物理网络、KV 驻留和迁移。目标是随上游能力更新完成接入。上游有实现与本入口已接通须分别记录，不能把上游全部能力直接写成本入口已支持。

大模型由上游固定 vLLM 0.11 CPU Scheduler 排批，按模型选择原有的 0.27.1 或 0.30 虚拟前向。既有 BF16 GQA Cell 保留 0.31 CPU Scheduler 和 session/KV 适配。运行报告分别记录 Scheduler 和实际计算配置。

## 框架与职责

| 层次 | 当前实现 | 职责 |
|---|---|---|
| 卫星、地面和设备资源 | [物理网络](../src/network_sim/physical/) | 配置卫星、GPU、交换机、终端和候选连接。`resources.py` 校验逻辑 rank 和请求端点是否符合物理资源 |
| 轨道与链路条件 | [environment.py](../src/network_sim/physical/environment.py) | 复用 O-S 的 SGP4/ITRS。由可见性和链路参数得到可用窗口，由建链策略管理终端捕获与释放 |
| 请求与多个 Cell | `engine.py` | 创建共享时钟、Comm 和网络事件驱动，为各 Cell 建立独立执行器，安排请求输入和 token 输出 |
| Cell、Scheduler、KV | `src/astermind/cell.py`、`src/astermind/ArchPrism/scheduler/` | 原生 vLLM 决定 batch、阶段、KV block，Cell 保存进度，OrbitServe 协调迁移 |
| 计算图与 kernel | M-L 的构图、lowering、预测和本地访存模块 | 提供各 rank 的任务、张量形状、通信和依赖，预测本地 kernel 耗时 |
| 执行映射与协调 | `src/astermind/ArchPrism/execution/` | 每个 Cell 的 rank 映射、批次依赖与算子推进，通过共享驱动提交跨 GPU 传输 |
| 传输与迁移 | [Comm](../src/network_sim/comm/)、`migration.py` | 按物理链路状态选路、计算竞争和回报终态。迁移按终态提交或回滚 KV |

计算侧提供计算和通信的依赖、点对点端点或集合通信参与组、数据量及其含义。执行映射层按选定算法把集合通信展开为逐阶段传输，将逻辑 rank 映射到 GPU 和终端，再提交给通信后端。点对点或已经展开的传输直接复用。当前接通 Ring AllReduce/AllGather、PP 点对点和部分显式变长通信，并非所有集合算法均已接通。

通信后端根据实际连接、链路状态和资源竞争返回完成/失败事件及时间。外部协调层收到事件后释放后续计算和通信；整轮结束再反馈 Scheduler。M-L 原先的跨 GPU 通信预测不重复计费，GPU 本地 kernel、HBM、L2 等仍由 M-L 计时。

Cell 选择目前为初始策略：优先已分配但未计算结束的请求较少者，再按孤立输入传输估计、Cell ID 排序；同 session 默认留在原 Cell。原 O-S 全部研究策略尚未逐项接入。完整 batch 结束后再调度下一轮，每轮保留每条请求的阶段、query 长度和已有 KV 长度。

## 多 GPU、PD/AF 与 KV

普通 Cell 的卡数为 `TP × PP`。`parallel.ep` 取 1 或 TP，EP 复用阶段内的 TP ranks。旧配置中的 `native_expert_parallel` 仍可开启同一功能。大模型实验配置使用固定上游切片。

| 模型 | 硬件与卡数 | 并行 | 部署 |
|---|---|---|---|
| Kimi-K2.6 | 16 × A800 | TP8 / PP2 / EP8 | 共置 |
| Kimi-K2.6 | 8 × H200 | TP8 / PP1 / EP8 | 共置 |
| DeepSeek-V3 | 8 × H200 | TP8 / PP1 / EP1 | 共置 |
| DeepSeek-V4 Flash | 4 × H200 | TP4 / PP1 / EP1 | 共置 |
| DeepSeek-V4 Pro | 8 × H200 | TP8 / PP1 / EP1 | 共置 |
| Kimi-K2.6 | 8 + 8 × H200 | 每池 TP8 / PP1 / EP8 | P/D 试验切片 |

大模型示例使用短合成请求、显式 KV block 预算和有固定种子的专家路由假设。设备内核时间是上游 Roofline 预测。Kimi-K3 与 DeepSeek-V3 L32 仍缺少该快照下可绑定的完整 Serving 前向。

`deployment: pd_disagg` 启用 Kimi H200 双池切片。P 池用 rank 0–7，D 池用 rank 8–15。首 token 从 P 池返回，D 池在逐 rank KV 传输全部完成后接收请求。P 池在交接时暂停新批次。当前切片使用上游的 KV 元数据交接与外部网络耗时，独立于下面的 session KV 迁移路径。

session KV 复用基于声明的同 session 线性历史和完整块，不模拟真实张量、任意文本匹配或跨 session 分支。容量以原生 vLLM block 池为准；可使用原生 `memory_policy_file` 推导预算。不配置时，显式 block 数仅为测试容量，不能视为真实 GPU 的 HBM 可容纳性结论。

迁移支持兼容单 GPU Cell，以及相同模型、KV 格式、TP/PP 分片和逐 rank KV 几何的多 GPU Cell。它在 batch 边界保护源 block、预留目标 block，逐 rank 读写和传输，等所有分片到达并完成本地写入后提交。任一分片失败时，等所有在途分片终止，再回滚预留并释放源保护。现有接口提供 copy/move。

## 星地接入、轨道和异构设备

地面位置通过 `physical.satellites` 中地面 owner 的 `ecef_position_km` 输入，并在 `ground_owners` 中标记为地面；不会当作卫星传播。当前直接输入 ECEF/ITRS 三维位置，不直接解析经纬度。

每根天线或独立接入通道表示为一个 `kind: terminal` 设备。每个终端有独立连接、可选总带宽和 `max_external_links`，通常一根天线取 1；需要多波束时显式填写更大值。多个地面终端可共同属于同一地面站。地面位置、卫星位置及 `orbit.min_ground_elevation_deg` 决定星地可用窗口；这是地面站看卫星的最低仰角，不是轨道倾角。目前该仰角门限是场景级配置。候选星地连接需显式提供，并非自动让所有可见卫星都连接。

纯计算星、纯通信星和混合星，以及单网关、多网关和分布式出口可配置。物理连接支持四邻接 Mesh、FC、Ring、Switch 和自由边；不提供 8-neighbor。终端和交换机可以中继，GPU 不默认充当路由器。

同一物理网络可连接使用不同卡型的多个 Cell，每个 Cell 通过 `hardware`/`hardware_id` 选择 M-L 硬件模型。一个多 GPU Cell 内混用不同卡型尚未接通；当前每个 Cell 的一个计算 pool 使用同一种硬件，不能靠修改物理设备 ID 冒充异构计算。

轨道支持逐成员传播，或参考星位置加固定 ITRS 偏移；固定偏移误差尚未验证。轨道可见性按采样产生窗口，不是连续几何求根。FlowSim 消费窗口、终端名额和建链状态；`simple_keep`、O-S 参考策略和自定义选择接口已存在。自动生成的轨道配置仍使用零建链耗时；60s/2s 可显式配置，但当前按终端使用历史计费，按卫星对记录首次/重建的规则仍待补。

星间链路参数可手动配置或通过 ISLSim 生成。动态 ISLSim 预算目前取可行带宽下界、延迟上界，不连续更新在途速率。ns-3 使用真实 PointToPoint/PPP/DropTail 包级传输，目前仅支持静态连接；没有 TCP、NCCL 协议或重传模型。FlowSim 和 ns-3 不能都宣称已支持动态星座建链。

## 运行和核查

大模型示例使用 vLLM 0.11.0 的 Linux x86-64 CPU 隔离环境。既有 BF16 GQA、前缀与迁移示例的 vLLM 0.31 环境见 `src/astermind/ArchPrism/scheduler/runtime/requirements-vllm-0.31.lock`。两种环境分别运行。更新后从项目根目录重新构建本地扩展。

```bash
python3.12 -m venv .venv-astermind
. .venv-astermind/bin/activate
python -m pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cpu -r src/astermind/ArchPrism/scheduler/runtime/requirements-vllm-0.11-cpu.lock
```

```bash
python -m simulator.build_midlayer
python -m network_sim.build_native --upstream-tests
python -m simulator.examples.astermind_large_models simulator/runs/large-models.json
python -m simulator --scenario simulator/runs/large-models.json --output simulator/runs/large-models.result.json
python -m simulator.audit simulator/runs/large-models.result.json --output simulator/runs/large-models.audit.json
python -m simulator.examples.render_astermind_timeline simulator/runs/large-models.result.json simulator/runs/large-models.png
python -m simulator.examples.astermind_large_models simulator/runs/kimi-pd.json kimi-k2.6-pd
python -m simulator --scenario simulator/runs/kimi-pd.json --output simulator/runs/kimi-pd.result.json
python -m simulator.audit simulator/runs/kimi-pd.result.json --output simulator/runs/kimi-pd.audit.json
```

Serving 运行使用现有 Linux/WSL CPU 隔离环境，不运行真实 GPU：

```powershell
wsl.exe -d Ubuntu-24.04 --cd '<仓库在 WSL 中的路径>' <已配置的 Python 路径> -m simulator --scenario simulator/examples/sequential_cells.json --output simulator/runs/sequential_cells.json
```

Linux 项目根目录中的其他入口：

```bash
python -m simulator --scenario simulator/examples/tp2_two_satellites.json --output simulator/runs/tp2_two_satellites.json
python -m simulator --scenario simulator/examples/session_migration_tp2.json --output simulator/runs/session_migration_tp2.json
python -m simulator --scenario simulator/examples/heterogeneous_cells.json --output simulator/runs/heterogeneous_cells.json
python -m simulator.checks
```

`--validate` 只检查资源、配置和拓扑，不证明模型或硬件 kernel 能执行。`audit.py` 核查实际执行的依赖与计时；当前验收见 `runs/acceptance.json`，测试见 `runs/unit_tests.json`，源文件身份由 `provenance.py` 记录。所有示例均采用合成通信参数，不代表真实卫星部署测量。早期核查文件描述的是当时版本，不替代此处现状。

2026-10-17 本轮 14 项测试通过，三个新增端到端样例通过依赖与计时审计：四个 Cell 在 2.010/5.010/8.010/11.090 ms 开始计算，12 个请求全部完成，峰值 4 个 Cell 同时运行，出现 8 个 P/D 混合 step，共享 1 Gbps 地面终端最多有 4 条流竞争；TP2→TP2 迁移按两个 rank 各传 2,359,296 字节，全部到齐后提交，目标实际命中 32 token 前缀；A800 与 C500 两个独立 Cell 共存运行。以上为合成配置的功能验证，不是硬件精度实测。汇总见 `runs/20261017_summary.json`；`runs/midlayer_boundary.json` 再次核对 M-L 的 147 个源码、111 个配置文件均未改动。

2026-11-07 的接入验证见 `runs/astermind_20261107/`。共置双 Cell 样例的 4 个请求和 128,312 条依赖均通过核查。Kimi P/D 的审计核对逐 rank KV 分片的完成时刻及随后 Decode 的开始时刻。
