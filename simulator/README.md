# 计算与通信仿真框架：Mid-Layer 接入

运行入口为 `python -m simulator`，计算只使用 Mid-Layer。先前的 FlexET 路径、静态图导出、复制源码、测试和示例已移除。上游 M-L 的模型、并行规则、kernel 与本地访存模型不在本目录另写一套。

## 版本与分工

当前默认固定为 M-L 2026-10-10 提交 `6fe573479c37a931744a6442e7a4b3503eec02e6`，完整源码、配置和已展开的校准数据直接保存在本仓库，不使用外部子模块。这是当前接入版本，后续会接入上游更新版本；可通过 `--midlayer-root` 在新进程选择快照，更新后需复查计算图、通信描述、容量配置和事件接口。10 月 10 日创建的旧归档封装的是更早提交 `98d207a`，不能与本提交混同。

模型、合法并行组合和 kernel 正确性由 M-L 开发线持续建设；本框架负责卫星部署、请求与多个 Cell、计算通信协同、轨道与物理网络、KV 驻留和迁移。目标是随上游能力更新完成接入。上游有实现与本入口已接通须分别记录，不能把上游全部能力直接写成本入口已支持。

M-L `src/` 和 `configs/` 保持原始内容。外部适配包括 vLLM 0.31 CPU Scheduler、session/KV 状态、逐事件执行，以及 A800 单 query Attention 的已有 launch 选择适配。它们不是完整复现上游 vLLM 0.11 环境；源码未改不单独构成端到端硬件精度证明。

## 框架与职责

| 层次 | 当前实现 | 职责 |
|---|---|---|
| 卫星、地面和设备资源 | `config.py`、`resources.py` | 独立配置卫星、GPU、交换机、终端和候选连接；rank 部署绑定这些实际设备 |
| 轨道与链路条件 | `environment.py` | 复用 O-S 的 SGP4/ITRS；由可见性和链路参数得到网络可用条件 |
| 请求与多个 Cell | `engine.py` | 选择 Cell，安排输入和逐 token 输出传输；共用事件时钟 |
| Cell、Scheduler、KV | 原接入模块及 `native_cell.py` | 原生 vLLM 决定 batch、阶段、KV block；外层维护声明的 session 关系与迁移状态 |
| 计算图与 kernel | M-L 的构图、lowering、预测和本地访存模块 | 提供各 rank 的任务、张量形状、通信和依赖，预测本地 kernel 耗时 |
| 执行映射与协调 | 原 `NetworkBatchExecutor`、`transport.py` | rank 到 GPU/端点映射、集合通信展开、就绪判断和完成事件推进 |
| 传输与迁移 | FlowSim/ns-3、`migration.py` | 实际拓扑、共享资源竞争、传输完成/失败，以及 KV 提交/回滚 |

计算侧提供计算和通信的依赖、点对点端点或集合通信参与组、数据量及其含义。执行映射层按选定算法把集合通信展开为逐阶段传输，将逻辑 rank 映射到 GPU 和终端，再提交给通信后端。点对点或已经展开的传输直接复用。当前接通 Ring AllReduce/AllGather、PP 点对点和部分显式变长通信，并非所有集合算法均已接通。

通信后端根据实际连接、链路状态和资源竞争返回完成/失败事件及时间。外部协调层收到事件后释放后续计算和通信；整轮结束再反馈 Scheduler。M-L 原先的跨 GPU 通信预测不重复计费，GPU 本地 kernel、HBM、L2 等仍由 M-L 计时。

Cell 选择目前为初始策略：优先已分配但未计算结束的请求较少者，再按孤立输入传输估计、Cell ID 排序；同 session 默认留在原 Cell。原 O-S 全部研究策略尚未逐项接入。完整 batch 结束后再调度下一轮，每轮保留每条请求的阶段、query 长度和已有 KV 长度。

## 多 GPU、PD/AF 与 KV

多 GPU 计算不是单卡限定。当前原生 Cell 入口覆盖 TP1/PP1、TP2/PP1、TP2/PP2，以及指定 Qwen3-MoE 的 TP2/EP2；多 GPU 路径限定 M-L 已接通的 A800 BF16 Qwen3 范围。`native_expert_parallel` 复用 TP ranks，不增加独立 EP 卡数轴。本入口没有额外发明并行切分；DP/CP 和更多模型/卡型随上游版本逐项接入。

需要区分三个功能：多 GPU 计算、session 历史 KV 迁移、PD/AF 分离部署。上游已有受限 PD/AF/MTP 路径，原接入目录也有专门 PD 入口，但当前 `python -m simulator` 仍为 colocated Cell，不能把 session KV 迁移写成完整 PD/AF 已接通。P/D 的资源池、逐 rank KV 和 A/F activation 传输都应按上游语义接入，尚未接通部分明确列为待补。

session KV 复用基于声明的同 session 线性历史和完整块，不模拟真实张量、任意文本匹配或跨 session 分支。容量以原生 vLLM block 池为准；可使用原生 `memory_policy_file` 推导预算。不配置时，显式 block 数仅为测试容量，不能视为真实 GPU 的 HBM 可容纳性结论。

迁移支持兼容单 GPU Cell，以及相同模型、KV 格式、TP/PP 分片和逐 rank KV 几何的多 GPU Cell：在 batch 边界暂停、保护源 block、预留目标 block，逐 rank 读取和发送各自 KV 分片，全部传输完成并完成目标本地写入后才提交。多 GPU 的物理传输总量为各 rank 分片之和，不把整份 KV 仅发到 rank 0。任一分片失败时，等所有在途分片终止，再回滚预留并释放源保护。支持 copy/move；不同 TP/PP 分片之间的重排、自动重试/续传仍未接通。`native_cell.py` 只描述兼容布局，不修改 M-L 构图和 kernel。

## 星地接入、轨道和异构设备

地面位置通过 `physical.satellites` 中地面 owner 的 `ecef_position_km` 输入，并在 `ground_owners` 中标记为地面；不会当作卫星传播。当前直接输入 ECEF/ITRS 三维位置，不直接解析经纬度。

每根天线或独立接入通道表示为一个 `kind: terminal` 设备。每个终端有独立连接、可选总带宽和 `max_external_links`，通常一根天线取 1；需要多波束时显式填写更大值。多个地面终端可共同属于同一地面站。地面位置、卫星位置及 `orbit.min_ground_elevation_deg` 决定星地可用窗口；这是地面站看卫星的最低仰角，不是轨道倾角。目前该仰角门限是场景级配置。候选星地连接需显式提供，并非自动让所有可见卫星都连接。

纯计算星、纯通信星和混合星，以及单网关、多网关和分布式出口可配置。物理连接支持四邻接 Mesh、FC、Ring、Switch 和自由边；不提供 8-neighbor。终端和交换机可以中继，GPU 不默认充当路由器。

同一物理网络可连接使用不同卡型的多个 Cell，每个 Cell 通过 `hardware`/`hardware_id` 选择 M-L 硬件模型。一个多 GPU Cell 内混用不同卡型尚未接通；当前每个 Cell 的一个计算 pool 使用同一种硬件，不能靠修改物理设备 ID 冒充异构计算。

轨道支持逐成员传播，或参考星位置加固定 ITRS 偏移；固定偏移误差尚未验证。轨道可见性按采样产生窗口，不是连续几何求根。FlowSim 消费窗口、终端名额和建链状态；`simple_keep`、O-S 参考策略和自定义选择接口已存在。自动生成的轨道配置仍使用零建链耗时；60s/2s 可显式配置，但当前按终端使用历史计费，按卫星对记录首次/重建的规则仍待补。

星间链路参数可手动配置或通过 ISLSim 生成。动态 ISLSim 预算目前取可行带宽下界、延迟上界，不连续更新在途速率。ns-3 使用真实 PointToPoint/PPP/DropTail 包级传输，目前仅支持静态连接；没有 TCP、NCCL 协议或重传模型。FlowSim 和 ns-3 不能都宣称已支持动态星座建链。

## 运行和核查

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

后续工作为接入更新 M-L、补齐 PD/AF 对外事件和传输、异分片 KV 迁移、卫星对建链状态、研究策略及大规模开销测算。仿真加速、热/能耗/成本评估后续开展。
