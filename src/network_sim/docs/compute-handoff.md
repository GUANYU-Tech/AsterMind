# 计算侧对接入口

## AsterMind 的算子级估时

构造 `OperatorCommunication(PhysicalTopology)`。OrbitServe 在初始化时提供拓扑，AsterMind 在批内调用 `estimate(CommunicationOperator)`，同步取回 `OperatorEstimate`。这个接口不读全局时钟、不维护在途流，也不向 GlobalTimeline 登记事件。

| `CommunicationOperator` 字段 | 口径 |
|---|---|
| `operation`, `algorithm` | 算子及算法。支持 ring AllReduce／AllGather、点对点 send_recv、直接 broadcast、变长 AllGatherV／ReduceScatterV、pairwise AllToAll。 |
| `participants` | 物理 GPU ID。点对点按源、目的排列，广播第一个 GPU 是源，分片和矩阵按参与者顺序排列。 |
| `bytes`, `bytes_semantics` | 传输量及口径。点对点和广播用 `payload`，均匀集合通信用 `per_rank_input`，AllGatherV 和显式 AllToAll 矩阵用 `aggregate_input`。ReduceScatterV 用 `per_rank_input` 表示一个 rank 的完整输入，等于各输出分片的总字节数。 |
| `ring_order`, `reduction` | 环形算子给出完整环顺序。AllReduce 使用 `sum`。 |
| `partition_bytes`, `traffic_matrix_bytes` | 变长分片的逐 GPU 字节数，或 AllToAll 的有向非本地流量矩阵。 |

Comm 根据算法展开传输轮次，在初始化拓扑上为每条流选择确定性的最短传播时延路径。单条流按 **字节数 ÷ 路径最窄容量 + 沿途传播时延** 估算。同轮取最大值，轮间串行累加。设备内部容量可限制单条路径，两条流同时经过也各自按孤立路径估时。动态可用窗口、捕获和流量竞争不进入这个估算。

`OperatorEstimate.duration_ns` 是整数纳秒的相对时长。另返回 `rounds`、`flow_count`、`cross_satellite_flows` 和 `cross_satellite_bytes`，供批次审计使用。这些计数表示拆分出的估算流。非法输入和不可达路径抛出 `ValueError`，不支持的算子或算法抛出 `NotImplementedError`。

```python
from network_sim import CommunicationOperator, OperatorCommunication

comm = OperatorCommunication(topology)
gpu_ids = ("gpu0", "gpu1")  # 在 topology 中配置了互相可达的两个 GPU
estimate = comm.estimate(CommunicationOperator(
    operation="all_reduce", algorithm="ring", participants=gpu_ids,
    bytes=1_000_000, bytes_semantics="per_rank_input",
    ring_order=gpu_ids, reduction="sum",
))
print(estimate.duration_ns)
```

Execution 用返回时长计算批内前驱和后继的起止时刻。整批最晚结束的时刻才登记为 `batch:complete`。

## 原 dev 的通信估时

GLM-5.2 A800 实验选择 `analytical_roofline_v1`。普通环形 AllReduce 把算法轮次折进每 rank 的字节系数，再除以带宽得到算子总时长。普通 AllToAll 同样按每 rank 输入乘 `(N−1)/N` 估算总传输量，未逐条构造物理路径。dev 的专用交换网络 AllToAll 可以接收显式流量矩阵，按各 NIC 的收发字节峰值和启动成本给出总时长，仍没有链路逐轮推进。参见 [dev 快照中的通信预测器](../../astermind/ArchPrism/midlayer/upstream/src/kernel/predictors/communication.py)。

## 批外系统传输（v0.2）

OrbitServe 的请求上传、token 交付和 KV 交接可继续通过 `CommunicationRuntime` 使用动态建链、流量共享和传输完成／失败事件。下文是该独立传输接口的用法，供批外流程和网络单测使用。图见 architecture.md，完整契约见 interfaces.md。

## 先跑通

解压后在 network_sim 的父目录执行：

```powershell
python -m pip install -r network_sim/requirements.txt
python -X utf8 -B -m unittest discover -s network_sim/tests -v
python -X utf8 -B -m network_sim --scenario network_sim/examples/public_basic.json --transfers network_sim/examples/public_transfers.json --output-dir network_sim/runs/basic
python -X utf8 -B -m network_sim --scenario network_sim/examples/handoff_dynamic.json --transfers network_sim/examples/public_transfers.json --until-ns 3000000 --output-dir network_sim/runs/dynamic
```

内部交接包带 Windows x64 DLL；其他平台或修改 native 源码须重建：python -m network_sim.build_native --compiler g++ --upstream-tests，也可指定 clang++ / Zig 路径。需要 Python 3.10+、requirements.txt，不需要完整 AsterMind 安装。

动态示例使用合成参数：两笔 1 MB 传输共用 8 Gbps Switch，捕获 200,000 ns，检查间隔 100,000 ns；一条链在 1,000,000 ns 断开，另一条持续可用。它验证等待、失败和容量释放，不代表真实星间测量。

## 批外传输调用方要提供

1. 稳定物理 GPU ID、rank→GPU 映射。逻辑组与物理连接解耦，改部署不必改通信算法。
2. 已展开点对点传输。每条提供起终点、字节数、就绪时刻、唯一 ID 和关联 ID。
3. 统一事件时钟。比较网络 next_event_time_ns 与计算／请求事件，调用 advance_to；不把孤立估计当并发场景耗时。
4. 消费成功和失败反馈。KV 目标空间、源释放、失败回滚由 O-S／KV 管理模块处理。
5. 明确计时归属。一次跨设备通信只在 M-L 或此网络计一次，也只由一个来源提交。GPU 内访存仍归 M-L。

## 最小调用

```python
from network_sim import CommunicationRuntime, PhysicalTopology, Scenario, TransferRequest

topology = PhysicalTopology(Scenario.load("network_sim/examples/public_basic.json"))
with CommunicationRuntime(topology) as network:
    rank_to_gpu = {0: "gpu0", 1: "gpu2"}  # 上层维护
    network.submit(TransferRequest(
        id="op-1-send-0", source=rank_to_gpu[0], target=rank_to_gpu[1],
        size_bytes=1_000_000, ready_time_ns=0, origin="multi_gpu",
        operation="send", correlation_id="op-1"))
    # 集成时应与其他模块事件比较；这里仅跑一个静态网络任务。
    while (time_ns := network.next_event_time_ns()) is not None:
        for event in network.advance_to(time_ns):
            if event.kind == "completed":
                print("数据就绪", event.id, event.completion_time_ns)
            else:
                print("上层处理失败", event.id, event.failure_time_ns, event.reason)
```

上例用静态配置，循环会结束。动态策略有周期检查，须设置系统期限，独立测试用 network.run(until_ns=...)。

topology.view 给配置图，candidate_path 给候选结构路径，route 给当前可用路径，snapshot.link_policy 给 down / acquiring / up 和历史。先用静态配置验证计算依赖，再开动态窗口验证等待／失败。轨道模块未来输出可用窗口；当前未接自动轨道、续传、包级协议或 surrogate。

## 发布边界

内部交接包比基础源码包多一个合成动态例子和 Windows DLL。公共基础包仍只含 public_basic.json、public_transfers.json 两个配置；私有参数、运行结果、Mesh／8-neighbor 等实验内容排除。测试验证模型约定，不验证真实耗时准确度。
