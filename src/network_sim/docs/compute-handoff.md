# 计算侧对接入口（v0.2）

本包独立实现物理拓扑、动态建链、基本流量共享；没有改计算侧。阅读顺序 model.py → topology.py → link_policy.py → runtime.py。图见 architecture.md，完整契约见 interfaces.md。

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

## 首次对接要提供

1. 稳定物理 GPU ID、rank→GPU 映射。逻辑组与物理连接解耦，改部署不必改通信算法。
2. 已展开点对点步骤：起终点、字节数、就绪时刻、唯一 ID、关联 ID；集合通信算法由多 GPU 操作层展开。
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
