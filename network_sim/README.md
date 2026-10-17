# 设备拓扑与通信模块（v0.2）

静态基础与动态建链已经实现：自由配置 Cell／卫星／GPU／Switch／多终端，查询实际设备路径，以及按链路和设备总容量进行公平带宽共享。它独立运行，未修改多 GPU 操作层、OrbitServe 或 Mid-Layer 的计算代码。

先看 [分层架构图与代码对应](docs/architecture.md)；浏览器版在 [architecture.html](docs/architecture.html)，包含框架、两星设备图和共享瓶颈三张图。

## 当前支持

| 能力 | 实现情况 |
|---|---|
| Cell → 卫星 → 设备 | 独立身份和成员关系；每星设备数量可以不同；允许纯通信星 |
| 静态连接 | GPU–GPU、GPU–Switch、Switch–Switch、GPU–终端、Switch–终端、终端–终端；跨星边连接两个终端 |
| 多终端 | 独立编号、连接和容量；静态检查连接数量，动态约束捕获／已建链名额 |
| 显式 Switch | 支持多个 Switch、多个端口和多个终端；可选整体共享容量 |
| 模板与自由配置 | FC／Ring／Switch 是便捷生成器；自由边配置是基本能力；不读取 TP／PP 等分组 |
| 静态 Cell 间连接 | 显式跨 Cell 终端连接；两端必须声明 `external_access`；未连接轨道和可见性 |
| 路径 | 确定性最小传播延迟路径；不按拥塞改路；GPU 不自动充当中继 |
| 流量共享 | 直接使用 AsterMind FlowSim 的 max-min 分配，任务加入／结束时重新分配 |
| 外部时钟 | `next_event_time_ns` + `advance_to`，可供未来 O-S 全局事件时钟驱动 |
| 终态反馈 | 完成／断链失败各携带任务来源、关联编号、实际路径和时刻；每任务只通知一次 |
| 动态建链 | simple_keep 保持旧链；os_reference 复用 O-S 通用／Walker 选择；三项时间可配 |
| 可用窗口 | 外部输入可见区间，失效时中断在途任务，未开始任务等待建链；尚未接轨道 |
| 三层数据视图 | 多 Cell／单 Cell／单星从同一设备图导出 JSON |

## 在当前工作区运行

PowerShell，在 `network_sim` 的父目录运行。原来的 AsterMind 目录不必放入 `PYTHONPATH`；需要的少量源码副本已保留来源。

```powershell
$networkPython = 'python'
& $networkPython -X utf8 -B -m network_sim.build_native --upstream-tests
& $networkPython -X utf8 -B -m unittest discover -s network_sim/tests -v
& $networkPython -X utf8 -B -m network_sim --scenario network_sim/examples/public_basic.json --transfers network_sim/examples/public_transfers.json --output-dir network_sim/runs/public_basic
```

在别的电脑，使用 Python 3.10+，安装 `requirements.txt`，通过 `--compiler` 指定 g++、clang++ 或 Zig。构建脚本在当前工作区也能找到此前安装的 Zig。生成的动态库不能直接跨操作系统使用。当前 Windows 编译需要在可正常运行 Zig 的环境中执行；受限沙箱内 Zig 曾报告无法识别自己的可执行文件路径，已在外部构建环境完成验证。

## 最简单的公开示例

配置：[public_basic.json](examples/public_basic.json)。一个 Cell、两颗星、四张 GPU；星 A 有一台总容量为 8 Gbps 的 Switch，并连两个终端；星 B 两个终端分别接到目标 GPU。两条星间链路各为 8 Gbps，数据和带宽只是说明机制的合成数值，不是研究测量参数。

两笔 1 MB 传输分别使用两个终端：

```text
gpu0 → switchA → termA0 → termB0 → gpu2
gpu1 → switchA → termA1 → termB1 → gpu3
          ↑ 两条路径共用 Switch 的 8 Gbps 总容量
```

单独执行各需 1,000,103 ns；并发执行各需 2,000,103 ns。传播项为每条路径 103 ns。若去掉 Switch 整体容量，两条独立通路各需 1,000,103 ns。输出 `report.json` 可以看到实际路径、完成事件及共享资源的最大并发数。

## 如何接入三个输入方

```python
from network_sim import CommunicationRuntime, PhysicalTopology, Scenario, TransferRequest

topology = PhysicalTopology(Scenario.load("network_sim/examples/public_basic.json"))
with CommunicationRuntime(topology) as network:
    network.submit(TransferRequest(
        id="kv-001", origin="orbitserve", operation="kv_migration",
        correlation_id="request-001", source="gpu0", target="gpu2",
        size_bytes=1_000_000, ready_time_ns=0,
    ))
    # 独立测试可调用 run()；接入 O-S 时由全局事件时钟调用 advance_to。
    events = network.run()
```

三个输入方使用同一个 `TransferRequest`，共用一个 `CommunicationRuntime` 才会竞争相同资源。多 GPU 操作层先解析 rank→GPU 映射和通信步骤；O-S 提供 KV、请求或输出的实际端点及字节数；M-L 仅提交应由外部网络处理的传输。上层收到完成事件后推进依赖计算或提交 KV 状态。

接口的错误、时序和计时约定见 [接口说明](docs/interfaces.md)。这里没有集成三个实际运行时，也没有调用 M-L 的 surrogate；预留的是共同输入和事件接口，不是已完成集成。

## 参考策略与自由配置

静态拓扑保留为预建链基线；动态默认 simple_keep，旧链能用就保持，空闲终端再选伙伴。检查间隔、初次和重新捕获时间均须显式配置。另有 os_reference，复用原 O-S 连接选择方法；自由策略可传回调。细节见 [建链与续传说明](docs/link-policy-reference.md)。这份说明明确区分“连接保持”“重新建链”和“数据续传”。

Mesh／8-neighbor 等登记为实验项，当前不实现、不随基础发布包带出。自由配置仍允许用户自行表达相应图。公开示例与研究配置的分离方式见 [发布范围](docs/release.md)。

## 复用与计时边界

- 直接复用 AsterMind 的 canonical topology 模型和引用校验、FlowSim 路由、事件推进及 max-min 带宽分配。没有另写 Python 带宽分配器。
- 新增 Cell 归属、多设备配置、Switch／终端整体容量到内部通路的转换、Python/native 接口、外部时钟及状态查询。副本改动和原始哈希见 [复用来源](docs/reuse.md)。
- `bidirectional=true` 表示两个方向各有独立链路容量。设备整体容量是经过该设备的所有流共用的容量，包含两个方向；不是自动推断的真实 NVSwitch 参数。未配置整体容量时只受连接带宽限制。
- 使用流水流量模型：孤立传输时间 = 数据量／路径最小容量 + 沿途延迟之和。不是对每跳分别重复计一次整份数据串行发送。传播期间不再占用带宽。
- 设备总容量只表达带宽；不根据功率推算吞吐。GPU 本地访存仍归 M-L。
- 零字节任务沿用 FlowSim 行为，立即完成，不收传播费用。同设备任务被拒绝，交给本地模型。
- 未实现自动轨道／可见性计算、换路、重传、持久化恢复、包级协议或 surrogate 校准。可用窗口已能驱动建链和在途断链失败。候选配置不原地修改，当前连接状态由 LinkController 管理。

测试证明路径与共享实现符合这些约定，不代表空间通信或真实硬件耗时准确度已经得到验证。

## 当前计算侧交接

先读 [计算侧对接入口](docs/compute-handoff.md)。动态独立运行必须给 --until-ns；内部交接包另含合成 handoff_dynamic.json 和 Windows x64 DLL，公共基础包仍只带两个静态示例配置。
