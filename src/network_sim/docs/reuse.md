# 复用来源与改动范围

来源为用户提供的 `astermind-feature-flowsim-web-integration` 本地代码包。原 AsterMind、OrbitServe 和 Mid-Layer 目录未修改。

| 本目录代码 | 复用方式 |
|---|---|
| `physical/_vendor/astermind_interfaces.py` | 完整复制 AsterMind `src/astermind/interfaces.py`，未修改；复用 canonical topology 与校验 |
| `comm/_vendor/flowsim/src/event_simulator.cc` | 复用事件执行、max-min 分配、发送进度、传播与统计；增加外部时钟、状态查询、启用／失效与失败反馈 |
| `comm/_vendor/flowsim/src/topology.cc` | 复用路径搜索；为零延迟内部容量通路添加循环保护、按启用状态选路 |
| `comm/_vendor/flowsim/src/simulator.cc` | 原样复制，用于运行原有回归 |
| `comm/_vendor/flowsim/tests/test_flowsim.cc` | 原样复制，12 项原有测试 |
| `physical/_vendor/orbitserve_link_selection.py` | 原样复制 O-S 通用及 Walker 连接选择方法；适配器提供明确平面关系 |
| `comm/native/bridge.cc` | 新增 C 接口，不实现另一套分配器 |
| `physical/topology.py` | 新增 Cell／多终端包装与容量转换；没有调用原“一星一个终端”的 composer |

没有直接使用原 FlowSim JSON compiler，因为它没有设备总容量通路，也不能表达 GPU 不作为隐式中继的转换；native 适配器直接使用同一个 FlowSim C++ 引擎。

本地副本的修改包括：

1. EventSimulator 新增 `advance_until`、`next_event_time_ns`、`progress` 和 `completed_report`，使未完成流保留状态并接受外部时钟驱动。
2. 本适配版本的 EventSimulator 使用确定性最短延迟路径，避免内部零延迟容量通路使无 ECMP 配置的逐跳选路走入死路；原 Simulator 的 ECMP 测试仍保留。
3. `shortest_latency_route` 不允许候选路径重复节点，避免零延迟环导致路径比较无限绕圈。
4. 完成报告筛选已完成任务；共享算法和正常发送／传播计时逻辑没有重写。
5. v0.2 增加动态启用边、失效终止流及失败查询；此路径的失败／传播约定是新增适配策略，不声称来自原 FlowSim。
6. O-S 只复用连接选择，不整体移植原轨道时间线与捕获参数；两类策略使用本版捕获状态机。

逐文件原始哈希和副本哈希在 [source_manifest.json](source_manifest.json)，早期接入差异在 [vendor_changes.patch](vendor_changes.patch)。这不是“上游文件完全没改”的接入；准确说法是“原目录没改，在明确标记的局部副本上增加接口与路由保护”。

基础包采用允许列表，只复制上述必要代码，不复制项目整体或任何研究场景配置。当前提供的是本地开发与发布预览；所给代码包没有随附清晰的独立 LICENSE 文件，正式公开分发前需由项目所有者确认来源授权与最终许可证。这里不替上游声明一个不存在的许可证。
