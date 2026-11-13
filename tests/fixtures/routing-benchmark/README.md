# 路由策略修改与回归记录

基线：version/AsterMind-V1.0，ae6ee46a93c3d61be8ed00303b0a990d540f6553。

## 已确认行为

- 全局默认 min_propagation。Scheduler 查询和 Comm 的模型/系统通信都显式传入同一策略。
- min_hops 只统计跨卫星或地面归属的链路，内部连接和设备容量门不计跳数。
- earliest_arrival 按累计传播时延 + 数据比特数 / 预计瓶颈速率评分。每个资源按容量 / (当前竞争者 + 1) 估算。
- 保留优先队列搜索、剪枝和路径缓存，不限制候选路径数量。缓存存在本次仿真内存中，复用路径但重新估时；负载变化本身不会自动换路。
- 流级完成时间仍由 Comm/FlowSim 决定。可行性检查和返回的逐边记录仍使用原有逐跳时间；有等待时，评分额外包含等待时间。未来容量变化不由快照评分完整预测。
- 非提交查询的首个返回值是新评分预测的到达时间；commit=True 的旧预留日历仍返回逐跳到达时间。

## 回归

完整 tests/unit/runtime：229 项通过。覆盖全局策略、模型/系统入口、缓存、断链、容量变化、竞争、时间线、H-S 适用性及 Roofline 拒绝边界。新增评分解析测试覆盖单次序列化、传播累加、共享池，以及不能仅按前缀总分剪枝的反例。

## 6000 节点复测

固定合成 60×100 无环绕网格，11840 条双向边。500 km 相邻距离、10 Gbps、1 MiB。所有链路持续可用。uneven 按边索引注入 0–7 个竞争计数，不是真实背景流。每项三次中位数；计时运行不带计数包装器，结构及物理查询已预热，路径缓存每次清空。

|负载|目标|策略|展开状态|逐边预测|首次 ms|缓存 ms|
|---|---|---|---:|---:|---:|---:|
|empty|0:1|min_hops|1|2|0.061|0.023|
|empty|0:1|min_propagation|1|2|0.055|0.022|
|empty|0:1|earliest_arrival|1|2|0.053|0.021|
|empty|20:20|min_hops|840|2440|39.656|0.746|
|empty|20:20|min_propagation|840|2440|47.069|0.675|
|empty|20:20|earliest_arrival|840|2440|75.413|1.011|
|empty|59:99|min_hops|5999|17680|493.532|3.825|
|empty|59:99|min_propagation|5999|17680|484.410|3.628|
|empty|59:99|earliest_arrival|5999|17680|653.125|2.591|
|uneven|0:1|min_hops|1|2|0.042|0.015|
|uneven|0:1|min_propagation|1|2|0.051|0.017|
|uneven|0:1|earliest_arrival|1|2|0.064|0.026|
|uneven|20:20|min_hops|937|2689|49.895|0.466|
|uneven|20:20|min_propagation|937|2689|52.994|0.431|
|uneven|20:20|earliest_arrival|2529|7207|233.804|0.980|
|uneven|59:99|min_hops|7060|20672|457.553|1.996|
|uneven|59:99|min_propagation|7060|20672|500.211|3.765|
|uneven|59:99|earliest_arrival|23549|68581|2538.847|4.930|

新 Earliest Arrival 在不均匀负载下保留更多不可相互支配的状态，最远查询约 2.54 秒。缓存命中不展开状态。这个限制保留记录，没有加候选上限，也没有声称实现提速。

before.json / before.md 是改 Earliest Arrival 之前的历史结果；after.json 是本次结果。不同运行的宿主负载未控制，耗时仅用于数量级判断；展开状态数可直接核查。

复跑：python -m tests.scripts.bench_route_6000 --output tests/fixtures/routing-benchmark/after.json。
策略和缓存接口细节见 astermind/engine/COMMUNICATION.md。
