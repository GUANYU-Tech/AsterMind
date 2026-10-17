# 建链策略与续传边界（v0.2）

## 默认动态策略：simple_keep

动态配置启用后，跨星终端连接成为候选边；星内连接继续固定。每到检查时刻，保留仍可用的旧链，包括正在捕获的连接，再给空闲终端选择新伙伴。优先距离短者，其次按卫星 ID、连接 ID 排序。没有距离和坐标时仅按稳定 ID 排序，不假造距离。

两端都须有名额。每个终端的 max_external_links 约束正在捕获／建立的连接，不约束候选总数。捕获期间占名额、不能传输；完成后才可传输。检查不重置捕获计时。旧链不会因为出现更近伙伴而切换，也不按每次检查／每笔传输重复收捕获开销。

三项时间必须显式配置，整数 ns：

```json
"link_policy": {
  "mode": "simple_keep",
  "check_interval_ns": 10000000000,
  "initial_acquisition_ns": 100000000,
  "reacquisition_ns": 200000000
}
```

数字只展示填写方法，没有测量依据。若任意一端终端曾成功建立外部连接，新建／换伙伴用重新捕获时间；两端都从未成功建立则用初次时间。被取消且未成功的初次捕获不算“曾建立”。不设 link_policy 则沿用静态预建链基线、不收捕获开销。

## 可用性与周期检查

Link.availability_windows 表达外部已计算的可用区间 [start_ns, end_ns)。null 表示始终可用，空列表表示从不可用。distance_km 只用于选伙伴；传播时间仍独立输入 latency_ns，不自动联动。

新伙伴在下一次周期检查选择。已知窗口结束则在结束时刻立即断链／取消捕获，不等下次检查，避免明知不可见还继续传输。若输入只有采样快照，外部构造的窗口只能达到采样精度。

尚未接 Orbit 传播、姿态／指向、随机故障和动态距离。窗口是这些模块未来提供的结果。终端失效可通过关闭相关候选窗口表达；当前不支持运行中原地改配置。

## O-S 参考版本：os_reference

原 O-S 目录和工作区上游副本未修改。_vendor/orbitserve_link_selection.py 直接复制原 _select_terminal_edges、_edge_sort_key、_degree_limited_edges；源码／AST 核对一致：

- 不设 os_layout：按距离等排序、按卫星终端数限额选择。
- 设置 os_layout：保留同轨指定邻居；每对相邻轨道平面按距离做一对一匹配；同轨优先，再施加每星连接数上限。

```json
"link_policy": {
  "mode": "os_reference",
  "check_interval_ns": 10000000000,
  "initial_acquisition_ns": 100000000,
  "reacquisition_ns": 200000000,
  "os_terminal_count": 2,
  "os_layout": {
    "plane_by_node": {"a": 0, "b": 0, "c": 1, "d": 1},
    "same_plane_neighbor_pairs": [["a", "b"], ["c", "d"]],
    "adjacent_plane_pairs": [[0, 1]]
  }
}
```

plane_by_node 填卫星 ID，覆盖全部配置卫星；配对 ID 须小者在前。参考方法先选卫星级候选，再由适配器落实到候选已指定的终端并施加终端名额约束。物理名额不足可能少于原卫星级选择结果，不自动找替代终端。

参考版每次重新选择，不承诺旧链必保持。两个策略共用本版捕获状态机；保留的是 O-S 连接选择算法，不是移植整个快照引擎。原 O-S 冷启动参数、轨道输入、时间线未自动接入。原代码在工作区 integration/upstream/orbitserve/src/orbitserve/runtime/engine/communication.py。

## 自由策略与传输失败

CommunicationRuntime(..., link_selector=callback) 可替换选择器。回调接收排序后的候选 Link、旧预留连接 ID、当前 ns，返回连接 ID。控制器继续校验可用性、名额、捕获时间；私有策略无需进入公共包。

next_event_time_ns / advance_to 统一驱动检查、已知失效、捕获及传输事件。动态独立运行必须 run(until_ns=...)，避免永不停息的周期检查或无限等路径。

未开始任务无当前路径时 waiting_link，建链后自动开始；候选结构也不可达则提交失败。在途路径失效返回一次 FailureEvent(kind=failed, reason=link_down)，释放发送容量；重新建链不自动恢复失败任务。尚未交付的传播阶段也按失败处理；交付与断链同一 ns 时交付先完成。这是保守整流失败约定，没有包级确认模型。

remaining_bytes 是未发送字节，不是接收端未确认字节，不能直接当续传偏移。确认、换路、按块重试、续传和持久化恢复待后续。原 O-S 的 fixed_at_flow_start_no_reroute 标记也不证明这些能力已存在。
