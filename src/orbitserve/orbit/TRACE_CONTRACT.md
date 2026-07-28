# Orbit Trace 合同

## 状态

- 合同 ID：`orbitserve.orbit_trace_contract`
- 决策状态：已实现
- 冻结日期：2026-07-20
- 规范 schema：`orbitserve.orbit_trace`

本合同定义了 Orbit 唯一的、基于 SGP4 且仅包含几何信息的输出。在数据提供方
与使用方共同确认后，旧圆轨道仿真器、替代 schema、轨迹构建器和兼容别名均已删除。

## 模块边界

Orbit 以确定性方式生成物理几何与环境事实，不负责通信、调度、部署或请求执行决策。

### Orbit 负责的输入

- 结构化 SGP4 平均根数和稳定卫星标识；
- 仿真历元和采样网格；
- 地面入口标识、坐标和最低仰角；
- 显式指定的候选 ISL 对；
- 几何模型和日照模型的控制参数；
- schema、算法及不可变参考数据的版本。

输出路径、主机名、进程 ID、墙上时钟时间戳和临时名称均不是物理输入，不得影响轨迹标识或字节内容。

### Orbit 负责的输出

- 轨迹元数据、相对文件引用、SHA-256 值、数据溯源信息和警告；
- 仅包含几何信息的卫星目录；
- 有序的 TEME 和 ITRS 星历样本；
- 包含距离和传播时延事实的 SGL 与 ISL 几何窗口；
- 日照和阴影窗口。

### Runtime 负责的派生状态

| 字段或概念 | 权威来源 |
|---|---|
| `satellite_initial_soc` | 受 Architecture 约束的 Runtime 配置 |
| `terminal_catalog` | Orbit 卫星 ID 加上 Architecture 和 Runtime 配置 |
| `active_graph_snapshots` | Orbit 可见性加上 Runtime 状态和策略 |
| 带宽和容量 | Architecture 加上 Runtime 配置 |
| 终端分配 | Runtime 状态 |
| 路由、队列、拥塞、DVFS、SOC、能量 | Runtime 和调度器状态 |

这些字段不得出现在 Orbit 生成的载荷或 `OrbitTraceBundle` 中。

## 规范公开合同

其他模块只能从 `orbitserve.orbit` 导入。

- 生产方入口：`generate_orbit_trace`、`write_sgp4_orbit_trace`、
  `write_orbit_trace`
- 使用方入口：`load_orbit_trace`
- 公开类型：`OrbitTraceManifest`、`OrbitTraceBundle`

`OrbitTraceBundle` 提供 `trace_manifest`、`satellite_catalog`、
`ephemeris_samples`、`sgl_windows`、`isl_windows` 和 `sunlight_windows`，
以及只读元数据属性。不存在 `RuntimeTrace*` 或替代别名。

## 清单与记录的不变量

- 各文件中的 `trace_id`、schema、时间范围和卫星标识必须一致。
- `trace_id` 由规范物理输入和算法版本派生，绝不能由输出目录派生。
- 文件路径采用相对 POSIX 路径。
- 每个被引用的数据文件都有 SHA-256 值。
- 星历记录按 `(time_s, satellite_id)` 排序。
- SGL 使用共享的 `ingress_id` 命名空间。
- ISL 端点顺序采用规范顺序。
- 窗口采用左闭右开区间 `[start_s, end_s)`，且 `start_s < end_s`。
- 拒绝 Runtime 和 Architecture 负责的字段。

写入器输出：

- `trace_manifest.json`
- `satellite_catalog.json`
- `ephemeris_trace.json`
- `sgl_windows.json`
- `isl_windows.json`
- `sunlight_windows.json`

## 字节级可复现性

对于完全相同的规范输入、代码、schema、算法和参考数据：

- 所有 JSON 均使用不带 BOM 的 UTF-8、按键排序、紧凑分隔符和一个 LF；
- 数值必须为有限值；负零转换为 `0.0`；
- 时间和窗口边界四舍五入至小数点后九位；
- 其他浮点数规范化为 15 位有效数字；
- 类集合输入和警告以确定性方式排序并去重；
- 不序列化主机特定元数据或墙上时钟元数据；
- 完整文件集、原始字节、SHA-256 值和 `trace_id` 均完全相同。

清单内容哈希根据规范清单计算，计算时省略其自身的
`trace_manifest_content_sha256` 字段。

必要测试会比较不同输出目录、映射插入顺序、输入集合顺序以及相互独立的
`PYTHONHASHSEED` 值。在相同的受支持环境下，Windows 和 CI Linux 必须生成完全相同的字节。

## 使用方义务

- 使用方只能使用 Orbit 的公开入口。
- Runtime 加载仅包含几何信息的轨迹后，派生 SOC、终端目录、活动图、容量和运行时状态。
- Workload 网关数据通过公开合同进行适配；网关带宽、成本和运行时可用性不进入 Orbit。
- DSE 和 Experiments 必须提供或选择结构化 SGP4 根数包；已删除的圆轨道配置不能作为回退方案。

未来对任何公开字段或语义的修改，都必须先由 Orbit 数据提供方与受影响的使用方达成一致，
之后才能修改代码和接口测试。
