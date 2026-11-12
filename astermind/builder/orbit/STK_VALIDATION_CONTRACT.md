# STK 11.6 验证合同

## 状态

- 合同 ID：`astermind.builder.orbit.stk_validation`
- 决策状态：已为诊断性试点冻结
- 冻结日期：2026-07-27
- 已验证的参考环境：STK 11.6.0、build 11.6.0.453、Windows x64
- Orbit 生产合同：不变

本合同规定如何将 STK 用作基于 SGP4 的 Orbit 实现的独立离线参考。
STK 仅作为验证工具。它不是生产依赖，不参与 Orbit Trace 生成，也不
改变任何公开的 `astermind.builder.orbit` 输入或输出。

诊断性试点将环境、映射、比较指标和证据格式冻结。数值通过/失败阈值
仅在试点语料完成评估后冻结。在该阈值修订通过评审之前，STK 结果仅为
诊断证据，不得将其描述为正式的精度认证。

## 归属与仓库边界

Orbit 验证工作仅限于 `astermind/builder/orbit/` 和
`tests/unit/orbit/`。

- 后续 STK 适配器代码应位于 `astermind/builder/orbit/validation/`。
- 根 `pyproject.toml` 已经项目负责人明确授权，加入
  `astermind.builder.orbit.validation` 及固定诊断语料的 package-data 条目。
- 映射、解析器、规范化导出和指标的单元测试应位于
  `tests/unit/orbit/`。
- 实时 STK 运行属于需主动启用的本地验证，不是常规单元测试。
- 本项工作不修改 Runtime、DSE、Workload、Architecture、Serving
  及其测试。
- 其他模块继续仅通过公开入口 `astermind.builder.orbit` 使用本模块。

不得向规范化 Orbit Trace 合同加入任何 STK 字段、产物、依赖或场景路径。

## 已验证的 STK 执行环境

参考环境固定如下：

| 控制项 | 冻结值 |
|---|---|
| 操作系统 | Windows x64 |
| STK 版本 | 11.6.0 |
| STK build | 11.6.0.453 |
| 自动化传输方式 | 通过 `127.0.0.1:5001` 使用 STK Connect |
| Connect 模式 | 开启确认，关闭异步输出 |
| 场景和报告命令读取超时 | 120 秒 |
| 传播器模块 | 内部 `SGP4` 模块 |
| TLE 更新模式 | `None` |
| 在线星表或地球数据访问 | 禁止 |
| 场景持久化 | 临时使用且绝不保存 |
| 规范时间尺度 | UTC |
| 位置和速度单位 | km 和 km/s |

COM 自动化可作为可选的本地回退方式，但它不是 STK 11.6 基线的规范
传输方式。适配器必须记录 STK 版本、build、传输方式、传播器模块、报告
样式和验证设置。不得序列化安装路径、主机名、进程 ID、实际运行耗时或
其他主机特定元数据。

任何不同的 STK 版本、build、传播器模块或自动化传输方式均构成不同的
参考环境；其结果与此基线合并前，必须先完成环境验证。

## 验证通道

### 通道 A：全精度直接根数

通道 A 是主要的生产合同对比通道。它将权威的
`Sgp4ElementBundle` 直接映射为 STK SGP4 根数，传播前不生成和解析 TLE。

### 通道 B：审计 TLE 互操作性

通道 B 向 STK 和 `python-sgp4` 提供同一确定性审计 TLE。它用于隔离
TLE 解析和定宽表示的兼容性，但包含 TLE 量化误差。通道 B 仅用于诊断，
不得取代通道 A 成为生产合同对比通道。

两个通道必须分别报告，其误差绝不得合并为单一统计量。

## 直接根数映射

STK 使用以下命令配置：

```text
SetPropagatorDefaults / SGP4 Module SGP4 UpdateMode None ValidateEphemeris On Use1PtInterpolation Off Output1PtWarning On
```

随后使用 `SetState ... SGP4 ... TLESource DefineElements Source Elements`
定义每颗卫星。

| Orbit 字段 | STK 字段或处理方式 |
|---|---|
| `satellite_id` | 在结果映射中保留；可使用确定性的 STK 安全本地对象名 |
| `satnum` | `SSCNumber` |
| `classification` | `Classification` |
| `object_id` | 确定性地转换为紧凑的 `IntlDesignator`；省略 `null` |
| `epoch_utc` | UTC 格式的 `OrbitEpoch`；无法表示的时刻按失败关闭处理 |
| `mean_motion_rev_per_day` | 第一个 `ElementSet` 值，直接以 rev/day 传入 |
| `eccentricity` | 第二个 `ElementSet` 值 |
| `inclination_deg` | 第三个 `ElementSet` 值，单位为度 |
| `argument_of_perigee_deg` | 第四个 `ElementSet` 值，单位为度 |
| `raan_deg` | 第五个 `ElementSet` 值，单位为度 |
| `mean_anomaly_deg` | 第六个 `ElementSet` 值，单位为度 |
| `bstar_inv_earth_radii` | 第七个 `ElementSet` 值 |
| `mean_motion_dot_rev_per_day2` | `MeanMotionDot`，保留真实导数语义 |
| `mean_motion_ddot_rev_per_day3` | `MeanMotionDotDot`，保留真实导数语义 |
| `revolution_number_at_epoch` | `RevNumber` |
| `ephemeris_type` | Orbit 输入合同要求其为零；无其他 STK 控制项 |
| `element_set_number` | 记录为源元数据；不改变传播结果 |

STK Connect `ElementSet` 的平均运动值单位为 rev/day。禁止传入
`python-sgp4` 内部使用的 rad/min 值。

在非零导数用例可纳入正式阈值前，必须有一个夹具证明
`MeanMotionDot` 和 `MeanMotionDotDot` 保留了 Orbit 合同的真实导数语义，
而非 TLE 显示约定 `ndot/2` 和 `nddot/6`。若导数夹具失败，则这些用例的
通道 A 判定失败；不得通过将导数设为零来掩盖问题。

Orbit 输入继续采用 WGS-72 和 improved operation mode。STK 清单必须标明
所使用的内部 SGP4 实现。由外部模块或 AFSPC 模块生成的结果不属于本基线。

## 场景与采样网格规则

- 输入根数包和 Orbit 传播请求是不可变的验证输入。
- 至少需要两个采样时刻。
- 采样时刻必须为有限值、唯一、非负、有序且等间隔，与当前 Orbit Trace
  适配器的要求一致。
- STK 场景区间从第一个请求的绝对 UTC 时刻开始，到最后一个请求的绝对
  UTC 时刻结束。
- STK SGP4 星历步长等于 Orbit 采样步长。
- 每份报告均必须在 Orbit 的精确采样时刻请求。
- STK 采样如有缺失、重复、顺序变化或额外样本，该用例即判定失败。
- STK 输出导出后绝不插值到 Orbit 网格。

所有日期解析和格式化均显式使用 UTC。禁止使用本地时区和依赖区域设置的
日期格式。

## 状态向量证据

以下已安装的 STK 报告样式为参考来源：

| Orbit 量 | STK 报告样式 |
|---|---|
| TEME 位置和速度 | `TEMEofDate Position Velocity` |
| 地固系位置和速度 | `Fixed Position Velocity` |

TEME 是主要的 SGP4 对比项。STK Fixed 与 Orbit ITRS 之间的对比属于独立的
坐标系转换对比，不得混入 TEME 传播误差。

对于每颗卫星和每个样本，对比报告包含：

- 位置向量差及其欧几里得范数，单位为米；
- 速度向量差及其欧几里得范数，单位为米每秒；
- 径向、沿迹向和交叉轨迹向的位置差；
- 每项指标的 RMS、P95、最大值以及最差卫星/时刻。

状态指标采用以下固定统计约定：

- 所有有符号差值均定义为 `Orbit - STK`；
- RIC 正交基由同一时刻的 STK 参考位置和速度构造，径向为位置方向，
  交叉轨迹向为位置与速度的叉积方向，沿迹向补全右手系；
- RIC 逐样本分量保留符号；其 RMS、P95、最大值按绝对值统计；
- P95 使用 Type-7 线性插值，即排序后索引为
  `(样本数 - 1) * 0.95`；
- 最大值并列时，按规范顺序最先出现的
  `(time_s, satellite_id)` 记录为最差样本。

任何坐标系均不得仅标记为 `ECI` 或 `ECEF`。

## 几何证据

### SGL

- 每个 `Sgp4GroundIngress` 都使用相同的 WGS-84 大地纬度、经度和高度
  转换为 STK Facility。
- 唯一的策略约束为 `ElevationAngle Min <min_elevation_deg>`。
- 禁用地形、折射、光行时延迟、像差、通信、终端、带宽和可用性约束。
- 对每个请求的地面入口—卫星对收集 STK Access 和 AER/range 证据。
- AER 必须从 Facility 侧生成，使用 `TimeStep Bound <sample_step_s>` 对齐
  Orbit 采样网格；Satellite 侧的局部仰角不得作为地面仰角。
- STK 11.6 可将零窗口返回为带对象路径前缀的 `NoAccesses`。仅当对应
  Access 为空时，空 AER 报告才是合法结果。

### ISL

- 仅评估显式候选 `isl_pairs`。
- STK satellite-to-satellite Access 提供独立的视线窗口。
- 不引入星座拓扑、终端分配、活动图、带宽或路由决策。

### 日照

- Orbit `sunlit=false` 主要与 STK Umbra 对比。
- STK Penumbra 保留为诊断证据，不得在未说明的情况下与 Umbra 合并。
- 由 Orbit 圆锥形本影模型、大气余量或 STK 阴影约定造成的差异必须明确
  报告。

STK 区间统一转换为 Orbit 的半开区间约定 `[start_s, end_s)`，并裁剪至
请求的验证区间。

SGL、ISL 和日照的冻结指标如下：

- 采样状态一致性和混淆计数；
- 起始边界和结束边界的绝对误差；
- 持续时间误差；
- 区间交并比；
- 最差链路对、卫星、ingress 和边界时刻；
- 当双方均提供相应量时，比较距离和仰角差异。

## 诊断语料

首轮试点必须包含固定、本地且不可变的输入，并覆盖：

1. 近圆 LEO；
2. 有倾角且中等偏心率的 LEO；
3. 非零 BSTAR 和非零平均运动导数；
4. 深空/GEO SGP4 分支；
5. SGL 地平线和 ISL 地球边缘边界用例；
6. Umbra 和 Penumbra 边界用例；
7. 根数历元时刻的样本，以及跨越多天、分布在历元两侧的样本。

任何用例均不得在验证时下载 TLE 或参考数据。合成用例与基于观测拟合的
用例继续通过现有 Orbit 来源合同加以区分。

状态向量子语料已冻结为：

- 语料 ID：`stk-state-pilot-2026-07`；
- 文件：`data/stk_validation/state_corpus.json`；
- SHA-256：
  `0d47eec875b6cd7a98a6eb3c2d6e31b0da099d77216364624fec21b522b9c805`；
- 用例顺序：`near-circular-leo`、`moderate-eccentricity-leo`、
  `nonzero-drag-derivatives-leo`、`deep-space-geo`。

几何事件子语料已冻结为：

- 语料 ID：`stk-geometry-pilot-2026-07`；
- 文件：`data/stk_validation/geometry_corpus.json`；
- SHA-256：
  `639f28892bc0e23e7fd4bcdd898d7f2e98b0ea9ff3b67ecd9c363444b3a2cc5e`；
- 用例顺序：`leo-boundary-geometry`；
- 固定网格：`[0, 21600]` 秒，步长 30 秒；
- 覆盖：SGL 地平线、ISL 地球边缘、Umbra 和 Penumbra 边界。

状态向量子语料与几何事件子语料合并覆盖上述第 1 至第 7 项。该覆盖仅表示
诊断输入完整，不表示数值阈值已经冻结，也不得描述为正式精度认证。

## 证据产物

后续验证器至少输出以下规范化、机器可读的证据：

- `validation_manifest.json`;
- `stk_teme_samples.json`;
- `stk_fixed_samples.json`;
- `stk_sgl_windows.json`;
- `stk_isl_windows.json`;
- `stk_lighting_windows.json`;
- `comparison_metrics.json`.

验证清单包含输入哈希、Orbit 算法/数据版本、STK 版本/build/设置、报告样式、
采样网格控制项和明确的适用资格状态。原始 STK 场景文件不是规范证据，
不得提交。

规范证据 JSON 使用 UTF-8、排序后的键、紧凑分隔符、有限数值、规范化的
负零和一个结尾 LF。在同一已验证环境中重新运行同一用例时，必须复现
规范证据的字节内容。跨实现验收仍采用数值比较；不要求 STK 文件与
Orbit Trace 文件彼此达到字节级一致。

## 失败与清理行为

发生以下任一情况时，用例按失败关闭处理：

- STK 环境与已验证基线不匹配；
- Connect 不可用或返回 `NACK`；
- 无法建立内部 SGP4 模块或离线设置；
- 输入值必须通过未声明的转换才能表示；
- 传播或报告生成失败；
- 卫星标识或采样网格未精确匹配；
- 必需值缺失、重复、非有限或无法解析；
- 临时场景无法干净关闭。

验证器创建一个名称唯一的内存场景，绝不保存，并在 `finally` 清理路径中
卸载。场景存在性使用 `CheckScenario /` 判断，不使用可能在未完成初始化
场景上返回 NACK 的 `GetTimePeriod /`。若场景创建或后续命令的 ACK 因
传输中断而丢失，验证器必须关闭失步 socket、建立新连接、重新检查场景
状态并完成清理。它必须拒绝替换或关闭在验证开始前已经存在的不相关场景。

## 阈值策略

本合同有意不设定未经依据的数值验收阈值。诊断性试点首先记录完整的指标
分布，并识别算法、坐标系数据和事件模型之间的差异。

正式阈值需要经评审的合同修订，该修订必须：

1. 指明已接受的语料和参考产物；
2. 分别设置 TEME、Fixed/ITRS、SGL、ISL 和日照阈值；
3. 区分算法误差、TLE 量化误差和坐标系数据误差；
4. 记录每个例外，而不是放宽一个全局容差；
5. 在任何结果被称为 STK 认证前获得批准。

## 合同冻结验收标准

- STK 与生产 Orbit 路径隔离。
- STK 11.6 环境和自动化传输方式可复现。
- 每个 Orbit 根数都有一项明确的 STK 映射或失败关闭规则。
- TEME 与 Fixed/ITRS 误差分别评估。
- SGL、ISL 和日照对比保持 Orbit 模块边界。
- 试点语料和证据产物已固定。
- 在基于证据的阈值通过评审前，禁止作出正式精度声明。
