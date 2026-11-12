# SGP4 轨道根数输入合同

## 状态

- 合同 ID：`orbitserve.sgp4_element_bundle.v1`
- 决策状态：已实现
- 冻结日期：2026-07-20
- 代码状态：已实现公开输入模型、严格校验器、基于直接轨道根数的
  SGP4 至 TEME 传播、基于固定数据的 TEME 至 ITRS 转换、
  仅涉及几何的 SGL/ISL/日照评估，以及规范化 Orbit Trace
  序列化

独立的 STK 11.6 验证遵循
[`STK_VALIDATION_CONTRACT.md`](STK_VALIDATION_CONTRACT.md)。

本合同定义提供给 Orbit、用于 SGP4 传播的参数。
规范的端到端公开入口为 `generate_orbit_trace`。

公开输入入口为 `Sgp4ElementBundle`、
`Sgp4SatelliteElements`、`Sgp4ElementProvenance` 和
`load_sgp4_element_bundle`，均从 `astermind.builder.orbit` 导出。加载器接受
JSON 路径、JSON 兼容映射或已校验的 bundle。加载器本身
不会初始化 SGP4 引擎。

公开传播入口为 `propagate_sgp4_teme`。它接受
一个已校验的轨道根数包、规范的仿真 UTC 历元，以及唯一且
非负的相对采样时刻。它使用 `python-sgp4==2.27`、WGS-72 和
improved 模式；任何初始化或采样错误都会使整个调用失败；
返回的记录按 `(time_s, satellite_id)` 排序，TEME 位置和速度单位分别为
千米和千米每秒。

公开坐标系入口为 `transform_sgp4_teme_to_itrs`。它使用
`astropy==6.1.3`，禁用 IERS 网络更新，并要求本地 IERS-A 和
闰秒文件覆盖每个采样时刻。结果中记录这些文件的 SHA-256 值、数据发布版本、
已安装的 `astropy-iers-data` 版本，以及地球定向值是最终值还是预测值。
绝对数据路径不纳入结果。Orbit 默认使用随项目提供的
`astropy-iers-data-0.2026.5.11.1.8.52` 文件，并在使用前校验其固定哈希值；
调用方也可以显式提供其他不可变文件。
数据超出范围时，整个转换失败，而不会静默外推。

公开几何入口为 `compute_sgp4_sun_positions_itrs` 和
`evaluate_sgp4_geometry`。几何评估器接受完整的
`Sgp4EarthFixedResult`、仅含几何信息的地面接入点，以及显式指定的候选
ISL 端点对。它生成确定性采样，以及覆盖
`[first_sample_time, last_sample_time)` 的左闭右开窗口。它不选择 ISL
拓扑，也不输出带宽、终端、路由、图、SOC 或其他
Runtime/Architecture 状态。

地面几何使用 WGS-84 大地坐标和局部大地法线向上的方向。
ISL 遮挡测试使用完整线段与 WGS-84 椭球体相交判定，并支持可选的非负净空裕量。
太阳位置使用 Astropy 的地心太阳星历，并在地固坐标结果所固定的精确
IERS/闰秒溯源条件下转换至 ITRS。食判定使用圆锥形本影、
IAU 标称太阳半径和可选的大气层地球半径裕量。布尔值交叉点通过对相应
带符号边界函数进行线性插值来细化；该细化方法由
`sgp4-wgs84-geometry-v1` 声明。

Trace 适配器入口为 `write_sgp4_orbit_trace`。它接受前述公开阶段产生的
已校验轨道根数包、TEME 传播结果、ITRS 转换结果和
几何结果。在写入任何内容前，它要求轨道根数集标识、仿真历元、有序采样
键、卫星成员集合、几何源溯源和均匀采样网格完全一致。
它将规范 JSON、文件哈希、Runtime/Architecture 字段拒绝逻辑和
清单构建委托给保持不变的公开
`write_orbit_trace` 入口。

Trace 星历记录有意使用明确的 `position_teme_km`、
`velocity_teme_km_s`、`position_itrs_km` 和 `velocity_itrs_km_s` 字段。
它们不会复用旧的 `r_eci_km` 名称。卫星目录包含规范的结构化
SGP4 平均根数记录。窗口记录只包含几何信息和采样所得的距离/时延边界；
不包含带宽、终端、图、路由、容量或 SOC 状态。Trace 溯源记录所有
轨道根数、传播、坐标系、地球定向和几何算法/数据版本。
合成数据和预测数据状态会作为显式警告输出，绝不会自动推断其是否可用于正式结果。

## 已冻结决策

1. 权威输入是结构化、全精度且兼容 SGP4 的平均根数集，
   而不是两行定宽 TLE 文本。
2. Orbit 直接使用结构化数值初始化 SGP4。可以生成规范 TLE
   行，作为审计和互操作产物，但量化后的文本行不会被重新读取为传播源。
3. 输入根数具有 SGP4 平均根数语义。它们不是瞬时
   开普勒根数，不得仅通过重命名笛卡尔状态记录或经典轨道根数记录来生成。
4. 合同 v1 仅支持 WGS-72 引力常数和 improved SGP4
   运行模式。
5. 合成输入必须标识为合成数据，不得声称经过观测拟合。
   Orbit 不会静默地将合成输出提升为正式实验依据。
6. 相同的规范轨道根数输入、传播请求、固定地球
   定向数据、schema 版本和算法版本，必须生成字节级完全一致的 Orbit 输出。

## 模块边界

上游星座或场景构建器负责生成或拟合平均根数。Orbit 负责校验、
确定性规范化、SGP4 初始化与传播、坐标转换、几何窗口
计算和 Trace 序列化。

上游提供方不得依赖 Orbit 实现文件。输入通过公开的
`astermind.builder.orbit` 入口传入。Runtime 仅消费公开的 Orbit Trace
合同，不得自行初始化或调用 SGP4 引擎。

本合同不会将 Architecture 或 Runtime 数据划归 Orbit。初始
SOC、终端能力或分配、活动图、带宽、路由、队列和能量状态
均不属于本输入，也不属于 Orbit 输出。

## 规范根数包结构

传输表示是具有以下结构的 JSON 兼容映射：

```json
{
  "schema_version": "orbitserve.sgp4_element_bundle.v1",
  "element_semantics": "sgp4_mean_elements",
  "gravity_model": "wgs72",
  "ops_mode": "improved",
  "time_scale": "utc",
  "satellites": [
    {
      "satellite_id": "sat-00001",
      "satnum": 10001,
      "classification": "U",
      "object_id": "2026-001A",
      "epoch_utc": "2026-07-20T00:00:00.000000Z",
      "inclination_deg": 53.0,
      "raan_deg": 0.0,
      "eccentricity": 0.001,
      "argument_of_perigee_deg": 0.0,
      "mean_anomaly_deg": 0.0,
      "mean_motion_rev_per_day": 15.25,
      "mean_motion_dot_rev_per_day2": 0.0,
      "mean_motion_ddot_rev_per_day3": 0.0,
      "bstar_inv_earth_radii": 0.00001,
      "ephemeris_type": 0,
      "element_set_number": 1,
      "revolution_number_at_epoch": 1
    }
  ],
  "provenance": {
    "source_kind": "synthetic_sgp4_mean_elements",
    "source_name": "scenario-builder",
    "source_version": "example-v1",
    "observationally_fitted": false
  }
}
```

示例值仅用于说明表示形式；它们不是经过校验的
参考轨道。

## 根数包字段

| 字段 | 类型 | 必需值或规则 |
|---|---|---|
| `schema_version` | 字符串 | 必须为 `orbitserve.sgp4_element_bundle.v1` |
| `element_semantics` | 字符串 | 必须为 `sgp4_mean_elements` |
| `gravity_model` | 字符串 | v1 中必须为 `wgs72` |
| `ops_mode` | 字符串 | v1 中必须为 `improved` |
| `time_scale` | 字符串 | v1 中必须为 `utc` |
| `satellites` | 数组 | 非空；记录唯一，并按 `satellite_id` 规范排序 |
| `provenance` | 对象 | 必须包含下文所述的来源和拟合声明 |

v1 拒绝未知的根数包、卫星或溯源字段。此失败关闭规则可防止拼写错误的参数
和未声明的单位被静默忽略。新增字段必须使用后续的增量 schema 版本。

## 卫星轨道根数字段

### 标识与记录信息

| 字段 | 类型 | 规则 |
|---|---|---|
| `satellite_id` | 字符串 | 必填、非空且稳定的 Orbit 标识；在根数包中唯一 |
| `satnum` | 整数 | 必填，`1..339999`，在根数包中唯一；用于 SGP4/TLE 标识 |
| `classification` | 字符串 | 必填，为 `U`、`C` 或 `S` 中的一个 ASCII 字符 |
| `object_id` | 字符串或 null | 可选的 CCSDS 风格国际编号 `YYYY-NNNPPP`；没有编号的合成对象使用 `null` |
| `ephemeris_type` | 整数 | 必填，且 v1 中必须为 `0` |
| `element_set_number` | 整数 | 必填，`0..9999` |
| `revolution_number_at_epoch` | 整数 | 必填，`0..99999` |

`satellite_id` 是 Orbit Trace 记录使用的标识符。`satnum` 和
`object_id` 是外部目录或 TLE 记录信息，不得替代它。
重新排列输入记录不得改变卫星标识或输出字节。

### 历元和物理根数

| 字段 | 单位 | 范围或解释 |
|---|---|---|
| `epoch_utc` | UTC 文本 | RFC 3339 UTC，规范形式为 `YYYY-MM-DDTHH:MM:SS.ffffffZ`；秒为 `00..59` |
| `inclination_deg` | 度 | `0 <= value <= 180` |
| `raan_deg` | 度 | `0 <= value < 360` |
| `eccentricity` | 无量纲 | `0 <= value < 1` |
| `argument_of_perigee_deg` | 度 | `0 <= value < 360` |
| `mean_anomaly_deg` | 度 | `0 <= value < 360` |
| `mean_motion_rev_per_day` | 圈/天 | 有限且严格为正 |
| `mean_motion_dot_rev_per_day2` | 圈/天^2 | 有限，真实一阶导数 `dn/dt` |
| `mean_motion_ddot_rev_per_day3` | 圈/天^3 | 有限，真实二阶导数 `d2n/dt2` |
| `bstar_inv_earth_radii` | 1/地球半径 | 有限的 SGP4 BSTAR 值 |

导数字段包含数学导数，不包含 TLE 第 1 行显示值
`ndot/2` 和 `nddot/6`。只有 TLE 格式化器会应用这些表示约定。

所有数值输入都必须是 JSON 整数或有限的 JSON 数值。布尔值
不视为数值。单位转换在 Orbit-to-SGP4 适配器中显式完成：
度转换为弧度，圈/天转换为引擎的角速度单位，导数单位仅转换一次。

## 溯源字段

| 字段 | 类型 | 规则 |
|---|---|---|
| `source_kind` | 字符串 | `synthetic_sgp4_mean_elements` 或 `observationally_fitted_general_perturbations` |
| `source_name` | 字符串 | 必填、非空的生成器、目录或拟合流水线名称 |
| `source_version` | 字符串 | 必填、非空且不可变的版本、修订版或数据发布标识 |
| `observationally_fitted` | 布尔值 | 合成输入必须为 `false`，观测拟合输入必须为 `true` |

溯源信息可以描述上游提供方如何生成平均根数，
但不得包含主机名、绝对路径、实际生成时间、
随机进程 ID 或其他不可复现的元数据。

对于合成输入，未来的 SGP4 路径默认将
`valid_for_paper_main_results` 设为 `false`。任何允许合成场景用于正式结果的
策略都归实验负责人所有，并且必须经过显式评审；
Orbit 绝不会根据数值上看似合理的根数自行推断。

## 权威性和 TLE 生成规则

- 上述结构化字段是唯一的传播权威输入。
- 输入字段 `tle_line1` 和 `tle_line2` 不属于 schema v1，
  将作为未知字段被拒绝。
- 审计 TLE 格式化器使用同一结构化记录、确定性舍入、定宽字段、
  校验和规则，并在需要时使用 Alpha-5 卫星编号编码。
- 如果出现溢出或无法表示的值，审计 TLE 生成必须将其报告为错误，
  不得静默截断。
- 仅允许在格式化器兼容性测试中，将生成的 TLE 重新解析到 SGP4。
  生产传播不得使用该往返转换。
- 由于 TLE 是精度较低的定宽表示，因此不要求从审计文本行传播的结果
  与直接从全精度结构化输入传播的结果达到位级一致。

## 传播和坐标系解释

轨道根数包不包含仿真采样网格。Orbit 传播请求使用现有的
Trace 时间语义，分别提供开始时间、结束时间和步长。每个请求的绝对采样时刻
均由仿真历元和 `time_s` 确定性推导。

SGP4 原生生成 TEME 坐标系下的位置和速度，单位分别为千米和千米每秒。
任何地固地面几何都使用显式的 TEME 至 ITRS 转换。
简单的地球自转角或未声明的 ECI/ECEF 别名都不能作为可接受的高精度替代方案。

实现固定并记录：

- SGP4 库和算法版本；
- WGS-72 常数选择和 improved 运行模式；
- 时间转换实现；
- TEME 至 ITRS 转换实现；
- 地球定向数据发布版本或不可变数据哈希；
- 所有窗口检测和边界细化算法版本。

在可复现运行期间，禁止通过 Runtime 网络访问更新地球定向数据。
固定数据缺失或超出范围时，必须失败关闭，或输出具有显式版本、
且不具备正式结果资格的回退结果。

## 校验和失败行为

未来的公开校验器必须在传播前拒绝整个根数包，
只要出现以下任一情况：

- 缺少必填字段、存在未知字段或类型错误；
- 标识符为空或重复；
- 数值非有限或超出声明范围；
- 角度依赖隐式折返，而不是使用规范输入；
- 历元不是规范 UTC，或包含闰秒字面值；
- 轨道根数语义、引力模型、运行模式或时间尺度
  不受支持；
- 溯源信息与 `observationally_fitted` 不一致；
- 任一卫星的 SGP4 初始化器返回错误。

每个采样时刻的 SGP4 错误码必须与卫星标识和采样时间一起保留。
失败采样绝不能被序列化为看似合理的零值或陈旧位置。
后续实现阶段将冻结以下选择：整个 Trace 失败，还是支持显式的部分结果且
不具备正式结果资格的诊断 Trace；初始生产路径将使整个运行失败。

## 确定性和规范标识

在哈希或传播前，Orbit 按如下方式规范化根数包：

- 卫星记录按 `satellite_id` 排序；
- JSON 对象键按字典序排序；
- 使用 UTF-8、LF、一个文件末尾换行，并遵循 Orbit Trace 有限浮点数规则；
- 负零转换为 `0.0`；
- 不执行隐式角度折返、历元舍入或重复项自动去重；
- 标识符和溯源字符串在校验后原样保留；
- 传播请求及所有固定的算法/数据版本均参与 Trace 标识计算。

未来的规范化输入会公开形如
`sgp4-elements-v1-<sha256>` 的 `element_set_id`。其摘要覆盖规范的物理控制参数
和卫星记录，但不包含非物理溯源信息。完整输出字节仍包含确定性溯源信息，
因此，即使 `element_set_id` 保持不变，修改溯源信息也可能改变文件字节。

## 跨模块消费方要求

Runtime、DSE 和 Experiments 必须使用公开的 `astermind.builder.orbit` 入口
和结构化 SGP4 轨道根数包。圆轨道配置、旧的
`r_eci_km` Trace 结构、Trace 构建器和兼容别名均已删除。
DSE/Experiments 的生成和搜索策略由其负责人后续处理；
Orbit 不保留圆轨道回退方案。

根依赖和随项目提供的 IERS package-data 在
`pyproject.toml` 中声明。

## 合同冻结验收标准

- 上游每个必需参数都有唯一的名称、类型、单位和语义解释。
- 结构化全精度输入与审计 TLE 的权威关系明确无歧义。
- 合成数据和经过观测拟合的数据溯源不会混淆。
- 校验、错误处理、坐标系要求和确定性标识均有明确规定。
- 跨模块消费方具有明确的迁移要求，并且只使用公开的
  Orbit 接口。
