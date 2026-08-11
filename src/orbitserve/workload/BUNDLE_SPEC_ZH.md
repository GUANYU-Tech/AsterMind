# OrbitServe Workload 模块与标准 Bundle 接口（中文）

workload 模块负责把不同来源的请求数据集转换成 OrbitServe 仿真和运行时共同
使用的稳定接口。

```text
原始数据集 + 地理分布 + 日内曲线
                 |
                 v
             数据集 adapter
                 |
                 v
          标准六文件 bundle
                 |
                 v
        loader -> upsample -> runtime
```

数据集特有的解析与语义放在 `src/workloads/<name>/adapter.py`；下游模块只依赖
`src/orbitserve/workload` 和标准 bundle，不直接依赖原始数据集格式。

## 模块职责

### 数据集 adapter

adapter 读取原始数据，映射地理与时间 pattern，完成 token、SLO 和会话转换，
最后写出标准 bundle。

当前两类参考 pattern：

| Pattern | Adapter | 数据语义 |
|---|---|---|
| ChatLLM | `src/workloads/main_composite_chat/adapter.py` | ShareChat 的 user/assistant 多轮会话 |
| Agent | `src/workloads/semianalasys/adapter.py` | 带 agent、subagent 和 hash KV 复用信息的模型调用链 |

两类 pattern 都可以使用四条区域日内曲线：`work_weekday`、`work_weekend`、
`nonwork_weekday`、`nonwork_weekend`。曲线决定一天内的名义请求分布；原始数据集
决定请求内容、token 分布、会话结构和因果依赖。

### `contract.py`

定义 schema、精确字段集合、数据类、标准序列化、bundle 哈希、类型转换和跨文件
校验。该模块只依赖 Python 标准库，可同时供 adapter 和 runtime 使用。

### `loader.py`

`load_workload_bundle(path)` 读取并完整校验一个 bundle，返回
`WorkloadGenerationResult`，其中包含：

- `manifest`
- `regions`
- `gateways`
- `session_requests`
- `timebin_demand`
- `summary`

`materialize_workload_bundle(ref, base_dir)` 负责解析 `WorkloadBundleRef`，随后调用
同一个 loader。

### `upsample.py`

`stream_upsampled_sessions(workload, factor, seed=...)` 以流式方式复制 session。
它重新生成 session ID、request ID 和前驱 ID；对整个 session 的名义时间线施加
一次确定性的 bin 内偏移；保持 turn 顺序和 `inter_turn_gap_s` 不变。

## Bundle 目录接口

一个标准 bundle 必须且只能包含六个文件：

| 文件 | 格式 | 用途 |
|---|---|---|
| `workload_manifest.json` | JSON 对象 | Bundle 标识、schema、时间范围、来源和完整性哈希 |
| `regions.json` | JSON 数组 | 用户区域坐标和需求权重 |
| `gateways.json` | JSON 数组 | 地面站坐标及其服务区域 |
| `session_requests.jsonl` | JSONL | 请求、session、token 和因果依赖记录 |
| `timebin_demand.csv` | CSV | 名义请求曲线聚合 |
| `workload_summary.json` | JSON 对象 | Adapter 生成的 workload 统计摘要 |

所有 JSON 使用 UTF-8。标准 JSON 采用 key 排序、紧凑分隔符、一个结尾换行、有限
数值以及最多六位小数。

## Schema

| `schema_version` | 用途 |
|---|---|
| `orbitserve.workload` | 使用 24 个基础请求字段的绝对到达 workload |
| `orbitserve.workload.causal.v1` | 在 24 个基础字段上增加四个依赖/溯源字段的完成相对 workload |

新的多轮 ChatLLM 和 Agent adapter 应输出 `orbitserve.workload.causal.v1`。

## `workload_manifest.json`

通用字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | string | 上述受支持 schema 之一 |
| `canonicalization` | string | `orbitserve-json-c14n` |
| `workload_id` | string | 稳定的 bundle 标识 |
| `seed` | integer | 基础确定性随机种子 |
| `time_horizon_s` | number | 名义 workload 时间范围（秒） |
| `bundle_sha256` | string | 五个数据文件与 `workload_id` 的组合哈希 |

adapter 可以增加来源数据集、profile pattern、到达语义和 token 复用语义等信息。

## `regions.json`

每行严格包含五个字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `region_id` | string | 请求引用的唯一地区 ID |
| `region_name` | string | 可读名称 |
| `lat_deg` | number | 代表纬度 |
| `lon_deg` | number | 代表经度 |
| `demand_weight` | number | 该地区在 bundle 中的需求占比 |

## `gateways.json`

每行严格包含四个字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `gateway_id` | string | 唯一地面站 ID |
| `lat_deg` | number | 地面站纬度 |
| `lon_deg` | number | 地面站经度 |
| `served_region_ids` | string 数组 | 可由该地面站服务的地区 |

## `session_requests.jsonl`

### 基础字段

| 分组 | 字段 | 类型 | 含义 |
|---|---|---|---|
| 标识 | `request_id` | string | 全局唯一请求 ID |
| 时间 | `arrival_time_s` | number | 绝对 schema 的释放时间；causal schema 的名义曲线时间 |
| 地理 | `user_region_id` | string | `regions.json` 中的地区 |
| 接入 | `access_mode` | string | `direct_user_sgl` 或 `gateway_sgl` |
| 接入 | `ingress_id` | string | 接入地区或地面站 ID |
| 接入 | `gateway_id` | string 或 null | 通过地面站接入时的地面站 ID |
| 模型 | `model_family` | string | 请求的模型族 |
| Token | `prompt_tokens` | integer | 本次请求表示的完整模型输入 |
| Token | `new_user_tokens` | integer | 本次请求需要执行 prefill 的 token |
| Token | `context_tokens_before_turn` | integer | 本次请求可以复用 KV 的 prompt token |
| Token | `output_tokens` | integer | 本次请求 decode 的 token |
| 传输 | `input_bytes` | integer | 传输模型使用的请求 payload 字节数 |
| 传输 | `output_bytes` | integer | 传输模型使用的响应 payload 字节数 |
| 模型 | `context_window_tokens` | integer | 模型上下文窗口上限 |
| SLO | `slo_class` | string | workload 定义的服务等级 |
| SLO | `ttft_deadline_s` | number 或 null | 相对实际到达时间的 TTFT deadline |
| SLO | `e2e_deadline_s` | number | 相对实际到达时间的 E2E deadline |
| Session | `session_id` | string 或 null | 上下文链 ID；独立单轮请求为 null |
| Session | `turn_index` | integer | `session_id` 内从 0 开始的序号 |
| Session | `session_turn_count` | integer 或 null | 该 session 的请求行数 |
| Session | `inter_turn_gap_s` | number 或 null | 前驱完成到本次提交之间的 think time |
| Session | `session_start_time_s` | number 或 null | 该 session 第一轮的名义时间 |
| Session | `session_ttl_s` | number 或 null | Runtime KV/session TTL |
| 标识 | `seed` | integer | 每请求确定性随机种子 |

### Causal 字段

`orbitserve.workload.causal.v1` 额外严格包含四个字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `arrival_time_mode` | string | `absolute` 或 `after_dependency_completion` |
| `depends_on_request_id` | string 或 null | 完成后才能释放本请求的前驱请求 |
| `source_trace_time_s` | number | 用于溯源的原始请求开始偏移 |
| `source_api_time_s` | number | 用于溯源的原始请求观测时长 |

## Token 计算合同

每条请求满足：

```text
prompt_tokens
  = context_tokens_before_turn
  + new_user_tokens
```

`context_tokens_before_turn` 是 bundle 声明的可复用 KV 部分。KV 不驻留、过期、
被驱逐或位于其他设备时，runtime 可以降低实际命中量。`new_user_tokens` 表示本轮
prefill 工作量；在 Agent trace 中，它还可以包含无法复用的历史 token 或工具 token，
不只表示人类新输入。

对于带 hash block 的 Agent trace，adapter 根据已知 hash 重合计算可复用 token。
对于 ChatLLM 会话，adapter 声明前序对话上下文可复用；实际命中由 runtime 的部署
位置和 KV 策略决定。

## 到达时间与依赖合同

### 绝对请求

```text
arrival_time_mode = "absolute"
depends_on_request_id = null
inter_turn_gap_s = null
actual_arrival_time_s = arrival_time_s
```

### 前驱完成后释放的请求

```text
arrival_time_mode = "after_dependency_completion"

actual_arrival_time_s
  = completion_time_s(depends_on_request_id)
  + inter_turn_gap_s
```

`completion_time_s` 是仿真中的完整 E2E 完成时间，不是 TTFT。
`inter_turn_gap_s` 是非负的客户端、工具或 agent think time，即从前驱完成到当前
请求提交之间的时间。不同根请求可以并发执行。

causal 请求的 `arrival_time_s` 保留从日内曲线采样得到的名义时间，用于 bundle
检查、可视化和 `timebin_demand.csv`。Runtime 性能统计使用动态解析后的实际到达时间。

参考 adapter 的计算方式：

- SemiAnalysis Agent：优先使用源数据 `think_time`；缺失时计算
  `max(0, current.t - (previous.t + previous.api_time))`。
- ShareChat：计算 `current_user_timestamp - previous_assistant_timestamp`。

## Session 与引用约束

- Request ID 全局唯一。
- 同一个 session 中的 `turn_index` 从 0 连续递增。
- `session_turn_count` 等于该 session 的实际请求行数。
- Causal 前驱必须存在于同一 bundle，且不能指向自身。
- 依赖图必须无环。
- `user_region_id` 必须能在 `regions.json` 中解析。
- 非空 `gateway_id` 必须能在 `gateways.json` 中解析。
- `gateway_sgl` 要求 `ingress_id == gateway_id`。
- `direct_user_sgl` 要求 `ingress_id == user_region_id`。

## `timebin_demand.csv`

标准列：

```text
time_bin_start_s, time_bin_end_s, region_id, request_count,
prompt_tokens_total, new_user_tokens_total, output_tokens_total
```

adapter 按名义 `arrival_time_s` 和地区聚合；当前参考 adapter 使用 60 秒窗口。
对于 causal bundle，该文件描述输入需求曲线，不表示受性能影响的实际 runtime 释放时间。

## `workload_summary.json`

summary 由 adapter 定义，主要用于检查和展示。至少应包含 `workload_id`、
`request_count` 和 `session_count`；可以按数据集增加 percentile、源 trace 数、KV 复用率
和 pattern 信息。Runtime 正确性不能依赖可选 summary 字段。

## Loader 校验顺序

`load_workload_bundle` 依次执行：

1. 校验 schema 与 canonicalization；
2. 校验严格六文件集合和组合哈希；
3. 按 schema 校验精确字段集合；
4. 执行类型转换并拒绝 runtime 结果字段泄漏；
5. 校验 token 和 session 约束；
6. 校验 causal 与地理引用。

示例：

```python
from orbitserve.workload import load_workload_bundle

workload = load_workload_bundle("path/to/experiment_bundle_1x")
print(len(workload.session_requests))
```
