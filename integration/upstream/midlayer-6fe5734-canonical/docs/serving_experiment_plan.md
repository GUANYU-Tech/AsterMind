# Serving 投稿实验与多 Agent 分工

## 这一阶段要完成什么

这一阶段只做三件事：先使用三种硬件都已有完整结果的九个逻辑 case 完成实测精度对齐；在用户后续提供 C500 和 910B 数据后，再纳入目前只有 A800 结果的六个逻辑 case；推进 GLM-5.2 和 Kimi-K2.6 在三种硬件配置上的 projected Serving 主链，对尚未闭合的模型、并行或通信组合明确报告 unsupported。Analytical PD 只在大模型 lowering 和通信前置合同闭合后启动，本阶段暂不补采 PD 真机数据。

第一阶段固定为 Qwen3-4B TP1、Qwen3-8B TP1 和 Llama-3.1-8B TP1，各使用 QPS 4/16/64，共九个逻辑 case、27 个物理硬件 case；其中 case ID 沿用历史标签 `Llama-3.1-8B`，配置和三硬件 raw 中被严格校验的实际模型 basename 是 `Llama-3.1-8B-Instruct`。第二阶段固定为 Qwen3-32B TP2 和 Qwen3-30B-A3B TP2+EP2，各使用 QPS 1/4/16，共六个逻辑 case、最终 18 个物理硬件 case；目前只处理并保留已有的 6 个 A800 case，缺少的 12 个 C500/910B case 标记为 pending external data，而不是当前阶段失败。大模型固定为 GLM-5.2 和 Kimi-K2.6，不增加其他模型。所有真实数据均从现有 ShareGPT raw 采集 或用户后续提供的完整结果复用，本计划不默认安排硬件重采，也不把完整 ShareGPT raw 采集 拉回或作为日常 evaluation 输入。

截至 2026-08-22，第一阶段 27 个物理 case 已按 compact schema 发布到 `serving_dataset/sharegpt/`，统一 evaluator 已在精确提交 `2b7e0b9acd43cc08e76c1a54019cca44ab5c3e79` 上完成 CPU-only 全矩阵执行。27 个 case 的请求、实测 metrics 与当前预测合同均通过输入和 provenance 校验，但当前 prediction 全部在 production launch selector 处 fail loud：A800 的 9 个 cell 缺少实际动态 M 的 GEMM launch fact，C500 的 9 个 cell 缺少 `gather` Vector launch fact，910B 的 9 个 cell 缺少对应动态 prefill workload 的 FlashAttention launch fact。因此本轮得到的是 27/27 `unsupported_launch_coverage`，尚未产生可报告的 Serving TTFT/TPOT/E2E 精度数值；这不是数据转换失败，也不得用 heuristic launch 或旧 prediction 绕过。

对这三类阻塞的离线审计表明它们不能通过现有 LUT 静默补齐。A800 当前首先遇到 9 个不同的动态 M GEMM key，现行和归档 LUT 均无这些 key，且 Scheduler 批次会随预测时间变化，不能用旧 Scheduler 轨迹枚举值冒充当前运行的完整 key 集；C500 的旧 MACA 3.2.1 observation 支持一个明确的 `gather` launch 候选规则，但当前 production 合同是 MACA 3.7.2，且继续执行还会遇到 3.7.2 FlashAttention selector 整体未闭合；910B 的现有 FlashAttention facts 没有任何 `causal=true + paged_varlen + uniform` 组合，历史九 case Scheduler 轨迹估算出的数千个 prefill/decode exact key 也均未命中。下一步应优先接入可审计的 host tiler 或静态 launch 规则；确需硬件时只采 non-duration launch evidence，并先由用户批准范围。

## 实验到底需要哪些数据

完整 ShareGPT raw 采集（`profiler_output/serving/<hardware_id>/sharegpt_raw/`）不是本阶段的输入合同。当前本地该采集约 61 GiB；此前审计的 A800 Qwen3-4B、Llama-3.1-8B、Qwen3-32B 和 Qwen3-30B-A3B 共 12 个唯一 case 合计 19.812 GiB，若把历史 `_serving`、`_serving_inputs` 重复目录算入则约 29.04 GiB。该统计不包含本轮新增为第一阶段目标的 Qwen3-8B，第一阶段完整 inventory 必须单独给出其体积。上述 12 个 case 的核心文件合计只有约 0.46 MiB，Scheduler sidecar 合计约 6.01 MiB；主要空间来自完整 plugin trace、每 rank operator JSONL、nsys SQLite/rep、regen cache 和旧 prediction/LUT，这些不应全部复制到主仓实验环境。

当前仓库已经形成“当前代码独立生成 prediction、compact observation 只在 evaluation 对齐”的主路径：`run_sharegpt_compact_evaluation.py` 严格读取 27-cell experiment config 和 validated compact case，使用 canonical `requests.csv` 重新执行当前 pinned-vLLM Forward，并只在 Forward 返回后读取 `alignment/metrics.json`。历史 `comparison.json`、旧 prediction/LUT、regen cache 和完整 profiler timeline 均不进入该路径。`full eval --part serving` 仍是单请求 conformance gate，不能替代这条实测精度路径；Scheduler/operator/launch 的逐项 exact semantic bridge 也仍未闭合，当前报告必须明确标为 unsupported，而不能把 observation 存在误写为已经完成 exact alignment。

| 数据层 | 每个 case 需要的内容 | 用于哪些实验 | 是否需要从三台机器拉回 |
|---|---|---|---:|
| E2E 精度核心 | `meta.json`、`arrival_schedule.json`、可用时的 `plugin_trace.json_provenance.json`、`serving_metrics.json` | 第一阶段 27 个物理 case 的 TTFT/TPOT/E2E/throughput 精度；第二阶段先保留 6 个 A800 case并等待 12 个外部结果 | 是，只拉回已存在且通过身份审计的 case；缺失 provenance 不能伪造 |
| Scheduler 结构 | `plugin_trace.json_scheduler.jsonl` | admission、iteration、scheduled tokens、prefill/decode 结构 | 是，目标 case 全部拉回，体积很小 |
| Operator 结构 observation | signature dictionary，加按 rank/step/layer 保序的 signature ID 序列、关键 attrs 和 sequence hash | 当前可用 case 的算子类型、shape、次数和顺序 exact 对齐 | 拉无损压缩 observation，不拉重复展开的完整 operator JSONL |
| Launch observation | signature dictionary，加按 rank/step 保序的 Kernel family/name、grid、block、smem、regs、split/tile 与 sequence hash | 验证 production launch selector 是否与真实 runtime 一致 | 拉无 duration 的压缩 launch observation，不拉 profiler 数据库 |
| Execution timing observation | 每 step 的 phase、Scheduler→forward、forward 内 Host、device kernel span、inter-forward gap、同步和 idle interval，以及来源 checksum | 解释 TTFT/TPOT/E2E 误差并研究 Host 调度/Forward CPU 开销 | 能从 raw 可靠提取的目标 case 均保留；evaluation-only，不进入 Forward |
| 完整 raw archive | monolithic plugin trace、完整 operator JSONL、nsys/mcTracer/msprof 原始文件 | 复核摘要或新的深度诊断 | 不拉回，保留在原硬件机器 |
| 历史派生产物 | regen cache、`plugin_trace_regen*`、`plugin_trace_g2.json`、旧 prediction/LUT/comparison | 当前实验不使用 | 不拉回 |

### 请求输入与 Production Trace 统一

ShareGPT、Synthetic Trace、GLM-5.2 Production Trace 和 Kimi-K2.6 Production Trace 的物理请求输入统一使用 `requests.csv`。Canonical input schema 固定为 `Query Time Stamp [ms]`、`In Token [tokens]`、`Out Token [tokens]` 和 `Cached hit [tokens]`；同一 strict reader 将四列规范化为 RequestStream，未提供 request ID 时按源文件行号生成稳定 ID，源文件顺序决定相同 arrival time 请求的顺序。文件名使用 `requests.csv`，因为每行表示一个请求，整个 case 才表示 workload。

现有 GLM-5.2/Kimi-K2.6 CSV 还包含 `Normal Input Tokens [tokens]`、`Total Token [tokens]`、`Inference Time [ms]`、`Ttft [ms]` 和 `gen_speed [token/s]`。转换时使用前两列做冗余一致性校验，把四个请求输入列写入 `requests.csv`，把后三个逐请求线上真值写入 `alignment/metrics.json`，并通过稳定 request ID 对齐；不得把 `gen_speed` 静默解释成其他指标。转换、行数和逐字段校验全部通过后，active dataset 不保留原始九列文件，也不建立 `source_trace.csv`。Production metrics 先保留供后续使用，当前 GLM-5.2/Kimi-K2.6 实验不启用该对齐。

ShareGPT 的 `Query Time Stamp [ms]` 来自已有 arrival schedule，`In Token [tokens]` 来自该硬件 case 的输入或可证明的一致 provenance，`Out Token [tokens]` 表示本次性能 replay 使用的输出 token 数。`manifest.json` 必须逐字段记录请求来源，至少区分 `requested`、`observed_actual`、`scheduler_derived`、`assumed_zero`、`scheduler_equivalent_representative` 和 `unavailable`，因此不能把实际生成长度描述成 requested output。`Cached hit [tokens]` 继续表示 Forward 的逻辑 prefix 长度 H；只有 case contract 明确证明 H=0 时才能以 `assumed_zero` 写 0。历史 ShareGPT raw 无法观测 H，但独立 prompt length、actual output 与每个请求的累计 Scheduler token conservation 可以证明实际调度没有获得净 prefix-work reduction；此时允许以 `scheduler_equivalent_representative` 写性能等价代表值 0，并在 `cached_hit_value_evidence` 中明确记录 H 不可观测、net prefix-work reduction 为 0、固定推导规则、可用 KV block size、证据 source IDs 和完整 raw source set SHA256。该代表值只用于重放相同净调度工作量，不能在报告中描述为 observed logical cache hit、prefix cache disabled、actual cache reuse 为零或不存在 recompute。Converter 还必须验证历史请求顺序、真实 metrics 与可用 provenance 的对应关系，再规范化为稳定的行号 ID。

每个 ShareGPT 硬件 case 必须在 `serving_dataset/sharegpt/<hardware_id>/<model_id>/<case_id>/` 中物理保存自己的 `requests.csv`，单硬件精度评估只读取该目录内的文件，不要求同名 case 在三种硬件上的 arrival、input length 和 output length 完全一致，也不设置额外的中央 ShareGPT 参考副本。Writer 必须从最终写出的 canonical `requests.csv` 字节计算 `requests_sha256`，不得接受调用方声明的 workload hash；只有三硬件 `requests_sha256` 相同且模型、tokenizer、并行方式、dtype 和 Scheduler contract 也一致时，结果才能标记为严格的同 workload 横向比较。SHA256 不同时，各硬件仍可独立进行 prediction-versus-alignment 精度评估，但不得把结果解释为完全相同 workload 下的硬件性能比较。

Production Trace 位于 `serving_dataset/production/<model_id>/<case_id>/`，与 ShareGPT 一样显式使用 model 和 case 两层；ShareGPT 额外保留 hardware 层，因为请求和实测真值可能随硬件 case 不同。`serving_dataset/production/` 只保存严格转换后的请求 payload，并与其他 canonical Serving 数据一样由 Git LFS 跟踪；原始九列 CSV 不进入 canonical case。现有 Production 源未记录模型与运行 setting，因此不发布 alignment ground truth。

Production 九列 CSV 的 strict raw-to-compact converter 已归入 `dataset_pipeline`：它从同一份 immutable bytes 内部解析并绑定 source bytes/SHA，保持物理行顺序并校验两条 token 冗余恒等式。8 个 Production case 已完成请求转换、上游成功请求确认、逐行审计和 Git LFS 发布；由于源文件没有记录模型与运行 setting，Inference Time、TTFT 和 generation speed 不作为 alignment ground truth 发布。

第一阶段 27 个物理 case 已完成只读盘点。对 Qwen3-4B、Qwen3-8B 和 Llama-3.1-8B 的每个相同 QPS，三硬件各自 128 个请求的 arrival、执行 input length 和 actual output length 均逐项一致；A800 与 C500 的 raw request ID 可显式对应，910B 的 Scheduler request ID 与 metrics ID 属于不同命名空间，只能按 metrics 物理顺序和首次调度顺序建立 ordinal-derived 双射，并通过累计 scheduled tokens 与 actual output 反推 input 后全量校验。27 个 case 均具备 metrics、Scheduler、operator 和 device timeline raw，但都缺少部分 model/tokenizer/runtime/source-commit provenance，因此只能标为 provenance partial，不能冒充 strict provenance complete。910B 的 capture log 明确记录 prefix cache 开启，A800/C500 未保留同等开关证据；三硬件逐请求均满足累计 scheduled tokens 等于 prompt tokens 加 actual output tokens 减一，证明实际调度没有获得净 prefix-work reduction，但不能据此声称逻辑 prefix 长度 H=0 或不存在 prefix/recompute 抵消。该结论不要求整体重采；缺失 observation 按字段或文件标记 unavailable，`Cached hit [tokens]` 使用带结构化证据的 `0 / scheduler_equivalent_representative`。现有一致性只证明本轮 request-shape sequence 相等，不等于 schema 强制未来三硬件使用相同请求，也不能替代 model revision、tokenizer、dtype 和 runtime identity 校验。

Profiler observation 的可用边界按 raw 证据分别表达：A800 只消费 case 根目录的 nsys SQLite，9 个 case 的 launch coverage 为 complete，timing 因跨时钟与边界证据有限为 partial；C500 通过 native typed tracer event 解析保留 9 个 case 的 partial launch/timing observation，不再依靠 kernel-name substring 猜测 memcpy 与 compute；910B 当前 raw 未提供可追溯的 Scheduler-to-TASK 时钟映射及误差，launch/timing 为 unavailable。后三者不影响 requests、metrics、Scheduler 和可证明的 operator alignment 发布，也不能通过其他硬件数据补值。

### Compact serving case 保留什么

目录先按 workload 来源分为 `production/` 与 `sharegpt/`，两者都显式使用 model 和 case 两层。ShareGPT 在 model 层之前增加 hardware 层，每个物理硬件 case 仍保持自包含；Production 请求只保存一份，不为三个预测目标硬件重复复制。case 顶层只放 identity 和 Forward 输入，用于对齐的实测真值统一放入 `alignment/`。

```text
serving_dataset/
  production/<model_id>/<case_id>/
    manifest.json
    requests.csv
    dynamic_inputs.jsonl             # 有才创建
    alignment/
      metrics.json                   # 仅保留，当前不启用 Production 对齐
  sharegpt/<hardware_id>/<model_id>/<case_id>/
    manifest.json
    requests.csv
    dynamic_inputs.jsonl             # 有才创建
    alignment/
      metrics.json
      scheduler.jsonl
      operators.jsonl
      launches.jsonl
      timing.jsonl
```

同一叶子目录始终表示一个 case。身份和结构化 case 指标使用 JSON，其中 `metrics.json` 明确包含逐请求记录与 case summary 两部分；按 step、operator 或 launch 重复的事件使用 JSONL；请求表使用 CSV。字段 schema 已作为 compact schema v1 实现并由 repository gate 校验，后续不得另建同义 schema。

| Compact 文件 | 保留的信息 | 是否进入 Forward |
|---|---|---:|
| `manifest.json` | schema version、case ID、hardware ID、model ID/revision、TP/EP/PP、dtype、Scheduler/runtime 关键参数、QPS、seed、canonical `requests.csv` 文件 SHA256、逐字段 provenance、结构化 request-ID mapping、dynamic-input provenance、alignment availability，以及每个 raw source 的唯一 archive ID、tool/version、相对文件名、bytes 和 SHA256 | 只读取运行身份和配置引用，不读取 alignment 真值 |
| `requests.csv` | 四个 canonical 请求列：arrival、input tokens、output tokens、cached-hit tokens | 是 |
| `dynamic_inputs.jsonl` | 可获得时保存影响实际执行的逐 step 动态输入，例如 MoE expert-token histogram 和 MTP proposed/accepted count；每条记录带 request/step/layer identity 与 `observed` 或 `generated` provenance | 是；缺失时只能使用显式配置的确定性生成规则，不能静默补值 |
| `alignment/metrics.json` | ShareGPT 保存逐请求 TTFT、TPOT、E2E、实际 output-token 数和 case 汇总；Production 保存原 CSV 中逐请求 `Inference Time [ms]`、`Ttft [ms]`、`gen_speed [token/s]` 及其明确单位和源字段 | 否；Production 当前只保留该文件，不启用对齐 |
| `alignment/scheduler.jsonl` | iteration ID、相对时间、scheduled request/tokens、total tokens 和 raw 中实际存在的 finished、preempted、recompute 状态 | 否 |
| `alignment/operators.jsonl` | rank/step/phase/layer、canonical operator signature、attrs 和保序 identity | 否 |
| `alignment/launches.jsonl` | rank/step/order、Kernel family/name、grid、block、smem、regs、tile/split 和保序 identity，不含 duration | 否 |
| `alignment/timing.jsonl` | 每 step phase、Scheduler/forward 边界、Host interval、device kernel span、launch/inter-forward gap、同步和 idle interval | 否 |

### Compact schema v1

所有 case 文件使用 `schema_version: 1`。`requests.csv` 的物理数据行按零起始编号为 `request_idx`，运行时 request ID 固定生成为 `request-{request_idx:012d}`；所有 alignment 和 dynamic-input 文件只用 `request_idx` 关联，不依赖删除后的原始 CSV 行号或原始业务 ID。`step_id` 是 converter 生成的 case 内稳定 ID，raw iteration 另存为 `source_iter_id`；`operator_id` 和 `launch_id` 分别由 rank、step 和 sequence index 生成，只要求在本 case 内唯一。时间字段必须在名称中写明单位，禁止裸 `timestamp` 或 `duration`；SHA256 必须是小写 64 位十六进制字符串。

`manifest.json` 顶层固定包含 `schema_version`、`identity`、`runtime`、`requests`、`dynamic_inputs`、`alignment`、`raw_sources`、`provenance` 和 `converter`。`identity` 包含 source kind、case ID、hardware ID、model ID/revision 和 tokenizer revision；Production 的 hardware ID 可以为 `null`。`runtime` 包含 dtype、deployment、TP/EP/PP、target QPS、seed、Scheduler config SHA、runtime version 和 source commit，其中历史证据无法证明的 dtype、deployment、parallel size、seed、version 或 commit 必须显式为 `null`，不能猜填；ShareGPT case 的 target QPS 必须存在且为正，hardware-independent Production Trace 没有配置目标 QPS 时必须将其写为 `null`，不能由观测 arrival 或 metrics 反推。`requests` 包含固定相对路径、行数、由 writer 对 canonical `requests.csv` 字节计算且不可由调用方声明的 `requests_sha256`、ID normalization、相同 arrival 的排序规则、结构化 source-ID mapping 以及四个字段各自的 provenance；910B 当前 ordinal-derived mapping 必须显式记录 source/target namespace、evidence source 和原因。`dynamic_inputs` 和每种 `alignment` entry 固定包含 status、coverage、reason、path、SHA 和 `source_archive_ids`；`raw_sources` 记录全 case 唯一的 archive ID、role、相对路径、bytes、SHA 以及该 source 自己的 tool/tool version，tool/version 未知时显式为 `null`，不得用 case 级单一 capture tool 冒充多 source provenance。`provenance` 使用 `complete | partial` 与结构化 gaps 精确覆盖 identity、runtime 和 raw-source tool provenance 中的所有 null 字段；`converter` 记录实现 ID 和 source commit。

文件级 status 只能是 `available`、`unavailable` 或 `not_applicable`，coverage 只能是 `complete`、`partial` 或 `null`。`available` 必须给出 path、SHA 和固定存在的 `source_archive_ids`；Scheduler、operator、launch、metrics 和 timing observation 必须至少绑定一个 manifest raw source，且每个 ID 都必须与同一 manifest 的 raw source 成对校验。`unavailable` 必须给出非空 reason；不存在的文件不能用空 payload 代替。Optional key 始终存在并以 `null` 表达不适用或不可恢复。请求字段 provenance kind 只能是 `requested`、`observed_actual`、`scheduler_derived`、`assumed_zero`、`scheduler_equivalent_representative` 或 `unavailable`；四个实际写入 CSV 的字段不能是 `unavailable`。`assumed_zero` 必须绑定证明逻辑 H=0 的 `cached_hit_value_evidence`；`scheduler_equivalent_representative` 必须绑定证明净 prefix-work reduction 为 0 而 H 不可观测的 evidence，二者不得混用。

`requests.csv` 严格只含以下四列且顺序固定：

```csv
Query Time Stamp [ms],In Token [tokens],Out Token [tokens],Cached hit [tokens]
```

到达时间必须有限且非负，input tokens 必须为正整数，output tokens 必须为非负整数，cached-hit tokens 必须满足 `0 <= cached hit <= input`。CSV 不保存 request ID、prompt、token ID 或原始九列；`request_idx` 是写入后 CSV 的物理数据行号，不是按 arrival 重新排序后的序号。

`dynamic_inputs.jsonl` 每行固定包含 `schema_version`、`record_id`、`record_kind`、writer 注入的 `requests_sha256`、`step_id`、`request_idxs`、`layer_idx`、`sequence_idx`、`schedule_step_sha256`、`provenance`、`moe` 和 `mtp`。`record_kind` 只能是 `moe_expert_histogram` 或 `mtp_acceptance`；前者的 `moe` 包含 total tokens 和 `tokens_per_expert`，后者的 `mtp` 包含 proposed、verified 和 accepted token counts，另一 payload 必须为 `null`。`provenance.kind` 只能是 `observed` 或 `generated`，并记录 source ID、source SHA、algorithm、seed 和 policy-descriptor SHA；generated record 的 source SHA 由 writer 注入为同 case 的 `requests_sha256`，调用方不得声明。`request_idxs: []` 只表示 raw 能证明 batch/step histogram 但无法恢复单请求归属，此时 Forward 只能按 exact step 消费，不能猜测归属。Dynamic input 是 Forward 输入，不依赖 evaluation-only Scheduler alignment 存在；`schedule_step_sha256` 保留为 Forward 运行时匹配键，有 Scheduler alignment 时 validator 必须交叉验证，无 Scheduler alignment 时依靠 record provenance 和请求 SHA 绑定。动态输入不得包含 duration。

`alignment/metrics.json` 固定包含 `schema_version`、writer 注入的 `requests_sha256`、`metric_provenance`、`per_request` 和 `summary`；Production metrics 还必须包含字段级 `field_coverage`，分别记录 inference time、TTFT 和 generation speed 的 `complete | partial` 状态、不可用请求数和规范化原因。每个 per-request entry 包含 `request_idx`、actual output tokens、TTFT、TPOT、E2E latency、inference time 和 generation speed；每个可选数值字段都必须存在，未知时为 `null`。summary 包含 request count、request throughput、output-token throughput 和 total-token throughput。Production 只从原始列填 `inference_time_ms`、`ttft_ms` 和 `generation_speed_tokens_per_s`，不得把 Inference Time 改名为 E2E，也不得从 generation speed 倒推 TPOT；即使个别字段不可用，metrics 仍必须为每个 canonical request 保留一条记录，只能在对应字段中写入带固定原因的 `null`。ShareGPT 按 raw 证据填 TTFT、TPOT、E2E 和 actual output tokens；coverage 为 `complete` 时 E2E 与 actual output 必须存在，actual output 为 0 时 TTFT/TPOT 必须为 `null`，为 1 时 TTFT 必须有限且 TPOT 固定为 `0.0`，大于 1 时 TTFT/TPOT 都必须有限。

`alignment/scheduler.jsonl` 每行表示一个真实 Scheduler iteration，固定包含 schema version、event/step/source-iteration ID、相对 Scheduler 起点的微秒时间、pool、total scheduled tokens、scheduled requests、finished request indexes、preempted request indexes 和 field availability。每个 scheduled request 包含 request index、scheduled tokens、phase、computed tokens、recomputed tokens 和 prefix-replay tokens，同一步内 request index 不得重复。raw 没有某字段时值必须为 `null` 且 availability 为 `unavailable`；只有 raw 明确记录集合为空时才能写 `[]`。

`alignment/operators.jsonl` 使用 signature 与 event 两类记录。Signature 保存 canonical operator kind、shape、dtype、variant 和 attrs，signature ID 是其 canonical JSON SHA256；历史 raw 未记录 dtype 时必须写 `null` 并由 manifest provenance gap 说明，不能写字符串 `"unavailable"` 或从其他硬件猜填。event 保存 operator ID、signature ID、rank、step、phase、layer、sequence index、request indexes 和 `exact | step_only | unavailable` link status；`(rank, step_id, sequence_idx)` 必须唯一，同一 rank/step 的 sequence index 按 JSONL 顺序从 0 严格连续递增。Complete coverage 只要求覆盖 `scheduled_requests` 非空的 work-bearing Scheduler step，idle step 不要求 operator event。Operator alignment 不保存 duration。

`alignment/launches.jsonl` 同样使用 signature 与 event 两类记录。Signature 保存 Kernel name/family、geometry kind、GPU grid/block 或 Ascend block/mix-block dim、shared-memory bytes、registers per thread、tile 和 split；event 保存 launch ID、signature ID、rank、step、sequence index、可证明的 operator IDs 和 `exact | step_order_only | unavailable` link status。`(rank, step_id, sequence_idx)` 必须唯一且同一 rank/step 从 0 严格连续递增；complete coverage 同样只要求 work-bearing Scheduler step。A800/C500 使用 `cuda_dim3`；910B 使用 `ascend_block_dim`，但 raw TASK 的 `blockDim=0` 表示 geometry 未提供，必须保存为 `geometry_kind="unavailable"` 且四个 geometry 字段均为 `null`，同时 launch coverage 必须为 `partial`，不得把 0 写成真实 Ascend block dim 或伪造 CUDA grid/block。Launch alignment 不保存 duration。

`alignment/timing.jsonl` 每行固定包含 schema version、interval ID/kind、rank、step ID、launch ID、phase、start/end/duration ns、clock domain、time origin、source archive ID 和 correlation。Interval kind 只能来自 Scheduler-to-forward、forward、host、device-kernel、device-copy、inter-forward-gap、synchronize 或 idle；`duration_ns` 必须等于 `end_ns - start_ns`。`device_kernel` 必须绑定 launch ID，C500 memcpy 等 `device_copy` 必须令 launch ID 为 `null`，source-specific adapter 必须按 raw event type 明确分类，禁止用 kernel-name substring 泛化。不同 clock domain 不得直接相减，存在经过验证的映射时必须在 manifest 记录 mapping source 和误差。相同 Kernel name或时间邻近不足以建立 launch ID；不能证明一一关联时 correlation 必须为 `unavailable`。Timing 只能由 raw 直接转换，不能用已经丢弃 end/duration 的 compatibility loader 冒充完整数据。

`manifest.json` 对每种 alignment 数据使用 `available`、`unavailable` 或 `not_applicable` 三态；`unavailable` 必须带原因。缺少 raw 证据时不创建空 alignment 文件，也不能从其他硬件或相邻 case 补值。所有 duration 的单位和时钟域必须显式记录。Production `src/serving` 只允许读取 `requests.csv`、可用的 `dynamic_inputs.jsonl` 和正常的模型、部署、Scheduler、HWSpec 配置，不得扫描或导入 `alignment/`。Prefix-cache 的逐请求命中 token 数归 `requests.csv`；aggregate 指标归 `alignment/metrics.json`，不再建立含义混杂的 `runtime.json`。

Production Trace 与 ShareGPT compact dataset 共用 `serving_dataset/` 根目录、版本化 schema、Forward/evaluation 边界和 Git LFS 跟踪策略；当前 canonical Production case 保存四列 `requests.csv`、manifest 与 `alignment/metrics.json`。

### Compact serving case 不保留什么

| 不进入 compact 的信息 | 处理方式 | 为什么主实验不需要 |
|---|---|---|
| Prompt 文本、token IDs、生成文本和 Production 原始九列文件 | 当前 compact schema 不保存；九列文件用于生成 `requests.csv` 和 `alignment/metrics.json`，转换审计后保留 source SHA | Serving 时间模型只需要 token 数、到达时间和 prefix-hit 数；线上指标已有独立 alignment 文件 |
| 绝对业务时间、hostname、PID、用户 ID 和原始 request ID | 只保留相对时间和重新编号的 canonical request ID；必要 provenance 使用非敏感 archive ID | 调度只依赖时间间隔和请求对应关系；机器身份由 hardware/tool provenance 表达，不需要暴露运营标识 |
| Monolithic `plugin_trace.json` | 保留在远端 raw archive | 其请求、Scheduler 和 operator 信息已分别由 compact 文件表达，完整文件体积大且高度重复 |
| 完整 `plugin_trace.json_operators.jsonl.rank*` | 远端保留，compact 保存去重 signature 与只引用 signature ID 的保序 event | 重复 attrs 只保存一次；compact event 保留结构对齐所需的 rank/step/order，不复制完整 raw payload |
| `nsys_profile.nsys-rep`、`nsys_profile.sqlite`、mcTracer/msprof raw | 远端保留；从可解析 case 生成 `alignment/timing.jsonl` | E2E 精度和 Host attribution 使用 compact alignment 数据；完整 profiler 数据库只在边界恢复失败、Kernel attribution 或同步诊断时需要 |
| Per-invocation 硬件 counters 和完整 launch raw | 留在 profiler raw 或 canonical Kernel dataset/launch data；可证明的 device interval 只以 evaluation-only timing record 保留 | Serving Forward 应调用统一 Kernel predictor；真实 duration 只能用于 alignment，不能进入 Serving 输入或替代单 Kernel predictor |
| Token→expert 的逐 token 映射、token 内容与完整 MTP token 序列 | 原始 observation 留在远端；`dynamic_inputs.jsonl` 只保存性能建模所需的每 step expert-token histogram 和 proposed/accepted count | Histogram/count 足以让 Forward 使用与实测相同的动态 workload，并对齐 routed rows、负载不均衡和 Scheduler 接受行为，无需保存敏感 token 内容 |
| `prediction*.json`、`prediction_lut*.json`、`comparison.json` | 不拉回，使用当前 commit 重新生成到 `experiment_logs/` | 它们是旧代码的派生结果，不是真值；复用会混淆版本并可能把旧拟合或 fallback 带回主结果 |
| `.regen_cache_*`、`plugin_trace_regen*`、`plugin_trace_g2.json` | 不拉回，远端归档 | 它们是旧工作流缓存或重复派生物，可由 raw/当前代码重新生成，不属于可复现输入合同 |

E2E 精度数值本身可以只由 RequestStream、`alignment/metrics.json` 和 prediction 计算，但若要解释误差并判断现有 Host 机制是否属于重要误差来源，`alignment/timing.jsonl` 必须尽量覆盖第一阶段 27 个物理 case；第二阶段在新增数据到齐后按同一规则扩展。目标 case 的 timing alignment 只能决定 evaluation coverage、误差归因和 unsupported 边界，不能用来选择 Forward 公式、参数、阈值或分段；任何进入 Forward 的 Host 参数仍只能来自独立机制证据。无法从某硬件 raw 中可靠恢复 Host/Device 时钟映射或 forward 边界时必须标记 unavailable，不能用另一硬件或相邻 case 代替。完整 profiler raw 始终不进入 production Forward。

小模型 E2E 数值比较只依赖 E2E 核心和 Scheduler sidecar，结构、launch、runtime behavior 与 Host/Device attribution 使用由同一 raw 生成的 compact observations。完整 operator trace 在目标 A800 12 case 中约 2.67 GiB，nsys SQLite/rep 约 5.98 GiB，monolithic plugin trace 约 3.00 GiB，regen cache/regen trace 约 5.92 GiB；它们不应因为运行精度表而被复制三份。Compact observations 保存 raw checksum 和 converter 版本，完整 raw 继续留在远端。

GLM-5.2/Kimi-K2.6 projected 实验需要本地 `requests.csv`、模型 config/provenance、HWSpec、部署和通信配置，以及显式的 `dynamic_inputs.jsonl` 或版本化确定性生成规则，不需要任何 ShareGPT raw 采集。Production 的 `alignment/metrics.json` 保留已有线上逐请求真值，但当前实验不消费它做对齐；这不等于数据没有真值。MTP 只有平均接受率时，dynamic input 记录与 writer 计算的 `requests_sha256` 绑定，并记录生成规则和 seed；有逐 step proposed/accepted 数据时可以直接作为 Forward 条件输入，但不得携带 duration。Analytical PD 同样不需要 ShareGPT raw 采集 真值。Agent 5 的单 Kernel 精度任务只使用 canonical `kernel_dataset`、`launch_data`、HWSpec 和合法 microbenchmark evidence，也不依赖 ShareGPT raw 采集。

最终 ShareGPT compact 目录为 `serving_dataset/sharegpt/<hardware_id>/<model_id>/<case_id>/`，每个硬件 case 都保存自己的 requests、可用 dynamic inputs、alignment 数据和 raw checksums；不通过链接共享 requests，也不因另一个硬件已有同名 case 而省略本地副本。Production 目录为 `serving_dataset/production/<model_id>/<case_id>/`，当前只保存 requests 和 manifest，manifest 将缺少 setting 的 alignment 显式标为 unavailable；这些 compact payload 由 Git LFS 跟踪。完整 ShareGPT raw 采集 继续作为各硬件机器上的 ignored raw archive，不删除、不改写、不纳入 Git/LFS。

新的第一阶段主评估已只消费 compact dataset；部分 legacy evaluation 仍可通过 `historical_alignment_root` 读取 ShareGPT raw 采集，trace-alignment 深度诊断路径也仍会访问 `nsys_profile.sqlite`。因此不能仅因 compact 主路径已闭合就删除本地 raw；compact manifest 保存的 raw checksum 用于回溯，是否归档或删除原始 A800 数据仍由用户决定。

## Phase 0：执行前冻结

任何 writer worktree 开始前，主 Agent 必须先完成以下合同冻结，并形成一个包含本计划及相关规范修改的本地 baseline commit：

- 更新 `AGENTS.md`、`README.md`、`.gitignore` 和 `.gitattributes`：`serving_dataset/sharegpt/` 与 `serving_dataset/production/` 的 canonical CSV、JSON、JSONL 均使用 Git LFS；Production 原始九列输入不进入仓库。
- 冻结上述目录、文件名、四列 CSV header、schema version、单位、workload 字段 provenance、request-ID normalization 和 observation availability 三态；同一含义不得再建立第二套 schema。
- 冻结第一阶段九个逻辑 case，并让 active ShareGPT evaluation 配置包含 Qwen3-4B、Qwen3-8B 和 Llama-3.1-8B 的 QPS 4/16/64；Qwen3-32B 与 Qwen3-30B-A3B 的 A800 case 保留但不混入第一阶段三硬件结果，等待另外两种硬件数据到齐后再启用第二阶段矩阵。
- 冻结 writer owner：Serving reader 及 RequestStream normalization 归 `src/serving/request_gen/`；raw-to-compact extraction 归 `dataset_pipeline/`，不得 import `src.serving` 或 `evaluation`，也不得复制 predictor/RequestStream 实现；跨 owner 一致性只在测试中通过公开文件合同验证。
- 把当前 dirty 文档状态整理为可审计的本地 commit 后再派生 worktree；Phase 0 不移动、转换或发布任何数据。

## 一共几个 Agent

一共 7 个 Agent：1 个主 Agent 和 6 个任务 Agent。单个 session 最多同时运行 5 个线程，因此任务分两波进入；修改同一 contract 的 Agent 不同时写代码。Agent 5 独立负责 Kernel 精度，可以与 Serving 开发并行，但只能在主 Agent 设定的 checkpoint 合入，不能让 Serving evaluation 的 predictor baseline 在一次实验中途变化。Agent 6 独立负责 ProtoBridge 更新与接口集成，不负责上游算法、硬件 profile 或校准。

| Agent | 负责的功能 | 是否修改代码 | 最终产出 |
|---|---|---:|---|
| 主 Agent | 冻结公共 contract、同步远端数据、审阅合并、运行总门禁、生成最终结论 | 是 | 集成后的 main、本阶段完整结果、问题与 unsupported 清单 |
| Agent 1：数据审计 | 盘点 A800/C500/910B 已有 Serving 数据并确认每个硬件 case 的输入与 observation 身份 | 否 | 第一阶段 27 个物理 case 的完整 inventory，以及第二阶段 6 个 A800 case 和 12 个 pending external-data cell 的独立清单 |
| Agent 2：Serving 主链 | 统一 Trace 输入，并完成六个模型从请求到 Scheduler、算子与 KernelPlan 的分析生成 | 是 | 一个本地 commit、统一 RequestStream、六模型支持、主链测试 |
| Agent 3：精度与报告 | 使用现有真值运行小模型精度实验，并运行大模型 projected 实验和生成图表数据 | 是 | 一个本地 commit、实验 manifest、当前可用小模型精度表、大模型 projected 表、聚合 CSV/JSON |
| Agent 4：Analytical PD | 在统一主链完成后实现和运行 PD 分离分析，不做真机采集 | 是 | 一个本地 commit、PD projected 结果、KV transfer/排队/利用率报告 |
| Agent 5：Kernel 精度 | 独立持续优化三硬件单 Kernel 解析预测精度，并维护 Serving 使用的统一 predictor | 是 | 阶段性本地 commits、三硬件 full kernel eval、物理证据与误差归因、改动前后对比 |
| Agent 6：ProtoBridge 集成 | 跟踪用户指定的上游 commit，更新 verbatim snapshot、provenance、Midlayer adapter 和接口测试 | 是 | 一个本地 commit、上游差异与 SHA 记录、adapter 兼容性结果、需要上游作者处理的需求清单 |

## 主 Agent

### 负责什么

- 选择干净的 baseline commit，并冻结模型、并行方式、Trace 字段和结果 schema。
- 为 Agent 2、3、4 创建独立 worktree；Agent 1 是只读任务，不创建 worktree。
- 根据 Agent 1 的审计，在 Agent 2 converter 合入后把同一精确 commit 或审计 snapshot 部署到 C500 和 910B，仅以 CPU 串行读取现有 raw 并在 `profiler_output/serving/<hardware_id>/` 生成 candidate；拉回通过 checksum 与 completeness 初审的 compact candidate，不拉完整 ShareGPT raw 采集，不使用 GPU，不重采，不让远程执行任务直接改 canonical 数据。A800 也先生成 candidate，由 dataset auditor 复核后才发布到 `serving_dataset/`。
- 在 Production converter 通过测试后，把现有顶层 `production_workloads/` 逐 case 转换到 Git LFS 跟踪的 `serving_dataset/production/<model_id>/<case_id>/`：四个请求输入列写入 `requests.csv`，冗余 token 列只做一致性验证；由于三个线上指标缺少对应模型与运行 setting，不发布为 alignment ground truth。该迁移现已完成，canonical case 通过 loader 与逐行审计，旧顶层源目录已移出工作树。
- 审阅并按 Agent 2→Agent 3→Agent 4 的依赖顺序集成 commit。
- 运行仓库门禁和最终全部实验，确认 measured 与 projected 没有混淆。
- 如果发现 Kernel 精度问题，先完成归因并向用户报告；不能为了改善 Serving 误差直接调 Forward。
- 为 Agent 5 冻结单 Kernel baseline，审阅其参数来源和物理证据，并只在完整 kernel gate 通过后形成阶段性 predictor checkpoint。
- 每次 predictor checkpoint 合入后重新运行 Agent 3 的全部第一阶段 Serving evaluation，不能混用不同 Kernel commit 的结果；第二阶段在用户提供另外两种硬件数据前不进入当前汇总。
- 为 Agent 6 指定精确 ProtoBridge 上游 commit，审阅 verbatim source/license SHA 和 adapter 变化；缺失的硬件 author profile 由主 Agent 报告给用户协调上游作者。

### 产出

- 按硬件独立记录 workload identity、observation provenance 和跨硬件可比性的数据 manifest。
- 集成后的代码和测试结果。
- 第一阶段 27 个物理 case 经逐项审计后的精度表，以及 provenance 不完整和 observation unavailable 清单；第二阶段单独记录 6 个已有 A800 case 与 12 个 pending external-data cell，不混入第一阶段结果。
- GLM-5.2、Kimi-K2.6 projected 结果。
- analytical PD 结果及“无 PD 真机真值”的明确说明。
- 未支持组合、数据冲突和后续需处理问题的列表。
- 每个 Serving 结果所使用的精确 Kernel predictor commit。
- 每个通信结果所使用的 ProtoBridge upstream commit、snapshot SHA 和 author-profile digest，或明确的 unsupported 原因。

## Agent 1：数据审计

### 负责什么

只读检查本地 A800 与远端 C500、910B 的现有 ShareGPT raw 采集（`profiler_output/serving/<hardware_id>/sharegpt_raw/`）。逐 case 核对 model、hardware、TP/EP/PP、dtype、QPS、请求数、seed、arrival、input/output tokens、canonical `requests.csv` 字节 SHA、runtime 版本、Serving metrics、Scheduler trace 和 operator trace，并分别统计 compact 必需文件与完整 raw 的体积。

第一阶段检查矩阵为：Qwen3-4B TP1、Qwen3-8B TP1 和 Llama-3.1-8B TP1，各使用 QPS 4/16/64，并分别检查 A800、C500、910B，共 9 个逻辑 case、27 个物理 case。第二阶段矩阵为 Qwen3-32B TP2 和 Qwen3-30B-A3B TP2+EP2，各使用 QPS 1/4/16；当前只登记 6 个 A800 case，C500/910B 的 12 个 cell 标记 pending external data，不在本轮重复搜索、推测或重采。

### 不负责什么

- 不修改代码或数据。
- 不启动 GPU，不运行 Serving，不重采 profiler。
- 不根据目录名推断 case 正确，不把不一致数据改名后纳入。
- 不拉回或覆盖 canonical 数据，文件同步由主 Agent 完成。

### 产出

- 第一阶段一张 27 行 physical-case inventory，逐项标记 usable、provenance incomplete、observation unavailable 或 identity mismatch；第二阶段另列 18 行状态，其中 6 行是已有 A800 case，12 行是 pending external data。
- 各硬件 case 自身由 canonical `requests.csv` 字节计算的 `requests_sha256`，以及同名 case 是否具备严格跨硬件可比性的结论。
- 每个可用 case 的真实路径、大小、checksum 和 provenance 摘要。
- 主 Agent 应拉回的精确文件列表、Operator 摘要生成输入和不应拉回的 raw/旧派生文件列表。

## Agent 2：Serving 主链

### Worktree 与文件范围

Worktree 建议为 `midlayer-wt-serving-core`。第一交付只负责 `src/serving/request_gen/`、Serving compact conversion 所需的 `dataset_pipeline/` leaf 及其对应测试，不修改 evaluation、canonical 数据或生产 CSV。该交付合入后，若继续 GLM/Kimi 主链，再从新 baseline 单独分配 `src/serving/workload/` 与 `src/serving/kernel_lowering/`，不能把数据合同与大模型 lowering 混成一个不可审阅提交。

### 负责什么

- 复用当前已有的 Production Trace strict reader，让 Production、ShareGPT 和 Synthetic Trace 生成相同四列 canonical `requests.csv`，并通过该 reader 归一化为 RequestStream；不建立第二套 parser。
- ShareGPT 从每个硬件 case 的已有 arrival、metrics、可用 provenance 和 Scheduler evidence 重建输入，不能读取历史 operator trace 作为模拟器算子输入。910B 的 ordinal-derived 映射、observed-actual output 和 cache-hit assumption 必须显式写入 manifest。
- 第一交付提供 CPU-only normalized-payload candidate writer 和 strict validator；writer 只接收已经规范化的内存 payload，不读取或删除 raw，也不冒充 source-specific raw converter。Production API 只允许写入仓库 `profiler_output/serving/<hardware_id>/...`，hardware-independent Production Trace 使用保留 scope `production`，不暴露任意 candidate-root 参数；测试只能 monkeypatch 模块内 candidate-root 常量到隔离目录。后续再由每种真实 raw 格式的 adapter 生成 Forward 可读且带明确 provenance 的 `dynamic_inputs.jsonl`，以及位于 `alignment/` 下的 metrics、Scheduler、operator、launch、execution timing 和 raw checksum provenance；adapter 不生成预测值，alignment 中所有 duration 字段均标记为 evaluation-only 且不能被 production Forward import。`dataset_pipeline` 实现只处理冻结的文件 schema，不 import `src.serving` 或 `evaluation`。
- 在第一交付完成后，单独审计并扩展 Qwen3-4B、Qwen3-8B 和 Llama-3.1-8B 的真实 Scheduler→logical workload→KernelPlan 主链；已有能力必须复用，不能重写。Qwen3-32B 与 Qwen3-30B-A3B 的代码能力可以继续闭合，但其跨硬件结果必须等待第二阶段数据。
- GLM-5.2 的 DSA/MoE/MTP 已在 colocated TP=EP=8/PP4 范围内通过固定 vLLM 0.11 CPU Scheduler 和 production runner、A800 projected launch、零拟合解析 predictor 到达统一时间线；MTP acceptance 是带明确 source ID/root seed 的可配置 projected assumption，并非 observation。这些新增 kernel 尚未做真机性能校准，且 DSA 上游 CUDA 要求 SM90/SM100，A800 结果不能解释为原生 vLLM kernel 实机支持。Kimi-K2.6 已在固定 TP=EP=8/PP2 路径通过相同主链，但其固定配置声明零个 MTP layer；KV offload 和三硬件实机精度继续显式 unsupported。
- Production Trace 缺少 expert ID 时复用当前已有的模型 top-k、uniform without replacement 和固定 seed contract，并把确定性生成结果写入 `dynamic_inputs.jsonl`、在 manifest 中记录规则和 seed；MTP 同样使用绑定 `requests_sha256` 的版本化生成合同，生成合同本身缺失时 fail loud。
- 所有算子由模型配置和 Scheduler 分析生成，不能回放真实 operator trace，不能读取真实 latency。

### 产出

- 一个可审阅的本地 commit。
- 统一 Trace/RequestStream、compact converter 和数据合同测试。
- 第一交付覆盖第一阶段三个小模型、九个逻辑 case 的 compact workload 与 observation 路径，并保证第二阶段相同 schema 可直接扩展。
- 第二交付才处理小模型主链缺口及 GLM-5.2、Kimi-K2.6 的完整 projected 主链；无法闭合的组合必须明确 unsupported。
- 每个模型一组端到端 contract tests，证明请求能够生成 Scheduler decision、logical operators 和 KernelPlan。

## Agent 3：精度与报告

### Worktree 与文件范围

Worktree 建议为 `midlayer-wt-serving-eval`。主要负责 `evaluation/serving/`、`scripts/evaluation/serving/`、实验配置和对应测试，不修改 production predictor 或 `src/serving/`。

### 负责什么

- 根据 Agent 1 的 inventory 和 compact serving candidate 生成第一阶段 27 个物理 case 的 experiment manifest；第二阶段 12 个待提供的 C500/910B cell 只进入状态清单，不进入本轮 evaluation manifest，也不要求完整 ShareGPT raw 采集 存在于本地。
- 对每个 cell 使用当前 production simulation 重新预测，不复用旧 prediction、旧 LUT 或旧 comparison。
- 按 request ID 对齐真实与预测的 TTFT、TPOT 和 E2E，并比较 request throughput 与 output-token throughput。
- 对所有通过 availability 与 provenance 审计的 case exact 比较 Scheduler iteration、scheduled tokens、prefill/decode 分类、保序 operator signature 和保序 launch signature；QPS 64 不因负载较高而自动排除，某项 raw 不能可靠映射时只把该 observation 标为 unavailable。
- 对具备可靠 execution timing observation 的全部目标 cell 分解 Scheduler→forward、forward 内 Host、device kernel span、inter-forward gap 与 idle time，分别报告 Prefill 和 Decode 的 observed/predicted/unsupported 边界。
- 第一阶段分别绘制 Qwen3-4B、Qwen3-8B 和 Llama-3.1-8B 在三硬件上的 QPS 4→16→64 负载曲线。Qwen3-32B/Qwen3-30B-A3B 的 QPS 1→4→16 曲线留到第二阶段；其中 Qwen3-32B 三个 A800 QPS 的历史 dtype 不完全一致，数据到齐后也不能把未统一 dtype 的结果解释为纯 QPS scaling。
- 在 Agent 2 大模型主链完成后，先运行 GLM-5.2、Kimi-K2.6 在三硬件上的 6 个 debug smoke，再运行 low/medium/high Production Trace 的 18 个 colocated projected cell。

### 产出

- 一个可审阅的本地 commit。
- 第一阶段最多 27 个物理 case 的小模型精度结果；第二阶段保持独立状态清单，等待数据到齐后再追加结果。
- 每个 hardware/model 的 TTFT、TPOT、E2E 和 throughput 负载曲线数据。
- 当前有完整 observation 的 Scheduler/算子/launch exact 对齐报告。
- 三硬件 Host/Device execution attribution 报告，逐项注明 observation available 或 unavailable。
- 6 个大模型 debug smoke 结果和 18 个正式 projected 结果。
- 可用于论文作图的聚合 CSV/JSON，以及所有失败或 unsupported cell 的清单。

## Agent 4：Analytical PD

### Worktree 与文件范围

Agent 4 不是本轮第一波任务。它必须同时等待 Agent 2 的大模型 lowering 与目标并行方式集成，以及 Agent 6 的 ProtoBridge Send/Recv、collective 和三硬件 author profile 闭合后才能开始写代码。届时 Worktree 建议为 `midlayer-wt-serving-pd`，负责 PD deployment、KV handoff/transfer、P/D queue、timeline 和对应 evaluation/test，不修改模型 lowering、Trace schema 或 canonical 数据。

### 负责什么

- 使用与 colocated 实验完全相同的 Production Trace 和模型配置建立 P/D 双实例分析。
- 目标研究配置暂定为 GLM-5.2 P(TP32+EP32)+D(TP32+EP32)、Kimi-K2.6 P(TP16+EP16)+D(TP16+EP16)，但当前主链只验证了远小于该规模的受限组合，ProtoBridge scaleout 也尚未支持。只有 topology、device count、scale-up/scale-out 边界和对应 profile 均被验证后才能启用这些配置，否则结果必须为 unsupported。
- 使用明确的 topology 和 ProtoBridge surrogate contract 计算 KV transfer；某硬件缺少合法通信参数时报告 unsupported，不能借用其他硬件。
- 前置合同闭合后先运行两个模型 × 三硬件的 6 个 debug smoke，再运行 low/medium/high 的 18 个正式 projected cell；前置合同未闭合时只报告 blocked/unsupported，不生成占位数值。
- 报告 Prefill queue、Decode queue、KV transfer bytes/time、P/D utilization、transfer/compute overlap、TTFT、TPOT、E2E 和 throughput。

### 不负责什么

- 不运行 PD 真机实验。
- 不创建或伪造 PD ground truth。
- 不把 colocated profile 拼接成 PD 实测结果。

### 产出

- 一个可审阅的本地 commit。
- 前置合同闭合时产出 6 个 PD debug smoke 和 18 个正式 analytical PD 结果；否则产出逐项 blocked/unsupported 矩阵而不是占位数值。
- PD 关键路径、KV transfer 和 P/D 利用率报告。
- 清楚标记为 projected only、no real PD ground truth 的论文图表数据。

## Agent 5：Kernel 精度

### Worktree 与文件范围

第一波只做 frozen-baseline 的只读 full eval 与证据审计，不创建 writer worktree。ProtoBridge 更新集成完成后，如已形成一个有独立物理证据、边界明确的单一机制修改，再从新 baseline 创建 `midlayer-wt-kernel-accuracy`，负责约定的 `src/kernel/`、`evaluation/kernel/` 和对应 tests；确有独立机制证据需要时可以提出 `microbench/` 修改方案，但未经用户确认不执行硬件采集，不修改 canonical `kernel_dataset`、`launch_data` 或 Serving 代码。

### 负责什么

- 在固定 baseline 上运行 A800、C500、910B 的完整单 Kernel evaluation，按 GEMM、FlashAttention、Fused MoE 和 Vector 分类，同时按 shape、launch、split、并行度和硬件机制定位主要误差。当前报告已经具备 MAPE、MedAPE、WMAE 和主要分桶；新增 P50/P90 APE、最大误差 case 和 complete/partial/unsupported 汇总属于报告补全，不得被描述成重新实现 full eval。
- 检查每个生产参数的来源、单位、物理含义和适用范围，删除或拒绝任何 per-shape/per-kernel duration 拟合与静默 fallback。
- 只使用硬件规格、算子/编译器静态语义、合法 launch observation 和与 evaluation case 独立的机制 microbenchmark 改进统一公式。
- 每次修改后运行三硬件相同的 full kernel eval，报告每算子 MAPE、P50/P90 absolute percentage error、最大误差 case、complete/partial/unsupported 数量及改动前后变化。
- 分析 Kernel 改动对 Serving 的预期影响，但不读取 Serving E2E 误差来选择参数；Serving 影响由主 Agent 在 checkpoint 合入后通过 Agent 3 的固定实验矩阵验证。

### 不负责什么

- 不以降低 evaluation MAPE 为依据反解参数、选择分段点或建立 duration LUT。
- 不修改或替换 canonical 数据，不自行安排重采。
- 不修改 Serving Scheduler、Trace、model lowering 或报告代码。
- 不直接 merge、rebase 或 push，也不让未验收的 predictor commit 进入其他 Agent 的 baseline。

### 产出

- 一个或多个经过阶段性整理的本地 commit；每次交付只包含一个可独立审阅的物理机制修改。
- 修改前后三硬件完整单 Kernel 精度表。
- 按 GEMM、FlashAttention、Fused MoE、Vector 分类的误差归因和 remaining issue 清单。
- 每个新增或修改参数的 provenance、单位、物理含义、适用范围和独立验证证据。
- 建议主 Agent 接受或拒绝该 checkpoint 的明确结论，以及未执行的硬件验证需求。

## Agent 6：ProtoBridge 更新集成

### Worktree 与文件范围

ProtoBridge 已于 2026-08-27 同步到上游 `54320aa712ad190ac647e69164233f4b228b4997`。该版本新增 FullMesh；按用户确认，MCCL FC2 映射到 FullMesh、MCCL Tree 映射到 Tree、HCCL Mesh 映射到 FullMesh。对应 adapter、versioned selector 消费和 repository/kernel tests 位于活动代码中；上游源码仍保持逐字快照。

### 负责什么

- 对比当前 vendored commit `32a3f50b43204296bb68b943faeb54770d615e95` 与候选上游 commit `f196675f57fae29bb143a7b93ab1eefd133c76be`，确认文档所述 collective 类型、Send/Recv、data-bytes-per-step 和 traffic 语义确实存在，并列出 surrogate API、输入参数、单位、拓扑语义和输出语义变化。
- 按仓库特许规则逐字更新 `whitebox_surrogate_v2.py`，同步上游 commit、source SHA256、license SHA256 和 provenance；禁止在 vendored 文件中加入 Midlayer patch。
- 更新 Midlayer `adapter.py`，严格完成 KernelSpec communication contract 到上游 surrogate 的字段、单位和返回值映射。
- 验证 all-reduce、all-gather、all-to-all、P2P/KV transfer 中哪些是上游明确支持的 operation；不支持的 operation fail loud，不用近似分支冒充。
- 接入由 ProtoBridge 作者提供并带版本/digest 的 A800、C500、910B profile；文档下方包含的 wire rate、startup、efficiency 和 topology 目前仍是待核验候选，其中明确标注的 H200默认值和占位值不能注册为 production profile。如果作者尚未提供可追溯 profile，只验证 synthetic contract并输出缺失字段和接口需求。
- 运行 snapshot hash、license、adapter、单位、topology、profile digest 和 communication predictor 测试。

### 不负责什么

- 不修改 ProtoBridge surrogate 公式、算法、参数定义或硬件支持逻辑。
- 不自行训练、校准或拟合 A800/C500/910B profile。
- 不从 Midlayer Serving/Kernel 真值反解 ProtoBridge 参数。
- 不把 H200 preset 或 synthetic test profile 声称为三种生产硬件支持。
- 需要上游修改、校准或新增 profile 时只形成精确需求，由用户协调 ProtoBridge 作者处理。

### 产出

- 一个可独立审阅的本地 commit。
- ProtoBridge old/new upstream commit 与 source SHA 对照。
- 更新后的 verbatim snapshot、license/provenance 和 Midlayer adapter。
- operation × hardware 支持矩阵及 fail-loud 测试。
- A800/C500/910B 所缺 author profile、参数、单位或 topology 信息清单，供用户转交上游作者。

### 用户补充说明
ProtoBrdige侧，对应作者已完成（已推送）：
surrogate接口已增加collective类型参数（现在可区分AllReduce/AllGather/AllToAll）
Send/Recv接口已添加
data_bytes / step / traffic语义已定义
scaleout暂时未支持，后续会支持。

对应三种硬件配置（需要你帮忙确认）
```
A800（NVSwitch switched，非阻塞，无需 bisection）：
s = make_surrogate_from_wire(25000, 6, 'ring',
        eta_ring=0.40,                  # H200 默认值，A800 占位
        startup_us=55.0,                 # 小消息 LL/LL128；大消息/alltoall 改 100
        link_latency_us=0.5,
        spray_chunk_bytes=4096)
topo = make_topology_from_family('switched', N)   # 非阻塞，无需 per_link_gbps
lat = s.predict(D, N, 'ring', credits=64, ber=0, collective='allreduce', topology=topo)

C500（两卡 FC2 映射到 FullMesh；四卡按 selector 选择 Tree/Ring）：
# 2-GPU FC2
s = make_surrogate_from_wire(64000, 3, 'fullmesh',
        eta_ring=0.40,                  # 仅 Ring 占位；不作用于 FullMesh 路径
        fullmesh_bw_bytes_per_us=64000, # 每 peer 路径；并发已由 FullMesh 公式表达
        startup_us=5.0,                 # LL/LL128；大消息改 10
        link_latency_us=0.5,
        spray_chunk_bytes=4096)
topo2 = make_topology_from_family('fullmesh', 2, links_per_gpu=1, per_link_gbps=64)
# 8-GPU: 两个 4-GPU 域 + host path (PCIe5 aggregate 1024 Gbps, hop=2)
topo8 = make_topology(bisection_gbps=1024, hop_count=2)
lat = s.predict(D, 2, 'fullmesh', credits=64, ber=0, collective='allreduce', topology=topo2)

910B（HCCS fullmesh，8-GPU）：
s = make_surrogate_from_wire(28000, 7, 'fullmesh',
        eta_ring=0.40,                  # 占位
        fullmesh_bw_bytes_per_us=28000, # 每 peer HCCS 路径，不取 7 路聚合
        startup_us=60.0,                # 只有 simpleNs=60000
        link_latency_us=0.2,
        spray_chunk_bytes=524288)
topo = make_topology_from_family('fullmesh', 8, links_per_gpu=7, per_link_gbps=28)
lat = s.predict(D, 8, 'fullmesh', credits=64, ber=0, collective='allreduce', topology=topo)
```

新版 ProtoBridge 的 `collective` 是 `predict()` / `predict_tail()` 的必填 keyword-only 参数，不是 `make_surrogate_from_wire()` 的构造参数；传给构造器会直接报错。上述数值仍是待作者确认的候选或占位信息，不能注册为 production profile。

FullMesh 的 `fullmesh_bw_bytes_per_us` 表示单个并发 peer 路径的带宽，不是每卡所有 peer 链路之和；因此 C500 使用 64,000 B/us，910B 使用 28,000 B/us。多 peer 聚合吞吐由 FullMesh 的并发阶段语义自然产生，不能再次乘以 3 或 7。


## 实验一览

| 实验 | 谁负责 | 运行多少 | 回答什么问题 | 是否使用真值 |
|---|---|---:|---|---:|
| 数据身份审计 | Agent 1 | 第一阶段 27 个物理 case；第二阶段 18 个状态项 | 每个硬件 case 的 workload 与 observation 是否可信，以及同名 case 是否严格跨硬件可比 | 是，仅读取 |
| 小模型 E2E 精度 | Agent 3 | 第一阶段最多 27 个物理 case | TTFT/TPOT/E2E/throughput 预测是否准确；第二阶段等待外部数据 | 是 |
| Scheduler/算子/launch exact 对齐 | Agent 3 | 当前有完整 observation 的 case | 请求是否被正确调度、生成正确算子并选择正确 launch | 是，evaluation only |
| Host/Device execution attribution | Agent 3 | 第一阶段最多 27 个物理 case | Prefill/Decode 的外部调度、forward 内 CPU 和 device 时间分别占多少 | 是，evaluation only |
| 小模型负载曲线 | Agent 3 | 复用当前可用 case | 是否预测排队、饱和和尾延迟趋势 | 是 |
| 大模型 debug smoke | Agent 2+3 | 目标 6 cells | 六条 model×hardware 主链能否完整运行；未闭合的 MLA/DSA、TP/EP、通信组合报告 unsupported | 否 |
| 大模型 colocated | Agent 3 | 目标 18 cells | 三硬件上的 large-model projected Serving 表现；只对合同闭合的 cell 生成数值 | 否 |
| 大模型 PD smoke | Agent 4 | 前置闭合后目标 6 cells | PD 主链是否完整 | 否 |
| 大模型 analytical PD | Agent 4 | 前置闭合后目标 18 cells | KV transfer、P/D queue 和关键路径影响 | 否 |
| 单 Kernel 基线与优化 | Agent 5 | 三硬件 full kernel matrix | GEMM/FA/MoE/Vector 的解析预测精度和物理误差来源 | 是，evaluation only |
| ProtoBridge 更新与兼容性 | Agent 6 | upstream diff + operation/hardware matrix | 通信 surrogate 是否能合法服务三硬件 Serving/PD | 仅使用作者 profile |

## 执行顺序

Phase 0 先由主 Agent 完成仓库合同、compact schema、模型矩阵和 baseline commit，不创建 writer worktree、不移动数据。之后第一波启动 Agent 1 的只读 inventory、Agent 2 的 compact contract/converter、Agent 5 的只读 Kernel baseline 审计和 Agent 6 的 ProtoBridge upstream diff/update，加主 Agent 共最多 5 个并发线程。Agent 5 在 Agent 6 修改 `src/kernel` 期间不得写该 owner。Agent 1 完成 inventory 后释放线程；Agent 3 只能先做不依赖 schema 实现的只读设计，必须等 Agent 2 第一交付合入后再从新 baseline 创建 writer worktree。

第一波代码集成顺序为 Agent 2 的 compact 第一交付→Agent 6→Agent 3；Agent 6 集成并形成新 baseline 后，Agent 5 才能获得一个具体、证据驱动的 Kernel writer 任务。Agent 6 缺少 author profile 时允许以更新后的 fail-loud adapter 和 blocker 清单结束，但不能伪造三硬件 communication 支持。若 Agent 5 后续 checkpoint 合入，主 Agent 必须使用新的精确 commit 重新生成第一阶段全部可用小模型结果和合同已闭合的大模型 colocated 结果，之前的结果只作为基线对比；第二阶段保持独立，等待用户提供 12 个 C500/910B case 后再纳入。

第二波先从新的 main baseline 执行 Agent 2 的大模型 lowering 第二交付及 Agent 3 的 compact evaluation。只有 GLM/Kimi execution、launch、KV、目标 TP/EP 与 ProtoBridge Send/Recv、collective、author profile、scale-up/scale-out 边界全部闭合后，才创建 Agent 4 worktree并运行 analytical PD；否则 Agent 4 保持 blocked。Agent 5 的下一项改动不得在任一固定 Serving/PD 实验中途合入。

最终由主 Agent 检查每个结果的 commit、HWSpec、model provenance、Trace SHA、deployment 和 prediction method；任何 measured/projected 混淆、真实 duration 进入 Forward、旧 LUT 被复用或 silent fallback 都视为验收失败。所有结果写入独立的 `experiment_logs/` run，不覆盖旧结果，未经用户明确授权不 push。
