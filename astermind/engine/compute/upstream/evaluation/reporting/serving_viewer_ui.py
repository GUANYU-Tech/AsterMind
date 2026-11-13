"""Shared bilingual presentation for the standalone serving viewers.

Translation touches UI text and canvas labels only; input values, identifiers,
raw evidence and embedded data remain unchanged.
"""
import json
import hashlib
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path


def report_run_context(payload, coverage):
    """Keep cost provenance optional and unknown costs distinct from zero."""
    execution = payload["main_chain_status"] if "main_chain_status" in payload else None
    if execution is not None:
        execution = {key: execution[key] for key in ("status", "chain", "execution_artifact_sha256")
            if key in execution}
    recording = coverage["resource_recording_s"] if "resource_recording_s" in coverage else None
    source = payload["visualization_run_costs"] if "visualization_run_costs" in payload else None
    costs = ({"html_generation_s": source["html_generation_s"] if "html_generation_s" in source else None,
        "scope": source["scope"] if "scope" in source else None} if source is not None else None)
    if source is not None:
        for key in ("total_wall_through_result_write_s", "simulation_wall_s", "recording_projection_s",
                "finalization_wall_s", "result_json_projection_write_s"):
            if key in source:
                costs[key] = source[key]
    values = [recording] + ([value for key, value in costs.items() if key != "scope"] if costs is not None else [])
    if any(value is not None and (type(value) not in (int, float)
            or not math.isfinite(value) or value < 0) for value in values):
        raise ValueError("recorded report costs must be finite nonnegative seconds")
    result = {"execution": execution, "resource_recording_s": recording, "report_costs": costs}
    if "performance_timing_status" in payload:
        from evaluation.serving.simulation.execution_artifact import _require_performance_timing_status
        result["performance_timing_status"] = _require_performance_timing_status(payload["performance_timing_status"])
    return result


def _configuration_metadata(value):
    private = {"prompt", "prompt_text", "request_text", "prompt_token_ids",
        "output_token_ids", "embedding_token_ids", "token_ids", "access_token", "api_key",
        "prompt_texts", "prompts", "request_texts", "input_text", "input_texts",
        "sglang_request_token_transcripts", "request_token_transcripts", "mtp_draft_token_ids_by_step",
        "prompt_token_ids_by_request", "output_token_ids_by_request", "token_ids_by_request"}
    if isinstance(value, Enum):
        return _configuration_metadata(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("configuration metadata keys must be strings")
        return {key: _configuration_metadata(item) for key, item in value.items()
            if key not in private and not key.endswith("_token_ids") and not key.endswith("_token_transcripts")}
    if isinstance(value, (list, tuple)):
        return [_configuration_metadata(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("experiment configuration metadata must be JSON-compatible")


def report_experiment_configuration(payload):
    """Project recorded configuration metadata; never export request contents."""
    if "visualization_experiment_configuration" in payload:
        result = _configuration_metadata(payload["visualization_experiment_configuration"])
        _validate_configuration_files(result["files"])
        return result
    groups = {
        "model": ("model_config_provenance", "model_compute_dtype"),
        "deployment": ("deployment", "deployment_plan", "hardware_ids", "routing_policy",
            "execution_mode", "serving_timing_mode", "kv_handoff_mode", "prediction_method", "launch_policy"),
        "scheduler": ("scheduler_runtime",),
        "inputs": ("request_source", "arrival_process", "workload_hash", "workload_assumption_binding"),
        "capacity": ("kv_capacity", "kv_lifecycle_replay"),
        "speculation": ("mtp_scenario", "mtp_pd_handoff_contract"),
        "provenance": ("execution_provenance",),
    }
    result = {group: {key: payload[key] for key in keys if key in payload}
        for group, keys in groups.items()}
    result["host_timing"] = ({key: payload["host_timing"][key] for key in (
        "schema_version", "timing_mode", "launch_measurement_boundary",
        "scheduler_measurement_boundary", "profiles", "runtime_prepare_profiles")
        if key in payload["host_timing"]} if "host_timing" in payload else {})
    result["hardware_display"] = (payload["visualization_hardware"]
        if "visualization_hardware" in payload and payload["visualization_hardware"] is not None else {})
    result["bandwidth_peaks"] = payload["visualization_memory_bandwidth_reference"] if "visualization_memory_bandwidth_reference" in payload else None
    files = payload["visualization_configuration_files"] if "visualization_configuration_files" in payload else []
    _validate_configuration_files(files)
    result["files"] = files
    result["unrecorded"] = [name for name in ("model_config_provenance", "deployment_plan",
        "scheduler_runtime", "request_source", "execution_provenance") if name not in payload]
    if "visualization_configuration_snapshot_missing" in payload:
        result["unrecorded"].extend(payload["visualization_configuration_snapshot_missing"])
    return _configuration_metadata(result)


def _validate_configuration_files(files):
    for record in files:
        if record["kind"] not in ("model_config", "hardware_config"):
            raise ValueError("unsupported experiment configuration file kind")
        if hashlib.sha256(record["content"].encode()).hexdigest() != record["sha256"]:
            raise ValueError("configuration snapshot content does not match recorded hash")


_TRANSLATIONS = dict(line.split("\t", 1) for line in r'''端到端执行与资源泳道	End-to-end execution and resource timeline
CPU、Device、每次 Forward 和资源曲线共用一条模拟时间轴。数值来自解析预测，不是 nsys 实机计数器。	CPU, device, forward iterations and resource curves share one simulated time axis. Values are analytical predictions, not measured nsys counters.
点击 Forward 色块或选择编号，在下方展开这一轮的算子明细。所有 Forward 均显示；没有保存算子明细的 Forward 用虚线边框标出。滚轮左右平移，Alt＋滚轮缩放，按住图表左右拖动也可平移；全程视图已覆盖完整范围，放大后即可平移。竖向光标对齐各资源行。小于 1 像素的 Forward 会显示为定位标记，放大后查看真实间隔。CPU 调度未建模的部分显示“未建模”，不会把它当成实测零开销。	Click a forward iteration to open its breakdown. Dashed borders mean operator details were not retained. Wheel to pan; Alt + wheel to zoom; drag to pan. Zoom in before panning the full-run view. Subpixel iterations become position markers. Unmodeled CPU scheduling time stays unknown.
Forward 选项：main 是实例；末尾数字是该实例的调度轮次，不是请求 ID。Prefill=预填充，Decode=解码，Mixed=两者混合。req 是本轮参与请求数；tok 是本轮调度 token（P 输入 + D 解码，不含 padding），不是请求总长度。自动纵轴时各行刻度不同，应比较数值而非曲线高度。点击左侧资源名或双击曲线可以单独放大；悬停显示浮动读数，点击曲线固定读数。	Forward IDs identify instance and scheduling iteration, not a request. req = participating requests; tok = scheduled prefill + decode tokens, excluding padding. Auto-scaled tracks use different scales: compare values, not heights. Click a resource label or double-click its curve to enlarge it. Hover for readings; click to pin them.
Tensor / Vector 是工作量相对标称算力的需求；Vector 取各类向量单元归一化需求的最大值。L2/HBM 使用预测器已有缓存/LRU内存路径的读写流量。H200 的 L2 读、写分别用 NCU 理论峰值；综合占比取读写两者较高值，不把百分比相加。HBM 读写共用标称带宽，综合占比按读写流量合计计算。按步骤汇总的需求均匀分摊到该步骤区间，局部曲线是近似；Device 忙碌按区间并集统计。没有成对读写路径的操作单独计为缺失，不补零；没有额外运行第二套缓存仿真。不同资源需求可能重叠，不能相加成总利用率。	Tensor/Vector demand is normalized by nominal compute capacity; Vector uses the maximum across vector units. L2/HBM traffic comes from the existing predictor cache/LRU paths. H200 L2 reads and writes use separate NCU peaks; combined pressure is the larger directional percentage. HBM uses total read + write traffic over shared nominal bandwidth. Demand is spread uniformly within each forward, so local curves are approximate. Device busy time uses exact interval unions. Missing traffic remains unknown. Resource percentages overlap and must not be added together.
默认放大已保留请求最多批次的第一层；点击色块查看形状、时间和资源依据。实例、rank、操作、请求和批次筛选会叠加。请求筛选显示它参与的批次，共享算子的耗时不归属于某个单独请求。选择批次时滑条在该批次内平移；“全程总览”切换到所有批次。滚轮左右平移，Alt＋滚轮缩放，拖动图表或滑条也可平移；小于 1 像素的色块按 1 像素显示。空白表示没有此类事件或该段资源明细未保留；灰色表示依据不足。计算、通信和搬运按操作类别分行，不代表它们一定使用独立硬件队列。	The first layer is initially enlarged. Click an operation for shape, timing and evidence. Filters combine. Request filtering shows shared batches containing that request, not exclusive request costs. Wheel to pan; Alt + wheel to zoom; drag the chart or slider to pan. Blank areas have no matching events or no retained details; gray means insufficient evidence. Tracks group operation types, not necessarily independent hardware queues.
同一模拟时间轴：按实例和设备查看计算、通信、KV 搬运与 Host 提交。所有时间均为模型预测。	One simulated time axis for compute, communication, KV transfers and host submissions, grouped by instance and device. All timings are model predictions.
忙碌占比按设备区间并集计算。资源表使用当前时间、实例和 rank 下的所有操作，不受操作类型、请求或批次筛选影响。HBM 仅是逻辑需求下界；未建模的物理流量、内存占用和 Host 行为保持未知。	Busy time uses the union of device intervals. Resource tables cover all operations in the selected time, instance and rank, independently of operation/request filters. HBM here is a logical-demand lower bound. Unmodeled physical traffic, memory usage and host behavior remain unknown.
“HBM 需求”把生产模型的逻辑字节均匀摊在对应操作区间；不是独立的实测内存操作或物理带宽曲线。“下界较大项”只比较工作量/峰值算力和逻辑字节/峰值带宽，不代表端到端关键路径。	Logical HBM demand spreads logical bytes over each operation interval; it is not measured physical traffic. The larger lower bound compares work/compute capacity with logical bytes/bandwidth, not an end-to-end critical path.
这次 Forward 只有全程汇总，采样版未保存它的逐算子明细。上方 CPU / Device 和资源统计仍包含它。	Only aggregate data was retained for this iteration. CPU/device and resource totals above still include it.
整步各组预测操作时长的汇总；细边颜色标记累计最多的组，不能表示组别发生顺序、持续占用或端到端关键路径	Predicted operation durations summed by group. The border marks the largest group, not execution order, continuous activity or the end-to-end critical path.
。整步数值不按当前缩放窗口裁剪，不受操作类型筛选影响，也不是端到端关键路径。按请求筛选时，只汇总该请求参与且保留明细的批次，仍包含同批次其他请求的工作。	. Whole-step totals are not clipped to the zoom window and are not an end-to-end critical path. Request filters include other requests sharing the retained batches.
。滚轮左右平移；Alt＋滚轮缩放；拖动平移；Shift＋拖动框选时间段；悬停读取准确数值。	. Wheel: pan; Alt + wheel: zoom; drag: pan; Shift + drag: select a time range. Hover for exact readings.
。已加载这一轮明细。	. Iteration details loaded.
已加载这一轮明细。时间仍为全程的绝对模拟时间；下面可以继续放大到某一层。	Iteration loaded. Times remain absolute simulated times; zoom further to inspect a layer.
移动光标，可查看对齐时间窗口内各资源的数值。数值基于当前显示的分箱宽度。	Hover to compare aligned resource readings. Each point is an average over the displayed time bin.
当前窗口没有保留具体操作明细；聚合条只表示范围，不能据此判定瓶颈。	No operator detail was retained in this window. Aggregate envelopes alone cannot establish the bottleneck.
工作量/峰值下界的比较，不包含启动和同步成本，不能代表实际关键路径	Work/capacity lower-bound comparison; excludes launch and synchronization, and does not identify the actual critical path.
逻辑字节需求按操作区间均匀分摊，非实际 HBM 事务	Logical bytes spread over the operation interval; not physical HBM transactions.
每个操作的预测器瓶颈组；不表示跨操作的端到端关键路径	Predictor bottleneck group per operation; not a cross-operation end-to-end critical path.
当前数据没有保留整步的模型瓶颈组；灰色条表示未知。	Whole-step bottleneck groups were not retained; gray means unknown.
本报告未配置独立标称峰值，使用记录中的 HWSpec 配置带宽。	No independent nominal peak is configured; recorded HWSpec model bandwidth is used.
读写共用带宽；综合占比按读写流量合计计算。	Read/write share bandwidth; combined pressure uses total read + write traffic.
综合占比取读占比、写占比中较高值。	Combined pressure is max(read %, write %).
调度 / 完成事件标记；未记录 Scheduler 耗时	Scheduling/completion markers; scheduler CPU duration was not recorded.
点击上方的 Forward。明细按需解压，一次只加载一轮。	Select a forward above. Details are decompressed on demand, one iteration at a time.
将鼠标移到曲线上查看每个点的窗口和数值。	Hover over the curve to inspect a point's window and value.
未细分、相等或未保留；灰底细边的聚合条只表示整步组别统计，不表示该组持续占用全段	Unspecified, tied or unavailable. Gray aggregate bars describe whole-step totals, not continuous activity.
此浏览器不支持自动解压，请使用较新版本的 Chrome、Edge、Firefox 或 Safari	Automatic decompression is unavailable; use a recent Chrome, Edge, Firefox or Safari.
请使用支持自动解压的现代浏览器	Use a modern browser with automatic decompression support.
当前没有可用的流量 / 资源数据	No traffic/resource data is available.
N/A · 此数据没有成对的 L2 读写路径	N/A: no paired L2 read/write traffic was exported.
N/A · 当前部署无该类链路证据	N/A: no applicable link evidence for this deployment.
N/A · 资源输入不可用	N/A: resource inputs unavailable.
相交步骤的模型瓶颈组（整步累计，含重叠）：	Bottleneck groups of overlapping steps (whole-step totals, including overlap):
当前窗口已保留明细中，累计操作时长最多：	Largest cumulative operation duration among retained details in this window:
下界比较中，HBM 较大的操作其预测时长累计	Cumulative predicted duration of operations with larger HBM lower bounds:
 个聚合区间不作逐操作判断；该比较不含启动、同步及物理内存事务。	 aggregate intervals excluded; comparison excludes launch, synchronization and physical transactions.
。缺失或超预算	. Missing or over budget:
。这是各操作时长之和，重叠和多 rank 会重复计数，不能作为端到端关键路径。	. Durations are summed; overlapping work and multiple ranks can be counted more than once. This is not an end-to-end critical path.
（单算子内部模型路径，非请求级关键路径）	(intra-operation model path, not a request-level critical path)
Host/排队事件不提供设备资源工作量	Host/wait events have no device resource work estimate.
此投影未保留逐操作计算/HBM 数据	Per-operation compute/HBM evidence was not retained.
旧投影未记录请求数	Request counts were not recorded in this older projection.
所选实例没有记录到路由选择。	No routing choices were recorded for this instance.
已知 activation 峰值：不可用（没有来源分配记录）	Known activation peak: unavailable (no source allocation records).
没有正的计算或逻辑字节需求	No positive compute or logical-byte demand.
没有保留此操作的资源输入	Resource inputs for this operation were not retained.
聚合区间或未保留资源输入	Aggregate interval or unavailable resource inputs.
未保留；不可从最终时延反推	Not retained; cannot infer from final latency.
模型未列出其他限制	The model reports no additional limitations.
此操作未保留诊断	Diagnostics were not retained for this operation.
不支持的报告存储版本	Unsupported report storage version
不支持的明细存储版本	Unsupported detail storage version
报告包含无效的表引用	Invalid table reference in report
无效的明细表引用	Invalid detail table reference
正在解压所选 Forward…	Decompressing selected forward…
正在解压并加载时间线…	Decompressing and loading timeline…
正在加载全程总览…	Loading overview…
当前筛选没有 Forward	No forwards match the current filter
报告加载失败：	Report load failed:
明细加载失败：	Detail load failed:
起点 → 终点	Source → destination
来源 → 选中实例	Source → selected instance
全部批次（全程汇总，明细按覆盖范围）	All batches (full-run aggregates; retained details only)
全部已保留明细的请求	All requests with retained details
选中 Forward 的 Timeline Breakdown	Selected forward breakdown
端到端模拟时间	Simulated end-to-end time
推理执行时间线	Inference execution timeline
一层一层看：操作与资源	Layer-by-layer operation breakdown
时间窗口内的资源与排队	Resources and queues in this time window
单位、分母、容量与原始证据	Units, denominators, capacity and raw evidence
模型瓶颈组和下界行：	Bottleneck groups and lower-bound tracks:
模型瓶颈组（整步汇总）	Model bottleneck groups (whole step)
常见算子模型路径（整步）	Common operation paths (whole step)
模型启动 / 分派时间	Model launch/dispatch time
模型资源服务时间	Model resource service time
路径支持级别 / 限制	Path support / limitations
设备计算资源（调度口径）	Device compute resource (schedule)
工作量下界比较	Work lower-bound comparison
算子模型关键路径	Intra-operation model critical path
HBM 逻辑字节 / 带宽下界较大	Larger logical HBM / bandwidth lower bound
HBM 逻辑需求下界	Logical HBM demand lower bound
HBM 逻辑需求	Logical HBM demand
HBM 带宽需求	HBM bandwidth demand
L2 带宽需求	L2 bandwidth demand
设备忙碌占比	Device busy fraction
最大算力资源需求	Maximum compute demand
最大算力需求	Maximum compute demand
平均每 token 延迟 TPOT	Mean TPOT
平均首字延迟 TTFT	Mean TTFT
选中的操作	Selected operation
展开原始字段和资源证据	Raw fields and resource evidence
操作颜色（仅列本次记录）：	Operation colors (recorded types only):
点击时间线中的色块。	Click an operation in the timeline.
路由与请求排队	Routing and request queues
资源随时间变化	Resource history
统计口径和资源覆盖	Definitions and evidence coverage
横向拖动	Pan timeline
内存带宽方向	Memory bandwidth view
综合占比（读写均考虑）	Combined pressure (read and write)
固定 0–100%（超限扩展）	Fixed 0–100% (expands above 100%)
自动放大（各行独立刻度）	Auto scale (independent per track)
定位选中 Forward	Focus selected forward
Forward 总数	Forward iterations
可展开 Forward	Detailed forwards
请求（筛选参与批次）	Request (shared batch filter)
排队时长 / 派发时刻（秒）	Queue duration / dispatch time (s)
候选实例排队 token	Queued tokens by candidate instance
所属步骤 / 阶段	Step / phase
同一批次请求	Requests sharing the batch
请求 / 阶段	Request / phase
完整区间	Full interval
聚合范围	Aggregate extent
聚合区间	Aggregate interval
模型瓶颈字段	Model bottleneck fields
模型瓶颈组	Model bottleneck group
下界较大项	Larger lower bound
下界数值	Lower-bound values
模型层	Model layer
图上这一行	This track
起点	Start
终点	End
预填充	Prefill
逐 token 解码	Token-by-token decode
Prefill 预填充	Prefill
Decode 解码	Decode
Mixed 混合	Mixed
模型组：	Model group:
依赖路径 ·	Dependency path ·
流水线 / 依赖	Pipeline / dependencies
多资源流水线	Multi-resource pipeline
解析执行下限（资源未细分）	Analytical execution floor (resources unspecified)
设备启动 / 分派	Device launch / dispatch
启动 / 分派	Launch / dispatch
采样 · HBM 逻辑需求	Sampling · logical HBM demand
采样 / 其他	Sampling / other
点对点交接	Point-to-point handoff
KV 交接	KV handoff
KV 更新 / 搬运	KV update / transfer
聚合（各类别混合）	Aggregate (mixed categories)
QKV 投影	QKV projection
FFN 投影	FFN projection
输出投影	Output projection
MoE 计算	MoE compute
专家路由	Expert routing
稀疏索引	Sparse indexing
归一化 / 激活	Normalization / activation
其他通信	Other communication
Warp 调度	Warp scheduling
Tensor Core（fp16 资源组）	Tensor Core (FP16 resource group)
Tensor Core 管线	Tensor Core pipeline
ALU 管线	ALU pipeline
FMA 管线	FMA pipeline
SFU 管线	SFU pipeline
特殊函数单元	Special function unit
向量单元	Vector unit
归约单元	Reduction unit
归约依赖	Reduction dependency
共享内存	Shared memory
片上互连	On-chip interconnect
互联资源	Interconnect resource
算力资源	Compute resource
调度资源	Scheduled resource
HBM / 存储资源	HBM / storage resource
Roofline（未细分）	Roofline (unspecified)
L1 缓存	L1 cache
L2 缓存	L2 cache
CPU 调度	CPU scheduling
CPU 提交	CPU submission
Host 提交	Host submission
Device 执行	Device execution
Tensor 需求	Tensor demand
Vector 需求	Vector demand
模型瓶颈	Model bottleneck
HBM 下界	HBM lower bound
HBM 合计	HBM total
L2 综合	L2 combined
本轮请求数	Requests per forward
本轮 token 数	Tokens per forward
全程平均（内存为综合占比）	Full-run mean (combined memory pressure)
Device 忙碌	Device busy
资源证据：可用	Evidence: available
每个 Forward 的数量，纵轴 0–	Count per forward; y-axis 0–
每个点是此窗口均值	Each point is the window mean
分箱峰值	Peak bin
纵轴 0–	Y-axis 0–
窗口均值	Window mean
当前区间均值	Selected interval mean
读数窗口	Reading window
相交 Forward 中，流量路径可用	Overlapping forwards: traffic evidence available for
读写合计流量	Total read + write traffic
合计流量	Total traffic
综合占比	Combined pressure
读峰值	Read peak
写峰值	Write peak
每点窗口	Bin width
每点为	Bin width:
调度调用	Scheduler calls
完成事件	Completion events
调度 / 完成事件标记	Scheduling / completion markers
Scheduler 耗时按所选 timing mode 记录；当前未计时不代表耗时为零。	Scheduler CPU time follows the selected timing mode; unrecorded time is not zero.
点击资源名放大 ↗	Click label to enlarge ↗
当前区间	Selected interval
设备 rank	Device rank
实例 / rank	Instance / rank
完成请求	Completed requests
请求吞吐	Request throughput
仿真程序耗时	Simulator wall time
预测持续时间	Predicted duration
总览覆盖所有	Overview includes all
可展开算子明细	with operator details
每个批次可能服务多个请求。全程仿真耗时	A batch may serve multiple requests. Simulator wall time:
与横轴的模拟时间不同。	distinct from simulated time on the axis.
正常容量验收	Normal-capacity acceptance
已知 activation 峰值	Known activation peak
不完整	incomplete
每张设备 HBM 容量	HBM capacity per device
预测模型配置带宽	Configured predictor bandwidth
B 算法传输量	B algorithmic traffic
B/s（算法字节）	B/s (algorithmic bytes)
B/s / 端点	B/s per endpoint
每端	Per endpoint
累计操作时长最多	Largest cumulative operation duration
算力下界较大的操作累计	Cumulative duration of operations with larger compute lower bounds
操作没有可比较的资源证据	operations lack comparable resource evidence
算力下界较大	Larger compute lower bound
两项下界相等	Equal lower bounds
旧投影未保留	Not retained in older projection
旧投影未记录	Not recorded in older projection
有算子明细	Operator details available
仅全程汇总	Aggregates only
请求数据缺失	Request data unavailable
请求数未知	Request count unknown
自动放大	Auto scale
固定 0–100%	Fixed 0–100%
标准大图	Standard
更高	Tall
紧凑	Compact
时间放大 2 倍	Zoom time in 2×
时间缩小 2 倍	Zoom time out 2×
放大 2 倍	Zoom in 2×
缩小 2 倍	Zoom out 2×
放大这条曲线	Enlarge curve
曲线高度	Track height
单独放大	Enlarge resource
内存方向	Memory view
平移时间范围	Pan time range
开始（秒）	Start (s)
结束（秒）	End (s)
首层细节	First layer
全程总览	Full-run overview
请求 / 秒	requests/s
（已有模型部分）	(modeled portion only)
（原模型输出）	(original model output)
完整路径见下方原始数据	full path in raw evidence below
此浏览器	This browser
不支持	Unsupported
未记录该字段	did not record this field
未细分	Unspecified
未建模	Unmodeled
未保留	Not retained
未提供	Not provided
不可用	Unavailable
已有依据	Evidence available
有依据	with evidence
发射策略	Launch policy
预测方法	Prediction method
资源下界	Resource lower bound
资源证据	Resource evidence
算力	Compute
调度批次	Scheduled batch
实例	Instance
操作类型	Operation type
操作	Operation
整步汇总	Whole-step aggregate
设备忙碌	Device busy
请求	Request
调度 token	scheduled tokens
Forward；其中	Forwards; of these
Forward 次数	Forward count
个事件；区间包含空隙，具体操作未保留	events; interval includes gaps, operation details not retained
组别顺序	group ordering
仅为累加，不提供	sum only; no
各资源时间不能直接相加为关键路径	resource times cannot be added to obtain the critical path
资源服务时间不能直接等同关键路径	resource service time is not a critical path
非时间顺序、非请求级关键路径	not chronological or a request-level critical path
页面数值来自解析仿真，非实机观测。	Values are analytical predictions, not hardware observations.
相交步骤	Overlapping steps
此操作	This operation
缺失或超预算	Missing or over budget
不可从最终时延反推	cannot infer from final latency
正常容量	Normal capacity
模型路径	Model path
全程仿真耗时	Simulator wall time
全程	Full run
全部	All
关闭	Close
平移	Pan
纵轴	Y-axis
只看读	Read only
只看写	Write only
流量	Traffic
混合	Mixed
相等	Equal
等待	Wait
排队 / 等待	Queue / wait
计算	Compute
通信	Communication
缺失	Missing
未计时	Not timed
次操作	operations
次 Forward	Forwards
次；	times;
次数	Count
次。	times.
次	times
秒	s
读	Read
写	Write
是	Yes
否	No'''.splitlines())


_TRANSLATIONS.update({"已计分": "Scored", "不支持": "Unsupported", "超出定价域": "Out of domain", "部分完成": "Partial", "完成": "Complete", "失败": "Failed", "已完成": "Completed", "中位数": "Median"})
_TRANSLATIONS.update({"硬件": "Hardware", "评估模式": "Evaluation mode", "Case 总数": "Cases", " 误差 / 覆盖": " error / coverage", "Kernel 输出成本": "Kernel output costs", "完整 Kernel semantic 结果": "Complete Kernel semantic results", "（预测 / 实测）": " (predicted / observed)", "预览覆盖": "Preview coverage", "仅 Kernel 评估；误差仅计已定价 case，unsupported 与 out-of-domain 单列。原始延迟为纳秒，预览用微秒；无 Serving 时间线。": "Kernel evaluation only. Errors score priced cases; unsupported and out-of-domain cases are listed separately. Raw latencies are nanoseconds; the preview uses microseconds. No Serving timeline is recorded.", "仅展示前 ": "First ", " 个 case，全部 case 见 semantic 结果链接。": " cases shown; all cases are in the semantic results link."})
_TRANSLATIONS.update({"折叠全部": "Collapse all", "展开全部": "Expand all", "点击实例或 rank 标题可折叠；拖动导航选区移动时间，拖动两端调整范围。": "Click an instance or rank header to collapse it. Drag the navigator selection to pan; drag its edges to resize the time window.", "全程导航 · 拖动选区或两端": "Full-run navigator · drag selection or edges"})

_TRANSLATIONS.update({" 次 Forward；其中 ": " forward iterations; ", " 次可展开算子明细。每个批次可能服务多个请求。全程仿真耗时 ": " have operator details. Batches may serve multiple requests. Simulator wall time: ", " s，与横轴的模拟时间不同。": " s (separate from simulated execution time).", "交互与口径说明": "Navigation and definitions", "Forward 选项与读图说明": "Forward IDs and reading the chart", "带宽峰值与模型假设": "Bandwidth peaks and model assumptions"})

_TRANSLATIONS.update({"个请求 ·": "requests ·", "个不可用": "unavailable", "个调度 token（Prefill": "scheduled tokens (Prefill", "逻辑 HBM": "Logical HBM", "；点击查看依据": "; click for evidence", "综合": "combined", "个有依据 /": "with evidence /", "个操作没有可比较的资源证据，": "operations lack comparable resource evidence,", "个相交步骤未记录该字段。": "overlapping steps did not record this field.", "48 个时间窗": "48 time bins"})

_TRANSLATIONS.update({"窗口 ": "Window ", "；保留的具体操作 ": "; retained operations ", " 个；具体明细外包络（非连续保证） ": "; detail envelope (not necessarily continuous) ", "；其余 ": "; remaining ", " 个事件的细节未保留。聚合条只表示范围，包含空隙；若需更多逐操作细节，需要提高记录上限并重新仿真。": " events have no retained details. Aggregate bars may include gaps. Increase the recording budget and rerun to retain more operation detail.", "明细覆盖 ": "Detail coverage ", " 个请求、": " requests, ", " 个模型批次，分布在 ": " model batches across ", " 段采样区间。请求被覆盖表示至少一个参与批次有明细，不代表其全部生成过程均有明细。": " sampled intervals. A covered request has at least one retained participating batch, not necessarily its entire generation history.", " 个请求": " requests"})

_TRANSLATIONS.update({"推测解码": "Speculative decoding", "实验配置": "Experiment configuration", "来自本次运行保存的配置与来源证据；requested → effective 分别表示请求值和实际采用值。未记录的字段明确列出，不从文件名推断。": "Recorded configuration and provenance for this run. Requested → effective distinguishes requested settings from settings actually used. Missing fields are listed explicitly; nothing is inferred from filenames.", "计算精度": "Compute precision", "部署方式": "Deployment", "Serving 时间模式": "Serving timing mode", "请求数量": "Request count", "输入数据路径": "Input dataset path", "输入数据 SHA256": "Input dataset SHA256", "运行代码 commit": "Simulation code commit", "模型执行来源": "Model execution source", "CUDA Graph 模式": "CUDA Graph mode", "最大 batch token 数": "Max batch tokens", "最大请求数": "Max requests", "最大上下文长度": "Max context length", "KV block token 数": "KV block tokens", "前缀缓存": "Prefix caching", "分块 Prefill": "Chunked prefill", "异步调度": "Async scheduling", "未记录的配置项：": "Configuration fields not recorded: ", "已展示保存的运行配置。": "Recorded run configuration is shown below.", "下载配置 JSON": "Download configuration JSON", "模型与精度": "Model and precision", "部署与并行": "Deployment and parallelism", "调度器完整配置": "Full scheduler configuration", "请求集与输入身份": "Requests and input identity", "KV 容量与缓存": "KV capacity and cache", "Host 时间模型": "Host timing model", "版本与来源证据": "Versions and provenance", "硬件摘要": "Hardware summary", "显示带宽峰值": "Display bandwidth peaks", "已校验的配置文件快照": "Verified configuration file snapshots", "硬件": "Hardware", "模型": "Model", "未记录": "Not recorded"})

_TRANSLATIONS.update({"逐 rank 资源均值":"Per-rank resource averages","完成范围：":"Completion: ","输入总数未记录":"input total not recorded","次有明细":"with details","读写流量证据：":"Read/write evidence: ","模型主要组别：":"Top model group: ","模型身份未记录":"model identity not recorded","精度未记录":"precision not recorded","解析预测":"analytical prediction","已建模":"modeled","模型组别按预测算子时长累计，含重叠，不等于端到端关键路径。未计时或缺失的信息不填零。":"Groups sum predicted operation durations including overlap, not the end-to-end critical path. Missing or unmodeled information is not filled with zero."})

_TRANSLATIONS.update({"精确范围与显示设置":"Exact range and display settings","全部参数与来源路径":"All parameters and source paths"})

_TRANSLATIONS.update({"运行状态与成本":"Run status and costs","当前窗口：":"Selected window: ","相交 Forward":"intersecting forwards","参与请求（按完整 Forward 计数）":"participating requests (whole forwards)","资源范围：":"Resource scope: ","最高资源需求：":"Highest resource demand: ","全程平均模型需求中最高的一项，不代表实测利用率或端到端关键路径。":"Highest full-run mean modeled demand; not measured utilization or the end-to-end critical path.","执行状态":"Execution status","模拟时间":"Simulated duration","资源记录耗时（包含在仿真内）":"Resource recording (included in simulation)","HTML 生成耗时":"HTML build time","失败和 unsupported 计数":"Failure and unsupported counts","未单独记录":"Not separately recorded","使用说明":"Help","按参数名筛选：TP、cache、SHA256…":"Filter settings: TP, cache, SHA256…"})

_TRANSLATIONS.update({"实例数量：":"Instances: ","rank 数量：":"Ranks: ","已完成：":"Completed: "})

_TRANSLATIONS.update({"资源记录耗时":"Resource recording time","阶段耗时重叠关系":"Timing overlap between phases","未记录；不要直接相加":"Not recorded; do not add phase durations directly"})

_TRANSLATIONS.update({"硬件身份未记录":"hardware identity not recorded"})

_TRANSLATIONS.update({"平均首字延迟 TTFT（毫秒）":"Mean TTFT (ms)", "平均每 token 延迟 TPOT（毫秒）":"Mean TPOT (ms)", "串行预测耗时（毫秒）":"Serial predicted time (ms)"})

_TRANSLATIONS.update({"平均首字延迟 TTFT（秒）":"Mean TTFT (s)", "平均每 token 延迟 TPOT（秒）":"Mean TPOT (s)", "仅为静态模型源码投影；Serving、正常容量和实际 runtime 尚未验收。":"Static model source projection only; Serving, normal capacity and runtime acceptance are separate.", "历史结果没有保留可视化投影；仅展示已记录的指标与配置，活动区间不可用。":"Historical result has no retained visualization projection; recorded metrics/configuration only, activity unavailable.", "完整模型图":"Complete model graph", "算子调用数":"Operator calls", "未计价调用":"Unpriced calls", "串行预测耗时（秒）":"Serial predicted time (seconds)", "A800 解析 launch 尚不支持性能结论":"A800 analytical launch does not support performance conclusions", "使用开发通信参数；性能结论未验收":"Development communication inputs; performance conclusions unvalidated", "交接元数据覆盖：":"Transfer metadata coverage: ", "总实验耗时（截至结果写入）":"Total experiment wall (through result write)", "结果 JSON 写入耗时":"Result JSON write time", "成本元数据写入另计，完整值见成本记录。":"Cost-metadata writing is measured separately; see the full cost receipt.", "实验状态与配置":"Experiment status and configuration", "本次没有可用的完整时间线；没有生成活动区间或利用率数值。":"A complete timeline is unavailable for this run. No activity intervals or utilization values were generated.", "完整成本记录（JSON）":"Full cost receipt (JSON)", "N/A · 活动区间未记录；Forward 包络不代表忙碌":"N/A: activity intervals not recorded; Forward span is not busy time", "滚轮平移 · Alt＋滚轮缩放 · 拖动导航选区":"Wheel: pan · Alt + wheel: zoom · Drag navigator selection", "全负载模拟跨度":"Full-workload simulated span", "明细覆盖与计数":"Detail coverage and counts", "运行成本 · 资源记录 ":"Run costs · recording ", "记录、投影、序列化和 HTML 均属于报告成本；HTML 生成耗时不能代表完整开销。":"Report cost includes recording, projection, serialization and HTML. HTML generation alone is not total overhead.", "完整报告成本":"Total report cost", "需计入记录、投影、序列化和 HTML；当前未完整记录":"Must include recording, projection, serialization and HTML; not fully recorded", "全负载的起止跨度，不是平均请求端到端延迟。":"Span of the full workload, not mean request end-to-end latency."})

_TRANSLATIONS.update({"物理卡时间线": "Physical-card timeline", "展开细分泳道（默认折叠）": "Detailed lanes (collapsed by default)", "时间分桶": "Time buckets", "瓶颈、逻辑HBM与资源诊断（默认折叠）": "Bottleneck, logical HBM and resource diagnostics (collapsed)", "展开此操作的瓶颈与资源诊断": "Operation bottleneck and resource diagnostics", "保存窗口": "Saved window", "物理卡": "Physical card", "全部卡": "All cards", "统计口径": "Accounting scope", "累计操作时间": "Cumulative operator time", "GPU计算通道（含空闲）": "GPU compute channel (including idle)", "FPGA计算通道（含空闲）": "FPGA compute channel (including idle)", "FPGA DDR通道（含空闲）": "FPGA DDR channel (including idle)", "路由 / 合并 / 本地归约": "Routing / combine / local reduction", "路由专家": "Routed experts", "共享专家": "Shared expert", "DDR搬运": "DDR staging", "其他 / 归一化 / 采样": "Other / normalization / sampling", "占比": "Share", "时长": "Duration", "操作名称": "Operation name", "操作类型": "Operation type"})

_TRANSLATIONS.update({
    "细分时间线": "Timeline Breakdown",
    "单次首层": "Single first-layer invocation",
    "其他并发活动仍按实际时间显示": "Concurrent activity remains visible at its scheduled times",
    "定位一个 decode step、一个 microbatch 的首层执行；保留窗口内的其他并发活动。":
        "Focus one first-layer invocation for one decode step and microbatch; concurrent activity remains visible.",
    "无法定位单次首层：保存的事件缺少 decode step / microbatch 标识。":
        "Cannot locate a single first-layer invocation: saved events lack decode-step/microbatch identifiers.",
    "一张卡一行，颜色表示操作类型；点卡名看细分，滚轮平移，Alt+滚轮缩放。":
        "One row per card; colors identify operations. Click a card name for details, wheel to pan, Alt + wheel to zoom.",
    "FPGA同卡双芯片的计算、DDR与通信分带显示，保留实际重叠与起止时间。点击色块查看操作；拖动时间线可平移。":
        "Each FPGA chip has separate compute, DDR and communication bands, preserving actual overlap and endpoints. Click a block for operation details; drag to pan.",
    "FPGA按芯片分别显示计算、DDR与通信；Host与等待明细保留。":
        "FPGA compute, DDR and communication are shown per chip; host and wait details are retained.",
    "逻辑HBM需求与资源下界不是实测带宽或端到端关键路径；诊断数据保留，未记录字段保持未知。":
        "Logical HBM demand and resource lower bounds are separate from measured bandwidth and the end-to-end critical path. Unrecorded fields remain unknown.",
    "跨设备累计算子时长，重叠活动可重复累计，不是wall-time分解。":
        "Operator durations are summed across devices, including overlaps; they do not partition wall time.",
    "通道累计时长包含空闲；占比以GPU的8×窗口、FPGA各通道的32×窗口归一化，累计时长不是wall-time。":
        "Channel duration includes idle time. Shares use 8 × window for GPUs and 32 × window for each FPGA channel; cumulative duration is not wall time.",
    "饼图统计固定保存窗口，不随上方缩放改变。":
        "The pie chart uses the saved window, independently of timeline zoom.",
    "此保存窗口没有操作明细": "No operation details in the saved window",
    "时间分桶饼图": "Time-bucket pie chart",
    "累计操作时长": "Cumulative operator duration",
    "通道累计时长": "Cumulative channel duration",
    "张物理卡": "physical cards",
    "明细范围": "Detail coverage",
    "完整记录": "Full record",
    "稳态输出吞吐": "Steady-state output throughput",
    "稳态 TPOT": "Steady-state TPOT",
    "总并发": "Total concurrency",
})

_TRANSLATIONS.update({"实测": "Observed", "预测": "Predicted", "已完成 / Case 总数": "Completed / cases", "Serving 结果表": "Serving results", "已保存的 Serving 评估汇总；完成运行不代表精度达标。": "Saved Serving evaluation. Completion does not imply passing accuracy targets.", "完整 Markdown 报告": "Full Markdown report"})

_TRANSLATIONS.update({"Case 明细（点击展开与搜索）": "Case details (expand and search)", "Case 明细预览": "Case preview", "Forward 延迟散点图": "Forward latency scatter", "三硬件 full Serving 总览": "Serving overview across three hardware platforms", "全部 Kernel 延迟散点图": "All-kernel latency scatter", "各算子延迟与误差分布": "Operator latency and error distributions"})

_SCRIPT = r'''<style>
#viewer-language{position:fixed;top:12px;right:20px;z-index:1000;background:#fff;color:#24435e;border:1px solid #b7c9da;border-radius:20px;padding:8px 16px;box-shadow:0 2px 9px #16324715;cursor:pointer;font:13px system-ui}body{visibility:hidden}
</style><script>
(()=>{'use strict';
const embedded=document.documentElement.dataset.embeddedDetail==='true';
const dictionary=__DICTIONARY__,entries=Object.entries(dictionary).sort((a,b)=>b[0].length-a[0].length),originals=new WeakMap(),titles=new WeakMap();let language='zh',updating=false,observer=null;const mutationOptions={childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:['title']};
window.reportLanguage=language;
function translate(value,target=language){const source=String(value);if(target==='zh'){let result=source;for(const [cn,en] of entries)if(en.length>=4)result=result.split(en).join(cn);return result;}let result=source;for(const [cn,en] of entries)result=result.split(cn).join(en);return result.replaceAll('，',', ').replaceAll('。','. ').replaceAll('；','; ').replaceAll('：',': ').replaceAll('（',' (').replaceAll('）',')').replaceAll('＋',' + ');}
window.reportTranslate=translate;
window.renderExperimentConfiguration=function(config){if(embedded)return;let panel=document.querySelector('#experiment-configuration');if(panel)return;panel=document.createElement('details');panel.id='experiment-configuration';panel.className='card';panel.style.margin='16px 0';const heading=document.createElement('summary');heading.style.fontSize='17px';heading.style.fontWeight='650';heading.textContent='实验配置';panel.append(heading);const note=document.createElement('p');note.textContent='来自本次运行保存的配置与来源证据；requested → effective 分别表示请求值和实际采用值。未记录的字段明确列出，不从文件名推断。';panel.append(note);
const table=document.createElement('table');table.style.cssText='width:100%;border-collapse:collapse;font-size:13px';function row(name,value){const tr=document.createElement('tr'),th=document.createElement('th'),td=document.createElement('td');th.textContent=name;th.style.cssText='text-align:left;padding:7px 12px;width:36%;border-bottom:1px solid #e2e9f0;color:#526c82';td.style.cssText='padding:7px 12px;border-bottom:1px solid #e2e9f0;overflow-wrap:anywhere';if(value!==undefined&&value!==null)td.setAttribute('data-no-translate','');td.textContent=value===undefined||value===null?'未记录':typeof value==='object'?JSON.stringify(value):String(value);tr.dataset.name=name;tr.dataset.search=(name+' '+translate(name,'en')+' '+String(value)).toLowerCase();tr.append(th,td);table.append(tr);}
const model=config.model.model_config_provenance,dep=config.deployment,inputs=config.inputs,provenance=config.provenance.execution_provenance;
row('模型',model?(model.model_id||model.architecture||model.model_type):null);row('计算精度',config.model.model_compute_dtype||model?.compute_dtype);row('硬件',Object.keys(config.hardware_display).join(', ')||null);row('部署方式',dep.deployment||dep.deployment_plan?.mode);row('预测方法',dep.prediction_method);row('发射策略',dep.launch_policy);row('Serving 时间模式',dep.serving_timing_mode||config.host_timing.timing_mode);row('请求数量',inputs.request_source?.row_count);row('输入数据路径',inputs.request_source?.input_path);row('输入数据 SHA256',inputs.request_source?.source_sha256);row('运行代码 commit',provenance?.repository?.git_commit);
const runtime=provenance?.runtime_inputs;row('模型执行来源',runtime?.model_execution_version);row('CUDA Graph 模式',runtime?.graph_execution?.configured_mode);
for(const [pool,v] of Object.entries(config.scheduler.scheduler_runtime||{})){row(pool+' · Scheduler runtime',v.version);row(pool+' · TP / PP',String(v.effective_tensor_parallel_size??'未记录')+' / '+String(v.effective_pipeline_parallel_size??'未记录'));row(pool+' · 最大 batch token 数',v.effective_max_num_batched_tokens??v.max_num_batched_tokens);row(pool+' · 最大请求数',v.max_num_seqs);row(pool+' · 最大上下文长度',v.max_model_len);row(pool+' · KV block token 数',v.effective_cache_block_size??v.kv_block_size_tokens);for(const [key,label] of [['prefix_caching','前缀缓存'],['chunked_prefill','分块 Prefill'],['async_scheduling','异步调度']])row(pool+' · '+label+' (requested → effective)',String(v['requested_'+key]??'未记录')+' → '+String(v['effective_'+key]??'未记录'));}
const extra=document.createElement('details'),extraHeading=document.createElement('summary'),extraTable=document.createElement('table');extra.className='config-extra';extraHeading.textContent='全部参数与来源路径';extraTable.style.cssText=table.style.cssText;for(const tr of [...table.rows]){const name=tr.dataset.name;if(!['模型','计算精度','硬件','部署方式','预测方法'].includes(name)&&!['最大 batch token 数','最大请求数','前缀缓存'].some(key=>name.includes(key)))extraTable.append(tr);}extra.append(extraHeading,extraTable);const search=document.createElement('input');search.type='search';search.className='config-search';search.setAttribute('aria-label','Search configuration settings');search.placeholder='Filter settings: TP, cache, SHA256…';let priorOpen=false,lastQuery='';search.oninput=()=>{const query=search.value.trim().toLowerCase();if(!lastQuery&&query)priorOpen=extra.open;for(const tr of [...table.rows,...extraTable.rows])tr.hidden=!!query&&!tr.dataset.search.includes(query);extra.open=query?true:priorOpen;lastQuery=query;};panel.append(search,table,extra);const missing=document.createElement('p');missing.textContent=config.unrecorded.length?'未记录的配置项：'+config.unrecorded.join(', '):'已展示保存的运行配置。';panel.append(missing);
const download=document.createElement('button');download.type='button';download.textContent='下载配置 JSON';download.onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(config,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='midlayer-experiment-configuration.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};panel.append(download);
const names={model:'模型与精度',deployment:'部署与并行',scheduler:'调度器完整配置',inputs:'请求集与输入身份',capacity:'KV 容量与缓存',speculation:'推测解码',host_timing:'Host 时间模型',provenance:'版本与来源证据',hardware_display:'硬件摘要',bandwidth_peaks:'显示带宽峰值',files:'已校验的配置文件快照'};
for(const [key,value] of Object.entries(config)){if(key==='unrecorded')continue;const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent=names[key]||key;pre.textContent=JSON.stringify(value,null,2);pre.style.cssText='white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;font:12px/1.6 ui-monospace,monospace;background:#f4f7fa;padding:12px;border-radius:5px';details.append(summary,pre);panel.append(details);}
const diagnostics=document.querySelector('#bottleneck-diagnostics'),anchor=document.querySelector('.page > .card');if(diagnostics)diagnostics.append(panel);else if(anchor?.querySelector('#metrics'))anchor.append(panel);else if(anchor)anchor.after(panel);else document.body.append(panel);sync();};

const fill=CanvasRenderingContext2D.prototype.fillText,measure=CanvasRenderingContext2D.prototype.measureText;
CanvasRenderingContext2D.prototype.fillText=function(value,...args){return fill.call(this,translate(value),...args);};
CanvasRenderingContext2D.prototype.measureText=function(value){return measure.call(this,translate(value));};
function translateNode(node){if(!node.parentElement||node.parentElement.closest('script,style,pre,[data-no-translate]'))return;const old=originals.get(node);if(old&&old.rendered===node.nodeValue){const rendered=translate(old.source);if(rendered!==node.nodeValue)node.nodeValue=rendered;old.rendered=rendered;return;}if(!/[\u3400-\u9fff]/.test(node.nodeValue))return;const source=node.nodeValue,rendered=translate(source);originals.set(node,{source,rendered});if(rendered!==source)node.nodeValue=rendered;}
function visit(root){if(!root)return;if(root.nodeType===Node.TEXT_NODE){translateNode(root);return;}const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);while(walker.nextNode())translateNode(walker.currentNode);}
function translateTitle(element){const current=element.getAttribute('title');if(current===null)return;const old=titles.get(element),source=old&&old.rendered===current?old.source:current,rendered=translate(source);titles.set(element,{source,rendered});if(rendered!==current)element.setAttribute('title',rendered);}
function translateTitles(root=document.body){if(root.nodeType===Node.ELEMENT_NODE)translateTitle(root);if(root.querySelectorAll)for(const element of root.querySelectorAll('[title]'))translateTitle(element);}
function sync(records=null){if(updating||!document.body)return;updating=true;if(observer)observer.disconnect();try{if(records===null){visit(document.body);translateTitles();}else{const roots=new Set(),attributes=new Set();for(const record of records){if(record.type==='attributes')attributes.add(record.target);else if(record.type==='characterData')roots.add(record.target);else for(const node of record.addedNodes)roots.add(node);}for(const node of roots){if(!node.isConnected)continue;let covered=false;for(let parent=node.parentNode;parent;parent=parent.parentNode)if(roots.has(parent)){covered=true;break;}if(!covered){visit(node);translateTitles(node);}}for(const element of attributes)if(element.isConnected)translateTitle(element);}document.documentElement.lang=language==='en'?'en':'zh-CN';const button=document.querySelector('#viewer-language');if(button){const label=language==='en'?'中文':'English';if(button.textContent!==label)button.textContent=label;const aria=language==='en'?'Switch to Chinese':'切换到英文';if(button.getAttribute('aria-label')!==aria)button.setAttribute('aria-label',aria);}for(const toggle of document.querySelectorAll('[data-viewer-language]')){const label=language==='en'?'中文':'English';if(toggle.textContent!==label)toggle.textContent=label;toggle.onclick=()=>window.setReportLanguage(language==='en'?'zh':'en');}const search=document.querySelector('.config-search');if(search)search.placeholder=translate('按参数名筛选：TP、cache、SHA256…');document.body.style.visibility='visible';}finally{updating=false;if(observer)observer.observe(document.body,mutationOptions);}}

function broadcast(){for(const frame of document.querySelectorAll('iframe'))if(frame.contentWindow)frame.contentWindow.postMessage({type:'midlayer-language',language},'*');}
window.setReportLanguage=function(value){if(!['en','zh'].includes(value))throw new Error('Invalid report language');language=value;window.reportLanguage=value;sync();if(typeof draw==='function'&&(window.swimlaneReady||window.midlayerReportReady))draw();broadcast();};
window.addEventListener('message',event=>{if(event.source===parent&&event.data?.type==='midlayer-language')window.setReportLanguage(event.data.language);});
window.addEventListener('DOMContentLoaded',()=>{const button=document.createElement('button');button.id='viewer-language';button.type='button';button.onclick=()=>window.setReportLanguage(language==='en'?'zh':'en');if(embedded)button.style.display='none';document.body.prepend(button);sync();observer=new MutationObserver(sync);observer.observe(document.body,mutationOptions);for(const frame of document.querySelectorAll('iframe'))frame.addEventListener('load',broadcast);});
})();
</script>'''


def with_serving_viewer_ui(html):
    """Embed offline translations before any viewer scripts execute."""
    dictionary = {key.strip(): value for key, value in _TRANSLATIONS.items()}
    script = _SCRIPT.replace("__DICTIONARY__", json.dumps(dictionary, ensure_ascii=True).replace("<", "\\u003c"))
    return html.replace('<meta charset="utf-8">', '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">' + script, 1)


def attach_report_costs(report, costs):
    """Attach producer-measured costs, keeping replay inputs self-contained.

    The producer must measure this metadata write separately in its final receipt;
    the embedded total is explicitly the earlier result-write boundary.
    """
    import time
    started = time.perf_counter()
    normalized = report_run_context({"visualization_run_costs": costs}, {})["report_costs"]
    path, data_path = Path(report["path"]), Path(report["data_path"])
    payload = json.loads(data_path.read_text())
    payload["visualization_run_costs"] = normalized
    data_path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    encoded = json.dumps(normalized, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    with path.open("a") as stream:
        stream.write('<script type="application/json" id="recorded-report-costs">' + encoded + '</script>')
    report["html_bytes"], report["data_bytes"] = path.stat().st_size, data_path.stat().st_size
    return time.perf_counter() - started


def configuration_file_snapshots(model_path, hardware_paths):
    """Snapshot readable config sources before execution, retaining missingness."""
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / "config.json"
    inputs = [("model_config", model_path)] + [("hardware_config", Path(path)) for path in hardware_paths]
    records, missing = [], []
    for kind, path in inputs:
        if not path.is_file():
            missing.append(kind + " snapshot: " + str(path))
            continue
        content = path.read_text()
        records.append({"kind": kind, "path": str(path.resolve()), "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest()})
    return records, missing
