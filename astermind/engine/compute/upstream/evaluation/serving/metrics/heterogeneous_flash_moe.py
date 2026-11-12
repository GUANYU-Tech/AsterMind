"""Small CPU-only report for the user-declared Flash GPU/FPGA MoE boundary."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import yaml

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.kernel_lowering.execution_timeline import (
    ExecutionResourceCalendar, build_execution_timeline)
from src.serving.workload.model_config import load_model_config
from src.serving.workload.traced_heterogeneous_moe import HeterogeneousMoETrace, trace_flash_heterogeneous_moe, trace_flash_heterogeneous_decode_layers
from src.serving.workload.traced_heterogeneous_prefill import trace_flash_heterogeneous_prefill
from src.serving.workload.workload_graph import WorkloadKind


LIMITATIONS = (
    "One decode MoE boundary only: Attention/mHC, prefill, KV, sampler and full request are outside this report.",
    "Uniform distinct top6 synthetic routing, seed recorded; not checkpoint-derived router decisions or numerical forward.",
    "FPGA vector32 elements/cycle is a user-declared whole-operation service; compiler/core fusion is unbound.",
    "Quantization uses logical group128 FP32 activation scales and original group32 E8M0 MXFP4 weight scales; physical packing/provider unbound.",
    "BF16 payload only: expert IDs/row IDs/packet framing and protocol overhead are not priced.",
    "DDR bandwidth work is explicit; complete DDR controller/DMA startup remains unknown (not zero).",
    "Network-only service uses the FPGA endpoint's declared50GB/s and20us in both directions; no GPU compute identity is substituted.",
    "GPU NIC HBM reads/writes and the weighted-merge implementation are unbound; BW1100 vector mathematical programs and vector/SFU throughput bindings remain incomplete.",
    "Memory objects have automatic lifetimes and HBM/DDR/buffer domains; physical buffer capacity and normal startup reservation are unbound.",
    "No complete MoE/request latency, QPS, native/runtime validation or normal-capacity acceptance follows from a priced subset.",
)


def _hardware_specs(gpu_vector_memory_only, *, gpu_utilization=1.0, fpga_utilization=1.0,
        gpu_hardware_config="configs/hardware/hygon_bw1100.yaml"):
    """Load canonical hardware; apply the explicit user-approved scenario only."""
    if type(gpu_vector_memory_only) is not bool:
        raise TypeError("gpu_vector_memory_only must be boolean")
    with open(gpu_hardware_config) as source:
        gpu_config = yaml.safe_load(source)
    if gpu_vector_memory_only:
        gpu_config["features"]["gpu_vector_memory_only"] = True
    gpu = HWSpec(gpu_config).with_roofline_utilization(
        compute=gpu_utilization, bandwidth=gpu_utilization)
    if not HWSpec.source_contract_supported("aggregate_gpu_source_roofline", gpu.hardware_id):
        raise ValueError("heterogeneous GPU requires its declared aggregate source Roofline contract")
    return {gpu.hardware_id: gpu,
        "intel_agmf039r47": HWSpec("configs/hardware/intel_agmf039r47.yaml").with_roofline_utilization(
            compute=fpga_utilization, bandwidth=fpga_utilization)}


GPU_APPROXIMATION_LIMITATIONS = (
    "Explicit user-approved approximation: scenario GPU vector/scalar/SFU work contributes only its logical memory traffic; GPU tensor/matrix work and FPGA compute remain priced.",
    "Roofline uses rated bandwidth/compute and dependency/resource contention; unspecified GPU launch, software, DMA/controller and network protocol startup are excluded, not measured as zero.",
    "Predictions are analytical scenario results, not hardware/provider or numerical-checkpoint acceptance.",
)


def _case(arguments):
    concurrency, microbatches, fpga_cards, seed, scope, layer_count, kv_length, gpu_vector_memory_only = arguments
    model = load_model_config("configs/model/deepseek-v4-flash.json")
    if scope == "prefill_model":
        if concurrency != 1 or microbatches != 1 or layer_count != 43:
            raise ValueError("isolated TP8 prefill report requires one prompt/microbatch and all43 layers")
        graph, traces, deployment, outputs = trace_flash_heterogeneous_prefill(model,
            input_ids=tuple(i % model.vocab_size for i in range(kv_length)), root_seed=seed)
        traced = HeterogeneousMoETrace(graph, traces, deployment, (), outputs, (tuple(range(kv_length)),))
        layer_routes = ()
    elif scope == "moe":
        traced = trace_flash_heterogeneous_moe(model, concurrency=concurrency, microbatches=microbatches,
            fpga_cards=fpga_cards, root_seed=seed)
        layer_routes = (traced.routes,)
    elif scope in ("decode_layer", "decode_model"):
        traced = trace_flash_heterogeneous_decode_layers(model, concurrency=concurrency,
            microbatches=microbatches, fpga_cards=fpga_cards, root_seed=seed,
            layer_count=layer_count, kv_length=kv_length, include_model_boundary=scope == "decode_model")
        layer_routes = traced.routes
    else:
        raise ValueError("unknown heterogeneous evaluation scope")
    hardware = _hardware_specs(gpu_vector_memory_only)
    plan = lower_workload(traced.graph, traced.deployment, hardware)
    calls, failures = [], []
    for call in plan.topological_order():
        spec = call.kernel_spec
        record = {"call_id": call.call_id, "hardware": spec.hardware_id, "kind": spec.kind.value,
            "variant": spec.variant, "dependencies": call.dependencies,
            "input_dtype": spec.dtype, "shape": dict(spec.shape),
            "output_dtype": spec.attributes["output_dtype"] if spec.kind is KernelKind.GEMM else None}
        try:
            prediction = predict(spec, hardware[spec.hardware_id], "roofline")
        except NotImplementedError as exc:
            record.update(status="unpriced", seconds=None, reason=str(exc))
            failures.append(record)
        else:
            record.update(status="priced_component", seconds=prediction.t_total_s)
        calls.append(record)
    routes = traced.routes
    count_by_chip = Counter(e // (256 // (2 * fpga_cards)) for layer in layer_routes for row in layer for e in row)
    return {"status": "partial", "concurrency": concurrency, "microbatches_requested": microbatches,
        "microbatches_nonempty": len(traced.microbatch_token_ids), "fpga_cards": fpga_cards,
        "fpga_chips": 2 * fpga_cards, "decode_gpu_cards": 0 if scope == "prefill_model" else 8,
        "prefill_gpu_cards": 8 if scope == "prefill_model" else 0, "seed": seed,
        "scope": scope, "layer_count": 1 if scope == "moe" else layer_count, "kv_length": kv_length,
        "model_output_shapes": tuple(t.shape for t in traced.outputs), "model_output_dtypes": tuple(t.dtype for t in traced.outputs),
        "layer_index": 3 if scope == "moe" else None, "routes": routes, "expert_assignments": kv_length * 6 * 43 if scope == "prefill_model" else concurrency * 6 * len(layer_routes),
        "assignments_by_chip": tuple(count_by_chip[i] for i in range(2 * fpga_cards)),
        "workload_nodes": len(traced.graph.nodes), "kernel_calls": len(plan.calls),
        "priced_calls": len(calls) - len(failures), "unpriced_calls": len(failures),
        "unpriced_by_variant": dict(Counter(call["variant"] for call in failures)),
        "unpriced_by_reason": dict(Counter(call["reason"] for call in failures)),
        "network_bf16_payload_bytes": sum(c.kernel_spec.transfer.bytes for c in plan.calls
            if c.kernel_spec.kind is KernelKind.COMMUNICATION),
        "ddr_combined_read_write_bytes": sum(c.kernel_spec.transfer.bytes for c in plan.calls
            if c.kernel_spec.kind is KernelKind.MEMCPY),
        "full_moe_latency_s": None, "full_request_latency_s": None,
        "all_graph_nodes_lowered": {s for c in plan.calls for s in c.source_node_ids} == {n.node_id for n in traced.graph.nodes},
        "all_fpga_calls_priced": None if scope == "prefill_model" else not any(c["hardware"] == "intel_agmf039r47" for c in failures),
        "fpga_service_calls": sum(c["hardware"] == "intel_agmf039r47" for c in calls),
        "model_config_path": "configs/model/deepseek-v4-flash.json", "model_config_sha256": model.config_sha256,
        "normal_capacity_accepted": False, "calls": calls}


def evaluate_heterogeneous_flash_moe(*, seed=0, scope="moe", layer_count=1, kv_length=1,
        concurrencies=(1, 4, 16, 32), microbatches=(1, 2), fpga_cards=(8, 16),
        output_tokens=3, sdk_python, sdk_source, workers=None,
        scheduler_kv_bytes_per_rank=512 * 1024 * 1024, gpu_vector_memory_only=False):
    if scope == "request":
        if (concurrencies != (1,) or microbatches != (1,)
                or len(fpga_cards) != 1 or fpga_cards[0] not in (8, 16)):
            raise ValueError("request scope requires concurrency1, microbatch1 and exactly one FPGA card count8/16")
        return _evaluate_request(seed=seed, prompt_tokens=kv_length,
            output_tokens=output_tokens, fpga_cards=fpga_cards[0],
            sdk_python=sdk_python, sdk_source=sdk_source,
            scheduler_kv_bytes_per_rank=scheduler_kv_bytes_per_rank,
            gpu_vector_memory_only=gpu_vector_memory_only)
    if scope not in ("moe", "decode_layer", "decode_model", "prefill_model"):
        raise ValueError("unknown heterogeneous report scope")
    if scope in ("decode_model", "prefill_model") and layer_count != 43:
        raise ValueError("decode model report requires all43 unchanged checkpoint layers")
    arguments = tuple((*case, seed, scope, layer_count, kv_length, gpu_vector_memory_only)
        for case in product(concurrencies, microbatches, fpga_cards))
    if not arguments:
        raise ValueError("heterogeneous sweep requires a nonempty case set")
    # Independent CPU metadata cases; the caller controls worker count.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        cases = list(executor.map(_case, arguments))
    scope_limitations = (LIMITATIONS[:7] + LIMITATIONS[8:] + GPU_APPROXIMATION_LIMITATIONS
        if gpu_vector_memory_only else LIMITATIONS)
    limitations = scope_limitations if scope == "moe" else (
        ("Isolated TP8 prefill source, all MoE on GPU; no FPGA offload, PD transfer or request loop." if scope == "prefill_model" else
        "Declared existing KV decode source; no Prefill/PD transfer/request loop or actual Scheduler acceptance."),
        "Native mHC scratch and FlashMLA split/combine remain explicitly unbound; no H200 CU or split count is invented.",
        "Complete checkpoint identity retained; partial layer scopes cannot be reported as a complete model.", *scope_limitations[1:])
    return {"scope": scope, "method": "roofline",
        "status": "partial", "GPU_used": False, "real_model_tensors": False,
        "gpu_vector_memory_only": gpu_vector_memory_only,
        "limitations": limitations, "cases": cases}


def render_report(result):
    if result["scope"] == "request":
        return _render_request_report(result)
    if result["scope"] == "moe":
        scope_text = "独立MoE边界"
    elif result["scope"] == "decode_layer":
        scope_text = "checkpoint前N个完整decoder层"
    elif result["scope"] == "prefill_model":
        scope_text = "43层TP8 GPU Prefill模型，GPU专家保留，不向FPGA卸载"
    else:
        scope_text = "43层decode模型，包含embedding/head/greedy的输入输出图"
    lines = ["# Flash GPU/FPGA 仿真进度", "",
        ("已跑通逐 token 路由、虚拟 Tensor 建图和全部算子转换。GPU向量仅计访存近似已开启，GPU矩阵与FPGA计算保留。此报告为局部建图范围，完整请求时延见request模式。"
            if result["gpu_vector_memory_only"] else
            "已跑通逐 token 路由、虚拟 Tensor 建图和全部算子转换。FPGA 专家/量化/DDR/网络及GPU路由投影/矩阵/打包复制可独立计价；GPU其余向量程序和吞吐绑定仍缺。完整请求和完整时延仍未通过；本报告的具体建图范围见下行。"), "",
        f"本报告范围：{scope_text}。Decode使用8 BW1100 DP8 + FPGA EP16/32；独立Prefill使用8 BW1100 TP8；MTP关闭。仅CPU元数据，不创建真实模型Tensor。", "",
        "| 并发 | microbatch 请求/非空 | FPGA卡 | 调用 | 已计价 | 未计价 |", "|---:|---:|---:|---:|---:|---:|"]
    for c in result["cases"]:
        cards = "不使用" if result["scope"] == "prefill_model" else c["fpga_cards"]
        lines.append(f'| {c["concurrency"]} | {c["microbatches_requested"]}/{c["microbatches_nonempty"]} | {cards} | {c["kernel_calls"]} | {c["priced_calls"]} | {c["unpriced_calls"]} |')
    lines.extend(("", "未计价调用（第一个case）：", ""))
    for variant, count in result["cases"][0]["unpriced_by_variant"].items():
        lines.append(f"- `{variant}`：{count}次。")
    lines.extend(("", "具体拒绝原因（第一个case）：", ""))
    for reason, count in result["cases"][0]["unpriced_by_reason"].items():
        lines.append(f"- {count}次：{reason}。")
    lines.extend(("", "适用范围与缺口：", ""))
    lines.extend("- " + limitation for limitation in result["limitations"])
    return "\n".join(lines) + "\n"


def _evaluate_request(*, seed, prompt_tokens, output_tokens, fpga_cards, sdk_python, sdk_source,
        scheduler_kv_bytes_per_rank, gpu_vector_memory_only):
    """Run a genuine CPU Scheduler driver and the full virtual request source."""
    from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
    from src.serving.execution import execute_heterogeneous_flash_request

    if type(prompt_tokens) is not int or prompt_tokens < 1:
        raise ValueError("request prompt length must be a positive integer")
    if type(output_tokens) is not int or output_tokens < 1:
        raise ValueError("request output length must be a positive integer")
    if type(scheduler_kv_bytes_per_rank) is not int or scheduler_kv_bytes_per_rank <= 0:
        raise ValueError("logical Scheduler KV byte budget must be explicit and positive")
    config_path = "configs/model/deepseek-v4-flash.json"
    model = load_model_config(config_path)
    settings = VLLMSchedulerConfig(config_path, tensor_parallel_size=4,
        max_model_len=max(128, prompt_tokens + output_tokens), scheduler_version="0.30.0",
        enable_chunked_prefill=False, dummy_token_id=42, trust_remote_code=True)
    budget = (scheduler_kv_bytes_per_rank,) * 4
    backend = VLLMBackend.from_vllm030_byte_budget(settings,
        # Preserve the venv executable path: resolving its symlink selects
        # the base interpreter and loses the already installed SDK packages.
        python_executable=Path(sdk_python).absolute(), source_root=Path(sdk_source).resolve(),
        # vLLM requires this capacity >= max_model_len with chunking off;
        # the actual scheduled rows remain prompt length, then one per step.
        pool_name="main", max_batched_tokens=settings.max_model_len, max_requests=1,
        kv_block_size_tokens=256, available_memory_by_rank=budget,
        supported_layouts_by_rank=(("BLHNC", "BLNHC"),) * 4)
    # The actual Scheduler bridge emits declared length-only token IDs. They
    # are source inputs, never a claim to have computed checkpoint outputs.
    program, steps, runtime = execute_heterogeneous_flash_request(model, backend,
        input_ids=(42,) * prompt_tokens, output_tokens=output_tokens,
        fpga_cards=fpga_cards, root_seed=seed)
    hardware = _hardware_specs(gpu_vector_memory_only)
    graph = program.graph
    plan = lower_workload(graph, program.deployment, hardware)
    calls, failures, predictions = [], [], {}
    for call in plan.topological_order():
        spec = call.kernel_spec
        record = {"call_id": call.call_id, "hardware": spec.hardware_id,
            "kind": spec.kind.value, "variant": spec.variant,
            "dependencies": call.dependencies, "shape": dict(spec.shape),
            "input_dtype": spec.dtype,
            "output_dtype": spec.attributes["output_dtype"] if spec.kind is KernelKind.GEMM else None}
        try:
            prediction = predict(spec, hardware[spec.hardware_id], "roofline")
        except NotImplementedError as exc:
            record.update(status="unpriced", seconds=None, reason=str(exc))
            failures.append(record)
        else:
            record.update(status="priced_component", seconds=prediction.t_total_s)
            predictions[call.call_id] = prediction
        calls.append(record)
    scheduler_steps = [{"step_id": step.schedule.step_id,
        "requests": [{"request_id": request.request_id, "phase": request.phase,
            "computed_tokens": request.computed_tokens, "input_tokens": request.input_tokens,
            "output_tokens": request.output_tokens} for request in step.schedule.requests],
        "generated_tokens": dict(step.generated_tokens),
        "finished_request_ids": step.finished_request_ids,
        "embedding_token_source": step.embedding_token_source} for step in steps]
    finished = tuple(request_id for step in steps for request_id in step.finished_request_ids)
    generated = sum(count for step in steps for count in step.generated_tokens.values())
    coverage = {node for call in plan.calls for node in call.source_node_ids} == {node.node_id for node in graph.nodes}
    if generated != output_tokens or program.emitted_tokens != output_tokens or not finished:
        raise RuntimeError("request driver did not finish the declared output length")
    if not coverage:
        raise RuntimeError("full virtual request contains source nodes omitted from lowering")
    prefill_start = 8 + 2 * fpga_cards
    timing = {"full_request_latency_s": None, "qps": None,
        "time_to_first_token_s": None, "mean_time_per_output_token_s": None,
        "output_tokens_per_second": None, "token_completion_times_s": None,
        "timing_basis": "dependency_and_resource_calendar_roofline"}
    if not failures:
        timeline = build_execution_timeline(plan, predictions,
            shared_resource_calendar=ExecutionResourceCalendar())
        calls_by_id = {call.call_id: call for call in plan.calls}
        token_completion_times = tuple(sorted(interval.end_time_s
            for interval in timeline.intervals
            if calls_by_id[interval.call_id].kernel_spec.variant == "vllm_plain_greedy_sample"
            and calls_by_id[interval.call_id].kernel_spec.attributes["stage"] == "output_int32"
            and calls_by_id[interval.call_id].rank in (0, prefill_start)))
        if len(token_completion_times) != output_tokens:
            raise RuntimeError("timed request must contain one selected sampler completion per emitted token")
        duration = timeline.duration_s
        if duration <= 0:
            raise RuntimeError("fully priced request must have positive duration")
        timing.update(full_request_latency_s=duration, qps=1.0 / duration,
            time_to_first_token_s=token_completion_times[0],
            mean_time_per_output_token_s=(None if output_tokens == 1 else
                (token_completion_times[-1] - token_completion_times[0]) / (output_tokens - 1)),
            output_tokens_per_second=output_tokens / duration,
            token_completion_times_s=token_completion_times)
    case = {"status": ("request_roofline_timing_completed" if not failures else
            "request_control_source_completed_timing_partial"), "scope": "request",
        "concurrency": 1, "microbatches_requested": 1, "microbatches_nonempty": 1,
        "fpga_cards": fpga_cards, "fpga_chips": 2 * fpga_cards, "seed": seed,
        "layer_count": model.num_hidden_layers, "prompt_tokens": prompt_tokens,
        "requested_output_tokens": output_tokens, "emitted_tokens": program.emitted_tokens,
        "scheduler_generated_tokens": generated, "scheduler_finished_request_ids": finished,
        "scheduler_steps": scheduler_steps, "source_step_records": program.step_records,
        "prefill_gpu_ranks": tuple(range(prefill_start, prefill_start + 8)),
        "decode_gpu_ranks": tuple(range(8)), "fpga_chip_ranks": tuple(range(8, prefill_start)),
        "handoff_bytes": program.handoff_bytes,
        "handoff_program": None if program.decode_state is None else "full_logical_arena_replica_copy_v1",
        "workload_nodes": len(graph.nodes), "kernel_calls": len(plan.calls),
        "priced_calls": len(calls) - len(failures), "unpriced_calls": len(failures),
        "unpriced_by_variant": dict(Counter(call["variant"] for call in failures)),
        "unpriced_by_reason": dict(Counter(call["reason"] for call in failures)),
        "all_graph_nodes_lowered": coverage,
        "fpga_service_calls": sum(call["hardware"] == "intel_agmf039r47" for call in calls),
        "all_fpga_calls_priced": (None if not any(call["hardware"] == "intel_agmf039r47" for call in calls)
            else not any(call["hardware"] == "intel_agmf039r47" for call in failures)),
        "model_config_path": config_path, "model_config_sha256": model.config_sha256,
        **timing, "normal_capacity_accepted": False,
        "native_provider_accepted": False, "calls": calls}
    return {"scope": "request", "method": "roofline",
        "status": "roofline_timing_completed" if not failures else "partial",
        "gpu_vector_memory_only": gpu_vector_memory_only,
        "request_control_source_completed": True, "roofline_numeric_timing_complete": not failures,
        "GPU_used": False, "real_model_tensors": False, "runtime_info": runtime,
        "scheduler_driver": {"version": "0.30.0", "tensor_parallel_size": 4,
            "role": "colocated", "available_memory_by_rank": budget,
            "kv_block_size_tokens": 256, "normal_capacity_accepted": False,
            "physical_placement_equivalent": False, "dummy_token_id": 42,
            "token_semantics": "declared_length_only_not_numerical_checkpoint_outputs"},
        "limitations": (
            "Real vLLM0.30 CPU Scheduler is a TP4 colocated length driver only; physical Prefill TP8 and Decode DP8/FPGA EP differ.",
            "Full43-layer virtual source executes prefill, explicit cache/token transport and repeated decode; no numerical model computation.",
            "PD copies complete logical cache arenas from one replicated TP source; this is a simulator transport contract, not an installed native PD connector.",
            f"Scheduler uses an explicit synthetic{scheduler_kv_bytes_per_rank} bytes/rank KV budget, not normal model capacity.",
            *(("Unpriced calls remain explicit; complete latency/QPS are null.",) if failures else (
                "QPS is reciprocal single-request makespan; not a saturated multi-request throughput benchmark.",)),
            *(GPU_APPROXIMATION_LIMITATIONS if gpu_vector_memory_only else ()),
            "No native provider/runtime, numerical correctness, normal capacity or W1-W7 acceptance follows from request source/control completion.",
            *LIMITATIONS[1:7], LIMITATIONS[8]), "cases": [case]}


def _render_request_report(result):
    case = result["cases"][0]
    lines = ["# Flash GPU＋FPGA 完整虚拟请求", "",
        "已完成真实CPU Scheduler驱动下的完整虚拟请求，实际Prefill/Decode阶段见下表。每次forward均包含完整43层；全部为CPU虚拟Tensor元数据，未执行真实模型数值计算。", "",
        ("**完整请求Roofline数值时延已跑通。** 全部调用已计价，按依赖图和共享资源日历计算。"
            if result["roofline_numeric_timing_complete"] else
            "**请求控制与源码图执行完成；Roofline完整数值时延尚未完成。** 未计价调用保留，完整时延和QPS均为null。"), "",
        ("GPU向量仅计访存：已开启；GPU矩阵、FPGA计算/向量/固定延时保留。"
            if result["gpu_vector_memory_only"] else "GPU向量仅计访存：未开启（严格模式）。"), "",
        f'输入{case["prompt_tokens"]} tokens，目标输出{case["requested_output_tokens"]} tokens；Scheduler实际生成{case["scheduler_generated_tokens"]} tokens，结束ID：`{case["scheduler_finished_request_ids"]}`。', "",
        f'物理Prefill GPU ranks：`{case["prefill_gpu_ranks"]}`（TP8）；Decode GPU ranks：`{case["decode_gpu_ranks"]}`（DP8）；FPGA芯片ranks：`{case["fpga_chip_ranks"]}`。三组设备互不重叠。', "",
        "实际CPU Scheduler仅为TP4 colocated请求长度驱动，不代表物理TP8/DP8的原生PD Scheduler或provider验收。", "",
        "| 调用 | 已计价 | 未计价 | FPGA调用全计价 | KV/token交接字节 |", "|---:|---:|---:|---|---:|",
        f'| {case["kernel_calls"]} | {case["priced_calls"]} | {case["unpriced_calls"]} | {case["all_fpga_calls_priced"]} | {case["handoff_bytes"]} |', ""]
    if result["roofline_numeric_timing_complete"]:
        tpot = case["mean_time_per_output_token_s"]
        lines.extend(("", f'完整请求：**{case["full_request_latency_s"]:.6f} s**；首token：**{case["time_to_first_token_s"]:.6f} s**。',
            f'平均后续token间隔：**{tpot:.6f} s**。' if tpot is not None else "单输出token，无后续token间隔。",
            f'单请求等效QPS：{case["qps"]:.6f}；输出吞吐：{case["output_tokens_per_second"]:.6f} token/s。', ""))
    lines.extend(("| 源码阶段 | Query tokens | KV长度 | 新生成tokens |", "|---|---:|---:|---:|"))
    for step in case["source_step_records"]:
        lines.append(f'| {step["phase"]} | {step["query_tokens"]} | {step["kv_length"]} | {step["generated_tokens"]} |')
    lines.extend(("", "未计价调用：", ""))
    lines.extend(f'- `{variant}`：{count}次。' for variant, count in case["unpriced_by_variant"].items())
    lines.extend(("", "具体拒绝原因：", ""))
    lines.extend(f'- {count}次：{reason}。' for reason, count in case["unpriced_by_reason"].items())
    lines.extend(("", "范围与保留缺口：", ""))
    lines.extend("- " + limitation for limitation in result["limitations"])
    return "\n".join(lines) + "\n"
