"""Project actual D-only steady-window events into the existing Serving viewer.

The wrapper below is report metadata, not a native Scheduler step. It never
predicts work, changes dependencies, clips event endpoints or invents events.
"""
from dataclasses import asdict
from html import escape
import math
from pathlib import Path
import re
from urllib.parse import quote

from evaluation.reporting.serving_interactive import _operation_display, write_interactive_report
from evaluation.serving.simulation.execution_artifact import _execution_timeline_manifest
from src.serving.kernel_lowering.execution_timeline import ExecutionTimeline
from src.serving.kernel_lowering.kernel_plan import KernelPlan


def _source_operation_display(call):
    sources = call["source_node_ids"]
    layers = {int(match[1]) for source in sources for match in
        re.finditer(r"(?:^|:)layer-(\d+)(?=:|$)", source)}
    layer = next(iter(layers)) if len(layers) == 1 else None
    for fragment, category in (("expert-group:w13", "MoE GEMM W13"),
            ("expert-group:w2", "MoE GEMM W2"), ("local-sum", "本地归约"),
            ("expert-dispatch", "MoE 传入"), ("expert-combine", "MoE 传出"), (":shared", "共享专家")):
        if any(fragment in source for source in sources):
            return category, layer
    category, _ = _operation_display(call)
    return category, layer


def write_flash_decode_steady_html(plan, predictions, timeline, metrics, *, output,
        configuration, detail_interval):
    """Display all real events intersecting the caller's two-cycle interval.

    Validate the complete input through the existing execution-artifact owner
    before selecting a detail projection. Repeated occurrences, when present,
    must already have real intervals in the canonical timeline. The original
    plan and its dependencies remain unchanged; no sub-plan is reconstructed.
    Full-request statistics and TTFT are unrecorded, not observed zeros.
    """
    if not isinstance(plan, KernelPlan) or not isinstance(timeline, ExecutionTimeline):
        raise TypeError("steady HTML requires a canonical KernelPlan and ExecutionTimeline")
    if not isinstance(configuration, dict) or not isinstance(metrics, dict):
        raise TypeError("steady HTML configuration and metrics must be dictionaries")
    required = ("steady_window_start_s", "steady_window_end_s", "steady_output_tokens_per_s",
        "steady_tpot_mean_s", "concurrency", "simulation_wall_s")
    if any(name not in metrics for name in required):
        raise ValueError("steady HTML requires the complete recorded steady metric fields")
    for name in required:
        value = metrics[name]
        if name == "concurrency":
            if type(value) is not int or value <= 0:
                raise ValueError("steady HTML concurrency must be a positive integer")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("steady HTML metrics must be finite nonnegative recorded values")
    if metrics["steady_output_tokens_per_s"] <= 0 or metrics["steady_tpot_mean_s"] <= 0:
        raise ValueError("steady HTML throughput and TPOT must be positive recorded values")
    if (not isinstance(detail_interval, tuple) or len(detail_interval) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in detail_interval)):
        raise ValueError("steady HTML detail interval must be a finite start/end tuple")
    start, end = detail_interval
    steady_start, steady_end = metrics["steady_window_start_s"], metrics["steady_window_end_s"]
    if not 0 <= steady_start <= start < end <= steady_end <= timeline.duration_s:
        raise ValueError("steady HTML detail interval must lie inside the actual steady window")
    # This owner checks full call/prediction coverage, placement, duration,
    # resource reservations and dependency completion before serialization.
    recorded = _execution_timeline_manifest(plan, timeline, predictions, origin_time_s=0.0)

    def intersects(row):
        left, right = row["absolute_start_time_s"], row["absolute_end_time_s"]
        return start <= left < end if left == right else left < end and right > start

    repeated_ids = {call_id for block in timeline.block_repeats for call_id, _ in block.call_intervals}
    ordinary = [row for row in recorded["intervals"] if row["call_id"] not in repeated_ids]
    selected = [row for row in ordinary if intersects(row)]
    host = [row for row in recorded["host_events"] if intersects(row)]
    selected_ids = {row["call_id"] for row in (*selected, *host)}
    blocks, repeated_events, selected_repeated_events = [], 0, 0
    if timeline.block_repeats:
        for block in recorded["block_repeats"]:
            calls = []
            for call in block["call_intervals"]:
                repeated_events += len(call["intervals"])
                intervals = [row for row in call["intervals"] if intersects(row)]
                if intervals:
                    calls.append({"call_id": call["call_id"], "intervals": intervals})
                    selected_ids.add(call["call_id"])
                    selected_repeated_events += len(intervals)
            if calls:
                blocks.append({"block_id": block["block_id"], "repeat_count": block["repeat_count"],
                    "call_intervals": calls})
    if not selected_ids:
        raise ValueError("steady HTML detail interval contains no recorded events")
    calls, projected_predictions = [], []
    for call in plan.calls:
        if call.call_id not in selected_ids:
            continue
        spec, prediction = call.kernel_spec, predictions[call.call_id]
        calls.append({"call_id": call.call_id, "pool_name": call.pool_name, "rank": call.rank,
            "resource": call.resource, "source_node_ids": call.source_node_ids,
            "dependencies": call.dependencies,
            "kernel_spec": {"kind": spec.kind.value, "hardware_id": spec.hardware_id,
                "shape": dict(spec.shape), "dtype": spec.dtype, "variant": spec.variant,
                "transfer": None if spec.transfer is None else asdict(spec.transfer)}})
        category, layer = _source_operation_display(calls[-1])
        calls[-1].update(display_category=category, layer=layer)
        projected_predictions.append({"call_id": call.call_id, "t_total_s": prediction.t_total_s,
            "t_launch_s": prediction.t_launch_s, "bottleneck_group": prediction.bottleneck_group,
            "t_per_resource_s": dict(prediction.t_per_resource)})
    detail_timeline = {"intervals": selected, "host_events": host, "block_repeats": blocks}
    detail_events = len(selected) + selected_repeated_events + len(host)
    total_events = len(ordinary) + repeated_events + len(recorded["host_events"])
    payload = {"execution_artifact": {"steps": [{"step_id": "steady_decode",
        "pool_name": "declared_D_only_source", "schedule_interval": {
            "start_time_s": start, "end_time_s": end},
        "timing": {"scope": "synthetic_report_wrapper_of_actual_D_only_events_not_native_Scheduler"},
        "execution_timeline": detail_timeline, "kernel_plan": {"calls": calls},
        "predictions": projected_predictions, "backend_step": {"schedule": {"requests": []}}}],
        "transfers": [], "schedule_trace": {"route_decisions": []}},
        "metrics": {"requests": {}, "request_throughput_per_s": None},
        "simulation_wall_time_s": metrics["simulation_wall_s"], "schedule_end_time_s": timeline.duration_s,
        "prediction_method": "recorded_analytical_predictions", "launch_policy": "declared_D_only_source",
        "visualization_coverage": {"detail_events": detail_events, "total_events": total_events,
            "detail_intervals_s": [[start, end]], "detailed_requests": 0, "total_requests": metrics["concurrency"],
            "detailed_steps": 1, "total_steps": 1,
            "selection": "all actual intervals intersecting selected steady two-cycle window; endpoints unchanged",
            "activity_scope": "actual selected Device/Host events; no native Scheduler or request-completion statistics",
            "resource_scope": "per-operator physical resource evidence not supplied by this adapter",
            "activity_availability": "available"},
        "visualization_experiment_configuration": {"model": {},
            "deployment": {"deployment": "declared_D_only_source",
                "actual_scenario_configuration": configuration},
            "scheduler": {}, "inputs": {"source": "declared_prefilled_KV_and_symbolic_D_token"},
            "capacity": {}, "speculation": {}, "provenance": {}, "host_timing": {},
            "hardware_display": {}, "bandwidth_peaks": None, "files": [],
            "steady_metrics": {name: metrics[name] for name in required},
            "selected_detail_interval_s": [start, end],
            "source_scope": "actual_interval_projection_with_synthetic_wrapper_not_native_Scheduler",
            "unrecorded": ["TTFT", "full_request_completion_statistics", "request_QPS",
                "native_provider", "per_operator_physical_resource_evidence"]}}
    return write_interactive_report(payload, output)


def write_flash_decode_experiment_report(output_dir, results, capacities):
    """Write the frozen 14-result/18-capacity tables; never price missing work."""
    from evaluation.serving.metrics.flash_decode_steady import HARDWARE, SCHEMES, case_settings

    if not isinstance(results, (list, tuple)) or not isinstance(capacities, (list, tuple)):
        raise TypeError("experiment report requires recorded result and capacity sequences")
    output = Path(output_dir)
    expected_results = set(case_settings())
    expected_capacities = {(hardware, scheme, mb) for hardware in HARDWARE for scheme in SCHEMES for mb in (1, 2, 3)}
    recorded_results, recorded_capacities = {}, {}
    for records, target in ((results, recorded_results), (capacities, recorded_capacities)):
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("experiment report records must be dictionaries")
            key = tuple(record[name] for name in ("hardware_id", "scheme", "microbatches"))
            if key in target:
                raise ValueError("experiment report contains duplicate case identities")
            target[key] = record
    if set(recorded_results) != expected_results or set(recorded_capacities) != expected_capacities:
        raise ValueError("experiment report requires exactly the frozen14 results and18 capacity combinations")
    hardware_labels = {"hygon_bw1100": "BW1100", "nvidia_rtx_pro_5500": "RTX PRO5500"}
    scheme_labels = {"tp8_ep8": "TP8/EP8", "dp8_ep8": "DP8/EP8", "dp8_fpga": "DP8＋16卡FPGA（32芯片）"}
    performance_rows, capacity_rows = [], []
    for hardware in HARDWARE:
        for scheme in SCHEMES:
            for mb in (1, 2, 3):
                key = (hardware, scheme, mb)
                capacity = recorded_capacities[key]["capacity"]
                if (capacity["deployment"] != scheme or capacity["concurrency"] != 128 * mb
                        or capacity["initial_context_tokens"] != 100000 or capacity["decode_tokens"] != 32
                        or capacity["normal_capacity_accepted"] is not False):
                    raise ValueError("capacity input disagrees with the declared100K/32-token experiment")
                if len(capacity["gpu_ranks"]) != 8:
                    raise ValueError("capacity report requires all8 actual GPU-rank inventories")
                known = max(rank["known_residency_inventory_bytes"] for rank in capacity["gpu_ranks"])
                maximum = capacity["conditional_static_max_initial_context_tokens"]
                if type(known) is not int or known < 0 or (maximum is not None and
                        (type(maximum) is not int or not 0 <= maximum <= 1048576 - 32)):
                    raise ValueError("capacity report received invalid recorded bytes or context bound")
                if capacity["status"] not in ("capacity_rejected", "known_inventory_fits_workspace_and_reserve_unbound"):
                    raise ValueError("capacity report received an unknown capacity status")
                capacity_rows.append((hardware_labels[hardware], scheme_labels[scheme], str(mb),
                    str(capacity["concurrency"]), f"{known / 1e9:.4f}",
                    "null" if maximum is None else str(maximum), capacity["status"]))
                if key not in expected_results:
                    continue
                result = recorded_results[key]
                status = result["status"]
                link = None
                if status == "analytical_completed":
                    metrics = result["steady_metrics"]
                    if (not isinstance(metrics, dict) or result["concurrency"] != capacity["concurrency"]
                            or metrics["concurrency"] != capacity["concurrency"]
                            or result["initial_context_tokens"] != 100000
                            or result["actual_decode_tokens_per_request"] != 32 or result["target_layers"] != 43):
                        raise ValueError("completed steady result disagrees with full43-layer frozen workload")
                    tpot, throughput = metrics["steady_tpot_mean_s"], metrics["steady_output_tokens_per_s"]
                    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                            for value in (tpot, throughput)):
                        raise ValueError("completed steady result requires positive recorded TPOT and throughput")
                    tpot_text, throughput_text = f"{tpot * 1000:.3f}", f"{throughput:.3f}"
                    if "html" in result:
                        page = Path(result["html"]["path"]).resolve()
                        if not page.is_file():
                            raise ValueError("recorded case timeline HTML is missing")
                        link = quote(page.relative_to(output.resolve()).as_posix(), safe="/")
                elif status in ("failed", "capacity_rejected", "complete_step_checkpoint"):
                    if "steady_metrics" in result and result["steady_metrics"] is not None:
                        raise ValueError("unfinished case cannot supply completed steady metrics")
                    tpot_text = throughput_text = "null"
                else:
                    raise ValueError("experiment report received an unknown case status")
                performance_rows.append(((hardware_labels[hardware], scheme_labels[scheme], str(mb),
                    str(capacity["concurrency"]), tpot_text, throughput_text), link, status))
    headers = ("硬件", "部署", "MB", "总并发", "稳态TPOT ms", "稳态输出token/s", "时间线", "状态")
    cap_headers = ("硬件", "部署", "MB", "总并发", "100K已知GB/卡（最大rank）", "条件最大初始context", "状态")
    intro = "DeepSeek-V4-Flash，完整43层；初始100000 tokens，32个连续decode输出。GPU均为8卡；FPGA方案为16张双芯片卡，共32芯片。每个MB为128请求，MB1/2/3总并发128/256/384。下表只读取实际记录的稳态指标，失败/未完成项保持null。"
    method = "所有矩阵权重均按FP4值＋group32 scale存储；BW1100使用FP8矩阵计算，RTX使用原生FP4，GPU算力/本地带宽分别取峰值0.5/0.7。FPGA算力不打折，HBM与DDR带宽取峰值0.6。集合通信每次启动50µs；无Ring，TP归约使用直接全互联广播＋GPU本地求和。RTX GPU互联采用两组各4卡PCIe5.0×16交换域，每域一个共享×16上行到非阻塞根节点。"
    measurement = "稳态窗口固定排除每请求前8与后8个输出：吞吐按公共稳态时间窗实际输出计数，TPOT按同请求连续完成间隔；短窗口吞吐含少量端点计数误差。GPU向量按逻辑访存近似，NIC DMA对GPU HBM的额外争用未独立计价。"
    boundary = "容量使用FP4矩阵值＋group32 scale、非矩阵参数、resident KV页及明确静态buffer清单；条件最大初始context已扣32个输出，模型总长封顶1048576。未知workspace/runtime reserve仍未知，不能当0称部署通过；结果为解析情景，非native provider或实卡精度验收。"
    markdown = ["# Flash decode 稳态对照", "", intro, "", "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers)]
    html_performance = []
    for values, link, status in performance_rows:
        markdown.append("| " + " | ".join((*values, "null" if link is None else f"[timeline.html]({link})", status)) + " |")
        cells = "".join("<td>" + escape(value) + "</td>" for value in values)
        cells += "<td>" + ("null" if link is None else f'<a href="{escape(link, quote=True)}">timeline.html</a>') + "</td>"
        html_performance.append("<tr>" + cells + "<td>" + escape(status) + "</td></tr>")
    markdown += ["", "| " + " | ".join(cap_headers) + " |", "|" + "---|" * len(cap_headers)]
    markdown.extend("| " + " | ".join(row) + " |" for row in capacity_rows)
    markdown += ["", method, "", measurement, "", boundary, ""]
    def table(titles, rows):
        return "<table><thead><tr>" + "".join("<th>" + escape(title) + "</th>" for title in titles) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    html_capacity = ["<tr>" + "".join("<td>" + escape(value) + "</td>" for value in row) + "</tr>" for row in capacity_rows]
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Flash decode 稳态对照</title>'
        '<style>body{font:14px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#253747}'
        'table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:8px;border:1px solid #d8e1e7;text-align:left}'
        'th{background:#eef3f7}p{line-height:1.6}a{color:#24649a}</style><h1>Flash decode 稳态对照</h1><p>'
        + escape(intro) + '</p>' + table(headers, html_performance) + table(cap_headers, html_capacity)
        + '<p>' + escape(method) + '</p><p>' + escape(measurement) + '</p><p>' + escape(boundary) + '</p></html>')
    output.mkdir(parents=True, exist_ok=True)
    report_path, index_path = output / "serving_report.md", output / "index.html"
    report_path.write_text("\n".join(markdown))
    index_path.write_text(page)
    return {"serving_report_path": str(report_path), "index_path": str(index_path),
        "performance_cases": len(performance_rows), "capacity_cases": len(capacity_rows)}
