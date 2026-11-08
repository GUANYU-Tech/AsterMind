"""Assemble separate canonical Kernel and full-only Serving reports."""

from __future__ import annotations

from evaluation.serving.metrics.sharegpt_evaluation_report import (
    MODEL_RANK,
    has_scheduler_progress_evidence,
)

from datetime import datetime
import json
import os
from pathlib import Path
from statistics import fmean
import time
from typing import Mapping, Sequence
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS

from evaluation.json_shape import require_object as _mapping

from evaluation.json_shape import require_finite_number as _finite

from evaluation.json_shape import require_list_or_tuple as _sequence


HARDWARE_ORDER = EVALUATED_HARDWARE_IDS
# Kernel evaluation also reports nvidia_h200.  The three paper hardware are
# required; the others are rendered from whatever status their card produced
# (PASS, SKIPPED or FAIL) rather than dropped.
KERNEL_HARDWARE_ORDER = (
    "nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
REQUIRED_KERNEL_HARDWARE = HARDWARE_ORDER
_KERNEL_SHORT_LABELS = {
    "nvidia_a800": "A800",
    "nvidia_h200": "H200",
    "metax_c500": "C500",
    "ascend_910b": "910B",
}
# Hardware whose kernel records are NSYS launch facts only (no Nsight Compute
# L2/HBM counters), so they have no measured traffic to compare against.
_LAUNCH_ONLY_KERNEL_HARDWARE = ("nvidia_h200",)
# Serving additionally reports H200 when a run selects an H200 candidate root.
SERVING_HARDWARE_ORDER = (*HARDWARE_ORDER, "nvidia_h200")
HARDWARE_LABELS = {
    "nvidia_a800": "NVIDIA A800",
    "nvidia_h200": "NVIDIA H200",
    "metax_c500": "MetaX C500",
    "ascend_910b": "Ascend 910B",
    "nvidia_h200": "NVIDIA H200",
}
MIDLAYER_WALL_TIME_PHASES = (
    "workload_construction",
    "kernel_lowering",
    "kernel_prediction",
    "timeline_construction",
    "artifact_generation",
    "scheduler_and_orchestration",
)


def write_kernel_evaluation_report(
    output_path: str | Path,
    state_value: Mapping[str, object],
    *,
    with_html: bool = False,
) -> float:
    """Validate and write the standalone Kernel evaluation report."""

    started = time.perf_counter()
    output = Path(output_path).resolve()
    state = _validate_state(state_value)
    kernel = _validate_component(state["kernel"], "kernel")
    kernel_wall_times = _mapping(
        kernel["hardware_evaluation_wall_time_s"],
        "Kernel hardware evaluation wall times",
    )
    if (not set(REQUIRED_KERNEL_HARDWARE) <= set(kernel_wall_times)
            or not set(kernel_wall_times) <= set(KERNEL_HARDWARE_ORDER)):
        raise ValueError("Kernel hardware wall-time set is invalid")
    for hardware_id in kernel_wall_times:
        if _finite(
            kernel_wall_times[hardware_id], f"{hardware_id} Kernel wall time"
        ) < 0.0:
            raise ValueError("Kernel hardware wall time must be nonnegative")
    kernel_payload = _mapping(
        json.loads(Path(str(kernel["result_path"])).read_text()),
        "kernel result",
    )
    traffic_payload = _mapping(
        json.loads(Path(str(kernel["traffic_summary_path"])).read_text()),
        "kernel memory traffic summary",
    )
    lines = _render_kernel_markdown(
        output, state, kernel, kernel_payload, traffic_payload,
        report_assembly_wall_time_s=0.0,
        with_html=with_html,
    )
    if with_html:
        from evaluation.reporting.kernel_overview import write_kernel_matrix_overview
        write_kernel_matrix_overview(kernel_payload, {"state": state, "kernel": kernel}, output.with_suffix(".html"))
    assembly_wall = time.perf_counter() - started
    lines = _render_kernel_markdown(
        output, state, kernel, kernel_payload, traffic_payload,
        report_assembly_wall_time_s=assembly_wall,
        with_html=with_html,
    )
    _write_markdown_atomic(output, lines)
    return time.perf_counter() - started


def write_serving_evaluation_report(
    output_path: str | Path,
    state_value: Mapping[str, object],
    *,
    llmservingsim_results: Path | None = None,
    refresh_latency_plots: bool = False,
) -> float:
    """Validate and write the standalone full-only Serving report."""
    started = time.perf_counter()
    state = _validate_state(state_value)
    serving = _validate_component(state["serving"], "serving")
    payload = json.loads(Path(str(serving["result_path"])).read_text())
    return _write_serving_evaluation_report_payload(output_path, state_value, payload,
        started=started, llmservingsim_results=llmservingsim_results,
        refresh_latency_plots=refresh_latency_plots)


def _write_serving_evaluation_report_payload(
    output_path: str | Path,
    state_value: Mapping[str, object],
    serving_payload: Mapping[str, object],
    *,
    started: float,
    llmservingsim_results: Path | None = None,
    refresh_latency_plots: bool = False,
) -> float:
    """Validate and write the standalone full-only Serving report."""

    output = Path(output_path).resolve()
    state = _validate_state(state_value)
    serving = _validate_component(state["serving"], "serving")
    serving_payload = _mapping(serving_payload, "Serving result")
    cells = _sequence(serving_payload["cells"], "Serving cells")
    if not cells:
        raise ValueError("standard report requires at least one Serving cell")
    baseline_statuses = None
    if llmservingsim_results is not None:
        from evaluation.reporting.llmservingsim_report import match_standard_baselines
        baseline_statuses = match_standard_baselines(llmservingsim_results.resolve(), serving_payload)
        coverage_path = output.parent / "llmservingsim_coverage.json"
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_path.write_text(json.dumps({"source": str(llmservingsim_results.resolve()),
                                             "cases": baseline_statuses}, indent=2, ensure_ascii=False) + "\n")
    if llmservingsim_results is not None or refresh_latency_plots:
        from evaluation.serving.metrics.sharegpt_evaluation_report import write_sharegpt_request_latency_plots
        paired_cells = [cell for cell in cells if cell["prediction_status"] == "completed"]
        if paired_cells:
            write_sharegpt_request_latency_plots(Path(str(serving["result_path"])).parent / "plots",
                                                paired_cells, baseline_by_cell=baseline_statuses)
    lines = _render_serving_markdown(
        output, state, serving, cells, baseline_statuses=baseline_statuses,
        llmservingsim_results=llmservingsim_results,
        report_assembly_wall_time_s=0.0,
    )
    assembly_wall = time.perf_counter() - started
    lines = _render_serving_markdown(
        output, state, serving, cells, baseline_statuses=baseline_statuses,
        llmservingsim_results=llmservingsim_results,
        report_assembly_wall_time_s=assembly_wall,
    )
    _write_markdown_atomic(output, lines)
    return time.perf_counter() - started


def _validate_state(value: object) -> Mapping[str, object]:
    state = _mapping(value, "standard report state")
    if set(state) != {
        "schema_version", "source_commit", "generated_at", "update_mode",
        "kernel", "serving",
    }:
        raise ValueError("standard report state fields are invalid")
    if state["schema_version"] != 1:
        raise ValueError("standard report state schema_version must equal 1")
    return state


def _write_markdown_atomic(output: Path, lines: Sequence[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output)


def _validate_component(value: object, label: str) -> Mapping[str, object]:
    component = _mapping(value, f"{label} component")
    required = {
        "generated_at", "source_commit", "result_path",
        "evaluation_wall_time_s", "plot_wall_time_s",
    }
    if label == "kernel":
        required.update((
            "hardware_evaluation_wall_time_s",
            "traffic_plot_path",
            "traffic_summary_path",
            "traffic_plot_wall_time_s",
        ))
    elif label == "serving":
        pass
    else:
        raise ValueError(f"unknown standard report component: {label}")
    if not required <= set(component):
        raise ValueError(f"{label} component fields are incomplete")
    for field in ("generated_at", "source_commit", "result_path"):
        if not isinstance(component[field], str) or not component[field]:
            raise TypeError(f"{label} {field} must be non-empty text")
    result_path = Path(str(component["result_path"]))
    if not result_path.is_file():
        raise ValueError(f"{label} result does not exist: {result_path}")
    for field in ("evaluation_wall_time_s", "plot_wall_time_s"):
        value = _finite(component[field], f"{label} {field}")
        if value < 0.0:
            raise ValueError(f"{label} {field} must be nonnegative")
    if label == "kernel":
        value = _finite(
            component["traffic_plot_wall_time_s"],
            "kernel traffic plot wall time",
        )
        if value < 0.0:
            raise ValueError("kernel traffic plot wall time must be nonnegative")
    return component


def _kernel_wall_time_table(kernel: Mapping[str, object]) -> list[str]:
    timings = _mapping(
        kernel["hardware_evaluation_wall_time_s"],
        "Kernel hardware evaluation wall times",
    )
    lines = [
        "| Hardware | Dataset evaluation wall time |",
        "|---|---:|",
    ]
    for hardware_id in KERNEL_HARDWARE_ORDER:
        if hardware_id not in timings:
            continue
        lines.append(
            f"| {HARDWARE_LABELS[hardware_id]} | "
            f"{_finite(timings[hardware_id], f'{hardware_id} Kernel wall time'):.3f} s |"
        )
    return lines


def _kernel_traffic_table(payload: Mapping[str, object]) -> list[str]:
    hardware_entries = {}
    for entry in _sequence(payload["hardware"], "traffic hardware"):
        hardware = _mapping(entry, "traffic hardware entry")
        hardware_id = str(hardware["hardware_id"])
        if hardware_id in hardware_entries:
            raise ValueError(f"duplicate traffic hardware entry: {hardware_id}")
        hardware_entries[hardware_id] = hardware
    if set(hardware_entries) != set(HARDWARE_ORDER):
        raise ValueError("traffic summary hardware set is invalid")
    lines = [
        "| Hardware | L2 cases | L2 MAPE | L2 MedAPE | L2 mean P/M | "
        "HBM cases | HBM MAPE | HBM MedAPE | HBM mean P/M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hardware_id in HARDWARE_ORDER:
        metrics = _mapping(
            hardware_entries[hardware_id]["metrics"],
            f"{hardware_id} traffic metrics",
        )
        cells = []
        for metric in ("l2", "hbm"):
            summary = _mapping(metrics[metric], f"{hardware_id} {metric} traffic")
            case_count = summary["case_count"]
            if isinstance(case_count, bool) or not isinstance(case_count, int):
                raise TypeError(f"{hardware_id} {metric} case count must be an integer")
            cells.append(str(case_count))
            cells.append(
                f"{_finite(summary['mape_percent'], f'{hardware_id} {metric} MAPE'):.2f}%")
            cells.append(
                f"{_finite(summary['median_ape_percent'], f'{hardware_id} {metric} MedAPE'):.2f}%")
            cells.append(
                f"{_finite(summary['mean_prediction_over_measured'], f'{hardware_id} {metric} mean P/M'):.3f}×")
        lines.append(f"| {HARDWARE_LABELS[hardware_id]} | {' | '.join(cells)} |")
    return lines


def _kernel_traffic_exclusion_note(payload: Mapping[str, object]) -> list[str]:
    parts = []
    for entry in _sequence(payload["hardware"], "traffic hardware"):
        hardware = _mapping(entry, "traffic hardware entry")
        count = len(_sequence(
            hardware["fa_serving_capture_case_ids"],
            "FA serving capture case ids",
        ))
        if count:
            parts.append(f"{HARDWARE_LABELS[str(hardware['hardware_id'])]} {count} 个")
    if not parts:
        return []
    return [
        "FA serving capture 行只有 op-level 证据，不提供 per-kernel traffic，未参与本图（"
        + "、".join(parts)
        + "）；按 metric 聚合时被过滤的 case 明细见结构化结果。",
        "",
    ]


def _render_kernel_markdown(
    output: Path,
    state: Mapping[str, object],
    kernel: Mapping[str, object],
    kernel_payload: Mapping[str, object],
    traffic_payload: Mapping[str, object],
    *,
    report_assembly_wall_time_s: float,
    with_html: bool = False,
) -> list[str]:
    header = [
        "# Midlayer Single-kernel Forward 标准评估报告",
        "",
    ]
    if with_html:
        header += [
            "离线 Kernel 概要：[HTML](" + str(output.with_suffix(".html")) + ")",
            "",
        ]
    return [
        *header,
        f"生成时间：`{state['generated_at']}`；Kernel 结果时间：`{kernel['generated_at']}`；源码 commit：`{kernel['source_commit']}`。",
        "",
        "## Kernel dataset 运行时间",
        "",
        "| Evaluation wall time | Plot wall time | Traffic plot wall time | Markdown assembly wall time |",
        "|---:|---:|---:|---:|",
        f"| {_finite(kernel['evaluation_wall_time_s'], 'kernel evaluation wall'):.3f} s | {_finite(kernel['plot_wall_time_s'], 'kernel plot wall'):.3f} s | {_finite(kernel['traffic_plot_wall_time_s'], 'kernel traffic plot wall'):.3f} s | {report_assembly_wall_time_s:.3f} s |",
        "",
        *_kernel_wall_time_table(kernel),
        "",
        "## Single-kernel Forward 精度",
        "",
        *_kernel_table(kernel_payload),
        "",
        *_kernel_unsupported_note(kernel_payload),
        f"![各硬件全部 Kernel bucket 总散点]({_relative(output, kernel['overall_plot_path'])})",
        "",
        f"![各硬件逐 bucket scatter 与 OP-level APE CDF]({_relative(output, kernel['detailed_plot_path'])})",
        "",
        "CDF 横轴为累计 OP 比例，纵轴为 absolute percentage error；它不会像 log-log latency scatter 那样在视觉上压缩相对误差。",
        "",
        "## Memory traffic Forward 精度",
        "",
        "每个点是一个 dataset case 的 L2 或 HBM traffic：对该 case 内所有同时具备预测与实测字节的 kernel 分别求和。实测 L2 取 source-interface 字节、HBM 取下游全局内存字节（与 mechanism validation 口径一致）；预测取 Forward 的 `l2_cache` / `dram` 资源 demand。该对比只是 evaluation 诊断，不回流入 Forward 参数。",
        "",
        *_kernel_traffic_table(traffic_payload),
        "",
        *_kernel_traffic_exclusion_note(traffic_payload),
        *_kernel_launch_only_traffic_note(kernel_payload),
        *(
            [f"![三硬件 L2 与 HBM memory traffic 预测-实测散点]({_relative(output, kernel['traffic_plot_path'])})",
             ""]
            if Path(str(kernel["traffic_plot_path"])).is_file() else
            ["本报告为 report-only 重建，输入 semantic artifact 未保留 per-kernel L2/HBM counter "
             "mechanisms，traffic 图需要 `capture_mechanisms=True` 的原始采集运行，本次不生成该图。",
             ""]
        ),
        "## 可复核产物",
        "",
        f"- Kernel 结构化结果：`{kernel['result_path']}`",
        f"- Memory traffic 结构化结果：`{kernel['traffic_summary_path']}`",
        f"- 报告状态：`{output.parent / 'state.json'}`",
        "",
    ]


def _render_serving_markdown(
    output: Path,
    state: Mapping[str, object],
    serving: Mapping[str, object],
    cells: Sequence[object],
    *,
    report_assembly_wall_time_s: float,
    baseline_statuses: Mapping[str, object] | None = None,
    llmservingsim_results: Path | None = None,
) -> list[str]:
    normalized_cells = tuple(_mapping(cell, "Serving cell") for cell in cells)
    hardware_present = {str(cell["hardware_id"]) for cell in normalized_cells}
    unknown_hardware = hardware_present - set(SERVING_HARDWARE_ORDER)
    if unknown_hardware:
        raise ValueError(f"unknown Serving hardware: {sorted(unknown_hardware)}")
    serving_root = Path(str(serving["result_path"])).parent
    timing_modes = sorted({str(cell["serving_timing_mode"]) for cell in normalized_cells
        if cell["prediction_status"] == "completed" and "serving_timing_mode" in cell})
    if not timing_modes:
        timing_modes = [str(serving["serving_timing_mode"]) if "serving_timing_mode" in serving
            else "device_plus_launch_plus_scheduler"]
    timing_description = "、".join(f"`{mode}`" for mode in timing_modes)
    lines = [
        "# Midlayer Full Serving 标准评估报告",
        "",
        f"生成时间：`{state['generated_at']}`；Serving 结果时间：`{serving['generated_at']}`；源码 commit：`{serving['source_commit']}`。",
        "",
        f"Serving timing mode：{timing_description}。主延迟图按同一 request index 配对 measured 与 predicted；排序图用于比较整体分布。报告中的 wall time 是 Midlayer 实际运行耗时，不是被模拟的请求时间。",
        "",
        "## 运行时间",
        "",
        "| Evaluation wall time | Plot wall time | Markdown assembly wall time |",
        "|---:|---:|---:|",
        f"| {_finite(serving['evaluation_wall_time_s'], 'Serving evaluation wall'):.3f} s | {_finite(serving['plot_wall_time_s'], 'Serving plot wall'):.3f} s | {report_assembly_wall_time_s:.3f} s |",
        "",
        "独立 case 会并行运行，因此下面各 case/phase 的累计 CPU wall time不能与父进程 elapsed 直接相加比较。",
        "",
        "## Full Serving 总览",
        "",
        *_serving_summary_table(normalized_cells),
        "",
        "Scored / completed 是精度汇总的实际分母：全部 completed case 都参与统计，各 case 的逐请求配对 MAPE 等权平均。采集差异超过 5% 仅作诊断标签，不阻止运行、不排除 case；历史采集校验结论保留，不改写真值或原 passed 标记。Dev-only 列统计 completed 中 `accuracy_claim_status=unsupported_development_only` 的 case：collective 计时模型本身是生产 ProtoBridge 解析模型，但三硬件的通信 profile 是显式 non-production 的 development 参数（语义映射 gap 清单见 `profiler_output/collective/protobridge_mapping_gaps_20260827.json`），精度解释因此受限；Failed 列为执行失败的 case，Unsupported 列仅统计明确不支持的 case；两者均不进入精度指标。",
        "",
        f"![三硬件 full Serving 总览]({_relative(output, serving_root / 'plots/three_hardware_serving_overview.png')})",
        "",
        "## Midlayer 实际 wall-time breakdown",
        "",
        *_wall_time_table(normalized_cells),
        "",
    ]
    links = []
    for cell in normalized_cells:
        if "interactive_report" in cell:
            receipt = cell["interactive_report"]
            if "path" in receipt:
                links.append("| " + str(cell["cell_id"]) + " | [HTML](" + _relative(output, Path(str(receipt["path"]))) + ") |")
    if links:
        lines.extend(["## 离线交互报告", "", "| Case | Report |", "|---|---|", *links, ""])
    if baseline_statuses is not None:
        available = sum(entry["status"] == "available" for entry in baseline_statuses.values())
        lines.extend((
            "## LLMServingSim 2.0 对比口径", "",
            f"完整标准矩阵 {len(normalized_cells)} 个 case 均列出 baseline 状态，其中 {available} 个 case 有匹配且已校验的 LLMServingSim 2.0 结果，直接叠加到下方 TTFT/TPOT paired scatter 和 sorted distribution。实测及 Midlayer 使用本次标准结果；LLMServingSim 使用已有兼容 LUT 进行 CPU 仿真或复用已验收结果，并逐项核对 canonical 请求、runtime 与实测指标。", "",
            "LLMServingSim 的计算后端为实测 LUT，保留单层 dummy-weight/eager 采集、插值/外推与软件执行差异；analytical 指 network 后端，不是纯解析 Transformer compute。请求/KV预算已对齐但运行时不完全等价。没有其调度轨迹或 host/Wait/Processing 分解的 case 不生成对应曲线。", "",
            f"[逐 case 状态、原始结果位置与比较限制]({_relative(output, output.parent / 'llmservingsim_coverage.json')})", "",
            f"Baseline LUT 来源：[历史比较报告]({_relative(output, llmservingsim_results / 'report.md')})、[独立验收 commit 与范围]({_relative(output, llmservingsim_results / 'main-acceptance.json')})。Midlayer 源码 commit 及仿真时间见本报告顶部；各 case 的原生运行 manifest 见覆盖 JSON；新扩展的 QPS case 不下载权重、不重新采集 LUT。", "",
        ))
    for hardware_id in SERVING_HARDWARE_ORDER:
        hardware_cells = sorted(
            [cell for cell in normalized_cells if cell["hardware_id"] == hardware_id],
            key=_case_sort_key,
        )
        if not hardware_cells:
            continue
        label = HARDWARE_LABELS[hardware_id]
        lines.extend((f"## {label} Serving", ""))
        if hardware_id == "nvidia_h200":
            source = (
                "canonical `serving_dataset/sharegpt/nvidia_h200`"
                if serving["nvidia_h200_dataset_root"] is None
                else "非 canonical 候选数据集 "
                     f"`{serving['nvidia_h200_dataset_root']}`")
            lines.extend((
                f"H200 真值为 vLLM `{serving['nvidia_h200_vllm_version']}` 的 "
                f"ShareGPT 采集（{source}）；Host timing 使用该 vLLM 版本的 "
                "`nvidia_h200_vllm*` profile，其中未在该版本实测的项在 profile "
                "引用的 reference 中以 `substitution` 注明。",
                "",
            ))
        lines.extend((
            "### TTFT：逐请求配对与排序分布",
            "",
            f"![{label} TTFT paired scatter and sorted distribution]({_relative(output, serving_root / 'plots' / f'{hardware_id}_ttft_request_comparison.png')})",
            "",
            "画法：每个 case 的左图按 request index 严格配对，每个点的横轴是该请求的 measured TTFT，纵轴是同一请求的 predicted TTFT，对角线表示完全一致；右图不再保持请求配对，而是将 measured 与 predicted 分别从小到大排序后画成分位曲线。所有坐标从 0 开始。用途：左图检查逐请求偏差、离群点和系统性高估/低估，右图检查两组 TTFT 的整体分布与长尾是否一致。",
            "",
            "### TTFT 版本一：按 measured / predicted 的总量拆分",
            "",
            f"![{label} TTFT wait and processing breakdown]({_relative(output, serving_root / 'plots' / f'{hardware_id}_ttft_breakdown.png')})",
            "",
            "画法：每个 case 先按 measured TTFT 从小到大排列请求，左侧 measured 与右侧 predicted 都沿用这一份顺序，因此左右相同横坐标始终对应同一请求；横轴不是时间顺序，也不是 Wait 分位数。深色区域是 Wait，浅色区域是 Processing，堆叠外边界是该请求的总 TTFT；只有 measured 总边界必然单调，Wait 边界和 predicted 总边界出现上下跳动并不表示时间序列毛刺。右图同时标出 mean signed ΔWait、ΔProcessing 和 ΔTTFT。",
            "用途：保持逐请求配对，在总 TTFT 的背景下判断同一请求的预测误差主要来自 Wait 还是 Processing。Wait 定义为 canonical 计划到达时间到首次进入 Scheduler step，包含首次调度之前的全部入口与排队延迟，并非纯 Scheduler 内部排队；Processing 等于 TTFT 减去 Wait，包含 prefill、调度/launch、Device 执行、Scheduler 收尾和首 token 返回客户端的路径，不等同于纯 Device 时间。",
            "",
            "### TTFT 版本二：按处理 / 排队组件对比",
            "",
            f"![{label} TTFT component-first comparison]({_relative(output, serving_root / 'plots' / f'{hardware_id}_ttft_component_comparison.png')})",
            "",
            "画法：这是同一数据的组件优先视图，左图只画 Processing，右图只画 Wait；蓝色表示 measured，橙色表示 predicted，实线表示 Processing，虚线表示 Wait。每个组件的 measured 与 predicted 分别独立排序（均从小到大），因此同一横坐标表示相同分位点，不保证对应同一请求。",
            "用途：比较 Processing 和 Wait 各自的整体分布、分位误差与长尾形状，不用于判断某一条请求具体差了多少；逐请求配对应看前面的 paired scatter 或版本一。",
            "",
            "### TPOT：逐请求配对与排序分布",
            "",
            f"![{label} TPOT paired scatter and sorted distribution]({_relative(output, serving_root / 'plots' / f'{hardware_id}_tpot_request_comparison.png')})",
            "",
            "### Measured vs predicted 调度推进",
            "",
        ))
        progress_plot = serving_root / "plots" / f"{hardware_id}_scheduler_progress.png"
        if (all(cell["prediction_status"] == "completed" for cell in hardware_cells)
                and has_scheduler_progress_evidence(hardware_cells) and progress_plot.is_file()):
            lines.extend((
                f"![{label} merged scheduler progress]({_relative(output, progress_plot)})",
                "",
            ))
        elif any(cell["prediction_status"] != "completed" for cell in hardware_cells):
            lines.extend((
                "部分 case 执行失败或明确不支持，因此不生成该硬件的合并调度推进图；具体状态与原因见逐 case 表。",
                "",
            ))
        else:
            lines.extend((
                "本轮缺少完整的实测 Scheduler steps 或预测推进证据，未生成合并调度推进图；不会用零值或合成轨迹补齐。",
                "",
            ))
        lines.extend((
            "### 逐 case signed error",
            "",
            *_serving_case_table(hardware_cells),
            "",
            "Status=Dev-only 含义同总览表；Capture Δ 列出 capture 与预测 runtime contract 不一致的字段，`source_commit` 的常驻分歧（预测侧 pin Scheduler 源码 commit，capture 记录完整部署 commit 链）按设计不列入。",
            "",
        ))
        if baseline_statuses is not None:
            from evaluation.reporting.llmservingsim_report import baseline_status_table
            lines.extend((*baseline_status_table(hardware_cells, baseline_statuses), ""))
    lines.extend((
        "## 三硬件 Host timing overhead share",
        "",
        f"![三硬件 host timing overhead share]({_relative(output, serving_root / 'plots/three_hardware_host_timing_overhead_percent.png')})",
        "",
        "## 可复核产物",
        "",
        f"- Serving 结构化结果：`{serving['result_path']}`",
        f"- 报告状态：`{output.parent / 'state.json'}`",
        "",
        "## Serving case 运行时间",
        "",
        f"![三硬件 Serving case evaluation wall time]({_relative(output, serving_root / 'plots/serving_case_evaluation_wall_time.png')})",
        "",
        "每根柱表示该 case 在本次并行批次中的 worker wall time；横向比较可定位长尾，但它不是独占 CPU 条件下的单 case benchmark。",
        "",
        "## Serving case 峰值内存",
        "",
        f"![三硬件 Serving case peak RSS]({_relative(output, serving_root / 'plots/serving_case_peak_rss.png')})",
        "",
        "每根柱表示隔离 case 子进程的峰值 RSS；纵轴使用 GiB 对数刻度，便于同时比较不同模型和请求到达速率。",
        "",
    ))
    if "workload_path" in serving and serving["workload_path"] == "virtual_tensor":
        lines[4:4] = ["虚拟 source 路径的逐请求指标已配对；source/provider 等价及正常容量尚未验收，状态为 Source-unvalidated；记录的 commit 不能单独证明运行源码快照冻结。实测 Scheduler、Host 分解及 CPU wall time 仅使用本次结果和 canonical 中实际存在的证据。", ""]
        rendered = []
        for line in lines:
            if line.startswith("![") and "](" in line and line.endswith(")"):
                label, relative = line[2:-1].split("](", 1)
                if not (output.parent / relative).is_file():
                    reason = ("所选 timing mode 不包含完整 Host 开销或没有已完成的 Host 结果"
                        if "host_timing" in relative else "没有已完成且具备该图所需配对/调度证据的 case")
                    rendered.append(f"{label}：未生成（{reason}）。")
                    continue
            rendered.append(line)
        return rendered
    return lines


def _kernel_card_entries(
        payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    matrix = _mapping(payload["kernel_matrix"], "kernel matrix")
    entries = {}
    for hardware_id in KERNEL_HARDWARE_ORDER:
        if hardware_id not in matrix:
            if hardware_id in REQUIRED_KERNEL_HARDWARE:
                raise ValueError(f"{hardware_id} kernel result is missing")
            continue
        entry = _mapping(matrix[hardware_id], f"{hardware_id} kernel result")
        if (hardware_id in REQUIRED_KERNEL_HARDWARE
                and entry["status"] != "PASS"):
            raise ValueError(f"{hardware_id} kernel evaluation is not PASS")
        entries[hardware_id] = entry
    return entries


def _forward_scored_counts(
        entry: Mapping[str, object], label: str) -> dict[int, int]:
    if "forward_coverage" not in entry or entry["forward_coverage"] is None:
        return {}
    coverage = _mapping(entry["forward_coverage"], f"{label} Forward coverage")
    scored = {}
    for item in _sequence(coverage["scored"], f"{label} scored rows"):
        row_index, count, total = _sequence(item, f"{label} scored row")
        scored[_integer(row_index, f"{label} scored row index")] = _integer(
            count, f"{label} scored count")
    return scored


def _kernel_table(payload: Mapping[str, object]) -> list[str]:
    entries = _kernel_card_entries(payload)
    normalized = {}
    ordered_keys = []
    labels = {}
    for hardware_id, entry in entries.items():
        if entry["status"] != "PASS":
            continue
        scored_counts = _forward_scored_counts(entry, hardware_id)
        rows = _sequence(entry["rows"], f"{hardware_id} kernel rows")
        by_key = {}
        parent = "ALL"
        for row_index, row_value in enumerate(rows):
            row = _sequence(row_value, f"{hardware_id} kernel row")
            if len(row) != 6:
                raise ValueError("kernel matrix row must contain six fields")
            label = str(row[0])
            clean_label = _kernel_row_key(label)
            if label == "ALL" or not label.startswith("│"):
                parent = clean_label
                key = clean_label
            else:
                key = f"{parent}/{clean_label}"
            count = _integer(row[1], f"{hardware_id} {key} count")
            count_text = (
                f"{scored_counts[row_index]}/{count}"
                if row_index in scored_counts else str(count))
            if row[3] is None:
                by_key[key] = (count_text, None)
            else:
                forward = _sequence(
                    row[3], f"{hardware_id} {key} Forward metrics")
                if len(forward) != 3:
                    raise ValueError(
                        "Forward metrics must contain MAPE/MedAPE/WMAE")
                by_key[key] = (
                    count_text,
                    *(
                        _finite(value, f"{hardware_id} {key} metric")
                        for value in forward
                    ),
                )
            labels[key] = label
            if key not in ordered_keys:
                ordered_keys.append(key)
        normalized[hardware_id] = by_key
    if not ordered_keys:
        raise ValueError("kernel matrix is empty")
    header = "| Bucket |"
    for hardware_id in entries:
        short = _KERNEL_SHORT_LABELS[hardware_id]
        header += f" {short} N | {short} MAPE | {short} MedAPE | {short} WMAE |"
    lines = [header, "|---|" + "---:|" * (4 * len(entries))]
    for key in ordered_keys:
        values = []
        for hardware_id, entry in entries.items():
            if entry["status"] != "PASS":
                tag = str(entry["status"]).lower()
                values.extend((tag, tag, tag, tag))
                continue
            if key not in normalized[hardware_id]:
                values.extend(("—", "—", "—", "—"))
                continue
            metrics = normalized[hardware_id][key]
            count = metrics[0]
            if metrics[1] is None:
                values.extend((count, "Err", "Err", "Err"))
                continue
            _, mape, medape, wmae = metrics
            values.extend((
                count, f"{mape:.2f}%", f"{medape:.2f}%", f"{wmae:.2f}%"
            ))
        lines.append(f"| {labels[key]} | {' | '.join(values)} |")
    return lines


def _kernel_unsupported_note(payload: Mapping[str, object]) -> list[str]:
    lines = []
    for hardware_id, entry in _kernel_card_entries(payload).items():
        if entry["status"] != "PASS":
            lines.append(
                f"{HARDWARE_LABELS[hardware_id]} Kernel evaluation "
                f"{entry['status']}：{entry['summary']}。")
            continue
        if "forward_coverage" not in entry or entry["forward_coverage"] is None:
            continue
        coverage = _mapping(entry["forward_coverage"], hardware_id)
        unsupported = _mapping(
            coverage["unsupported_cases"], f"{hardware_id} unsupported cases")
        if not unsupported:
            continue
        parts = "、".join(
            f"{op_type} {count}" for op_type, count in unsupported.items()
            if op_type != "ALL")
        lines.append(
            f"{HARDWARE_LABELS[hardware_id]}：Forward 显式 unsupported 的 case "
            f"共 {unsupported['ALL']} 个（{parts}），不计入 MAPE；N 列的 "
            "`scored/N` 表示该 bucket 实际计分的 case 数。原因明细见该硬件 "
            "evaluation log 的 “Forward unsupported cases” 段。")
    if not lines:
        return []
    return [*lines, ""]


def _kernel_launch_only_traffic_note(
        payload: Mapping[str, object]) -> list[str]:
    present = [
        HARDWARE_LABELS[hardware_id]
        for hardware_id, entry in _kernel_card_entries(payload).items()
        if hardware_id in _LAUNCH_ONLY_KERNEL_HARDWARE
        and entry["status"] == "PASS"
    ]
    if not present:
        return []
    return [
        "、".join(present)
        + " 的 kernel 记录以 NSYS launch 事实为基础，部分记录已附 "
        "Nsight Compute counter sidecar。当前 standard traffic evaluation "
        "尚未消费这些 sidecar，因此暂不出现在本表与本图中。",
        "",
    ]


def _serving_prediction_status(cell: Mapping[str, object]) -> str:
    status = cell["prediction_status"]
    if status not in ("completed", "failed", "unsupported", "unsupported_launch_coverage"):
        raise ValueError(f"unknown Serving prediction status: {status!r}")
    return status


def _serving_summary_table(cells: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| Hardware | Completed / total | Scored / completed | Observer Δ>5% | Dev-only | Failed | Unsupported | Req/s MAPE | TTFT paired MAPE | TPOT paired MAPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hardware_id in SERVING_HARDWARE_ORDER:
        selected = [cell for cell in cells if cell["hardware_id"] == hardware_id]
        if not selected:
            continue
        statuses = [_serving_prediction_status(cell) for cell in selected]
        completed = [cell for cell in selected if cell["prediction_status"] == "completed"]
        scored = completed
        provisional = sum(bool(_observer_delta_metrics(cell)) for cell in completed)
        dev_only = sum(
            1
            for cell in completed
            if cell["accuracy_claim_status"] == "unsupported_development_only"
        )
        failed = statuses.count("failed")
        unsupported = sum(status in ("unsupported", "unsupported_launch_coverage") for status in statuses)
        if not completed:
            lines.append(
                f"| {HARDWARE_LABELS[hardware_id]} | 0 / {len(selected)} | "
                f"0 / 0 | 0 | 0 | {failed} | {unsupported} | Err | Err | Err |"
            )
            continue
        throughput_errors = []
        latency_errors = {"ttft_ms": [], "tpot_ms": []}
        for cell in scored:
            metrics = _mapping(cell["metrics"], "Serving metrics")
            throughput = _mapping(metrics["throughput"], "Serving throughput")
            request_rate = _mapping(
                throughput["request_throughput_per_s"], "request throughput"
            )
            throughput_errors.append(_finite(
                request_rate["absolute_percentage_error"], "request throughput APE"
            ))
            paired = _mapping(metrics["paired_request_error"], "paired errors")
            for metric_name in latency_errors:
                latency_errors[metric_name].append(_finite(
                    _mapping(paired[metric_name], metric_name)["mape_percent"],
                    f"{metric_name} MAPE",
                ))
        lines.append(
            f"| {HARDWARE_LABELS[hardware_id]} | {len(completed)} / "
            f"{len(selected)} | {len(scored)} / {len(completed)} | "
            f"{provisional} | {dev_only} | {failed} | {unsupported} | "
            f"{fmean(throughput_errors):.2f}% | "
            f"{fmean(latency_errors['ttft_ms']):.2f}% | "
            f"{fmean(latency_errors['tpot_ms']):.2f}% |"
        )
    return lines


def _wall_time_table(cells: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| Hardware | Cases | Simulation | Workload | Lowering | Kernel prediction | Timeline | Artifact | Scheduler/orchestration | Validation/comparison |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hardware_id in SERVING_HARDWARE_ORDER:
        selected = [
            cell for cell in cells
            if cell["hardware_id"] == hardware_id
            and cell["prediction_status"] == "completed"
        ]
        if not selected:
            continue
        totals = {phase: 0.0 for phase in MIDLAYER_WALL_TIME_PHASES}
        simulation_total = 0.0
        validation_total = 0.0
        for cell in selected:
            provenance = _mapping(cell["forward_provenance"], "forward provenance")
            timing = _mapping(provenance["midlayer_wall_time"], "Midlayer wall time")
            simulation_wall = _finite(
                timing["simulation_wall_time_s"], "simulation wall time"
            )
            phases = _mapping(timing["phase_wall_time_s"], "Midlayer phases")
            simulation_total += simulation_wall
            for phase in MIDLAYER_WALL_TIME_PHASES:
                totals[phase] += _finite(phases[phase], phase)
            validation_total += max(
                0.0,
                _finite(cell["evaluation_wall_time_seconds"], "case wall time")
                - simulation_wall,
            )
        lines.append(
            f"| {HARDWARE_LABELS[hardware_id]} | {len(selected)} | {simulation_total:.3f}s | "
            f"{totals['workload_construction']:.3f}s | {totals['kernel_lowering']:.3f}s | "
            f"{totals['kernel_prediction']:.3f}s | {totals['timeline_construction']:.3f}s | "
            f"{totals['artifact_generation']:.3f}s | {totals['scheduler_and_orchestration']:.3f}s | "
            f"{validation_total:.3f}s |"
        )
    return lines


_ACCURACY_CLAIM_LABELS = {
    "source_projected_unvalidated": "Source-unvalidated",
    "supported_for_accuracy_interpretation": "OK",
    "unsupported_development_only": "Dev-only",
    "debug_capacity_not_for_accuracy": "Debug-capacity",
    "provisional_observer_effect": "Observer diagnostic",
}
_CAPTURE_DELTA_OMITTED_FIELDS = frozenset({"source_commit"})


def _observer_delta_metrics(cell: Mapping[str, object]) -> tuple[str, ...] | None:
    if "observation_quality" not in cell:
        return None
    quality = _mapping(cell["observation_quality"], "Serving observation quality")
    if "metric_delta_percent" not in quality or quality["metric_delta_percent"] is None:
        return None
    deltas = _mapping(quality["metric_delta_percent"], "observer effect deltas")
    return tuple(sorted(name for name, value in deltas.items()
                        if _finite(value, f"observer delta {name}") > 5.0))


def _capture_delta(cell: Mapping[str, object]) -> str:
    comparison = _mapping(
        cell["capture_runtime_comparison"], "capture runtime comparison"
    )
    fields = _mapping(comparison["fields"], "capture runtime fields")
    mismatches = [
        field
        for field, entry in fields.items()
        if field not in _CAPTURE_DELTA_OMITTED_FIELDS
        and _mapping(entry, f"capture field {field}")["status"] == "mismatch"
    ]
    return ", ".join(mismatches) if mismatches else "—"


def _serving_case_table(cells: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| Model / case | Status | Observer Δ (5% label) | Req/s pred/measured | TTFT signed error | TPOT signed error | TTFT paired MAPE | TPOT paired MAPE | Capture Δ |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        prediction_status = _serving_prediction_status(cell)
        if prediction_status != "completed":
            reason = str(cell["prediction_reason"])
            label = "Failed" if prediction_status == "failed" else "Unsupported"
            lines.append(
                f"| `{cell['model_id']} / {cell['case_id']}` | "
                f"{label}: {reason} | — | Err | Err | Err | Err | Err | — |"
            )
            continue
        claim = str(cell["accuracy_claim_status"])
        if claim not in _ACCURACY_CLAIM_LABELS:
            raise ValueError(f"unknown accuracy claim status: {claim!r}")
        status = _ACCURACY_CLAIM_LABELS[claim]
        observer_metrics = _observer_delta_metrics(cell)
        quality_text = (
            "legacy" if observer_metrics is None
            else "Δ≤5%" if not observer_metrics
            else "Δ>5%: " + ", ".join(observer_metrics)
        )
        capture_delta = _capture_delta(cell)
        metrics = _mapping(cell["metrics"], "Serving metrics")
        throughput = _mapping(
            _mapping(metrics["throughput"], "throughput")["request_throughput_per_s"],
            "request throughput",
        )
        latency = _mapping(metrics["latency"], "latency")
        paired = _mapping(metrics["paired_request_error"], "paired error")
        req_ratio = _finite(throughput["predicted"], "predicted req/s") / _finite(
            throughput["observed"], "measured req/s"
        )
        values = []
        for metric_name in ("ttft_ms", "tpot_ms"):
            mean = _mapping(
                _mapping(latency[metric_name], metric_name)["mean"],
                f"{metric_name} mean",
            )
            observed = _finite(mean["observed"], f"{metric_name} observed")
            predicted = _finite(mean["predicted"], f"{metric_name} predicted")
            values.append((predicted - observed) / observed * 100.0)
        ttft_mape = _finite(
            _mapping(paired["ttft_ms"], "TTFT paired error")["mape_percent"],
            "TTFT MAPE",
        )
        tpot_mape = _finite(
            _mapping(paired["tpot_ms"], "TPOT paired error")["mape_percent"],
            "TPOT MAPE",
        )
        lines.append(
            f"| `{cell['model_id']} / {cell['case_id']}` | {status} | {quality_text} | "
            f"{req_ratio:.3f}× | "
            f"{values[0]:+.2f}% | {values[1]:+.2f}% | "
            f"{ttft_mape:.2f}% | {tpot_mape:.2f}% | {capture_delta} |"
        )
    return lines


def _relative(output: Path, target: object) -> str:
    return Path(os.path.relpath(Path(str(target)).resolve(), output.parent)).as_posix()


def _kernel_row_key(label: str) -> str:
    return (
        label.replace("├", "").replace("└", "").replace("─", "")
        .replace("│", "").strip()
    )


def _case_sort_key(cell: Mapping[str, object]) -> tuple[int, float, str]:
    model_id = str(cell["model_id"])
    if model_id not in MODEL_RANK:
        raise ValueError(f"unknown standard-report model: {model_id}")
    return (
        MODEL_RANK[model_id],
        _finite(cell["target_qps"], "target QPS"),
        str(cell["case_id"]),
    )


def _plot_slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in value
    ).strip("_").lower()








def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def timestamp_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
