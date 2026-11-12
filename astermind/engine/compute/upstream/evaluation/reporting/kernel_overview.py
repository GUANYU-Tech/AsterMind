"""Offline Kernel overviews over the existing semantic artifact, without prediction."""
from pathlib import Path
import json
import time

from evaluation.reporting.serving_interactive import write_status_report
from evaluation.reporting.serving_viewer_ui import (
    attach_report_costs, report_experiment_configuration, _configuration_metadata,
)


def default_kernel_report_path():
    import tempfile
    root = Path(__file__).resolve().parents[2] / "experiment_logs/evaluation/kernel/reports"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=root)) / "kernel.html"


def write_kernel_overview(artifact, metrics, arguments, output, *, elapsed_s,
                          hardware_snapshot=None, failure=None):
    """Publish recorded comparisons; None stays unavailable and unsupported stays explicit."""
    started = time.perf_counter()
    metrics = json.loads(json.dumps(metrics, allow_nan=False))
    output = Path(output).resolve()
    facts = [{"label": "硬件", "value": arguments["hardware_id"]},
             {"label": "评估模式", "value": arguments["mode"]}]
    links = [{"label": "Kernel 输出成本", "path": str(output.with_suffix(".report.json"))}]
    if failure is not None:
        status = "unsupported" if failure["type"] == "NotImplementedError" else "failed"
        reason = failure["type"] + ": " + failure["message"]
    else:
        if artifact is None:
            raise ValueError("successful Kernel overview requires semantic results")
        facts.append({"label": "Case 总数", "value": artifact["case_count"]})
        status = "complete"
        for method, values in metrics.items():
            if values is None:
                continue
            text = {"ok": "已完成", "unsupported": "不支持"}[values["status"]]
            if values["status"] == "ok":
                text = f"已计分 {values['scored']} / {values['N']} · MAPE {values['MAPE']:.2f}% · 中位数 {values['MedAPE']:.2f}% · P90 {values['P90APE']:.2f}%"
            text += f" · 不支持 {values['unsupported']}" if "unsupported" in values else ""
            text += f" · 超出定价域 {values['out_of_domain']}" if "out_of_domain" in values else ""
            facts.append({"label": method + " 误差 / 覆盖", "value": text})
            if values["status"] != "ok" or ("unsupported" in values and values["unsupported"]) or ("out_of_domain" in values and values["out_of_domain"]):
                status = "partial"
        if not any(value is not None and value["status"] == "ok" for value in metrics.values()):
            status = "unsupported"
        reason = "仅 Kernel 评估；误差仅计已定价 case，unsupported 与 out-of-domain 单列。原始延迟为纳秒，预览用微秒；无 Serving 时间线。"
        # The complete existing artifact is linked; HTML keeps only a bounded preview.
        semantic_path = output.with_suffix(".semantic.json")
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_path.write_text(json.dumps(artifact, allow_nan=False, indent=2) + "\n")
        links.append({"label": "完整 Kernel semantic 结果", "path": str(semantic_path)})
        preview_cases = artifact["cases"][:20]
        case_rows = []
        for case in preview_cases:
            for method in metrics:
                pair = case[method]
                if pair is not None:
                    text = f"{pair[0] / 1000:.3f} / {pair[1] / 1000:.3f} µs" if isinstance(pair, (list, tuple)) else pair
                    case_rows.append({"label": case["case_id"] + " · " + method + "（预测 / 实测）", "value": text})
        if case_rows:
            facts.append({"label": "Case 明细预览", "value": case_rows})
        if artifact["case_count"] > 20:
            facts.append({"label": "预览覆盖", "value": f"仅展示前 {len(preview_cases)} 个 case，全部 case 见 semantic 结果链接。"})
        # First-screen result chart: actual-vs-Forward latency scatter from the
        # same saved pairs the accuracy table scores, colored by kernel family
        # with the existing scatter palette.
        chart_path = output.with_suffix(".latency.png")
        import math

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties, fontManager

        cjk_names = ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei")
        available_fonts = {font.name for font in fontManager.ttflist}
        label_font = FontProperties(family=[name for name in cjk_names if name in available_fonts] + ["DejaVu Sans"])

        from evaluation.kernel.latency_scatter_report import (
            SERIES as _SCATTER_SERIES,
            _family_index as _family_index,
        )
        points = []
        for case in artifact["cases"]:
            pair = case["forward"]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            op_type = case["report_meta"]["op_type"]
            points.append((pair[1] / 1000.0, pair[0] / 1000.0,
                           _family_index(op_type)))
        if points:
            values = [v for point in points for v in point[:2]]
            lower = 10 ** math.floor(math.log10(min(values)))
            upper = 10 ** math.ceil(math.log10(max(values)))
            figure, axis = plt.subplots(figsize=(6.4, 6.0))
            for family_index_i, (_family, color, marker) in enumerate(_SCATTER_SERIES):
                subset = [p for p in points if p[2] == family_index_i]
                if subset:
                    axis.scatter([p[0] for p in subset], [p[1] for p in subset],
                        s=14, c=color, marker=marker, alpha=0.75, linewidths=0, label=_family)
            axis.plot([lower, upper], [lower, upper], color="#666666",
                linewidth=1.0, linestyle="--", label="理想：预测 = 实测")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_xlabel("实测延迟（µs）", fontproperties=label_font)
            axis.set_ylabel("预测延迟（µs）", fontproperties=label_font)
            axis.set_title(arguments["hardware_id"] + " · Forward 预测 vs 实测（" + str(len(points)) + " cases）", fontproperties=label_font)
            axis.legend(prop=label_font, fontsize=8)
            axis.grid(True, which="both", linewidth=0.3, alpha=0.4)
            figure.tight_layout()
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(chart_path, dpi=150, facecolor="white")
            plt.close(figure)
            links.append({"label": "Forward 延迟散点图", "path": str(chart_path)})
    payload = {"main_chain_status": {"status": status, "reason": reason}, "report_facts": facts,
        "report_links": links,
        "visualization_run_costs": {"scope": "Kernel evaluation before report finalization", "evaluation_before_report_s": elapsed_s}}
    config = report_experiment_configuration(payload)
    config["provenance"]["kernel_evaluation"] = {"arguments": _configuration_metadata(arguments),
        "dataset": artifact["dataset"] if artifact is not None else None,
        "hardware_config": artifact["hardware_config"] if artifact is not None else None, "metrics": metrics}
    config["files"] = [] if hardware_snapshot is None else [hardware_snapshot]
    payload["visualization_experiment_configuration"] = config
    receipt = write_status_report(payload, output)
    costs = {"scope": "Kernel main body through HTML and projection write; receipt attachment/write measured separately",
        "evaluation_before_report_s": elapsed_s, "report_generation_s": time.perf_counter() - started,
        "total_wall_through_result_write_s": elapsed_s + time.perf_counter() - started}
    attach_started = time.perf_counter()
    attach_report_costs(receipt, costs)
    costs["cost_metadata_attachment_s"] = time.perf_counter() - attach_started
    receipt["report_output_costs"] = costs
    receipt_path = output.with_suffix(".report.json")
    receipt["cost_receipt_path"] = str(receipt_path)
    receipt_path.write_text(json.dumps(receipt, allow_nan=False, indent=2) + "\n")
    receipt["total_wall_through_report_receipt_s"] = elapsed_s + time.perf_counter() - started
    return receipt


def write_kernel_matrix_overview(payload, configuration, output):
    """Link existing matrix outcomes and child reports without reparsing predictions."""
    started = time.perf_counter()
    output = Path(output).resolve()
    facts, links = [], []
    if "kernel" in configuration:
        for field, label in (("overall_plot_path", "全部 Kernel 延迟散点图"),
                             ("detailed_plot_path", "各算子延迟与误差分布")):
            links.append({"label": label, "path": configuration["kernel"][field]})
    statuses = []
    for hardware_id, entry in payload["kernel_matrix"].items():
        status = entry["status"]
        if status not in ("PASS", "FAIL", "SKIPPED"):
            raise ValueError("unknown Kernel matrix status: " + status)
        statuses.append(status)
        facts.append({"label": hardware_id + " · " + status, "value": entry["summary"]})
    reports = payload["kernel_reports"] if "kernel_reports" in payload else {}
    for hardware_id, report in reports.items():
        if report["status"] == "available":
            links.append({"label": hardware_id + " Kernel overview", "path": report["path"]})
        else:
            facts.append({"label": hardware_id + " report coverage", "value": report["status"]})
    if not reports:
        facts.append({"label": "Per-hardware overview coverage", "value": "Not recorded in this historical matrix"})
    status = "failed" if "FAIL" in statuses else "partial" if "SKIPPED" in statuses else "complete"
    run = {"main_chain_status": {"status": status, "reason": "Kernel matrix overview. Existing accuracy rows and plots remain in the standard Markdown report; Serving results are separate."},
        "report_facts": facts, "report_links": links}
    config = report_experiment_configuration(run)
    config["provenance"]["kernel_report"] = _configuration_metadata(configuration)
    run["visualization_experiment_configuration"] = config
    receipt = write_status_report(run, output)
    receipt["report_output_costs"] = {"report_generation_s": time.perf_counter() - started,
        "scope": "matrix HTML/projection generation and writes; evaluation/plots/Markdown assembly reported separately"}
    path = output.with_suffix(".report.json")
    receipt["cost_receipt_path"] = str(path)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
