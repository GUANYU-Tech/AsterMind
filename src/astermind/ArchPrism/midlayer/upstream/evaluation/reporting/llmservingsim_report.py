"""Render an optional, explicitly scoped external-baseline TTFT comparison."""

from __future__ import annotations

import json
import math
import os
import shlex
from pathlib import Path

from evaluation.serving.metrics.llmservingsim_comparison import distribution, finite_nonnegative


MODELS = ("qwen3-4b", "llama-3.1-8b")
SERIES = ("Measured", "Midlayer", "LLMServingSim 2.0")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())



def _native_result_path(directory: Path, comparison: dict) -> Path:
    historical = Path(comparison["native_run_manifest"])
    if (historical.name != "run.json"
            or historical.parts[-5:-2] != (directory.parent.name, directory.name, "llmservingsim")
            or not historical.parent.name.startswith("run-")):
        raise ValueError("accepted native run manifest does not match the case")
    return directory / "llmservingsim" / historical.parent.name / "normalized.json"

def load_ttft_case(directory: Path, model: str, *, hardware_id: str = "ascend_910b",
                   case_id: str = "tp1_qps16") -> dict:
    """Validate the accepted case identities and summaries without running simulation."""
    mapping = _read(directory / "request_mapping.json")["requests"]
    truth = _read(directory / "canonical/metrics.json")["per_request"]
    prediction = _read(directory / "midlayer-request-metrics/prediction.json")["requests"]
    comparison = _read(directory / "comparison.json")
    native = _read(_native_result_path(directory, comparison))["requests"]
    expected = list(range(128))
    if (comparison["case"] != f"{hardware_id}/{model}/{case_id}"
            or comparison["llmservingsim_status"] != "completed"):
        raise ValueError("baseline case identity/status mismatch")
    if ([r["native_request_id"] for r in mapping] != expected
            or [r["request_id"] for r in mapping] != [str(i) for i in expected]
            or len(truth) != 128 or len(native) != 128
            or set(prediction) != {f"request-{i:012d}" for i in expected}):
        raise ValueError("expected exactly 128 matching request identities")
    truth_by_id = {r["request_idx"]: r for r in truth}
    native_by_id = {r["request_id"]: r for r in native}
    if set(truth_by_id) != set(expected) or set(native_by_id) != {str(i) for i in expected}:
        raise ValueError("duplicate, missing or unknown request identity")
    values = {name: [] for name in SERIES}
    for index, request in enumerate(mapping):
        measured = truth_by_id[index]
        simulated = native_by_id[str(index)]
        arrival_ms = finite_nonnegative(request["arrival_time_ms"], "canonical arrival")
        native_arrival_ms = finite_nonnegative(simulated["arrival_time_s"], "native arrival") * 1000
        if (simulated["native_request_id"] != index
                or measured["actual_output_tokens"] != request["output_tokens"]
                or simulated["output_tokens"] != request["output_tokens"]
                or simulated["input_tokens"] != request["input_tokens"]
                or abs(native_arrival_ms - arrival_ms) > 0.000001):
            raise ValueError("request workload identity mismatch")
        values["Measured"].append(finite_nonnegative(measured["ttft_ms"], "measured TTFT") / 1000)
        values["Midlayer"].append(finite_nonnegative(prediction[f"request-{index:012d}"]["ttft_s"], "Midlayer TTFT"))
        values["LLMServingSim 2.0"].append(finite_nonnegative(simulated["ttft_s"], "LLMServingSim TTFT"))
    summaries = {name: distribution(rows) for name, rows in values.items()}
    saved = (comparison["midlayer_vs_truth"]["ttft_s"]["reference"],
             comparison["midlayer_vs_truth"]["ttft_s"]["prediction"],
             comparison["llmservingsim_vs_truth"]["ttft_s"]["prediction"])
    for name, reference in zip(SERIES, saved):
        for key in ("count", "mean", "p50", "p90", "p99"):
            if not math.isclose(summaries[name][key], reference[key], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"{name} TTFT differs from accepted comparison: {key}")
    return {"model": model, "source": str(directory.resolve()), "ttft_s": values,
            "summary_s": summaries, "limitations": comparison["profile_evidence"]["limitations"]}


def render_llmservingsim_section(results_root: Path, output: Path) -> list[str]:
    """Write both model panels once and return Markdown for a scoped report section."""
    cases = [load_ttft_case(results_root / "targets/ascend_910b" / model, model) for model in MODELS]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / "ascend_910b_llmservingsim_ttft_cdf.png"
    pdf = png.with_suffix(".pdf")
    colors = ("#222222", "#0072B2", "#D55E00")
    styles = ("-", "--", "-.")
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
        for ax, case in zip(axes, cases):
            for name, color, style in zip(SERIES, colors, styles):
                ordered = sorted(value * 1000 for value in case["ttft_s"][name])
                ax.step([0.0, *ordered], [0.0, *[(i + 1) / len(ordered) for i in range(len(ordered))]],
                        where="post", label=name, color=color, linestyle=style, linewidth=2)
            ax.set_title({"qwen3-4b": "Qwen3-4B", "llama-3.1-8b": "Llama-3.1-8B"}[case["model"]])
            ax.set(xlabel="TTFT (ms)", xlim=(0, None), ylim=(0, 1))
            ax.grid(alpha=0.2)
            ax.legend(loc="lower right", fontsize=9)
        axes[0].set_ylabel("Cumulative fraction of requests")
        fig.suptitle("Ascend 910B | TP1 | QPS16 | BF16 | 128 requests per model")
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
    summary_path = output.parent / "llmservingsim_ttft_summary.json"
    summary_path.write_text(json.dumps({"unit": "seconds", "cases": cases}, indent=2) + "\n")
    relative = lambda path: os.path.relpath(path, output.parent)
    lines = ["## LLMServingSim 2.0：910B TTFT 分布", "",
        "本节复用已验收的历史对比结果，范围为 Ascend 910B、BF16、TP1、QPS16、两模型各 128 请求；不表示本次重新运行或刷新了全矩阵。Measured、Midlayer 和 LLMServingSim 2.0 使用相同请求 ID、输入/输出长度与到达时间（原生纳秒舍入误差不超过 1 ns）。", "",
        f"![910B three-way TTFT distribution]({relative(png)})", "",
        f"[PDF]({relative(pdf)}) · [逐请求数据与统计]({relative(summary_path)})", "",
        "横轴为首 token 时延（毫秒），纵轴为经验累积分布；坐标从零开始，每条曲线独立排序，不代表同一横坐标对应同一请求。未进行平滑、插补或重新采样。", "",
        "| Model | Method | Requests | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        for name in SERIES:
            stat = case["summary_s"][name]
            lines.append(f"| {case['model']} | {name} | {stat['count']} | " + " | ".join(f"{stat[k] * 1000:.3f}" for k in ("mean", "p50", "p90", "p99")) + " |")
    lines += ["", "LLMServingSim 2.0 的计算后端为实测 LUT，采自单层 dummy-weight/eager 执行，保留插值、外推及软件执行差异；analytical 指网络后端，并非纯解析 Transformer compute。请求与 KV 预算已对齐，但运行时并不完全等价。本图不将 TTFT 差异归因于某一项 Graph/CPU 开销。", "", "输入目录及原比较限制：", ""]
    for case in cases:
        lines += [f"- `{case['source']}`", f"- {case['model']} limitations: `{json.dumps(case['limitations'], ensure_ascii=False)}`"]
    lines += ["", "重生成（仅读取已保存结果，不运行性能实验）：", "", "```bash",
              "python scripts/evaluation/run_standard_evaluation_report.py "
              "--llmservingsim-report-only --llmservingsim-results "
              + shlex.quote(str(results_root.resolve())) + " --output-dir "
              + shlex.quote(str(output.parent.resolve())), "```", ""]
    return lines


def write_llmservingsim_report(results_root: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Serving TTFT 对比报告", "", *render_llmservingsim_section(results_root, output)]
    output.write_text("\n".join(lines) + "\n")
    return output


def _same_canonical_inputs(current: Path, archived: Path, cell: dict) -> bool:
    """Match workload semantics, not only model and display labels."""
    import csv
    current_manifest = _read(current / "manifest.json")
    archived_manifest = _read(archived / "manifest.json")
    for name in ("hardware_id", "model_id", "case_id"):
        if (current_manifest["identity"][name] != cell[name]
                or current_manifest["identity"][name] != archived_manifest["identity"][name]):
            return False
    for name in ("dtype", "tensor_parallel_size", "pipeline_parallel_size",
                 "expert_parallel_size", "target_qps", "capture_enable_prefix_caching",
                 "scheduler_pipeline_depth"):
        if current_manifest["runtime"][name] != archived_manifest["runtime"][name]:
            return False
    if current_manifest["runtime"]["target_qps"] != cell["target_qps"]:
        return False
    with (current / "requests.csv").open(newline="") as stream:
        current_rows = list(csv.DictReader(stream))
    with (archived / "requests.csv").open(newline="") as stream:
        archived_rows = list(csv.DictReader(stream))
    if len(current_rows) != cell["request_count"] or len(current_rows) != len(archived_rows):
        return False
    for now, old in zip(current_rows, archived_rows, strict=True):
        if set(now) != set(old):
            return False
        for field in now:
            a = finite_nonnegative(float(now[field]), "canonical request field")
            b = finite_nonnegative(float(old[field]), "archived request field")
            if a != b:
                return False
    return True


def match_standard_baselines(results_root: Path, serving_payload: dict) -> dict:
    """Return one explicit baseline status for every standard case; never synthesize data."""
    statuses = {}
    for cell in serving_payload["cells"]:
        cell_id = cell["cell_id"]
        if cell_id in statuses:
            raise ValueError("duplicate standard report case")
        hardware, model, case_id = cell["hardware_id"], cell["model_id"], cell["case_id"]
        if cell_id != f"{hardware}/{model}/{case_id}":
            raise ValueError("standard report cell identity mismatch")
        status = {"status": "missing_profile", "reason": "本次已验收性能表缺少该硬件/模型/精度/TP的匹配 LUT；现有表仅覆盖三硬件的 Qwen3-4B/Llama-3.1-8B BF16 TP1，不代表上游架构不支持。",
                  "metrics_ms": None, "summary_ms": None, "source": None}
        statuses[cell_id] = status
        if hardware == "nvidia_h200":
            status.update(status="deferred_by_user", reason="用户暂缓 H200 的 LLMServingSim 实验。")
            continue
        if hardware not in ("nvidia_a800", "metax_c500", "ascend_910b") or model not in MODELS or case_id not in ("tp1_qps4", "tp1_qps16", "tp1_qps64"):
            continue
        if "workload_path" in serving_payload and serving_payload["workload_path"] == "virtual_tensor":
            if cell["prediction_status"] != "completed":
                status.update(status="not_comparable", reason="当前虚拟 case 未完成预测，不能建立逐请求比较。")
                continue
            metrics = cell["metrics"]
            if metrics["comparison_kind"] == "measured_vs_predicted_anonymous_distributions":
                if metrics["paired_request_comparison"] != "not_available_saved_result_omits_request_mapping":
                    raise ValueError("unknown anonymous virtual comparison layout")
                status.update(status="not_comparable", reason="当前虚拟结果仅保存匿名延迟分布，缺少逐请求配对事实；不能叠加为逐请求比较。")
                continue
            if (metrics["comparison_kind"] != "measured_vs_predicted_by_request_idx"
                    or metrics["paired_request_comparison"] != "paired_before_anonymous_projection"):
                raise ValueError("unknown virtual LLMServingSim comparison layout")
        base_directory = results_root / "targets" / hardware / model
        if (base_directory / case_id).is_dir():
            directory = (base_directory / case_id).resolve()
        elif case_id == "tp1_qps16" and (base_directory / "comparison.json").is_file():
            directory = base_directory
        else:
            status.update(status="not_run", reason="已有兼容 TP1 LUT，但该 QPS case 尚未生成完整原生仿真结果。")
            continue
        validated = load_ttft_case(directory, model, hardware_id=hardware, case_id=case_id)
        status["source"] = str(directory.resolve())
        base = (Path(serving_payload["hardware_dataset_roots"][hardware])
                if hardware in serving_payload["hardware_dataset_roots"]
                else Path(serving_payload["dataset_root"]))
        canonical = base / hardware / model / case_id
        if not _same_canonical_inputs(canonical, directory / "canonical", cell):
            status.update(status="input_mismatch", reason="标准 case 与已验收 baseline 的 canonical identity/runtime/request 输入不同，禁止叠加。")
            continue
        paired = cell["metrics"]["paired_request_latency"]
        if paired is None:
            status.update(status="not_comparable", reason="LLMServingSim 有结果，但此标准 case 未生成 Midlayer 逐请求配对指标；不叠加。")
            continue
        comparison = _read(directory / "comparison.json")
        native = _read(_native_result_path(directory, comparison))["requests"]
        native_by_id = {row["request_id"]: row for row in native}
        truth = {row["request_idx"]: row for row in _read(directory / "canonical/metrics.json")["per_request"]}
        metrics, summaries = {}, {}
        matched = True
        for field_ms, field_s in (("ttft_ms", "ttft_s"), ("tpot_ms", "tpot_s"), ("e2e_latency_ms", "latency_s")):
            records = paired["metrics"][field_ms]
            if len(records) != 128 or [r["request_idx"] for r in records] != list(range(128)):
                raise ValueError("standard paired metrics have invalid request identities")
            values = [finite_nonnegative(native_by_id[str(i)][field_s], field_s) for i in range(128)]
            stat = distribution(values)
            reference = comparison["llmservingsim_vs_truth"][field_s]["prediction"]
            for name in ("count", "mean", "p50", "p90", "p99"):
                if not math.isclose(stat[name], reference[name], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("native metric differs from accepted comparison")
            for i, row in enumerate(records):
                observed = finite_nonnegative(row["measured_ms"], "standard measured metric")
                if not math.isclose(observed, truth[i][field_ms], rel_tol=1e-10, abs_tol=1e-8):
                    matched = False
            metrics[field_ms] = [v * 1000 for v in values]
            summaries[field_ms] = {name: stat[name] * 1000 for name in ("mean", "p50", "p90", "p99")}
        if not matched:
            status.update(status="input_mismatch", reason="标准报告与 baseline 的逐请求实测 TTFT/TPOT/E2E 不同，禁止叠加。")
            continue
        status.update(status="available", reason="128请求输入与逐请求实测一致；计算为实测 LUT，保留采集和runtime限制。",
                      metrics_ms=metrics, summary_ms=summaries, limitations=validated["limitations"],
                      native_run_manifest=str(_native_result_path(directory, comparison).with_name("run.json")))
    return statuses


def baseline_status_table(cells, statuses: dict) -> list[str]:
    lines = ["### LLMServingSim 2.0：逐 case 覆盖与误差", "",
             "误差为均值 signed relative error；Missing/deferred 不参与精度均值，也不代表上游 unsupported。TTFT/TPOT 的 available 结果已加入上面的逐请求图与分布图。", "",
             "| Model / Case | LLMServingSim 2.0 | TTFT mean error | TPOT mean error | E2E mean error | 原因 |",
             "|---|---|---:|---:|---:|---|"]
    for cell in cells:
        entry = statuses[cell["cell_id"]]
        errors = []
        for metric in ("ttft_ms", "tpot_ms", "e2e_latency_ms"):
            if entry["status"] == "available":
                observed = sum(row["measured_ms"] for row in cell["metrics"]["paired_request_latency"]["metrics"][metric]) / cell["request_count"]
                error = f"{(entry['summary_ms'][metric]['mean'] / observed - 1) * 100:+.2f}%" if observed > 0 else "undefined (zero mean)"
            else:
                error = "—"
            errors.append(error)
        lines.append(f"| {cell['model_id']} / {cell['case_id']} | {entry['status']} | " + " | ".join(errors) + f" | {entry['reason']} |")
    return lines
