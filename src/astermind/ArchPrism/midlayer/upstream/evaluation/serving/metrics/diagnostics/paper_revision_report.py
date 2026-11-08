"""Render revision figures and verify sensitivity control cases."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#3477a4", "#333333", "#c65a35")


def read(path):
    return json.loads(Path(path).read_text())


def render(root, figure_dir):
    root, figure_dir = Path(root), Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data = root / "experiment_logs/paper_revision"
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight",
    })
    cases = {c["cell_id"]: c for c in read(data / "official_cells.json")}
    studies = []
    for path in sorted((data / "sensitivity").glob("*.json")):
        value = read(path)
        if "summary" in value:
            studies.append(value)
    if len(studies) != 21:
        raise ValueError("revision sensitivity figure requires exactly 21 studies")
    by_key = {
        (d["intervention"]["kind"], int(d["case_id"].split("qps")[-1]),
         d["intervention"]["factor"]): d for d in studies}
    rows = []
    for d in studies:
        kind, factor = d["intervention"]["kind"], d["intervention"]["factor"]
        qps = int(d["case_id"].split("qps")[-1])
        baseline_kind = "l2_scale" if kind == "reserve" else kind
        baseline = by_key[(baseline_kind, qps, 1.0)]["summary"]
        if factor == 1 and kind != "reserve":
            for metric in ("ttft_ms", "tpot_ms"):
                if d["summary"][metric] != (
                        cases[d["case_id"]]["metrics"]["latency"][metric]
                        ["mean"]["predicted"]):
                    raise ValueError("sensitivity control differs from the official baseline")
        row = {"case_id": d["case_id"], "kind": kind, "factor": factor}
        row.update(d["summary"])
        for key in ("ttft_ms", "tpot_ms", "request_throughput_per_s"):
            row[key + "_delta_percent"] = (
                100 * (d["summary"][key] / baseline[key] - 1))
        row["num_kv_blocks"] = d["kv_capacity"]["scheduler_pools"][0]["num_kv_blocks"]
        rows.append(row)
    with (data / "sensitivity/summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Show small and nonmonotone effects without concealing axis scales.
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.55))
    for ax, metric, title in zip(
            axes, ("ttft_ms", "tpot_ms", "request_throughput_per_s"),
            ("Mean TTFT", "Mean TPOT", "Request throughput")):
        for color, qps in zip(COLORS, (4, 16, 64)):
            selected = sorted(
                (r for r in rows if r["kind"] == "l2_scale"
                 and r["case_id"].endswith(f"qps{qps}")),
                key=lambda r: r["factor"])
            ax.plot([20, 40, 80], [r[metric + "_delta_percent"] for r in selected],
                    "o-", color=color, label=f"{qps} req/s", ms=4)
        ax.axhline(0, color=".7", lw=.6)
        ax.set(xlabel="L2 capacity (MiB)", title=title)
        ax.set_xticks([20, 40, 80])
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Change from 40 MiB (%)")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "l2_sensitivity_revision.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9, 4.5))
    for col, qps in enumerate((4, 16, 64)):
        for color, factor in zip(COLORS, (.9, 1., 1.1)):
            trajectory = by_key[("kernel_scale", qps, factor)]["trajectory"]
            axes[0, col].plot(
                [r["step_index"] for r in trajectory],
                [r["scheduled_tokens"] for r in trajectory],
                color=color, lw=.55, alpha=.8, label=f"{factor:g}× kernel time")
            axes[1, col].plot(
                [r["start_s"] for r in trajectory],
                [r["batch_requests"] for r in trajectory],
                color=color, lw=.8)
        axes[0, col].set(yscale="log", xlabel="Step index", title=f"{qps} req/s")
        axes[1, col].set(xlabel="Simulated schedule time (s)")
    axes[0, 0].set_ylabel("Scheduled tokens")
    axes[1, 0].set_ylabel("Requests per batch")
    axes[0, 2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "scheduler_feedback_revision.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.55))
    for ax, metric, title in zip(
            axes, ("ttft_ms", "tpot_ms", "request_throughput_per_s"),
            ("Mean TTFT", "Mean TPOT", "Request throughput")):
        for color, qps in zip(COLORS, (4, 16, 64)):
            ax.plot([.9, 1., 1.1],
                    [100 * (by_key[("kernel_scale", qps, f)]["summary"][metric]
                     / by_key[("kernel_scale", qps, 1.)]["summary"][metric] - 1)
                     for f in (.9, 1., 1.1)],
                    "o-", color=color, label=f"{qps} req/s")
        ax.axhline(0, color=".7", lw=.6)
        ax.set(xlabel="Kernel-time multiplier", title=title)
        ax.set_xticks([.9, 1., 1.1])
    axes[0].set_ylabel("Change from baseline (%)")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "kernel_time_sensitivity_revision.pdf")
    plt.close(fig)

    ladder = read(data / "ladder/summary.json")["cases"]
    order = (
        "nvidia_a800/qwen3-4b/tp1_qps4",
        "metax_c500/qwen3-4b/tp1_qps4",
        "ascend_910b/qwen3-4b/tp1_qps4",
        "ascend_910b/qwen3-8b/tp1_qps4",
        "ascend_910b/llama-3.1-8b/tp1_qps4")
    ladder.sort(key=lambda item: order.index(item["case_id"]))
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.1))
    labels = ["Kernel", "+Graph", "+Pad", "+CPU", "+Async"]
    for column, case in enumerate(ladder):
        hw, model, _ = case["case_id"].split("/")
        title = {"nvidia_a800": "A800", "metax_c500": "C500",
                 "ascend_910b": "910B"}[hw] + "\n" + (
                     model.replace("qwen3", "Qwen3").replace("llama-3.1", "Llama-3.1"))
        for row, metric in enumerate(("ttft", "tpot")):
            ax = axes[row, column]
            ax.plot(range(5), case[metric + "_rungs_ms"], "o-", color=COLORS[0], ms=3)
            ax.axhline(case["observed"][metric + "_ms"], color=COLORS[2],
                       ls="--", lw=1, label="Measured")
            ax.set_xticks(range(5), labels, rotation=55, fontsize=7)
            if row == 0:
                ax.set_title(title)
    axes[0, 0].set_ylabel("TTFT (ms)")
    axes[1, 0].set_ylabel("TPOT (ms)")
    axes[0, -1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "mechanism_ablation_revision.pdf")
    plt.close(fig)

    with (data / "ttft_analysis/per_qps_metrics_and_breakdown.csv").open() as stream:
        breakdown = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, model in zip(axes, ("qwen3-4b", "qwen3-8b", "llama-3.1-8b")):
        selected = sorted(
            (r for r in breakdown if r["hardware_id"] == "ascend_910b"
             and r["model_id"] == model), key=lambda r: int(r["target_qps"]))
        x = np.arange(3)
        wait = [float(r["wait_signed_error_ms"]) for r in selected]
        processing = [float(r["processing_signed_error_ms"]) for r in selected]
        ax.bar(x - .18, wait, .36, color=COLORS[0], label="Arrival to first schedule")
        ax.bar(x + .18, processing, .36, color=COLORS[2], label="First schedule to token")
        ax.plot(x, [w + p for w, p in zip(wait, processing)],
                "kd--", ms=4, lw=.8, label="Total TTFT error")
        ax.axhline(0, color=".5", lw=.5)
        ax.set_xticks(x, ["4", "16", "64"])
        ax.set(title=model, xlabel="Arrival rate (req/s)")
    axes[0].set_ylabel("Predicted − measured (ms)")
    axes[1].legend(frameon=False, fontsize=6.7)
    fig.tight_layout()
    fig.savefig(figure_dir / "ttft_decomposition_revision.pdf")
    plt.close(fig)

    graph = read(data / "graph_padding/summary.json")
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    hardware = ("nvidia_a800", "metax_c500", "ascend_910b")
    x = np.arange(3)
    for offset, key, label, color in (
            (-.25, "time", "Device iteration time", COLORS[0]),
            (0, "l2_cache", "L2 traffic (covered)", "#73a890"),
            (.25, "dram", "HBM traffic (covered)", COLORS[2])):
        values = [
            graph[h]["all_steps"]["timeline_delta_pct"] if key == "time"
            else graph[h]["memory_diagnostic_scope"][key]["bytes_delta_pct"]
            for h in hardware]
        ax.bar(x + offset, values, .24, label=label, color=color)
    ax.axhline(0, color=".4", lw=.6)
    ax.set_xticks(x, ["A800", "C500", "910B"])
    ax.set_ylabel("Change from exact capture (%)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "graph_padding_revision.pdf")
    plt.close(fig)
    return rows
