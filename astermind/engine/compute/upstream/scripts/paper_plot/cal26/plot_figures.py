#!/usr/bin/env python3
"""Draw the CAL'26 paper data figures from one collect_data.py run.

  memory_traffic.pdf                     Fig. 2  measured vs predicted L2/HBM traffic
  serving_mechanism_ablation.pdf         Fig. 3  cumulative mechanism ladder
  serving_request_comparison_<model>.pdf Fig. 4  request TTFT/TPOT CDFs (paper: qwen3-4b)
  production_tradeoff.pdf                Fig. 5  DeepSeek-R1 H200 vs InferenceX

Every figure is also written as .png.  Fig. 1 (overview) is a drawn diagram
with no data behind it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARENT = ROOT / "experiment_logs/paper_plot/cal26"

HARDWARE = (
    ("nvidia_a800", "A800"),
    ("metax_c500", "C500"),
    ("ascend_910b", "910B"),
)
# QPS 4 / 16 / 64 hues of the CDF figure, plus a purple for Vector.
KERNEL_SERIES = (
    ("GEMM", "#355f7d", "o"),
    ("Attention", "#6b9f8d", "o"),
    ("MoE", "#cc6677", "o"),
    ("Vector", "#a85b9e", "o"),
)
SCATTER_ALPHA = {"GEMM": 0.3, "Attention": 0.35, "MoE": 0.35, "Vector": 0.4}
QPS_COLORS = ("#355f7d", "#6b9f8d", "#cc6677")
SERVING_MODELS = (
    ("qwen3-4b", "tp1_qps{}", (4, 16, 64)),
    ("qwen3-8b", "tp1_qps{}", (4, 16, 64)),
    ("llama-3.1-8b", "tp1_qps{}", (4, 16, 64)),
    ("qwen3-32b", "tp2_ep1_qps{}", (1, 4, 16)),
    ("qwen3-30b-a3b", "tp2_ep2_qps{}", (1, 4, 16)),
)
ABLATION_LABELS = {
    "Kernel only": "Kernel Only",
    "+Graph": "+Graph Replay",
    "+Graph padding": "+Graph Padding",
    "+Sched CPU": "+Scheduler CPU",
    "+Async depth": "+Async Depth",
}
ABLATION_MEASURED = "#1f4e79"
ABLATION_MODEL = "#1b9e84"

FIGURE_STYLE = {
    "font.size": 7.2,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.2,
    "legend.fontsize": 6.8,
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
ABLATION_STYLE = {
    "font.size": 7.0,
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextlib.contextmanager
def style(params: dict[str, object]):
    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams.update(params)
        yield


def save(figure, directory: Path, name: str, *, pad: float = 0.025) -> None:
    pdf_path = directory / name
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=pad, dpi=220)
    figure.savefig(pdf_path.with_suffix(".png"), bbox_inches="tight",
                   pad_inches=pad, dpi=300)
    plt.close(figure)


def _read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def memory_traffic(manifest: dict, out: Path) -> None:
    """Measured vs Forward-predicted L2 and HBM traffic per kernel case."""

    document = _read_json(manifest["memory_traffic"])
    by_hardware = {hw["hardware_id"]: hw for hw in document["hardware"]}
    figure, axes = plt.subplots(2, 3, figsize=(3.45, 2.42), squeeze=False)
    metrics = (("l2", "LLC"), ("hbm", "HBM"))

    limits: dict[str, tuple[float, float]] = {}
    for metric, _label in metrics:
        values = []
        for hardware_id, _title in HARDWARE:
            for case in by_hardware[hardware_id]["metrics"][metric]["cases"]:
                values.extend((case["measured_bytes"] / 1e6,
                               case["predicted_bytes"] / 1e6))
        limits[metric] = (10 ** math.floor(math.log10(min(values))),
                          10 ** math.ceil(math.log10(max(values))))

    for row, (metric, metric_label) in enumerate(metrics):
        for column, (hardware_id, hardware_label) in enumerate(HARDWARE):
            lower, upper = limits[metric]
            axis = axes[row, column]
            cases = by_hardware[hardware_id]["metrics"][metric]["cases"]
            # Drawn bottom to top so GEMM covers Attention, MoE and Vector.
            for family, color, marker in reversed(KERNEL_SERIES):
                # The traffic data names this family "FlashAttention".
                data_family = ("FlashAttention" if family == "Attention"
                               else family)
                selected = [c for c in cases if c["family"] == data_family]
                axis.scatter(
                    [c["measured_bytes"] / 1e6 for c in selected],
                    [c["predicted_bytes"] / 1e6 for c in selected],
                    s=2, alpha=SCATTER_ALPHA[family],
                    color=color, marker=marker,
                    edgecolors="none", label=family,
                )
            axis.plot((lower, upper), (lower, upper), color="#666666",
                      linewidth=0.6, linestyle="--", zorder=0)
            axis.set_xscale("log")
            axis.set_yscale("log")
            ticks = [10.0 ** exponent for exponent in range(
                int(math.log10(lower)), int(math.log10(upper)) + 1, 3
            )]
            if ticks[-1] != upper:
                ticks.append(upper)
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            if row == 0:
                # The upper L2 bound is only a frame limit; leaving its label
                # off avoids a dangling label at the top/right edge.
                tick_labels = [rf"$10^{{{int(math.log10(tick))}}}$"
                               for tick in ticks]
                tick_labels[-1] = ""
                axis.set_xticklabels(tick_labels)
                axis.set_yticklabels(tick_labels)
            axis.set_aspect("equal", adjustable="box")
            observed = np.asarray([c["measured_bytes"] for c in cases])
            predicted = np.asarray([c["predicted_bytes"] for c in cases])
            errors = np.abs(predicted - observed) / observed * 100.0
            axis.text(
                0.08, 0.88,
                f"MAPE={errors.mean():.1f}%",
                transform=axis.transAxes, ha="left", va="top", fontsize=5.4,
            )
            axis.set_title(f"{hardware_label} {metric_label} Traffic",
                           fontweight="bold", fontsize=5.6, pad=1.5)
            # Both rows share the axes; the upper-row x label is omitted to
            # leave room for the panel titles.
            if row == 1:
                axis.set_xlabel("Measured (MB)", fontsize=5.1,
                                labelpad=1.0)
            if column == 0:
                axis.set_ylabel("Predicted (MB)", fontsize=5.1,
                                labelpad=1.0)
                axis.yaxis.set_label_coords(-0.23, 0.5)
            axis.grid(which="major", color="#b0b0b0", linewidth=0.55, alpha=0.45)
            axis.grid(which="minor", color="#c8c8c8", linewidth=0.35, alpha=0.35)
            axis.tick_params(labelsize=4.5, pad=0.8)

    # Points stay translucent to show density; legend keys are opaque and a
    # little larger, in the operator row order of the kernel-accuracy table.
    series = {family: (color, marker)
              for family, color, marker in KERNEL_SERIES}
    legend_handles = [
        Line2D([0], [0], linestyle="none", marker=series[family][1],
               color=series[family][0], markersize=2.6, markeredgewidth=0.0,
               alpha=1.0, label=family)
        for family in ("GEMM", "Attention", "MoE", "Vector")
    ]
    figure.subplots_adjust(left=0.17, right=0.995, bottom=0.12, top=0.90,
                           wspace=0.31, hspace=0.20)
    # Spread the four keys over 75% of the width of a row of three panels.
    for axis in axes.flat:
        axis.apply_aspect()
    row_left = axes[0, 0].get_position().x0
    row_right = axes[0, -1].get_position().x1
    legend_width = 0.75 * (row_right - row_left)
    legend_left = (row_left + row_right - legend_width) / 2.0
    figure.legend(handles=legend_handles, ncol=4, frameon=False,
                  loc="upper left", mode="expand", borderaxespad=0.0,
                  bbox_to_anchor=(legend_left, 1.0, legend_width, 0.0),
                  handletextpad=0.18, fontsize=5)
    save(figure, out, "memory_traffic.pdf")


def _ablation_panel(axis, payload, metric: str, title: str,
                    show_labels: bool) -> None:
    rungs = payload["rungs"]
    measured = payload["observed"][metric]
    bars = [measured] + [rung[metric] for rung in rungs]
    ceiling = max(bars)
    pitch, height = 0.62, 0.44
    rows = [row * pitch for row in range(len(bars))][::-1]
    for index, (row, value) in enumerate(zip(rows, bars)):
        axis.barh(row, value, height=height, zorder=3, edgecolor="none",
                  color=ABLATION_MEASURED if index == 0 else ABLATION_MODEL)
        # Labels of bars that overshoot the measured value end just left of
        # the dotted measured line instead of underneath it.
        anchor = measured if value > measured else value
        axis.text(anchor - 0.012 * ceiling, row, f"{value:.2f}", va="center",
                  ha="right", fontsize=4.6, color="white", fontweight="bold",
                  zorder=6)
    axis.axvline(measured, color=ABLATION_MEASURED, linestyle=":",
                 linewidth=0.8, zorder=5)
    axis.set_yticks(rows)
    axis.set_yticklabels(
        ["Measured"] + [ABLATION_LABELS[r["label"].replace("\n", " ")]
                        for r in rungs],
        fontsize=5.2)
    axis.tick_params(axis="y", labelleft=show_labels, length=0, pad=1.6)
    axis.set_ylim(rows[-1] - pitch / 2.0, rows[0] + pitch / 2.0)
    axis.set_xlim(0, ceiling * 1.04)
    axis.set_xlabel(title, fontsize=5.4, labelpad=1.0)
    axis.tick_params(axis="x", labelsize=4.8, pad=1.0)
    axis.grid(axis="x", color="#cccccc", linewidth=0.4, alpha=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)


def serving_mechanism_ablation(manifest: dict, out: Path) -> None:
    """Mean TTFT/TPOT of each cumulative mechanism rung vs the measurement."""

    payload = _read_json(manifest["mechanism_ablation"])
    figure, axes = plt.subplots(1, 2, figsize=(3.45, 1.0))
    _ablation_panel(axes[0], payload, "ttft_ms", "Mean TTFT (ms)", True)
    _ablation_panel(axes[1], payload, "tpot_ms", "Mean TPOT (ms)", False)
    figure.subplots_adjust(left=0.25, right=0.98, bottom=0.30, top=0.98,
                           wspace=0.08)
    save(figure, out, "serving_mechanism_ablation.pdf", pad=0.02)


def _plot_cdf(axis, values: np.ndarray, *, color: str, linestyle: str) -> None:
    ordered = np.sort(values)
    cumulative = np.arange(1, len(ordered) + 1, dtype=float) / len(ordered)
    axis.step(np.r_[ordered[0], ordered], np.r_[0.0, cumulative],
              where="post", color=color, linestyle=linestyle, linewidth=0.8)


def serving_request_comparison(manifest: dict, out: Path) -> None:
    """Measured vs predicted request TTFT/TPOT CDFs, one figure per model."""

    cells = {cell["cell_id"]: cell
             for cell in _read_json(manifest["serving_results"])["cells"]}
    for model_id, setting_template, qps_values in SERVING_MODELS:
        figure, axes = plt.subplots(2, 3, figsize=(3.45, 2.42), sharey=True)
        for column, (hardware_id, hardware_label) in enumerate(HARDWARE):
            for row, (metric, metric_label) in enumerate((
                    ("ttft_ms", "TTFT"), ("tpot_ms", "TPOT"))):
                axis = axes[row, column]
                panel_max = 0.0
                for qps, color in zip(qps_values, QPS_COLORS, strict=True):
                    cell = cells[f"{hardware_id}/{model_id}/"
                                 f"{setting_template.format(qps)}"]
                    records = cell["metrics"]["paired_request_latency"][
                        "metrics"][metric]
                    measured = np.asarray([r["measured_ms"] for r in records],
                                          dtype=float)
                    predicted = np.asarray([r["predicted_ms"] for r in records],
                                           dtype=float)
                    panel_max = max(panel_max, float(measured.max()),
                                    float(predicted.max()))
                    _plot_cdf(axis, measured, color=color, linestyle="-")
                    _plot_cdf(axis, predicted, color=color, linestyle="--")
                axis.set_xlim(0.0, max(1.0, panel_max * 1.03))
                axis.set_ylim(0.0, 1.03)
                axis.set_title(f"{hardware_label} {metric_label}",
                               fontsize=5.6, fontweight="bold", pad=1.5)
                if column == 0:
                    axis.set_ylabel("CDF", fontsize=5.1, labelpad=1.0)
                    axis.yaxis.set_label_coords(-0.23, 0.5)
                if row == 1:
                    axis.set_xlabel("Latency (ms)", fontsize=5.1,
                                    labelpad=1.0)
                axis.grid(color="#b0b0b0", linewidth=0.55, alpha=0.45)
                axis.tick_params(labelsize=4.5, pad=0.8, labelleft=True)
                axis.set_box_aspect(1)

        # Linestyle is the only encoding shared by every panel; colour maps to
        # the QPS of this model's settings.
        handles = [
            Line2D([0], [0], color="#333333", linewidth=0.8, linestyle="-",
                   label="Measured"),
            Line2D([0], [0], color="#333333", linewidth=0.8, linestyle="--",
                   label="Predicted"),
        ] + [
            Line2D([0], [0], color=color, linewidth=0.8, label=f"QPS {qps}")
            for qps, color in zip(qps_values, QPS_COLORS, strict=True)
        ]
        figure.legend(handles=handles, ncol=5, frameon=False,
                      loc="upper left", bbox_to_anchor=(0.21125, 0.95, 0.7425, 0.05),
                      mode="expand", borderaxespad=0.0, fontsize=4.5,
                      handlelength=2.8, handletextpad=0.38)
        figure.subplots_adjust(left=0.17, right=0.995, bottom=0.12, top=0.93,
                               wspace=0.31, hspace=0.11)
        save(figure, out, f"serving_request_comparison_{model_id}.pdf")


def _inferencex_dsr1_colocated(raw: list[dict],
                               spec_method: str) -> list[tuple[float, float]]:
    """Same-config InferenceX SGLang H200 DSR1 8K/1K TP8/EP1 points."""

    rows = sorted((
        row for row in raw
        if row["hardware"] == "h200" and row["framework"] == "sglang"
        and row["precision"] == "fp8" and not row["disagg"]
        and row["isl"] == 8192 and row["osl"] == 1024
        and row["spec_method"] == spec_method
        and row["decode_tp"] == 8 and row["decode_ep"] == 1
        and not row["decode_dp_attention"]
    ), key=lambda row: row["conc"])
    concurrencies = [row["conc"] for row in rows]
    if not rows or len(set(concurrencies)) != len(concurrencies):
        raise ValueError(f"InferenceX rows for {spec_method}: {concurrencies}")
    return [(row["metrics"]["p90_intvty"], row["metrics"]["tput_per_gpu"])
            for row in rows]


def production_tradeoff(manifest: dict, out: Path) -> None:
    """DeepSeek-R1 8K/1K on H200: simulated deployments vs InferenceX.

    Left MTP off, right MTP on, shared throughput axis.  Colocated uses 8
    GPUs and PD 8 prefill + 8 decode GPUs, TP8/EP1, closed loop; throughput
    is divided by all GPUs of the deployment (the InferenceX convention).
    """

    directory = Path(manifest["dsr1_sweep"])
    # Version 1 reports predate the immutable suite and use unprefixed groups.
    # Version 2 explicitly binds the published vLLM settings and five inputs.
    if manifest["schema_version"] == 1:
        concurrencies = (1, 2, 4, 8, 16, 32, 64)
        setting_prefix = ""
        model_label = "DeepSeek-R1"
    elif manifest["schema_version"] == 2:
        concurrencies = (4, 8, 16, 32, 64)
        setting_prefix = "vllm_"
        model_label = "DeepSeek-V3 / R1 recipe"
    else:
        raise ValueError("unsupported CAL'26 manifest schema")
    raw = _read_json(manifest["inferencex_raw"]["path"])
    panels = (("off", "none", f"{model_label} (MTP Off)"),
              ("on", "mtp", f"{model_label} (MTP On)"))
    deployments = (
        ("colocated", "Colocated (Sim)", "#355f7d", "o", 1.55, 8.0),
        ("pd", "PD Disaggregated (Sim)", "#cc6677", "D", 1.45, 16.0),
    )
    figure, axes = plt.subplots(1, 2, figsize=(3.45, 1.45), sharey=True)
    y_max = 0.0
    for axis, (mtp, spec_method, title) in zip(axes, panels, strict=True):
        for deployment, label, color, marker, size, gpus in deployments:
            points = []
            for concurrency in concurrencies:
                result = _read_json(directory / f"{setting_prefix}{deployment}_mtp_{mtp}"
                                    / f"c{concurrency}.json")
                quantiles = result["metrics"]["latency_distributions"][
                    "tpot_s"]["quantiles"]
                p90_tpot = quantiles["values_s"][quantiles["percentiles"].index(90)]
                points.append((1.0 / p90_tpot,
                               result["metrics"]["token_throughput_per_s"] / gpus))
            y_max = max([y_max] + [point[1] for point in points])
            axis.plot([point[0] for point in points],
                      [point[1] for point in points],
                      color=color, linestyle="-", marker=marker,
                      markersize=size, linewidth=0.72, label=label)
        measured = _inferencex_dsr1_colocated(raw, spec_method)
        y_max = max([y_max] + [point[1] for point in measured])
        axis.plot([point[0] for point in measured],
                  [point[1] for point in measured],
                  color="#355f7d", linestyle=(0, (3.0, 1.6)), marker="s",
                  markersize=1.55, linewidth=0.72,
                  label="Colocated (Measured by InferenceX)")
        axis.set_title(title, fontsize=6.5, fontweight="bold", pad=2.0)
        axis.set_xlabel("P90 Interactivity (tok/s/user)", fontsize=5.5,
                        labelpad=2.0)
        axis.set_xlim(left=0.0)
        axis.grid(color="#b8b8b8", alpha=0.28, linewidth=0.45)
        axis.tick_params(labelsize=5.0, pad=1.0, width=0.7, length=2.7)
        for spine in axis.spines.values():
            spine.set_linewidth(0.75)
    axes[0].set_ylim(0.0, 1.06 * y_max)
    # Both panels label the shared throughput axis.
    axes[1].tick_params(labelleft=True)
    axes[0].set_ylabel("Token Throughput\nper GPU (tok/s/GPU)", fontsize=5.5,
                       labelpad=2.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=5.0, ncol=3,
                  loc="upper center", bbox_to_anchor=(0.54, 0.975),
                  handlelength=1.8, columnspacing=0.4, handletextpad=0.3)
    figure.subplots_adjust(left=0.14, right=0.99, bottom=0.25, top=0.775,
                           wspace=0.24)
    save(figure, out, "production_tradeoff.pdf")


def _latest_run() -> Path:
    runs = sorted(path.parent for path in DEFAULT_PARENT.glob("*/manifest.json"))
    if not runs:
        raise FileNotFoundError(f"no collect_data.py run under {DEFAULT_PARENT}")
    return runs[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="collect_data.py run directory (default: latest)")
    parser.add_argument("--figure-dir", type=Path, default=None,
                        help="output directory (default: <run-dir>/figs)")
    parser.add_argument("--figure", choices=("all", "production_tradeoff"),
                        default="all", help="draw all figures or only the serving tradeoff")
    args = parser.parse_args()
    run_dir = (args.run_dir or _latest_run()).resolve()
    manifest = _read_json(run_dir / "manifest.json")
    out = (args.figure_dir or run_dir / "figs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    with style(FIGURE_STYLE):
        if args.figure == "all":
            memory_traffic(manifest, out)
            serving_request_comparison(manifest, out)
        production_tradeoff(manifest, out)
    if args.figure == "all":
        with style(ABLATION_STYLE):
            serving_mechanism_ablation(manifest, out)
    print(f"figures from {run_dir} (source commit {manifest['source_commit']}) "
          f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
