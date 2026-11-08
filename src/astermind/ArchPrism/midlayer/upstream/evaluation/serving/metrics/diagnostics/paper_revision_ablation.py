"""Plot Fig. 3 with explicit simulated TTFT components."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Annotation
from matplotlib.textpath import TextPath
from matplotlib.transforms import ScaledTranslation


def require_plot(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--show-scheduling", action="store_true",
                        help="Split out the first scheduling call from remaining TTFT.")
    split.add_argument("--waiting-before-execution", action="store_true",
                       help="Show simulated arrival-to-execution wait; keep measured total unsplit.")
    split.add_argument("--processing-other", action="store_true",
                       help="Split simulated TTFT into Processing and Other latency.")
    parser.add_argument("--show-output-return", action="store_true",
                        help="Split execution and output return after execution.")
    parser.add_argument("--other-style", choices=("original", "green", "outline", "hatch", "dense-hatch"),
                        default="original", help="Style of Other latency in the two-part preview.")
    parser.add_argument("--other-color",
                        help="Override the fill color of Other latency.")
    parser.add_argument("--other-text-color",
                        help="Override the label color of Other latency.")
    parser.add_argument("--label-decimals", type=int, choices=(0, 1, 2, 3),
                        help="Use a uniform decimal precision for bar labels.")
    parser.add_argument("--other-first", action="store_true",
                        help="Place Other latency to the left of Processing.")
    parser.add_argument("--waiting-color", default="#e7b873",
                        help="Fill color for the waiting segment.")
    parser.add_argument("--inside-totals", action="store_true",
                        help="Use the original total-label layout for the two-part preview.")
    parser.add_argument("--right-align-other", action="store_true",
                        help="Right-align Other latency labels inside their segment.")
    parser.add_argument("--legend-bottom-left", action="store_true",
                        help="Place a two-row legend below the configuration labels.")
    parser.add_argument("--center-kernel-only-other", action="store_true",
                        help="Center the narrow Other latency label for Kernel Only.")
    parser.add_argument("--device-only", action="store_true",
                        help="Use device-only execution paths for the dark TTFT segment.")
    parser.add_argument("--labels-top", action="store_true",
                        help="Place the legend and metric titles above the bars.")
    parser.add_argument("--split-tpot", action="store_true",
                        help="Also split TPOT into device-only execution and remaining latency.")
    parser.add_argument("--compact-bottom-labels", action="store_true",
                        help="Keep metric labels below the axes and bring the top legend closer.")
    parser.add_argument("--optical-center-labels", action="store_true",
                        help="Center numeric glyphs vertically within their bars.")
    parser.add_argument("--width-inches", type=float, default=3.45,
                        help="Figure width before tight cropping.")
    parser.add_argument("--small-tpot-inside", action="store_true",
                        help="Center narrow TPOT residual labels inside, omitting their leading zero.")
    parser.add_argument("--wide-legend", action="store_true",
                        help="Increase spacing between legend entries without lengthening swatches.")
    parser.add_argument("--optical-label-offset-pt", type=float, default=.22,
                        help="Additional upward optical adjustment for bar numbers in points.")
    args = parser.parse_args()
    def label(value, default_decimals=2):
        decimals = default_decimals if args.label_decimals is None else args.label_decimals
        return f"{value:.{decimals}f}"
    if args.show_output_return and not args.waiting_before_execution:
        parser.error("--show-output-return requires --waiting-before-execution")
    if args.other_style != "original" and not args.processing_other:
        parser.error("--other-style requires --processing-other")
    if args.other_color is not None and not args.processing_other:
        parser.error("--other-color requires --processing-other")
    if args.other_text_color is not None and not args.processing_other:
        parser.error("--other-text-color requires --processing-other")
    if args.other_first and not args.processing_other:
        parser.error("--other-first requires --processing-other")
    if args.inside_totals and not args.processing_other:
        parser.error("--inside-totals requires --processing-other")
    if args.right_align_other and not args.inside_totals:
        parser.error("--right-align-other requires --inside-totals")
    if args.device_only and not args.processing_other:
        parser.error("--device-only requires --processing-other")
    if args.labels_top and args.legend_bottom_left:
        parser.error("--labels-top conflicts with --legend-bottom-left")
    if args.split_tpot and not args.device_only:
        parser.error("--split-tpot requires --device-only")
    if args.compact_bottom_labels and (args.labels_top or args.legend_bottom_left):
        parser.error("--compact-bottom-labels requires a top legend and bottom metric labels")
    payload = json.loads(args.input.read_text())
    rows = payload["rows"]
    require_plot(payload["original_totals_verified"] is True, "original totals must be verified")
    require_plot(len(rows) == 6, "ablation figure requires six rows")
    require_plot(args.output.name.endswith("_revision.pdf"), "output must end with _revision.pdf")
    labels = [
        "Measured", "Kernel Only", "+Graph Replay", "+Graph Padding",
        "+Scheduler CPU", "+Async Depth",
    ]
    if args.waiting_before_execution or args.processing_other:
        labels[0] = "Measured (total)"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if args.other_style in ("hatch", "dense-hatch"):
        plt.rcParams["hatch.linewidth"] = .35
    three_parts = args.show_scheduling or args.show_output_return
    figure, axes = plt.subplots(
        1, 2, figsize=(args.width_inches, 1.18 if args.labels_top else .98 if args.legend_bottom_left else 1.25 if three_parts else 1.05))
    positions = [0.62 * i for i in range(6)][::-1]
    measured_color, model_color, waiting_color = "#1f4e79", "#1b9e84", args.waiting_color
    other_text_color = "#392b15"
    if args.other_style == "green":
        waiting_color = "#bce6da"
        other_text_color = "#184d40"
    elif args.other_style in ("outline", "hatch"):
        waiting_color = "#ffffff"
        other_text_color = "#184d40"
    elif args.other_style == "dense-hatch":
        waiting_color = model_color
        other_text_color = "white"
    hatch_color = "#c6e7dc" if args.other_style == "dense-hatch" else model_color
    hatch_pattern = "////////" if args.other_style == "dense-hatch" else "////"
    if args.other_color is not None:
        waiting_color = args.other_color
    if args.other_text_color is not None:
        other_text_color = args.other_text_color
    scheduling_color = "#b04491"
    for column, metric in enumerate(("ttft_ms", "tpot_ms")):
        axis = axes[column]
        values = [r[metric] for r in rows]
        ceiling = max(values)
        for i, (y, row) in enumerate(zip(positions, rows, strict=True)):
            color = measured_color if i == 0 else model_color
            value = row[metric]
            if (column == 0 or args.split_tpot) and (args.show_output_return or args.processing_other) and i > 0:
                if args.processing_other:
                    device_ms = (row["device_only_tpot_ms"] if column == 1 else
                                 row["device_only_ms"] if args.device_only else row["execution_ms"])
                    other_latency = value - device_ms
                    if abs(other_latency) < 1e-10:
                        other_latency = 0.0
                    if args.device_only:
                        upper_bound = row["execution_ms"] if column == 0 else value
                        require_plot(0 <= device_ms <= upper_bound + 1e-8, "device duration exceeds its execution interval")
                    else:
                        require_plot(abs(other_latency - row["wait_ms"] - row["output_return_ms"]) < 1e-8, "other latency components do not conserve the total")
                    segments = (
                        (device_ms, model_color, "white"),
                        (other_latency, waiting_color, other_text_color),
                    )
                    if args.other_first:
                        segments = segments[::-1]
                else:
                    segments = (
                        (row["wait_ms"], waiting_color, "#392b15"),
                        (row["execution_ms"], model_color, "white"),
                        (row["output_return_ms"], scheduling_color, "white"),
                    )
                require_plot(abs(sum(segment[0] for segment in segments) - value) < 1e-8, "latency components do not conserve the total")
                left = 0.0
                for segment_index, (duration, segment_color, text_color) in enumerate(segments):
                    require_plot(duration >= 0, "latency component must be non-negative")
                    if duration:
                        other_index = 0 if args.other_first else 1
                        outlined = args.processing_other and args.other_style == "outline" and segment_index == other_index
                        hatched = args.processing_other and args.other_style in ("hatch", "dense-hatch") and segment_index == other_index
                        axis.barh(y, duration, left=left, height=.44,
                                  color=segment_color,
                                  edgecolor=hatch_color if hatched else model_color if outlined else "none",
                                  linewidth=.5 if outlined else .35 if hatched else 0,
                                  linestyle=(0, (2, 1.5)) if outlined else "-",
                                  hatch=hatch_pattern if hatched else None,
                                  zorder=3)
                        if args.inside_totals:
                            if segment_index == other_index:
                                right_align = (
                                    args.right_align_other
                                    and not (args.center_kernel_only_other and i == 1))
                                label_x = (
                                    left + duration - .012 * ceiling
                                    if right_align else left + duration / 2)
                                number = label(duration, 1)
                                if column == 1 and args.other_first and duration < .1 * ceiling:
                                    if args.small_tpot_inside:
                                        label_x = left + duration / 2
                                        right_align = False
                                        if number.startswith("0."):
                                            number = number[1:]
                                    else:
                                        label_x = -.012 * ceiling
                                        right_align = True
                                axis.text(label_x, y, number,
                                          fontsize=4.6 if args.legend_bottom_left or args.labels_top or args.compact_bottom_labels else 4.1,
                                          color=text_color,
                                          ha="right" if right_align else "center",
                                          bbox=(
                                              {"facecolor": model_color, "edgecolor": "none", "pad": .15}
                                              if args.other_style == "dense-hatch" else None),
                                          va="center", zorder=6)
                        elif duration >= 4:
                            axis.text(left + duration / 2, y, label(duration),
                                      fontsize=3.4 if args.processing_other and duration < 8 else 4.0,
                                      color=text_color,
                                      ha="center", va="center", zorder=6)
                        else:
                            axis.annotate(
                                label(duration),
                                xy=(left + duration / 2, y + .22),
                                xytext=(left + duration / 2, y + .32),
                                fontsize=3.8, color=segment_color,
                                ha="center", va="center",
                                arrowprops={"arrowstyle": "-", "color": segment_color,
                                            "lw": .45, "shrinkA": 0, "shrinkB": 0},
                                zorder=7,
                            )
                    left += duration
            elif column == 0 and not (
                    (args.waiting_before_execution or args.processing_other) and i == 0):
                scheduling = row["scheduling_ms"] if args.show_scheduling else 0.0
                remaining = (
                    row["remaining_ttft_ms"] if args.show_scheduling
                    else row["processing_ms"])
                require_plot(min(row["wait_ms"], scheduling, remaining) >= 0, "latency components must be non-negative")
                require_plot(abs(row["wait_ms"] + scheduling + remaining - value) < 1e-8, "latency components do not conserve the total")
                axis.barh(y, row["wait_ms"], height=.44, color=waiting_color,
                          edgecolor="none", zorder=3)
                axis.barh(y, remaining, left=row["wait_ms"] + scheduling,
                          height=.44, color=color, edgecolor="none", zorder=3)
                if args.show_scheduling:
                    axis.barh(y, scheduling, left=row["wait_ms"],
                              height=.44, color=scheduling_color,
                              edgecolor="none", zorder=4)
                    if scheduling:
                        axis.annotate(
                            label(scheduling),
                            xy=(row["wait_ms"] + scheduling / 2, y + .22),
                            xytext=(row["wait_ms"] + scheduling + 3, y + .30),
                            fontsize=3.8, color=scheduling_color,
                            ha="left", va="center",
                            arrowprops={"arrowstyle": "-", "color": scheduling_color,
                                        "lw": .45, "shrinkA": 0, "shrinkB": 0},
                            zorder=7,
                        )
                axis.text(row["wait_ms"] / 2, y, label(row["wait_ms"], 1),
                          fontsize=4.1, color="#392b15", ha="center",
                          va="center", zorder=6)
            else:
                axis.barh(y, value, height=.44, color=color,
                          edgecolor="none", zorder=3)
            if column == 0 and (args.show_output_return or args.processing_other) and not args.inside_totals:
                axis.text(value + .018 * ceiling, y, label(value),
                          fontsize=4.6, color="#222222", fontweight="bold",
                          ha="left", va="center", zorder=6)
            else:
                anchor = min(values[0], value)
                axis.text(anchor - .012 * ceiling, y, label(value),
                          fontsize=4.6, color="white", fontweight="bold",
                          ha="right", va="center", zorder=6)
        axis.axvline(values[0], color=measured_color, linestyle=":",
                     linewidth=.8, zorder=5)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=5.2)
        axis.tick_params(axis="y", labelleft=column == 0, length=0, pad=1.6)
        axis.set_ylim(positions[-1] - .31, positions[0] + .31)
        axis.set_xlim(0, ceiling * (
            1.23 if column == 0 and (args.show_output_return or args.processing_other) and not args.inside_totals else 1.04))
        metric_label = "Mean TTFT (ms)" if column == 0 else "Mean TPOT (ms)"
        if args.labels_top:
            axis.set_title(metric_label, fontsize=5.2, pad=3)
        else:
            axis.set_xlabel(metric_label,
                            fontsize=5.2 if args.legend_bottom_left or args.compact_bottom_labels else 5.4, labelpad=1)
        axis.tick_params(axis="x", labelsize=4.8, pad=1)
        axis.grid(axis="x", color="#cccccc", linewidth=.4, alpha=.6)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
    waiting_label = (
        "Waiting before execution (simulation)" if args.waiting_before_execution
        else "Waiting before first scheduling")
    handles = [Patch(facecolor=waiting_color, edgecolor="none",
                     label=waiting_label)]
    if args.show_scheduling:
        handles.extend([
            Patch(facecolor=scheduling_color, edgecolor="none",
                  label="First scheduling call"),
            Patch(facecolor=model_color, edgecolor="none",
                  label="Remaining TTFT"),
        ])
    elif args.show_output_return:
        handles.extend([
            Patch(facecolor=model_color, edgecolor="none",
                  label="Execution"),
            Patch(facecolor=scheduling_color, edgecolor="none",
                  label="Output return after execution"),
        ])
    elif args.processing_other:
        handles = [
            Patch(facecolor=model_color, edgecolor="none",
                  label="Device Execution" if args.device_only else "Processing"),
            Patch(facecolor=waiting_color,
                  edgecolor=hatch_color if args.other_style in ("hatch", "dense-hatch") else model_color if args.other_style == "outline" else "none",
                  linewidth=.5,
                  linestyle=(0, (2, 1.5)) if args.other_style == "outline" else "-",
                  hatch=hatch_pattern if args.other_style in ("hatch", "dense-hatch") else None,
                  label="Other Latency"),
        ]
        if args.other_first:
            handles.reverse()
    figure.legend(
        handles=handles,
        loc="upper right" if args.legend_bottom_left else "upper center",
        bbox_to_anchor=(.60, .99) if args.compact_bottom_labels else (.60, 1.0) if args.labels_top else (.235, .29) if args.legend_bottom_left else (.60, 1.01),
        frameon=False, fontsize=5.2 if args.legend_bottom_left or args.labels_top or args.compact_bottom_labels else 5.1,
        handlelength=1.2,
        columnspacing=4.0 if args.wide_legend else 2.0,
        ncol=1 if args.legend_bottom_left else 2 if args.processing_other else 1,
        borderaxespad=0, handletextpad=.4,
    )
    figure.subplots_adjust(
        left=.25, right=.98,
        bottom=.25 if args.compact_bottom_labels else .14 if args.labels_top else .30 if args.legend_bottom_left else .24 if three_parts else .28,
        top=.88 if args.compact_bottom_labels else .77 if args.labels_top else .98 if args.legend_bottom_left else .72 if three_parts else .84,
        wspace=.08)
    if args.optical_center_labels:
        for axis in axes:
            for artist in axis.texts:
                if isinstance(artist, Annotation):
                    continue
                bounds = TextPath((0, 0), artist.get_text(),
                                  prop=artist.get_fontproperties()).get_extents()
                midpoint_pt = (bounds.ymin + bounds.ymax) / 2
                artist.set_verticalalignment("baseline")
                artist.set_transform(artist.get_transform() + ScaledTranslation(
                    0, (-midpoint_pt + args.optical_label_offset_pt) / 72,
                    figure.dpi_scale_trans))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=.02)
    figure.savefig(args.output.with_suffix(".png"), bbox_inches="tight",
                   pad_inches=.02, dpi=350)
    plt.close(figure)


if __name__ == "__main__":
    main()
