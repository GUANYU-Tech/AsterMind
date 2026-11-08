"""Render event and Cell-state tracks from a completed simulation report."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


PHASE_COLORS = {"prefill": "#3873bb", "decode": "#478a62", "mixed": "#7958af"}


def render(report, output):
    cells = list(report["cells"])
    requests = list(report["requests"])
    cell_label = {key: key.removeprefix("cell-") for key in cells}
    steps = {(cell_id, step["step"]["schedule"]["step_id"]): step
             for cell_id, cell in report["cells"].items() for step in cell["steps"]}
    fig, (request_ax, step_ax, state_ax) = plt.subplots(
        3, 1, sharex=True, figsize=(12, 7.7), layout="constrained",
        gridspec_kw={"height_ratios": (1.35, 1.0, 1.15)})
    fig.suptitle("OrbitServe / AsterMind   |   Shared simulation timeline",
                 fontsize=16, fontweight="bold", x=.02, ha="left")

    for row, request_id in enumerate(requests):
        state = report["requests"][request_id]
        arrive, ready, done = (state[key] / 1e6 for key in
                               ("arrival_ns", "cell_ready_ns", "compute_complete_ns"))
        request_ax.barh(row, ready - arrive, left=arrive, height=.4, color="#cb8a3e")
        request_ax.barh(row, done - ready, left=ready, height=.4, color="#b5cae2")
        received = [event["time_ns"] / 1e6 for event in report["request_state_events"]
                    if event["request_id"] == request_id and event["reason"] == "token_delivered"]
        request_ax.scatter(received, [row] * len(received), color="#18324b", s=26, zorder=4)
    request_ax.set_yticks(range(len(requests)), requests)
    request_ax.invert_yaxis()
    request_ax.set_title("A   Request events", loc="left", fontweight="bold")
    request_ax.legend(handles=[
        Patch(color="#cb8a3e", label="Ingress transfer"),
        Patch(color="#b5cae2", label="Queued or executing in Cell"),
        Line2D([0], [0], marker="o", color="#18324b", linestyle="", label="Token delivered"),
    ], loc="upper right", ncols=3, frameon=False)
    request_ax.margins(y=.34)

    for batch in report["batches"]:
        key = (batch["cell_id"], batch["step_id"])
        row = cells.index(key[0])
        step = steps[key]
        phases = {req["phase"] for req in step["step"]["schedule"]["requests"]}
        phase = next(iter(phases)) if len(phases) == 1 else "mixed"
        start, end = (step[k] * 1e3 for k in ("start_time_s", "end_time_s"))
        step_ax.barh(row, end - start, left=start, height=.43, color=PHASE_COLORS[phase],
                     linewidth=.65, edgecolor="white")
        for call in batch["calls"]:
            if any(phase and phase[0]["operation"] == "pp_send_recv" for phase in call["phases"]):
                at = call["start_time_ns"] / 1e6
                step_ax.barh(row + .27, call["end_time_ns"] / 1e6 - at, left=at,
                             height=.07, color="#dc704d", linewidth=0)
    step_ax.set_yticks(range(len(cells)), [cell_label[c] for c in cells])
    step_ax.invert_yaxis()
    step_ax.set_title("B   Completed batches and cross-satellite pipeline traffic",
                      loc="left", fontweight="bold")
    step_ax.legend(handles=[
        *(Patch(color=color, label=phase.capitalize()) for phase, color in PHASE_COLORS.items()),
        Patch(color="#dc704d", label="Pipeline transfer"),
    ], loc="upper right", ncols=4, frameon=False)
    step_ax.margins(y=.51)

    horizon = report["measurement"]["simulated_time_ns"] / 1e6
    for key, color in zip(cells, plt.cm.tab10.colors, strict=False):
        history = report["cells"][key]["load_history"]
        times = [0, *(record["time_s"] * 1e3 for record in history), horizon]
        resident = [0, *(record["pending_request_count"] for record in history), 0]
        waiting = [0, *(record["waiting_request_count"] for record in history), 0]
        state_ax.step(times, resident, where="post", label=f"{cell_label[key]} resident", color=color)
        state_ax.step(times, waiting, where="post", label=f"{cell_label[key]} waiting",
                      color=color, linestyle="--", alpha=.75)
    state_ax.set_title("C   Cell state after admission and batch completion",
                       loc="left", fontweight="bold")
    state_ax.set_ylabel("Requests")
    state_ax.set_xlabel("Global simulation time (ms)")
    state_ax.legend(loc="upper right", ncols=2, frameon=False)
    state_ax.margins(y=.22)
    for axis in (request_ax, step_ax, state_ax):
        axis.grid(axis="x", color="#e0e6ed", linewidth=.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    state_ax.set_xlim(-horizon * .012, horizon * 1.04)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        parser.error("a completed run is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(report, args.output)


if __name__ == "__main__":
    main()
