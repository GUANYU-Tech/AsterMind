#!/usr/bin/env python3
"""Summarize 910B ShareGPT step/device/Host decomposition.

Observed scheduler traces and Forward predictions have different Scheduler
trajectories, so this tool compares phase aggregates and never pairs a
prediction with a differently numbered observed step.  The observed
``step_end-step_start`` envelope is retained separately from the Device
execution interval; it is not used to fit a Forward parameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import zstandard as zstd


def _load_jsonl_zst(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        raw = zstd.ZstdDecompressor().stream_reader(stream).read()
    lines = raw.decode("utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


HARDWARE_ID = "ascend_910b"


def _phase(observation: dict[str, Any]) -> str:
    prefill = observation["prefill_tokens"]
    decode = observation["decode_tokens"]
    if prefill and decode:
        return "mixed"
    if prefill:
        return "prefill"
    if decode:
        return "decode"
    raise ValueError("scheduler observation has no prefill/decode tokens")


def _sum_observed(rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    selected = [row for row in rows if _phase(row) == phase]
    wall_ms = sum(
        (row["step_end_ns"] - row["step_start_ns"]) / 1_000_000.0
        for row in selected
    )
    device_ms = sum(
        (row["device_execution_end_ns"] - row["device_execution_start_ns"])
        / 1_000_000.0
        for row in selected
    )
    return {
        "step_count": len(selected),
        "observed_step_envelope_ms": wall_ms,
        "observed_device_busy_ms": device_ms,
        "observed_host_dispatch_envelope_ms": wall_ms - device_ms,
        "fallback_step_count": sum(
            row["graph_fallback_reason"] is not None for row in selected
        ),
    }


def _sum_predicted(steps: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    selected = [step for step in steps if step["phase"] == phase]
    device_ms = sum(step["predicted_device_s"] * 1_000.0 for step in selected)
    wall_ms = sum(step["step_wall_s"] * 1_000.0 for step in selected)
    launch_cpu_ms = sum(step["launch_cpu_s"] * 1_000.0 for step in selected)
    scheduler_ms = sum(
        (step["scheduler_plan_cpu_s"] + step["scheduler_update_cpu_s"])
        * 1_000.0
        for step in selected
    )
    return {
        "step_count": len(selected),
        "predicted_step_envelope_ms": wall_ms,
        "predicted_device_ms": device_ms,
        "predicted_host_critical_path_ms": wall_ms - device_ms,
        "predicted_host_submit_cpu_ms": launch_cpu_ms,
        "predicted_scheduler_cpu_ms": scheduler_ms,
    }


def build_decomposition(
    repository_root: Path,
    result_paths: list[Path],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for result_path in result_paths:
        payload = json.loads(result_path.read_text())
        for cell in payload["cells"]:
            hardware = cell["hardware_id"]
            if hardware != HARDWARE_ID:
                # This report reads the 910B scheduler-trace schema, which the
                # other cards do not share.  Record the skip rather than
                # failing inside a foreign row's keys.
                excluded.append(
                    {"cell_id": cell["cell_id"], "hardware_id": hardware}
                )
                continue
            model = cell["model_id"]
            case_id = cell["case_id"]
            dataset_case = (
                repository_root / "serving_dataset" / "sharegpt" / hardware
                / model / case_id
            )
            observations = _load_jsonl_zst(
                dataset_case / "scheduler_steps.jsonl.zst"
            )
            phases = {}
            for phase in ("prefill", "mixed", "decode"):
                observed = _sum_observed(observations, phase)
                predicted = _sum_predicted(cell["host_timing"]["steps"], phase)
                # This is a phase aggregate across independent trajectories;
                # retain it as a closure diagnostic, never as a fit target.
                observed_host = observed["observed_host_dispatch_envelope_ms"]
                phases[phase] = {
                    "observed": observed,
                    "predicted": predicted,
                    "phase_host_closure_gap_ms": (
                        observed_host
                        - predicted["predicted_host_critical_path_ms"]
                    ),
                }
            cells.append(
                {
                    "cell_id": cell["cell_id"],
                    "result_path": str(result_path),
                    "observed_trace_path": str(
                        dataset_case / "scheduler_steps.jsonl.zst"
                    ),
                    "phases": phases,
                    "trajectory_note": (
                        "observed canonical trace and Forward use independent "
                        "Scheduler trajectories; phase totals are diagnostic "
                        "and are not paired observations"
                    ),
                }
            )
    if not cells:
        raise ValueError(
            f"no {HARDWARE_ID} cell in the supplied results; "
            f"skipped {len(excluded)} cell(s) from other hardware"
        )
    return {
        "schema_version": 2,
        "hardware_id": HARDWARE_ID,
        "excluded_cells": excluded,
        "evidence_policy": {
            "forward_duration_observation_consumed": False,
            "serving_duration_fitted": False,
            "observed_host_dispatch_is": "step envelope minus device execution interval",
        },
        "cells": cells,
        "mechanism_evidence": [
            {
                "mechanism": "capture-limit eager fallback",
                "evidence": "scheduler trace runtime_cudagraph_mode=NONE and graph_fallback_reason=above_capture_limit",
                "independent_source": "microbench/ascend_910b/runtime/launch/references/host_launch_20260830.json",
            },
            {
                "mechanism": "compiled-piece/TorchNPU/MoE/attention framework dispatch",
                "evidence": "same-run selected-step msprof + stack correlation",
                "independent_source": "microbench/ascend_910b/runtime/launch and communication/collectives count sweeps",
            },
        ],
        "repair_status": {
            "status": "blocked_pending_source_authored_helper_contract",
            "required_scope": [
                "attention TensorMove helper occurrences",
                "MoE routing Index/NotEqual/Mul occurrences",
                "model/input/output boundary helper occurrences",
            ],
            "forbidden_shortcut": (
                "Do not fit the phase closure gap or add a case/model constant."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Ascend 910B ShareGPT step decomposition",
        "",
        "Observed scheduler traces and Forward predictions are independent trajectories. Phase totals below are closure diagnostics; no serving duration is fitted or consumed by Forward.",
        "",
    ]
    for cell in report["cells"]:
        lines.extend([f"## `{cell['cell_id']}`", "", "| phase | observed steps | observed Device busy (ms) | observed Host/dispatch envelope (ms) | modeled Host critical path (ms) | closure gap (ms) | fallback steps |", "|---|---:|---:|---:|---:|---:|---:|"])
        for phase in ("prefill", "mixed", "decode"):
            item = cell["phases"][phase]
            obs = item["observed"]
            pred = item["predicted"]
            lines.append(
                f"| {phase} | {obs['step_count']} | {obs['observed_device_busy_ms']:.3f} | {obs['observed_host_dispatch_envelope_ms']:.3f} | {pred['predicted_host_critical_path_ms']:.3f} | {item['phase_host_closure_gap_ms']:.3f} | {obs['fallback_step_count']} |"
            )
        lines.append("")
    if report["excluded_cells"]:
        lines.extend([
            "## Excluded cells",
            "",
            "This report reads the 910B scheduler-trace schema; cells from other hardware are skipped:",
            "",
        ])
        lines.extend(
            f"- `{item['cell_id']}` (`{item['hardware_id']}`)"
            for item in report["excluded_cells"]
        )
        lines.append("")
    lines.extend([
        "## Where the closure gap sits",
        "",
        "Measured from the table above, per phase and per step:",
        "",
        "| cell | prefill | mixed | decode |",
        "|---|---:|---:|---:|",
    ])
    for cell in report["cells"]:
        columns = []
        for phase in ("prefill", "mixed", "decode"):
            item = cell["phases"][phase]
            count = item["observed"]["step_count"]
            gap = item["phase_host_closure_gap_ms"]
            columns.append(
                f"{gap:+.1f} ms / {count} = {gap / count:+.3f} ms"
                if count
                else "-"
            )
        lines.append(f"| `{cell['cell_id']}` | " + " | ".join(columns) + " |")
    lines.extend([
        "",
        "## Mechanism conclusion",
        "",
        "The existing Host launch profile measures submit-only service, while runtime-prepare rules cover only source-authored occurrences represented in the current KernelPlan. The independent count sweeps are microsecond-scale and do not justify a fitted per-step residual. The remaining legal repair is a source-authored 0/1/N helper-launch contract in Serving/lowering, with graph-domain shape and first-occurrence elision retained explicitly.",
        "",
        "The per-phase distribution above is measured per run and is not asserted here: read it off the table rather than carrying a remembered shape, because the phase that dominates has changed as the model changed.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_decomposition(args.repository_root, args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(report) + "\n")
    print(json.dumps({"cells": len(report["cells"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
