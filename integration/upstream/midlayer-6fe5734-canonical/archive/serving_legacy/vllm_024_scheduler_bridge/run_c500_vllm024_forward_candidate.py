#!/usr/bin/env python3
"""Run one C500 Forward candidate with the local real vLLM 0.24 Scheduler.

The bridge consumes scheduler decisions only.  Midlayer's ordinary production
Forward still owns workload construction, kernel lowering, and prediction.
The result is intentionally labelled candidate because the current C500
runtime source hashes may not equal the source snapshot that produced the
ground-truth capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.serving.metrics import sharegpt_compact_evaluation as compact  # noqa: E402
from evaluation.serving.simulation import simulation_runner  # noqa: E402
from src.serving.backend.metax_c500_vllm_024_scheduler import (  # noqa: E402
    C500VLLM024SchedulerControlPlane,
)
from src.serving.backend.vllm import VLLMBackend  # noqa: E402


_DEFAULT_CONFIG = Path("configs/evaluation/experiments/sharegpt_phase1.json")
_DEFAULT_OUTPUT = Path(
    "experiment_logs/serving/scheduler_candidates/"
    "c500-qwen3-4b-tp1-qps16-vllm024-current-source/results.json"
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", required=True, type=Path,
        help="local copy of the C500 container's vLLM Python package root")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--case", default="metax_c500/qwen3-4b/tp1_qps16",
        help="one compact cell: hardware/model/case")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _candidate_runner(
    argv: Sequence[str], source_root: Path, diagnostic_state: dict[str, object],
) -> Mapping[str, object]:
    original_from_model = VLLMBackend.__dict__["from_model"]

    def from_c500_vllm024(cls, backend_config, scheduler_config):
        control_plane = C500VLLM024SchedulerControlPlane(
            backend_config,
            scheduler_config,
            source_root=source_root,
        )
        diagnostic_state["control_plane"] = control_plane
        return cls(backend_config, control_plane)

    VLLMBackend.from_model = classmethod(from_c500_vllm024)
    try:
        return simulation_runner.run_cli_args(argv)
    finally:
        VLLMBackend.from_model = original_from_model


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _arguments(argv)
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise ValueError("--source-root does not exist")
    config_path = (REPOSITORY_ROOT / args.config).resolve()
    config = compact.load_phase1_config(config_path)
    cells = compact.select_phase1_cells(config, [args.case], all_cases=False)
    if len(cells) != 1:
        raise RuntimeError("C500 vLLM 0.24 candidate requires exactly one cell")
    cell = cells[0]
    if (cell.hardware.hardware_id, cell.model.dataset_model_id, cell.case_id) != (
            "metax_c500", "qwen3-4b", "tp1_qps16"):
        raise NotImplementedError(
            "C500 vLLM 0.24 candidate currently supports only "
            "metax_c500/qwen3-4b/tp1_qps16")
    case = load_compact_case(compact.compact_case_path(
        REPOSITORY_ROOT, config, cell))
    forward_argv = compact.build_simulation_argv(
        REPOSITORY_ROOT,
        config,
        cell,
        case,
        execution_mode="accelerated",
        serving_timing_mode="device_only",
    )
    diagnostic_state: dict[str, object] = {}
    try:
        payload = _candidate_runner(forward_argv, source_root, diagnostic_state)
    except BaseException as error:
        failure_path = args.output
        if not failure_path.is_absolute():
            failure_path = REPOSITORY_ROOT / failure_path
        control_plane = diagnostic_state["control_plane"]
        if not isinstance(control_plane, C500VLLM024SchedulerControlPlane):
            raise AssertionError("candidate did not construct its Scheduler")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.with_name("failure.json").write_text(json.dumps({
            "schema_version": 1,
            "error_type": type(error).__name__,
            "error": str(error),
            "scheduler_protocol_tail": control_plane.debug_history,
        }, indent=2, sort_keys=True) + "\n")
        raise
    scheduler_runtime = _mapping(payload["scheduler_runtime"], "scheduler runtime")
    if set(scheduler_runtime) != {"main"}:
        raise RuntimeError("candidate Forward returned unexpected Scheduler pools")
    runtime = _mapping(scheduler_runtime["main"], "main Scheduler runtime")
    if runtime["version"] != "0.24.0":
        raise RuntimeError("candidate Forward did not use vLLM 0.24")
    if runtime["source_snapshot_status"] != (
            "current_runtime_source_not_capture_exact"):
        raise RuntimeError("candidate source provenance label mismatched")
    observed = compact._observed_metrics(case)
    comparison = compact._compare_prediction(observed, payload, len(case.requests))
    result = {
        "schema_version": 1,
        "result_kind": "c500_vllm_0_24_scheduler_forward_candidate",
        "accuracy_claim_status": "candidate_current_runtime_source_not_capture_exact",
        "reason": (
            "The Scheduler is real vLLM 0.24 current C500 runtime source, "
            "but its EngineCore source SHA differs from the capture snapshot; "
            "this result is not a capture-exact accuracy claim."),
        "cell_id": cell.cell_id,
        "forward": {
            "main_chain_status": payload["main_chain_status"],
            "performance_timing_status": payload["performance_timing_status"],
            "execution_mode": payload["execution_mode"],
            "serving_timing_mode": payload["serving_timing_mode"],
            "execution_artifact": payload["execution_artifact"],
            "scheduler_runtime": runtime,
        },
        "metrics": comparison,
    }
    output_path = args.output
    if not output_path.is_absolute():
        output_path = REPOSITORY_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    result = run(argv)
    print(json.dumps({
        "cell_id": result["cell_id"],
        "output": str(_DEFAULT_OUTPUT),
        "result_kind": result["result_kind"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
