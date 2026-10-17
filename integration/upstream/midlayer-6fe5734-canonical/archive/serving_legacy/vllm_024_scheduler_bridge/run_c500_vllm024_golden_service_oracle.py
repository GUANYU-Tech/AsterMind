#!/usr/bin/env python3
"""Run a fail-closed golden-step-timing counterfactual on C500 Scheduler 0.24.

The Scheduler remains free running.  It must independently reproduce every
canonical batch before this diagnostic supplies that batch's observed plan,
service, and update timings.  A changed batch is an invalid counterfactual:
the runner stops rather than reusing a duration from a different workload.
This is evaluation-only oracle evidence and never enters Forward.
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
from src.serving import simulation as serving_simulation  # noqa: E402
from src.serving.schedule.schedule_step import BackendStep  # noqa: E402
from scripts.evaluation.serving.diagnostics.run_c500_vllm024_forward_candidate import (  # noqa: E402
    _candidate_runner,
)


_DEFAULT_CONFIG = Path("configs/evaluation/experiments/sharegpt_phase1.json")
_DEFAULT_OUTPUT = Path(
    "experiment_logs/serving/scheduler_candidates/"
    "c500-qwen3-4b-tp1-qps16-vllm024-golden-service-oracle/results.json"
)
_CELL_ID = "metax_c500/qwen3-4b/tp1_qps16"


class GoldenStepMismatch(RuntimeError):
    """The free-running Scheduler emitted a workload absent from golden trace."""


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--case", default=_CELL_ID)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _expected_signature(row: Mapping[str, object]) -> tuple[object, ...]:
    scheduled = row["scheduled_requests"]
    if not isinstance(scheduled, list) or not scheduled:
        raise ValueError("golden Scheduler row requires scheduled requests")
    requests = []
    for order, value in enumerate(scheduled):
        request = _mapping(value, f"golden scheduled request {order}")
        if set(request) != {
                "request_idx", "prefill_tokens", "decode_tokens",
                "scheduled_tokens"}:
            raise ValueError("golden scheduled request fields differ")
        request_index = _integer(request["request_idx"], "golden request index")
        prefill_tokens = _integer(
            request["prefill_tokens"], "golden prefill tokens")
        decode_tokens = _integer(
            request["decode_tokens"], "golden decode tokens")
        scheduled_tokens = _integer(
            request["scheduled_tokens"], "golden scheduled tokens", minimum=1)
        if prefill_tokens + decode_tokens != scheduled_tokens:
            raise ValueError("golden request token totals disagree")
        requests.append((request_index, prefill_tokens, decode_tokens))
    finished = row["finished_request_indexes"]
    if not isinstance(finished, list):
        raise TypeError("golden finished request indexes must be a list")
    finished_indexes = tuple(
        _integer(value, "golden finished request index") for value in finished)
    return tuple(requests), finished_indexes


def _actual_signature(step: BackendStep) -> tuple[object, ...]:
    if not isinstance(step, BackendStep):
        raise TypeError("oracle Scheduler step must be BackendStep")
    requests = []
    for request in step.schedule.requests:
        request_id = request.request_id
        prefix = "request-"
        if not request_id.startswith(prefix):
            raise GoldenStepMismatch(
                f"Scheduler request ID is not canonical: {request_id!r}")
        suffix = request_id.removeprefix(prefix)
        if not suffix.isdecimal() or str(int(suffix)) != suffix.lstrip("0"):
            # The compact request formatter uses twelve digits, including zero.
            if suffix != "000000000000":
                raise GoldenStepMismatch(
                    f"Scheduler request ID is not a canonical index: {request_id!r}")
        request_index = int(suffix)
        requests.append((
            request_index,
            request.input_tokens,
            request.output_tokens,
        ))
    finished_indexes = []
    for request_id in step.schedule.finished_request_ids:
        if not request_id.startswith("request-"):
            raise GoldenStepMismatch(
                f"Scheduler finished non-canonical request ID: {request_id!r}")
        suffix = request_id.removeprefix("request-")
        if not suffix.isdecimal():
            raise GoldenStepMismatch(
                f"Scheduler finished non-numeric request ID: {request_id!r}")
        finished_indexes.append(int(suffix))
    return tuple(requests), tuple(finished_indexes)


class _GoldenTimingOracle:
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        if not rows:
            raise ValueError("golden Scheduler trace is empty")
        self._expected = []
        self._step_indexes: dict[int, int] = {}
        previous_update_ns = 0
        for index, row in enumerate(rows):
            schedule_end_ns = _integer(
                row["schedule_end_ns"], "golden schedule end", minimum=0)
            update_start_ns = _integer(
                row["update_start_ns"], "golden update start", minimum=0)
            execution_start_ns = max(schedule_end_ns, previous_update_ns)
            if update_start_ns <= execution_start_ns:
                raise ValueError("golden Scheduler trace has non-positive service")
            step_start_ns = _integer(
                row["step_start_ns"], "golden step start", minimum=0)
            step_end_ns = _integer(
                row["step_end_ns"], "golden step end", minimum=0)
            if schedule_end_ns < step_start_ns or step_end_ns < update_start_ns:
                raise ValueError("golden Scheduler trace has unordered host timing")
            self._expected.append({
                "signature": _expected_signature(row),
                "plan_s": (schedule_end_ns - step_start_ns) / 1_000_000_000.0,
                "service_s": (
                    (update_start_ns - execution_start_ns) / 1_000_000_000.0),
                "update_s": (step_end_ns - update_start_ns) / 1_000_000_000.0,
            })
            previous_update_ns = update_start_ns
        self._index = 0

    def prepare(self, step: BackendStep) -> BackendStep:
        if self._index >= len(self._expected):
            raise GoldenStepMismatch(
                "free-running Scheduler emitted more steps than golden trace")
        actual = _actual_signature(step)
        expected = self._expected[self._index]
        if actual != expected["signature"]:
            raise GoldenStepMismatch(
                "golden workload mismatch at step "
                f"{self._index}: expected={expected['signature']!r}, "
                f"actual={actual!r}")
        timed = BackendStep(
            step.schedule,
            step.kv_cache_events,
            step.generated_tokens,
            float(expected["plan_s"]),
        )
        self._step_indexes[id(timed)] = self._index
        return timed

    def duration(self, step: BackendStep) -> float:
        if self._index >= len(self._expected):
            raise GoldenStepMismatch("golden duration requested after trace end")
        if id(step) not in self._step_indexes:
            raise GoldenStepMismatch("golden duration has no planned step")
        step_index = self._step_indexes[id(step)]
        if step_index != self._index:
            raise GoldenStepMismatch("golden duration order differs from Scheduler order")
        expected = self._expected[step_index]
        actual = _actual_signature(step)
        if actual != expected["signature"]:
            raise GoldenStepMismatch("golden duration workload changed after plan")
        self._index += 1
        return float(expected["service_s"])

    def update(self, step: BackendStep) -> float:
        if id(step) not in self._step_indexes:
            raise GoldenStepMismatch("golden update has no planned step")
        completed_index = self._step_indexes.pop(id(step))
        if completed_index >= len(self._expected):
            raise GoldenStepMismatch("golden update requested outside trace range")
        expected = self._expected[completed_index]
        if _actual_signature(step) != expected["signature"]:
            raise GoldenStepMismatch("golden update workload changed after execution")
        return float(expected["update_s"])

    def assert_consumed(self) -> None:
        if self._index != len(self._expected):
            raise GoldenStepMismatch(
                "free-running Scheduler stopped before consuming golden trace: "
                f"consumed={self._index}, golden={len(self._expected)}")

    @property
    def consumed_steps(self) -> int:
        return self._index

    @property
    def golden_steps(self) -> int:
        return len(self._expected)


class _GoldenTimingBackend:
    """Overlay observed plan/update clock values without changing scheduling."""

    def __init__(self, backend: object, oracle: _GoldenTimingOracle) -> None:
        self._backend = backend
        self._oracle = oracle
        if not hasattr(backend, "config"):
            raise TypeError("golden timing backend requires backend config")
        self.config = backend.config
        self._original_steps: dict[int, BackendStep] = {}

    def add_request(self, request: object) -> None:
        self._backend.add_request(request)

    def add_requests(self, requests: tuple[object, ...]) -> None:
        self._backend.add_requests(requests)

    def plan_step(self, virtual_time_s: float) -> BackendStep | None:
        original = self._backend.plan_step(virtual_time_s)
        if original is None:
            return None
        if not isinstance(original, BackendStep):
            raise TypeError("wrapped Scheduler returned non-BackendStep")
        timed = self._oracle.prepare(original)
        self._original_steps[id(timed)] = original
        return timed

    def complete_step(self, step: BackendStep, virtual_time_s: float) -> float:
        if id(step) not in self._original_steps:
            raise ValueError("golden timing completion has no original step")
        original = self._original_steps.pop(id(step))
        self._backend.complete_step(original, virtual_time_s)
        return self._oracle.update(step)

    def has_work(self) -> bool:
        return self._backend.has_work()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _arguments(argv)
    if args.case != _CELL_ID:
        raise NotImplementedError("golden service oracle currently supports only 4B QPS16")
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise ValueError("--source-root does not exist")
    config_path = (REPOSITORY_ROOT / args.config).resolve()
    config = compact.load_phase1_config(config_path)
    cells = compact.select_phase1_cells(config, [args.case], all_cases=False)
    if len(cells) != 1:
        raise RuntimeError("golden service oracle requires exactly one cell")
    cell = cells[0]
    case = load_compact_case(compact.compact_case_path(
        REPOSITORY_ROOT, config, cell))
    if case.scheduler_steps is None:
        raise ValueError("golden service oracle requires Scheduler trace")
    oracle = _GoldenTimingOracle(case.scheduler_steps)
    forward_argv = compact.build_simulation_argv(
        REPOSITORY_ROOT,
        config,
        cell,
        case,
        execution_mode="accelerated",
        serving_timing_mode="device_only",
    )
    original_schedule_runner = serving_simulation.run_single_pool_schedule

    def run_with_golden_timing(
        backend, requests, step_duration, scheduler_pipeline_depth,
        on_step_complete=None, tail_drain=None, timing_mode="device_only",
    ):
        if tail_drain is not None:
            raise RuntimeError("golden service oracle requires ordinary scheduling")

        def duration(step: BackendStep) -> float:
            # Preserve the normal Forward execution artifact, but make only
            # the virtual Scheduler clock consume the golden service value.
            step_duration(step)
            return oracle.duration(step)

        timing_backend = _GoldenTimingBackend(backend, oracle)
        return original_schedule_runner(
            timing_backend,
            requests,
            duration,
            scheduler_pipeline_depth,
            on_step_complete=on_step_complete,
            tail_drain=None,
            timing_mode="device_plus_launch_plus_scheduler",
        )

    serving_simulation.run_single_pool_schedule = run_with_golden_timing
    output_path = args.output
    if not output_path.is_absolute():
        output_path = REPOSITORY_ROOT / output_path
    try:
        payload = _candidate_runner(forward_argv, source_root, {})
        oracle.assert_consumed()
    except BaseException as error:
        result = {
            "schema_version": 1,
            "result_kind": "c500_vllm_0_24_golden_step_timing_oracle",
            "status": "invalid_counterfactual",
            "cell_id": cell.cell_id,
            "reason": (
                "A golden duration is valid only for its exact canonical "
                "workload; no cross-step fallback was permitted."),
            "golden_trace_steps": oracle.golden_steps,
            "consumed_exact_steps": oracle.consumed_steps,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json(output_path, result)
        return result
    finally:
        serving_simulation.run_single_pool_schedule = original_schedule_runner

    observed = compact._observed_metrics(case)
    comparison = compact._compare_prediction(observed, payload, len(case.requests))
    scheduler_runtime = _mapping(payload["scheduler_runtime"], "scheduler runtime")
    result = {
        "schema_version": 1,
        "result_kind": "c500_vllm_0_24_golden_step_timing_oracle",
        "status": "completed_candidate_current_runtime_source_not_capture_exact",
        "cell_id": cell.cell_id,
        "timing_source": (
            "canonical Scheduler trace plan/service/update boundaries; service="
            "update_start - max(schedule_end, previous_update_start)"),
        "duration_observation_role": "evaluation_oracle_only",
        "forward_duration_observation_consumed": False,
        "golden_trace_steps": oracle.golden_steps,
        "consumed_exact_steps": oracle.consumed_steps,
        "scheduler_runtime": scheduler_runtime,
        "metrics": comparison,
    }
    _write_json(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    result = run(argv)
    print(json.dumps({
        "output": str(_DEFAULT_OUTPUT),
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
