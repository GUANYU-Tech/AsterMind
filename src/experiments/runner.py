"""Compose Workload, Architecture, Orbit/network, Serving and Runtime."""

from __future__ import annotations

import gzip
import json
import uuid
from pathlib import Path

from orbitserve.workload import materialize_workload_bundle
from .design_evaluation import evaluate_materialized_design
from .models import ExperimentResult


def run_experiment(spec, *, output_dir=None, progress=None, _workers=None):
    if spec.physical_network:
        from orbitserve.runtime import check_network_backend

        check_network_backend(
            spec.physical_network["network"].get("transport_model", "queue")
        )
    root = Path(
        output_dir
        or Path("results/experiments")
        / (spec.experiment_id + "-" + uuid.uuid4().hex[:8])
    )
    root.mkdir(parents=True, exist_ok=True)
    workload = materialize_workload_bundle(
        spec.workload, base_dir=Path(spec.base_dir or ".")
    ).to_dict()
    if spec.ground_baseline is not None:
        from orbitserve.baselines.ground.contracts import (
            GroundScenario,
            GroundStrategySpec,
        )
        from .runtime_dispatch import evaluate_runtime_performance

        ground = evaluate_runtime_performance(
            GroundScenario(
                scenario_id=spec.experiment_id,
                requests=tuple(workload["session_requests"]),
                regions=tuple(workload["regions"]),
                gateways=tuple(workload["gateways"]),
                runtime_config=GroundStrategySpec.from_dict(spec.ground_baseline),
                time_horizon_s=workload["manifest"].get("time_horizon_s"),
            )
        )
        return ExperimentResult(
            spec.experiment_id,
            spec.to_dict(),
            {},
            {},
            {},
            ground,
            ground.details["terrestrial_costing"],
            {"evaluated": False, "rows": []},
            {
                "workload": "evaluated",
                "runtime": "evaluated",
                "orbit": "not_required_terrestrial_baseline",
            },
            ground.warnings,
        )
    evaluation = evaluate_materialized_design(
        spec,
        physical_network=spec.physical_network,
        workload_generation=workload,
        output_dir=root,
        progress=progress,
        _workers=_workers,
    )
    dse = {"evaluated": False, "rows": []}
    if spec.dse.get("enabled"):
        from orbitserve.dse import run_dse

        dse = run_dse(
            spec,
            workload_generation=workload,
            output_dir=root / "dse",
            progress=progress,
            _workers=_workers,
        )
    runtime = evaluation.runtime
    for name, rows in (
        ("runtime_events.jsonl.gz", runtime.state_summary["event_log"]),
        ("runtime_requests.jsonl.gz", runtime.request_records),
    ):
        with gzip.open(root / name, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    )
                    + "\n"
                )
    result = ExperimentResult(
        spec.experiment_id,
        spec.to_dict(),
        evaluation.architecture,
        evaluation.serving.to_dict(),
        {
            "checks": evaluation.network["checks"],
            "result_index": str(root / "network/network_result.json"),
            "trace_id": evaluation.network['trace_id'],
            "physical_dataset": evaluation.network.get('provenance', {}).get('physical_dataset'),
        },
        runtime,
        evaluation.cost,
        dse,
        {
            name: "evaluated"
            for name in (
                "architecture",
                "orbit",
                "network",
                "workload",
                "serving",
                "runtime",
                "cost",
            )
        },
        tuple(dict.fromkeys((*evaluation.network["warnings"], *runtime.warnings))),
    )
    (root / "experiment_result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result
