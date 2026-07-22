"""Shared evaluation pipeline for one fully materialized constellation design.

The evaluator deliberately knows nothing about DSE ranges.  Callers must pass
one complete :class:`ConstellationDesign`; fixed experiments and DSE candidates
therefore exercise the same architecture, hardware, cost, simulation, serving,
and current-runtime path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .constellation_cost import estimate_constellation_cost
from ..architecture import resolve_experiment_hardware
from ..runtime import build_runtime_scenario
from .runtime_dispatch import evaluate_runtime_performance
from ..runtime import get_scheduler
from ..serving import estimate_serving_throughput
from ..orbit import run_simulation


ARCHITECTURE_SURROGATE_WARNING = (
    "constellation architecture analysis is a lightweight surrogate; STK-calibrated visibility, "
    "shadow, routing, and queueing are not evaluated"
)

_FILE_BACKED_TRACE_MODELS = {"surrogate_orbit_trace", "circular_geometry_trace"}


@dataclass(frozen=True)
class MaterializedDesignEvaluation:
    """Outputs produced by the common single-design evaluation path."""

    design: Any
    constellation_architecture: dict[str, Any]
    scheduler: Any
    scheduler_decision: Any
    resolved_hardware: Any
    cost: Any
    simulation_config: Any
    simulation: Any
    serving_spec: Any
    serving_throughput: Any
    runtime_scenario: Any
    runtime_performance: Any
    warnings: tuple[str, ...]


def evaluate_materialized_design(
    experiment_spec: Any,
    *,
    design: Any,
    workload_generation: Any,
    serving_workload: Any,
    evaluation_id: str,
    simulation_config: Any | None = None,
    input_warnings: tuple[str, ...] = (),
) -> MaterializedDesignEvaluation:
    """Evaluate exactly one complete design without consulting DSE ranges."""

    base_dir = Path(experiment_spec.base_dir) if getattr(experiment_spec, "base_dir", None) else None
    input_simulation_config = (
        simulation_config
        if simulation_config is not None
        else getattr(experiment_spec, "simulation", None)
    )
    if input_simulation_config is None:
        raise ValueError("a materialized design evaluation requires simulation configuration")
    input_simulation_config = _simulation_config_with_default_trace_dir(
        input_simulation_config,
        evaluation_id=evaluation_id,
    )

    # Keep this order explicit: it is the shared fixed-design/DSE contract.
    constellation_architecture = design.architecture_analysis()
    scheduler = get_scheduler(experiment_spec.scheduler)
    scheduler_decision = scheduler.select_action(state=None, request=None)
    resolved_hardware = resolve_experiment_hardware(
        getattr(experiment_spec, "hardware", None),
        base_dir=base_dir,
    )
    cost = estimate_constellation_cost(
        design,
        experiment_spec.costing,
        base_dir=base_dir,
        cost_limits=resolved_hardware.cost_limits,
        cost_config=resolved_hardware.cost_config,
        single_satellite_case_path=resolved_hardware.single_satellite_case_path,
    )
    resolved_simulation_config = _simulation_config_with_link_hardware(
        input_simulation_config,
        resolved_hardware=resolved_hardware,
    )
    simulation = run_simulation(
        resolved_simulation_config,
        constellation=design,
        workload_generation=workload_generation,
        base_dir=base_dir,
        experiment_id=evaluation_id,
    )
    serving_spec, serving_warnings = _serving_spec_for_resolved_hardware(
        experiment_spec,
        resolved_hardware=resolved_hardware,
    )
    serving_throughput = estimate_serving_throughput(
        serving_spec,
        hardware=resolved_hardware.satellite_profile,
        token_distribution=serving_workload,
        constellation_satellite_count=design.satellite_count,
    )

    pre_runtime_warnings = _dedupe(
        (
            *input_warnings,
            *resolved_hardware.warnings,
            *serving_warnings,
            *cost.warnings,
            *simulation.warnings,
            *serving_throughput.warnings,
            ARCHITECTURE_SURROGATE_WARNING,
            *tuple(constellation_architecture.get("warnings", ())),
        )
    )
    runtime_scenario = build_runtime_scenario(
        scenario_id=evaluation_id,
        design_id=str(design.name),
        workload_generation=workload_generation,
        workload_spec=getattr(experiment_spec, "workload", None),
        constellation=design,
        constellation_architecture=constellation_architecture,
        serving_spec=serving_spec,
        serving_throughput=serving_throughput,
        simulation_result=simulation,
        simulation_config=resolved_simulation_config,
        cost_result=cost,
        scheduler_spec=scheduler.to_runtime_strategy(),
        hardware=getattr(experiment_spec, "hardware", None),
        resolved_hardware=resolved_hardware,
        warnings=pre_runtime_warnings,
    )
    runtime_performance = evaluate_runtime_performance(
        runtime_scenario,
        runtime_scenario.runtime_config,
    )
    warnings = _dedupe((*pre_runtime_warnings, *runtime_performance.warnings))
    return MaterializedDesignEvaluation(
        design=design,
        constellation_architecture=constellation_architecture,
        scheduler=scheduler,
        scheduler_decision=scheduler_decision,
        resolved_hardware=resolved_hardware,
        cost=cost,
        simulation_config=resolved_simulation_config,
        simulation=simulation,
        serving_spec=serving_spec,
        serving_throughput=serving_throughput,
        runtime_scenario=runtime_scenario,
        runtime_performance=runtime_performance,
        warnings=warnings,
    )


def _serving_spec_for_resolved_hardware(
    experiment_spec: Any,
    *,
    resolved_hardware: Any,
) -> tuple[Any, tuple[str, ...]]:
    """Use resolver output as the single-design hardware authority."""

    serving = experiment_spec.serving
    if resolved_hardware.satellite_profile is None:
        return serving, ()

    cost_or_top_level_hardware = (
        getattr(experiment_spec, "hardware", None) is not None
        or resolved_hardware.single_satellite_case_path is not None
    )
    if not cost_or_top_level_hardware:
        return replace(serving, hardware=None), ()

    return replace(serving, hardware=None), ()


def _dedupe(values: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if value is not None))


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        return dict(converted) if isinstance(converted, dict) else {}
    return {}


def safe_trace_path_component(value: Any) -> str:
    """Return one deterministic path component for a trace identity.

    Design and experiment identifiers are user input.  Keeping path separators
    or ``..`` in them would let a DSE row escape its trace root; replacing an
    unsafe identifier without a digest would also make distinct identifiers
    collide.  Safe short identifiers stay readable, while normalized or long
    identifiers receive a stable digest suffix.
    """

    text = str(value or "trace").strip()
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in text
    ).strip("._")
    if not normalized:
        normalized = "trace"
    unsafe = (
        normalized != text
        or text in {".", ".."}
        or len(normalized.encode("utf-8")) > 120
    )
    if not unsafe:
        return normalized
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    readable = _truncate_utf8(normalized, 96).rstrip("._-") or "trace"
    return f"{readable}.hash-{digest}"


def _truncate_utf8(value: str, max_bytes: int) -> str:
    byte_count = 0
    characters: list[str] = []
    for character in value:
        encoded_size = len(character.encode("utf-8"))
        if byte_count + encoded_size > max_bytes:
            break
        characters.append(character)
        byte_count += encoded_size
    return "".join(characters)


def _simulation_config_with_default_trace_dir(
    simulation_config: Any,
    *,
    evaluation_id: str,
) -> Any:
    """Materialize a safe default trace directory for fixed evaluations."""

    trace_model = str(getattr(simulation_config, "trace_model", "skeleton"))
    if trace_model not in _FILE_BACKED_TRACE_MODELS:
        return simulation_config
    if getattr(simulation_config, "trace_output_dir", None):
        return simulation_config
    try:
        return replace(
            simulation_config,
            trace_output_dir=str(
                Path("results")
                / "simulation_traces"
                / safe_trace_path_component(evaluation_id)
            ),
        )
    except TypeError:
        return simulation_config


def _simulation_config_with_link_hardware(simulation_config: Any, *, resolved_hardware: Any) -> Any:
    """Inject the immutable single-satellite terminal capability into trace inputs."""

    profile = getattr(resolved_hardware, "satellite_profile", None)
    if profile is None:
        return simulation_config
    options = dict(getattr(simulation_config, "trace_options", {}) or {})
    options["satellite_link_hardware"] = {
        "terminal_count_per_satellite": int(getattr(profile, "laser_terminal_count", 0)),
        "terminal_bandwidth_gbps": float(getattr(profile, "resolved_laser_terminal_bandwidth_gbps", 0.0)),
        "total_bandwidth_gbps": float(getattr(profile, "inter_satellite_bandwidth_gbps", 0.0)),
    }
    return replace(simulation_config, trace_options=options)
