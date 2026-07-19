"""Shared evaluation pipeline for one fully materialized constellation design.

The evaluator deliberately knows nothing about DSE ranges.  Callers must pass
one complete :class:`ConstellationDesign`; fixed experiments and DSE candidates
therefore exercise the same architecture, hardware, cost, simulation, serving,
and current-runtime path.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .constellation_cost import estimate_constellation_cost
from ..runtime import EnergyIntensity, build_planned_energy_demand
from ..runtime import attach_energy_alignment, summarize_runtime_sunlight
from ..architecture import resolve_satellite_architecture
from ..runtime import build_runtime_scenario
from .runtime_dispatch import evaluate_runtime_performance
from ..runtime import apply_runtime_energy_envelope
from ..runtime import get_scheduler
from ..serving import estimate_serving_capacity
from ..orbit import run_simulation
from ..orbit import load_surrogate_trace_bundle


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
    satellite_architecture: Any
    planned_energy_demand: Any
    cost: Any
    simulation_config: Any
    runtime_energy_envelope: Any
    simulation: Any
    runtime_sunlight_summary: dict[str, Any]
    serving_spec: Any
    serving_throughput: Any
    runtime_scenario: Any
    runtime_performance: Any
    energy_alignment: Any
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
    satellite_architecture = resolve_satellite_architecture(
        getattr(experiment_spec, "hardware", None),
        base_dir=base_dir,
    )
    planned_energy_demand, energy_planning_warnings = _planned_energy_demand_for_design(
        experiment_spec,
        workload_generation=workload_generation,
        satellite_count=design.satellite_count,
        satellite_architecture=satellite_architecture,
        simulation_duration_s=getattr(input_simulation_config, "duration_s", None),
    )
    cost_limits = None
    cost_config = None
    single_satellite_case_path = None
    if satellite_architecture is not None:
        cost_limits, cost_config, single_satellite_case_path = satellite_architecture.cost_inputs()
    cost = estimate_constellation_cost(
        design,
        experiment_spec.costing,
        base_dir=base_dir,
        cost_limits=cost_limits,
        cost_config=cost_config,
        single_satellite_case_path=single_satellite_case_path,
    )
    resolved_simulation_config, runtime_energy_envelope = apply_runtime_energy_envelope(
        input_simulation_config,
        cost_result=cost,
        satellite_architecture=satellite_architecture,
    )
    resolved_simulation_config = _simulation_config_with_link_hardware(
        resolved_simulation_config,
        satellite_architecture=satellite_architecture,
    )
    simulation = run_simulation(
        resolved_simulation_config,
        constellation=design,
        workload_generation=workload_generation,
        base_dir=base_dir,
        experiment_id=evaluation_id,
    )
    runtime_sunlight_summary, sunlight_summary_warnings = _runtime_sunlight_summary(
        simulation,
        runtime_energy_envelope=runtime_energy_envelope,
        base_dir=base_dir,
    )
    serving_spec, serving_warnings = _serving_spec_for_satellite_architecture(
        experiment_spec,
        satellite_architecture=satellite_architecture,
    )
    serving_throughput = estimate_serving_capacity(
        serving_spec,
        hardware=satellite_architecture.hardware if satellite_architecture is not None else None,
    )

    pre_runtime_warnings = _dedupe(
        (
            *input_warnings,
            *energy_planning_warnings,
            *serving_warnings,
            *runtime_energy_envelope.warnings,
            *cost.warnings,
            *simulation.warnings,
            *sunlight_summary_warnings,
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
        satellite_architecture=satellite_architecture,
        warnings=pre_runtime_warnings,
    )
    runtime_performance = evaluate_runtime_performance(
        runtime_scenario,
        runtime_scenario.runtime_config,
    )
    runtime_performance, energy_alignment = attach_energy_alignment(
        runtime_performance,
        planned_energy_demand,
        runtime_energy_envelope=runtime_energy_envelope,
        runtime_sunlight_summary=runtime_sunlight_summary,
        tolerance_fraction=float(
            getattr(experiment_spec.costing, "energy_alignment_tolerance", 0.05)
        ),
    )
    warnings = _dedupe(
        (*pre_runtime_warnings, *runtime_performance.warnings, *energy_alignment.warnings)
    )
    return MaterializedDesignEvaluation(
        design=design,
        constellation_architecture=constellation_architecture,
        scheduler=scheduler,
        scheduler_decision=scheduler_decision,
        satellite_architecture=satellite_architecture,
        planned_energy_demand=planned_energy_demand,
        cost=cost,
        simulation_config=resolved_simulation_config,
        runtime_energy_envelope=runtime_energy_envelope,
        simulation=simulation,
        runtime_sunlight_summary=runtime_sunlight_summary,
        serving_spec=serving_spec,
        serving_throughput=serving_throughput,
        runtime_scenario=runtime_scenario,
        runtime_performance=runtime_performance,
        energy_alignment=energy_alignment,
        warnings=warnings,
    )


def _runtime_sunlight_summary(
    simulation: Any,
    *,
    runtime_energy_envelope: Any,
    base_dir: Path | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    trace_model = str(getattr(simulation, "trace_model", "") or "")
    if trace_model not in _FILE_BACKED_TRACE_MODELS or not bool(
        getattr(simulation, "trace_evaluated", False)
    ):
        return {}, ()
    try:
        trace_bundle = load_surrogate_trace_bundle(
            getattr(simulation, "trace_manifest", None),
            base_dir=base_dir,
        )
        summary = summarize_runtime_sunlight(trace_bundle.energy_windows)
    except (OSError, TypeError, ValueError) as exc:
        return {}, (f"runtime sunlight summary unavailable: {exc}",)
    if not summary.get("evaluated", False):
        return summary, (
            "runtime sunlight summary is invalid; energy alignment cannot validate the cost sunlit fraction",
        )
    if trace_model == "surrogate_orbit_trace":
        envelope_payload = _mapping(runtime_energy_envelope)
        trace_options = _mapping(envelope_payload.get("trace_options"))
        shadow_fraction = trace_options.get("shadow_fraction")
        energy_trace = _mapping(getattr(trace_bundle, "energy_trace", {}))
        energy_trace_provenance = _mapping(energy_trace.get("provenance"))
        orbit_period_value = energy_trace_provenance.get("orbit_period_s")
        if orbit_period_value is None:
            orbit_period_value = _mapping(
                _mapping(envelope_payload.get("provenance")).get("runtime_fields")
            ).get("orbit_period_s")
        try:
            declared_sunlit_fraction = 1.0 - float(shadow_fraction)
            actual_orbit_period_s = float(orbit_period_value)
        except (TypeError, ValueError, OverflowError):
            return summary, (
                "periodic runtime sunlight contract is unavailable from runtime_energy_envelope",
            )
        if (
            not math.isfinite(declared_sunlit_fraction)
            or not 0.0 <= declared_sunlit_fraction <= 1.0
            or not math.isfinite(actual_orbit_period_s)
            or actual_orbit_period_s <= 0.0
        ):
            return summary, (
                "periodic runtime sunlight contract has an invalid shadow fraction or materialized orbit period",
            )
        summary = {
            **summary,
            "observed_finite_horizon_sunlit_fraction": summary.get(
                "actual_sunlit_fraction"
            ),
            "actual_sunlit_fraction": declared_sunlit_fraction,
            "maximum_shadow_window_s": float(shadow_fraction) * actual_orbit_period_s,
            "periodic_orbit_period_s": actual_orbit_period_s,
            "measurement_basis": "materialized_periodic_shadow_contract",
        }
    elif trace_model == "circular_geometry_trace":
        geometry_summary = _mapping(
            getattr(trace_bundle, "energy_trace", {}).get(
                "sunlight_contract_summary"
            )
            if isinstance(getattr(trace_bundle, "energy_trace", None), dict)
            else None
        )
        if geometry_summary.get("evaluated", False):
            summary = {
                **summary,
                "observed_finite_horizon_sunlit_fraction": summary.get(
                    "actual_sunlit_fraction"
                ),
                "actual_sunlit_fraction": geometry_summary.get(
                    "actual_sunlit_fraction"
                ),
                "maximum_shadow_window_s": geometry_summary.get(
                    "maximum_shadow_window_s"
                ),
                "measurement_basis": "geometry_full_orbit_sampling",
                "geometry_orbit_average": geometry_summary,
            }
    return summary, ()


def _planned_energy_demand_for_design(
    experiment_spec: Any,
    *,
    workload_generation: Any,
    satellite_count: int,
    satellite_architecture: Any,
    simulation_duration_s: float | None,
) -> tuple[Any | None, tuple[str, ...]]:
    """Build an offered-workload energy audit without sizing static hardware."""

    costing = experiment_spec.costing
    workload_payload = _mapping(workload_generation)
    requests = workload_payload.get("requests")
    configured = bool(workload_payload.get("configured", False)) or isinstance(requests, (list, tuple))
    if not configured:
        return None, (
            "offered-workload energy accounting is unavailable because canonical offered requests are not configured; "
            "the energy audit is unavailable",
        )
    energy = getattr(satellite_architecture, "energy", None)
    if energy is None:
        return None, (
            "offered-workload energy accounting is unavailable because resolved cost power configuration is missing; "
            "the energy audit is unavailable",
        )
    intensity = EnergyIntensity(
        compute_standby_power_w=float(energy.compute_standby_power_w),
        communication_standby_power_w=float(
            energy.communication_standby_power_w
        ),
        compute_energy_j_per_token=float(energy.compute_energy_j_per_token),
        link_energy_j_per_byte=float(energy.link_energy_j_per_byte),
    )
    planned = build_planned_energy_demand(
        workload_generation,
        satellite_count=satellite_count,
        intensity=intensity,
        target_served_ratio=float(getattr(costing, "target_served_ratio", 1.0)),
        nominal_isl_hops_per_request=float(
            getattr(costing, "nominal_isl_hops_per_request", 0.0)
        ),
        simulation_duration_s=simulation_duration_s,
    )
    return planned, (
        "energy alignment audits offered-workload token/byte demand and does not read runtime served requests",
    )


def _serving_spec_for_satellite_architecture(
    experiment_spec: Any,
    *,
    satellite_architecture: Any,
) -> tuple[Any, tuple[str, ...]]:
    """Use resolver output as the single-design hardware authority."""

    serving = experiment_spec.serving
    if satellite_architecture is None:
        return serving, ()
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


def _simulation_config_with_link_hardware(simulation_config: Any, *, satellite_architecture: Any) -> Any:
    """Inject the immutable single-satellite terminal capability into trace inputs."""

    profile = getattr(satellite_architecture, "hardware", None)
    if profile is None:
        return simulation_config
    options = dict(getattr(simulation_config, "trace_options", {}) or {})
    options["satellite_link_hardware"] = {
        "terminal_count_per_satellite": int(getattr(profile, "laser_terminal_count", 0)),
        "terminal_bandwidth_gbps": float(getattr(profile, "resolved_laser_terminal_bandwidth_gbps", 0.0)),
        "total_bandwidth_gbps": float(getattr(profile, "inter_satellite_bandwidth_gbps", 0.0)),
    }
    return replace(simulation_config, trace_options=options)
