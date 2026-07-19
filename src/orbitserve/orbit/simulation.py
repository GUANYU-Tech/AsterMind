"""Simulation trace entry points for OrbitServe experiments.

The v1 framework still does not implement high-fidelity orbit propagation, but
it can emit explicit placeholder results, the compute-only ideal trace, and a
deterministic surrogate access/link/energy trace for preliminary runtime checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import to_jsonable
from ..common import PerformanceMetrics, SLOMetrics
from .simulation_traces import CIRCULAR_GEOMETRY_TRACE, RuntimeTraceManifest, SURROGATE_ORBIT_TRACE


_REMOVED_TRACE_PHYSICAL_INPUTS = (
    "link_profiles",
    "isl_bandwidth_bps",
    "sgl_bandwidth_bps",
)


def _pick(data: dict[str, Any], key: str, default: Any) -> Any:
    return data[key] if key in data else default


@dataclass(frozen=True)
class SimulationConfig:
    duration_s: float = 21600.0
    step_s: float = 60.0
    epoch: str = "2026-06-16T00:00:00Z"
    use_stk_calibrated_surrogate: bool = False
    trace_model: str = "skeleton"
    trace_output_dir: str | None = None
    trace_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        removed = sorted(key for key in _REMOVED_TRACE_PHYSICAL_INPUTS if key in self.trace_options)
        if not removed:
            return
        fields = ", ".join(f"simulation.trace_options.{key}" for key in removed)
        raise ValueError(
            f"{fields} are no longer supported physical-link inputs; "
            "define ISL parameters in constellation orbital-set clusters and SGL parameters in "
            "constellation.gateway_profile"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SimulationConfig":
        data = data or {}
        trace_options = dict(data.get("trace_options") or {})
        return cls(
            duration_s=float(_pick(data, "duration_s", 21600.0)),
            step_s=float(_pick(data, "step_s", 60.0)),
            epoch=str(_pick(data, "epoch", "2026-06-16T00:00:00Z")),
            use_stk_calibrated_surrogate=bool(
                _pick(
                    data,
                    "use_stk_calibrated_surrogate",
                    _pick(data, "use_stk_surrogate", False),
                )
            ),
            trace_model=str(_pick(data, "trace_model", "skeleton")),
            trace_output_dir=(
                str(data["trace_output_dir"])
                if data.get("trace_output_dir") is not None
                else None
            ),
            trace_options=trace_options,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class SimulationResult:
    evaluated: bool
    performance: PerformanceMetrics
    slo: SLOMetrics
    warnings: tuple[str, ...]
    trace_model: str = "skeleton"
    trace_evaluated: bool = False
    trace_manifest: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "performance": self.performance.to_dict(),
            "slo": self.slo.to_dict(),
            "warnings": list(self.warnings),
            "trace_model": self.trace_model,
            "trace_evaluated": self.trace_evaluated,
            "trace_manifest": to_jsonable(self.trace_manifest or {}),
        }


def run_simulation(
    config: SimulationConfig,
    *,
    constellation: Any = None,
    workload_generation: Any = None,
    base_dir: str | Path | None = None,
    experiment_id: str | None = None,
) -> SimulationResult:
    """Return placeholder metrics plus any requested runtime trace manifest."""

    if config.trace_model == "ideal_full_visibility":
        manifest = RuntimeTraceManifest.ideal_full_visibility(
            trace_id=f"ideal_full_visibility:{config.epoch}:{config.duration_s:g}s"
        )
        return SimulationResult(
            evaluated=False,
            performance=PerformanceMetrics(),
            slo=SLOMetrics(),
            trace_model=manifest.trace_model,
            trace_evaluated=True,
            trace_manifest=manifest.to_dict(),
            warnings=(
                "ideal_full_visibility trace is enabled for idealized compute-only runtime checks",
                "orbit/visibility simulation is not implemented",
                "valid_for_paper_main_results=false",
                f"received simulation window duration_s={config.duration_s} step_s={config.step_s}",
            ),
        )

    if config.trace_model == SURROGATE_ORBIT_TRACE:
        if constellation is None or workload_generation is None:
            return SimulationResult(
                evaluated=False,
                performance=PerformanceMetrics(),
                slo=SLOMetrics(),
                trace_model=SURROGATE_ORBIT_TRACE,
                trace_evaluated=False,
                trace_manifest=RuntimeTraceManifest(
                    trace_model=SURROGATE_ORBIT_TRACE,
                    warnings=(
                        "surrogate_orbit_trace requires constellation and materialized workload_generation",
                        "valid_for_paper_main_results=false",
                    ),
                ).to_dict(),
                warnings=(
                    "surrogate_orbit_trace requires constellation and materialized workload_generation",
                    "valid_for_paper_main_results=false",
                ),
            )
        try:
            from .simulation_trace_builder import build_surrogate_orbit_trace

            trace_result = build_surrogate_orbit_trace(
                config=config,
                constellation=constellation,
                workload_generation=workload_generation,
                base_dir=base_dir,
                experiment_id=experiment_id,
            )
        except Exception as exc:
            warning = f"surrogate_orbit_trace build failed: {exc}"
            return SimulationResult(
                evaluated=False,
                performance=PerformanceMetrics(),
                slo=SLOMetrics(),
                trace_model=SURROGATE_ORBIT_TRACE,
                trace_evaluated=False,
                trace_manifest=RuntimeTraceManifest(
                    trace_model=SURROGATE_ORBIT_TRACE,
                    warnings=(warning, "valid_for_paper_main_results=false"),
                ).to_dict(),
                warnings=(warning, "valid_for_paper_main_results=false"),
            )
        return SimulationResult(
            evaluated=False,
            performance=PerformanceMetrics(),
            slo=SLOMetrics(),
            trace_model=SURROGATE_ORBIT_TRACE,
            trace_evaluated=True,
            trace_manifest=trace_result.to_trace_manifest(),
            warnings=(
                *trace_result.warnings,
                "surrogate_orbit_trace evaluates deterministic runtime input traces only",
                "valid_for_paper_main_results=false",
                f"received simulation window duration_s={config.duration_s} step_s={config.step_s}",
            ),
        )

    if config.trace_model == CIRCULAR_GEOMETRY_TRACE:
        if constellation is None or workload_generation is None:
            return SimulationResult(
                evaluated=False,
                performance=PerformanceMetrics(),
                slo=SLOMetrics(),
                trace_model=CIRCULAR_GEOMETRY_TRACE,
                trace_evaluated=False,
                trace_manifest=RuntimeTraceManifest(
                    trace_model=CIRCULAR_GEOMETRY_TRACE,
                    warnings=(
                        "circular_geometry_trace requires constellation and materialized workload_generation",
                        "valid_for_paper_main_results=false",
                    ),
                ).to_dict(),
                warnings=(
                    "circular_geometry_trace requires constellation and materialized workload_generation",
                    "valid_for_paper_main_results=false",
                ),
            )
        try:
            from .orbit_traces import build_satellite_motion_trace

            trace_result = build_satellite_motion_trace(
                config=config,
                constellation=constellation,
                workload_generation=workload_generation,
                base_dir=base_dir,
                experiment_id=experiment_id,
            )
        except Exception as exc:
            warning = f"circular_geometry_trace build failed: {exc}"
            return SimulationResult(
                evaluated=False,
                performance=PerformanceMetrics(),
                slo=SLOMetrics(),
                trace_model=CIRCULAR_GEOMETRY_TRACE,
                trace_evaluated=False,
                trace_manifest=RuntimeTraceManifest(
                    trace_model=CIRCULAR_GEOMETRY_TRACE,
                    warnings=(warning, "valid_for_paper_main_results=false"),
                ).to_dict(),
                warnings=(warning, "valid_for_paper_main_results=false"),
            )
        return SimulationResult(
            evaluated=False,
            performance=PerformanceMetrics(),
            slo=SLOMetrics(),
            trace_model=CIRCULAR_GEOMETRY_TRACE,
            trace_evaluated=True,
            trace_manifest=trace_result.to_trace_manifest(),
            warnings=(
                *trace_result.warnings,
                "circular_geometry_trace evaluates deterministic geometry input traces only",
                "trace is not STK-calibrated",
                "valid_for_paper_main_results=false",
                f"received simulation window duration_s={config.duration_s} step_s={config.step_s}",
            ),
        )

    return SimulationResult(
        evaluated=False,
        performance=PerformanceMetrics(),
        slo=SLOMetrics(),
        trace_model=config.trace_model,
        trace_evaluated=False,
        trace_manifest=RuntimeTraceManifest(warnings=("simulation trace is not implemented",)).to_dict(),
        warnings=(
            "orbit/visibility simulation is not implemented",
            f"received simulation window duration_s={config.duration_s} step_s={config.step_s}",
        ),
    )
