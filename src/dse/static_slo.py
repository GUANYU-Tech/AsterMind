"""Geometry-only screening; end-user latency and energy require Runtime."""

from dataclasses import dataclass, field
from orbitserve.common import to_jsonable
from orbitserve.orbit import OrbitTraceBundle, OrbitTraceManifest, load_orbit_trace


@dataclass(frozen=True)
class StaticSLOProxyResult:
    evaluated: bool = False
    geometric_access_fraction: float | None = None
    request_count: int = 0
    runtime_metrics: dict = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)

    def to_dict(self):
        return to_jsonable(self)


def evaluate_static_slo_proxy(
    *,
    simulation_result,
    workload_generation,
    serving_throughput=None,
    base_dir=None,
    **unused,
):
    try:
        if isinstance(simulation_result, OrbitTraceBundle):
            trace = simulation_result
        elif isinstance(simulation_result, OrbitTraceManifest):
            trace = load_orbit_trace(simulation_result, base_dir=base_dir)
        else:
            raise ValueError(
                "geometry screening requires an OrbitTraceBundle or OrbitTraceManifest"
            )
        if not trace.trace_manifest.evaluated:
            raise ValueError("orbit trace is not evaluated")
        workload = (
            workload_generation.to_dict()
            if hasattr(workload_generation, "to_dict")
            else dict(workload_generation or {})
        )
        requests = workload.get("session_requests", workload.get("requests", ()))
        if not requests:
            raise ValueError("geometry screening requires materialized requests")
        covered = sum(
            any(
                str(w["ingress_id"]) == str(r["ingress_id"])
                and float(w["start_s"])
                <= float(r["arrival_time_s"])
                < float(w["end_s"])
                for w in trace.sgl_windows
            )
            for r in requests
        )
        return StaticSLOProxyResult(
            True,
            covered / len(requests),
            len(requests),
            {},
            (
                "Geometric candidate coverage is an upper bound; run Runtime for terminal allocation, latency, queues and energy.",
            ),
            {
                "scope": "geometric_access_upper_bound",
                "runtime_metrics_inferred": False,
            },
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        return StaticSLOProxyResult(
            warnings=(str(error),),
            provenance={
                "scope": "geometry_unavailable",
                "runtime_metrics_inferred": False,
            },
        )
