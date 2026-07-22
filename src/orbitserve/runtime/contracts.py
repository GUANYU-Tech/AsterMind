"""Stable runtime data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..common import to_jsonable

RUNTIME_INTERFACE_DOC = "docs/research/parameters/llm_serving_runtime_interface.md"
RUNTIME_PERFORMANCE_DOC = "docs/research/parameters/llm_serving_runtime_performance_model.md"
RUNTIME_OPTIMIZATION_DOC = "docs/research/parameters/llm_serving_runtime_optimization.md"

RUNTIME_RESULT_FIELDS = frozenset(
    {
        "assigned_satellite",
        "assigned_satellite_id",
        "assigned_deployment",
        "access_satellite_id",
        "selected_path",
        "session_action",
        "completed",
        "completion_ratio",
        "uncompleted_reason",
        "TTFT",
        "E2E",
        "ttft_s",
        "ttft_p95_s",
        "observed_ttft_s",
        "e2e_latency_s",
        "e2e_p95_s",
        "observed_e2e_s",
        "energy_violation",
        "energy_violation_fraction",
        "cost_per_request",
        "cost_per_served_request_usd",
        "cost_per_served_mtoken_usd",
        "soc_before",
        "soc_after",
        "compute_sunlit",
        "access_sunlit",
        "dvfs_scale",
        "shadow_strategy",
        "sgl_queue_wait_s",
        "isl_queue_wait_s",
        "link_transmission_s",
        "energy_violation_count",
        "fallback_used",
        "deferred",
        "defer_until_s",
        "ground_origin_region_id",
        "ground_serving_region_id",
        "ground_replica_id",
        "ground_cross_region",
        "ground_network_delay_s",
        "ground_load_balancer_queue_wait_s",
        "ground_replica_pending_wait_s",
        "ground_prefix_cache_hit_ratio",
    }
)



def _default_weights() -> dict[str, Any]:
    return {
        "latency": None,
        "energy": None,
        "link": None,
        "queue": None,
        "kv": None,
        "session": None,
        "fallback": None,
    }

def _strategy_defaults(policy: str) -> dict[str, Any]:
    if policy == "ground_regional":
        return {
            "policy_family": "terrestrial_baseline",
            "enabled_actions": ("ground_local_compute", "ground_reject_drop"),
            "session_actions_enabled": ("sticky_kv_owner", "recompute_context"),
            "path_policy": "ground_region_local_only",
            "queue_policy": "regional_fcfs",
            "max_hop_count": 0,
            "fallback_enabled": False,
            "defer_enabled": False,
            "hard_gate_overrides": {},
        }
    if policy == "skywalker_like_ground":
        return {
            "policy_family": "terrestrial_baseline",
            "enabled_actions": (
                "ground_local_compute",
                "ground_cross_region_forward",
                "ground_reject_drop",
            ),
            "session_actions_enabled": ("sticky_kv_owner", "recompute_context"),
            "path_policy": "skywalker_two_layer_prefix_snapshot",
            "queue_policy": "skywalker_pending_selective_fcfs",
            "max_hop_count": 1,
            "fallback_enabled": False,
            "defer_enabled": False,
            "hard_gate_overrides": {},
        }
    if policy in {
        "ground_central_rr",
        "ground_central_ll",
        "ground_central_ch",
        "ground_central_sgl_like",
        "ground_gke_gateway_like",
    }:
        topology = (
            "regional_gke_gateway_like_global_backend"
            if policy == "ground_gke_gateway_like"
            else "single_us_global_coordinator"
        )
        return {
            "policy_family": "terrestrial_paper_comparison_baseline",
            "enabled_actions": ("ground_global_blind_push", "ground_reject_drop"),
            "session_actions_enabled": ("sticky_kv_owner", "recompute_context"),
            "path_policy": topology,
            "queue_policy": "blind_push_replica_admission_queue",
            "max_hop_count": 0,
            "fallback_enabled": False,
            "defer_enabled": False,
            "hard_gate_overrides": {},
        }
    if policy == "local_only":
        return {
            "policy_family": "baseline",
            "enabled_actions": ("local_compute", "reject_drop"),
            "session_actions_enabled": ("none", "recompute_context"),
            "path_policy": "nearest",
            "queue_policy": "fifo",
            "max_hop_count": 0,
            "fallback_enabled": False,
            "defer_enabled": False,
        }
    if policy == "shortest_path":
        return {
            "policy_family": "baseline",
            "enabled_actions": ("local_compute", "same_plane_forward", "cross_plane_forward", "reject_drop"),
            "session_actions_enabled": ("none", "recompute_context"),
            "path_policy": "shortest_path",
            "queue_policy": "fifo",
            "fallback_enabled": False,
            "defer_enabled": False,
        }
    if policy == "load_aware":
        return {
            "policy_family": "baseline",
            "enabled_actions": ("local_compute", "same_plane_forward", "cross_plane_forward", "reject_drop"),
            "session_actions_enabled": ("none", "recompute_context"),
            "path_policy": "k_shortest",
            "queue_policy": "least_queue_work",
            "fallback_enabled": False,
            "defer_enabled": False,
        }
    if policy == "energy_aware":
        return {
            "policy_family": "baseline",
            "enabled_actions": (
                "local_compute",
                "same_plane_forward",
                "cross_plane_forward",
                "shadow_offload",
                "reject_drop",
            ),
            "session_actions_enabled": ("none", "recompute_context"),
            "shadow_dvfs_enabled": True,
            "shadow_dvfs_scale": 0.5,
            "path_policy": "energy_aware",
            "queue_policy": "fifo",
            "fallback_enabled": False,
            "defer_enabled": False,
        }
    if policy == "orbitserve":
        return {
            "policy_family": "orbitserve",
            "enabled_actions": (
                "local_compute",
                "same_plane_forward",
                "cross_plane_forward",
                "shadow_dvfs",
                "shadow_offload",
                "defer",
                "ground_fallback",
                "reject_drop",
            ),
            "session_actions_enabled": (
                "sticky_kv_owner",
                "migrate_kv",
                "replicate_kv",
                "recompute_context",
                "evict_session",
            ),
            "shadow_dvfs_enabled": True,
            "shadow_dvfs_scale": 0.5,
            "path_policy": "k_shortest_score",
            "queue_policy": "resource_aware_score",
            "fallback_enabled": False,
            "defer_enabled": True,
        }
    return {
        "policy_family": "baseline",
        "enabled_actions": (),
        "session_actions_enabled": (),
        "fallback_enabled": False,
        "defer_enabled": False,
    }

@dataclass(frozen=True)
class RuntimeStrategySpec:
    name: str
    policy: str = "orbitserve"
    policy_family: str = "orbitserve"
    enabled_actions: tuple[str, ...] = ()
    session_actions_enabled: tuple[str, ...] = ()
    shadow_dvfs_enabled: bool = False
    shadow_dvfs_scale: float = 0.5
    lookahead_orbits: float | None = 1.0
    h_lookahead_s: float | None = None
    max_hop_count: int | None = None
    fallback_enabled: bool = False
    defer_enabled: bool = False
    path_policy: str | None = None
    queue_policy: str | None = None
    hard_gate_mode: str = "strict"
    hard_gate_overrides: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, Any] = field(default_factory=_default_weights)
    normalization_refs: dict[str, Any] = field(default_factory=dict)
    kv_policy: dict[str, Any] = field(default_factory=dict)
    ground_config: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    trace_level: str = "summary"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.shadow_dvfs_scale) <= 1.0:
            raise ValueError("shadow_dvfs_scale must be in (0, 1]")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RuntimeStrategySpec":
        return cls.from_value(data)

    @classmethod
    def from_value(cls, value: Any = None) -> "RuntimeStrategySpec":
        data = _mapping_from_result(value)
        if not data and value is not None:
            data = _object_public_fields(value)
        policy = _canonical_policy(str(data.get("policy", "orbitserve")))
        defaults = _strategy_defaults(policy)
        weights = _default_weights()
        weights.update(_dict(data.get("weights")))
        name = _optional_str(data.get("name")) or policy
        hard_gate_overrides = _dict(defaults.get("hard_gate_overrides"))
        hard_gate_overrides.update(_dict(data.get("hard_gate_overrides")))
        return cls(
            name=name,
            policy=policy,
            policy_family=_optional_str(data.get("policy_family")) or str(defaults["policy_family"]),
            enabled_actions=_string_tuple_or_default(
                data,
                "enabled_actions",
                tuple(defaults["enabled_actions"]),
            ),
            session_actions_enabled=_string_tuple_or_default(
                data,
                "session_actions_enabled",
                tuple(defaults["session_actions_enabled"]),
            ),
            shadow_dvfs_enabled=bool(data.get("shadow_dvfs_enabled", defaults.get("shadow_dvfs_enabled", False))),
            shadow_dvfs_scale=float(data.get("shadow_dvfs_scale", defaults.get("shadow_dvfs_scale", 0.5))),
            lookahead_orbits=_optional_float(data.get("lookahead_orbits", 1.0)),
            h_lookahead_s=_optional_float(data.get("h_lookahead_s", data.get("H_lookahead"))),
            max_hop_count=_optional_int(data.get("max_hop_count", defaults.get("max_hop_count"))),
            fallback_enabled=bool(data.get("fallback_enabled", defaults["fallback_enabled"])),
            defer_enabled=bool(data.get("defer_enabled", defaults["defer_enabled"])),
            path_policy=_optional_str(data.get("path_policy")) or _optional_str(defaults.get("path_policy")),
            queue_policy=_optional_str(data.get("queue_policy")) or _optional_str(defaults.get("queue_policy")),
            hard_gate_mode=str(data.get("hard_gate_mode", "strict")),
            hard_gate_overrides=hard_gate_overrides,
            weights=weights,
            normalization_refs=_dict(data.get("normalization_refs")),
            kv_policy=_dict(data.get("kv_policy")),
            ground_config=_dict(data.get("ground_config")),
            random_seed=_optional_int(data.get("random_seed")),
            trace_level=str(data.get("trace_level", "summary")),
            provenance={
                "source": "fixed RuntimeStrategySpec",
                "strategy_mutable_during_run": False,
                **_dict(data.get("provenance")),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class ServingDeploymentView:
    deployment_id: str
    member_satellites: tuple[str, ...] = ()
    serving_spec: dict[str, Any] = field(default_factory=dict)
    throughput_estimate: dict[str, Any] = field(default_factory=dict)
    evaluated: bool = False
    feasible: bool | None = None
    request_throughput_per_s: float | None = None
    prefill_tokens_per_s: float | None = None
    decode_tokens_per_s: float | None = None
    kv_cache_capacity_tokens: int | None = None
    kv_cache_bytes_per_token: float | None = None
    satellites_per_deployment: int | None = None
    stage_satellite_count: int | None = None
    internal_comm_summary: dict[str, Any] = field(default_factory=dict)
    internal_comm_accounting: str = "embedded"
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: Any, *, deployment_id: str = "deployment_0") -> "ServingDeploymentView":
        payload = _mapping_from_result(value)
        details = _dict(payload.get("details"))
        if details:
            throughput_payload = {**details, **{key: item for key, item in payload.items() if item is not None}}
        else:
            throughput_payload = payload
        return cls.from_inputs(deployment_id=deployment_id, throughput_estimate=throughput_payload)

    @classmethod
    def from_inputs(
        cls,
        *,
        deployment_id: str,
        serving_spec: Any = None,
        throughput_estimate: Any = None,
        internal_comm_accounting: str = "embedded",
    ) -> "ServingDeploymentView":
        serving_payload = _mapping_from_result(serving_spec)
        throughput_payload = _mapping_from_result(throughput_estimate)
        internal_comm_summary = {
            key: throughput_payload.get(key)
            for key in (
                "effective_isl_bandwidth_gbps",
                "isl_required_flows",
                "available_isl_laser_heads",
                "isl_head_occupancy",
                "prefill_interconnect_time_s",
                "decode_interconnect_time_s",
                "tp_communication_time_s",
                "ep_communication_time_s",
                "pp_communication_time_s",
                "communication_bottleneck",
                "bottlenecks",
                "warnings",
                "provenance",
            )
            if key in throughput_payload
        }
        return cls(
            deployment_id=deployment_id,
            member_satellites=tuple(
                str(item)
                for item in _first_present(
                    throughput_payload.get("member_satellites"),
                    serving_payload.get("member_satellites"),
                    (),
                )
                or ()
            ),
            serving_spec=serving_payload,
            throughput_estimate=throughput_payload,
            evaluated=bool(throughput_payload.get("evaluated", False)),
            feasible=_optional_bool(throughput_payload.get("feasible")),
            request_throughput_per_s=_optional_float(throughput_payload.get("request_throughput_per_s")),
            prefill_tokens_per_s=_optional_float(throughput_payload.get("prefill_tokens_per_s")),
            decode_tokens_per_s=_optional_float(throughput_payload.get("decode_tokens_per_s")),
            kv_cache_capacity_tokens=_optional_int(throughput_payload.get("kv_cache_capacity_tokens")),
            kv_cache_bytes_per_token=_optional_float(throughput_payload.get("kv_cache_bytes_per_token")),
            satellites_per_deployment=_optional_int(throughput_payload.get("satellites_per_deployment")),
            stage_satellite_count=_optional_int(throughput_payload.get("stage_satellite_count")),
            internal_comm_summary=internal_comm_summary,
            internal_comm_accounting=internal_comm_accounting,
            warnings=_warnings_from_payload(throughput_payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class RuntimePredictionEnvelope:
    evaluated: bool = False
    predicted_ttft_s: float | None = None
    predicted_e2e_s: float | None = None
    access_wait_s: float | None = None
    prefill_queue_wait_s: float | None = None
    decode_queue_wait_s: float | None = None
    link_queue_wait_s: float | None = None
    prefill_service_s: float | None = None
    decode_service_s: float | None = None
    kv_migration_visible_s: float | None = None
    context_recompute_s: float | None = None
    path_setup_delay_s: float | None = None
    average_hop_count: float | None = None
    delta_compute_j: float | None = None
    delta_sgl_j: float | None = None
    delta_isl_j: float | None = None
    delta_energy_j: float | None = None
    soc_min_predicted: float | None = None
    energy_feasible: bool | None = None
    bottleneck_reason: str | None = None
    warnings: tuple[str, ...] = ("runtime prediction is not implemented",)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class RuntimeAction:
    action_id: str
    request_id: str | None = None
    action_type: str = "reject_drop"
    access_node: str | None = None
    serving_deployment: str | None = None
    route_in: Any | None = None
    route_out: Any | None = None
    kv_route: Any | None = None
    session_action: str = "none"
    energy_action: str = "none"
    dvfs_profile: str | None = None
    defer_until_s: float | None = None
    fallback_mode: str = "none"
    expected_resource_delta: dict[str, Any] = field(default_factory=dict)
    hard_gate_results: dict[str, Any] = field(default_factory=dict)
    prediction: RuntimePredictionEnvelope = field(default_factory=RuntimePredictionEnvelope)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class RuntimeDecisionTrace:
    request_id: str | None = None
    arrival_time_s: float | None = None
    decision_time_s: float | None = None
    policy_name: str | None = None
    state_view_hash: str | None = None
    candidate_action_count: int = 0
    feasible_action_count: int = 0
    selected_action: RuntimeAction | dict[str, Any] | None = None
    rejected_actions: tuple[dict[str, Any], ...] = ()
    gate_results: dict[str, Any] = field(default_factory=dict)
    prediction: RuntimePredictionEnvelope = field(default_factory=RuntimePredictionEnvelope)
    score_components: dict[str, Any] = field(default_factory=dict)
    session_decision: dict[str, Any] = field(default_factory=dict)
    route_summary: dict[str, Any] = field(default_factory=dict)
    bottleneck_reason: str | None = "not_implemented"
    warnings: tuple[str, ...] = ("runtime decision trace is not implemented",)
    trace_schema_version: str = "orbitserve.runtime_decision_trace.v1"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class RuntimeScenario:
    scenario_id: str = "runtime_scenario"
    design_id: str = "runtime_design"
    time_horizon_s: float | None = None
    workload_bundle: dict[str, Any] = field(default_factory=dict)
    requests: tuple[dict[str, Any], ...] = ()
    sessions: tuple[dict[str, Any], ...] = ()
    regions: tuple[dict[str, Any], ...] = ()
    gateways: tuple[dict[str, Any], ...] = ()
    simulation_trace_ref: dict[str, Any] = field(default_factory=dict)
    constellation: dict[str, Any] = field(default_factory=dict)
    serving_deployments: tuple[ServingDeploymentView, ...] = ()
    cost_power: dict[str, Any] = field(default_factory=dict)
    runtime_config: RuntimeStrategySpec = field(default_factory=lambda: RuntimeStrategySpec.from_value(None))
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: str = "orbitserve.runtime_scenario.v1"
    time_mode: str = "event_driven"
    time_step_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class RuntimePerformanceResult:
    evaluated: bool
    scenario: dict[str, Any]
    strategy: dict[str, Any]
    workload: dict[str, Any] = field(default_factory=dict)
    constellation: dict[str, Any] = field(default_factory=dict)
    serving: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    slo: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    runtime_metrics: dict[str, Any] = field(default_factory=dict)
    bottleneck_breakdown: dict[str, Any] = field(default_factory=dict)
    session_metrics: dict[str, Any] = field(default_factory=dict)
    communication_metrics: dict[str, Any] = field(default_factory=dict)
    energy_metrics: dict[str, Any] = field(default_factory=dict)
    region_metrics: tuple[dict[str, Any], ...] = ()
    prediction_error: dict[str, Any] = field(default_factory=dict)
    non_separability: dict[str, Any] = field(default_factory=dict)
    decision_trace: dict[str, Any] = field(default_factory=dict)
    request_records_path: str | None = None
    request_records: tuple[dict[str, Any], ...] = ()
    trace_model: str | None = None
    scope: str | None = None
    valid_for_paper_main_results: bool = False
    module_status: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    module: str = "llm_serving_runtime_performance"
    schema_version: str = "orbitserve.runtime_performance.v1"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

def _mapping_from_result(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return to_jsonable(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = to_jsonable(value.to_dict())
        return converted if isinstance(converted, dict) else {}
    converted = to_jsonable(value)
    return converted if isinstance(converted, dict) else {}

def _object_public_fields(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            item = getattr(value, key)
        except Exception:  # pragma: no cover - defensive adapter.
            continue
        if callable(item):
            continue
        result[key] = item
    return to_jsonable(result)

def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}

def _warnings_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = payload.get("warnings", ())
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw)

def _canonical_policy(value: str) -> str:
    normalized = value.strip().lower()
    return normalized

def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)

def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)

def _string_tuple_or_default(data: dict[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if key not in data:
        return default
    return _string_tuple(data.get(key))

def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
