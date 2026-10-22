"""Terrestrial baseline contracts, independent of the satellite scheduler."""

from dataclasses import dataclass, field, asdict, replace
from typing import Any
from orbitserve.common import to_jsonable


@dataclass(frozen=True)
class GroundRequestRecord:
    request_id: str
    accepted: Any = False
    action_type: Any = None
    arrival_time_s: Any = 0.0
    bottleneck_reason: Any = None
    communication_energy_j: Any = None
    completed: Any = False
    compute_energy_j: Any = None
    context_recompute_tokens: Any = None
    decode_queue_wait_s: Any = None
    decode_service_s: Any = None
    deployment_id: Any = None
    e2e_deadline_s: Any = None
    e2e_s: Any = None
    energy_violation: Any = False
    fallback_used: Any = False
    ground_cross_region: Any = False
    ground_load_balancer_queue_wait_s: Any = None
    ground_network_delay_s: Any = None
    ground_origin_region_id: Any = None
    ground_prefix_cache_hit_ratio: Any = None
    ground_replica_id: Any = None
    ground_replica_pending_wait_s: Any = None
    ground_serving_region_id: Any = None
    kv_hit_tokens: Any = None
    kv_miss_tokens: Any = None
    link_queue_wait_s: Any = None
    link_transmission_s: Any = None
    output_tokens: Any = 0
    prefill_queue_wait_s: Any = None
    prefill_service_s: Any = None
    prompt_tokens: Any = 0
    provenance: Any = field(default_factory=dict)
    region_id: Any = None
    reject_reason: Any = None
    rejected: Any = False
    scope: Any = None
    selected_path: Any = ()
    served: Any = False
    session_action: Any = None
    session_id: Any = None
    soc_after: Any = None
    soc_before: Any = None
    timeout: Any = False
    timeout_reason: Any = None
    tpot_s: Any = None
    trace_model: Any = None
    ttft_deadline_s: Any = None
    ttft_s: Any = None
    uncompleted_reason: Any = None
    valid_for_paper_main_results: Any = None

    def to_dict(self):
        return to_jsonable(asdict(self))

    def with_updates(self, **values):
        return replace(self, **values)


@dataclass(frozen=True)
class GroundStrategySpec:
    policy: str = "skywalker_like_ground"
    ground_config: dict = field(default_factory=dict)
    fallback_enabled: bool = False
    enabled_actions: tuple = (
        "ground_local_compute",
        "ground_cross_region_forward",
        "ground_reject_drop",
    )
    path_policy: str = "skywalker_two_layer_prefix_snapshot"
    queue_policy: str = "skywalker_pending_selective_fcfs"
    max_hop_count: int = 1
    hard_gate_overrides: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data=None):
        return cls(**(data or {}))

    def to_dict(self):
        return to_jsonable(asdict(self))

    @property
    def spec(self):
        return self


@dataclass(frozen=True)
class GroundScenario:
    scenario_id: str = "ground"
    requests: tuple = ()
    regions: tuple = ()
    gateways: tuple = ()
    runtime_config: Any = field(default_factory=GroundStrategySpec)
    time_horizon_s: float | None = None
    cost_power: dict = field(default_factory=dict)
    simulation_trace_ref: dict = field(default_factory=dict)
    serving_profile: Any = None
    constellation: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def get_ground_scheduler(spec):
    return (
        spec
        if isinstance(spec, GroundStrategySpec)
        else GroundStrategySpec.from_dict(spec)
    )
