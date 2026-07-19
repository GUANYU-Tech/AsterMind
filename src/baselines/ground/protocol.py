"""Paper-shaped SkyWalker terrestrial comparison matrix.

This is a harness, not a workload generator.  It accepts a materialized
request corpus (for example an adapted Chatbot Arena, WildChat, or ToT trace)
and runs every policy against the same request/program corpus, calibrated
profiles, client harness, replica allocation, and measurement window.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Iterable, Mapping, Sequence

from ...common import to_jsonable
from .costing import estimate_terrestrial_ground_cost
from .global_runtime import (
    is_centralized_ground_policy,
    run_centralized_ground_runtime,
)
from .runtime import run_ground_runtime
from ...runtime import RuntimeStrategySpec


SKYWALKER_PAPER_POLICY_IDS = (
    "central_rr",
    "central_ll",
    "central_ch",
    "central_sgl_like",
    "gke_gateway_like",
    "skywalker_ch",
    "skywalker_prefix",
    "region_local",
)


def even_replica_allocations(
    region_ids: Sequence[str], totals: Iterable[int] = (3, 6, 9, 12, 15, 18)
) -> tuple[dict[str, int], ...]:
    """Return the paper's evenly distributed 3-region capacity sweep shape."""

    normalized_regions = tuple(str(region_id) for region_id in region_ids)
    if not normalized_regions:
        raise ValueError("replica allocation needs at least one region")
    result: list[dict[str, int]] = []
    for raw_total in totals:
        total = int(raw_total)
        if total <= 0 or total % len(normalized_regions) != 0:
            raise ValueError(
                "even replica totals must be positive multiples of the region count"
            )
        per_region = total // len(normalized_regions)
        result.append({region_id: per_region for region_id in normalized_regions})
    return tuple(result)


def build_skywalker_paper_strategy(
    policy_id: str,
    base_ground_config: Mapping[str, Any],
    replica_allocation: Mapping[str, int],
) -> RuntimeStrategySpec:
    """Build a first-class topology+dispatcher baseline from one policy id."""

    policy = str(policy_id)
    if policy not in SKYWALKER_PAPER_POLICY_IDS:
        raise ValueError(f"unknown SkyWalker paper comparison policy: {policy!r}")
    config = deepcopy(dict(base_ground_config))
    regions = _apply_replica_allocation(config.get("regions"), replica_allocation)
    config["regions"] = regions
    if policy == "skywalker_prefix":
        runtime_policy = "skywalker_like_ground"
        config.update(
            {"routing_policy": "prefix_tree", "selective_pushing": "pending_requests"}
        )
    elif policy == "skywalker_ch":
        runtime_policy = "skywalker_like_ground"
        config.update(
            {"routing_policy": "consistent_hash", "selective_pushing": "pending_requests"}
        )
    elif policy == "region_local":
        runtime_policy = "ground_regional"
        config.update(
            {"routing_policy": "prefix_tree", "selective_pushing": "pending_requests"}
        )
    else:
        runtime_policy = {
            "central_rr": "ground_central_rr",
            "central_ll": "ground_central_ll",
            "central_ch": "ground_central_ch",
            "central_sgl_like": "ground_central_sgl_like",
            "gke_gateway_like": "ground_gke_gateway_like",
        }[policy]
    return RuntimeStrategySpec.from_dict(
        {"policy": runtime_policy, "ground_config": config}
    )


def run_skywalker_paper_protocol(
    *,
    requests: Sequence[Mapping[str, Any]],
    base_ground_config: Mapping[str, Any],
    policy_ids: Sequence[str] = SKYWALKER_PAPER_POLICY_IDS,
    replica_allocations: Sequence[Mapping[str, int]] | None = None,
    warmup_s: float = 0.0,
    measurement_duration_s: float | None = None,
) -> dict[str, Any]:
    """Run the same materialized corpus through paper-comparison policies.

    ``warmup_s`` and ``measurement_duration_s`` filter *completed* requests
    by client-visible terminal time.  The raw full-run metrics remain nested
    in every row, preventing an accidental comparison of closed-loop and
    open-loop results with an ambiguous denominator.
    """

    if warmup_s < 0:
        raise ValueError("warmup_s must be non-negative")
    if measurement_duration_s is not None and measurement_duration_s <= 0:
        raise ValueError("measurement_duration_s must be positive when supplied")
    materialized_requests = tuple(dict(item) for item in requests if isinstance(item, Mapping))
    if not materialized_requests:
        raise ValueError("SkyWalker protocol needs at least one materialized request")
    config = dict(base_ground_config)
    region_ids = tuple(
        str(item.get("region_id"))
        for item in config.get("regions", ())
        if isinstance(item, Mapping) and item.get("region_id") is not None
    )
    if not region_ids:
        raise ValueError("base_ground_config.regions is required for the protocol")
    allocations = tuple(replica_allocations or even_replica_allocations(region_ids))
    rows: list[dict[str, Any]] = []
    for allocation in allocations:
        for policy_id in policy_ids:
            strategy = build_skywalker_paper_strategy(
                policy_id, config, allocation
            )
            runner = (
                run_centralized_ground_runtime
                if is_centralized_ground_policy(strategy.policy)
                else run_ground_runtime
            )
            result = runner(
                requests=materialized_requests,
                runtime_strategy=strategy,
                policy=strategy.policy,
            )
            window = _measurement_window(
                result.records,
                warmup_s=warmup_s,
                duration_s=measurement_duration_s,
            )
            cost = estimate_terrestrial_ground_cost(
                records=result.records,
                ground_config=strategy.ground_config,
                runtime_observation_window_s=window["measurement_duration_s"],
            )
            rows.append(
                {
                    "policy_id": policy_id,
                    "runtime_policy": strategy.policy,
                    "replica_allocation": dict(allocation),
                    "replica_count": sum(int(value) for value in allocation.values()),
                    "measurement": window,
                    "runtime_metrics": result.runtime_metrics,
                    "communication_metrics": result.communication_metrics,
                    "terrestrial_costing": cost.to_dict(),
                    "warnings": tuple(dict.fromkeys((*result.warnings, *cost.warnings))),
                    "provenance": result.provenance,
                    "valid_for_paper_main_results": False,
                }
            )
    return {
        "schema_version": "orbitserve.skywalker_paper_protocol.v1",
        "policy_ids": tuple(policy_ids),
        "replica_allocations": to_jsonable(allocations),
        "measurement_contract": {
            "warmup_s": warmup_s,
            "measurement_duration_s": measurement_duration_s,
            "completion_inclusion": "client_visible_terminal_time_in_window",
            "comparability": "all rows consume the same materialized request/program corpus and configured profiles",
        },
        "rows": rows,
        "warnings": (
            "This protocol is a paper-shaped comparison harness. It becomes a result reproduction only after the original workload preprocessing, client/program dependencies, L4/SGLang service measurements, network measurements, and price evidence are supplied.",
            "valid_for_paper_main_results=false",
        ),
        "valid_for_paper_main_results": False,
    }


def _apply_replica_allocation(
    regions: Any, allocation: Mapping[str, int]
) -> list[dict[str, Any]]:
    if not isinstance(regions, (tuple, list)):
        raise ValueError("base_ground_config.regions must be a list")
    result: list[dict[str, Any]] = []
    supplied = {str(key) for key in allocation}
    configured = set()
    for raw_region in regions:
        if not isinstance(raw_region, Mapping):
            raise ValueError("ground regions must be mappings")
        region = dict(raw_region)
        region_id = str(region.get("region_id") or "")
        if not region_id:
            raise ValueError("ground regions require region_id")
        configured.add(region_id)
        count = allocation.get(region_id)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"replica allocation for {region_id!r} must be a positive integer")
        region["replica_count"] = count
        result.append(region)
    if supplied != configured:
        raise ValueError(
            "replica allocation region ids must exactly match base_ground_config.regions"
        )
    return result


def _measurement_window(
    records: Sequence[Any], *, warmup_s: float, duration_s: float | None
) -> dict[str, Any]:
    arrivals = [float(getattr(record, "arrival_time_s", 0.0)) for record in records]
    start_s = min(arrivals, default=0.0) + warmup_s
    terminals = [
        float(record.arrival_time_s + record.e2e_s)
        for record in records
        if bool(getattr(record, "completed", False)) and getattr(record, "e2e_s", None) is not None
    ]
    end_s = start_s + duration_s if duration_s is not None else max(terminals, default=start_s)
    effective_duration_s = max(1e-9, end_s - start_s)
    included = [
        record
        for record in records
        if bool(getattr(record, "completed", False))
        and getattr(record, "e2e_s", None) is not None
        and start_s <= record.arrival_time_s + record.e2e_s <= end_s + 1e-12
    ]
    completed_tokens = sum(
        int(getattr(record, "prompt_tokens", 0)) + int(getattr(record, "output_tokens", 0))
        for record in included
    )
    return {
        "start_s": start_s,
        "end_s": end_s,
        "measurement_duration_s": effective_duration_s,
        "completed_request_count": len(included),
        "completed_token_count": completed_tokens,
        "completed_request_throughput_per_s": len(included) / effective_duration_s,
        "completed_token_throughput_per_s": completed_tokens / effective_duration_s,
        "ttft_p50_s": _percentile([getattr(record, "ttft_s", None) for record in included], 0.5),
        "ttft_p90_s": _percentile([getattr(record, "ttft_s", None) for record in included], 0.9),
        "e2e_p50_s": _percentile([getattr(record, "e2e_s", None) for record in included], 0.5),
        "e2e_p90_s": _percentile([getattr(record, "e2e_s", None) for record in included], 0.9),
        "cache_hit_rate_tokens": _cache_hit_rate(included),
    }


def _cache_hit_rate(records: Sequence[Any]) -> float | None:
    hit = sum(int(getattr(record, "kv_hit_tokens", 0) or 0) for record in records)
    miss = sum(int(getattr(record, "kv_miss_tokens", 0) or 0) for record in records)
    return hit / (hit + miss) if hit + miss else None


def _percentile(values: Sequence[Any], quantile: float) -> float | None:
    data = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return data[low]
    fraction = position - low
    return data[low] * (1.0 - fraction) + data[high] * fraction


__all__ = [
    "SKYWALKER_PAPER_POLICY_IDS",
    "build_skywalker_paper_strategy",
    "even_replica_allocations",
    "run_skywalker_paper_protocol",
]
