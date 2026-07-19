"""Isolated cost accounting for terrestrial multi-region serving baselines.

This module deliberately does *not* use :mod:`orbitserve.costing` or any
single-satellite cost input.  A terrestrial result is only priced when an
explicit ``ground_config.terrestrial_costing`` contract supplies the regional
replica prices and the service period that those prices cover.  That keeps a
SkyWalker-like cloud baseline comparable without accidentally inheriting LEO
capital, launch, or satellite-operation costs.

The contract is intentionally small and auditable::

    {
      "service_period_s": 3600,
      "region_replica_hourly_price_usd": {"us_east": 0.71, "eu": 0.79},
      "region_egress_cost_per_gb_usd": {"us_east": 0.09, "eu": 0.09},
      "prompt_bytes_per_token": 4,
      "output_bytes_per_token": 4,
      "annualization_mode": "repeat_service_period"
    }

``region_egress_cost_per_gb_usd`` and the token-to-byte conversion are
optional.  If omitted, the result remains a valid compute-instance cost but
explicitly states that network egress is excluded.  If egress pricing is
partially supplied, the whole price result fails closed rather than silently
underpricing a region.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from ...common import to_jsonable


SECONDS_PER_HOUR = 3600.0
HOURS_PER_YEAR = 24.0 * 365.0
BYTES_PER_GB = 1_000_000_000.0
TERRESTRIAL_COSTING_SCHEMA = "orbitserve.terrestrial_ground_costing.v1"


@dataclass(frozen=True)
class TerrestrialGroundCostResult:
    """A priced service period, or an explicit fail-closed null result."""

    evaluated: bool
    total_cost_usd: float | None
    annualized_cost_usd_per_year: float | None
    compute_cost_usd: float | None
    egress_cost_usd: float | None
    total_operational_cost_usd: float | None
    cost_coverage: str
    excluded_cost_components: tuple[str, ...]
    service_period_s: float | None
    workload_observation_period_s: float | None
    workload_scale_to_service_period: float | None
    runtime_observation_window_s: float | None
    annualization_multiplier: float | None
    provisioned_instance_hours: float | None
    service_work_instance_hours: float | None
    service_work_utilization_proxy: float | None
    egress_total_gb: float | None
    completed_request_count: int
    served_request_count: int
    completed_token_count: int
    served_token_count: int
    cost_per_completed_request_usd: float | None
    cost_per_served_request_usd: float | None
    cost_per_completed_token_usd: float | None
    cost_per_served_token_usd: float | None
    cost_per_served_mtoken_usd: float | None
    region_costs: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]
    schema_version: str = TERRESTRIAL_COSTING_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def runtime_metrics(self) -> dict[str, Any]:
        """Return the small, top-level metric projection consumed by runtime/DSE."""

        return {
            "terrestrial_total_cost_usd": self.total_cost_usd,
            "terrestrial_annualized_cost_usd_per_year": self.annualized_cost_usd_per_year,
            "terrestrial_compute_cost_usd": self.compute_cost_usd,
            "terrestrial_egress_cost_usd": self.egress_cost_usd,
            "terrestrial_total_operational_cost_usd": self.total_operational_cost_usd,
            "terrestrial_cost_coverage": self.cost_coverage,
            "terrestrial_excluded_cost_components": self.excluded_cost_components,
            "terrestrial_workload_observation_period_s": self.workload_observation_period_s,
            "terrestrial_workload_scale_to_service_period": self.workload_scale_to_service_period,
            "terrestrial_provisioned_instance_hours": self.provisioned_instance_hours,
            "terrestrial_service_work_instance_hours": self.service_work_instance_hours,
            "terrestrial_service_work_utilization_proxy": self.service_work_utilization_proxy,
            "terrestrial_egress_total_gb": self.egress_total_gb,
            "cost_per_completed_request_usd": self.cost_per_completed_request_usd,
            "cost_per_request_usd": self.cost_per_completed_request_usd,
            "cost_per_served_request_usd": self.cost_per_served_request_usd,
            "cost_per_completed_token_usd": self.cost_per_completed_token_usd,
            "cost_per_token_usd": self.cost_per_completed_token_usd,
            "cost_per_served_token_usd": self.cost_per_served_token_usd,
            "cost_per_served_mtoken_usd": self.cost_per_served_mtoken_usd,
            "terrestrial_costing_evaluated": self.evaluated,
        }


@dataclass(frozen=True)
class _RegionCapacity:
    region_id: str
    replica_count: int


def estimate_terrestrial_ground_cost(
    *,
    records: Iterable[Any],
    ground_config: Mapping[str, Any] | None,
    runtime_observation_window_s: float | None = None,
) -> TerrestrialGroundCostResult:
    """Price a ground-runtime result from explicit cloud-price inputs.

    ``records`` may be ``RuntimeRequestRecord`` instances or their serialized
    dictionaries.  Compute instances are charged for every configured replica
    during the configured service period--they are not discounted by observed
    utilization.  The aggregate service-work ratio is reported only as an
    audit signal and can exceed one under batched overlap.
    """

    record_list = tuple(records)
    config = _mapping(ground_config)
    costing = config.get("terrestrial_costing")
    capacities = _configured_regions(config)
    counts = _record_counts(record_list)
    base_kwargs = {
        **counts,
        "region_costs": (),
        "schema_version": TERRESTRIAL_COSTING_SCHEMA,
    }

    if costing is None:
        return _null_result(
            warnings=(
                "terrestrial costing is not configured: ground_config.terrestrial_costing is required; "
                "LEO cost is intentionally not reused",
            ),
            provenance={
                "model": "terrestrial_cloud_instance_and_optional_egress",
                "costing_source": "not_configured",
                "leo_cost_reused": False,
            },
            **base_kwargs,
        )
    if not isinstance(costing, Mapping):
        return _null_result(
            warnings=("ground_config.terrestrial_costing must be an object",),
            provenance={
                "model": "terrestrial_cloud_instance_and_optional_egress",
                "costing_source": "invalid",
                "leo_cost_reused": False,
            },
            **base_kwargs,
        )
    if not capacities:
        return _null_result(
            warnings=(
                "terrestrial costing cannot resolve any configured ground regions/replica counts",
            ),
            provenance={
                "model": "terrestrial_cloud_instance_and_optional_egress",
                "costing_source": "invalid_ground_regions",
                "leo_cost_reused": False,
            },
            **base_kwargs,
        )

    errors: list[str] = _validate_ground_records(
        record_list,
        configured_region_ids={item.region_id for item in capacities},
    )
    warnings: list[str] = []
    service_period_s = _positive_float(costing.get("service_period_s"))
    if service_period_s is None:
        errors.append("terrestrial_costing.service_period_s must be a positive number")
    workload_observation_period_s = _positive_float(
        costing.get("workload_observation_period_s", service_period_s)
    )
    if workload_observation_period_s is None:
        errors.append(
            "terrestrial_costing.workload_observation_period_s must be positive when supplied"
        )
    observed_runtime_window_s = _positive_float(runtime_observation_window_s)
    if (
        observed_runtime_window_s is not None
        and workload_observation_period_s is not None
        and not _approximately_equal(observed_runtime_window_s, workload_observation_period_s)
    ):
        errors.append(
            "runtime observation window differs from terrestrial_costing.workload_observation_period_s; "
            "supply the logical workload observation period that the records represent"
        )
    price_map = _nonnegative_number_map(costing.get("region_replica_hourly_price_usd"))
    if price_map is None:
        errors.append(
            "terrestrial_costing.region_replica_hourly_price_usd must map every configured region to a non-negative hourly price"
        )
        price_map = {}
    missing_price_regions = sorted(region.region_id for region in capacities if region.region_id not in price_map)
    if missing_price_regions:
        errors.append(
            "terrestrial_costing.region_replica_hourly_price_usd is missing configured region(s): "
            + ", ".join(missing_price_regions)
        )

    mode = str(costing.get("annualization_mode", "repeat_service_period"))
    annualization_multiplier: float | None = None
    if mode == "repeat_service_period":
        hours_per_year = _positive_float(costing.get("hours_per_year", HOURS_PER_YEAR))
        if hours_per_year is None:
            errors.append("terrestrial_costing.hours_per_year must be positive when annualization_mode=repeat_service_period")
        elif service_period_s is not None:
            annualization_multiplier = hours_per_year / (service_period_s / SECONDS_PER_HOUR)
            warnings.append(
                "annualized terrestrial cost assumes the configured service period repeats uniformly through the year"
            )
    elif mode == "explicit_multiplier":
        annualization_multiplier = _positive_float(costing.get("annualization_multiplier"))
        if annualization_multiplier is None:
            errors.append(
                "terrestrial_costing.annualization_multiplier must be positive when annualization_mode=explicit_multiplier"
            )
    else:
        errors.append(
            "terrestrial_costing.annualization_mode must be 'repeat_service_period' or 'explicit_multiplier'"
        )

    egress_price_raw = costing.get("region_egress_cost_per_gb_usd")
    egress_price_map: dict[str, float] | None = None
    if egress_price_raw is None:
        warnings.append("terrestrial egress pricing is not configured; total cost excludes network egress")
    else:
        egress_price_map = _nonnegative_number_map(egress_price_raw)
        if egress_price_map is None:
            errors.append("terrestrial_costing.region_egress_cost_per_gb_usd must map regions to non-negative prices")
            egress_price_map = {}
        missing_egress_regions = sorted(
            region.region_id for region in capacities if region.region_id not in egress_price_map
        )
        if missing_egress_regions:
            errors.append(
                "terrestrial_costing.region_egress_cost_per_gb_usd is missing configured region(s): "
                + ", ".join(missing_egress_regions)
            )

    egress_records = _records_requiring_egress_validation(record_list)
    prompt_bytes_per_token = _nonnegative_float(costing.get("prompt_bytes_per_token"))
    output_bytes_per_token = _nonnegative_float(costing.get("output_bytes_per_token"))
    if egress_price_map is not None and egress_records:
        if prompt_bytes_per_token is None:
            errors.append(
                "terrestrial_costing.prompt_bytes_per_token is required when egress pricing and cross-region records are present"
            )
        if output_bytes_per_token is None:
            errors.append(
                "terrestrial_costing.output_bytes_per_token is required when egress pricing and cross-region records are present"
            )
        errors.extend(
            _validate_egress_paths(
                egress_records,
                configured_region_ids={item.region_id for item in capacities},
            )
        )
    elif egress_price_map is not None:
        # There is no billed transfer in this period, so token byte assumptions
        # are not needed.  Keep the zero egress line visible in the result.
        prompt_bytes_per_token = prompt_bytes_per_token if prompt_bytes_per_token is not None else 0.0
        output_bytes_per_token = output_bytes_per_token if output_bytes_per_token is not None else 0.0

    if errors:
        return _null_result(
            service_period_s=service_period_s,
            workload_observation_period_s=workload_observation_period_s,
            runtime_observation_window_s=observed_runtime_window_s,
            annualization_multiplier=annualization_multiplier,
            warnings=tuple((*errors, *warnings, "ground cost fields remain null because terrestrial pricing is incomplete")),
            provenance={
                "model": "terrestrial_cloud_instance_and_optional_egress",
                "costing_source": "incomplete",
                "leo_cost_reused": False,
                "annualization_mode": mode,
            },
            **base_kwargs,
        )

    assert service_period_s is not None
    assert workload_observation_period_s is not None
    assert annualization_multiplier is not None
    workload_scale_to_service_period = service_period_s / workload_observation_period_s
    period_hours = service_period_s / SECONDS_PER_HOUR
    region_rows: dict[str, dict[str, Any]] = {
        item.region_id: {
            "region_id": item.region_id,
            "replica_count": item.replica_count,
            "replica_hourly_price_usd": price_map[item.region_id],
            "provisioned_instance_hours": item.replica_count * period_hours,
            "compute_cost_usd": item.replica_count * period_hours * price_map[item.region_id],
            "service_work_seconds": 0.0,
            "prompt_egress_bytes": 0.0,
            "output_egress_bytes": 0.0,
            "egress_cost_usd": 0.0,
        }
        for item in capacities
    }
    egress_events: list[tuple[str, str, Any]] = []
    for record in record_list:
        serving_region = _optional_string(_value(record, "ground_serving_region_id"))
        service_s = _nonnegative_float(_value(record, "prefill_service_s")) or 0.0
        service_s += _nonnegative_float(_value(record, "decode_service_s")) or 0.0
        if serving_region in region_rows:
            region_rows[serving_region]["service_work_seconds"] += service_s
        if egress_price_map is not None:
            egress_events.extend(_egress_events_from_selected_path(record))

    for source_region, payload_kind, record in egress_events:
        if payload_kind == "prompt":
            region_rows[source_region]["prompt_egress_bytes"] += (
                (_nonnegative_float(_value(record, "prompt_tokens")) or 0.0)
                * float(prompt_bytes_per_token or 0.0)
            )
        else:
            region_rows[source_region]["output_egress_bytes"] += (
                (_nonnegative_float(_value(record, "output_tokens")) or 0.0)
                * float(output_bytes_per_token or 0.0)
            )

    compute_cost = 0.0
    egress_cost = 0.0
    provisioned_instance_hours = 0.0
    service_work_seconds = 0.0
    egress_total_bytes = 0.0
    for region_id, row in region_rows.items():
        if egress_price_map is not None:
            egress_bytes = row["prompt_egress_bytes"] + row["output_egress_bytes"]
            row["egress_cost_usd"] = egress_bytes / BYTES_PER_GB * egress_price_map[region_id]
            row["egress_cost_per_gb_usd"] = egress_price_map[region_id]
        else:
            row["egress_cost_per_gb_usd"] = None
        row["service_work_instance_hours"] = row["service_work_seconds"] / SECONDS_PER_HOUR
        row["service_work_utilization_proxy"] = (
            row["service_work_seconds"] / (row["replica_count"] * service_period_s)
            if row["replica_count"] > 0
            else None
        )
        row["total_cost_usd"] = row["compute_cost_usd"] + row["egress_cost_usd"]
        compute_cost += float(row["compute_cost_usd"])
        egress_cost += float(row["egress_cost_usd"])
        provisioned_instance_hours += float(row["provisioned_instance_hours"])
        service_work_seconds += float(row["service_work_seconds"])
        egress_total_bytes += float(row["prompt_egress_bytes"] + row["output_egress_bytes"])

    total_cost = compute_cost + egress_cost
    service_work_instance_hours = service_work_seconds / SECONDS_PER_HOUR
    completed = counts["completed_request_count"] * workload_scale_to_service_period
    served = counts["served_request_count"] * workload_scale_to_service_period
    completed_tokens = counts["completed_token_count"] * workload_scale_to_service_period
    served_tokens = counts["served_token_count"] * workload_scale_to_service_period
    cost_per_completed_request = _divide(total_cost, completed)
    cost_per_served_request = _divide(total_cost, served)
    cost_per_completed_token = _divide(total_cost, completed_tokens)
    cost_per_served_token = _divide(total_cost, served_tokens)
    warnings.extend(
        (
            "instance cost bills all configured replicas for the service period; utilization is reported as a service-work proxy and does not discount instance charges",
            "cross-region egress is inferred from selected_path: prompt bytes use the LB-to-LB or ingress-to-replica outbound leg and completed response bytes use the replica-to-client outbound leg",
            "terrestrial cost uses explicit cloud-price inputs only; no LEO capital, launch, satellite, or energy cost is reused",
        )
    )
    return TerrestrialGroundCostResult(
        evaluated=True,
        total_cost_usd=total_cost,
        annualized_cost_usd_per_year=total_cost * annualization_multiplier,
        compute_cost_usd=compute_cost,
        egress_cost_usd=egress_cost,
        total_operational_cost_usd=None,
        cost_coverage=(
            "compute_plus_cross_region_egress"
            if egress_price_map is not None
            else "compute_instance_only"
        ),
        excluded_cost_components=(
            "client_egress",
            "control_plane",
            "dns_tls",
            "retransmissions",
            "provider_tiered_pricing",
            "storage_and_networking_fixed_charges",
            "operations_and_support",
            *( () if egress_price_map is not None else ("cross_region_egress",) ),
        ),
        service_period_s=service_period_s,
        workload_observation_period_s=workload_observation_period_s,
        workload_scale_to_service_period=workload_scale_to_service_period,
        runtime_observation_window_s=observed_runtime_window_s,
        annualization_multiplier=annualization_multiplier,
        provisioned_instance_hours=provisioned_instance_hours,
        service_work_instance_hours=service_work_instance_hours,
        service_work_utilization_proxy=(
            service_work_seconds / (provisioned_instance_hours * SECONDS_PER_HOUR)
            if provisioned_instance_hours > 0
            else None
        ),
        # Without a supplied byte/token and egress-price contract, a zero
        # would mean either "no transfer" or "transfer not modeled".  Keep
        # the latter distinct as null.
        egress_total_gb=(
            egress_total_bytes / BYTES_PER_GB
            if egress_price_map is not None
            else None
        ),
        **counts,
        cost_per_completed_request_usd=cost_per_completed_request,
        cost_per_served_request_usd=cost_per_served_request,
        cost_per_completed_token_usd=cost_per_completed_token,
        cost_per_served_token_usd=cost_per_served_token,
        cost_per_served_mtoken_usd=(cost_per_served_token * 1_000_000.0 if cost_per_served_token is not None else None),
        region_costs=tuple(region_rows[item.region_id] for item in capacities),
        warnings=tuple(dict.fromkeys(warnings)),
        provenance={
            "model": "terrestrial_cloud_instance_and_optional_egress",
            "schema_version": TERRESTRIAL_COSTING_SCHEMA,
            "costing_source": "ground_config.terrestrial_costing",
            "leo_cost_reused": False,
            "annualization_mode": mode,
            "bytes_per_gb": BYTES_PER_GB,
            "egress_accounting": (
                "selected_path_cloud_outbound_legs"
                if egress_price_map is not None
                else "not_configured"
            ),
            "price_provenance": _price_provenance(costing, warnings),
            "service_work_utilization_proxy": "sum(prefill_service_s + decode_service_s) / configured_replica_seconds; not GPU utilization and may exceed one under batched overlap",
        },
    )


def _null_result(
    *,
    warnings: tuple[str, ...],
    provenance: dict[str, Any],
    completed_request_count: int,
    served_request_count: int,
    completed_token_count: int,
    served_token_count: int,
    region_costs: tuple[dict[str, Any], ...],
    service_period_s: float | None = None,
    workload_observation_period_s: float | None = None,
    runtime_observation_window_s: float | None = None,
    annualization_multiplier: float | None = None,
    schema_version: str = TERRESTRIAL_COSTING_SCHEMA,
) -> TerrestrialGroundCostResult:
    return TerrestrialGroundCostResult(
        evaluated=False,
        total_cost_usd=None,
        annualized_cost_usd_per_year=None,
        compute_cost_usd=None,
        egress_cost_usd=None,
        total_operational_cost_usd=None,
        cost_coverage="not_evaluated",
        excluded_cost_components=(),
        service_period_s=service_period_s,
        workload_observation_period_s=workload_observation_period_s,
        workload_scale_to_service_period=None,
        runtime_observation_window_s=runtime_observation_window_s,
        annualization_multiplier=annualization_multiplier,
        provisioned_instance_hours=None,
        service_work_instance_hours=None,
        service_work_utilization_proxy=None,
        egress_total_gb=None,
        completed_request_count=completed_request_count,
        served_request_count=served_request_count,
        completed_token_count=completed_token_count,
        served_token_count=served_token_count,
        cost_per_completed_request_usd=None,
        cost_per_served_request_usd=None,
        cost_per_completed_token_usd=None,
        cost_per_served_token_usd=None,
        cost_per_served_mtoken_usd=None,
        region_costs=region_costs,
        warnings=warnings,
        provenance={**provenance, "schema_version": schema_version},
        schema_version=schema_version,
    )


def _validate_ground_records(
    records: Iterable[Any], *, configured_region_ids: set[str]
) -> list[str]:
    """Reject malformed or mixed-domain records before any pricing arithmetic."""

    errors: list[str] = []
    for record in records:
        request_id = _optional_string(_value(record, "request_id")) or "<unknown-request>"
        action_type = _optional_string(_value(record, "action_type"))
        if action_type is None or not action_type.startswith("ground_"):
            errors.append(
                f"record {request_id} is not a terrestrial ground action and cannot enter terrestrial costing"
            )
            continue
        origin = _optional_string(_value(record, "ground_origin_region_id"))
        serving = _optional_string(_value(record, "ground_serving_region_id"))
        if origin not in configured_region_ids:
            errors.append(
                f"record {request_id} has missing or unconfigured ground_origin_region_id={origin!r}"
            )
        if serving is not None and serving not in configured_region_ids:
            errors.append(
                f"record {request_id} has unconfigured ground_serving_region_id={serving!r}"
            )
        if bool(_value(record, "ground_cross_region", False)) and serving is None:
            errors.append(
                f"cross-region record {request_id} has no ground_serving_region_id"
            )
    return errors


def _records_requiring_egress_validation(records: Iterable[Any]) -> tuple[Any, ...]:
    """Return every record whose path can incur a priced cloud egress leg.

    A centralized coordinator can forward a prompt from US to an EU replica
    even when the request's client and serving replica are both in Europe.
    Therefore ``origin != serving`` is insufficient; selected-path topology is
    authoritative for pricing.
    """

    result: list[Any] = []
    for record in records:
        policy = _optional_string(
            _mapping(_value(record, "provenance")).get("baseline_policy")
        )
        # Centralized and GKE-like baselines cannot infer their billable WAN
        # leg from origin/serving alone: US ingress -> EU replica -> EU client
        # has origin == serving yet still exports the prompt from US.  Their
        # selected path is therefore mandatory even if it happens to contain
        # no billable leg for a particular local placement.
        topology_requires_path = policy in {
            "ground_central_rr",
            "ground_central_ll",
            "ground_central_ch",
            "ground_central_sgl_like",
            "ground_gke_gateway_like",
        }
        if bool(_value(record, "ground_cross_region", False)):
            result.append(record)
            continue
        try:
            has_egress_events = bool(_egress_events_from_selected_path(record))
        except ValueError:
            if topology_requires_path:
                result.append(record)
            continue
        if has_egress_events or topology_requires_path:
            result.append(record)
    return tuple(result)


def _validate_egress_paths(
    cross_records: Iterable[Any], *, configured_region_ids: set[str]
) -> list[str]:
    """Ensure every priced WAN transfer has an auditable cloud outbound leg."""

    errors: list[str] = []
    for record in cross_records:
        request_id = _optional_string(_value(record, "request_id")) or "<unknown-request>"
        policy = _optional_string(
            _mapping(_value(record, "provenance")).get("baseline_policy")
        )
        topology_requires_path = policy in {
            "ground_central_rr",
            "ground_central_ll",
            "ground_central_ch",
            "ground_central_sgl_like",
            "ground_gke_gateway_like",
        }
        raw_path = _value(record, "selected_path")
        if topology_requires_path and (
            not isinstance(raw_path, (tuple, list)) or not raw_path
        ):
            errors.append(
                f"record {request_id} requires a non-empty selected_path for topology-aware egress costing"
            )
            continue
        if topology_requires_path:
            path_error = _validate_baseline_selected_path(raw_path)
            if path_error is not None:
                errors.append(
                    f"record {request_id} has invalid selected_path for topology-aware egress costing: {path_error}"
                )
                continue
        try:
            events = _egress_events_from_selected_path(record)
        except ValueError as exc:
            errors.append(f"cross-region record {request_id} has invalid selected_path for egress costing: {exc}")
            continue
        prompt_count = 0
        output_count = 0
        for source, kind, _ in events:
            if source not in configured_region_ids:
                errors.append(
                    f"cross-region record {request_id} selected_path charges unconfigured source region {source!r}"
                )
            if kind == "prompt":
                prompt_count += 1
            elif kind == "output":
                output_count += 1
        if (
            _optional_string(_value(record, "action_type"))
            == "ground_cross_region_forward"
            and prompt_count != 1
        ):
            errors.append(
                f"cross-region record {request_id} must have exactly one priced prompt outbound leg, found {prompt_count}"
            )
        if prompt_count > 1:
            errors.append(
                f"record {request_id} has more than one priced prompt outbound leg, found {prompt_count}"
            )
        # A completed request streams an output to its client, but that leg
        # is billable egress only when replica and client are in different
        # cloud regions.  Central US -> EU prompt forwarding with an EU
        # replica and EU client is therefore valid with zero output events.
        if output_count > 1:
            errors.append(
                f"cross-region record {request_id} has more than one priced response outbound leg: {output_count}"
            )
    return errors


def _validate_baseline_selected_path(raw_path: Any) -> str | None:
    """Validate the topology ledger required by central/GKE-like policies.

    In those topologies an arbitrary non-empty trace is not sufficient: the
    accounting decision depends on client -> ingress, ingress -> replica, and
    replica -> client being represented explicitly.  This catches corrupted
    traces before they can silently turn a US->EU prompt transfer into zero
    egress.
    """

    if not isinstance(raw_path, (tuple, list)):
        return "selected_path is missing"
    relations = {
        "client_to_baseline_ingress": ("client", "lb"),
        "baseline_ingress_to_replica": ("lb", "ground"),
        "replica_to_client_stream": ("ground", "client"),
    }
    matching: dict[str, list[Mapping[str, Any]]] = {
        relation: [] for relation in relations
    }
    for raw_hop in raw_path:
        hop = _mapping(raw_hop)
        relation = _optional_string(hop.get("relation"))
        if relation in matching:
            matching[relation].append(hop)
    for relation, (source_kind, target_kind) in relations.items():
        hops = matching[relation]
        if len(hops) != 1:
            return f"requires exactly one {relation} leg, found {len(hops)}"
        hop = hops[0]
        source = _optional_string(hop.get("source"))
        target = _optional_string(hop.get("target"))
        if source is None or target is None:
            return f"{relation} lacks source or target"
        if not source.startswith(f"{source_kind}:") or not target.startswith(
            f"{target_kind}:"
        ):
            return f"{relation} has invalid endpoint kinds"
        if _endpoint_region(source) is None or _endpoint_region(target) is None:
            return f"{relation} lacks parseable endpoint regions"
    return None


def _egress_events_from_selected_path(record: Any) -> tuple[tuple[str, str, Any], ...]:
    """Map one ground record's actual selected path to charged cloud egress legs.

    Cloud receives client input without an egress charge.  We therefore price
    only the SkyWalker LB-to-LB/central-ingress-to-replica prompt leg and the
    completed replica-to-client response leg.  The function deliberately
    refuses an incomplete path instead of guessing from origin/serving IDs.
    """

    raw_path = _value(record, "selected_path")
    if not isinstance(raw_path, (tuple, list)):
        raise ValueError("selected_path is missing")
    result: list[tuple[str, str, Any]] = []
    for raw_hop in raw_path:
        hop = _mapping(raw_hop)
        relation = _optional_string(hop.get("relation"))
        source = _endpoint_region(hop.get("source"))
        target = _endpoint_region(hop.get("target"))
        if relation in {"cross_region_lb_forward", "baseline_ingress_to_replica"}:
            if source is None or target is None:
                raise ValueError(f"{relation} does not encode source/target regions")
            if source != target:
                result.append((source, "prompt", record))
        elif relation == "replica_to_client_stream":
            if source is None or target is None:
                raise ValueError("replica_to_client_stream does not encode source/target regions")
            if source != target and bool(_value(record, "completed", False)):
                result.append((source, "output", record))
    return tuple(result)


def _endpoint_region(value: Any) -> str | None:
    text = _optional_string(value)
    if text is None:
        return None
    pieces = text.split(":")
    if len(pieces) < 2 or pieces[0] not in {"client", "lb", "ground"}:
        return None
    return _optional_string(pieces[1])


def _price_provenance(costing: Mapping[str, Any], warnings: list[str]) -> dict[str, Any]:
    supplied = _mapping(costing.get("price_provenance"))
    if not supplied:
        warnings.append(
            "terrestrial prices are user-supplied and unverified; provider/SKU/GPU-count/effective-date/source metadata is required for paper numerical claims"
        )
        return {"status": "user_supplied_unverified"}
    return {
        "status": str(supplied.get("status") or "metadata_supplied_not_verified"),
        "currency": supplied.get("currency"),
        "provider": supplied.get("provider"),
        "sku": supplied.get("sku"),
        "gpu_per_replica": supplied.get("gpu_per_replica"),
        "price_type": supplied.get("price_type"),
        "effective_date": supplied.get("effective_date"),
        "source_url": supplied.get("source_url"),
        "source_artifact_sha256": supplied.get("source_artifact_sha256"),
        "regions": _mapping(supplied.get("regions")),
    }


def _configured_regions(config: Mapping[str, Any]) -> tuple[_RegionCapacity, ...]:
    raw_regions = config.get("regions")
    if not isinstance(raw_regions, (list, tuple)):
        return ()
    result: list[_RegionCapacity] = []
    seen: set[str] = set()
    for raw in raw_regions:
        region = _mapping(raw)
        region_id = _optional_string(region.get("region_id"))
        replica_count = _nonnegative_int(region.get("replica_count"))
        if region_id is None or replica_count is None or replica_count <= 0 or region_id in seen:
            return ()
        seen.add(region_id)
        result.append(_RegionCapacity(region_id=region_id, replica_count=replica_count))
    return tuple(result)


def _record_counts(records: Iterable[Any]) -> dict[str, int]:
    completed = [record for record in records if bool(_value(record, "completed", False))]
    served = [record for record in records if bool(_value(record, "served", False))]
    return {
        "completed_request_count": len(completed),
        "served_request_count": len(served),
        "completed_token_count": int(
            sum((_nonnegative_float(_value(record, "prompt_tokens")) or 0.0) + (_nonnegative_float(_value(record, "output_tokens")) or 0.0) for record in completed)
        ),
        "served_token_count": int(
            sum((_nonnegative_float(_value(record, "prompt_tokens")) or 0.0) + (_nonnegative_float(_value(record, "output_tokens")) or 0.0) for record in served)
        ),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_float(value: Any) -> float | None:
    parsed = _nonnegative_float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _nonnegative_int(value: Any) -> int | None:
    parsed = _nonnegative_float(value)
    if parsed is None or int(parsed) != parsed:
        return None
    return int(parsed)


def _nonnegative_number_map(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = _optional_string(raw_key)
        parsed = _nonnegative_float(raw_value)
        if key is None or parsed is None:
            return None
        result[key] = parsed
    return result


def _divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _approximately_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


__all__ = [
    "BYTES_PER_GB",
    "HOURS_PER_YEAR",
    "TERRESTRIAL_COSTING_SCHEMA",
    "TerrestrialGroundCostResult",
    "estimate_terrestrial_ground_cost",
]
