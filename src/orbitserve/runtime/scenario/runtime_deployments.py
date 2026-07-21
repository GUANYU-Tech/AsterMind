"""Build the single-satellite-serving deployment catalog used by Runtime."""

from __future__ import annotations

from typing import Any

from ...common import to_jsonable
from .deployment_contracts import RuntimeDeployment, RuntimeDeploymentCatalog


def build_runtime_deployment_catalog(scenario: Any) -> RuntimeDeploymentCatalog:
    """Build one full-serving deployment from the scenario's public view.

    Runtime assumes every listed member satellite can execute both Prefill and
    Decode.  It never creates phase-specific groups or Prefill/Decode pairs.
    """

    serving_deployments = tuple(getattr(scenario, "serving_deployments", ()) or ())
    constellation = _mapping(getattr(scenario, "constellation", None))
    total_satellites = _constellation_satellite_count(constellation)
    if not serving_deployments:
        return RuntimeDeploymentCatalog(
            warnings=("runtime deployment catalog is empty because serving_deployments is empty",),
            provenance={"source": "runtime_deployments.build_runtime_deployment_catalog"},
        )

    payload = _mapping(serving_deployments[0])
    throughput = _mapping(payload.get("throughput_estimate"))
    deployment_id = str(payload.get("deployment_id") or "deployment_0")
    satellites_per_deployment = _optional_int(
        _first_present(payload.get("satellites_per_deployment"), throughput.get("satellites_per_deployment"))
    )
    member_satellites = _explicit_member_satellites(payload)
    if not member_satellites:
        member_count = (
            _serving_satellite_budget_count(payload, throughput)
            or total_satellites
            or satellites_per_deployment
            or 1
        )
        member_satellites = _synthetic_satellite_ids(max(1, member_count))

    prefill_tokens_per_s = _optional_float(
        _first_present(payload.get("prefill_tokens_per_s"), throughput.get("prefill_tokens_per_s"))
    )
    decode_tokens_per_s = _optional_float(
        _first_present(payload.get("decode_tokens_per_s"), throughput.get("decode_tokens_per_s"))
    )
    budget_payload = _serving_satellite_budget_payload(payload, throughput)
    warnings = tuple(dict.fromkeys(_string_tuple(payload.get("warnings"))))
    deployment = RuntimeDeployment(
        deployment_id=deployment_id,
        member_satellites=member_satellites,
        entry_satellite_ids=member_satellites,
        prefill_tokens_per_s=prefill_tokens_per_s,
        decode_tokens_per_s=decode_tokens_per_s,
        request_throughput_per_s=_optional_float(
            _first_present(payload.get("request_throughput_per_s"), throughput.get("request_throughput_per_s"))
        ),
        kv_cache_capacity_tokens=_optional_int(
            _first_present(payload.get("kv_cache_capacity_tokens"), throughput.get("kv_cache_capacity_tokens"))
        ),
        kv_cache_bytes_per_token=_optional_float(
            _first_present(payload.get("kv_cache_bytes_per_token"), throughput.get("kv_cache_bytes_per_token"))
        ),
        satellites_per_deployment=satellites_per_deployment,
        internal_comm_accounting="embedded",
        placement_model="single_satellite_full_serving",
        selection_policy="runtime_scheduler_selects_serving_satellite",
        provenance={
            "source": "ServingDeploymentView",
            "input_deployment_id": deployment_id,
            "member_satellite_assignment": _member_assignment_source(payload, constellation),
            "serving_assumption": "each_member_satellite_executes_prefill_and_decode",
            **budget_payload,
        },
        warnings=warnings,
    )
    return RuntimeDeploymentCatalog(
        deployments=(deployment,),
        placement_model="single_satellite_full_serving",
        selection_policy="runtime_scheduler_selects_serving_satellite",
        warnings=warnings,
        provenance={
            "source": "runtime_deployments.build_runtime_deployment_catalog",
            "constellation_satellite_count": total_satellites,
            "serving_assumption": "single_satellite_full_serving",
            **budget_payload,
        },
    )


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return to_jsonable(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = to_jsonable(value.to_dict())
        return converted if isinstance(converted, dict) else {}
    converted = to_jsonable(value)
    return converted if isinstance(converted, dict) else {}


def _serving_satellite_budget_payload(payload: dict[str, Any], throughput: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "serving_satellite_budget_count",
        "constellation_satellite_count",
        "satellite_budget_mode",
        "satellite_budget_source",
        "comparison_group_id",
        "satellite_budget_unused_count",
    )
    return {
        key: value
        for key in keys
        if (value := _first_present(payload.get(key), throughput.get(key))) is not None
    }


def _serving_satellite_budget_count(payload: dict[str, Any], throughput: dict[str, Any]) -> int | None:
    count = _optional_int(_serving_satellite_budget_payload(payload, throughput).get("serving_satellite_budget_count"))
    return count if count is not None and count > 0 else None


def _constellation_satellite_count(constellation: dict[str, Any]) -> int | None:
    return _optional_int(
        _first_present(
            constellation.get("satellite_count"),
            _mapping(constellation.get("design")).get("satellite_count"),
        )
    )


def _explicit_member_satellites(payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in payload.get("member_satellites", ()) or ())


def _synthetic_satellite_ids(count: int) -> tuple[str, ...]:
    return tuple(f"sat-{index:05d}" for index in range(max(0, count)))


def _member_assignment_source(payload: dict[str, Any], constellation: dict[str, Any]) -> str:
    if _explicit_member_satellites(payload):
        return "payload.member_satellites"
    if constellation.get("satellite_catalog"):
        return "constellation.satellite_catalog"
    return "deterministic synthetic sat-xxxxx ids"


def _first_present(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


__all__ = ["RuntimeDeployment", "RuntimeDeploymentCatalog", "build_runtime_deployment_catalog"]
