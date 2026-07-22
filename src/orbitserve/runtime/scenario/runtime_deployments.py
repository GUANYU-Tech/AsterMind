"""Build the single-satellite-serving deployment catalog used by Runtime."""

from __future__ import annotations

from typing import Any

from ...common import to_jsonable
from .deployment_contracts import RuntimeDeployment, RuntimeDeploymentCatalog


def build_runtime_deployment_catalog(
    scenario: Any,
    *,
    satellite_catalog: tuple[dict[str, Any], ...] | None = None,
) -> RuntimeDeploymentCatalog:
    """Build one full-serving deployment from the scenario's public view.

    Runtime assumes every listed member satellite can execute both Prefill and
    Decode.  It never creates phase-specific groups or Prefill/Decode pairs.
    """

    serving_deployments = tuple(getattr(scenario, "serving_deployments", ()) or ())
    constellation = _mapping(getattr(scenario, "constellation", None))
    if not serving_deployments:
        return RuntimeDeploymentCatalog(
            warnings=("runtime deployment catalog is empty because serving_deployments is empty",),
            provenance={"source": "runtime_deployments.build_runtime_deployment_catalog"},
        )

    payload = _mapping(serving_deployments[0])
    throughput = _mapping(payload.get("throughput_estimate"))
    deployment_id = str(payload.get("deployment_id") or "deployment_0")
    catalog_rows = satellite_catalog or tuple(
        _mapping(item)
        for item in constellation.get("satellite_catalog", ()) or ()
    )
    member_satellites = tuple(
        str(row["satellite_id"])
        for row in catalog_rows
        if row.get("satellite_id") is not None
    )
    if not member_satellites:
        return RuntimeDeploymentCatalog(
            warnings=("runtime service catalog requires Orbit satellite_catalog",),
            provenance={"source": "runtime_deployments.build_runtime_deployment_catalog"},
        )

    prefill_tokens_per_s = _optional_float(
        _first_present(payload.get("prefill_tokens_per_s"), throughput.get("prefill_tokens_per_s"))
    )
    decode_tokens_per_s = _optional_float(
        _first_present(payload.get("decode_tokens_per_s"), throughput.get("decode_tokens_per_s"))
    )
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
        satellites_per_deployment=1,
        internal_comm_accounting="embedded",
        placement_model="single_satellite_full_serving",
        selection_policy="runtime_scheduler_selects_serving_satellite",
        provenance={
            "source": "single_satellite_serving_capacity",
            "serving_assumption": "all_orbit_satellites_run_full_prefill_decode_serving",
            "satellite_id_source": "Orbit satellite_catalog",
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
            "constellation_satellite_count": len(member_satellites),
            "serving_assumption": "all_orbit_satellites_run_full_prefill_decode_serving",
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
