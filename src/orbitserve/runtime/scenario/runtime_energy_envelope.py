"""Adapters for closing the static cost model to runtime energy traces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from ...common import to_jsonable


COST_ENERGY_TRACE_OPTION_KEYS = (
    "battery_capacity_j",
    "battery_charge_efficiency",
    "battery_discharge_efficiency",
    "charge_rate_w",
    "idle_power_w",
    "idle_power_w_sunlit",
    "idle_power_w_shadow",
    "initial_soc",
    "compute_energy_j_per_token",
    "link_energy_j_per_byte",
    "soc_min",
    "shadow_fraction",
    "energy_accounting_mode",
)


@dataclass(frozen=True)
class RuntimeEnergyEnvelope:
    evaluated: bool = False
    trace_options: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def apply_runtime_energy_envelope(
    simulation_config: Any,
    *,
    cost_result: Any,
    resolved_hardware: Any = None,
) -> tuple[Any, RuntimeEnergyEnvelope]:
    """Return a simulation config whose runtime energy options come from cost/hardware."""

    envelope = derive_runtime_energy_envelope(
        cost_result,
        resolved_hardware=resolved_hardware,
        base_trace_options=getattr(simulation_config, "trace_options", {}) or {},
    )
    current_options = dict(getattr(simulation_config, "trace_options", {}) or {})
    next_options = dict(current_options)
    next_options["runtime_energy_envelope"] = envelope.to_dict()
    if envelope.evaluated:
        next_options.update(envelope.trace_options)
        next_options["runtime_energy_envelope_required"] = True
    elif _trace_backed_runtime_model(simulation_config):
        next_options["runtime_energy_envelope_required"] = True
        next_options["runtime_energy_envelope_unavailable"] = True
    try:
        return replace(simulation_config, trace_options=next_options), envelope
    except TypeError:
        return simulation_config, envelope


def derive_runtime_energy_envelope(
    cost_result: Any,
    *,
    resolved_hardware: Any = None,
    base_trace_options: dict[str, Any] | None = None,
) -> RuntimeEnergyEnvelope:
    """Project cost-model battery/solar/power closure into runtime trace options."""

    cost_payload = _mapping(cost_result)
    if not bool(cost_payload.get("evaluated", False)):
        return RuntimeEnergyEnvelope(
            evaluated=False,
            warnings=("runtime energy envelope was not derived because cost.evaluated=false",),
            provenance={"source": "cost_hardware_orbit_runtime", "cost_evaluated": False},
        )
    single = _dict(cost_payload.get("single_satellite"))
    if not single:
        return RuntimeEnergyEnvelope(
            evaluated=False,
            warnings=("runtime energy envelope was not derived because cost.single_satellite is missing",),
            provenance={"source": "cost_hardware_orbit_runtime", "cost_evaluated": True},
        )
    single_provenance = _dict(single.get("provenance"))
    power_sizing_mode = str(
        single_provenance.get("power_sizing_mode") or "conservative_peak_load"
    )

    battery_wh = _first_float(single.get("E_battery_req_wh"), single.get("E_battery_req"))
    solar_w = _first_float(single.get("P_solar_req_w"), single.get("P_solar_req"))
    p_avg_w = _first_float(single.get("P_avg_w"), single.get("P_avg"))
    p_shadow_w = _first_float(single.get("P_shadow_avg_w"), single.get("P_shadow_avg"))
    missing = [
        name
        for name, value in (
            ("E_battery_req_wh", battery_wh),
            ("P_solar_req_w", solar_w),
            ("P_avg_w", p_avg_w),
            ("P_shadow_avg_w", p_shadow_w),
        )
        if value is None or not math.isfinite(value) or value <= 0
    ]
    if missing:
        return RuntimeEnergyEnvelope(
            evaluated=False,
            warnings=(f"runtime energy envelope was not derived because cost fields are missing/nonpositive: {', '.join(missing)}",),
            provenance={"source": "cost_hardware_orbit_runtime", "missing_fields": missing},
        )

    cost_config = getattr(resolved_hardware, "cost_config", None)
    power_config = getattr(cost_config, "power", None)
    platform_config = getattr(cost_config, "platform", None)
    dod_max = _optional_float(getattr(power_config, "dod_max", None))
    base_options = dict(base_trace_options or {})
    initial_soc = _optional_float(base_options.get("initial_soc", 0.8))
    battery_eol_fraction = _optional_float(getattr(power_config, "battery_eol_fraction", None))
    eta_pmad = _optional_float(getattr(power_config, "eta_pmad", None))
    pv_eol_fraction = _optional_float(getattr(power_config, "pv_eol_fraction", None))
    battery_charge_efficiency = _optional_float(
        getattr(power_config, "battery_charge_efficiency", 1.0)
    )
    battery_discharge_efficiency = _optional_float(
        getattr(power_config, "battery_discharge_efficiency", 1.0)
    )
    platform_power_w = _optional_float(getattr(platform_config, "power_w", None))
    compute_standby_power_w = _optional_float(
        getattr(power_config, "compute_standby_power_w", None)
    )
    communication_standby_power_w = _optional_float(
        getattr(power_config, "communication_standby_power_w", None)
    )
    compute_energy_j_per_token = _optional_float(
        getattr(power_config, "compute_energy_j_per_token", None)
    )
    link_energy_j_per_byte = _optional_float(
        getattr(power_config, "link_energy_j_per_byte", None)
    )
    orbit_condition = _dict(single_provenance.get("orbit_energy_condition"))
    sunlit_fraction = _optional_float(orbit_condition.get("sunlit_fraction"))
    eclipse_duration_min = _optional_float(orbit_condition.get("eclipse_duration_min"))
    contract_values = {
        "battery_eol_fraction": battery_eol_fraction,
        "eta_pmad": eta_pmad,
        "pv_eol_fraction": pv_eol_fraction,
        "battery_charge_efficiency": battery_charge_efficiency,
        "battery_discharge_efficiency": battery_discharge_efficiency,
        "sunlit_fraction": sunlit_fraction,
        "platform_power_w": platform_power_w,
    }
    invalid_contract = [
        name
        for name, value in contract_values.items()
        if value is None
        or not math.isfinite(value)
        or value <= 0
        or (name != "platform_power_w" and value > 1)
    ]
    if dod_max is not None and (not math.isfinite(dod_max) or not 0 <= dod_max <= 1):
        invalid_contract.append("dod_max")
    if initial_soc is None or not math.isfinite(initial_soc) or not 0 <= initial_soc <= 1:
        invalid_contract.append("initial_soc")
    elif dod_max is not None and initial_soc <= 1.0 - dod_max:
        invalid_contract.append("initial_soc_not_above_soc_min")
    if eclipse_duration_min is not None and (
        not math.isfinite(eclipse_duration_min) or eclipse_duration_min < 0
    ):
        invalid_contract.append("eclipse_duration_min")
    nonnegative_contract_values = {
        "compute_standby_power_w": compute_standby_power_w,
        "communication_standby_power_w": communication_standby_power_w,
        "compute_energy_j_per_token": compute_energy_j_per_token,
        "link_energy_j_per_byte": link_energy_j_per_byte,
    }
    invalid_contract.extend(
        name
        for name, value in nonnegative_contract_values.items()
        if value is None or not math.isfinite(value) or value < 0
    )
    if invalid_contract:
        return RuntimeEnergyEnvelope(
            evaluated=False,
            warnings=(
                "runtime energy envelope was not derived because the event-incremental energy contract "
                f"is missing/invalid: {', '.join(invalid_contract)}",
            ),
            provenance={"source": "cost_hardware_orbit_runtime", "invalid_contract_fields": invalid_contract},
        )

    effective_battery_j = battery_wh * 3600.0 * float(battery_eol_fraction)
    available_solar_bus_w = solar_w * float(eta_pmad) * float(pv_eol_fraction)
    always_on_baseline_power_w = float(platform_power_w)
    always_on_baseline_power_w += float(compute_standby_power_w) + float(
        communication_standby_power_w
    )
    trace_options = {
        "battery_capacity_j": effective_battery_j,
        "battery_charge_efficiency": float(battery_charge_efficiency),
        "battery_discharge_efficiency": float(battery_discharge_efficiency),
        "initial_soc": float(initial_soc),
        "charge_rate_w": available_solar_bus_w,
        "idle_power_w": always_on_baseline_power_w,
        "idle_power_w_sunlit": always_on_baseline_power_w,
        "idle_power_w_shadow": always_on_baseline_power_w,
        "compute_energy_j_per_token": float(compute_energy_j_per_token),
        "link_energy_j_per_byte": float(link_energy_j_per_byte),
        "shadow_fraction": _clip01(1.0 - float(sunlit_fraction)),
        "energy_accounting_mode": "event_incremental_v1",
    }
    if dod_max is not None:
        trace_options["soc_min"] = _clip01(1.0 - dod_max)
    orbit_period_s = _first_float(base_options.get("orbit_period_s"), 5700.0)
    if orbit_period_s is None or not math.isfinite(orbit_period_s) or orbit_period_s <= 0:
        return RuntimeEnergyEnvelope(
            evaluated=False,
            warnings=("runtime energy envelope was not derived because orbit_period_s is invalid",),
            provenance={"source": "cost_hardware_orbit_runtime", "invalid_contract_fields": ["orbit_period_s"]},
        )

    warnings = [
        "runtime energy envelope uses event_incremental_v1: always-on baseline plus per-event compute/link energy",
    ]
    for key, next_value in trace_options.items():
        if key in base_options and (
            str(base_options.get(key)) != str(next_value)
            if isinstance(next_value, str)
            else _optional_float(base_options.get(key)) != _optional_float(next_value)
        ):
            warnings.append(f"simulation.trace_options.{key} overridden by cost/hardware runtime energy envelope")
    warnings.append(
        "runtime compute/link per-request energy coefficients are baseline-exclusive and forced from cost/hardware"
    )

    provenance = {
        "source": "cost_hardware_orbit_runtime",
        "adapter": "orbitserve.runtime_energy_envelope.derive_runtime_energy_envelope",
        "cost_fields": {
            "battery_nameplate_bol_wh": battery_wh,
            "solar_nameplate_bol_w": solar_w,
            "sizing_average_power_w": p_avg_w,
            "sizing_shadow_average_power_w": p_shadow_w,
            "power_sizing_mode": power_sizing_mode,
        },
        "runtime_fields": {
            "battery_effective_eol_j": effective_battery_j,
            "solar_bus_available_eol_w": available_solar_bus_w,
            "always_on_baseline_power_w": always_on_baseline_power_w,
            "platform_power_w": platform_power_w,
            "compute_standby_power_w": compute_standby_power_w,
            "communication_standby_power_w": communication_standby_power_w,
            "compute_energy_j_per_token": compute_energy_j_per_token,
            "link_energy_j_per_byte": link_energy_j_per_byte,
            "battery_charge_efficiency": battery_charge_efficiency,
            "battery_discharge_efficiency": battery_discharge_efficiency,
            "initial_soc": initial_soc,
            "soc_min": _clip01(1.0 - dod_max) if dod_max is not None else None,
            "sunlit_fraction": sunlit_fraction,
            "orbit_period_s": orbit_period_s,
            "energy_accounting_mode": "event_incremental_v1",
        },
        "hardware_power": {
            "dod_max": dod_max,
            "battery_eol_fraction": battery_eol_fraction,
            "eta_pmad": eta_pmad,
            "pv_eol_fraction": pv_eol_fraction,
            "eclipse_duration_min": eclipse_duration_min,
            "sunlit_fraction": sunlit_fraction,
        },
        "energy_sizing": {
            "sizing_mode": power_sizing_mode,
            "uses_runtime_served_requests": False,
            "planned_energy_demand": single_provenance.get("planned_energy_demand"),
        },
        "overrides": tuple(key for key in COST_ENERGY_TRACE_OPTION_KEYS if key in trace_options),
    }
    return RuntimeEnergyEnvelope(
        evaluated=True,
        trace_options=trace_options,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=provenance,
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


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _trace_backed_runtime_model(simulation_config: Any) -> bool:
    return str(getattr(simulation_config, "trace_model", "")) in {
        "surrogate_orbit_trace",
        "circular_geometry_trace",
    }


__all__ = [
    "COST_ENERGY_TRACE_OPTION_KEYS",
    "RuntimeEnergyEnvelope",
    "apply_runtime_energy_envelope",
    "derive_runtime_energy_envelope",
]
