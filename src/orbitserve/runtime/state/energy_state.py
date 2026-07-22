"""Dynamic per-satellite energy and SOC state transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable
from .energy_accounting import EnergyIntensity, compute_dynamic_energy_j


REQUIRED_RUNTIME_ENERGY_WINDOW_FIELDS = (
    "battery_capacity_j",
    "charge_rate_w",
    "idle_power_w",
    "compute_energy_j_per_token",
    "link_energy_j_per_byte",
    "soc_min",
    "energy_accounting_mode",
)


@dataclass(frozen=True)
class EnergyUpdateResult:
    evaluated: bool = False
    soc_before: float | None = None
    soc_after: float | None = None
    energy_violation: bool | None = None
    delta_compute_j: float | None = None
    delta_link_j: float | None = None
    warnings: tuple[str, ...] = ("energy/SOC integration is not implemented",)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)




class EnergyWindowError(ValueError):
    def __init__(
        self,
        reason: str,
        *,
        satellite_id: str,
        time_s: float,
        sample_time_s: float | None = None,
        warnings: tuple[str, ...] = (),
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.satellite_id = satellite_id
        self.time_s = time_s
        self.sample_time_s = sample_time_s
        self.warnings = warnings


def query_energy_window_at(trace_bundle: Any, satellite_id: str, time_s: float) -> dict[str, Any]:
    method = getattr(trace_bundle, "energy_window_at", None)
    if callable(method):
        return _as_energy_query_result(
            method(satellite_id, time_s),
            unavailable_reason="no_energy_window",
            satellite_id=satellite_id,
            time_s=time_s,
        )
    candidates = [
        window
        for window in getattr(trace_bundle, "energy_windows", ()) or ()
        if str(window.get("satellite_id")) == str(satellite_id)
        and float(window.get("start_s", 0.0)) <= time_s < float(window.get("end_s", 0.0))
    ]
    if candidates:
        window = dict(
            sorted(
                candidates,
                key=lambda item: (float(item.get("start_s", 0.0)), float(item.get("end_s", 0.0))),
            )[0]
        )
        return {
            "available": True,
            "reason": None,
            "sample_time_s": time_s,
            "stale": False,
            "value": window,
            "warnings": _string_tuple(window.get("warnings")),
        }
    return {
        "available": False,
        "reason": "no_energy_window",
        "sample_time_s": time_s,
        "stale": False,
        "value": None,
        "warnings": ("no_energy_window",),
    }


def energy_window_at(trace_bundle: Any, satellite_id: str, time_s: float) -> dict[str, Any]:
    result = query_energy_window_at(trace_bundle, satellite_id, time_s)
    if result["available"] and not result.get("stale", False):
        window = dict(result.get("value") or {})
        _validate_runtime_energy_window(window, satellite_id=satellite_id, time_s=time_s)
        return window
    reason = str(result.get("reason") or "no_energy_window")
    raise EnergyWindowError(
        reason,
        satellite_id=str(satellite_id),
        time_s=float(time_s),
        sample_time_s=result.get("sample_time_s"),
        warnings=_string_tuple(result.get("warnings")) or (reason,),
    )


def advance_satellite_energy(satellite_state: Any, trace_bundle: Any, to_s: float) -> None:
    """Materialize a satellite SOC snapshot at ``to_s`` from its event ledger.

    Replaying from the immutable initial SOC makes out-of-order future queue
    reservations safe: asking for an earlier time never reuses a SOC snapshot
    that already contains later request debits.  State is committed only after
    the full trace interval validates, so a window gap cannot half-update SOC.
    """

    to_s = float(to_s)
    if not math.isfinite(to_s) or to_s < 0:
        raise EnergyWindowError(
            "invalid_energy_time",
            satellite_id=str(getattr(satellite_state, "satellite_id", "<unknown>")),
            time_s=to_s,
            warnings=("invalid_energy_time",),
        )
    initial_soc = _finite_fraction(
        getattr(satellite_state, "initial_soc", getattr(satellite_state, "soc", 0.8)),
        "initial_soc",
        satellite_id=str(satellite_state.satellite_id),
        time_s=0.0,
    )
    all_events = tuple(dict(event) for event in tuple(getattr(satellite_state, "energy_events", ()) or ()))
    for event in all_events:
        event_time_s = _nonnegative_finite(
            event.get("event_time_s"),
            "event_time_s",
            satellite_id=str(satellite_state.satellite_id),
            time_s=to_s,
        )
        if event.get("event_id") is None:
            raise _energy_value_error(
                "event_id",
                satellite_id=str(satellite_state.satellite_id),
                time_s=event_time_s,
            )
    events = sorted(
        (event for event in all_events if float(event["event_time_s"]) <= to_s),
        key=lambda event: (float(event.get("event_time_s", 0.0)), int(event.get("sequence", 0))),
    )
    last_update_s = float(getattr(satellite_state, "last_energy_update_s", 0.0))
    materialized_ids = {
        str(item)
        for item in tuple(getattr(satellite_state, "materialized_energy_event_ids", ()) or ())
    }
    unmaterialized_before_snapshot = any(
        float(event["event_time_s"]) < last_update_s - 1e-9
        and str(event["event_id"]) not in materialized_ids
        for event in events
    )
    if to_s >= last_update_s and not unmaterialized_before_snapshot:
        current_s = last_update_s
        soc = float(getattr(satellite_state, "soc", initial_soc))
        ledger = list(tuple(getattr(satellite_state, "energy_ledger", ()) or ()))
        for event in events:
            event_id = str(event["event_id"])
            if event_id in materialized_ids:
                continue
            event_time_s = float(event["event_time_s"])
            soc, rows = _integrate_idle_charge_interval(
                satellite_state,
                trace_bundle,
                soc=soc,
                start_s=current_s,
                end_s=event_time_s,
            )
            ledger.extend(rows)
            window = energy_window_at(trace_bundle, satellite_state.satellite_id, event_time_s)
            soc, row = _apply_stored_event(satellite_state, event, window, soc=soc)
            ledger.append(row)
            materialized_ids.add(event_id)
            current_s = event_time_s
        soc, rows = _integrate_idle_charge_interval(
            satellite_state,
            trace_bundle,
            soc=soc,
            start_s=current_s,
            end_s=to_s,
        )
        ledger.extend(rows)
        final_window = _energy_window_at_boundary(trace_bundle, satellite_state.satellite_id, to_s)
        satellite_state.soc = soc
        satellite_state.last_energy_update_s = to_s
        satellite_state.sunlit = bool(final_window.get("sunlit", True))
        satellite_state.energy_ledger = tuple(ledger)
        satellite_state.materialized_energy_event_ids = tuple(sorted(materialized_ids))
        return
    current_s = 0.0
    soc = initial_soc
    ledger: list[dict[str, Any]] = []
    for event in events:
        event_time_s = float(event.get("event_time_s", 0.0))
        soc, rows = _integrate_idle_charge_interval(
            satellite_state,
            trace_bundle,
            soc=soc,
            start_s=current_s,
            end_s=event_time_s,
        )
        ledger.extend(rows)
        window = energy_window_at(trace_bundle, satellite_state.satellite_id, event_time_s)
        soc, row = _apply_stored_event(
            satellite_state,
            event,
            window,
            soc=soc,
        )
        ledger.append(row)
        current_s = event_time_s
    soc, rows = _integrate_idle_charge_interval(
        satellite_state,
        trace_bundle,
        soc=soc,
        start_s=current_s,
        end_s=to_s,
    )
    ledger.extend(rows)
    final_window = _energy_window_at_boundary(trace_bundle, satellite_state.satellite_id, to_s)
    satellite_state.soc = soc
    satellite_state.last_energy_update_s = to_s
    satellite_state.sunlit = bool(final_window.get("sunlit", True))
    satellite_state.energy_ledger = tuple(ledger)
    satellite_state.materialized_energy_event_ids = tuple(str(event["event_id"]) for event in events)


def _integrate_idle_charge_interval(
    satellite_state: Any,
    trace_bundle: Any,
    *,
    soc: float,
    start_s: float,
    end_s: float,
) -> tuple[float, list[dict[str, Any]]]:
    if end_s <= start_s:
        return soc, []
    current_s = start_s
    indexed_windows = getattr(trace_bundle, "_energy_windows_by_satellite", None)
    if callable(indexed_windows):
        window_source = indexed_windows().get(str(satellite_state.satellite_id), ())
    else:
        window_source = getattr(trace_bundle, "energy_windows", ()) or ()
    windows = sorted(
        [
            window
            for window in window_source
            if str(window.get("satellite_id")) == str(satellite_state.satellite_id)
            and float(window.get("end_s", 0.0)) > start_s
            and float(window.get("start_s", 0.0)) < end_s
        ],
        key=lambda item: (float(item.get("start_s", 0.0)), float(item.get("end_s", 0.0))),
    )
    rows: list[dict[str, Any]] = []
    for window in windows:
        segment_start = max(current_s, float(window.get("start_s", current_s)))
        segment_end = min(end_s, float(window.get("end_s", end_s)))
        if segment_start > current_s:
            raise EnergyWindowError(
                "no_energy_window",
                satellite_id=str(satellite_state.satellite_id),
                time_s=current_s,
                warnings=("no_energy_window",),
            )
        if segment_end <= segment_start:
            continue
        soc, row = _idle_charge_delta(
            satellite_state,
            window,
            duration_s=segment_end - segment_start,
            soc=soc,
            event_time_s=segment_end,
        )
        rows.append(row)
        current_s = segment_end
    if current_s < end_s:
        raise EnergyWindowError(
            "no_energy_window",
            satellite_id=str(satellite_state.satellite_id),
            time_s=current_s,
            warnings=("no_energy_window",),
        )
    return soc, rows


def apply_request_energy(
    satellite_state: Any,
    *,
    prompt_tokens: int,
    output_tokens: int,
    link_bytes: float,
    energy_window: dict[str, Any],
    event_time_s: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    satellite_id = str(getattr(satellite_state, "satellite_id", "<unknown>"))
    current_time_s = float(getattr(satellite_state, "last_energy_update_s", 0.0))
    _validate_runtime_energy_window(
        energy_window,
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    _positive_finite(
        energy_window["battery_capacity_j"],
        "battery_capacity_j",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    compute_coefficient = _nonnegative_finite(
        energy_window["compute_energy_j_per_token"],
        "compute_energy_j_per_token",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    link_coefficient = _nonnegative_finite(
        energy_window["link_energy_j_per_byte"],
        "link_energy_j_per_byte",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    try:
        dynamic = compute_dynamic_energy_j(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            link_bytes=link_bytes,
            intensity=EnergyIntensity(
                compute_energy_j_per_token=compute_coefficient,
                link_energy_j_per_byte=link_coefficient,
            ),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        field_name = _dynamic_energy_error_field(exc)
        raise _energy_value_error(
            field_name,
            satellite_id=satellite_id,
            time_s=current_time_s,
        ) from None
    prompt_tokens_accounted = _nonnegative_finite(
        prompt_tokens,
        "prompt_tokens",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    output_tokens_accounted = _nonnegative_finite(
        output_tokens,
        "output_tokens",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    compute_tokens = _nonnegative_finite(
        prompt_tokens_accounted + output_tokens_accounted,
        "compute_tokens",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    link_bytes_accounted = _nonnegative_finite(
        link_bytes,
        "link_bytes",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    compute_j = _nonnegative_finite(
        dynamic.compute_j,
        "compute_j",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    link_j = _nonnegative_finite(
        dynamic.link_j,
        "link_j",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    _nonnegative_finite(
        dynamic.total_j,
        "dynamic_energy_j",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    base_compute_coefficient = _nonnegative_finite(
        energy_window.get("base_compute_energy_j_per_token", compute_coefficient),
        "base_compute_energy_j_per_token",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    base_link_coefficient = _nonnegative_finite(
        energy_window.get("base_link_energy_j_per_byte", link_coefficient),
        "base_link_energy_j_per_byte",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    runtime_compute_energy_scale = _nonnegative_finite(
        energy_window.get("runtime_compute_energy_scale", 1.0),
        "runtime_compute_energy_scale",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    runtime_link_energy_scale = _nonnegative_finite(
        energy_window.get("runtime_link_energy_scale", 1.0),
        "runtime_link_energy_scale",
        satellite_id=satellite_id,
        time_s=current_time_s,
    )
    if not math.isclose(
        compute_coefficient,
        base_compute_coefficient * runtime_compute_energy_scale,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise _energy_value_error(
            "runtime_compute_energy_scale",
            satellite_id=satellite_id,
            time_s=current_time_s,
        )
    if not math.isclose(
        link_coefficient,
        base_link_coefficient * runtime_link_energy_scale,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise _energy_value_error(
            "runtime_link_energy_scale",
            satellite_id=satellite_id,
            time_s=current_time_s,
        )
    event_time = float(
        current_time_s
        if event_time_s is None
        else event_time_s
    )
    _nonnegative_finite(
        event_time,
        "event_time_s",
        satellite_id=str(getattr(satellite_state, "satellite_id", "<unknown>")),
        time_s=float(getattr(satellite_state, "last_energy_update_s", 0.0)),
    )
    if abs(event_time - float(getattr(satellite_state, "last_energy_update_s", 0.0))) > 1e-9:
        raise _energy_value_error(
            "event_time_s_not_materialized",
            satellite_id=str(getattr(satellite_state, "satellite_id", "<unknown>")),
            time_s=event_time,
        )
    soc_before = float(getattr(satellite_state, "soc", 0.8))
    events = tuple(getattr(satellite_state, "energy_events", ()) or ())
    event = {
        "event_id": f"energy-event-{len(events)}",
        "sequence": len(events),
        "event_time_s": event_time,
        "compute_j": compute_j,
        "link_j": link_j,
        "compute_tokens": compute_tokens,
        "link_bytes_accounted": link_bytes_accounted,
        "compute_energy_j_per_token": compute_coefficient,
        "link_energy_j_per_byte": link_coefficient,
        "base_compute_energy_j_per_token": base_compute_coefficient,
        "base_link_energy_j_per_byte": base_link_coefficient,
        "runtime_compute_energy_scale": runtime_compute_energy_scale,
        "runtime_link_energy_scale": runtime_link_energy_scale,
        "metadata": dict(metadata or {}),
    }
    satellite_state.energy_events = (*events, event)
    soc_after, row = _apply_stored_event(satellite_state, event, energy_window, soc=soc_before)
    satellite_state.soc = soc_after
    satellite_state.energy_ledger = (*tuple(getattr(satellite_state, "energy_ledger", ()) or ()), row)
    satellite_state.materialized_energy_event_ids = (
        *tuple(getattr(satellite_state, "materialized_energy_event_ids", ()) or ()),
        str(event["event_id"]),
    )
    soc_min = float(energy_window["soc_min"])
    return {
        "soc_before": soc_before,
        "soc_after": soc_after,
        "energy_violation": soc_after < soc_min or row["unserved_energy_j"] > 0,
        "compute_j": compute_j,
        "link_j": link_j,
        "dynamic_energy_j": compute_j + link_j,
        "compute_tokens": compute_tokens,
        "link_bytes_accounted": link_bytes_accounted,
        "compute_energy_j_per_token": compute_coefficient,
        "link_energy_j_per_byte": link_coefficient,
        "base_compute_energy_j_per_token": base_compute_coefficient,
        "base_link_energy_j_per_byte": base_link_coefficient,
        "runtime_compute_energy_scale": runtime_compute_energy_scale,
        "runtime_link_energy_scale": runtime_link_energy_scale,
        "unserved_energy_j": row["unserved_energy_j"],
        "ledger_entry": row,
        "soc_min": soc_min,
    }


def _idle_charge_delta(
    satellite_state: Any,
    window: dict[str, Any],
    *,
    duration_s: float,
    soc: float,
    event_time_s: float,
) -> tuple[float, dict[str, Any]]:
    _validate_runtime_energy_window(
        window,
        satellite_id=str(getattr(satellite_state, "satellite_id", window.get("satellite_id", "<unknown>"))),
        time_s=float(window.get("start_s", getattr(satellite_state, "last_energy_update_s", 0.0))),
    )
    capacity_j = _positive_finite(window["battery_capacity_j"], "battery_capacity_j", satellite_id=str(satellite_state.satellite_id), time_s=event_time_s)
    charge_w = _nonnegative_finite(window["charge_rate_w"], "charge_rate_w", satellite_id=str(satellite_state.satellite_id), time_s=event_time_s)
    idle_w = _nonnegative_finite(window["idle_power_w"], "idle_power_w", satellite_id=str(satellite_state.satellite_id), time_s=event_time_s)
    charge_eff = _efficiency(window.get("battery_charge_efficiency", 1.0), "battery_charge_efficiency", satellite_id=str(satellite_state.satellite_id), time_s=event_time_s)
    discharge_eff = _efficiency(window.get("battery_discharge_efficiency", 1.0), "battery_discharge_efficiency", satellite_id=str(satellite_state.satellite_id), time_s=event_time_s)
    duration_s = max(0.0, float(duration_s))
    generated_j = charge_w * duration_s
    baseline_j = idle_w * duration_s
    net_bus_j = generated_j - baseline_j
    stored_before_j = soc * capacity_j
    if net_bus_j >= 0:
        max_accepted_bus_j = max(0.0, capacity_j - stored_before_j) / charge_eff
        charge_input_bus_j = min(net_bus_j, max_accepted_bus_j)
        chemical_charge_j = charge_input_bus_j * charge_eff
        stored_j = stored_before_j + chemical_charge_j
        curtailed_j = max(0.0, net_bus_j - charge_input_bus_j)
        discharge_output_bus_j = 0.0
        unserved_bus_j = 0.0
        conversion_loss_j = charge_input_bus_j - chemical_charge_j
    else:
        requested_discharge_bus_j = -net_bus_j
        requested_chemical_j = requested_discharge_bus_j / discharge_eff
        supplied_chemical_j = min(stored_before_j, requested_chemical_j)
        discharge_output_bus_j = supplied_chemical_j * discharge_eff
        stored_j = stored_before_j - supplied_chemical_j
        curtailed_j = 0.0
        charge_input_bus_j = 0.0
        unserved_bus_j = max(0.0, requested_discharge_bus_j - discharge_output_bus_j)
        conversion_loss_j = supplied_chemical_j - discharge_output_bus_j
    next_soc = stored_j / capacity_j
    return next_soc, {
        "event_id": f"baseline:{float(event_time_s):.9f}",
        "event_time_s": float(event_time_s),
        "satellite_id": str(satellite_state.satellite_id),
        "energy_kind": "baseline_solar",
        "duration_s": duration_s,
        "generated_bus_j": generated_j,
        "baseline_bus_j": baseline_j,
        "battery_charge_input_bus_j": charge_input_bus_j,
        "battery_discharge_output_bus_j": discharge_output_bus_j,
        "conversion_loss_j": conversion_loss_j,
        "battery_delta_j": stored_j - stored_before_j,
        "curtailed_energy_j": curtailed_j,
        "unserved_energy_j": unserved_bus_j,
        "unserved_bus_j": unserved_bus_j,
        "soc_before": soc,
        "soc_after": next_soc,
    }


def _apply_stored_event(
    satellite_state: Any,
    event: dict[str, Any],
    window: dict[str, Any],
    *,
    soc: float,
) -> tuple[float, dict[str, Any]]:
    event_time_s = float(event.get("event_time_s", 0.0))
    capacity_j = _positive_finite(window["battery_capacity_j"], "battery_capacity_j", satellite_id=str(satellite_state.satellite_id), time_s=event_time_s)
    discharge_eff = _efficiency(window.get("battery_discharge_efficiency", 1.0), "battery_discharge_efficiency", satellite_id=str(satellite_state.satellite_id), time_s=event_time_s)
    compute_j = _nonnegative_finite(event.get("compute_j", 0.0), "compute_j", satellite_id=str(satellite_state.satellite_id), time_s=event_time_s)
    link_j = _nonnegative_finite(event.get("link_j", 0.0), "link_j", satellite_id=str(satellite_state.satellite_id), time_s=event_time_s)
    requested_chemical_j = (compute_j + link_j) / discharge_eff
    available_j = soc * capacity_j
    supplied_j = min(available_j, requested_chemical_j)
    delivered_bus_j = supplied_j * discharge_eff
    unserved_bus_j = max(0.0, compute_j + link_j - delivered_bus_j)
    conversion_loss_j = max(0.0, supplied_j - delivered_bus_j)
    next_soc = (available_j - supplied_j) / capacity_j
    row = {
        **dict(event.get("metadata") or {}),
        "event_id": str(event.get("event_id")),
        "event_time_s": event_time_s,
        "satellite_id": str(satellite_state.satellite_id),
        "energy_kind": "request_event",
        "compute_bus_j": compute_j,
        "link_bus_j": link_j,
        "dynamic_bus_j": compute_j + link_j,
        "compute_tokens": _nonnegative_finite(
            event.get("compute_tokens", 0.0),
            "compute_tokens",
            satellite_id=str(satellite_state.satellite_id),
            time_s=event_time_s,
        ),
        "link_bytes_accounted": _nonnegative_finite(
            event.get("link_bytes_accounted", 0.0),
            "link_bytes_accounted",
            satellite_id=str(satellite_state.satellite_id),
            time_s=event_time_s,
        ),
        "compute_energy_j_per_token": _nonnegative_finite(
            event.get("compute_energy_j_per_token", 0.0),
            "compute_energy_j_per_token",
            satellite_id=str(satellite_state.satellite_id),
            time_s=event_time_s,
        ),
        "link_energy_j_per_byte": _nonnegative_finite(
            event.get("link_energy_j_per_byte", 0.0),
            "link_energy_j_per_byte",
            satellite_id=str(satellite_state.satellite_id),
            time_s=event_time_s,
        ),
        "base_compute_energy_j_per_token": _nonnegative_finite(
            event.get("base_compute_energy_j_per_token", event.get("compute_energy_j_per_token", 0.0)),
            "base_compute_energy_j_per_token",
            satellite_id=str(satellite_state.satellite_id),
            time_s=event_time_s,
        ),
        "base_link_energy_j_per_byte": _nonnegative_finite(
            event.get("base_link_energy_j_per_byte", event.get("link_energy_j_per_byte", 0.0)),
            "base_link_energy_j_per_byte",
            satellite_id=str(satellite_state.satellite_id),
            time_s=event_time_s,
        ),
        "runtime_compute_energy_scale": _nonnegative_finite(
            event.get("runtime_compute_energy_scale", 1.0),
            "runtime_compute_energy_scale",
            satellite_id=str(satellite_state.satellite_id),
            time_s=event_time_s,
        ),
        "runtime_link_energy_scale": _nonnegative_finite(
            event.get("runtime_link_energy_scale", 1.0),
            "runtime_link_energy_scale",
            satellite_id=str(satellite_state.satellite_id),
            time_s=event_time_s,
        ),
        "battery_discharge_j": supplied_j,
        "battery_discharge_output_bus_j": delivered_bus_j,
        "conversion_loss_j": conversion_loss_j,
        "unserved_energy_j": unserved_bus_j,
        "unserved_bus_j": unserved_bus_j,
        "curtailed_energy_j": 0.0,
        "soc_before": soc,
        "soc_after": next_soc,
        "soc_min": float(window["soc_min"]),
    }
    return next_soc, row


def _dynamic_energy_error_field(exc: Exception) -> str:
    """Preserve the shared calculator's field name in fail-closed warnings."""

    text = str(exc)
    for field_name in (
        "prompt_tokens",
        "output_tokens",
        "link_bytes",
        "compute_energy_j_per_token",
        "link_energy_j_per_byte",
    ):
        if field_name in text:
            return field_name
    return "dynamic_energy_inputs"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate_runtime_energy_window(window: dict[str, Any], *, satellite_id: str, time_s: float) -> None:
    missing = tuple(field for field in REQUIRED_RUNTIME_ENERGY_WINDOW_FIELDS if field not in window)
    if missing:
        reason = "energy_window_missing_field"
        raise EnergyWindowError(
            reason,
            satellite_id=str(satellite_id),
            time_s=float(time_s),
            sample_time_s=_optional_float(window.get("start_s")),
            warnings=(reason, *(f"missing:{field}" for field in missing)),
        )
    _positive_finite(window["battery_capacity_j"], "battery_capacity_j", satellite_id=satellite_id, time_s=time_s)
    for field_name in (
        "charge_rate_w",
        "idle_power_w",
        "compute_energy_j_per_token",
        "link_energy_j_per_byte",
    ):
        _nonnegative_finite(window[field_name], field_name, satellite_id=satellite_id, time_s=time_s)
    _finite_fraction(window["soc_min"], "soc_min", satellite_id=satellite_id, time_s=time_s)
    _efficiency(window.get("battery_charge_efficiency", 1.0), "battery_charge_efficiency", satellite_id=satellite_id, time_s=time_s)
    _efficiency(window.get("battery_discharge_efficiency", 1.0), "battery_discharge_efficiency", satellite_id=satellite_id, time_s=time_s)
    if str(window["energy_accounting_mode"]) != "event_incremental_v1":
        raise _energy_value_error(
            "energy_accounting_mode",
            satellite_id=satellite_id,
            time_s=time_s,
        )


def _energy_window_at_boundary(trace_bundle: Any, satellite_id: str, time_s: float) -> dict[str, Any]:
    try:
        return energy_window_at(trace_bundle, satellite_id, time_s)
    except EnergyWindowError:
        source = getattr(trace_bundle, "energy_windows", ()) or ()
        matches = [
            dict(window)
            for window in source
            if str(window.get("satellite_id")) == str(satellite_id)
            and abs(float(window.get("end_s", -1.0)) - float(time_s)) <= 1e-9
        ]
        if not matches:
            raise
        window = sorted(matches, key=lambda row: float(row.get("start_s", 0.0)))[-1]
        _validate_runtime_energy_window(window, satellite_id=satellite_id, time_s=time_s)
        return window


def _energy_value_error(
    field_name: str,
    *,
    satellite_id: str,
    time_s: float,
) -> EnergyWindowError:
    reason = "energy_window_invalid_field"
    return EnergyWindowError(
        reason,
        satellite_id=str(satellite_id),
        time_s=float(time_s),
        warnings=(reason, f"invalid:{field_name}"),
    )


def _positive_finite(value: Any, field_name: str, *, satellite_id: str, time_s: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise _energy_value_error(field_name, satellite_id=satellite_id, time_s=time_s) from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise _energy_value_error(field_name, satellite_id=satellite_id, time_s=time_s)
    return parsed


def _nonnegative_finite(value: Any, field_name: str, *, satellite_id: str, time_s: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise _energy_value_error(field_name, satellite_id=satellite_id, time_s=time_s) from None
    if not math.isfinite(parsed) or parsed < 0:
        raise _energy_value_error(field_name, satellite_id=satellite_id, time_s=time_s)
    return parsed


def _finite_fraction(value: Any, field_name: str, *, satellite_id: str, time_s: float) -> float:
    parsed = _nonnegative_finite(value, field_name, satellite_id=satellite_id, time_s=time_s)
    if parsed > 1:
        raise _energy_value_error(field_name, satellite_id=satellite_id, time_s=time_s)
    return parsed


def _efficiency(value: Any, field_name: str, *, satellite_id: str, time_s: float) -> float:
    parsed = _positive_finite(value, field_name, satellite_id=satellite_id, time_s=time_s)
    if parsed > 1:
        raise _energy_value_error(field_name, satellite_id=satellite_id, time_s=time_s)
    return parsed


def _as_energy_query_result(
    value: Any,
    *,
    unavailable_reason: str,
    satellite_id: str,
    time_s: float,
) -> dict[str, Any]:
    if isinstance(value, dict) and ("available" in value or "reason" in value or "value" in value):
        return {
            "available": bool(value.get("available", value.get("value") is not None)),
            "reason": value.get("reason"),
            "sample_time_s": value.get("sample_time_s", time_s),
            "stale": bool(value.get("stale", False)),
            "value": value.get("value"),
            "warnings": _string_tuple(value.get("warnings")),
        }
    if hasattr(value, "available"):
        return {
            "available": bool(getattr(value, "available")),
            "reason": getattr(value, "reason", None),
            "sample_time_s": getattr(value, "sample_time_s", time_s),
            "stale": bool(getattr(value, "stale", False)),
            "value": getattr(value, "value", None),
            "warnings": _string_tuple(getattr(value, "warnings", ())),
        }
    if value is None:
        return {
            "available": False,
            "reason": unavailable_reason,
            "sample_time_s": time_s,
            "stale": False,
            "value": None,
            "warnings": (unavailable_reason,),
        }
    return {
        "available": True,
        "reason": None,
        "sample_time_s": time_s,
        "stale": False,
        "value": dict(value),
        "warnings": _string_tuple(getattr(value, "warnings", ())),
    }


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "EnergyWindowError",
    "EnergyUpdateResult",
    "REQUIRED_RUNTIME_ENERGY_WINDOW_FIELDS",
    "advance_satellite_energy",
    "apply_request_energy",
    "energy_window_at",
    "query_energy_window_at",
]
