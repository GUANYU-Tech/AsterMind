"""Shared SGL access-window helpers.

These helpers keep SGL availability derived from continuous access windows while
leaving ISL/link-sample semantics to the routing and trace layers.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from ..common import to_jsonable


def active_sgl_access_window(
    access_windows: Iterable[dict[str, Any]],
    link_id: str,
    time_s: float,
) -> dict[str, Any] | None:
    """Return the active half-open SGL access window for ``link_id`` at ``time_s``."""

    candidates: list[dict[str, Any]] = []
    for window_value in access_windows:
        window = _dict(window_value)
        if str(window.get("sgl_link_id")) != str(link_id):
            continue
        start_s = _optional_float(window.get("start_s"))
        end_s = _optional_float(window.get("end_s"))
        if start_s is None or end_s is None:
            continue
        if start_s <= time_s < end_s:
            candidates.append(dict(window))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda window: (
            float(_first_present(window.get("propagation_delay_s"), window.get("propagation_delay_min_s"), 0.0)),
            str(window.get("satellite_id") or ""),
            str(window.get("window_id") or ""),
        ),
    )


def has_sgl_access_windows(access_windows: Iterable[dict[str, Any]], link_id: str) -> bool:
    """Return whether any access window exists for this SGL link."""

    return any(str(_dict(window).get("sgl_link_id")) == str(link_id) for window in access_windows)


def sgl_window_link_sample(
    access_windows: Iterable[dict[str, Any]],
    link_id: str,
    time_s: float,
    *,
    link: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a routing-compatible SGL link sample from an active access window."""

    link_value = _dict(link)
    if not _is_sgl_link(link_id, link_value):
        return None
    window = active_sgl_access_window(access_windows, link_id, time_s)
    if window is None:
        return None

    bandwidth_bps = _finite_float(
        _first_present(
            window.get("B_raw_bps"),
            window.get("bandwidth_bps"),
            link_value.get("B_raw_bps"),
            link_value.get("bandwidth_bps"),
            link_value.get("bandwidth_bps_nominal"),
        )
    )
    propagation_delay_s = float(
        _first_present(
            window.get("tau_prop_s"),
            window.get("propagation_delay_s"),
            window.get("propagation_delay_min_s"),
            link_value.get("tau_prop_s"),
            link_value.get("propagation_delay_s"),
            0.0,
        )
    )
    setup_delay_s = float(
        _first_present(
            window.get("tau_setup_s"),
            window.get("setup_delay_s"),
            link_value.get("tau_setup_s"),
            link_value.get("setup_delay_s"),
            0.0,
        )
    )
    terminal_count_raw = _first_present(
        window.get("terminal_count_per_satellite"),
        window.get("terminal_pool_capacity_per_satellite"),
        link_value.get("terminal_count_per_satellite"),
        link_value.get("terminal_pool_capacity_per_satellite"),
    )
    terminal_count = _finite_float(terminal_count_raw)
    generated_physical_input = _uses_generated_gateway_profile(window, link_value)
    if bandwidth_bps is None or bandwidth_bps <= 0.0:
        capacity_available = False
        unavailable_reason = "invalid_sgl_bandwidth"
    elif terminal_count_raw is not None and (terminal_count is None or terminal_count <= 0.0):
        capacity_available = False
        unavailable_reason = "invalid_sgl_terminal_count"
    elif terminal_count_raw is None and generated_physical_input:
        capacity_available = False
        unavailable_reason = "missing_sgl_terminal_count"
    else:
        capacity_available = True
        unavailable_reason = None

    value = dict(link_value)
    value.update(
        {
            "link_id": str(link_id),
            "kind": str(_first_present(link_value.get("kind"), "sgl")),
            "runtime_kind": "sgl",
            "endpoint_a": str(_first_present(link_value.get("endpoint_a"), window.get("ingress_id"), "")),
            "endpoint_b": str(_first_present(link_value.get("endpoint_b"), window.get("satellite_id"), "")),
            "time_s": time_s,
            "sample_time_s": time_s,
            "edge_available_external": capacity_available,
            "unavailable_reason": unavailable_reason,
            "bandwidth_bps": bandwidth_bps or 0.0,
            "B_raw_bps": bandwidth_bps or 0.0,
            "propagation_delay_s": propagation_delay_s,
            "tau_prop_s": propagation_delay_s,
            "setup_delay_s": setup_delay_s,
            "tau_setup_s": setup_delay_s,
            "access_window_id": window.get("window_id"),
            "access_window_start_s": window.get("start_s"),
            "access_window_end_s": window.get("end_s"),
            "availability_source": "access_window",
            "provenance": {
                **_mapping(link_value.get("provenance")),
                "availability_source": "access_window",
                "access_window_id": window.get("window_id"),
            },
        }
    )
    if terminal_count is not None:
        value["terminal_count_per_satellite"] = terminal_count
    return value


def _uses_generated_gateway_profile(window: dict[str, Any], link: dict[str, Any]) -> bool:
    for payload in (window, link, _mapping(window.get("provenance")), _mapping(link.get("provenance"))):
        if payload.get("physical_parameter_source") == "constellation.gateway_profile":
            return True
    return False


def _is_sgl_link(link_id: str, link: dict[str, Any]) -> bool:
    kind = str(_first_present(link.get("runtime_kind"), link.get("kind"), ""))
    return kind == "sgl" or str(link_id).startswith("sgl:")


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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    parsed = _optional_float(value)
    return parsed if parsed is not None and math.isfinite(parsed) else None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
