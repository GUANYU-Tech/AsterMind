"""Runtime trace contracts for OrbitServe simulation outputs.

The trace layer intentionally separates trace metadata from the runtime event
engine.  It supports the original idealized full-visibility trace for compute
queue checks and a deterministic surrogate access/link/energy trace for
trace-backed runtime analysis.  Neither trace is valid for paper main results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sgl_access_windows import has_sgl_access_windows, sgl_window_link_sample
from ..common import to_jsonable


IDEAL_FULL_VISIBILITY = "ideal_full_visibility"
SKELETON_TRACE = "skeleton"
SURROGATE_ORBIT_TRACE = "surrogate_orbit_trace"
CIRCULAR_GEOMETRY_TRACE = "circular_geometry_trace"
TRACE_BACKED_RUNTIME_SURROGATE_SCOPE = "trace_backed_runtime_surrogate"
SURROGATE_TRACE_SCHEMA = "orbitserve.surrogate_orbit_trace.v1"
SATELLITE_MOTION_TRACE_SCHEMA = "orbitserve.satellite_motion_trace.v1"

_LEGACY_TRACE_SCOPE_ALIASES = {
    "trace_backed_runtime_sanity": TRACE_BACKED_RUNTIME_SURROGATE_SCOPE,
}


_TRACE_PATH_KEYS = (
    "satellite_catalog_path",
    "ephemeris_trace_path",
    "access_trace_path",
    "link_trace_path",
    "energy_trace_path",
    "calibration_report_path",
)
_TRACE_SHA256_KEYS = (
    "satellite_catalog_sha256",
    "ephemeris_trace_sha256",
    "access_trace_sha256",
    "link_trace_sha256",
    "energy_trace_sha256",
    "calibration_report_sha256",
    "trace_manifest_sha256",
)
_MANIFEST_KNOWN_KEYS = {
    "schema_version",
    "trace_model",
    "model",
    "evaluated",
    "trace_id",
    "valid_for_paper_main_results",
    "scope",
    "warnings",
    "provenance",
    "time",
    "files",
    "sha256",
    "calibration",
    *_TRACE_PATH_KEYS,
    *_TRACE_SHA256_KEYS,
}


@dataclass(frozen=True)
class RuntimeTraceManifest:
    schema_version: str | None = None
    trace_model: str = SKELETON_TRACE
    evaluated: bool = False
    trace_id: str | None = None
    satellite_catalog_path: str | None = None
    ephemeris_trace_path: str | None = None
    access_trace_path: str | None = None
    link_trace_path: str | None = None
    energy_trace_path: str | None = None
    calibration_report_path: str | None = None
    satellite_catalog_sha256: str | None = None
    ephemeris_trace_sha256: str | None = None
    access_trace_sha256: str | None = None
    link_trace_sha256: str | None = None
    energy_trace_sha256: str | None = None
    calibration_report_sha256: str | None = None
    trace_manifest_sha256: str | None = None
    time: dict[str, Any] = field(default_factory=dict)
    files: dict[str, Any] = field(default_factory=dict)
    sha256: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    valid_for_paper_main_results: bool = False
    scope: str = "not_evaluated"
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any = None) -> "RuntimeTraceManifest":
        payload = _mapping(value)
        if not payload:
            return cls(warnings=("simulation trace manifest is not configured",))
        manifest = _mapping(payload.get("trace_manifest")) or payload
        files = _mapping(manifest.get("files"))
        sha256 = _mapping(manifest.get("sha256"))
        time = _mapping(manifest.get("time"))
        calibration = _mapping(manifest.get("calibration"))
        trace_model = str(manifest.get("trace_model") or manifest.get("model") or SKELETON_TRACE)
        paths = {
            key: _optional_str(_first_present(manifest.get(key), files.get(key)))
            for key in _TRACE_PATH_KEYS
        }
        hashes = {
            key: _optional_str(_first_present(manifest.get(key), sha256.get(key)))
            for key in _TRACE_SHA256_KEYS
        }
        normalized_files = dict(files)
        for key, item in paths.items():
            if item is not None:
                normalized_files.setdefault(key, item)
        normalized_sha256 = dict(sha256)
        for key, item in hashes.items():
            if item is not None:
                normalized_sha256.setdefault(key, item)
        return cls(
            schema_version=_optional_str(manifest.get("schema_version")),
            trace_model=trace_model,
            evaluated=bool(manifest.get("evaluated", payload.get("evaluated", False))),
            trace_id=_optional_str(manifest.get("trace_id")),
            satellite_catalog_path=paths["satellite_catalog_path"],
            ephemeris_trace_path=paths["ephemeris_trace_path"],
            access_trace_path=paths["access_trace_path"],
            link_trace_path=paths["link_trace_path"],
            energy_trace_path=paths["energy_trace_path"],
            calibration_report_path=paths["calibration_report_path"],
            satellite_catalog_sha256=hashes["satellite_catalog_sha256"],
            ephemeris_trace_sha256=hashes["ephemeris_trace_sha256"],
            access_trace_sha256=hashes["access_trace_sha256"],
            link_trace_sha256=hashes["link_trace_sha256"],
            energy_trace_sha256=hashes["energy_trace_sha256"],
            calibration_report_sha256=hashes["calibration_report_sha256"],
            trace_manifest_sha256=hashes["trace_manifest_sha256"],
            time=time,
            files=normalized_files,
            sha256=normalized_sha256,
            calibration=calibration,
            valid_for_paper_main_results=bool(manifest.get("valid_for_paper_main_results", False)),
            scope=normalize_trace_scope(str(manifest.get("scope") or _scope_for_trace_model(trace_model))),
            warnings=_string_tuple(manifest.get("warnings") or payload.get("warnings")),
            provenance=_mapping(manifest.get("provenance")),
            extra={key: item for key, item in manifest.items() if key not in _MANIFEST_KNOWN_KEYS},
        )

    @classmethod
    def ideal_full_visibility(cls, *, trace_id: str | None = None) -> "RuntimeTraceManifest":
        return cls(
            trace_model=IDEAL_FULL_VISIBILITY,
            evaluated=True,
            trace_id=trace_id or IDEAL_FULL_VISIBILITY,
            valid_for_paper_main_results=False,
            scope="idealized_compute_only",
            warnings=(
                "ideal_full_visibility is a compute-queue-only trace, not an orbit-aware trace",
                "valid_for_paper_main_results=false",
            ),
            provenance={"source": "orbitserve.simulation_traces.RuntimeTraceManifest.ideal_full_visibility"},
        )

    def to_dict(self) -> dict[str, Any]:
        files = dict(self.files)
        for key in _TRACE_PATH_KEYS:
            item = getattr(self, key)
            if item is not None:
                files.setdefault(key, item)
        sha256 = dict(self.sha256)
        for key in _TRACE_SHA256_KEYS:
            item = getattr(self, key)
            if item is not None:
                sha256.setdefault(key, item)
        payload: dict[str, Any] = dict(self.extra)
        payload.update(
            {
                "schema_version": self.schema_version,
                "trace_model": self.trace_model,
                "evaluated": self.evaluated,
                "trace_id": self.trace_id,
                "satellite_catalog_path": self.satellite_catalog_path,
                "ephemeris_trace_path": self.ephemeris_trace_path,
                "access_trace_path": self.access_trace_path,
                "link_trace_path": self.link_trace_path,
                "energy_trace_path": self.energy_trace_path,
                "calibration_report_path": self.calibration_report_path,
                "satellite_catalog_sha256": self.satellite_catalog_sha256,
                "ephemeris_trace_sha256": self.ephemeris_trace_sha256,
                "access_trace_sha256": self.access_trace_sha256,
                "link_trace_sha256": self.link_trace_sha256,
                "energy_trace_sha256": self.energy_trace_sha256,
                "calibration_report_sha256": self.calibration_report_sha256,
                "trace_manifest_sha256": self.trace_manifest_sha256,
                "time": self.time,
                "files": files,
                "sha256": sha256,
                "calibration": self.calibration,
                "valid_for_paper_main_results": self.valid_for_paper_main_results,
                "scope": self.scope,
                "warnings": self.warnings,
                "provenance": self.provenance,
            }
        )
        return to_jsonable(payload)


@dataclass(frozen=True)
class TraceQueryResult:
    available: bool
    reason: str | None = None
    sample_time_s: float | None = None
    stale: bool = False
    value: Any = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class RuntimeTraceBundle:
    trace_id: str
    access_trace: dict[str, Any] = field(default_factory=dict)
    link_trace: dict[str, Any] = field(default_factory=dict)
    energy_trace: dict[str, Any] = field(default_factory=dict)
    satellite_catalog: tuple[dict[str, Any], ...] = ()
    access_windows: tuple[dict[str, Any], ...] = ()
    visible_count_samples: tuple[dict[str, Any], ...] = ()
    physical_links: tuple[dict[str, Any], ...] = ()
    links: tuple[dict[str, Any], ...] = ()
    link_samples: tuple[dict[str, Any], ...] = ()
    active_graph_snapshots: tuple[dict[str, Any], ...] = ()
    terminal_catalog: tuple[dict[str, Any], ...] = ()
    energy_windows: tuple[dict[str, Any], ...] = ()
    shadow_windows: tuple[dict[str, Any], ...] = ()
    solar_samples: tuple[dict[str, Any], ...] = ()
    ephemeris_samples: tuple[dict[str, Any], ...] = ()
    satellite_initial_soc: dict[str, float] = field(default_factory=dict)
    trace_manifest: RuntimeTraceManifest | dict[str, Any] | None = None
    horizon_s: float | None = None
    step_s: float | None = None
    boundary_tolerance_s: float = 1e-9
    trace_model: str = SURROGATE_ORBIT_TRACE
    scope: str = TRACE_BACKED_RUNTIME_SURROGATE_SCOPE
    warnings: tuple[str, ...] = ()
    time_warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def _allow_static_link_compat(self) -> bool:
        cached = self.__dict__.get("_allow_static_link_compat_cache")
        if isinstance(cached, bool):
            return cached
        manifest = _manifest_mapping(self.trace_manifest)
        schema_version = str(manifest.get("schema_version") or "")
        manifest_model = str(manifest.get("trace_model") or manifest.get("model") or self.trace_model)
        allowed = (
            self.trace_model == SURROGATE_ORBIT_TRACE
            and manifest_model == SURROGATE_ORBIT_TRACE
            and schema_version != SATELLITE_MOTION_TRACE_SCHEMA
            and not self.active_graph_snapshots
            and not self.link_samples
        )
        object.__setattr__(self, "_allow_static_link_compat_cache", allowed)
        return allowed

    def _link_by_id(self, link_id: str) -> dict[str, Any] | None:
        cached = self.__dict__.get("_link_by_id_cache")
        if not isinstance(cached, dict):
            cached = {}
            for link in (*self.physical_links, *self.links):
                if link.get("link_id") is not None:
                    cached[str(link.get("link_id"))] = dict(link)
            object.__setattr__(self, "_link_by_id_cache", cached)
        selected = cached.get(str(link_id))
        return dict(selected) if selected is not None else None

    def access_at(self, ingress_id: str, time_s: float) -> TraceQueryResult:
        time_value, unavailable = self._guard_query_time(time_s)
        if unavailable is not None:
            return unavailable
        windows = tuple(
            sorted(
                (
                    dict(window)
                    for window in self.access_windows
                    if str(window.get("ingress_id")) == str(ingress_id)
                    and _float_value(window.get("start_s"), 0.0) <= time_value
                    and time_value < _float_value(window.get("end_s"), 0.0)
                ),
                key=lambda item: (
                    _float_value(item.get("propagation_delay_s"), 0.0),
                    str(item.get("satellite_id")),
                    str(item.get("window_id")),
                ),
            )
        )
        if not windows:
            return TraceQueryResult(
                available=False,
                reason="no_access_window",
                sample_time_s=time_value,
                warnings=self.time_warnings,
            )
        return TraceQueryResult(
            available=True,
            sample_time_s=time_value,
            value=windows,
            warnings=self.time_warnings,
        )

    def next_access_after(self, ingress_id: str, time_s: float) -> TraceQueryResult:
        time_value, unavailable = self._guard_query_time(time_s)
        if unavailable is not None:
            return unavailable
        candidates = [
            dict(window)
            for window in self.access_windows
            if str(window.get("ingress_id")) == str(ingress_id)
            and _float_value(window.get("end_s"), 0.0) > time_value
        ]
        if not candidates:
            return TraceQueryResult(
                available=False,
                reason="no_next_access_window",
                sample_time_s=time_value,
                warnings=self.time_warnings,
            )
        selected = min(
            candidates,
            key=lambda item: (
                max(time_value, _float_value(item.get("start_s"), 0.0)),
                _float_value(item.get("propagation_delay_s"), 0.0),
                str(item.get("satellite_id")),
                str(item.get("window_id")),
            ),
        )
        return TraceQueryResult(
            available=True,
            sample_time_s=max(time_value, _float_value(selected.get("start_s"), 0.0)),
            value=selected,
            warnings=self.time_warnings,
        )

    def active_links_at(self, time_s: float) -> TraceQueryResult:
        time_value, unavailable = self._guard_query_time(time_s)
        if unavailable is not None:
            return unavailable
        cached_result = self._cached_active_links_at(time_value)
        if cached_result is not None:
            return cached_result
        if self.active_graph_snapshots:
            sample_result = self._sample_at(self.active_graph_snapshots, time_value, "active_graph")
            if not sample_result.available:
                return self._store_active_links_at(time_value, sample_result)
            snapshot = _dict(sample_result.value)
            active_link_ids = tuple(str(item) for item in snapshot.get("active_link_ids", ()) or ())
        elif self.link_samples:
            sample_result = self._sample_at(self.link_samples, time_value, "link")
            if not sample_result.available:
                return self._store_active_links_at(time_value, sample_result)
            sample_time = sample_result.sample_time_s
            active_link_ids = tuple(
                sorted(
                    {
                        str(sample.get("link_id"))
                        for sample in self.link_samples
                        if _same_time(sample.get("time_s"), sample_time)
                        and bool(sample.get("edge_available_external", False))
                    }
                )
            )
            snapshot = {
                "time_s": sample_time,
                "active_link_ids": active_link_ids,
                "activation_policy": "derived_from_link_samples",
            }
        else:
            if not self._allow_static_link_compat():
                return TraceQueryResult(
                    available=False,
                    reason="active_graph_unavailable",
                    sample_time_s=time_value,
                    warnings=(*self.time_warnings, "active_graph_unavailable"),
                )
            static_value = self.__dict__.get("_static_active_links_value")
            if not isinstance(static_value, dict):
                static_links = tuple(dict(link) for link in self.links if bool(link.get("edge_available_external", True)))
                active_link_ids = tuple(str(link.get("link_id")) for link in static_links)
                static_value = {
                    "active_link_ids": active_link_ids,
                    "active_link_id_set": frozenset(active_link_ids),
                    "links": static_links,
                    "snapshot": {"activation_policy": "static_compat_links"},
                }
                object.__setattr__(self, "_static_active_links_value", static_value)
            value = {
                **static_value,
                "snapshot": {**_dict(static_value.get("snapshot")), "time_s": time_value},
            }
            return self._store_active_links_at(time_value, TraceQueryResult(
                available=True,
                sample_time_s=time_value,
                value=value,
                warnings=self.time_warnings,
            ))
        links_by_id = {str(link.get("link_id")): dict(link) for link in (*self.physical_links, *self.links)}
        active_links = tuple(links_by_id.get(link_id, {"link_id": link_id}) for link_id in active_link_ids)
        value = {
            "active_link_ids": active_link_ids,
            "active_link_id_set": frozenset(active_link_ids),
            "links": active_links,
            "snapshot": snapshot,
        }
        return self._store_active_links_at(time_value, TraceQueryResult(
            available=True,
            sample_time_s=sample_result.sample_time_s,
            stale=sample_result.stale,
            value=value,
            warnings=sample_result.warnings,
        ))

    def _cached_active_links_at(self, time_s: float) -> TraceQueryResult | None:
        cached = self.__dict__.get("_active_links_at_cache")
        if not isinstance(cached, dict):
            return None
        result = cached.get(float(time_s))
        return result if isinstance(result, TraceQueryResult) else None

    def _store_active_links_at(self, time_s: float, result: TraceQueryResult) -> TraceQueryResult:
        cached = self.__dict__.get("_active_links_at_cache")
        if not isinstance(cached, dict):
            cached = {}
            object.__setattr__(self, "_active_links_at_cache", cached)
        cached[float(time_s)] = result
        return result

    def link_sample_at(self, link_id: str, time_s: float) -> TraceQueryResult:
        time_value, unavailable = self._guard_query_time(time_s)
        if unavailable is not None:
            return unavailable
        sgl_windows = self._access_windows_by_sgl_link_id().get(str(link_id), ())
        if sgl_windows:
            sgl_window_sample = sgl_window_link_sample(
                sgl_windows,
                link_id,
                time_value,
                link=self._link_by_id(link_id),
            )
            if sgl_window_sample is not None:
                if not bool(sgl_window_sample.get("edge_available_external", False)):
                    return TraceQueryResult(
                        available=False,
                        reason=_optional_str(sgl_window_sample.get("unavailable_reason"))
                        or "invalid_sgl_capacity",
                        sample_time_s=time_value,
                        value=sgl_window_sample,
                        warnings=self.time_warnings,
                    )
                return TraceQueryResult(
                    available=True,
                    sample_time_s=time_value,
                    value=sgl_window_sample,
                    warnings=self.time_warnings,
                )
            return TraceQueryResult(
                available=False,
                reason="no_sgl_access_window",
                sample_time_s=time_value,
                warnings=(*self.time_warnings, "no_sgl_access_window"),
            )
        samples = self._link_samples_by_id().get(str(link_id), ())
        if not samples:
            static_link_allowed = (not self.link_samples) and (
                bool(self.active_graph_snapshots) or self._allow_static_link_compat()
            )
            if not static_link_allowed:
                return TraceQueryResult(
                    available=False,
                    reason="link_sample_missing",
                    sample_time_s=time_value,
                    warnings=(*self.time_warnings, "link_sample_missing"),
                )
            if self.active_graph_snapshots:
                active = self.active_links_at(time_value)
                if not active.available:
                    return TraceQueryResult(
                        available=False,
                        reason=active.reason or "active_graph_unavailable",
                        sample_time_s=active.sample_time_s,
                        warnings=active.warnings,
                    )
                active_ids = _active_link_id_set(active.value)
                if active_ids and str(link_id) not in active_ids:
                    return TraceQueryResult(
                        available=False,
                        reason="link_inactive_at_sample",
                        sample_time_s=active.sample_time_s,
                        warnings=active.warnings,
                    )
            static_link = self._link_by_id(link_id)
            if static_link is not None:
                if not bool(static_link.get("edge_available_external", True)):
                    return TraceQueryResult(
                        available=False,
                        reason=_optional_str(static_link.get("unavailable_reason")) or "link_unavailable",
                        sample_time_s=time_value,
                        value=static_link,
                        warnings=self.time_warnings,
                    )
                return TraceQueryResult(
                    available=True,
                    sample_time_s=time_value,
                    value=static_link,
                    warnings=self.time_warnings,
                )
            return TraceQueryResult(
                available=False,
                reason="link_sample_missing",
                sample_time_s=time_value,
                warnings=self.time_warnings,
            )
        result = self._sample_at(samples, time_value, "link")
        if not result.available:
            return result
        sample = dict(result.value)
        if not bool(sample.get("edge_available_external", True)):
            return TraceQueryResult(
                available=False,
                reason=_optional_str(sample.get("unavailable_reason")) or "link_unavailable",
                sample_time_s=result.sample_time_s,
                value=sample,
                warnings=result.warnings,
            )
        return TraceQueryResult(
            available=True,
            sample_time_s=result.sample_time_s,
            value=sample,
            warnings=result.warnings,
        )

    def _access_windows_by_sgl_link_id(self) -> dict[str, tuple[dict[str, Any], ...]]:
        cached = self.__dict__.get("_access_windows_by_sgl_link_id_cache")
        if isinstance(cached, dict):
            return cached
        grouped: dict[str, list[dict[str, Any]]] = {}
        for window in self.access_windows:
            link_id = window.get("sgl_link_id")
            if link_id is not None:
                grouped.setdefault(str(link_id), []).append(dict(window))
        indexed = {
            link_id: tuple(
                sorted(
                    windows,
                    key=lambda item: (
                        _float_value(item.get("start_s"), 0.0),
                        _float_value(item.get("end_s"), 0.0),
                        str(item.get("window_id") or ""),
                    ),
                )
            )
            for link_id, windows in grouped.items()
        }
        object.__setattr__(self, "_access_windows_by_sgl_link_id_cache", indexed)
        return indexed

    def _link_samples_by_id(self) -> dict[str, tuple[dict[str, Any], ...]]:
        cached = self.__dict__.get("_link_samples_by_id_cache")
        if isinstance(cached, dict):
            return cached
        grouped: dict[str, list[dict[str, Any]]] = {}
        for sample in self.link_samples:
            link_id = sample.get("link_id")
            if link_id is not None:
                grouped.setdefault(str(link_id), []).append(dict(sample))
        indexed = {
            link_id: tuple(
                sorted(
                    samples,
                    key=lambda item: (
                        _float_value(item.get("time_s"), 0.0),
                        str(item.get("link_id") or ""),
                    ),
                )
            )
            for link_id, samples in grouped.items()
        }
        object.__setattr__(self, "_link_samples_by_id_cache", indexed)
        return indexed

    def energy_window_at(self, satellite_id: str, time_s: float) -> TraceQueryResult:
        time_value, unavailable = self._guard_query_time(time_s)
        if unavailable is not None:
            return unavailable
        candidates = tuple(
            dict(window)
            for window in self._energy_windows_by_satellite().get(str(satellite_id), ())
            if _float_value(window.get("start_s"), 0.0) <= time_value
            and time_value < _float_value(window.get("end_s"), 0.0)
        )
        if not candidates:
            return TraceQueryResult(
                available=False,
                reason="no_energy_window",
                sample_time_s=time_value,
                warnings=self.time_warnings,
            )
        return TraceQueryResult(
            available=True,
            sample_time_s=time_value,
            value=candidates[0],
            warnings=self.time_warnings,
        )

    def _energy_windows_by_satellite(self) -> dict[str, tuple[dict[str, Any], ...]]:
        cached = self.__dict__.get("_energy_windows_by_satellite_cache")
        if isinstance(cached, dict):
            return cached
        grouped: dict[str, list[dict[str, Any]]] = {}
        for window in self.energy_windows:
            grouped.setdefault(str(window.get("satellite_id")), []).append(dict(window))
        indexed = {
            satellite_id: tuple(
                sorted(
                    windows,
                    key=lambda item: (
                        _float_value(item.get("start_s"), 0.0),
                        _float_value(item.get("end_s"), 0.0),
                    ),
                )
            )
            for satellite_id, windows in grouped.items()
        }
        object.__setattr__(self, "_energy_windows_by_satellite_cache", indexed)
        return indexed

    def ephemeris_at(self, satellite_id: str, time_s: float) -> TraceQueryResult:
        time_value, unavailable = self._guard_query_time(time_s)
        if unavailable is not None:
            return unavailable
        samples = tuple(
            sample for sample in self.ephemeris_samples if str(sample.get("satellite_id")) == str(satellite_id)
        )
        if not samples:
            return TraceQueryResult(
                available=False,
                reason="ephemeris_trace_missing",
                sample_time_s=time_value,
                warnings=self.time_warnings,
            )
        return self._sample_at(samples, time_value, "ephemeris")

    def _guard_query_time(self, time_s: float) -> tuple[float, TraceQueryResult | None]:
        time_value = _float_value(time_s, float("nan"))
        if time_value != time_value:
            return time_value, TraceQueryResult(
                available=False,
                reason="invalid_query_time",
                warnings=self.time_warnings,
            )
        horizon_s, _, _, warnings = self._time_authority()
        if horizon_s is None:
            return time_value, TraceQueryResult(
                available=False,
                reason="trace_horizon_missing",
                sample_time_s=time_value,
                warnings=warnings,
            )
        if time_value < 0.0 or time_value >= horizon_s:
            return time_value, TraceQueryResult(
                available=False,
                reason="request_time_outside_trace_horizon",
                sample_time_s=time_value,
                warnings=warnings,
            )
        return time_value, None

    def _sample_at(
        self,
        samples: tuple[dict[str, Any], ...],
        time_s: float,
        kind: str,
    ) -> TraceQueryResult:
        horizon_s, step_s, tolerance_s, warnings = self._time_authority()
        del horizon_s
        if step_s is None or step_s <= 0.0:
            return TraceQueryResult(
                available=False,
                reason="trace_step_missing",
                sample_time_s=time_s,
                warnings=warnings,
            )
        eligible = [
            dict(sample)
            for sample in samples
            if _optional_float(sample.get("time_s")) is not None
            and float(sample["time_s"]) <= time_s + tolerance_s
        ]
        if not eligible:
            return TraceQueryResult(
                available=False,
                reason=f"trace_sample_missing:{kind}",
                sample_time_s=time_s,
                warnings=warnings,
            )
        selected = max(
            eligible,
            key=lambda sample: (
                _float_value(sample.get("time_s"), 0.0),
                str(sample.get("link_id") or sample.get("satellite_id") or ""),
            ),
        )
        sample_time_s = _float_value(selected.get("time_s"), 0.0)
        if time_s - sample_time_s > step_s + tolerance_s:
            reason = f"stale_trace_sample:{kind}"
            return TraceQueryResult(
                available=False,
                reason=reason,
                sample_time_s=sample_time_s,
                stale=True,
                value=selected,
                warnings=(*warnings, reason),
            )
        return TraceQueryResult(
            available=True,
            sample_time_s=sample_time_s,
            value=selected,
            warnings=warnings,
        )

    def _time_authority(self) -> tuple[float | None, float | None, float, tuple[str, ...]]:
        if self.horizon_s is not None or self.step_s is not None:
            return (
                self.horizon_s,
                self.step_s,
                self.boundary_tolerance_s,
                self.time_warnings,
            )
        return _infer_time_authority(
            trace_manifest=self.trace_manifest,
            access_windows=self.access_windows,
            visible_count_samples=self.visible_count_samples,
            link_samples=self.link_samples,
            active_graph_snapshots=self.active_graph_snapshots,
            energy_windows=self.energy_windows,
            shadow_windows=self.shadow_windows,
            solar_samples=self.solar_samples,
            ephemeris_samples=self.ephemeris_samples,
        )


SurrogateTraceBundle = RuntimeTraceBundle


def trace_file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    trace_path = Path(path)
    if not trace_path.exists() or not trace_path.is_file():
        return None
    digest = hashlib.sha256()
    with trace_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_trace_path(
    path: str | Path | None,
    *,
    manifest_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> Path | None:
    if path is None:
        return None
    trace_path = Path(path).expanduser()
    if trace_path.is_absolute():
        return trace_path
    for root in (manifest_dir, base_dir, Path.cwd()):
        if root is None:
            continue
        return (Path(root).expanduser() / trace_path).resolve()
    return trace_path.resolve()


def normalize_trace_scope(scope: str | None) -> str:
    if scope is None:
        return "not_evaluated"
    scope_text = str(scope)
    return _LEGACY_TRACE_SCOPE_ALIASES.get(scope_text, scope_text)


def load_surrogate_trace_bundle(
    trace_manifest: RuntimeTraceManifest | dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> RuntimeTraceBundle:
    manifest = (
        trace_manifest
        if isinstance(trace_manifest, RuntimeTraceManifest)
        else RuntimeTraceManifest.from_value(trace_manifest)
    )
    if manifest.trace_model not in {SURROGATE_ORBIT_TRACE, CIRCULAR_GEOMETRY_TRACE}:
        raise ValueError(f"unsupported runtime trace_model: {manifest.trace_model!r}")
    trace_dir = _optional_str(manifest.provenance.get("trace_dir"))
    satellite_catalog_path = resolve_trace_path(
        manifest.satellite_catalog_path,
        manifest_dir=trace_dir,
        base_dir=base_dir,
    )
    ephemeris_path = resolve_trace_path(
        manifest.ephemeris_trace_path,
        manifest_dir=trace_dir,
        base_dir=base_dir,
    )
    access_path = resolve_trace_path(manifest.access_trace_path, manifest_dir=trace_dir, base_dir=base_dir)
    link_path = resolve_trace_path(manifest.link_trace_path, manifest_dir=trace_dir, base_dir=base_dir)
    energy_path = resolve_trace_path(manifest.energy_trace_path, manifest_dir=trace_dir, base_dir=base_dir)
    calibration_path = resolve_trace_path(manifest.calibration_report_path, manifest_dir=trace_dir, base_dir=base_dir)
    for label, trace_path, expected_sha in (
        ("satellite_catalog", satellite_catalog_path, manifest.satellite_catalog_sha256),
        ("ephemeris_trace", ephemeris_path, manifest.ephemeris_trace_sha256),
        ("access_trace", access_path, manifest.access_trace_sha256),
        ("link_trace", link_path, manifest.link_trace_sha256),
        ("energy_trace", energy_path, manifest.energy_trace_sha256),
        ("calibration_report", calibration_path, manifest.calibration_report_sha256),
    ):
        if trace_path is None and label in {"satellite_catalog", "ephemeris_trace", "calibration_report"}:
            continue
        if trace_path is None or not trace_path.exists():
            raise ValueError(f"missing surrogate trace file: {label}")
        actual_sha = trace_file_sha256(trace_path)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(f"surrogate trace sha256 mismatch for {label}")

    access_trace = _read_json(access_path)
    link_trace = _read_json(link_path)
    energy_trace = _read_json(energy_path)
    satellite_catalog_value = _read_json_value(satellite_catalog_path) if satellite_catalog_path is not None else None
    satellite_catalog_source = (
        satellite_catalog_value
        if isinstance(satellite_catalog_value, list)
        else access_trace.get("satellite_catalog", ())
    )
    satellite_catalog = tuple(_dict(item) for item in satellite_catalog_source if isinstance(item, dict))
    access_windows = tuple(_dict(item) for item in access_trace.get("access_windows", ()))
    visible_count_samples = tuple(_dict(item) for item in access_trace.get("visible_count_samples", ()))
    physical_links = tuple(_dict(item) for item in link_trace.get("physical_links", ()))
    links = tuple(_dict(item) for item in (link_trace.get("links") or link_trace.get("physical_links") or ()))
    link_samples = tuple(_dict(item) for item in link_trace.get("link_samples", ()))
    active_graph_snapshots = tuple(_dict(item) for item in link_trace.get("active_graph_snapshots", ()))
    terminal_catalog = tuple(_dict(item) for item in link_trace.get("terminal_catalog", ()))
    energy_windows = tuple(_dict(item) for item in energy_trace.get("energy_windows", ()))
    shadow_windows = tuple(_dict(item) for item in energy_trace.get("shadow_windows", ()))
    solar_samples = tuple(_dict(item) for item in energy_trace.get("solar_samples", ()))
    ephemeris_samples = _read_jsonl(ephemeris_path) if ephemeris_path is not None else ()
    initial_soc = {
        str(key): float(value)
        for key, value in _dict(energy_trace.get("satellite_initial_soc")).items()
    }
    horizon_s, step_s, boundary_tolerance_s, time_warnings = _infer_time_authority(
        trace_manifest=manifest,
        access_windows=access_windows,
        visible_count_samples=visible_count_samples,
        link_samples=link_samples,
        active_graph_snapshots=active_graph_snapshots,
        energy_windows=energy_windows,
        shadow_windows=shadow_windows,
        solar_samples=solar_samples,
        ephemeris_samples=ephemeris_samples,
    )
    warnings = tuple(
        dict.fromkeys(
            (
                *manifest.warnings,
                *_string_tuple(access_trace.get("warnings")),
                *_string_tuple(link_trace.get("warnings")),
                *_string_tuple(energy_trace.get("warnings")),
                *time_warnings,
            )
        )
    )
    provenance = {
        **manifest.provenance,
        **({"satellite_catalog_path": str(satellite_catalog_path)} if satellite_catalog_path is not None else {}),
        **({"ephemeris_trace_path": str(ephemeris_path)} if ephemeris_path is not None else {}),
        "access_trace_path": str(access_path),
        "link_trace_path": str(link_path),
        "energy_trace_path": str(energy_path),
        **({"calibration_report_path": str(calibration_path)} if calibration_path is not None else {}),
    }
    return RuntimeTraceBundle(
        trace_id=manifest.trace_id or str(access_trace.get("trace_id") or SURROGATE_ORBIT_TRACE),
        access_trace=access_trace,
        link_trace=link_trace,
        energy_trace=energy_trace,
        satellite_catalog=satellite_catalog,
        access_windows=access_windows,
        visible_count_samples=visible_count_samples,
        physical_links=physical_links,
        links=links,
        link_samples=link_samples,
        active_graph_snapshots=active_graph_snapshots,
        terminal_catalog=terminal_catalog,
        energy_windows=energy_windows,
        shadow_windows=shadow_windows,
        solar_samples=solar_samples,
        ephemeris_samples=ephemeris_samples,
        satellite_initial_soc=initial_soc,
        trace_manifest=manifest,
        horizon_s=horizon_s,
        step_s=step_s,
        boundary_tolerance_s=boundary_tolerance_s,
        trace_model=manifest.trace_model,
        scope=manifest.scope,
        warnings=warnings,
        time_warnings=time_warnings,
        provenance=provenance,
    )


def _scope_for_trace_model(trace_model: str) -> str:
    if trace_model == IDEAL_FULL_VISIBILITY:
        return "idealized_compute_only"
    if trace_model == SURROGATE_ORBIT_TRACE:
        return TRACE_BACKED_RUNTIME_SURROGATE_SCOPE
    if trace_model == CIRCULAR_GEOMETRY_TRACE:
        return "geometry_trace_surrogate"
    if trace_model == SKELETON_TRACE:
        return "not_evaluated"
    return "trace_backed_runtime"


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = _read_json_value(path)
    return payload if isinstance(payload, dict) else {}


def _read_json_value(path: Path | None) -> Any:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path | None) -> tuple[dict[str, Any], ...]:
    if path is None:
        return ()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(_dict(item))
    return tuple(rows)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _float_value(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _same_time(value: Any, expected: float | None, *, tolerance_s: float = 1e-9) -> bool:
    if expected is None:
        return False
    parsed = _optional_float(value)
    return parsed is not None and abs(parsed - expected) <= tolerance_s


def _active_link_id_set(value: Any) -> frozenset[str]:
    if isinstance(value, dict):
        cached = value.get("active_link_id_set")
        if isinstance(cached, frozenset):
            return cached
        if isinstance(cached, set):
            return frozenset(str(item) for item in cached)
        if value.get("active_link_ids") is not None:
            return frozenset(str(item) for item in value.get("active_link_ids") or ())
        if value.get("links") is not None:
            return frozenset(str(_dict(item).get("link_id")) for item in value.get("links") or ())
    if isinstance(value, (list, tuple)):
        return frozenset(str(item if isinstance(item, str) else _dict(item).get("link_id")) for item in value)
    return frozenset()


def _infer_time_authority(
    *,
    trace_manifest: RuntimeTraceManifest | dict[str, Any] | None,
    access_windows: tuple[dict[str, Any], ...],
    visible_count_samples: tuple[dict[str, Any], ...],
    link_samples: tuple[dict[str, Any], ...],
    active_graph_snapshots: tuple[dict[str, Any], ...],
    energy_windows: tuple[dict[str, Any], ...],
    shadow_windows: tuple[dict[str, Any], ...],
    solar_samples: tuple[dict[str, Any], ...],
    ephemeris_samples: tuple[dict[str, Any], ...],
) -> tuple[float | None, float | None, float, tuple[str, ...]]:
    manifest = _manifest_mapping(trace_manifest)
    time = _mapping(manifest.get("time"))
    horizon_s = _optional_float(time.get("duration_s"))
    step_s = _optional_float(time.get("step_s"))
    boundary_tolerance_s = _optional_float(time.get("boundary_tolerance_s"))
    if boundary_tolerance_s is None:
        boundary_tolerance_s = 1e-9

    sample_times = _sample_times(
        visible_count_samples,
        link_samples,
        active_graph_snapshots,
        solar_samples,
        ephemeris_samples,
    )
    if step_s is None:
        step_s = _infer_step_s(sample_times)
    warnings: tuple[str, ...] = ()
    if horizon_s is None:
        if sample_times and step_s is not None:
            horizon_s = max(sample_times) + step_s
        else:
            horizon_s = _max_window_end(access_windows, energy_windows, shadow_windows)
        if horizon_s is None:
            warnings = ("trace_horizon_missing",)
    return horizon_s, step_s, boundary_tolerance_s, warnings


def _manifest_mapping(value: RuntimeTraceManifest | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, RuntimeTraceManifest):
        return value.to_dict()
    return _mapping(value)


def _sample_times(*sample_groups: tuple[dict[str, Any], ...]) -> tuple[float, ...]:
    times = {
        parsed
        for group in sample_groups
        for sample in group
        for parsed in (_optional_float(sample.get("time_s")),)
        if parsed is not None
    }
    return tuple(sorted(times))


def _infer_step_s(sample_times: tuple[float, ...]) -> float | None:
    if len(sample_times) < 2:
        return None
    deltas = [
        later - earlier
        for earlier, later in zip(sample_times, sample_times[1:])
        if later > earlier
    ]
    return min(deltas) if deltas else None


def _max_window_end(*window_groups: tuple[dict[str, Any], ...]) -> float | None:
    ends = [
        parsed
        for group in window_groups
        for window in group
        for parsed in (_optional_float(window.get("end_s")),)
        if parsed is not None
    ]
    return max(ends) if ends else None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


__all__ = [
    "CIRCULAR_GEOMETRY_TRACE",
    "IDEAL_FULL_VISIBILITY",
    "RuntimeTraceManifest",
    "RuntimeTraceBundle",
    "SATELLITE_MOTION_TRACE_SCHEMA",
    "SKELETON_TRACE",
    "SURROGATE_ORBIT_TRACE",
    "SURROGATE_TRACE_SCHEMA",
    "SurrogateTraceBundle",
    "TraceQueryResult",
    "TRACE_BACKED_RUNTIME_SURROGATE_SCOPE",
    "load_surrogate_trace_bundle",
    "normalize_trace_scope",
    "resolve_trace_path",
    "trace_file_sha256",
]
