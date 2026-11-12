"""Frozen local input corpus for STK geometry-event validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from astermind.builder.orbit.sgp4.elements import Sgp4ElementBundle, load_sgp4_element_bundle
from astermind.builder.orbit.sgp4.geometry import Sgp4GroundIngress


STK_GEOMETRY_CORPUS_SCHEMA = 'orbitserve.orbit.stk-geometry-corpus'
STK_GEOMETRY_CORPUS_ID = "stk-geometry-pilot-2026-07"
STK_GEOMETRY_CORPUS_CASE_IDS = ("leo-boundary-geometry",)
STK_GEOMETRY_CORPUS_REQUIRED_COVERAGE = frozenset(
    {
        "isl_earth_limb_boundary",
        "penumbra_boundary",
        "sgl_horizon_boundary",
        "umbra_boundary",
    }
)
STK_GEOMETRY_CORPUS_SHA256 = (
    "639f28892bc0e23e7fd4bcdd898d7f2e98b0ea9ff3b67ecd9c363444b3a2cc5e"
)

_CORPUS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "stk_validation"
    / "geometry_corpus.json"
)
_TOP_FIELDS = {"schema_version", "corpus_id", "cases"}
_CASE_FIELDS = {
    "case_id",
    "coverage",
    "description",
    "elements",
    "ground_ingresses",
    "isl_clearance_km",
    "isl_pairs",
    "minimum_expected_event_counts",
    "sample_grid_s",
    "shadow_atmosphere_margin_km",
    "simulation_epoch_utc",
}
_EVENT_COUNT_FIELDS = {
    "isl_visible_windows",
    "penumbra_windows",
    "sgl_visible_windows",
    "umbra_windows",
}
_MAX_GRID_SAMPLES = 1_000_000


@dataclass(frozen=True, slots=True)
class StkGeometryCorpusCase:
    """One immutable geometry validation input."""

    case_id: str
    description: str
    coverage: tuple[str, ...]
    simulation_epoch_utc: str
    sample_times_s: tuple[float, ...]
    elements: Sgp4ElementBundle
    ground_ingresses: tuple[Sgp4GroundIngress, ...]
    isl_pairs: tuple[tuple[str, str], ...]
    isl_clearance_km: float
    shadow_atmosphere_margin_km: float
    minimum_expected_event_counts: tuple[tuple[str, int], ...]

    @property
    def expected_event_counts(self) -> dict[str, int]:
        return dict(self.minimum_expected_event_counts)


@dataclass(frozen=True, slots=True)
class StkGeometryCorpus:
    """Validated geometry corpus plus the hash of its exact source bytes."""

    corpus_id: str
    cases: tuple[StkGeometryCorpusCase, ...]
    source_sha256: str
    schema_version: str = STK_GEOMETRY_CORPUS_SCHEMA

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)


def load_stk_geometry_corpus(
    source: str | Path | Mapping[str, Any] | None = None,
) -> StkGeometryCorpus:
    """Load the frozen corpus or validate an equivalent caller-supplied mapping."""

    if source is None:
        raw = _CORPUS_PATH.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash != STK_GEOMETRY_CORPUS_SHA256:
            raise ValueError("frozen STK geometry corpus hash does not match its contract")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("STK geometry corpus must be valid UTF-8 JSON") from exc
    elif isinstance(source, Mapping):
        payload = source
        raw = _canonical_bytes(payload)
        source_hash = hashlib.sha256(raw).hexdigest()
    else:
        path = Path(source)
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("STK geometry corpus must be valid UTF-8 JSON") from exc

    root = _mapping(payload, "STK geometry corpus")
    _exact_fields(root, _TOP_FIELDS, "STK geometry corpus")
    if root["schema_version"] != STK_GEOMETRY_CORPUS_SCHEMA:
        raise ValueError("unsupported STK geometry corpus schema")
    corpus_id = _identifier(root["corpus_id"], "corpus_id")
    cases_value = _sequence(root["cases"], "cases")
    if not cases_value:
        raise ValueError("STK geometry corpus cases must be non-empty")
    cases = tuple(_case(value, index) for index, value in enumerate(cases_value))
    case_ids = tuple(case.case_id for case in cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("STK geometry corpus case_id values must be unique")
    coverage = {item for case in cases for item in case.coverage}
    missing = STK_GEOMETRY_CORPUS_REQUIRED_COVERAGE - coverage
    if missing:
        raise ValueError(f"STK geometry corpus is missing coverage: {sorted(missing)}")
    if source is None:
        if corpus_id != STK_GEOMETRY_CORPUS_ID:
            raise ValueError("frozen STK geometry corpus_id does not match its contract")
        if case_ids != STK_GEOMETRY_CORPUS_CASE_IDS:
            raise ValueError("frozen STK geometry corpus case order changed")
    return StkGeometryCorpus(corpus_id, cases, source_hash)


def _case(value: Any, index: int) -> StkGeometryCorpusCase:
    label = f"cases[{index}]"
    item = _mapping(value, label)
    _exact_fields(item, _CASE_FIELDS, label)
    case_id = _identifier(item["case_id"], f"{label}.case_id")
    description = _identifier(item["description"], f"{label}.description")
    coverage = tuple(
        sorted(
            {
                _identifier(entry, f"{label}.coverage")
                for entry in _sequence(item["coverage"], f"{label}.coverage")
            }
        )
    )
    if not coverage:
        raise ValueError(f"{label}.coverage must be non-empty")
    sample_times = _sample_grid(item["sample_grid_s"], f"{label}.sample_grid_s")
    elements = load_sgp4_element_bundle(item["elements"])
    ingresses = tuple(
        sorted(
            (
                Sgp4GroundIngress.from_value(entry)
                for entry in _sequence(
                    item["ground_ingresses"], f"{label}.ground_ingresses"
                )
            ),
            key=lambda entry: entry.ingress_id,
        )
    )
    ingress_ids = tuple(entry.ingress_id for entry in ingresses)
    if not ingresses or len(ingress_ids) != len(set(ingress_ids)):
        raise ValueError(f"{label}.ground_ingresses must have unique identifiers")
    satellite_ids = {entry.satellite_id for entry in elements.satellites}
    pairs = tuple(
        sorted(
            _isl_pair(entry, satellite_ids, f"{label}.isl_pairs")
            for entry in _sequence(item["isl_pairs"], f"{label}.isl_pairs")
        )
    )
    if not pairs or len(pairs) != len(set(pairs)):
        raise ValueError(f"{label}.isl_pairs must be non-empty and unique")
    counts = _mapping(
        item["minimum_expected_event_counts"],
        f"{label}.minimum_expected_event_counts",
    )
    _exact_fields(
        counts,
        _EVENT_COUNT_FIELDS,
        f"{label}.minimum_expected_event_counts",
    )
    expected_counts = tuple(
        (key, _positive_int(counts[key], f"{label}.{key}"))
        for key in sorted(_EVENT_COUNT_FIELDS)
    )
    return StkGeometryCorpusCase(
        case_id=case_id,
        description=description,
        coverage=coverage,
        simulation_epoch_utc=_canonical_utc(
            item["simulation_epoch_utc"], f"{label}.simulation_epoch_utc"
        ),
        sample_times_s=sample_times,
        elements=elements,
        ground_ingresses=ingresses,
        isl_pairs=pairs,
        isl_clearance_km=_nonnegative_float(
            item["isl_clearance_km"], f"{label}.isl_clearance_km"
        ),
        shadow_atmosphere_margin_km=_nonnegative_float(
            item["shadow_atmosphere_margin_km"],
            f"{label}.shadow_atmosphere_margin_km",
        ),
        minimum_expected_event_counts=expected_counts,
    )


def _sample_grid(value: Any, label: str) -> tuple[float, ...]:
    grid = _mapping(value, label)
    _exact_fields(grid, {"start", "stop", "step"}, label)
    start = _nonnegative_float(grid["start"], f"{label}.start")
    stop = _nonnegative_float(grid["stop"], f"{label}.stop")
    step = _positive_float(grid["step"], f"{label}.step")
    if stop <= start:
        raise ValueError(f"{label}.stop must be greater than start")
    count_float = (stop - start) / step
    count = round(count_float)
    if abs(count_float - count) > 1e-9 or count + 1 > _MAX_GRID_SAMPLES:
        raise ValueError(f"{label} must end exactly on a bounded uniform grid")
    return tuple(start + index * step for index in range(count + 1))


def _isl_pair(
    value: Any, satellite_ids: set[str], label: str
) -> tuple[str, str]:
    pair = _sequence(value, label)
    if len(pair) != 2:
        raise ValueError(f"{label} entries must contain two satellite identifiers")
    first = _identifier(pair[0], label)
    second = _identifier(pair[1], label)
    if first == second or first not in satellite_ids or second not in satellite_ids:
        raise ValueError(f"{label} entries must name two distinct corpus satellites")
    return tuple(sorted((first, second)))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a mapping with string keys")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _canonical_utc(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be canonical UTC text")
    from datetime import datetime

    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DDTHH:MM:SS.ffffffZ") from exc
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


__all__ = [
    "STK_GEOMETRY_CORPUS_CASE_IDS",
    "STK_GEOMETRY_CORPUS_ID",
    "STK_GEOMETRY_CORPUS_REQUIRED_COVERAGE",
    "STK_GEOMETRY_CORPUS_SCHEMA",
    "STK_GEOMETRY_CORPUS_SHA256",
    "StkGeometryCorpus",
    "StkGeometryCorpusCase",
    "load_stk_geometry_corpus",
]
