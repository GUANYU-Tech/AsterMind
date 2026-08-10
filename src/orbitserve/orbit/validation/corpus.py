"""Pinned, offline diagnostic corpus for STK state validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..sgp4.elements import Sgp4ElementBundle

STK_STATE_CORPUS_SCHEMA = "orbitserve.orbit.stk-state-corpus"
STK_STATE_CORPUS_ID = "stk-state-pilot-2026-07"
STK_STATE_CORPUS_SHA256 = (
    "0d47eec875b6cd7a98a6eb3c2d6e31b0da099d77216364624fec21b522b9c805"
)
STK_STATE_CORPUS_CASE_IDS = (
    "near-circular-leo",
    "moderate-eccentricity-leo",
    "nonzero-drag-derivatives-leo",
    "deep-space-geo",
)
STK_STATE_CORPUS_REQUIRED_COVERAGE = frozenset(
    {
        "deep_space_geo",
        "epoch_sample",
        "inclined_moderate_eccentricity_leo",
        "multi_day_span",
        "near_circular_leo",
        "nonzero_bstar",
        "nonzero_mean_motion_derivatives",
        "post_epoch_samples",
        "pre_epoch_samples",
    }
)

_DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "stk_validation"
    / "state_corpus.json"
)
_CASE_KEYS = {
    "case_id",
    "coverage",
    "description",
    "elements",
    "sample_times_s",
    "simulation_epoch_utc",
}


@dataclass(frozen=True, slots=True)
class StkStateCorpusCase:
    """One immutable state-vector validation case."""

    case_id: str
    description: str
    coverage: tuple[str, ...]
    simulation_epoch_utc: str
    sample_times_s: tuple[float, ...]
    elements: Sgp4ElementBundle

    @property
    def element_set_id(self) -> str:
        return self.elements.element_set_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "coverage": list(self.coverage),
            "description": self.description,
            "elements": self.elements.to_dict(),
            "sample_times_s": list(self.sample_times_s),
            "simulation_epoch_utc": self.simulation_epoch_utc,
        }


@dataclass(frozen=True, slots=True)
class StkStateCorpus:
    """Validated state corpus with pinned source identity."""

    cases: tuple[StkStateCorpusCase, ...]
    source_sha256: str
    schema_version: str = STK_STATE_CORPUS_SCHEMA
    corpus_id: str = STK_STATE_CORPUS_ID

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)

    @property
    def coverage(self) -> frozenset[str]:
        return frozenset(
            item
            for case in self.cases
            for item in case.coverage
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "corpus_id": self.corpus_id,
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                _normalise_json(self.to_dict()),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )


def load_stk_state_corpus(
    source: str | Path | None = None,
    *,
    expected_sha256: str | None = None,
) -> StkStateCorpus:
    """Load the pinned corpus or an explicitly hash-qualified local corpus."""

    path = _DEFAULT_CORPUS_PATH if source is None else Path(source)
    if expected_sha256 is None:
        if source is not None:
            raise ValueError("a custom STK corpus requires expected_sha256")
        expected_sha256 = STK_STATE_CORPUS_SHA256
    _require_sha256(expected_sha256)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"unable to read STK state corpus: {path}") from exc
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "STK state corpus SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("STK state corpus must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("invalid STK state corpus JSON") from exc
    root = _mapping(payload, "STK state corpus")
    _exact_keys(
        root,
        {"cases", "corpus_id", "schema_version"},
        "STK state corpus",
    )
    if root["schema_version"] != STK_STATE_CORPUS_SCHEMA:
        raise ValueError(
            f"schema_version must be {STK_STATE_CORPUS_SCHEMA!r}"
        )
    if root["corpus_id"] != STK_STATE_CORPUS_ID:
        raise ValueError(f"corpus_id must be {STK_STATE_CORPUS_ID!r}")
    raw_cases = root["cases"]
    if isinstance(raw_cases, (str, bytes)) or not isinstance(raw_cases, Sequence):
        raise TypeError("cases must be a JSON array")
    cases = tuple(_case(value) for value in raw_cases)
    if tuple(case.case_id for case in cases) != STK_STATE_CORPUS_CASE_IDS:
        raise ValueError(
            "STK state corpus cases must use the frozen IDs and order"
        )
    corpus = StkStateCorpus(cases=cases, source_sha256=actual_sha256)
    missing_coverage = STK_STATE_CORPUS_REQUIRED_COVERAGE - corpus.coverage
    if missing_coverage:
        raise ValueError(
            "STK state corpus is missing required coverage: "
            + ", ".join(sorted(missing_coverage))
        )
    return corpus


def _case(value: Any) -> StkStateCorpusCase:
    payload = _mapping(value, "STK state corpus case")
    _exact_keys(payload, _CASE_KEYS, "STK state corpus case")
    case_id = _nonempty_string(payload["case_id"], "case_id")
    description = _nonempty_string(payload["description"], "description")
    raw_coverage = payload["coverage"]
    if (
        isinstance(raw_coverage, (str, bytes))
        or not isinstance(raw_coverage, Sequence)
        or not raw_coverage
    ):
        raise TypeError("coverage must be a non-empty JSON array")
    coverage = tuple(
        _nonempty_string(item, "coverage item") for item in raw_coverage
    )
    if coverage != tuple(sorted(set(coverage))):
        raise ValueError("coverage must be unique and lexicographically ordered")
    unknown_coverage = set(coverage) - STK_STATE_CORPUS_REQUIRED_COVERAGE
    if unknown_coverage:
        raise ValueError(
            "unknown STK state corpus coverage: "
            + ", ".join(sorted(unknown_coverage))
        )
    simulation_epoch_utc = _canonical_utc(
        payload["simulation_epoch_utc"],
        "simulation_epoch_utc",
    )
    sample_times_s = _sample_grid(payload["sample_times_s"])
    elements = Sgp4ElementBundle.from_value(
        _mapping(payload["elements"], "elements")
    )
    if len(elements.satellites) != 1:
        raise ValueError("each STK state corpus case must contain one satellite")
    case = StkStateCorpusCase(
        case_id=case_id,
        description=description,
        coverage=coverage,
        simulation_epoch_utc=simulation_epoch_utc,
        sample_times_s=sample_times_s,
        elements=elements,
    )
    _validate_coverage_claims(case)
    return case


def _validate_coverage_claims(case: StkStateCorpusCase) -> None:
    element = case.elements.satellites[0]
    simulation_epoch = _utc_datetime(case.simulation_epoch_utc)
    element_epoch = _utc_datetime(element.epoch_utc)
    instants = tuple(
        simulation_epoch + timedelta(seconds=value)
        for value in case.sample_times_s
    )
    coverage = set(case.coverage)
    claims = {
        "epoch_sample": element_epoch in instants,
        "pre_epoch_samples": any(value < element_epoch for value in instants),
        "post_epoch_samples": any(value > element_epoch for value in instants),
        "multi_day_span": (
            case.sample_times_s[-1] - case.sample_times_s[0] >= 172800.0
        ),
        "near_circular_leo": (
            element.eccentricity <= 0.01
            and element.mean_motion_rev_per_day > 6.4
        ),
        "inclined_moderate_eccentricity_leo": (
            0.01 < element.eccentricity <= 0.2
            and element.inclination_deg >= 45.0
            and element.mean_motion_rev_per_day > 6.4
        ),
        "nonzero_bstar": element.bstar_inv_earth_radii != 0.0,
        "nonzero_mean_motion_derivatives": (
            element.mean_motion_dot_rev_per_day2 != 0.0
            and element.mean_motion_ddot_rev_per_day3 != 0.0
        ),
        "deep_space_geo": (
            0.8 <= element.mean_motion_rev_per_day <= 1.2
            and element.eccentricity <= 0.01
        ),
    }
    invalid = sorted(name for name in coverage if not claims[name])
    if invalid:
        raise ValueError(
            f"case {case.case_id!r} has invalid coverage claims: "
            + ", ".join(invalid)
        )


def _sample_grid(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("sample_times_s must be a JSON array")
    times = tuple(_finite_float(item, "sample_times_s") for item in value)
    if len(times) < 2:
        raise ValueError("sample_times_s must contain at least two values")
    if times[0] < 0.0:
        raise ValueError("sample_times_s must be non-negative")
    deltas = tuple(second - first for first, second in zip(times, times[1:]))
    if any(delta <= 0.0 for delta in deltas):
        raise ValueError("sample_times_s must be strictly increasing")
    step = deltas[0]
    tolerance = max(abs(step) * 1e-12, 1e-9)
    if any(abs(delta - step) > tolerance for delta in deltas[1:]):
        raise ValueError("sample_times_s must use a uniform step")
    for time_s in times:
        microseconds = time_s * 1_000_000.0
        if abs(microseconds - round(microseconds)) > 1e-6:
            raise ValueError(
                "sample_times_s must be exactly representable in microseconds"
            )
    return times


def _canonical_utc(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    _utc_datetime(value)
    return value


def _utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ValueError(
            "UTC instant must use YYYY-MM-DDTHH:MM:SS.ffffffZ"
        ) from exc
    return parsed.replace(tzinfo=timezone.utc)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must contain only numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must contain only finite numbers")
    return 0.0 if number == 0.0 else number


def _require_sha256(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal digits")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _normalise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical STK corpus contains a non-finite float")
        return 0.0 if value == 0.0 else value
    return value


__all__ = [
    "STK_STATE_CORPUS_CASE_IDS",
    "STK_STATE_CORPUS_ID",
    "STK_STATE_CORPUS_REQUIRED_COVERAGE",
    "STK_STATE_CORPUS_SCHEMA",
    "STK_STATE_CORPUS_SHA256",
    "StkStateCorpus",
    "StkStateCorpusCase",
    "load_stk_state_corpus",
]
