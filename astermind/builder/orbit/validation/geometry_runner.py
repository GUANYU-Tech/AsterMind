"""Batch runner for the frozen STK geometry validation corpus."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable

from astermind.builder.orbit.sgp4.frames import transform_sgp4_teme_to_itrs
from astermind.builder.orbit.sgp4.geometry import Sgp4GeometryResult, evaluate_sgp4_geometry
from astermind.builder.orbit.sgp4.propagation import propagate_sgp4_teme
from astermind.builder.orbit.validation.geometry_corpus import StkGeometryCorpus, load_stk_geometry_corpus
from astermind.builder.orbit.validation.geometry_metrics import StkGeometryComparisonReport, compare_stk_geometry_capture
from astermind.builder.orbit.validation.stk import STK_11_BASELINE_BUILD, STK_CONNECT_HOST, STK_CONNECT_PORT, STK_VALIDATION_CONTRACT_ID, StkEnvironmentError, StkValidationError
from astermind.builder.orbit.validation.stk_geometry import StkGeometryCapture, capture_stk_geometry


STK_GEOMETRY_CORPUS_RUN_SCHEMA = 'orbitserve.orbit.stk-geometry-corpus-run'


@dataclass(frozen=True, slots=True)
class StkGeometryCorpusCaseResult:
    case_id: str
    coverage: tuple[str, ...]
    capture: StkGeometryCapture
    orbit_geometry: Sgp4GeometryResult
    comparison: StkGeometryComparisonReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture": self.capture.to_dict(),
            "case_id": self.case_id,
            "comparison": self.comparison.to_dict(),
            "coverage": list(self.coverage),
            "orbit_geometry": self.orbit_geometry.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StkGeometryCorpusRun:
    corpus_id: str
    corpus_source_sha256: str
    cases: tuple[StkGeometryCorpusCaseResult, ...]
    schema: str = STK_GEOMETRY_CORPUS_RUN_SCHEMA
    contract_id: str = STK_VALIDATION_CONTRACT_ID
    evidence_scope: str = "sgl_isl_lighting_geometry"
    formal_thresholds_frozen: bool = False

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "contract_id": self.contract_id,
            "corpus_id": self.corpus_id,
            "corpus_source_sha256": self.corpus_source_sha256,
            "evidence_scope": self.evidence_scope,
            "formal_thresholds_frozen": self.formal_thresholds_frozen,
            "schema": self.schema,
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


_CaptureFunction = Callable[..., StkGeometryCapture]


def run_stk_geometry_corpus(
    corpus: StkGeometryCorpus | None = None,
    *,
    installed_build: str,
    host: str = STK_CONNECT_HOST,
    port: int = STK_CONNECT_PORT,
    configure_application_defaults: bool = False,
    connect_timeout_s: float = 10.0,
    read_timeout_s: float = 120.0,
    capture_function: _CaptureFunction = capture_stk_geometry,
) -> StkGeometryCorpusRun:
    """Run each frozen geometry case through STK and the Orbit public pipeline."""

    active_corpus = load_stk_geometry_corpus() if corpus is None else corpus
    if not isinstance(active_corpus, StkGeometryCorpus):
        raise TypeError("corpus must be StkGeometryCorpus")
    if installed_build != STK_11_BASELINE_BUILD:
        raise StkEnvironmentError(
            f"STK build {installed_build!r} does not match "
            f"{STK_11_BASELINE_BUILD!r}"
        )
    if not callable(capture_function):
        raise TypeError("capture_function must be callable")
    results: list[StkGeometryCorpusCaseResult] = []
    for case in active_corpus.cases:
        capture = capture_function(
            case.elements,
            simulation_epoch_utc=case.simulation_epoch_utc,
            sample_times_s=case.sample_times_s,
            ground_ingresses=case.ground_ingresses,
            isl_pairs=case.isl_pairs,
            host=host,
            port=port,
            installed_build=installed_build,
            configure_application_defaults=configure_application_defaults,
            connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s,
        )
        orbit_teme = propagate_sgp4_teme(
            case.elements,
            simulation_epoch_utc=case.simulation_epoch_utc,
            sample_times_s=case.sample_times_s,
        )
        orbit_fixed = transform_sgp4_teme_to_itrs(orbit_teme)
        orbit_geometry = evaluate_sgp4_geometry(
            orbit_fixed,
            ground_ingresses=case.ground_ingresses,
            isl_pairs=case.isl_pairs,
            isl_clearance_km=case.isl_clearance_km,
            shadow_atmosphere_margin_km=case.shadow_atmosphere_margin_km,
        )
        _validate_event_coverage(case.expected_event_counts, capture, orbit_geometry)
        comparison = compare_stk_geometry_capture(capture, orbit_geometry)
        results.append(
            StkGeometryCorpusCaseResult(
                case_id=case.case_id,
                coverage=case.coverage,
                capture=capture,
                orbit_geometry=orbit_geometry,
                comparison=comparison,
            )
        )
    return StkGeometryCorpusRun(
        corpus_id=active_corpus.corpus_id,
        corpus_source_sha256=active_corpus.source_sha256,
        cases=tuple(results),
    )


def _validate_event_coverage(
    expected: dict[str, int],
    capture: StkGeometryCapture,
    orbit: Sgp4GeometryResult,
) -> None:
    orbit_counts = {
        "sgl_visible_windows": len(orbit.sgl_windows),
        "isl_visible_windows": len(orbit.isl_windows),
        "umbra_windows": sum(not window.sunlit for window in orbit.sunlight_windows),
    }
    stk_counts = {
        "sgl_visible_windows": len(capture.sgl_windows),
        "isl_visible_windows": len(capture.isl_windows),
        "umbra_windows": sum(
            window.lighting == "umbra" for window in capture.lighting_windows
        ),
        "penumbra_windows": sum(
            window.lighting == "penumbra" for window in capture.lighting_windows
        ),
    }
    for key, minimum in expected.items():
        if key != "penumbra_windows" and orbit_counts[key] < minimum:
            raise StkValidationError(
                f"Orbit geometry did not exercise required event coverage: {key}"
            )
        if stk_counts[key] < minimum:
            raise StkValidationError(
                f"STK geometry did not exercise required event coverage: {key}"
            )


def _normalise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical STK geometry run contains non-finite data")
        return 0.0 if value == 0.0 else value
    return value


__all__ = [
    "STK_GEOMETRY_CORPUS_RUN_SCHEMA",
    "StkGeometryCorpusCaseResult",
    "StkGeometryCorpusRun",
    "run_stk_geometry_corpus",
]
