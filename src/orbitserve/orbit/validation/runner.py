"""Batch runner for the pinned STK state-validation corpus."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable

from ..sgp4.frames import transform_sgp4_teme_to_itrs
from ..sgp4.propagation import propagate_sgp4_teme
from .corpus import StkStateCorpus, load_stk_state_corpus
from .metrics import StkStateComparisonReport, compare_stk_state_capture
from .stk import (
    STK_11_BASELINE_BUILD,
    STK_CONNECT_HOST,
    STK_CONNECT_PORT,
    STK_VALIDATION_CONTRACT_ID,
    StkDirectStateCapture,
    StkEnvironmentError,
    capture_stk_direct_states,
)

STK_STATE_CORPUS_RUN_SCHEMA = "orbitserve.orbit.stk-state-corpus-run"


@dataclass(frozen=True, slots=True)
class StkStateCorpusCaseResult:
    """Complete diagnostic evidence for one frozen corpus case."""

    case_id: str
    coverage: tuple[str, ...]
    capture: StkDirectStateCapture
    comparison: StkStateComparisonReport

    def __post_init__(self) -> None:
        if self.capture.element_set_id != self.comparison.element_set_id:
            raise ValueError("capture and comparison element_set_id must match")
        if (
            self.capture.simulation_epoch_utc
            != self.comparison.simulation_epoch_utc
        ):
            raise ValueError(
                "capture and comparison simulation_epoch_utc must match"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture": self.capture.to_dict(),
            "case_id": self.case_id,
            "comparison": self.comparison.to_dict(),
            "coverage": list(self.coverage),
        }


@dataclass(frozen=True, slots=True)
class StkStateCorpusRun:
    """Canonical aggregate evidence for one complete state-corpus run."""

    corpus_id: str
    corpus_source_sha256: str
    cases: tuple[StkStateCorpusCaseResult, ...]
    schema: str = STK_STATE_CORPUS_RUN_SCHEMA
    contract_id: str = STK_VALIDATION_CONTRACT_ID
    evidence_scope: str = "state_vectors_only"
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


_CaptureFunction = Callable[..., StkDirectStateCapture]


def run_stk_state_corpus(
    corpus: StkStateCorpus | None = None,
    *,
    installed_build: str,
    host: str = STK_CONNECT_HOST,
    port: int = STK_CONNECT_PORT,
    configure_application_defaults: bool = False,
    connect_timeout_s: float = 10.0,
    read_timeout_s: float = 120.0,
    capture_function: _CaptureFunction = capture_stk_direct_states,
) -> StkStateCorpusRun:
    """Run every frozen state case through STK and the Orbit public pipeline."""

    active_corpus = load_stk_state_corpus() if corpus is None else corpus
    if not isinstance(active_corpus, StkStateCorpus):
        raise TypeError("corpus must be StkStateCorpus")
    if not callable(capture_function):
        raise TypeError("capture_function must be callable")
    if installed_build != STK_11_BASELINE_BUILD:
        raise StkEnvironmentError(
            f"STK build {installed_build!r} does not match "
            f"{STK_11_BASELINE_BUILD!r}"
        )
    if not isinstance(configure_application_defaults, bool):
        raise TypeError("configure_application_defaults must be a boolean")
    results: list[StkStateCorpusCaseResult] = []
    for case in active_corpus.cases:
        capture = capture_function(
            case.elements,
            simulation_epoch_utc=case.simulation_epoch_utc,
            sample_times_s=case.sample_times_s,
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
        comparison = compare_stk_state_capture(
            capture,
            orbit_teme,
            orbit_fixed,
        )
        results.append(
            StkStateCorpusCaseResult(
                case_id=case.case_id,
                coverage=case.coverage,
                capture=capture,
                comparison=comparison,
            )
        )
    return StkStateCorpusRun(
        corpus_id=active_corpus.corpus_id,
        corpus_source_sha256=active_corpus.source_sha256,
        cases=tuple(results),
    )


def _normalise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical STK corpus run contains a non-finite float")
        return 0.0 if value == 0.0 else value
    return value


__all__ = [
    "STK_STATE_CORPUS_RUN_SCHEMA",
    "StkStateCorpusCaseResult",
    "StkStateCorpusRun",
    "run_stk_state_corpus",
]
