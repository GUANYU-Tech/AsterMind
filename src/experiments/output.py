"""Layered experiment output helpers.

The runner keeps a fully auditable in-memory result, while the CLI can emit a
compact summary JSON plus external audit artifacts for bulky traces.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from orbitserve.common import to_jsonable


SUMMARY_SCHEMA_VERSION = "orbitserve.experiment_summary.v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "orbitserve.experiment_artifacts.v1"


def experiment_output_payload(
    result: Any,
    *,
    output_mode: str = "summary",
    artifact_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    indent: int = 2,
) -> dict[str, Any]:
    """Return the requested experiment output payload.

    ``output_mode="full"`` preserves the historical result JSON.  The default
    ``summary`` mode keeps plotting and batch-analysis fields in the main JSON
    and externalizes request-level audit records when ``artifact_dir`` is set.
    """

    normalized_mode = _normalize_output_mode(output_mode)
    payload = _result_to_payload(result)
    if normalized_mode == "full":
        return payload

    writer = _ArtifactWriter(
        Path(artifact_dir) if artifact_dir is not None else None,
        output_path=Path(output_path) if output_path is not None else None,
        indent=indent,
    )
    summary = _summary_payload(payload, writer)
    summary["artifact_manifest"] = writer.manifest()
    return summary


def default_artifact_dir_for_output(out_path: Path | str | None) -> Path | None:
    """Return the default artifact directory next to a summary output file."""

    if out_path is None:
        return None
    output_path = Path(out_path)
    stem = output_path.stem or "experiment"
    return output_path.with_name(f"{stem}_artifacts")


def _summary_payload(payload: dict[str, Any], writer: "_ArtifactWriter") -> dict[str, Any]:
    summary = deepcopy(payload)
    summary["schema_version"] = SUMMARY_SCHEMA_VERSION
    summary["output_mode"] = "summary"
    summary["output_contract"] = {
        "summary_json": "lightweight metrics/provenance for batch runs and plotting",
        "audit_artifacts": "request-level runtime and DSE audit data externalized from bulky fields",
        "full_json_mode": "rerun CLI with --output-mode full to reproduce the historical monolithic JSON",
    }

    performance_details = _dict_at(summary, ("performance", "details"))
    runtime = performance_details.get("runtime")
    if isinstance(runtime, dict):
        performance_details["runtime"] = _externalized_runtime_summary(
            runtime,
            writer=writer,
            artifact_prefix=("runtime",),
            artifact_name="runtime",
            json_path="performance.details.runtime",
        )

    dse = summary.get("dse")
    if isinstance(dse, dict):
        _externalize_dse(dse, writer=writer)

    return summary


def _externalize_dse(dse: dict[str, Any], *, writer: "_ArtifactWriter") -> None:
    rows = dse.get("rows")
    rows_ref = None
    if isinstance(rows, list) and rows:
        rows_ref = writer.write_jsonl(
            name="dse_rows_full",
            rel_path=("dse", "rows_full.jsonl"),
            rows=rows,
            json_path="dse.rows",
        )
        dse["rows"] = [
            _slim_dse_row(row, artifact_ref=rows_ref) if isinstance(row, dict) else row
            for row in rows
        ]
        dse["rows_artifact"] = rows_ref

    runtime_matrix = dse.get("runtime_matrix")
    if isinstance(runtime_matrix, dict):
        matrix_rows = runtime_matrix.get("matrix_rows")
        matrix_ref = None
        if isinstance(matrix_rows, list) and matrix_rows:
            matrix_ref = writer.write_jsonl(
                name="dse_runtime_matrix_rows_full",
                rel_path=("dse", "runtime_matrix_rows_full.jsonl"),
                rows=matrix_rows,
                json_path="dse.runtime_matrix.matrix_rows",
            )
            runtime_matrix["matrix_rows"] = [
                _slim_matrix_row(row, artifact_ref=matrix_ref) if isinstance(row, dict) else row
                for row in matrix_rows
            ]
            runtime_matrix["matrix_rows_artifact"] = matrix_ref

        selected_rows = runtime_matrix.get("selected_rows")
        if isinstance(selected_rows, list):
            runtime_matrix["selected_rows"] = [
                _slim_matrix_row(row, artifact_ref=matrix_ref) if isinstance(row, dict) else row
                for row in selected_rows
            ]

        selected_by_design = runtime_matrix.get("selected_by_design")
        if isinstance(selected_by_design, dict):
            runtime_matrix["selected_by_design"] = {
                key: _slim_matrix_row(value, artifact_ref=matrix_ref)
                if isinstance(value, dict)
                else value
                for key, value in selected_by_design.items()
            }

    frontier = dse.get("frontier")
    if isinstance(frontier, dict):
        frontier_ref = writer.write_json(
            name="dse_frontier_full",
            rel_path=("dse", "frontier_full.json"),
            value=frontier,
            json_path="dse.frontier",
        )
        _slim_frontier(frontier, artifact_ref=frontier_ref)


def _externalized_runtime_summary(
    runtime: dict[str, Any],
    *,
    writer: "_ArtifactWriter",
    artifact_prefix: tuple[str, ...],
    artifact_name: str,
    json_path: str,
) -> dict[str, Any]:
    request_records = _request_records_from_runtime(runtime)
    decision_records = _decision_records_from_runtime(runtime)
    request_ref = (
        writer.write_jsonl(
            name=f"{artifact_name}_request_records",
            rel_path=(*artifact_prefix, "request_records.jsonl"),
            rows=request_records,
            json_path=f"{json_path}.request_records",
        )
        if request_records
        else None
    )
    decision_ref = (
        writer.write_jsonl(
            name=f"{artifact_name}_decision_trace_records",
            rel_path=(*artifact_prefix, "decision_trace_records.jsonl"),
            rows=decision_records,
            json_path=f"{json_path}.decision_trace.records",
        )
        if decision_records
        else None
    )
    return _slim_runtime_result(
        runtime,
        request_records_artifact=request_ref,
        decision_trace_artifact=decision_ref,
    )


def _slim_score_row(row: dict[str, Any]) -> dict[str, Any]:
    slim = deepcopy(row)
    evaluator_result = slim.get("evaluator_result")
    if isinstance(evaluator_result, dict):
        slim["evaluator_result"] = _slim_runtime_result(evaluator_result)
    return slim


def _slim_dse_row(row: dict[str, Any], *, artifact_ref: dict[str, Any] | None) -> dict[str, Any]:
    slim = deepcopy(row)
    details = _dict_at(slim, ("performance", "details"))
    for key in ("runtime",):
        value = details.get(key)
        if isinstance(value, dict):
            details[key] = _slim_runtime_result(value)
    if artifact_ref is not None:
        slim["artifact_ref"] = artifact_ref
    return slim


def _slim_matrix_row(row: dict[str, Any], *, artifact_ref: dict[str, Any] | None) -> dict[str, Any]:
    slim = deepcopy(row)
    runtime_result = slim.get("runtime_result")
    if isinstance(runtime_result, dict):
        slim["runtime_result"] = _slim_runtime_result(runtime_result)
    if artifact_ref is not None:
        slim["artifact_ref"] = artifact_ref
    return slim


def _slim_runtime_result(
    runtime: dict[str, Any],
    *,
    request_records_artifact: dict[str, Any] | None = None,
    decision_trace_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slim = deepcopy(runtime)
    request_records = _pop_list(slim, "request_records")
    details = slim.get("details")
    detail_records: list[Any] = []
    if isinstance(details, dict):
        detail_records = _pop_list(details, "request_records")

    request_count = len(request_records) if request_records else len(detail_records)
    if request_count:
        slim["request_record_count"] = request_count
        if isinstance(details, dict):
            details["request_record_count"] = request_count
    if request_records_artifact is not None:
        slim["request_records_artifact"] = request_records_artifact
        slim["request_records_path"] = request_records_artifact.get("path")
        if isinstance(details, dict):
            details["request_records_artifact"] = request_records_artifact
            details["request_records_path"] = request_records_artifact.get("path")

    decision_records = _pop_list(slim, "decision_trace")
    if decision_records:
        slim["decision_trace_count"] = len(decision_records)
    if decision_trace_artifact is not None:
        slim["decision_trace_artifact"] = decision_trace_artifact
        slim["decision_trace_path"] = decision_trace_artifact.get("path")

    strategy = slim.get("strategy")
    if isinstance(strategy, dict):
        slim["strategy"] = _slim_strategy(strategy)

    constellation = slim.get("constellation")
    if isinstance(constellation, dict):
        slim["constellation"] = _constellation_summary(constellation)

    workload = slim.get("workload")
    if isinstance(workload, dict):
        slim["workload"] = _workload_summary(workload)

    provenance = slim.get("provenance")
    if isinstance(provenance, dict):
        slim["provenance"] = _provenance_summary(provenance)

    state_summary = slim.get("state_summary")
    if isinstance(state_summary, dict):
        slim["state_summary"] = _state_summary(state_summary)

    if isinstance(details, dict):
        state_summary = details.get("state_summary")
        if isinstance(state_summary, dict):
            details["state_summary"] = _state_summary(state_summary)
        deployment_catalog = details.get("deployment_catalog")
        if isinstance(deployment_catalog, dict):
            details["deployment_catalog"] = _deployment_catalog_summary(deployment_catalog)

    return slim


def _slim_frontier(frontier: dict[str, Any], *, artifact_ref: dict[str, Any] | None) -> None:
    for key in ("frontier_rows", "eligible_candidates", "excluded_candidates"):
        value = frontier.get(key)
        if isinstance(value, list):
            frontier[key] = [
                _frontier_candidate_summary(candidate) if isinstance(candidate, dict) else candidate
                for candidate in value
            ]
    if artifact_ref is not None:
        frontier["artifact_ref"] = artifact_ref


def _frontier_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    keep_fields = (
        "candidate_id",
        "design_strategy_id",
        "row_index",
        "design_id",
        "strategy_id",
        "policy",
        "source",
        "runtime_evaluated",
        "eligible",
        "excluded_reason",
        "missing_fields",
        "pareto_member",
        "valid_for_paper_main_results",
        "runtime_metric_scope",
        "calibration",
        "cost_metric",
        "cost_usd",
        "annualized_cost_usd_per_year",
        "cost_total_usd",
        "e2e_p95_s",
        "ttft_p95_s",
        "e2e_p99_s",
        "ttft_p99_s",
        "objective_score",
        "objective_score_scope",
        "trace_model",
        "scope",
        "warnings",
    )
    return {key: candidate.get(key) for key in keep_fields if key in candidate}


def _slim_strategy(strategy: dict[str, Any]) -> dict[str, Any]:
    keep_fields = (
        "ingress_selection",
        "latency_weight",
        "kv_reuse_weight",
        "energy_weight",
        "compute_load_weight",
        "shadow_dvfs_enabled",
        "shadow_dvfs_scale",
        "kv_migration_fraction",
    )
    return {key: strategy.get(key) for key in keep_fields if key in strategy}


def _constellation_summary(constellation: dict[str, Any]) -> dict[str, Any]:
    design = constellation.get("design")
    architecture = constellation.get("architecture")
    design_summary = _design_summary(design) if isinstance(design, dict) else {}
    architecture_summary = (
        _architecture_summary(architecture) if isinstance(architecture, dict) else {}
    )
    return {
        "satellite_count": constellation.get("satellite_count"),
        "design": design_summary,
        "architecture": architecture_summary,
    }


def _design_summary(design: dict[str, Any]) -> dict[str, Any]:
    keep_fields = (
        "name",
        "satellite_count",
        "plane_count",
        "sats_per_plane",
        "altitude_km",
        "inclination_deg",
        "orbit_family",
        "min_elevation_deg",
    )
    summary = {key: design.get(key) for key in keep_fields if key in design}
    orbital_sets = design.get("orbital_sets")
    if isinstance(orbital_sets, list):
        summary["orbital_set_count"] = len(orbital_sets)
    return summary


def _architecture_summary(architecture: dict[str, Any]) -> dict[str, Any]:
    keep_fields = (
        "evaluated",
        "source",
        "satellite_count",
        "total_satellite_count",
        "cluster_count",
        "cluster_satellite_count",
        "warnings",
    )
    summary = {key: architecture.get(key) for key in keep_fields if key in architecture}
    return summary


def _workload_summary(workload: dict[str, Any]) -> dict[str, Any]:
    keep_fields = (
        "status",
        "configured",
        "bundle_dir",
        "schema_version",
        "source_mode",
        "files",
        "request_count",
        "session_count",
        "warnings",
    )
    summary = {key: workload.get(key) for key in keep_fields if key in workload}
    workload_summary = workload.get("summary")
    if isinstance(workload_summary, dict):
        summary["summary"] = {
            key: workload_summary.get(key)
            for key in (
                "schema_version",
                "request_count",
                "session_count",
                "duration_s",
                "source_mode",
                "token_stats",
                "slo_mix",
                "regional_mix",
            )
            if key in workload_summary
        }
    return summary


def _provenance_summary(provenance: dict[str, Any]) -> dict[str, Any]:
    keep_fields = (
        "performance_evaluator",
        "runtime_interface_schema",
        "runtime_performance_schema",
        "valid_for_paper_main_results",
        "runtime_engine",
        "simulation_trace",
        "workload_bundle",
    )
    summary = {key: provenance.get(key) for key in keep_fields if key in provenance}
    trace_manifest = provenance.get("trace_manifest")
    if isinstance(trace_manifest, dict):
        summary["trace_manifest"] = {
            key: trace_manifest.get(key)
            for key in (
                "trace_model",
                "scope",
                "trace_id",
                "evaluated",
                "valid_for_paper_main_results",
                "access_trace_sha256",
                "link_trace_sha256",
                "energy_trace_sha256",
                "satellite_catalog_sha256",
            )
            if key in trace_manifest
        }
    omitted = [
        key
        for key in ("hardware_config", "cost_model")
        if key in provenance and key not in summary
    ]
    if omitted:
        summary["omitted_large_fields"] = omitted
    return summary


def _state_summary(state_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "satellite_count": len(state_summary.get("satellites", {}) or {}),
        "link_count": len(state_summary.get("links", {}) or {}),
        "session_count": len(state_summary.get("sessions", {}) or {}),
        "request_count": state_summary.get("request_count"),
        "decision_trace_count": state_summary.get("decision_trace_count"),
        "module_status": state_summary.get("module_status", {}),
    }


def _deployment_catalog_summary(deployment_catalog: dict[str, Any]) -> dict[str, Any]:
    deployments = deployment_catalog.get("deployments")
    default_deployment = deployment_catalog.get("default_deployment")
    return {
        "deployment_count": len(deployments) if isinstance(deployments, list) else None,
        "default_deployment": default_deployment,
        "warnings": deployment_catalog.get("warnings", ()),
    }


def _request_records_from_runtime(runtime: dict[str, Any]) -> list[Any]:
    records = runtime.get("request_records")
    if isinstance(records, list):
        return records
    details = runtime.get("details")
    if isinstance(details, dict):
        detail_records = details.get("request_records")
        if isinstance(detail_records, list):
            return detail_records
    return []


def _decision_records_from_runtime(runtime: dict[str, Any]) -> list[Any]:
    decision_trace = runtime.get("decision_trace")
    if isinstance(decision_trace, list):
        return decision_trace
    if isinstance(decision_trace, dict):
        records = decision_trace.get("records")
        return records if isinstance(records, list) else []
    return []


def _pop_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.pop(key, None)
    return value if isinstance(value, list) else []


def _dict_at(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _result_to_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        payload = to_jsonable(result)
    elif hasattr(result, "to_dict") and callable(result.to_dict):
        payload = to_jsonable(result.to_dict())
    else:
        payload = to_jsonable(result)
    if not isinstance(payload, dict):
        raise TypeError("experiment result must serialize to a JSON object")
    return payload


def _normalize_output_mode(output_mode: str) -> str:
    mode = str(output_mode or "summary").strip().lower()
    if mode not in {"summary", "full"}:
        raise ValueError("output_mode must be 'summary' or 'full'")
    return mode


class _ArtifactWriter:
    def __init__(
        self,
        root: Path | None,
        *,
        output_path: Path | None,
        indent: int,
    ) -> None:
        self.root = root
        self.output_path = output_path
        self.indent = indent
        self.artifacts: list[dict[str, Any]] = []

    def write_jsonl(
        self,
        *,
        name: str,
        rel_path: tuple[str, ...],
        rows: list[Any],
        json_path: str,
    ) -> dict[str, Any] | None:
        if self.root is None:
            return None
        path = self.root.joinpath(*rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(
            json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        )
        return self._write(
            name=name,
            path=path,
            text=text,
            content_type="application/x-jsonlines",
            json_path=json_path,
            record_count=len(rows),
        )

    def write_json(
        self,
        *,
        name: str,
        rel_path: tuple[str, ...],
        value: Any,
        json_path: str,
    ) -> dict[str, Any] | None:
        if self.root is None:
            return None
        path = self.root.joinpath(*rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(to_jsonable(value), ensure_ascii=False, indent=self.indent, sort_keys=True) + "\n"
        return self._write(
            name=name,
            path=path,
            text=text,
            content_type="application/json",
            json_path=json_path,
        )

    def _write(
        self,
        *,
        name: str,
        path: Path,
        text: str,
        content_type: str,
        json_path: str,
        record_count: int | None = None,
    ) -> dict[str, Any]:
        path.write_text(text, encoding="utf-8")
        encoded = text.encode("utf-8")
        ref: dict[str, Any] = {
            "name": name,
            "path": self._display_path(path),
            "content_type": content_type,
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "json_path": json_path,
        }
        if record_count is not None:
            ref["record_count"] = record_count
        self.artifacts.append(ref)
        return ref

    def _display_path(self, path: Path) -> str:
        if self.output_path is not None:
            try:
                return str(path.relative_to(self.output_path.parent))
            except ValueError:
                pass
        return str(path)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "enabled": self.root is not None,
            "artifact_dir": str(self.root) if self.root is not None else None,
            "artifact_count": len(self.artifacts),
            "artifacts": list(self.artifacts),
            "warnings": ()
            if self.root is not None
            else ("artifact_dir is not configured; summary JSON omits bulky audit records",),
        }


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "default_artifact_dir_for_output",
    "experiment_output_payload",
]
