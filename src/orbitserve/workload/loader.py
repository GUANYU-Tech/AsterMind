"""Consumer-side loading of canonical workload bundles (engine runtime).

``load_workload_bundle`` reads a six-file bundle, verifies the manifest and
bundle hash, coerces rows, and returns a fully validated
WorkloadGenerationResult.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .contract import (
    GATEWAY_KEYS,
    MANIFEST_FILE,
    REGION_KEYS,
    SESSION_REQUEST_KEYS,
    GatewaySpec,
    RegionSpec,
    WorkloadBundleRef,
    WorkloadGenerationResult,
    coerce_session_request_row,
    coerce_timebin_row,
    reject_runtime_fields,
    seed_from_manifest,
    validate_exact_keys,
    validate_manifest_and_files,
    validate_reference_resolution,
    validate_session_consistency,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(item)
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_regions(rows: list[dict[str, Any]]) -> tuple[RegionSpec, ...]:
    return tuple(RegionSpec.from_dict(item) for item in rows)


def _load_gateways(rows: list[dict[str, Any]]) -> tuple[GatewaySpec, ...]:
    return tuple(GatewaySpec.from_dict(item) for item in rows)


def resolve_bundle_path(path_value: Any, base_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    base_candidate = (base_dir / path).resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if path.parts and path.parts[0] in {"data", "docs", "examples", "results"}:
        return cwd_candidate
    if base_candidate.exists():
        return base_candidate
    if cwd_candidate.exists():
        return cwd_candidate
    if cwd_candidate.parent.exists() and not base_candidate.parent.exists():
        return cwd_candidate
    return base_candidate


def load_workload_bundle(bundle_dir: str | Path) -> WorkloadGenerationResult:
    """Load and fully validate one canonical workload bundle."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_json(root / MANIFEST_FILE)
    paths = validate_manifest_and_files(root, manifest)

    # Load data files
    region_rows = _read_json(paths["regions"])
    gateway_rows = _read_json(paths["gateways"])
    session_request_rows = _read_jsonl(paths["session_requests"])
    timebin_rows = _read_csv_rows(paths["timebin_demand"])
    summary = _read_json(paths["summary"])

    # Exact-key validation
    validate_exact_keys(region_rows, REGION_KEYS, "regions")
    validate_exact_keys(gateway_rows, GATEWAY_KEYS, "gateways")
    validate_exact_keys(session_request_rows, SESSION_REQUEST_KEYS, "session_requests")

    # Coerce
    seed = seed_from_manifest(manifest)
    session_requests = tuple(
        coerce_session_request_row(item, seed=seed) for item in session_request_rows
    )
    for index, row in enumerate(session_requests):
        reject_runtime_fields(row, context=f"session_request row {index}")

    regions = _load_regions(region_rows)
    gateways = _load_gateways(gateway_rows)

    # Cross-file validation
    validate_session_consistency(list(session_requests))
    validate_reference_resolution(session_requests, regions=regions, gateways=gateways)

    return WorkloadGenerationResult(
        bundle_dir=str(root),
        manifest=manifest,
        regions=regions,
        gateways=gateways,
        session_requests=session_requests,
        timebin_demand=tuple(coerce_timebin_row(item) for item in timebin_rows),
        summary=summary,
        status="loaded",
    )


def _base_dir(base_dir: str | Path | None) -> Path:
    return Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()


def materialize_workload_bundle(
    workload_ref: WorkloadBundleRef | WorkloadGenerationResult,
    base_dir: str | Path | None = None,
) -> WorkloadGenerationResult:
    if isinstance(workload_ref, WorkloadGenerationResult):
        if workload_ref.bundle_dir is None:
            raise ValueError("bundle_dir is required for an in-memory workload result")
        return load_workload_bundle(workload_ref.bundle_dir)
    if not isinstance(workload_ref, WorkloadBundleRef):
        raise TypeError("workload_ref must be WorkloadBundleRef or WorkloadGenerationResult")
    if not workload_ref.bundle_dir:
        raise ValueError("workload.bundle_dir is required and must reference a workload bundle")
    root = _base_dir(base_dir)
    return load_workload_bundle(resolve_bundle_path(workload_ref.bundle_dir, root))


__all__ = ["load_workload_bundle", "materialize_workload_bundle", "resolve_bundle_path"]
