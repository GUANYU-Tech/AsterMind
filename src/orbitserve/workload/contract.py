"""Canonical OrbitServe workload bundle contract.

This module is stdlib-only so dataset adapters can share it without pulling
optional dataset dependencies into the simulation package.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

BUNDLE_SCHEMA = "orbitserve.workload"
DESCRIPTION_SCHEMA = "orbitserve.workload_description"
SUMMARY_SCHEMA = "orbitserve.workload_summary"
CANONICALIZATION = "orbitserve-json-c14n"
BUNDLE_FILES = {
    "regions": ("regions.json", "application/json"),
    "gateways": ("gateways.json", "application/json"),
    "requests": ("requests.jsonl", "application/x-ndjson"),
    "sessions": ("sessions.jsonl", "application/x-ndjson"),
    "timebin_demand": ("timebin_demand.csv", "text/csv"),
    "summary": ("workload_summary.json", "application/json"),
}
REQUEST_KEYS = (
    "request_id", "arrival_time_s", "user_region_id", "region_id", "lat_deg", "lon_deg",
    "access_mode", "ingress_id", "ingress_lat_deg", "ingress_lon_deg", "gateway_id",
    "task_type", "model_family", "prompt_tokens", "new_user_tokens",
    "context_tokens_before_turn", "output_tokens", "input_bytes", "output_bytes",
    "context_window_tokens", "slo_class", "ttft_deadline_s", "e2e_deadline_s", "priority",
    "session_id", "turn_index", "session_turn_count", "inter_turn_gap_s", "session_ttl_s",
    "burst_id", "burst_multiplier", "seed",
)
SESSION_KEYS = (
    "session_id", "user_region_id", "session_start_time_s", "session_turn_count",
    "session_slo_class", "context_window_tokens", "session_ttl_s", "seed",
)
REGION_KEYS = (
    "region_id", "region_name", "lat_deg", "lon_deg", "timezone_utc_offset_h",
    "demand_weight", "weight_source", "market_class",
)
GATEWAY_KEYS = (
    "gateway_id", "lat_deg", "lon_deg", "served_region_ids", "gateway_bandwidth_bps",
    "availability_profile_id", "cost_usd_per_year",
)
TIMEBIN_FIELDS = (
    "time_bin_start_s", "time_bin_end_s", "region_id", "request_count",
    "prompt_tokens_total", "new_user_tokens_total", "output_tokens_total",
    "interactive_request_count", "burst_request_count",
)


def canonical_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN/Infinity is forbidden in workload")
        rounded = round(value, 6)
        return 0 if rounded == 0 else rounded
    if isinstance(value, Mapping):
        return {str(key): canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    payload = json.dumps(
        canonical_value(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    return (payload + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def canonical_timebin_csv_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=TIMEBIN_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(canonical_value(dict(row)) for row in rows)
    return output.getvalue().encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def descriptor(*, path: str, media_type: str, payload: bytes, record_count: int) -> dict[str, Any]:
    return {
        "path": path, "media_type": media_type,
        "sha256": sha256_bytes(payload), "record_count": int(record_count),
    }


def bundle_hash(workload_id: str, descriptors: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [BUNDLE_SCHEMA, workload_id]
    for item in sorted(descriptors.values(), key=lambda value: str(value["path"])):
        lines.append(f"{item['path']}\t{item['sha256']}\t{item['record_count']}")
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def validate_exact_keys(rows: Iterable[Mapping[str, Any]], keys: tuple[str, ...], label: str) -> None:
    expected = set(keys)
    for index, row in enumerate(rows):
        actual = set(row)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"{label}[{index}] violates workload fields; missing={missing}, extra={extra}"
            )


def validate_manifest_and_files(bundle_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError(
            f"unsupported workload schema {manifest.get('schema_version')!r}; expected {BUNDLE_SCHEMA!r}"
        )
    if manifest.get("canonicalization") != CANONICALIZATION:
        raise ValueError("workload canonicalization is missing or unsupported")
    descriptors = manifest.get("files")
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(BUNDLE_FILES):
        raise ValueError("workload manifest must describe exactly six data files")
    actual_files = {path.name for path in bundle_dir.iterdir() if path.is_file()}
    expected_files = {"workload_manifest.json", *(name for name, _ in BUNDLE_FILES.values())}
    if actual_files != expected_files:
        raise ValueError(
            f"workload bundle must contain exactly seven files; actual={sorted(actual_files)}"
        )
    resolved: dict[str, Path] = {}
    for key, (name, media_type) in BUNDLE_FILES.items():
        item = descriptors[key]
        if not isinstance(item, Mapping):
            raise ValueError(f"workload descriptor {key!r} must be an object")
        if item.get("path") != name or item.get("media_type") != media_type:
            raise ValueError(f"workload descriptor mismatch for {key}")
        path = bundle_dir / name
        if item.get("sha256") != sha256_bytes(path.read_bytes()):
            raise ValueError(f"workload SHA-256 mismatch for {name}")
        resolved[key] = path
    if manifest.get("bundle_sha256") != bundle_hash(str(manifest.get("workload_id")), descriptors):
        raise ValueError("workload bundle_sha256 mismatch")
    return resolved


__all__ = [
    "BUNDLE_FILES", "BUNDLE_SCHEMA", "CANONICALIZATION", "DESCRIPTION_SCHEMA",
    "GATEWAY_KEYS", "REGION_KEYS", "REQUEST_KEYS", "SESSION_KEYS", "SUMMARY_SCHEMA",
    "TIMEBIN_FIELDS", "bundle_hash", "canonical_json_bytes", "canonical_jsonl_bytes",
    "canonical_timebin_csv_bytes", "descriptor", "sha256_bytes", "validate_exact_keys",
    "validate_manifest_and_files",
]
