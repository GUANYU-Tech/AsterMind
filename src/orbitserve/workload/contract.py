"""Canonical OrbitServe workload bundle contract (shared layer).

This module is stdlib-only so dataset adapters can share it without pulling
optional dataset dependencies into the simulation package. It hosts the
shared data types (WorkloadBundleRef / RegionSpec / GatewaySpec /
WorkloadGenerationResult) and the shared validation logic (exact key sets,
prompt=context+new_user, turn contiguity, reference resolution) used by both
the consumer-side loader and the build-time generation tooling.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

BUNDLE_SCHEMA = "orbitserve.workload"
DESCRIPTION_SCHEMA = "orbitserve.workload_description"
SUMMARY_SCHEMA = "orbitserve.workload_summary"
CANONICALIZATION = "orbitserve-json-c14n"
WORKLOAD_BUNDLE_SCHEMA = BUNDLE_SCHEMA
WORKLOAD_DESCRIPTION_SCHEMA = DESCRIPTION_SCHEMA
BUNDLE_FILES = {
    "regions": ("regions.json", "application/json"),
    "gateways": ("gateways.json", "application/json"),
    "requests": ("requests.jsonl", "application/x-ndjson"),
    "sessions": ("sessions.jsonl", "application/x-ndjson"),
    "timebin_demand": ("timebin_demand.csv", "text/csv"),
    "summary": ("workload_summary.json", "application/json"),
}
REQUEST_FILES = {
    "regions": "regions.json",
    "gateways": "gateways.json",
    "requests": "requests.jsonl",
    "sessions": "sessions.jsonl",
    "timebin_demand": "timebin_demand.csv",
    "summary": "workload_summary.json",
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
RUNTIME_RESULT_FIELDS = frozenset(
    {
        "assigned_satellite",
        "assigned_satellite_id",
        "assigned_deployment",
        "access_satellite_id",
        "selected_path",
        "session_action",
        "served",
        "served_ratio",
        "TTFT",
        "E2E",
        "ttft_s",
        "ttft_p95_s",
        "observed_ttft_s",
        "e2e_latency_s",
        "e2e_p95_s",
        "observed_e2e_s",
        "energy_violation",
        "energy_violation_fraction",
        "cost_per_request",
        "cost_per_served_request_usd",
        "cost_per_served_mtoken_usd",
        "soc_before",
        "soc_after",
        "sgl_queue_wait_s",
        "isl_queue_wait_s",
        "link_transmission_s",
        "energy_violation_count",
        "fallback_used",
        "deferred",
        "defer_until_s",
    }
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


def stable_hex(*parts: str, length: int = 24) -> str:
    """Deterministic hex digest of string parts (sha256, NUL-joined)."""
    return hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:length]


def stable_seed(*parts: str) -> int:
    """Deterministic non-negative int seed derived from string parts."""
    return int(stable_hex(*parts, length=16), 16) & ((1 << 63) - 1)


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


# ---------------------------------------------------------------------------
# Shared data types and validation logic
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """JSON-safe conversion for dataclasses/paths/containers (stdlib-only).

    Mirrors orbitserve.common.serialization.to_jsonable; contract.py cannot
    import orbitserve internals, so the logic lives here for the shared data
    types' to_dict methods.
    """
    if is_dataclass(value):
        return {
            item.name: _to_jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def pick(data: dict[str, Any], key: str, default: Any) -> Any:
    return data[key] if key in data else default


def require_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping, got {type(value).__name__}")
    return value


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


@dataclass(frozen=True)
class TokenDistribution:
    name: str = "chatllm_smoke"
    prompt_tokens_p50: int = 512
    prompt_tokens_p95: int = 2048
    output_tokens_p50: int = 128
    output_tokens_p95: int = 512

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TokenDistribution":
        data = data or {}
        return cls(
            name=str(pick(data, "name", "chatllm_smoke")),
            prompt_tokens_p50=int(pick(data, "prompt_tokens_p50", 512)),
            prompt_tokens_p95=int(pick(data, "prompt_tokens_p95", 2048)),
            output_tokens_p50=int(pick(data, "output_tokens_p50", 128)),
            output_tokens_p95=int(pick(data, "output_tokens_p95", 512)),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class WorkloadBundleRef:
    """Experiment input reference to one canonical workload bundle."""

    bundle_dir: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WorkloadBundleRef":
        data = data or {}
        unsupported = sorted(set(data) - {"bundle_dir"})
        if unsupported:
            raise ValueError(
                "experiment workload accepts only a canonical bundle_dir; "
                f"unsupported fields: {unsupported}"
            )
        return cls(bundle_dir=optional_str(data.get("bundle_dir")))

    def to_dict(self) -> dict[str, Any]:
        return {"bundle_dir": self.bundle_dir}


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    region_name: str
    lat_deg: float
    lon_deg: float
    timezone_utc_offset_h: float = 0.0
    demand_weight: float = 1.0
    weight_source: str = "unspecified"
    market_class: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionSpec":
        return cls(
            region_id=str(data["region_id"]),
            region_name=str(pick(data, "region_name", data["region_id"])),
            lat_deg=float(data["lat_deg"]),
            lon_deg=float(data["lon_deg"]),
            timezone_utc_offset_h=float(pick(data, "timezone_utc_offset_h", 0.0)),
            demand_weight=float(pick(data, "demand_weight", 1.0)),
            weight_source=str(pick(data, "weight_source", "unspecified")),
            market_class=str(data["market_class"]) if data.get("market_class") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class GatewaySpec:
    gateway_id: str
    lat_deg: float
    lon_deg: float
    served_region_ids: tuple[str, ...] = ()
    gateway_bandwidth_bps: float | None = None
    availability_profile_id: str | None = None
    cost_usd_per_year: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GatewaySpec":
        return cls(
            gateway_id=str(data["gateway_id"]),
            lat_deg=float(data["lat_deg"]),
            lon_deg=float(data["lon_deg"]),
            served_region_ids=tuple(str(item) for item in data.get("served_region_ids", ())),
            gateway_bandwidth_bps=float(data["gateway_bandwidth_bps"]) if data.get("gateway_bandwidth_bps") is not None else None,
            availability_profile_id=str(data["availability_profile_id"]) if data.get("availability_profile_id") is not None else None,
            cost_usd_per_year=float(data["cost_usd_per_year"]) if data.get("cost_usd_per_year") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class WorkloadGenerationResult:
    bundle_dir: str | None
    manifest: dict[str, Any]
    regions: tuple[RegionSpec, ...]
    gateways: tuple[GatewaySpec, ...]
    requests: tuple[dict[str, Any], ...]
    sessions: tuple[dict[str, Any], ...]
    timebin_demand: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    evaluated: bool = True
    status: str = "generated"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "status": self.status,
            "bundle_dir": self.bundle_dir,
            "request_count": len(self.requests),
            "session_count": len(self.sessions),
            "manifest": _to_jsonable(self.manifest),
            "regions": [item.to_dict() for item in self.regions],
            "gateways": [item.to_dict() for item in self.gateways],
            "requests": [_to_jsonable(item) for item in self.requests],
            "sessions": [_to_jsonable(item) for item in self.sessions],
            "timebin_demand": [_to_jsonable(item) for item in self.timebin_demand],
            "summary": _to_jsonable(self.summary),
            "warnings": list(self.warnings),
        }


def reject_runtime_fields(value: Any, *, context: str) -> None:
    if isinstance(value, dict):
        leaked = sorted(str(key) for key in value if str(key) in RUNTIME_RESULT_FIELDS)
        if leaked:
            raise ValueError(f"{context} contains runtime-only field(s): {', '.join(leaked)}")
        for key, item in value.items():
            reject_runtime_fields(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_runtime_fields(item, context=f"{context}[{index}]")


def region_lookup(desc: dict[str, Any]) -> dict[str, RegionSpec]:
    geography_cfg = require_mapping(desc.get("geography"))
    return {
        region.region_id: region
        for region in (
            RegionSpec.from_dict(require_mapping(item))
            for item in geography_cfg.get("regions", ())
        )
    }


def gateway_lookup(desc: dict[str, Any]) -> dict[str, GatewaySpec]:
    gateway_cfg = require_mapping(desc.get("gateways"))
    return {
        gateway.gateway_id: gateway
        for gateway in (
            GatewaySpec.from_dict(require_mapping(item))
            for item in gateway_cfg.get("sites", ())
        )
    }


def input_bytes(new_user_tokens: int, token_cfg: dict[str, Any]) -> int:
    return int(
        new_user_tokens * int(pick(token_cfg, "bytes_per_input_token", 4))
        + int(pick(token_cfg, "request_metadata_bytes", 512))
    )


def output_bytes(output_tokens: int, token_cfg: dict[str, Any]) -> int:
    return int(
        output_tokens * int(pick(token_cfg, "bytes_per_output_token", 4))
        + int(pick(token_cfg, "response_metadata_bytes", 256))
    )


def coerce_request_row(
    row: dict[str, Any],
    *,
    seed: int,
    desc: dict[str, Any] | None = None,
    mark_fillable: bool = False,
) -> dict[str, Any]:
    reject_runtime_fields(row, context=f"request {row.get('request_id', '<unknown>')}")
    token_cfg = require_mapping(desc.get("tokens")) if desc else {}
    session_cfg = require_mapping(desc.get("sessions")) if desc else {}
    regions = region_lookup(desc) if desc else {}
    gateways = gateway_lookup(desc) if desc else {}
    gateway_cfg = require_mapping(desc.get("gateways")) if desc else {}
    data = dict(row)
    new_user_missing = "new_user_tokens" not in data
    context_missing = "context_tokens_before_turn" not in data
    input_bytes_missing = "input_bytes" not in data
    output_bytes_missing = "output_bytes" not in data
    if "user_region_id" not in data and "region_id" in data:
        data["user_region_id"] = data["region_id"]
    if "region_id" not in data and "user_region_id" in data:
        data["region_id"] = data["user_region_id"]
    region = regions.get(str(data.get("user_region_id")))
    if region is not None:
        data.setdefault("lat_deg", region.lat_deg)
        data.setdefault("lon_deg", region.lon_deg)
    if "new_user_tokens" not in data and "prompt_tokens" in data:
        data["new_user_tokens"] = data["prompt_tokens"]
    data.setdefault("context_tokens_before_turn", 0)
    data.setdefault("model_family", token_cfg.get("model_family") or token_cfg.get("model_family_fallback"))
    data.setdefault("access_mode", gateway_cfg.get("access_mode", "direct_user_sgl"))
    gateway = gateways.get(str(data.get("gateway_id")))
    if gateway is not None:
        data.setdefault("ingress_id", gateway.gateway_id)
        data.setdefault("ingress_lat_deg", gateway.lat_deg)
        data.setdefault("ingress_lon_deg", gateway.lon_deg)
    data.setdefault("ingress_id", data.get("gateway_id") or data.get("user_region_id"))
    data.setdefault("ingress_lat_deg", data.get("lat_deg"))
    data.setdefault("ingress_lon_deg", data.get("lon_deg"))
    data.setdefault("gateway_id", None)
    data.setdefault("priority", "normal")
    data.setdefault("session_id", None)
    data.setdefault("turn_index", 0 if data.get("session_id") in (None, "") else 1)
    data.setdefault("session_turn_count", None)
    data.setdefault("inter_turn_gap_s", None)
    data.setdefault("session_ttl_s", session_cfg.get("session_ttl_s"))
    data.setdefault("context_window_tokens", pick(session_cfg, "context_window_tokens", 32768))
    data.setdefault("burst_id", None)
    data.setdefault("burst_multiplier", 1.0)
    data.setdefault("seed", seed)
    if mark_fillable:
        data["_new_user_tokens_missing"] = new_user_missing
        data["_context_tokens_before_turn_missing"] = context_missing
        data["_input_bytes_missing"] = input_bytes_missing
        data["_output_bytes_missing"] = output_bytes_missing

    required = (
        "request_id",
        "arrival_time_s",
        "user_region_id",
        "region_id",
        "lat_deg",
        "lon_deg",
        "access_mode",
        "ingress_id",
        "ingress_lat_deg",
        "ingress_lon_deg",
        "task_type",
        "model_family",
        "prompt_tokens",
        "new_user_tokens",
        "context_tokens_before_turn",
        "output_tokens",
        "slo_class",
        "e2e_deadline_s",
        "context_window_tokens",
        "burst_multiplier",
        "seed",
    )
    missing = [key for key in required if data.get(key) is None or data.get(key) == ""]
    if missing:
        raise ValueError(f"explicit request row is missing required fields: {missing}")

    data["arrival_time_s"] = float(data["arrival_time_s"])
    data["lat_deg"] = float(data["lat_deg"])
    data["lon_deg"] = float(data["lon_deg"])
    data["ingress_lat_deg"] = float(data["ingress_lat_deg"])
    data["ingress_lon_deg"] = float(data["ingress_lon_deg"])
    for key in (
        "prompt_tokens",
        "new_user_tokens",
        "context_tokens_before_turn",
        "output_tokens",
        "turn_index",
        "context_window_tokens",
        "seed",
    ):
        data[key] = int(float(data[key]))
    for key in ("ttft_deadline_s", "e2e_deadline_s", "inter_turn_gap_s", "session_ttl_s", "burst_multiplier"):
        data[key] = optional_float(data.get(key))
    data["session_turn_count"] = (
        int(float(data["session_turn_count"])) if data.get("session_turn_count") is not None else None
    )
    if data["e2e_deadline_s"] is None:
        raise ValueError("explicit request row requires e2e_deadline_s")
    data["input_bytes"] = int(float(data["input_bytes"])) if data.get("input_bytes") is not None else input_bytes(int(data["new_user_tokens"]), token_cfg)
    data["output_bytes"] = int(float(data["output_bytes"])) if data.get("output_bytes") is not None else output_bytes(int(data["output_tokens"]), token_cfg)
    for key in (
        "request_id",
        "user_region_id",
        "region_id",
        "access_mode",
        "ingress_id",
        "task_type",
        "model_family",
        "slo_class",
    ):
        data[key] = str(data[key])
    if data.get("gateway_id") in ("", "None"):
        data["gateway_id"] = None
    if data.get("session_id") in ("", "None"):
        data["session_id"] = None
    if data.get("burst_id") in ("", "None"):
        data["burst_id"] = None
    return _to_jsonable(data)


def coerce_session_row(row: dict[str, Any], *, seed: int | None = None) -> dict[str, Any]:
    reject_runtime_fields(row, context=f"session {row.get('session_id', '<unknown>')}")
    data = dict(row)
    if seed is not None:
        data.setdefault("seed", seed)
    required = (
        "session_id",
        "user_region_id",
        "session_start_time_s",
        "session_turn_count",
        "session_slo_class",
        "context_window_tokens",
        "seed",
    )
    missing = [key for key in required if data.get(key) is None or data.get(key) == ""]
    if missing:
        raise ValueError(f"session row is missing required fields: {missing}")
    data["session_start_time_s"] = float(data["session_start_time_s"])
    data["session_turn_count"] = int(float(data["session_turn_count"]))
    data["context_window_tokens"] = int(float(data["context_window_tokens"]))
    data["seed"] = int(float(data["seed"]))
    data["session_ttl_s"] = optional_float(data.get("session_ttl_s"))
    return _to_jsonable(data)


def coerce_timebin_row(row: dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    data["time_bin_start_s"] = float(data["time_bin_start_s"])
    data["time_bin_end_s"] = float(data["time_bin_end_s"])
    for key in (
        "request_count",
        "prompt_tokens_total",
        "new_user_tokens_total",
        "output_tokens_total",
        "interactive_request_count",
        "burst_request_count",
    ):
        data[key] = int(float(data[key]))
    return data


def validate_session_consistency(
    requests: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in requests:
        session_id = row.get("session_id")
        prompt_tokens = int(row["prompt_tokens"])
        new_user_tokens = int(row["new_user_tokens"])
        context_tokens = int(row["context_tokens_before_turn"])
        if prompt_tokens != context_tokens + new_user_tokens:
            raise ValueError(
                f"request {row['request_id']} has prompt_tokens != context_tokens_before_turn + new_user_tokens"
            )
        if session_id is not None:
            grouped.setdefault(str(session_id), []).append(row)

    for session_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: (float(item["arrival_time_s"]), int(item["turn_index"])))
        turn_indices = [int(row["turn_index"]) for row in ordered]
        expected = list(range(turn_indices[0], turn_indices[0] + len(turn_indices)))
        if turn_indices != expected:
            raise ValueError(f"session {session_id} has non-contiguous turn_index values {turn_indices}")
        for row in ordered:
            declared_turn_count = row.get("session_turn_count")
            if declared_turn_count is not None and int(declared_turn_count) != len(rows):
                raise ValueError(
                    f"request {row['request_id']} declares session_turn_count={declared_turn_count}, "
                    f"but session {session_id} has {len(rows)} request rows"
                )

    session_rows = {str(row["session_id"]): row for row in sessions}
    for session_id, rows in grouped.items():
        if session_id in session_rows and int(session_rows[session_id]["session_turn_count"]) != len(rows):
            raise ValueError(
                f"session row {session_id} declares session_turn_count="
                f"{session_rows[session_id]['session_turn_count']}, but requests contain {len(rows)} rows"
            )


def validate_reference_resolution(
    requests: tuple[dict[str, Any], ...],
    sessions: tuple[dict[str, Any], ...],
    *,
    regions: tuple[RegionSpec, ...],
    gateways: tuple[GatewaySpec, ...],
) -> None:
    """Verify that request/session reference fields resolve to known regions/gateways."""

    region_ids = {region.region_id for region in regions}
    gateway_ids = {gateway.gateway_id for gateway in gateways}
    for row in requests:
        request_id = row.get("request_id")
        user_region_id = row.get("user_region_id")
        if user_region_id not in region_ids:
            raise ValueError(
                f"request {request_id}: user_region_id {user_region_id!r} is not a known region"
            )
        if row.get("region_id") != user_region_id:
            raise ValueError(
                f"request {request_id}: region_id {row.get('region_id')!r} does not match "
                f"user_region_id {user_region_id!r}"
            )
        gateway_id = row.get("gateway_id")
        if gateway_id is not None and gateway_id not in gateway_ids:
            raise ValueError(
                f"request {request_id}: gateway_id {gateway_id!r} is not a known gateway"
            )
        access_mode = row.get("access_mode")
        ingress_id = row.get("ingress_id")
        if access_mode == "gateway_sgl" and ingress_id != gateway_id:
            raise ValueError(
                f"request {request_id}: ingress_id {ingress_id!r} must equal gateway_id "
                f"{gateway_id!r} for access_mode='gateway_sgl'"
            )
        if access_mode == "direct_user_sgl" and ingress_id != user_region_id:
            raise ValueError(
                f"request {request_id}: ingress_id {ingress_id!r} must equal user_region_id "
                f"{user_region_id!r} for access_mode='direct_user_sgl'"
            )
    for row in sessions:
        user_region_id = row.get("user_region_id")
        if user_region_id not in region_ids:
            raise ValueError(
                f"session {row.get('session_id')}: user_region_id {user_region_id!r} "
                "is not a known region"
            )


def seed_from_manifest(manifest: dict[str, Any]) -> int:
    return int(manifest.get("seed") or pick(require_mapping(manifest.get("generator")), "seed", 1))


__all__ = [
    "BUNDLE_FILES",
    "BUNDLE_SCHEMA",
    "CANONICALIZATION",
    "DESCRIPTION_SCHEMA",
    "GATEWAY_KEYS",
    "REGION_KEYS",
    "REQUEST_FILES",
    "REQUEST_KEYS",
    "RUNTIME_RESULT_FIELDS",
    "SESSION_KEYS",
    "SUMMARY_SCHEMA",
    "TIMEBIN_FIELDS",
    "WORKLOAD_BUNDLE_SCHEMA",
    "WORKLOAD_DESCRIPTION_SCHEMA",
    "GatewaySpec",
    "RegionSpec",
    "TokenDistribution",
    "WorkloadBundleRef",
    "WorkloadGenerationResult",
    "bundle_hash",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "canonical_timebin_csv_bytes",
    "coerce_request_row",
    "coerce_session_row",
    "coerce_timebin_row",
    "descriptor",
    "input_bytes",
    "optional_float",
    "optional_str",
    "output_bytes",
    "pick",
    "reject_runtime_fields",
    "require_mapping",
    "seed_from_manifest",
    "sha256_bytes",
    "stable_hex",
    "stable_seed",
    "validate_exact_keys",
    "validate_manifest_and_files",
    "validate_reference_resolution",
    "validate_session_consistency",
]
