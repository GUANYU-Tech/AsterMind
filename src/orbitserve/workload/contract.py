"""Canonical OrbitServe workload bundle contract (shared layer).

This module is stdlib-only so dataset adapters can share it without pulling
optional dataset dependencies into the simulation package.
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
CAUSAL_BUNDLE_SCHEMA = "orbitserve.workload.causal.v1"
BUNDLE_SCHEMAS = frozenset({BUNDLE_SCHEMA, CAUSAL_BUNDLE_SCHEMA})
DESCRIPTION_SCHEMA = "orbitserve.workload_description"
SUMMARY_SCHEMA = "orbitserve.workload_summary"
CANONICALIZATION = "orbitserve-json-c14n"
WORKLOAD_BUNDLE_SCHEMA = BUNDLE_SCHEMA
WORKLOAD_DESCRIPTION_SCHEMA = DESCRIPTION_SCHEMA

# Five data files + workload_manifest.json = six total.
BUNDLE_FILES = {
    "regions":          ("regions.json",           "application/json"),
    "gateways":         ("gateways.json",          "application/json"),
    "session_requests": ("session_requests.jsonl", "application/x-ndjson"),
    "timebin_demand":   ("timebin_demand.csv",     "text/csv"),
    "summary":          ("workload_summary.json",  "application/json"),
}

MANIFEST_FILE = "workload_manifest.json"

SESSION_REQUEST_KEYS = (
    "request_id", "arrival_time_s", "user_region_id",
    "access_mode", "ingress_id", "gateway_id",
    "model_family",
    "prompt_tokens", "new_user_tokens", "context_tokens_before_turn",
    "output_tokens", "input_bytes", "output_bytes", "context_window_tokens",
    "slo_class", "ttft_deadline_s", "e2e_deadline_s",
    "session_id", "turn_index", "session_turn_count",
    "inter_turn_gap_s", "session_start_time_s", "session_ttl_s",
    "seed",
)

CAUSAL_SESSION_REQUEST_KEYS = SESSION_REQUEST_KEYS + (
    "arrival_time_mode", "depends_on_request_id",
    "source_trace_time_s", "source_api_time_s",
)

REGION_KEYS = (
    "region_id", "region_name", "lat_deg", "lon_deg", "demand_weight",
)

GATEWAY_KEYS = (
    "gateway_id", "lat_deg", "lon_deg", "served_region_ids",
)

TIMEBIN_FIELDS = (
    "time_bin_start_s", "time_bin_end_s", "region_id", "request_count",
    "prompt_tokens_total", "new_user_tokens_total", "output_tokens_total",
)

RUNTIME_RESULT_FIELDS = frozenset(
    {
        "assigned_satellite", "assigned_satellite_id", "assigned_deployment",
        "access_satellite_id", "selected_path", "session_action",
        "served", "served_ratio", "TTFT", "E2E",
        "ttft_s", "ttft_p95_s", "observed_ttft_s",
        "e2e_latency_s", "e2e_p95_s", "observed_e2e_s",
        "energy_violation", "energy_violation_fraction",
        "cost_per_request", "cost_per_served_request_usd", "cost_per_served_mtoken_usd",
        "soc_before", "soc_after", "sgl_queue_wait_s", "isl_queue_wait_s",
        "link_transmission_s", "energy_violation_count",
        "fallback_used", "deferred", "defer_until_s",
    }
)


# ── Canonical serialisation ────────────────────────────────────────

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


# ── Hashing ────────────────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hex(*parts: str, length: int = 24) -> str:
    return hashlib.sha256("\0".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


def stable_seed(*parts: str) -> int:
    return int(stable_hex(*parts, length=16), 16) & ((1 << 63) - 1)


def bundle_hash(workload_id: str, bundle_dir: Path) -> str:
    """Composite hash of all data files + workload_id.

    Computed from file contents directly — no per-file-SHA storage needed.
    """
    lines = [BUNDLE_SCHEMA, workload_id]
    for key, (name, _) in sorted(BUNDLE_FILES.items(), key=lambda kv: kv[1][0]):
        path = bundle_dir / name
        lines.append(f"{name}\t{sha256_bytes(path.read_bytes())}")
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


# ── Validation ─────────────────────────────────────────────────────

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
    if manifest.get("schema_version") not in BUNDLE_SCHEMAS:
        raise ValueError(
            f"unsupported workload schema {manifest.get('schema_version')!r}; "
            f"expected one of {sorted(BUNDLE_SCHEMAS)!r}"
        )
    if manifest.get("canonicalization") != CANONICALIZATION:
        raise ValueError("workload canonicalization is missing or unsupported")

    expected_files = {MANIFEST_FILE, *(name for name, _ in BUNDLE_FILES.values())}
    actual_files = {path.name for path in bundle_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            f"workload bundle must contain exactly {len(expected_files)} files; "
            f"actual={sorted(actual_files)}"
        )

    declared_hash = manifest.get("bundle_sha256")
    if not declared_hash:
        raise ValueError("workload manifest must contain bundle_sha256")
    computed = bundle_hash(str(manifest.get("workload_id")), bundle_dir)
    if declared_hash != computed:
        raise ValueError(
            f"workload bundle_sha256 mismatch: declared={declared_hash}, computed={computed}"
        )

    resolved: dict[str, Path] = {}
    for key, (name, _) in BUNDLE_FILES.items():
        resolved[key] = bundle_dir / name
    return resolved


def validate_session_consistency(requests: list[dict[str, Any]]) -> None:
    """Check prompt=context+new_user and turn_index contiguity within sessions."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    request_ids = {str(row.get("request_id")) for row in requests}
    for row in requests:
        prompt_tokens = int(row["prompt_tokens"])
        new_user_tokens = int(row["new_user_tokens"])
        context_tokens = int(row["context_tokens_before_turn"])
        if prompt_tokens != context_tokens + new_user_tokens:
            raise ValueError(
                f"request {row['request_id']} has prompt_tokens != "
                f"context_tokens_before_turn + new_user_tokens"
            )
        session_id = row.get("session_id")
        if session_id is not None:
            grouped.setdefault(str(session_id), []).append(row)

        if "arrival_time_mode" in row:
            mode = str(row["arrival_time_mode"])
            dependency = row.get("depends_on_request_id")
            if mode not in {"absolute", "after_dependency_completion"}:
                raise ValueError(f"request {row['request_id']} has unsupported arrival_time_mode={mode!r}")
            if mode == "absolute" and dependency is not None:
                raise ValueError(f"absolute request {row['request_id']} cannot declare a predecessor")
            if mode == "after_dependency_completion" and not dependency:
                raise ValueError(f"request {row['request_id']} requires depends_on_request_id")
            if mode == "after_dependency_completion" and row.get("inter_turn_gap_s") is None:
                raise ValueError(f"request {row['request_id']} requires inter_turn_gap_s")
            if row.get("inter_turn_gap_s") is not None and float(row["inter_turn_gap_s"]) < 0:
                raise ValueError(f"request {row['request_id']} has negative inter_turn_gap_s")
            if dependency is not None and str(dependency) not in request_ids:
                raise ValueError(
                    f"request {row['request_id']} depends on unknown request {dependency!r}"
                )
            if dependency is not None and str(dependency) == str(row["request_id"]):
                raise ValueError(f"request {row['request_id']} cannot depend on itself")

    for session_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda r: int(r["turn_index"]))
        turn_indices = [int(r["turn_index"]) for r in ordered]
        expected = list(range(turn_indices[0], turn_indices[0] + len(turn_indices)))
        if turn_indices != expected:
            raise ValueError(
                f"session {session_id} has non-contiguous turn_index values {turn_indices}"
            )
        for row in ordered:
            declared = row.get("session_turn_count")
            if declared is not None and int(declared) != len(rows):
                raise ValueError(
                    f"request {row['request_id']} declares session_turn_count={declared}, "
                    f"but session {session_id} has {len(rows)} request rows"
                )


def validate_reference_resolution(
    session_requests: tuple[dict[str, Any], ...],
    *,
    regions: tuple["RegionSpec", ...],
    gateways: tuple["GatewaySpec", ...],
) -> None:
    region_ids = {r.region_id for r in regions}
    gateway_ids = {g.gateway_id for g in gateways}
    for row in session_requests:
        rid = row.get("request_id")
        user_region_id = row.get("user_region_id")
        if user_region_id not in region_ids:
            raise ValueError(f"request {rid}: user_region_id {user_region_id!r} not a known region")
        gateway_id = row.get("gateway_id")
        if gateway_id is not None and gateway_id not in gateway_ids:
            raise ValueError(f"request {rid}: gateway_id {gateway_id!r} not a known gateway")
        access_mode = row.get("access_mode")
        ingress_id = row.get("ingress_id")
        if access_mode == "gateway_sgl" and ingress_id != gateway_id:
            raise ValueError(
                f"request {rid}: ingress_id {ingress_id!r} must equal "
                f"gateway_id {gateway_id!r} for access_mode='gateway_sgl'"
            )
        if access_mode == "direct_user_sgl" and ingress_id != user_region_id:
            raise ValueError(
                f"request {rid}: ingress_id {ingress_id!r} must equal "
                f"user_region_id {user_region_id!r} for access_mode='direct_user_sgl'"
            )


# ── Shared data types ──────────────────────────────────────────────

def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _to_jsonable(getattr(value, item.name)) for item in fields(value)}
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


def reject_runtime_fields(value: Any, *, context: str) -> None:
    if isinstance(value, dict):
        leaked = sorted(str(k) for k in value if str(k) in RUNTIME_RESULT_FIELDS)
        if leaked:
            raise ValueError(f"{context} contains runtime-only field(s): {', '.join(leaked)}")
        for key, item in value.items():
            reject_runtime_fields(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_runtime_fields(item, context=f"{context}[{index}]")


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
    demand_weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionSpec":
        return cls(
            region_id=str(data["region_id"]),
            region_name=str(pick(data, "region_name", data["region_id"])),
            lat_deg=float(data["lat_deg"]),
            lon_deg=float(data["lon_deg"]),
            demand_weight=float(pick(data, "demand_weight", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class GatewaySpec:
    gateway_id: str
    lat_deg: float
    lon_deg: float
    served_region_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GatewaySpec":
        return cls(
            gateway_id=str(data["gateway_id"]),
            lat_deg=float(data["lat_deg"]),
            lon_deg=float(data["lon_deg"]),
            served_region_ids=tuple(str(i) for i in data.get("served_region_ids", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class WorkloadGenerationResult:
    bundle_dir: str | None
    manifest: dict[str, Any]
    regions: tuple[RegionSpec, ...]
    gateways: tuple[GatewaySpec, ...]
    session_requests: tuple[dict[str, Any], ...]
    timebin_demand: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    evaluated: bool = True
    status: str = "generated"
    warnings: tuple[str, ...] = ()

    @property
    def requests(self) -> tuple[dict[str, Any], ...]:
        """Deprecated alias for session_requests."""
        return self.session_requests

    @property
    def sessions(self) -> tuple[dict[str, Any], ...]:
        """Deprecated compatibility property — sessions are now embedded in session_requests."""
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "status": self.status,
            "bundle_dir": self.bundle_dir,
            "request_count": len(self.session_requests),
            "manifest": _to_jsonable(self.manifest),
            "regions": [r.to_dict() for r in self.regions],
            "gateways": [g.to_dict() for g in self.gateways],
            "session_requests": [_to_jsonable(r) for r in self.session_requests],
            "timebin_demand": [_to_jsonable(t) for t in self.timebin_demand],
            "summary": _to_jsonable(self.summary),
            "warnings": list(self.warnings),
        }


# ── Row coercion ───────────────────────────────────────────────────

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


def region_lookup(desc: dict[str, Any]) -> dict[str, "RegionSpec"]:
    geography_cfg = require_mapping(desc.get("geography"))
    return {
        r.region_id: r
        for r in (
            RegionSpec.from_dict(require_mapping(item))
            for item in geography_cfg.get("regions", ())
        )
    }


def gateway_lookup(desc: dict[str, Any]) -> dict[str, "GatewaySpec"]:
    gateway_cfg = require_mapping(desc.get("gateways"))
    return {
        g.gateway_id: g
        for g in (
            GatewaySpec.from_dict(require_mapping(item))
            for item in gateway_cfg.get("sites", ())
        )
    }


def seed_from_manifest(manifest: dict[str, Any]) -> int:
    return int(manifest.get("seed", 1))


def coerce_session_request_row(
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
    gateway_cfg = require_mapping(desc.get("gateways")) if desc else {}
    data = dict(row)

    new_user_missing = "new_user_tokens" not in data
    input_bytes_missing = "input_bytes" not in data
    output_bytes_missing = "output_bytes" not in data

    region = regions.get(str(data.get("user_region_id")))
    if "new_user_tokens" not in data and "prompt_tokens" in data:
        data["new_user_tokens"] = data["prompt_tokens"]
    data.setdefault("context_tokens_before_turn", 0)
    data.setdefault("model_family", token_cfg.get("model_family") or token_cfg.get("model_family_fallback"))
    data.setdefault("access_mode", gateway_cfg.get("access_mode", "direct_user_sgl"))
    data.setdefault("ingress_id", data.get("gateway_id") or data.get("user_region_id"))
    data.setdefault("gateway_id", None)
    data.setdefault("session_id", None)
    data.setdefault("turn_index", 0 if data.get("session_id") in (None, "") else 1)
    data.setdefault("session_turn_count", None)
    data.setdefault("inter_turn_gap_s", None)
    data.setdefault("session_start_time_s", None)
    data.setdefault("session_ttl_s", session_cfg.get("session_ttl_s"))
    data.setdefault("context_window_tokens", pick(session_cfg, "context_window_tokens", 32768))
    data.setdefault("seed", seed)

    if mark_fillable:
        data["_new_user_tokens_missing"] = new_user_missing
        data["_input_bytes_missing"] = input_bytes_missing
        data["_output_bytes_missing"] = output_bytes_missing

    required = (
        "request_id", "arrival_time_s", "user_region_id",
        "access_mode", "ingress_id",
        "model_family", "prompt_tokens", "new_user_tokens",
        "context_tokens_before_turn", "output_tokens",
        "slo_class", "e2e_deadline_s", "context_window_tokens",
        "seed",
    )
    missing = [k for k in required if data.get(k) is None or data.get(k) == ""]
    if missing:
        raise ValueError(f"explicit request row is missing required fields: {missing}")

    data["arrival_time_s"] = float(data["arrival_time_s"])
    for key in (
        "prompt_tokens", "new_user_tokens", "context_tokens_before_turn",
        "output_tokens", "turn_index", "context_window_tokens", "seed",
    ):
        data[key] = int(float(data[key]))
    for key in ("ttft_deadline_s", "e2e_deadline_s", "inter_turn_gap_s", "session_ttl_s"):
        data[key] = optional_float(data.get(key))
    if "arrival_time_mode" in data:
        data["arrival_time_mode"] = str(data["arrival_time_mode"])
        data["depends_on_request_id"] = optional_str(data.get("depends_on_request_id"))
        for key in ("source_trace_time_s", "source_api_time_s"):
            data[key] = float(data[key])
    data["session_turn_count"] = (
        int(float(data["session_turn_count"])) if data.get("session_turn_count") is not None else None
    )
    if data["e2e_deadline_s"] is None:
        raise ValueError("explicit request row requires e2e_deadline_s")
    data["input_bytes"] = (
        int(float(data["input_bytes"])) if data.get("input_bytes") is not None
        else input_bytes(int(data["new_user_tokens"]), token_cfg)
    )
    data["output_bytes"] = (
        int(float(data["output_bytes"])) if data.get("output_bytes") is not None
        else output_bytes(int(data["output_tokens"]), token_cfg)
    )
    for key in ("request_id", "user_region_id", "access_mode", "ingress_id", "model_family", "slo_class"):
        data[key] = str(data[key])
    if data.get("gateway_id") in ("", "None"):
        data["gateway_id"] = None
    if data.get("session_id") in ("", "None"):
        data["session_id"] = None
        data["session_start_time_s"] = None
    elif data.get("session_start_time_s") is not None:
        data["session_start_time_s"] = float(data["session_start_time_s"])
    return _to_jsonable(data)


def coerce_timebin_row(row: dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    data["time_bin_start_s"] = float(data["time_bin_start_s"])
    data["time_bin_end_s"] = float(data["time_bin_end_s"])
    for key in (
        "request_count", "prompt_tokens_total",
        "new_user_tokens_total", "output_tokens_total",
    ):
        data[key] = int(float(data[key]))
    return data


# ── Legacy compatibility aliases ───────────────────────────────────
# These exist so that generation.py, upsample.py, and older adapters
# continue to import without changes.

REQUEST_KEYS = SESSION_REQUEST_KEYS
SESSION_KEYS = (
    "session_id", "user_region_id", "session_start_time_s", "session_turn_count",
    "session_slo_class", "context_window_tokens", "session_ttl_s", "seed",
)
REQUEST_FILES = {
    "regions": "regions.json",
    "gateways": "gateways.json",
    "requests": "session_requests.jsonl",
    "timebin_demand": "timebin_demand.csv",
    "summary": "workload_summary.json",
}


def coerce_request_row(*args, **kwargs):
    return coerce_session_request_row(*args, **kwargs)


def coerce_session_row(row, *, seed=None):
    reject_runtime_fields(row, context=f"session {row.get('session_id', '<unknown>')}")
    return _to_jsonable(dict(row))


def descriptor(*, path, media_type, payload, record_count):
    return {
        "path": path, "media_type": media_type,
        "sha256": sha256_bytes(payload), "record_count": int(record_count),
    }


# ── Public API ─────────────────────────────────────────────────────

__all__ = [
    "CAUSAL_BUNDLE_SCHEMA", "CAUSAL_SESSION_REQUEST_KEYS",
    "BUNDLE_FILES", "BUNDLE_SCHEMA", "BUNDLE_SCHEMAS", "CANONICALIZATION",
    "GATEWAY_KEYS", "MANIFEST_FILE", "REGION_KEYS",
    "RUNTIME_RESULT_FIELDS", "SESSION_REQUEST_KEYS", "TIMEBIN_FIELDS",
    "GatewaySpec", "RegionSpec", "TokenDistribution",
    "WorkloadBundleRef", "WorkloadGenerationResult",
    "bundle_hash", "canonical_json_bytes", "canonical_jsonl_bytes",
    "canonical_timebin_csv_bytes",
    "coerce_session_request_row", "coerce_timebin_row",
    "input_bytes", "optional_float", "optional_str", "output_bytes",
    "pick", "reject_runtime_fields", "require_mapping",
    "seed_from_manifest", "sha256_bytes", "stable_hex", "stable_seed",
    "validate_exact_keys", "validate_manifest_and_files",
    "validate_reference_resolution", "validate_session_consistency",
]
