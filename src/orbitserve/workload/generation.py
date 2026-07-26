"""Build-time workload generation tooling (inside the workload package).

Hosts the generation side of the former orbitserve.workload core:
generate_workload_bundle / compile_workload_description plus all
synthetic/trace/manifest assembly, write-back, and hashing helpers.
Shared data types and validation logic come from contract.py; bundle
loading/path resolution and materialize_workload_bundle live in loader.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import random
from typing import Any

from ..common import to_jsonable
from .contract import (
    BUNDLE_FILES, BUNDLE_SCHEMA, CANONICALIZATION, DESCRIPTION_SCHEMA,
    GATEWAY_KEYS, REGION_KEYS, REQUEST_FILES, REQUEST_KEYS, SESSION_KEYS,
    SUMMARY_SCHEMA, WORKLOAD_BUNDLE_SCHEMA, WORKLOAD_DESCRIPTION_SCHEMA,
    GatewaySpec, WorkloadGenerationResult, RegionSpec, WorkloadBundleRef,
    bundle_hash, canonical_json_bytes, canonical_jsonl_bytes,
    canonical_timebin_csv_bytes, descriptor,
    validate_exact_keys,
    coerce_request_row as _coerce_request_row,
    coerce_session_row as _coerce_session_row,
    input_bytes as _input_bytes,
    optional_float as _optional_float,
    optional_str as _optional_str,
    output_bytes as _output_bytes,
    pick as _pick,
    reject_runtime_fields as _reject_runtime_fields,
    require_mapping as _dict,
    validate_session_consistency as _validate_session_consistency,
)
from .loader import (
    _read_csv_rows,
    _read_jsonl,
    load_workload_bundle,
    resolve_bundle_path as _resolve_path,
)

GENERATOR_NAME = "orbitserve_workload_generator_or_loader"
GENERATOR_VERSION = "0.1.0"

SUPPORTED_SOURCE_MODES = frozenset(
    {"synthetic_generation", "explicit_requests", "trace_replay", "hybrid"}
)

def generate_workload_bundle(
    description: dict[str, Any] | str | Path | WorkloadGenerationResult,
    base_dir: str | Path | None = None,
) -> WorkloadGenerationResult:
    """Generate or normalize a canonical workload bundle from a description.

    The function returns the in-memory canonical bundle and writes it when the
    description has ``output.bundle_dir``. Paths inside the description are
    resolved relative to ``base_dir``; JSON is always supported, while YAML and
    Markdown require optional PyYAML for YAML parsing.
    """

    if isinstance(description, WorkloadGenerationResult):
        return description
    if isinstance(description, (str, Path)):
        return compile_workload_description(description, base_dir=base_dir)

    root = _base_dir(base_dir)
    desc = dict(description)
    _validate_description_header(desc)
    _reject_runtime_fields(desc, context="workload description")

    source_mode = str(desc["source_mode"])
    if source_mode == "synthetic_generation":
        result = _generate_synthetic_workload(desc, root)
    elif source_mode == "explicit_requests":
        result = _load_explicit_requests_workload(desc, root)
    elif source_mode in {"trace_replay", "hybrid"}:
        result = _load_replayed_trace_workload(desc, root, source_mode=source_mode)
    else:
        raise ValueError(f"unsupported source_mode: {source_mode!r}")

    result = _with_manifest(result)
    output = _dict(desc.get("output"))
    bundle_dir_value = output.get("bundle_dir")
    if bundle_dir_value is None:
        return result
    bundle_dir = _resolve_path(bundle_dir_value, root)
    result = replace(result, bundle_dir=str(bundle_dir))
    _write_workload_bundle(result, bundle_dir, overwrite=bool(output.get("overwrite", False)))
    return result

def compile_workload_description(
    path: str | Path,
    base_dir: str | Path | None = None,
) -> WorkloadGenerationResult:
    """Compile a JSON/YAML/Markdown workload description into a bundle."""

    description_path = Path(path).expanduser()
    if not description_path.is_absolute() and base_dir is not None:
        description_path = _resolve_path(description_path, _base_dir(base_dir))
    else:
        description_path = description_path.resolve()
    description_base_dir = _base_dir(base_dir) if base_dir is not None else description_path.parent.resolve()
    desc = _read_description_document(description_path)
    desc["_description_file"] = description_path.name
    desc["_description_sha256"] = _sha256_file(description_path)
    return generate_workload_bundle(desc, base_dir=description_base_dir)

def _generate_synthetic_workload(
    desc: dict[str, Any],
    base_dir: Path,
) -> WorkloadGenerationResult:
    time_cfg = _dict(desc.get("time"))
    arrival_cfg = _dict(desc.get("arrival"))
    token_cfg = _dict(desc.get("tokens"))
    session_cfg = _dict(desc.get("sessions"))
    validation_cfg = _dict(desc.get("validation"))
    session_mode = str(_pick(session_cfg, "session_mode", "open_loop_turn_arrivals"))
    if session_mode != "open_loop_turn_arrivals":
        raise NotImplementedError(
            "synthetic_generation currently supports session_mode='open_loop_turn_arrivals'; "
            "closed_loop_session_templates must be interpreted by a future runtime model"
        )
    workload_id = str(desc["workload_id"])
    seed = _description_seed(desc)
    rng = random.Random(seed)

    horizon_s = float(time_cfg["time_horizon_s"])
    time_bin_s = float(_pick(time_cfg, "time_bin_s", 60.0))
    if horizon_s <= 0 or time_bin_s <= 0:
        raise ValueError("time_horizon_s and time_bin_s must be positive")

    process = str(_pick(arrival_cfg, "process", "nhpp_synthetic"))
    if process not in {"nhpp_synthetic", "non_homogeneous_poisson"}:
        raise ValueError(
            "synthetic_generation requires arrival.process='nhpp_synthetic' "
            f"or legacy 'non_homogeneous_poisson', got {process!r}"
        )

    regions = _load_regions(desc, validation_cfg)
    gateways, access_mode, mapping_policy = _load_gateways(desc, regions)
    bursts = tuple(_dict(item) for item in desc.get("bursts", ()))
    for burst in bursts:
        shape = str(_pick(burst, "shape", "step"))
        if shape not in {"step", "ramp", "triangular"}:
            raise ValueError(
                f"unsupported burst shape {shape!r}; supported shapes are step, ramp, triangular"
            )

    lambda_global = float(_pick(arrival_cfg, "lambda_global", 0.0))
    load_scale = float(_pick(arrival_cfg, "load_scale", 1.0))
    if lambda_global < 0 or load_scale < 0:
        raise ValueError("lambda_global and load_scale must be non-negative")
    expected_rows_per_root_arrival = _expected_rows_per_root_arrival(session_cfg)
    root_lambda_global = (
        lambda_global / expected_rows_per_root_arrival
        if expected_rows_per_root_arrival > 0
        else lambda_global
    )

    diurnal_profile = str(_pick(arrival_cfg, "diurnal_profile", "mixed_chatllm"))
    max_generated = validation_cfg.get("max_generated_requests")
    max_generated_requests = int(max_generated) if max_generated is not None else None

    base_events: list[dict[str, Any]] = []
    bin_count = int(math.ceil(horizon_s / time_bin_s))
    for bin_index in range(bin_count):
        bin_start = bin_index * time_bin_s
        bin_end = min(horizon_s, bin_start + time_bin_s)
        bin_mid = (bin_start + bin_end) / 2.0
        for region in regions:
            local_hour = ((bin_mid / 3600.0) + region.timezone_utc_offset_h) % 24.0
            burst_multiplier, burst_id = _burst_multiplier_for(region.region_id, bin_mid, bursts)
            rate = (
                root_lambda_global
                * load_scale
                * region.demand_weight
                * _diurnal_factor(diurnal_profile, local_hour)
                * burst_multiplier
            )
            if rate <= 0:
                continue
            t = bin_start
            while True:
                t += rng.expovariate(rate)
                if t >= bin_end or t >= horizon_s:
                    break
                actual_burst_multiplier, actual_burst_id = _burst_multiplier_for(
                    region.region_id, t, bursts
                )
                base_events.append(
                    {
                        "arrival_time_s": t,
                        "region": region,
                        "burst_id": actual_burst_id,
                        "burst_multiplier": actual_burst_multiplier,
                    }
                )
                if max_generated_requests is not None and len(base_events) > max_generated_requests:
                    raise ValueError(
                        "synthetic workload exceeded validation.max_generated_requests="
                        f"{max_generated_requests}"
                    )

    requests: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    session_seq = 0
    for event in sorted(base_events, key=lambda item: (item["arrival_time_s"], item["region"].region_id)):
        event_rows, session_row = _build_synthetic_event_rows(
            event=event,
            workload_id=workload_id,
            session_seq=session_seq,
            rng=rng,
            seed=seed,
            horizon_s=horizon_s,
            token_cfg=token_cfg,
            slo_cfg=_dict(desc.get("slo")),
            session_cfg=session_cfg,
            gateways=gateways,
            access_mode=access_mode,
            mapping_policy=mapping_policy,
        )
        if session_row is not None:
            session_seq += 1
            sessions.append(session_row)
        requests.extend(event_rows)

    requests = _sort_and_number_requests(requests, workload_id=workload_id, seed=seed)
    if validation_cfg.get("require_unique_request_id", True):
        _require_unique_request_ids(requests)
    _validate_session_consistency(requests, sessions)
    timebin_demand = _build_timebin_demand(requests, horizon_s=horizon_s, time_bin_s=time_bin_s)
    summary = _build_summary(
        source_mode="synthetic_generation",
        workload_id=workload_id,
        requests=requests,
        sessions=sessions,
        regions=regions,
        gateways=gateways,
        timebin_demand=timebin_demand,
        seed=seed,
        generation_details={
            "lambda_global_target_request_per_s": lambda_global,
            "root_arrival_rate_per_s": root_lambda_global,
            "expected_rows_per_root_arrival": expected_rows_per_root_arrival,
            "arrival_rate_semantics": (
                "lambda_global targets final request rows per second; root arrivals are downscaled "
                "when session expansion can add follow-up turns"
            ),
        },
    )
    manifest = _build_manifest(
        desc=desc,
        source_mode="synthetic_generation",
        regions=regions,
        gateways=gateways,
        timebin_demand=timebin_demand,
        summary=summary,
        seed=seed,
        input_trace_files=(),
        base_dir=base_dir,
    )
    return WorkloadGenerationResult(
        bundle_dir=None,
        manifest=manifest,
        regions=regions,
        gateways=gateways,
        requests=tuple(requests),
        sessions=tuple(sessions),
        timebin_demand=tuple(timebin_demand),
        summary=summary,
    )

def _load_explicit_requests_workload(
    desc: dict[str, Any],
    base_dir: Path,
) -> WorkloadGenerationResult:
    explicit_cfg = _dict(desc.get("explicit_requests"))
    if not explicit_cfg:
        raise ValueError("source_mode='explicit_requests' requires explicit_requests section")
    seed = _description_seed(desc)
    time_cfg = _dict(desc.get("time"))
    horizon_s = float(time_cfg["time_horizon_s"])
    time_bin_s = float(_pick(time_cfg, "time_bin_s", 60.0))

    requests_path = _resolve_path(explicit_cfg["requests_path"], base_dir)
    request_format = _explicit_format(explicit_cfg.get("format"), requests_path)
    rows = _read_rows_by_format(requests_path, request_format)
    requests = [_coerce_request_row(row, seed=seed, desc=desc, mark_fillable=True) for row in rows]

    validation_cfg = _dict(desc.get("validation"))
    if validation_cfg.get("require_unique_request_id", True):
        _require_unique_request_ids(requests)
    if validation_cfg.get("require_sorted_requests", False):
        _require_sorted_requests(requests)
    if validation_cfg.get("sort_requests", True):
        requests = sorted(requests, key=lambda item: (float(item["arrival_time_s"]), str(item["request_id"])))
    _fill_explicit_session_context(requests, desc)
    _drop_internal_request_markers(requests)

    sessions_path_value = explicit_cfg.get("sessions_path")
    if sessions_path_value is None:
        sessions = _infer_sessions_from_requests(requests, desc)
        session_input_files: tuple[Path, ...] = ()
    else:
        sessions_path = _resolve_path(sessions_path_value, base_dir)
        session_format = _explicit_format(explicit_cfg.get("sessions_format", request_format), sessions_path)
        sessions = [_coerce_session_row(row, seed=seed) for row in _read_rows_by_format(sessions_path, session_format)]
        session_input_files = (sessions_path,)
    _validate_session_consistency(requests, sessions)

    regions = _load_regions_from_explicit(desc, requests)
    gateways, _, _ = _load_gateways(desc, regions, requests=requests)
    timebin_demand = _build_timebin_demand(requests, horizon_s=horizon_s, time_bin_s=time_bin_s)
    summary = _build_summary(
        source_mode="explicit_requests",
        workload_id=str(desc["workload_id"]),
        requests=requests,
        sessions=sessions,
        regions=regions,
        gateways=gateways,
        timebin_demand=timebin_demand,
        seed=seed,
        generation_details={},
    )
    input_files = (requests_path,) + session_input_files
    manifest = _build_manifest(
        desc=desc,
        source_mode="explicit_requests",
        regions=regions,
        gateways=gateways,
        timebin_demand=timebin_demand,
        summary=summary,
        seed=seed,
        input_trace_files=tuple(_source_file_record(path, base_dir) for path in input_files),
        base_dir=base_dir,
    )
    return WorkloadGenerationResult(
        bundle_dir=None,
        manifest=manifest,
        regions=regions,
        gateways=gateways,
        requests=tuple(requests),
        sessions=tuple(sessions),
        timebin_demand=tuple(timebin_demand),
        summary=summary,
    )

def _load_replayed_trace_workload(
    desc: dict[str, Any],
    base_dir: Path,
    *,
    source_mode: str,
) -> WorkloadGenerationResult:
    trace_cfg = _dict(desc.get("trace_replay"))
    if not trace_cfg:
        raise ValueError(f"source_mode={source_mode!r} requires trace_replay section")
    trace_path = _resolve_path(trace_cfg["path"], base_dir)
    trace_format = _explicit_format(trace_cfg.get("format"), trace_path)
    raw_rows = _read_rows_by_format(trace_path, trace_format)

    seed = _description_seed(desc)
    rng = random.Random(seed)
    time_cfg = _dict(desc.get("time"))
    horizon_s = float(time_cfg["time_horizon_s"])
    time_bin_s = float(_pick(time_cfg, "time_bin_s", 60.0))
    validation_cfg = _dict(desc.get("validation"))
    token_cfg = _dict(desc.get("tokens"))
    slo_cfg = _dict(desc.get("slo"))
    session_cfg = _dict(desc.get("sessions"))
    session_mode = str(_pick(session_cfg, "session_mode", "open_loop_turn_arrivals"))
    if session_mode != "open_loop_turn_arrivals":
        raise NotImplementedError(
            f"{source_mode} currently supports session_mode='open_loop_turn_arrivals'; "
            "closed_loop_session_templates must be interpreted by a future runtime model"
        )
    regions = _load_regions(desc, validation_cfg)
    gateways, access_mode, mapping_policy = _load_gateways(desc, regions)
    region_by_id = {region.region_id: region for region in regions}
    column_mapping = _dict(trace_cfg.get("column_mapping"))
    time_column = str(_pick(trace_cfg, "time_column", "arrival_time_s"))
    time_scale = float(_pick(trace_cfg, "time_scale", 1.0))
    time_unit = str(_pick(trace_cfg, "time_unit", "s")).lower()
    time_unit_scale = {"s": 1.0, "sec": 1.0, "second": 1.0, "ms": 0.001}.get(time_unit)
    if time_unit_scale is None:
        raise ValueError("trace_replay.time_unit must be s/sec/second or ms")
    request_id_column = trace_cfg.get("request_id_column")
    region_column = trace_cfg.get("region_column")

    requests: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        arrival = float(raw[time_column]) * time_unit_scale * time_scale
        if arrival < 0 or arrival > horizon_s:
            continue
        region = _trace_region(raw, regions, region_by_id, region_column, rng)
        task = _trace_task(raw, token_cfg, column_mapping, rng)
        slo = _trace_slo(raw, slo_cfg, column_mapping, rng)
        ingress = _ingress_for_region(region, gateways, access_mode, mapping_policy)
        row = {
            "request_id": str(raw.get(request_id_column)) if request_id_column and raw.get(request_id_column) is not None else f"{desc['workload_id']}_trace_{index:08d}",
            "arrival_time_s": round(arrival, 6),
            "user_region_id": region.region_id,
            "region_id": region.region_id,
            "lat_deg": region.lat_deg,
            "lon_deg": region.lon_deg,
            "access_mode": ingress["access_mode"],
            "ingress_id": ingress["ingress_id"],
            "ingress_lat_deg": ingress["ingress_lat_deg"],
            "ingress_lon_deg": ingress["ingress_lon_deg"],
            "gateway_id": ingress["gateway_id"],
            "task_type": task["task_type"],
            "model_family": str(
                raw.get(column_mapping.get("model_family", ""))
                or token_cfg.get("model_family")
                or token_cfg.get("model_family_fallback")
                or "llama_70b_like"
            ),
            "prompt_tokens": task["prompt_tokens"],
            "new_user_tokens": task["prompt_tokens"],
            "context_tokens_before_turn": 0,
            "output_tokens": task["output_tokens"],
            "slo_class": slo["slo_class"],
            "ttft_deadline_s": slo["ttft_deadline_s"],
            "e2e_deadline_s": slo["e2e_deadline_s"],
            "priority": slo["priority"],
            "session_id": raw.get(column_mapping.get("session_id", "")) or None,
            "turn_index": int(float(raw.get(column_mapping.get("turn_index", ""), 0) or 0)),
            "session_turn_count": None,
            "inter_turn_gap_s": None,
            "session_ttl_s": session_cfg.get("session_ttl_s"),
            "context_window_tokens": int(_pick(session_cfg, "context_window_tokens", 32768)),
            "burst_id": None,
            "burst_multiplier": 1.0,
            "seed": seed + index,
        }
        row["input_bytes"] = _input_bytes(int(row["new_user_tokens"]), token_cfg)
        row["output_bytes"] = _output_bytes(int(row["output_tokens"]), token_cfg)
        requests.append(_coerce_request_row(row, seed=seed, desc=desc))

    requests = sorted(requests, key=lambda item: (float(item["arrival_time_s"]), str(item["request_id"])))
    if validation_cfg.get("require_unique_request_id", True):
        _require_unique_request_ids(requests)
    _fill_explicit_session_context(requests, desc)
    _drop_internal_request_markers(requests)
    sessions = _infer_sessions_from_requests(requests, desc)
    _validate_session_consistency(requests, sessions)

    timebin_demand = _build_timebin_demand(requests, horizon_s=horizon_s, time_bin_s=time_bin_s)
    summary = _build_summary(
        source_mode=source_mode,
        workload_id=str(desc["workload_id"]),
        requests=requests,
        sessions=sessions,
        regions=regions,
        gateways=gateways,
        timebin_demand=timebin_demand,
        seed=seed,
        generation_details={"trace_rows_read": len(raw_rows), "trace_rows_kept": len(requests)},
    )
    manifest = _build_manifest(
        desc=desc,
        source_mode=source_mode,
        regions=regions,
        gateways=gateways,
        timebin_demand=timebin_demand,
        summary=summary,
        seed=seed,
        input_trace_files=(_source_file_record(trace_path, base_dir),),
        base_dir=base_dir,
    )
    return WorkloadGenerationResult(
        bundle_dir=None,
        manifest=manifest,
        regions=regions,
        gateways=gateways,
        requests=tuple(requests),
        sessions=tuple(sessions),
        timebin_demand=tuple(timebin_demand),
        summary=summary,
    )

def _build_synthetic_event_rows(
    *,
    event: dict[str, Any],
    workload_id: str,
    session_seq: int,
    rng: random.Random,
    seed: int,
    horizon_s: float,
    token_cfg: dict[str, Any],
    slo_cfg: dict[str, Any],
    session_cfg: dict[str, Any],
    gateways: tuple[GatewaySpec, ...],
    access_mode: str,
    mapping_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    region: RegionSpec = event["region"]
    session_ratio = float(_pick(session_cfg, "session_ratio", 0.0))
    is_session = rng.random() < session_ratio
    intended_turn_count = _sample_turn_count(session_cfg, rng) if is_session else 1
    intended_turn_count = max(1, int(intended_turn_count))
    session_id = f"{workload_id}_sess_{session_seq:06d}" if intended_turn_count > 1 else None
    session_ttl_s = _optional_float(session_cfg.get("session_ttl_s"))
    context_window_tokens = int(_pick(session_cfg, "context_window_tokens", 32768))
    slo = _sample_slo(slo_cfg, rng)

    rows: list[dict[str, Any]] = []
    arrival_time_s = float(event["arrival_time_s"])
    context_tokens = 0
    previous_gap_s: float | None = None
    for turn_index in range(1, intended_turn_count + 1):
        if turn_index > 1:
            previous_gap_s = _sample_gap(session_cfg, rng)
            arrival_time_s += previous_gap_s
        if arrival_time_s >= horizon_s:
            break
        task = _sample_task(token_cfg, rng)
        new_user_tokens = _sample_token_count(task["prompt_distribution"], rng)
        output_tokens = _sample_token_count(task["output_distribution"], rng)
        new_user_tokens = min(new_user_tokens, context_window_tokens)
        context_before = min(context_tokens, max(context_window_tokens - new_user_tokens, 0))
        prompt_tokens = context_before + new_user_tokens
        context_after = min(context_window_tokens, context_before + new_user_tokens + output_tokens)
        ingress = _ingress_for_region(region, gateways, access_mode, mapping_policy)
        row = {
            "request_id": "",
            "arrival_time_s": round(arrival_time_s, 6),
            "user_region_id": region.region_id,
            "region_id": region.region_id,
            "lat_deg": region.lat_deg,
            "lon_deg": region.lon_deg,
            "access_mode": ingress["access_mode"],
            "ingress_id": ingress["ingress_id"],
            "ingress_lat_deg": ingress["ingress_lat_deg"],
            "ingress_lon_deg": ingress["ingress_lon_deg"],
            "gateway_id": ingress["gateway_id"],
            "task_type": task["task_type"],
            "model_family": str(_pick(token_cfg, "model_family", "llama_70b_like")),
            "prompt_tokens": int(prompt_tokens),
            "new_user_tokens": int(new_user_tokens),
            "context_tokens_before_turn": int(context_before),
            "output_tokens": int(output_tokens),
            "input_bytes": _input_bytes(new_user_tokens, token_cfg),
            "output_bytes": _output_bytes(output_tokens, token_cfg),
            "slo_class": slo["slo_class"],
            "ttft_deadline_s": slo["ttft_deadline_s"],
            "e2e_deadline_s": slo["e2e_deadline_s"],
            "priority": slo["priority"],
            "session_id": session_id,
            "turn_index": turn_index if session_id else 0,
            "session_turn_count": intended_turn_count if session_id else None,
            "inter_turn_gap_s": previous_gap_s if session_id and turn_index > 1 else None,
            "session_ttl_s": session_ttl_s if session_id else None,
            "context_window_tokens": context_window_tokens,
            "burst_id": event.get("burst_id"),
            "burst_multiplier": float(event.get("burst_multiplier", 1.0)),
            "seed": seed,
        }
        rows.append(row)
        context_tokens = context_after

    if len(rows) <= 1:
        if rows:
            row = rows[0]
            row["session_id"] = None
            row["turn_index"] = 0
            row["session_turn_count"] = None
            row["inter_turn_gap_s"] = None
            row["session_ttl_s"] = None
        return rows, None

    actual_turn_count = len(rows)
    for row in rows:
        row["session_turn_count"] = actual_turn_count
    session_row = {
        "session_id": session_id,
        "user_region_id": region.region_id,
        "session_start_time_s": rows[0]["arrival_time_s"],
        "session_turn_count": actual_turn_count,
        "session_slo_class": slo["slo_class"],
        "session_ttl_s": session_ttl_s,
        "context_window_tokens": context_window_tokens,
        "seed": seed,
    }
    return rows, session_row

def _with_manifest(result: WorkloadGenerationResult) -> WorkloadGenerationResult:
    region_rows = [item.to_dict() for item in result.regions]
    gateway_rows = [item.to_dict() for item in result.gateways]
    request_rows = [dict(item) for item in result.requests]
    session_rows = [dict(item) for item in result.sessions]
    timebin_rows = [dict(item) for item in result.timebin_demand]
    validate_exact_keys(region_rows, REGION_KEYS, "regions")
    validate_exact_keys(gateway_rows, GATEWAY_KEYS, "gateways")
    validate_exact_keys(request_rows, REQUEST_KEYS, "requests")
    validate_exact_keys(session_rows, SESSION_KEYS, "sessions")
    payloads = {
        "regions": canonical_json_bytes(region_rows),
        "gateways": canonical_json_bytes(gateway_rows),
        "requests": canonical_jsonl_bytes(request_rows),
        "sessions": canonical_jsonl_bytes(session_rows),
        "timebin_demand": canonical_timebin_csv_bytes(timebin_rows),
        "summary": canonical_json_bytes(result.summary),
    }
    counts = {
        "regions": len(region_rows), "gateways": len(gateway_rows),
        "requests": len(request_rows), "sessions": len(session_rows),
        "timebin_demand": len(timebin_rows), "summary": 1,
    }
    descriptors = {
        key: descriptor(
            path=BUNDLE_FILES[key][0], media_type=BUNDLE_FILES[key][1],
            payload=payloads[key], record_count=counts[key],
        )
        for key in BUNDLE_FILES
    }
    previous = dict(result.manifest)
    previous_source = _dict(previous.get("source"))
    source_mode = str(previous.get("source_mode") or previous_source.get("source_mode") or "unknown")
    generator = _dict(previous.get("generator"))
    seed = int(previous.get("seed") or generator.get("seed") or 1)
    time_payload = _dict(previous.get("time"))
    provenance = _dict(previous.get("provenance"))
    manifest = {
        "schema_version": WORKLOAD_BUNDLE_SCHEMA,
        "workload_id": str(previous["workload_id"]),
        "source_mode": source_mode,
        "seed": seed,
        "module_version": GENERATOR_VERSION,
        "canonicalization": CANONICALIZATION,
        "generator": {**generator, "seed": seed},
        "time": {
            **time_payload,
            "epoch_kind": "absolute_utc" if str(time_payload.get("epoch", "")).endswith("Z") else "relative",
            "interval_semantics": "half_open",
        },
        "source": {
            "mode": source_mode,
            "inputs": list(previous_source.get("inputs") or previous_source.get("input_trace_files", ())),
        },
        "files": descriptors,
        "description_sha256": previous_source.get("description_sha256") or provenance.get("config_sha256"),
        "bundle_sha256": bundle_hash(str(previous["workload_id"]), descriptors),
        "counts": {
            "request_count": counts["requests"],
            "session_count": counts["sessions"],
            "region_count": counts["regions"],
            "gateway_count": counts["gateways"],
            "timebin_count": counts["timebin_demand"],
        },
        "provenance": provenance,
        "tokenizer": previous.get("tokenizer"),
        "validation": {
            "status": "validated",
            "checks": [
                "exact_keys",
                "sha256",
                "bundle_hash",
                "record_counts",
                "session_consistency",
                "reference_resolution",
            ],
        },
        "warnings": list(result.warnings),
    }
    for key in ("arrival", "geography", "tokens", "sessions", "slo", "burst_events", "summary_digest"):
        if key in previous:
            manifest[key] = previous[key]
    return replace(result, manifest=manifest)

def _build_manifest(
    *,
    desc: dict[str, Any],
    source_mode: str,
    regions: tuple[RegionSpec, ...],
    gateways: tuple[GatewaySpec, ...],
    timebin_demand: list[dict[str, Any]],
    summary: dict[str, Any],
    seed: int,
    input_trace_files: tuple[dict[str, Any], ...],
    base_dir: Path,
) -> dict[str, Any]:
    time_cfg = _dict(desc.get("time"))
    geography_cfg = _dict(desc.get("geography"))
    gateway_cfg = _dict(desc.get("gateways"))
    arrival_cfg = _dict(desc.get("arrival"))
    token_cfg = _dict(desc.get("tokens"))
    session_cfg = _dict(desc.get("sessions"))
    slo_cfg = _dict(desc.get("slo"))
    provenance_cfg = _dict(desc.get("provenance"))
    description_hash = desc.get("_description_sha256") or _sha256_text(
        json.dumps(to_jsonable(_without_internal_keys(desc)), sort_keys=True, ensure_ascii=False)
    )
    return {
        "schema_version": WORKLOAD_BUNDLE_SCHEMA,
        "workload_id": str(desc["workload_id"]),
        "source": {
            "source_mode": source_mode,
            "description_schema_version": str(desc.get("schema_version", WORKLOAD_DESCRIPTION_SCHEMA)),
            "description_file": desc.get("_description_file"),
            "description_sha256": description_hash,
            "input_trace_files": list(input_trace_files),
        },
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "seed": seed,
        },
        "time": {
            "epoch": str(_pick(time_cfg, "epoch", "synthetic_day_0")),
            "time_horizon_s": float(time_cfg["time_horizon_s"]),
            "time_bin_s": float(_pick(time_cfg, "time_bin_s", 60.0)),
        },
        "geography": {
            "mode": str(_pick(geography_cfg, "mode", "named_gateway_service_regions")),
            "region_count": len(regions),
            "gateway_mode": str(_pick(gateway_cfg, "access_mode", "direct_user_sgl")),
        },
        "arrival": {
            "arrival_process": str(_pick(arrival_cfg, "process", source_mode)),
            "lambda_global": _optional_float(arrival_cfg.get("lambda_global")),
            "load_scale": float(_pick(arrival_cfg, "load_scale", 1.0)),
            "diurnal_profile": str(_pick(arrival_cfg, "diurnal_profile", "flat")),
        },
        "tokens": {
            "model_family": token_cfg.get("model_family") or token_cfg.get("model_family_fallback"),
            "task_mix": _task_mix_label(token_cfg),
        },
        "tokenizer": to_jsonable(token_cfg.get("tokenizer")),
        "sessions": {
            "session_mode": str(_pick(session_cfg, "session_mode", "open_loop_turn_arrivals")),
            "session_ratio": _optional_float(session_cfg.get("session_ratio")),
            "turn_count_distribution": _distribution_label(
                session_cfg.get("turn_count_distribution", "single_turn_only")
            ),
            "gap_distribution": _distribution_label(session_cfg.get("gap_distribution", "mixed_gap")),
        },
        "slo": {"slo_mix": str(_pick(slo_cfg, "mix", "custom"))},
        "burst_events": [to_jsonable(item) for item in desc.get("bursts", ())],
        "files": dict(REQUEST_FILES),
        "summary_digest": {
            "request_count": summary["request_count"],
            "session_count": summary["session_count"],
            "timebin_count": len(timebin_demand),
        },
        "provenance": {
            "sources": list(provenance_cfg.get("sources", ())),
            "notes": list(provenance_cfg.get("notes", ())),
            "config_sha256": description_hash,
            "base_dir": str(base_dir),
            "expanded_description": to_jsonable(_without_internal_keys(desc)),
        },
    }

def _write_workload_bundle(
    result: WorkloadGenerationResult,
    bundle_dir: Path,
    *,
    overwrite: bool,
) -> None:
    if bundle_dir.exists() and not overwrite and (bundle_dir / "workload_manifest.json").exists():
        raise FileExistsError(
            f"workload bundle already exists at {bundle_dir}; set output.overwrite=true to replace bundle files"
        )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / REQUEST_FILES["regions"]).write_bytes(
        canonical_json_bytes([item.to_dict() for item in result.regions])
    )
    (bundle_dir / REQUEST_FILES["gateways"]).write_bytes(
        canonical_json_bytes([item.to_dict() for item in result.gateways])
    )
    (bundle_dir / REQUEST_FILES["requests"]).write_bytes(canonical_jsonl_bytes(result.requests))
    (bundle_dir / REQUEST_FILES["sessions"]).write_bytes(canonical_jsonl_bytes(result.sessions))
    (bundle_dir / REQUEST_FILES["timebin_demand"]).write_bytes(
        canonical_timebin_csv_bytes(result.timebin_demand)
    )
    (bundle_dir / REQUEST_FILES["summary"]).write_bytes(canonical_json_bytes(result.summary))
    (bundle_dir / "workload_manifest.json").write_bytes(canonical_json_bytes(result.manifest))

def _load_regions(
    desc: dict[str, Any],
    validation_cfg: dict[str, Any],
) -> tuple[RegionSpec, ...]:
    geography_cfg = _dict(desc.get("geography"))
    region_items = geography_cfg.get("regions")
    if region_items:
        regions = tuple(RegionSpec.from_dict(_dict(item)) for item in region_items)
    else:
        regions = _default_regions(
            count=int(_pick(geography_cfg, "region_count", 4)),
            weight_source=str(_pick(geography_cfg, "mode", "default_named_regions")),
        )
    return _normalize_regions(regions, geography_cfg, validation_cfg)

def _load_regions_from_explicit(
    desc: dict[str, Any],
    requests: list[dict[str, Any]],
) -> tuple[RegionSpec, ...]:
    geography_cfg = _dict(desc.get("geography"))
    validation_cfg = _dict(desc.get("validation"))
    if geography_cfg.get("regions"):
        return _load_regions(desc, validation_cfg)

    grouped: dict[str, dict[str, Any]] = {}
    for row in requests:
        region_id = str(row["user_region_id"])
        grouped.setdefault(
            region_id,
            {
                "region_id": region_id,
                "region_name": region_id,
                "lat_deg": float(row["lat_deg"]),
                "lon_deg": float(row["lon_deg"]),
                "timezone_utc_offset_h": 0.0,
                "demand_weight": 0.0,
                "weight_source": "explicit_requests",
            },
        )
        grouped[region_id]["demand_weight"] += 1.0
    if not grouped:
        raise ValueError("explicit request workload contains no requests from which to infer regions")
    total = sum(item["demand_weight"] for item in grouped.values())
    regions = tuple(
        RegionSpec.from_dict({**item, "demand_weight": item["demand_weight"] / total})
        for item in grouped.values()
    )
    return _normalize_regions(regions, geography_cfg, validation_cfg)

def _normalize_regions(
    regions: tuple[RegionSpec, ...],
    geography_cfg: dict[str, Any],
    validation_cfg: dict[str, Any],
) -> tuple[RegionSpec, ...]:
    if not regions:
        raise ValueError("at least one region is required")
    total = sum(region.demand_weight for region in regions)
    if total <= 0:
        raise ValueError("region demand weights must sum to a positive value")
    require_sum_one = geography_cfg.get("demand_weight_normalization") == "require_sum_one"
    normalize_flag = validation_cfg.get("normalize_region_weights")
    if normalize_flag is None:
        normalize_flag = not require_sum_one
    if abs(total - 1.0) > 1e-6:
        if require_sum_one and not normalize_flag:
            raise ValueError(f"region demand weights must sum to 1.0, got {total}")
        if not normalize_flag:
            raise ValueError(
                f"region demand weights sum to {total}; enable validation.normalize_region_weights"
            )
        regions = tuple(replace(region, demand_weight=region.demand_weight / total) for region in regions)
    return regions

def _load_gateways(
    desc: dict[str, Any],
    regions: tuple[RegionSpec, ...],
    *,
    requests: list[dict[str, Any]] | None = None,
) -> tuple[tuple[GatewaySpec, ...], str, str]:
    gateway_cfg = _dict(desc.get("gateways"))
    access_mode = str(_pick(gateway_cfg, "access_mode", "direct_user_sgl"))
    mapping_policy = str(_pick(gateway_cfg, "mapping_policy", "nearest_gateway"))
    site_items = gateway_cfg.get("sites") or ()
    gateways = tuple(GatewaySpec.from_dict(_dict(item)) for item in site_items)

    if not gateways and requests:
        gateways = _infer_gateways_from_requests(requests)
    if access_mode == "gateway_sgl" and not gateways:
        raise ValueError("gateways.sites are required when access_mode='gateway_sgl'")
    if access_mode not in {"direct_user_sgl", "gateway_sgl"}:
        raise ValueError(f"unsupported access_mode: {access_mode!r}")
    return gateways, access_mode, mapping_policy

def _infer_gateways_from_requests(requests: list[dict[str, Any]]) -> tuple[GatewaySpec, ...]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in requests:
        gateway_id = row.get("gateway_id")
        if gateway_id is None or gateway_id == "":
            continue
        grouped.setdefault(
            str(gateway_id),
            {
                "gateway_id": str(gateway_id),
                "lat_deg": float(row["ingress_lat_deg"]),
                "lon_deg": float(row["ingress_lon_deg"]),
                "served_region_ids": set(),
            },
        )
        grouped[str(gateway_id)]["served_region_ids"].add(str(row["user_region_id"]))
    return tuple(
        GatewaySpec.from_dict(
            {
                **item,
                "served_region_ids": sorted(item["served_region_ids"]),
            }
        )
        for item in grouped.values()
    )

def _default_regions(count: int, weight_source: str) -> tuple[RegionSpec, ...]:
    seeds = (
        ("na_east", "North America East", 39.0, -77.0, -5.0, "core_cloud"),
        ("na_west", "North America West", 37.4, -122.1, -8.0, "core_cloud"),
        ("europe_west", "Europe West", 50.1, 8.7, 1.0, "core_cloud"),
        ("east_asia", "East Asia", 35.7, 139.7, 9.0, "core_cloud"),
        ("south_asia", "South Asia", 19.1, 72.9, 5.5, "growth_cloud"),
        ("southeast_asia", "Southeast Asia", 1.3, 103.8, 8.0, "growth_cloud"),
        ("south_america", "South America", -23.6, -46.6, -3.0, "growth_cloud"),
        ("middle_east", "Middle East", 25.2, 55.3, 4.0, "growth_cloud"),
        ("africa", "Africa", -1.3, 36.8, 3.0, "sensitivity"),
        ("oceania", "Oceania", -33.9, 151.2, 10.0, "sensitivity"),
    )
    count = max(1, count)
    items: list[tuple[str, str, float, float, float, str]] = list(seeds[: min(count, len(seeds))])
    while len(items) < count:
        index = len(items)
        lat = -60.0 + (index * 37.0) % 120.0
        lon = -170.0 + (index * 53.0) % 340.0
        items.append((f"region_{index:02d}", f"Synthetic Region {index}", lat, lon, round(lon / 15.0), "synthetic"))
    weight = 1.0 / count
    return tuple(
        RegionSpec(
            region_id=region_id,
            region_name=name,
            lat_deg=lat,
            lon_deg=lon,
            timezone_utc_offset_h=tz,
            demand_weight=weight,
            weight_source=weight_source,
            market_class=market_class,
        )
        for region_id, name, lat, lon, tz, market_class in items
    )

def _ingress_for_region(
    region: RegionSpec,
    gateways: tuple[GatewaySpec, ...],
    access_mode: str,
    mapping_policy: str,
) -> dict[str, Any]:
    if access_mode == "direct_user_sgl":
        return {
            "access_mode": "direct_user_sgl",
            "ingress_id": region.region_id,
            "ingress_lat_deg": region.lat_deg,
            "ingress_lon_deg": region.lon_deg,
            "gateway_id": None,
        }

    candidates = [item for item in gateways if region.region_id in item.served_region_ids] or list(gateways)
    if not candidates:
        raise ValueError(f"no gateway is available for region {region.region_id!r}")
    if mapping_policy == "first_listed_gateway":
        gateway = candidates[0]
    else:
        gateway = min(
            candidates,
            key=lambda item: _geo_distance_sq(region.lat_deg, region.lon_deg, item.lat_deg, item.lon_deg),
        )
    return {
        "access_mode": "gateway_sgl",
        "ingress_id": gateway.gateway_id,
        "ingress_lat_deg": gateway.lat_deg,
        "ingress_lon_deg": gateway.lon_deg,
        "gateway_id": gateway.gateway_id,
    }

def _sample_task(token_cfg: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    task_mix = token_cfg.get("task_mix") or _default_task_mix()
    task = _choose_weighted([_dict(item) for item in task_mix], rng)
    return {
        "task_type": str(_pick(task, "task_type", "conversation")),
        "prompt_distribution": _dict(task.get("prompt_distribution") or _default_task_mix()[0]["prompt_distribution"]),
        "output_distribution": _dict(task.get("output_distribution") or _default_task_mix()[0]["output_distribution"]),
    }

def _trace_region(
    row: dict[str, Any],
    regions: tuple[RegionSpec, ...],
    region_by_id: dict[str, RegionSpec],
    region_column: Any,
    rng: random.Random,
) -> RegionSpec:
    if region_column and row.get(region_column) is not None:
        region_id = str(row[region_column])
        if region_id not in region_by_id:
            raise ValueError(f"trace row references unknown region_id {region_id!r}")
        return region_by_id[region_id]
    return RegionSpec.from_dict(
        _choose_weighted(
            [
                {
                    **region.to_dict(),
                    "weight": region.demand_weight,
                }
                for region in regions
            ],
            rng,
        )
    )

def _trace_task(
    row: dict[str, Any],
    token_cfg: dict[str, Any],
    column_mapping: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    sampled = _sample_task(token_cfg, rng)
    prompt_column = column_mapping.get("prompt_tokens")
    output_column = column_mapping.get("output_tokens")
    task_column = column_mapping.get("task_type")
    prompt_tokens = (
        int(float(row[prompt_column]))
        if prompt_column and row.get(prompt_column) is not None
        else _sample_token_count(sampled["prompt_distribution"], rng)
    )
    output_tokens = (
        int(float(row[output_column]))
        if output_column and row.get(output_column) is not None
        else _sample_token_count(sampled["output_distribution"], rng)
    )
    return {
        "task_type": str(row.get(task_column) or sampled["task_type"]) if task_column else sampled["task_type"],
        "prompt_tokens": max(1, prompt_tokens),
        "output_tokens": max(1, output_tokens),
    }

def _trace_slo(
    row: dict[str, Any],
    slo_cfg: dict[str, Any],
    column_mapping: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    sampled = _sample_slo(slo_cfg, rng)
    slo_column = column_mapping.get("slo_class")
    ttft_column = column_mapping.get("ttft_deadline_s")
    e2e_column = column_mapping.get("e2e_deadline_s")
    priority_column = column_mapping.get("priority")
    if slo_column and row.get(slo_column):
        sampled = {**sampled, "slo_class": str(row[slo_column])}
    if ttft_column and row.get(ttft_column) not in (None, ""):
        sampled = {**sampled, "ttft_deadline_s": float(row[ttft_column])}
    if e2e_column and row.get(e2e_column) not in (None, ""):
        sampled = {**sampled, "e2e_deadline_s": float(row[e2e_column])}
    if priority_column and row.get(priority_column) not in (None, ""):
        sampled = {**sampled, "priority": row[priority_column]}
    return sampled

def _sample_slo(slo_cfg: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    classes = slo_cfg.get("classes")
    if not classes:
        classes = _slo_mix_preset(str(_pick(slo_cfg, "mix", "balanced")))
    item = _choose_weighted([_dict(entry) for entry in classes], rng)
    return {
        "slo_class": str(item.get("slo_class") or item.get("name") or "interactive_5s"),
        "ttft_deadline_s": _optional_float(item.get("ttft_deadline_s")),
        "e2e_deadline_s": float(_pick(item, "e2e_deadline_s", 30.0)),
        "priority": item.get("priority", "normal"),
    }

def _expected_rows_per_root_arrival(session_cfg: dict[str, Any]) -> float:
    session_ratio = max(0.0, min(1.0, float(_pick(session_cfg, "session_ratio", 0.0))))
    spec = session_cfg.get("turn_count_distribution", "single_turn_only")
    if isinstance(spec, str):
        values = _turn_count_preset(spec)
    else:
        spec_dict = _dict(spec)
        if spec_dict.get("type") != "categorical":
            return 1.0
        values = {int(key): float(value) for key, value in _dict(spec_dict.get("values")).items()}
    total_weight = sum(values.values())
    if total_weight <= 0:
        return 1.0
    expected_turns = sum(max(1, turn_count) * weight for turn_count, weight in values.items()) / total_weight
    return (1.0 - session_ratio) + session_ratio * expected_turns

def _sample_turn_count(session_cfg: dict[str, Any], rng: random.Random) -> int:
    spec = session_cfg.get("turn_count_distribution", "single_turn_only")
    if isinstance(spec, str):
        values = _turn_count_preset(spec)
    else:
        spec_dict = _dict(spec)
        if spec_dict.get("type") != "categorical":
            raise ValueError("turn_count_distribution must be categorical or a named preset")
        values = {int(key): float(value) for key, value in _dict(spec_dict.get("values")).items()}
    return int(_choose_weighted([{"value": key, "weight": value} for key, value in values.items()], rng)["value"])

def _sample_gap(session_cfg: dict[str, Any], rng: random.Random) -> float:
    spec = session_cfg.get("gap_distribution", "mixed_gap")
    if isinstance(spec, str):
        spec = _gap_preset(spec)
    spec_dict = _dict(spec)
    if spec_dict.get("type") == "constant":
        return float(_pick(spec_dict, "value_s", 0.0))
    if spec_dict.get("type") != "bucketed_uniform":
        raise ValueError("gap_distribution must be bucketed_uniform, constant, or a named preset")
    bucket = _choose_weighted([_dict(item) for item in spec_dict.get("buckets", ())], rng)
    return round(rng.uniform(float(bucket["min_s"]), float(bucket["max_s"])), 6)

def _sample_token_count(distribution: dict[str, Any], rng: random.Random) -> int:
    kind = str(_pick(distribution, "type", "quantile_buckets"))
    if kind == "quantile_buckets":
        p50 = int(_pick(distribution, "p50", 512))
        p95 = int(_pick(distribution, "p95", max(p50, 2048)))
        p99 = int(_pick(distribution, "p99", p95))
        max_value = int(_pick(distribution, "max", p99))
        min_value = int(_pick(distribution, "min", max(1, p50 // 4)))
        u = rng.random()
        if u < 0.50:
            low, high = min_value, p50
        elif u < 0.95:
            low, high = p50, p95
        elif u < 0.99:
            low, high = p95, p99
        else:
            low, high = p99, max_value
        low, high = sorted((max(1, low), max(1, high)))
        return int(round(rng.uniform(low, high)))
    if kind == "lognormal":
        value = rng.lognormvariate(float(distribution["mu"]), float(distribution["sigma"]))
        return int(round(_clamp(value, float(_pick(distribution, "min", 1)), float(distribution["max"]))))
    raise ValueError(
        f"unsupported token distribution {kind!r}; supported types are quantile_buckets and lognormal"
    )

def _choose_weighted(items: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    if not items:
        raise ValueError("weighted choice requires at least one item")
    total = sum(float(_pick(item, "weight", 1.0)) for item in items)
    if total <= 0:
        raise ValueError("weighted choice weights must sum to a positive value")
    needle = rng.random() * total
    seen = 0.0
    for item in items:
        seen += float(_pick(item, "weight", 1.0))
        if needle <= seen:
            return item
    return items[-1]

def _diurnal_factor(profile: str, local_hour: float) -> float:
    raw = _raw_diurnal(profile, local_hour)
    norm = _diurnal_norm(profile)
    return raw / norm if norm > 0 else 1.0

def _raw_diurnal(profile: str, local_hour: float) -> float:
    if profile == "flat":
        return 1.0
    if profile == "consumer_chat":
        return 0.35 + 1.35 * _periodic_gaussian(local_hour, 21.0, 3.2) + 0.30 * _periodic_gaussian(local_hour, 12.0, 5.0)
    if profile == "enterprise_workday":
        return 0.25 + 1.55 * _periodic_gaussian(local_hour, 11.0, 3.5) + 0.55 * _periodic_gaussian(local_hour, 15.5, 2.8)
    if profile == "mixed_chatllm":
        return 0.55 * _raw_diurnal("consumer_chat", local_hour) + 0.45 * _raw_diurnal("enterprise_workday", local_hour)
    raise ValueError(
        f"unsupported diurnal_profile {profile!r}; supported profiles are flat, consumer_chat, enterprise_workday, mixed_chatllm"
    )

def _diurnal_norm(profile: str) -> float:
    samples = [_raw_diurnal(profile, index * 24.0 / 288.0) for index in range(288)]
    return sum(samples) / len(samples)

def _periodic_gaussian(hour: float, center: float, sigma: float) -> float:
    delta = abs((hour - center + 12.0) % 24.0 - 12.0)
    return math.exp(-((delta * delta) / (sigma * sigma)))

def _burst_multiplier_for(
    region_id: str,
    time_s: float,
    bursts: tuple[dict[str, Any], ...],
) -> tuple[float, str | None]:
    multiplier = 1.0
    active_ids: list[str] = []
    for burst in bursts:
        region_ids = tuple(str(item) for item in burst.get("region_ids", ()))
        if region_ids and region_id not in region_ids:
            continue
        start = float(burst["start_time_s"])
        duration = float(burst["duration_s"])
        if duration <= 0 or time_s < start or time_s >= start + duration:
            continue
        phase = (time_s - start) / duration
        target = float(_pick(burst, "multiplier", 1.0))
        shape = str(_pick(burst, "shape", "step"))
        if shape == "step":
            shaped = target
        elif shape == "ramp":
            shaped = 1.0 + (target - 1.0) * phase
        elif shape == "triangular":
            shaped = 1.0 + (target - 1.0) * (1.0 - abs(2.0 * phase - 1.0))
        else:
            raise ValueError(f"unsupported burst shape: {shape!r}")
        multiplier *= shaped
        active_ids.append(str(_pick(burst, "burst_id", "burst")))
    return multiplier, "+".join(active_ids) if active_ids else None

def _fill_explicit_session_context(requests: list[dict[str, Any]], desc: dict[str, Any]) -> None:
    token_cfg = _dict(desc.get("tokens"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    single_rows: list[dict[str, Any]] = []
    for row in requests:
        session_id = row.get("session_id")
        if session_id is None:
            single_rows.append(row)
        else:
            grouped.setdefault(str(session_id), []).append(row)

    for row in single_rows:
        _fill_explicit_context_row(row, previous_context_tokens=0, token_cfg=token_cfg)

    for rows in grouped.values():
        previous_context_tokens = 0
        for row in sorted(rows, key=lambda item: (float(item["arrival_time_s"]), int(item.get("turn_index") or 0))):
            previous_context_tokens = _fill_explicit_context_row(
                row,
                previous_context_tokens=previous_context_tokens,
                token_cfg=token_cfg,
            )

def _fill_explicit_context_row(
    row: dict[str, Any],
    *,
    previous_context_tokens: int,
    token_cfg: dict[str, Any],
) -> int:
    prompt_tokens = int(row["prompt_tokens"])
    output_tokens = int(row["output_tokens"])
    context_window_tokens = int(row["context_window_tokens"])

    context_missing = bool(row.get("_context_tokens_before_turn_missing"))
    new_user_missing = bool(row.get("_new_user_tokens_missing"))
    if context_missing and not new_user_missing:
        context_before = max(0, prompt_tokens - int(row["new_user_tokens"]))
    elif context_missing:
        context_before = min(previous_context_tokens, prompt_tokens, context_window_tokens)
    else:
        context_before = int(row["context_tokens_before_turn"])

    if new_user_missing:
        new_user_tokens = max(0, prompt_tokens - context_before)
    else:
        new_user_tokens = int(row["new_user_tokens"])

    row["context_tokens_before_turn"] = context_before
    row["new_user_tokens"] = new_user_tokens
    if row.get("_input_bytes_missing"):
        row["input_bytes"] = _input_bytes(new_user_tokens, token_cfg)
    if row.get("_output_bytes_missing"):
        row["output_bytes"] = _output_bytes(output_tokens, token_cfg)
    return min(context_window_tokens, context_before + new_user_tokens + output_tokens)

def _drop_internal_request_markers(requests: list[dict[str, Any]]) -> None:
    marker_keys = (
        "_new_user_tokens_missing",
        "_context_tokens_before_turn_missing",
        "_input_bytes_missing",
        "_output_bytes_missing",
    )
    for row in requests:
        for key in marker_keys:
            row.pop(key, None)

def _infer_sessions_from_requests(
    requests: list[dict[str, Any]],
    desc: dict[str, Any],
) -> list[dict[str, Any]]:
    session_cfg = _dict(desc.get("sessions"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in requests:
        session_id = row.get("session_id")
        if session_id is not None:
            grouped.setdefault(str(session_id), []).append(row)
    sessions: list[dict[str, Any]] = []
    for session_id, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item["turn_index"]))
        sessions.append(
            {
                "session_id": session_id,
                "user_region_id": rows[0]["user_region_id"],
                "session_start_time_s": rows[0]["arrival_time_s"],
                "session_turn_count": len(rows),
                "session_slo_class": rows[0]["slo_class"],
                "session_ttl_s": rows[0].get("session_ttl_s", session_cfg.get("session_ttl_s")),
                "context_window_tokens": rows[0]["context_window_tokens"],
                "seed": rows[0].get("seed", _description_seed(desc)),
            }
        )
    return sessions

def _sort_and_number_requests(
    requests: list[dict[str, Any]],
    *,
    workload_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    sorted_requests = sorted(
        requests,
        key=lambda item: (
            float(item["arrival_time_s"]),
            str(item["user_region_id"]),
            int(item.get("turn_index") or 0),
        ),
    )
    for index, row in enumerate(sorted_requests):
        row["request_id"] = f"{workload_id}_req_{index:08d}"
        row["seed"] = seed + index
        _reject_runtime_fields(row, context=row["request_id"])
    return sorted_requests

def _build_timebin_demand(
    requests: list[dict[str, Any]],
    *,
    horizon_s: float,
    time_bin_s: float,
) -> list[dict[str, Any]]:
    bins: dict[tuple[float, str], dict[str, Any]] = {}
    for row in requests:
        arrival = float(row["arrival_time_s"])
        if arrival < 0 or arrival > horizon_s:
            raise ValueError(f"request {row['request_id']} has arrival_time_s outside workload horizon")
        bin_start = math.floor(arrival / time_bin_s) * time_bin_s
        bin_end = min(horizon_s, bin_start + time_bin_s)
        key = (bin_start, str(row["user_region_id"]))
        item = bins.setdefault(
            key,
            {
                "time_bin_start_s": bin_start,
                "time_bin_end_s": bin_end,
                "region_id": str(row["user_region_id"]),
                "request_count": 0,
                "prompt_tokens_total": 0,
                "new_user_tokens_total": 0,
                "output_tokens_total": 0,
                "interactive_request_count": 0,
                "burst_request_count": 0,
            },
        )
        item["request_count"] += 1
        item["prompt_tokens_total"] += int(row["prompt_tokens"])
        item["new_user_tokens_total"] += int(row["new_user_tokens"])
        item["output_tokens_total"] += int(row["output_tokens"])
        if "interactive" in str(row["slo_class"]):
            item["interactive_request_count"] += 1
        if row.get("burst_id") is not None:
            item["burst_request_count"] += 1
    return [bins[key] for key in sorted(bins)]

def _build_summary(
    *,
    source_mode: str,
    workload_id: str,
    requests: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    regions: tuple[RegionSpec, ...],
    gateways: tuple[GatewaySpec, ...],
    timebin_demand: list[dict[str, Any]],
    seed: int,
    generation_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_tokens = [int(row["prompt_tokens"]) for row in requests]
    new_user_tokens = [int(row["new_user_tokens"]) for row in requests]
    output_tokens = [int(row["output_tokens"]) for row in requests]
    gaps = [
        float(row["inter_turn_gap_s"])
        for row in requests
        if row.get("inter_turn_gap_s") is not None
    ]
    return {
        "schema_version": SUMMARY_SCHEMA,
        "workload_id": workload_id,
        "source_mode": source_mode,
        "request_count": len(requests),
        "session_count": len(sessions),
        "region_count": len(regions),
        "gateway_count": len(gateways),
        "seed": seed,
        "request_time_range_s": {
            "min": min((float(row["arrival_time_s"]) for row in requests), default=None),
            "max": max((float(row["arrival_time_s"]) for row in requests), default=None),
        },
        "task_counts": _count_by(requests, "task_type"),
        "slo_counts": _count_by(requests, "slo_class"),
        "slo_class_counts": _count_by(requests, "slo_class"),
        "region_request_counts": _count_by(requests, "user_region_id"),
        "gateway_request_counts": _count_by(requests, "gateway_id"),
        "token_stats": {
            "prompt_tokens": _stats(prompt_tokens),
            "new_user_tokens": _stats(new_user_tokens),
            "output_tokens": _stats(output_tokens),
        },
        "session_stats": {
            "turn_count": _stats([int(row["session_turn_count"]) for row in sessions]),
            "inter_turn_gap_s": _stats(gaps),
        },
        "timebin_count": len(timebin_demand),
        "total_tokens": {
            "prompt_tokens": sum(prompt_tokens),
            "new_user_tokens": sum(new_user_tokens),
            "output_tokens": sum(output_tokens),
        },
        "provenance": {
            "generator": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
        },
        "generation_details": to_jsonable(generation_details or {}),
    }

def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        label = "null" if value is None else str(value)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))

def _stats(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    values = sorted(values)
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }

def _percentile(values: list[int | float], q: float) -> int | float:
    if not values:
        return None
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[index]

def _read_description_document(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(text, source=str(path))
    elif suffix == ".md":
        data = _load_markdown_description(text, source=str(path))
    else:
        raise ValueError("workload descriptions must be .json, .yaml, .yml, or .md")
    if not isinstance(data, dict):
        raise ValueError(f"workload description {path} did not parse to a mapping")
    return data

def _load_markdown_description(text: str, *, source: str) -> dict[str, Any]:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return _load_yaml(text[4:end], source=source)
    for fence in ("```json", "```yaml", "```yml"):
        start = text.find(fence)
        if start == -1:
            continue
        start += len(fence)
        end = text.find("```", start)
        if end == -1:
            raise ValueError(f"unterminated structured workload block in {source}")
        payload = text[start:end].strip()
        if fence == "```json":
            return json.loads(payload)
        return _load_yaml(payload, source=source)
    raise ValueError(
        "Markdown workload descriptions require YAML front matter or a fenced json/yaml block"
    )

def _load_yaml(text: str, *, source: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            f"YAML workload description parsing for {source} requires optional PyYAML; "
            "use .json for stdlib-only parsing"
        ) from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML workload description {source} did not parse to a mapping")
    return data

def _read_rows_by_format(path: Path, fmt: str) -> list[dict[str, Any]]:
    if fmt == "jsonl":
        return _read_jsonl(path)
    if fmt == "csv":
        return _read_csv_rows(path)
    if fmt == "parquet":
        raise ValueError("parquet is intentionally unsupported by the stdlib workload core")
    raise ValueError(f"unsupported explicit request format: {fmt!r}")

def _explicit_format(configured: Any, path: Path) -> str:
    fmt = str(configured or path.suffix.lstrip(".")).lower()
    if fmt == "json":
        raise ValueError("explicit request rows must be JSONL or CSV, not a single JSON array")
    if fmt not in {"jsonl", "csv", "parquet"}:
        raise ValueError(f"unsupported request row format: {fmt!r}")
    if fmt == "parquet":
        raise ValueError("parquet is intentionally unsupported by the stdlib workload core")
    return fmt

def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.write_text(
        "".join(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

def _write_timebin_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    fieldnames = (
        "time_bin_start_s",
        "time_bin_end_s",
        "region_id",
        "request_count",
        "prompt_tokens_total",
        "new_user_tokens_total",
        "output_tokens_total",
        "interactive_request_count",
        "burst_request_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

def _validate_description_header(desc: dict[str, Any]) -> None:
    schema = str(desc.get("schema_version", ""))
    if schema != WORKLOAD_DESCRIPTION_SCHEMA:
        raise ValueError(
            f"unsupported workload description schema_version {schema!r}; "
            f"expected {WORKLOAD_DESCRIPTION_SCHEMA!r}"
        )
    bundle_schema = str(desc.get("output_bundle_schema", WORKLOAD_BUNDLE_SCHEMA))
    if bundle_schema != WORKLOAD_BUNDLE_SCHEMA:
        raise ValueError(
            f"unsupported output_bundle_schema {bundle_schema!r}; expected {WORKLOAD_BUNDLE_SCHEMA!r}"
        )
    if not desc.get("workload_id"):
        raise ValueError("workload_id is required")
    source_mode = str(desc.get("source_mode", ""))
    if source_mode not in SUPPORTED_SOURCE_MODES:
        raise ValueError(
            f"unsupported source_mode {source_mode!r}; supported modes are {sorted(SUPPORTED_SOURCE_MODES)}"
        )
    for section in ("time", "geography", "arrival", "tokens", "slo", "sessions"):
        if section not in desc:
            raise ValueError(f"workload description missing required section {section!r}")

def _require_unique_request_ids(requests: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in requests:
        request_id = str(row["request_id"])
        if request_id in seen:
            duplicates.add(request_id)
        seen.add(request_id)
    if duplicates:
        raise ValueError(f"duplicate request_id values: {sorted(duplicates)}")

def _require_sorted_requests(requests: list[dict[str, Any]]) -> None:
    keys = [(float(row["arrival_time_s"]), str(row["request_id"])) for row in requests]
    if keys != sorted(keys):
        raise ValueError("explicit requests are not sorted by arrival_time_s, request_id")

def _description_seed(desc: dict[str, Any]) -> int:
    if desc.get("seed") is not None:
        return int(desc["seed"])
    generator = _dict(desc.get("generator"))
    if generator.get("seed") is not None:
        return int(generator["seed"])
    provenance = _dict(desc.get("provenance"))
    if provenance.get("seed") is not None:
        return int(provenance["seed"])
    return 1

def _base_dir(base_dir: str | Path | None) -> Path:
    return Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()

def _source_file_record(path: Path, base_dir: Path) -> dict[str, Any]:
    return {
        "path": _relative_or_absolute(path, base_dir),
        "sha256": _sha256_file(path),
        "format": path.suffix.lstrip(".").lower(),
    }

def _relative_or_absolute(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path.resolve())

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _without_internal_keys(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if not str(key).startswith("_")}

def _geo_distance_sq(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    return (lat_a - lat_b) ** 2 + ((lon_a - lon_b) * math.cos(math.radians(lat_a))) ** 2

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def _task_mix_label(token_cfg: dict[str, Any]) -> str:
    task_mix = token_cfg.get("task_mix")
    if not task_mix:
        return "default_chatllm_mix"
    return ",".join(str(_pick(_dict(item), "task_type", "task")) for item in task_mix)

def _distribution_label(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("type", "custom"))
    return "custom"

def _default_task_mix() -> list[dict[str, Any]]:
    return [
        {
            "task_type": "conversation",
            "weight": 0.70,
            "prompt_distribution": {
                "type": "quantile_buckets",
                "p50": 768,
                "p95": 2048,
                "p99": 4096,
                "max": 8192,
            },
            "output_distribution": {
                "type": "quantile_buckets",
                "p50": 256,
                "p95": 1024,
                "p99": 2048,
                "max": 4096,
            },
        },
        {
            "task_type": "summarization",
            "weight": 0.20,
            "prompt_distribution": {
                "type": "quantile_buckets",
                "p50": 2048,
                "p95": 8192,
                "p99": 12288,
                "max": 16384,
            },
            "output_distribution": {
                "type": "quantile_buckets",
                "p50": 256,
                "p95": 1024,
                "p99": 1536,
                "max": 2048,
            },
        },
        {
            "task_type": "code",
            "weight": 0.10,
            "prompt_distribution": {
                "type": "quantile_buckets",
                "p50": 1024,
                "p95": 4096,
                "p99": 8192,
                "max": 12000,
            },
            "output_distribution": {
                "type": "quantile_buckets",
                "p50": 512,
                "p95": 2048,
                "p99": 4096,
                "max": 8192,
            },
        },
    ]

def _slo_mix_preset(name: str) -> list[dict[str, Any]]:
    presets = {
        "interactive_heavy": [
            {"slo_class": "interactive_5s", "weight": 0.80, "ttft_deadline_s": 5.0, "e2e_deadline_s": 30.0},
            {"slo_class": "relaxed_60s", "weight": 0.20, "ttft_deadline_s": 10.0, "e2e_deadline_s": 60.0},
        ],
        "balanced": [
            {"slo_class": "interactive_5s", "weight": 0.60, "ttft_deadline_s": 5.0, "e2e_deadline_s": 30.0},
            {"slo_class": "relaxed_60s", "weight": 0.35, "ttft_deadline_s": 10.0, "e2e_deadline_s": 60.0},
            {"slo_class": "batch_300s", "weight": 0.05, "ttft_deadline_s": None, "e2e_deadline_s": 300.0},
        ],
        "relaxed_heavy": [
            {"slo_class": "interactive_10s", "weight": 0.30, "ttft_deadline_s": 10.0, "e2e_deadline_s": 60.0},
            {"slo_class": "relaxed_60s", "weight": 0.60, "ttft_deadline_s": 10.0, "e2e_deadline_s": 60.0},
            {"slo_class": "batch_300s", "weight": 0.10, "ttft_deadline_s": None, "e2e_deadline_s": 300.0},
        ],
    }
    if name not in presets:
        raise ValueError(
            f"unknown SLO mix preset {name!r}; provide slo.classes or use one of {sorted(presets)}"
        )
    return presets[name]

def _turn_count_preset(name: str) -> dict[int, float]:
    presets = {
        "single_turn_only": {1: 1.0},
        "light_session": {1: 0.75, 2: 0.20, 4: 0.05},
        "mixed_session": {1: 0.50, 2: 0.25, 4: 0.20, 8: 0.05},
        "session_stress": {1: 0.25, 2: 0.25, 4: 0.35, 8: 0.15},
    }
    if name not in presets:
        raise ValueError(f"unknown turn_count_distribution preset {name!r}")
    return presets[name]

def _gap_preset(name: str) -> dict[str, Any]:
    if name not in {"mixed_gap", "bucketed_default"}:
        raise ValueError(f"unknown gap_distribution preset {name!r}")
    return {
        "type": "bucketed_uniform",
        "buckets": [
            {"name": "short_gap", "weight": 0.50, "min_s": 5.0, "max_s": 30.0},
            {"name": "medium_gap", "weight": 0.35, "min_s": 60.0, "max_s": 300.0},
            {"name": "long_gap", "weight": 0.15, "min_s": 600.0, "max_s": 1800.0},
        ],
    }

