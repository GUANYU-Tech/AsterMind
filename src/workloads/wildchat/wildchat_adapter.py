#!/usr/bin/env python3
"""Convert a fixed official WildChat Parquet shard into a workload bundle.

The experiment path is deliberately local and reproducible: download the
chosen official Parquet shard once, verify its SHA-256, then map it with the
pinned target-model tokenizer. The bundle itself always has exactly seven
``orbitserve.workload`` files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


from orbitserve.workload import (
    BUNDLE_FILES, BUNDLE_SCHEMA, CANONICALIZATION, DESCRIPTION_SCHEMA,
    GATEWAY_KEYS, REGION_KEYS, REQUEST_KEYS, SESSION_KEYS, SUMMARY_SCHEMA,
    TIMEBIN_FIELDS, bundle_hash, canonical_json_bytes as canonical_bytes, sha256_bytes,
)


class AdapterError(ValueError):
    """A deterministic, user-correctable adapter input error."""


def stable_hex(*parts: str, length: int = 24) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:length]


def stable_seed(*parts: str) -> int:
    return int(stable_hex(*parts, length=16), 16) & ((1 << 63) - 1)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"{path} must contain one JSON object")
    return value


def read_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise AdapterError(f"{path} must contain one JSON array of objects")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise AdapterError(f"{path}:{line_number}: empty JSONL lines are not allowed")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AdapterError(f"{path}:{line_number}: each JSONL row must be an object")
        rows.append(row)
    if not rows:
        raise AdapterError(f"{path}: no records")
    return rows


def read_source_rows(path: Path) -> list[dict[str, Any]]:
    """Read either adapter-ready JSONL or an official WildChat Parquet shard."""
    if path.suffix.lower() != ".parquet":
        return read_jsonl(path)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise AdapterError("Parquet input requires pyarrow") from exc
    try:
        reader = parquet.ParquetFile(path)
        rows = [row for batch in reader.iter_batches(batch_size=4096) for row in batch.to_pylist()]
    except Exception as exc:  # pyarrow exposes several format-specific exception classes
        raise AdapterError(f"cannot read WildChat Parquet shard {path}: {exc}") from exc
    if not rows:
        raise AdapterError(f"{path}: no records")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical_bytes(row) for row in rows))


def parse_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if not isinstance(value, str) or not value:
        raise AdapterError("assistant timestamp is missing or not a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdapterError(f"invalid UTC timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise AdapterError(f"timestamp must carry a UTC offset: {value!r}")
    return parsed.astimezone(timezone.utc)


def simple_unicode_token_count(text: str) -> int:
    # A stable smoke-test tokenizer: words/numbers and each non-space symbol are tokens.
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


class TokenCounter:
    """Use a pinned target tokenizer; simple_unicode_v1 is only a fallback."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.backend = spec["backend"]
        self.tokenizer: Any | None = None
        if self.backend == "huggingface_auto":
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise AdapterError("huggingface_auto requires the transformers package") from exc
            source = spec.get("_resolved_local_path", spec["tokenizer_id"])
            self.tokenizer = AutoTokenizer.from_pretrained(
                source,
                revision=None if "_resolved_local_path" in spec else spec["tokenizer_revision"],
                trust_remote_code=False,
                local_files_only="_resolved_local_path" in spec,
            )
        elif self.backend != "simple_unicode_v1":
            raise AdapterError(f"unsupported tokenization backend: {self.backend!r}")

    @staticmethod
    def _simple_message_tokens(message: dict[str, Any]) -> int:
        return simple_unicode_token_count(f"<|{message.get('role', '')}|>\n{message.get('content', '')}\n")

    def prompt_counts(self, history: list[dict[str, Any]], user: dict[str, Any], context_window: int) -> tuple[int, int, int]:
        if self.backend == "simple_unicode_v1":
            context = sum(self._simple_message_tokens(message) for message in history)
            new_user = self._simple_message_tokens(user)
        else:
            assert self.tokenizer is not None
            history_ids = self.tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=False) if history else []
            prompt_ids = self.tokenizer.apply_chat_template(history + [user], tokenize=True, add_generation_prompt=True)
            context = len(history_ids)
            new_user = max(1, len(prompt_ids) - context)
        context = min(context, max(0, context_window - new_user))
        return context, new_user, context + new_user

    def output_tokens(self, assistant: dict[str, Any]) -> int:
        content = str(assistant.get("content", ""))
        if self.backend == "simple_unicode_v1":
            return self._simple_message_tokens(assistant)
        assert self.tokenizer is not None
        count = len(self.tokenizer.encode(content, add_special_tokens=False))
        return max(1, count + int(self.spec.get("assistant_token_policy") == "content_plus_eos"))


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0
    p1, p2 = math.radians(lat_a), math.radians(lat_b)
    dp, dl = p2 - p1, math.radians(lon_b - lon_a)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(h))


def select_gateway(region: dict[str, Any], gateways: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if policy != "nearest_gateway":
        raise AdapterError(f"unsupported gateways.selection_policy: {policy!r}")
    if not gateways:
        raise AdapterError("at least one gateway is required for gateway_sgl")
    return min(
        gateways,
        key=lambda gateway: (haversine_km(region["lat_deg"], region["lon_deg"], gateway["lat_deg"], gateway["lon_deg"]), gateway["gateway_id"]),
    )


def select_slo(output_tokens: int, rules: list[dict[str, Any]]) -> dict[str, Any]:
    for rule in rules:
        if rule.get("default") is True or output_tokens <= int(rule["max_output_tokens"]):
            required = {"slo_class", "ttft_deadline_s", "e2e_deadline_s", "priority", "session_ttl_s"}
            missing = required - set(rule)
            if missing:
                raise AdapterError(f"SLO rule is missing {sorted(missing)}")
            return rule
    raise AdapterError("slo_policy.rules needs a default rule")


def _location_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def load_country_centroids(config: dict[str, Any], config_dir: Path) -> dict[str, tuple[str, float, float]]:
    configured_path = config.get("country_centroids_path")
    if not configured_path:
        return {}
    path = config_dir / str(configured_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read country centroid catalog {path}: {exc}") from exc
    entries = raw.get("countries", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise AdapterError("country centroid catalog must be an array or an object with countries")
    catalog: dict[str, tuple[str, float, float]] = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        if "lat_deg" in item and "lon_deg" in item:
            lat, lon = float(item["lat_deg"]), float(item["lon_deg"])
        elif isinstance(item.get("location"), str):
            lat_text, lon_text = item["location"].split(",", maxsplit=1)
            lat, lon = float(lat_text), float(lon_text)
        else:
            continue
        catalog[_location_key(item["name"])] = (item["name"], lat, lon)
    if not catalog:
        raise AdapterError("country centroid catalog contains no usable coordinates")
    return catalog


def load_state_centroids(config: dict[str, Any], config_dir: Path) -> dict[str, dict[str, Any]]:
    configured_path = config.get("state_centroids_path")
    if not configured_path:
        return {}
    path = config_dir / str(configured_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read state centroid catalog {path}: {exc}") from exc
    entries = raw.get("states", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, dict):
        raise AdapterError("state centroid catalog must be an object keyed by country|state")
    catalog: dict[str, dict[str, Any]] = {}
    for key, item in entries.items():
        if not isinstance(key, str) or "|" not in key or not isinstance(item, dict):
            continue
        country, state = key.split("|", 1)
        if not isinstance(item.get("lat_deg"), (int, float)) or not isinstance(item.get("lon_deg"), (int, float)):
            continue
        normalized_key = f"{_location_key(country)}|{_location_key(state)}"
        catalog[normalized_key] = {
            "region_id": f"{_location_key(country)}_{_location_key(state)}",
            "region_name": item.get("region_name", f"{state}, {country}"),
            "lat_deg": float(item["lat_deg"]),
            "lon_deg": float(item["lon_deg"]),
            "timezone_utc_offset_h": max(-12, min(14, round(float(item["lon_deg"]) / 15))),
            "market_class": item.get("market_class"),
        }
    return catalog


def find_region(
    user_message: dict[str, Any],
    mapping: dict[str, Any],
    state_centroids: dict[str, dict[str, Any]],
    country_centroids: dict[str, tuple[str, float, float]],
) -> dict[str, Any] | None:
    country, state = user_message.get("country"), user_message.get("state")
    if isinstance(country, str) and isinstance(state, str):
        match = mapping.get(f"{country}|{state}")
        if match is not None:
            return match
        state_match = state_centroids.get(f"{_location_key(country)}|{_location_key(state)}")
        if state_match is not None:
            return state_match
    if isinstance(country, str):
        # A country-only fallback is deliberately explicit: use "Country|*" in the config.
        explicit = mapping.get(f"{country}|*")
        if explicit is not None:
            return explicit
        centroid = country_centroids.get(_location_key(country))
        if centroid is not None:
            name, lat_deg, lon_deg = centroid
            return {
                "region_id": f"country_{_location_key(name)}", "region_name": name,
                "lat_deg": lat_deg, "lon_deg": lon_deg,
                "timezone_utc_offset_h": max(-12, min(14, round(lon_deg / 15))),
                "market_class": None,
            }
    return None


def extract_pairs(record: dict[str, Any], source_key: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = record.get("conversation")
    if not isinstance(conversation, list):
        return []
    policy = config["adapter_policy"]
    mapping = config["region_mapping"]
    state_centroids = config.get("_state_centroids", {})
    country_centroids = config.get("_country_centroids", {})
    epoch = parse_utc(config["dataset"]["time_window"]["epoch_utc"])
    horizon = float(config["dataset"]["time_window"]["time_horizon_s"])
    history: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    index = 0
    while index < len(conversation):
        item = conversation[index]
        if not isinstance(item, dict):
            index += 1
            continue
        if item.get("role") != "user" or index + 1 >= len(conversation):
            history.append(item)
            index += 1
            continue
        assistant = conversation[index + 1]
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            history.append(item)
            index += 1
            continue
        content_ok = isinstance(item.get("content"), str) and item["content"].strip() and isinstance(assistant.get("content"), str) and assistant["content"].strip()
        region = find_region(item, mapping, state_centroids, country_centroids)
        try:
            arrival = (parse_utc(assistant.get("timestamp")) - epoch).total_seconds()
        except AdapterError:
            if policy["missing_timestamp_policy"] == "drop":
                history.extend([item, assistant])
                index += 2
                continue
            raise
        if not content_ok or region is None or not (0 <= arrival < horizon):
            if policy["invalid_pair_policy"] != "drop" and not content_ok:
                raise AdapterError(f"{source_key}: invalid user→assistant pair")
            if region is None and policy["unknown_region_policy"] != "drop":
                raise AdapterError(f"{source_key}: no configured region for user location")
            history.extend([item, assistant])
            index += 2
            continue
        pairs.append({
            "source_key": source_key,
            "pair_index": len(pairs),
            "arrival_time_s": arrival,
            "region": region,
            "history": deepcopy(history),
            "user": item,
            "assistant": assistant,
        })
        history.extend([item, assistant])
        index += 2
    return pairs


def build_requests(config: dict[str, Any], raw_rows: list[dict[str, Any]], input_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gateways_config = config["gateways"]
    raw_gateways = gateways_config["items"]
    rules = config["slo_policy"]["rules"]
    tokenization = config["tokenization"]
    token_counter = TokenCounter(tokenization)
    if config["slo_policy"].get("session_assignment") != "max_output_tokens":
        raise AdapterError("slo_policy.session_assignment must be max_output_tokens for canonical workload session consistency")

    all_requests: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    used_regions: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(raw_rows, start=1):
        source_key = stable_hex(config["dataset"]["revision"], input_sha256, str(line_number), length=32)
        pairs = extract_pairs(raw, source_key, config)
        if not pairs:
            continue
        session_id = f"wc_sess_{stable_hex(config['workload_id'], source_key)}"
        pair_outputs = [token_counter.output_tokens(pair["assistant"]) for pair in pairs]
        slo = select_slo(max(pair_outputs), rules)
        previous_arrival: float | None = None
        session_rows: list[dict[str, Any]] = []
        for pair, output_tokens in zip(pairs, pair_outputs, strict=True):
            arrival = float(pair["arrival_time_s"])
            if previous_arrival is not None and arrival < previous_arrival:
                raise AdapterError(f"{session_id}: assistant completion proxy moved backwards")
            window = int(tokenization["context_window_tokens"])
            context, new_user, prompt = token_counter.prompt_counts(pair["history"], pair["user"], window)
            region = pair["region"]
            gateway = select_gateway(region, raw_gateways, gateways_config["selection_policy"])
            source_turn_key = f"{source_key}:turn:{pair['pair_index']}"
            request_id = f"wc_req_{stable_hex(config['workload_id'], source_turn_key)}"
            row = {
                "request_id": request_id,
                "arrival_time_s": arrival,
                "user_region_id": region["region_id"],
                "region_id": region["region_id"],
                "lat_deg": float(region["lat_deg"]),
                "lon_deg": float(region["lon_deg"]),
                "access_mode": "gateway_sgl",
                "ingress_id": gateway["gateway_id"],
                "ingress_lat_deg": float(gateway["lat_deg"]),
                "ingress_lon_deg": float(gateway["lon_deg"]),
                "gateway_id": gateway["gateway_id"],
                "task_type": "conversation",
                "model_family": config["model_mapping"].get(raw.get("model"), config["model_mapping"].get("default", tokenization["target_model"])),
                "prompt_tokens": prompt,
                "new_user_tokens": new_user,
                "context_tokens_before_turn": context,
                "output_tokens": output_tokens,
                "input_bytes": prompt * int(tokenization["bytes_per_input_token"]),
                "output_bytes": output_tokens * int(tokenization["bytes_per_output_token"]),
                "context_window_tokens": window,
                "slo_class": slo["slo_class"],
                "ttft_deadline_s": float(slo["ttft_deadline_s"]),
                "e2e_deadline_s": float(slo["e2e_deadline_s"]),
                "priority": slo["priority"],
                "session_id": session_id,
                "turn_index": pair["pair_index"],
                "session_turn_count": None,
                "inter_turn_gap_s": None if previous_arrival is None else arrival - previous_arrival,
                "session_ttl_s": float(slo["session_ttl_s"]),
                "burst_id": None,
                "burst_multiplier": 1.0,
                "seed": stable_seed(config["seed"].__str__(), request_id),
            }
            session_rows.append(row)
            used_regions[region["region_id"]] = region
            previous_arrival = arrival
        for row in session_rows:
            row["session_turn_count"] = len(session_rows)
        sessions.append({
            "session_id": session_id,
            "user_region_id": session_rows[0]["user_region_id"],
            "session_start_time_s": session_rows[0]["arrival_time_s"],
            "session_turn_count": len(session_rows),
            "session_slo_class": session_rows[0]["slo_class"],
            "context_window_tokens": session_rows[0]["context_window_tokens"],
            "session_ttl_s": session_rows[0]["session_ttl_s"],
            "seed": stable_seed(config["seed"].__str__(), session_id),
        })
        all_requests.extend(session_rows)

    if not all_requests:
        raise AdapterError("no valid user→assistant pairs remained after filtering")
    all_requests.sort(key=lambda row: (row["arrival_time_s"], row["request_id"]))
    sessions.sort(key=lambda row: (row["session_start_time_s"], row["session_id"]))
    counts = Counter(row["region_id"] for row in all_requests)
    regions = []
    for region_id, region in sorted(used_regions.items()):
        regions.append({
            "region_id": region_id,
            "region_name": region["region_name"],
            "lat_deg": float(region["lat_deg"]),
            "lon_deg": float(region["lon_deg"]),
            "timezone_utc_offset_h": float(region["timezone_utc_offset_h"]),
            "demand_weight": counts[region_id] / len(all_requests),
            "weight_source": f"wildchat_pair_share:{config['dataset']['revision']}",
            "market_class": region.get("market_class"),
        })
    gateway_regions: dict[str, list[str]] = defaultdict(list)
    for request in all_requests:
        gateway_regions[request["gateway_id"]].append(request["region_id"])
    gateways = []
    for gateway in raw_gateways:
        gateways.append({
            "gateway_id": gateway["gateway_id"], "lat_deg": float(gateway["lat_deg"]), "lon_deg": float(gateway["lon_deg"]),
            "served_region_ids": sorted(set(gateway_regions[gateway["gateway_id"]])),
            "gateway_bandwidth_bps": int(gateway["gateway_bandwidth_bps"]),
            "availability_profile_id": gateway.get("availability_profile_id"), "cost_usd_per_year": gateway.get("cost_usd_per_year"),
        })
    gateways.sort(key=lambda row: row["gateway_id"])
    return all_requests, sessions, regions, gateways


def build_timebins(requests: list[dict[str, Any]], horizon: float, bin_size: float) -> list[dict[str, Any]]:
    bins: dict[tuple[float, str], dict[str, Any]] = {}
    for row in requests:
        start = math.floor(row["arrival_time_s"] / bin_size) * bin_size
        key = (start, row["region_id"])
        if key not in bins:
            bins[key] = {"time_bin_start_s": start, "time_bin_end_s": min(start + bin_size, horizon), "region_id": row["region_id"], "request_count": 0, "prompt_tokens_total": 0, "new_user_tokens_total": 0, "output_tokens_total": 0, "interactive_request_count": 0, "burst_request_count": 0}
        item = bins[key]
        item["request_count"] += 1
        item["prompt_tokens_total"] += row["prompt_tokens"]
        item["new_user_tokens_total"] += row["new_user_tokens"]
        item["output_tokens_total"] += row["output_tokens"]
        item["interactive_request_count"] += int(row["priority"] == "interactive")
        item["burst_request_count"] += int(row["burst_id"] is not None)
    return [bins[key] for key in sorted(bins)]


def stat_block(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None, "p99": None}
    ordered = sorted(values)
    def percentile(q: float) -> float | int:
        return ordered[math.ceil(q * len(ordered)) - 1]
    return {"count": len(values), "min": ordered[0], "max": ordered[-1], "mean": sum(values) / len(values), "p50": percentile(0.50), "p95": percentile(0.95), "p99": percentile(0.99)}


def build_summary(requests: list[dict[str, Any]], sessions: list[dict[str, Any]], regions: list[dict[str, Any]], gateways: list[dict[str, Any]], timebins: list[dict[str, Any]], description_hash: str, config: dict[str, Any]) -> dict[str, Any]:
    request_counts = Counter(row["region_id"] for row in requests)
    gateway_counts = Counter(row["gateway_id"] for row in requests if row["gateway_id"] is not None)
    gaps = [row["inter_turn_gap_s"] for row in requests if row["inter_turn_gap_s"] is not None]
    return {
        "schema_version": SUMMARY_SCHEMA, "workload_id": config["workload_id"], "source_mode": "explicit_requests", "seed": config["seed"],
        "request_count": len(requests), "session_count": len(sessions), "region_count": len(regions), "gateway_count": len(gateways), "timebin_count": len(timebins),
        "request_time_range_s": {"min": requests[0]["arrival_time_s"], "max": requests[-1]["arrival_time_s"]},
        "task_counts": dict(sorted(Counter(row["task_type"] for row in requests).items())),
        "model_family_counts": dict(sorted(Counter(row["model_family"] for row in requests).items())),
        "slo_class_counts": dict(sorted(Counter(row["slo_class"] for row in requests).items())),
        "priority_counts": {key: sum(row["priority"] == key for row in requests) for key in ("interactive", "normal", "batch")},
        "region_request_counts": {row["region_id"]: request_counts[row["region_id"]] for row in regions},
        "gateway_request_counts": {row["gateway_id"]: gateway_counts[row["gateway_id"]] for row in gateways},
        "direct_request_count": 0,
        "access_mode_counts": {"direct_user_sgl": 0, "gateway_sgl": len(requests)},
        "burst_request_counts": {},
        "token_stats": {key: stat_block([row[key] for row in requests]) for key in ("prompt_tokens", "new_user_tokens", "output_tokens")},
        "session_stats": {"turn_count": stat_block([row["session_turn_count"] for row in sessions]), "inter_turn_gap_s": stat_block(gaps)},
        "total_tokens": {key: sum(row[key] for row in requests) for key in ("prompt_tokens", "new_user_tokens", "output_tokens")},
        "total_bytes": {"input_bytes": sum(row["input_bytes"] for row in requests), "output_bytes": sum(row["output_bytes"] for row in requests)},
        "provenance": {"description_sha256": description_hash, "generator_version": GENERATOR_VERSION},
    }


def build_description(config: dict[str, Any], regions: list[dict[str, Any]], gateways: list[dict[str, Any]], source_sha: str, output_dir_name: str) -> dict[str, Any]:
    time = config["dataset"]["time_window"]
    region_access = [{"region_id": region["region_id"], "access_mode": "gateway_sgl", "gateway_id": min(gateways, key=lambda gateway: (haversine_km(region["lat_deg"], region["lon_deg"], gateway["lat_deg"], gateway["lon_deg"]), gateway["gateway_id"]))["gateway_id"]} for region in regions]
    rules = config["slo_policy"]["rules"]
    # The workload description requires the displayed (six-decimal) weights to sum to one.
    base_weight = round(1 / len(rules), 6)
    slo_classes = [
        {"slo_class": rule["slo_class"], "weight": base_weight if index < len(rules) - 1 else 1 - base_weight * (len(rules) - 1), "ttft_deadline_s": rule["ttft_deadline_s"], "e2e_deadline_s": rule["e2e_deadline_s"], "priority": rule["priority"]}
        for index, rule in enumerate(rules)
    ]
    provenance_notes = ["arrival_time_s is an assistant completion-time proxy, not an observed user submit time."]
    if config["tokenization"]["backend"] == "simple_unicode_v1":
        provenance_notes.append("simple_unicode_v1 is a deterministic smoke-test tokenizer; replace it with the target-model tokenizer for token-sensitive experiments.")
    else:
        provenance_notes.append(f"tokens use {config['tokenization']['tokenizer_id']} at revision {config['tokenization']['tokenizer_revision']} with its chat template.")
    return {
        "schema_version": DESCRIPTION_SCHEMA, "workload_id": config["workload_id"], "source_mode": "explicit_requests", "seed": config["seed"],
        "time": {"epoch_kind": "absolute_utc", "epoch": time["epoch_utc"], "time_horizon_s": time["time_horizon_s"], "time_bin_s": time["time_bin_s"]},
        "geography": {"demand_weight_policy": "require_sum_one", "regions": regions},
        "access": {"gateways": gateways, "region_access": sorted(region_access, key=lambda row: row["region_id"])},
        "arrival": {"process": "from_source"},
        "tokens": {"bytes_per_input_token": config["tokenization"]["bytes_per_input_token"], "bytes_per_output_token": config["tokenization"]["bytes_per_output_token"], "default_context_window_tokens": config["tokenization"]["context_window_tokens"], "task_mix": [{"task_type": "conversation", "model_family": config["tokenization"]["target_model"], "weight": 1.0, "new_user_tokens": {"type": "constant", "value": 1}, "output_tokens": {"type": "constant", "value": 1}}]},
        "slo": {"classes": slo_classes},
        "sessions": {"session_mode": "open_loop_turn_arrivals", "session_ratio": 1.0, "context_window_tokens": config["tokenization"]["context_window_tokens"], "session_ttl_s": config["slo_policy"]["rules"][0]["session_ttl_s"], "turn_count": {"type": "constant", "value": 1}, "inter_turn_gap_s": {"type": "constant", "value": 0}},
        "bursts": [],
        "source": {"mode": "explicit_requests", "requests_path": "canonical_requests.jsonl", "requests_format": "jsonl", "sessions_path": "canonical_sessions.jsonl", "sessions_format": "jsonl"},
        "validation": {"reject_unknown_fields": True, "reject_runtime_fields": True, "require_unique_ids": True, "require_sorted_input": False, "sort_output": True},
        "output": {"bundle_dir": output_dir_name, "overwrite": bool(config["bundle_output"]["overwrite"]), "float_precision_digits": 6},
        "provenance": {"sources": [{"source_id": "wildchat_input", "title": config["dataset"]["name"], "url": "https://huggingface.co/datasets/allenai/WildChat-4.8M", "license": None, "sha256": source_sha}], "notes": provenance_notes},
        "metadata": {"adapter": GENERATOR_NAME, "arrival_time_policy": config["adapter_policy"]["arrival_time_policy"]},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMEBIN_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def descriptor(path: Path, name: str, media_type: str, record_count: int) -> dict[str, Any]:
    return {"path": name, "media_type": media_type, "sha256": sha256_bytes(path.read_bytes()), "record_count": record_count}



def _description_hash(description: dict[str, Any]) -> str:
    digest_view = deepcopy(description)
    del digest_view["output"]["bundle_dir"]
    del digest_view["output"]["overwrite"]
    return sha256_bytes(canonical_bytes(digest_view))


def build(config_path: Path, input_path: Path, out_dir: Path, overwrite: bool) -> None:
    config = read_json(config_path)
    required = {"workload_id", "seed", "dataset", "region_mapping", "gateways", "tokenization", "model_mapping", "slo_policy", "adapter_policy", "bundle_output", "country_centroids_path", "state_centroids_path"}
    missing = required - set(config)
    if missing:
        raise AdapterError(f"config missing top-level fields: {sorted(missing)}")
    config["_country_centroids"] = load_country_centroids(config, config_path.parent)
    config["_state_centroids"] = load_state_centroids(config, config_path.parent)
    tokenizer_local_path = config["tokenization"].get("local_path")
    if tokenizer_local_path:
        resolved = config_path.parent / str(tokenizer_local_path)
        if not resolved.is_dir():
            raise AdapterError(f"pinned tokenizer directory is missing: {resolved}")
        config["tokenization"]["_resolved_local_path"] = str(resolved)
    if input_path.suffix.lower() != ".parquet":
        raise AdapterError("the experiment adapter accepts an official WildChat .parquet shard")
    input_sha256 = sha256_bytes(input_path.read_bytes())
    expected_sha = config["dataset"].get("expected_source_sha256")
    if expected_sha and input_sha256 != expected_sha:
        raise AdapterError(f"source SHA-256 mismatch for {input_path}; expected {expected_sha}, got {input_sha256}")
    raw_rows = read_source_rows(input_path)
    max_records = int(config["dataset"].get("max_source_records", len(raw_rows)))
    if max_records <= 0:
        raise AdapterError("dataset.max_source_records must be positive")
    raw_rows = raw_rows[:max_records]
    time_window = config["dataset"]["time_window"]
    if time_window.get("epoch_policy") == "most_populated_utc_day":
        day_counts: Counter[str] = Counter()
        for row in raw_rows:
            try:
                day_counts[parse_utc(row.get("timestamp")).date().isoformat()] += 1
            except AdapterError:
                continue
        if not day_counts:
            raise AdapterError("cannot infer an experiment day: no valid top-level WildChat timestamps")
        selected_day = min(day for day, count in day_counts.items() if count == max(day_counts.values()))
        time_window["epoch_utc"] = f"{selected_day}T00:00:00Z"
    requests, sessions, regions, gateways = build_requests(config, raw_rows, input_sha256)
    time = config["dataset"]["time_window"]
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    canonical_requests_path = out_dir.parent / "canonical_requests.jsonl"
    canonical_sessions_path = out_dir.parent / "canonical_sessions.jsonl"
    write_jsonl(canonical_requests_path, requests)
    write_jsonl(canonical_sessions_path, sessions)
    description = build_description(config, regions, gateways, input_sha256, out_dir.name)
    description_hash = _description_hash(description)
    timebins = build_timebins(requests, float(time["time_horizon_s"]), float(time["time_bin_s"]))
    summary = build_summary(requests, sessions, regions, gateways, timebins, description_hash, config)

    if out_dir.exists():
        if not overwrite:
            raise AdapterError(f"output directory exists: {out_dir}; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    write_json(out_dir / "regions.json", regions)
    write_json(out_dir / "gateways.json", gateways)
    write_jsonl(out_dir / "requests.jsonl", requests)
    write_jsonl(out_dir / "sessions.jsonl", sessions)
    write_csv(out_dir / "timebin_demand.csv", timebins)
    write_json(out_dir / "workload_summary.json", summary)
    descriptors = {
        "regions": descriptor(out_dir / "regions.json", *BUNDLE_FILES["regions"], len(regions)),
        "gateways": descriptor(out_dir / "gateways.json", *BUNDLE_FILES["gateways"], len(gateways)),
        "requests": descriptor(out_dir / "requests.jsonl", *BUNDLE_FILES["requests"], len(requests)),
        "sessions": descriptor(out_dir / "sessions.jsonl", *BUNDLE_FILES["sessions"], len(sessions)),
        "timebin_demand": descriptor(out_dir / "timebin_demand.csv", *BUNDLE_FILES["timebin_demand"], len(timebins)),
        "summary": descriptor(out_dir / "workload_summary.json", *BUNDLE_FILES["summary"], 1),
    }
    manifest = {
        "schema_version": BUNDLE_SCHEMA, "workload_id": config["workload_id"], "source_mode": "explicit_requests", "seed": config["seed"],
        "module_version": config["bundle_output"]["module_version"], "canonicalization": CANONICALIZATION,
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION, "prng": "sha256-stable-seed-v1"},
        "time": {"epoch_kind": "absolute_utc", "epoch": time["epoch_utc"], "time_horizon_s": time["time_horizon_s"], "time_bin_s": time["time_bin_s"], "interval_semantics": "half_open"},
        "source": {"mode": "explicit_requests", "inputs": [{"path": "canonical_requests.jsonl", "sha256": sha256_bytes(canonical_requests_path.read_bytes()), "record_count": len(requests)}, {"path": "canonical_sessions.jsonl", "sha256": sha256_bytes(canonical_sessions_path.read_bytes()), "record_count": len(sessions)}]},
        "files": descriptors, "description_sha256": description_hash, "bundle_sha256": bundle_hash(config["workload_id"], descriptors),
        "counts": {"request_count": len(requests), "session_count": len(sessions), "region_count": len(regions), "gateway_count": len(gateways), "timebin_count": len(timebins)},
        "provenance": description["provenance"],
    }
    write_json(out_dir / "workload_manifest.json", manifest)
    write_json(out_dir.parent / "generated_description.json", description)
    validate(out_dir, out_dir.parent / "generated_description.json")
    print(f"built {len(requests)} requests, {len(sessions)} sessions, {len(regions)} regions in {out_dir}")


def refresh_metadata(config_path: Path, bundle_dir: Path) -> None:
    """Regenerate sidecar inputs, description, summary and manifest for a bundle.

    This is used only to repair metadata around an already generated real-data
    bundle; it never alters requests, sessions, regions, gateways or time bins.
    """
    config = read_json(config_path)
    manifest = read_json(bundle_dir / "workload_manifest.json")
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise AdapterError("refresh requires an orbitserve.workload bundle")
    regions = read_json_array(bundle_dir / "regions.json")
    requests = read_jsonl_output(bundle_dir / "requests.jsonl")
    sessions = read_jsonl_output(bundle_dir / "sessions.jsonl")
    gateway_regions: dict[str, set[str]] = defaultdict(set)
    for request in requests:
        gateway_regions[request["gateway_id"]].add(request["region_id"])
    gateways = [
        {
            "gateway_id": gateway["gateway_id"],
            "lat_deg": float(gateway["lat_deg"]),
            "lon_deg": float(gateway["lon_deg"]),
            "served_region_ids": sorted(gateway_regions[gateway["gateway_id"]]),
            "gateway_bandwidth_bps": int(gateway["gateway_bandwidth_bps"]),
            "availability_profile_id": gateway.get("availability_profile_id"),
            "cost_usd_per_year": gateway.get("cost_usd_per_year"),
        }
        for gateway in config["gateways"]["items"]
    ]
    write_json(bundle_dir / "gateways.json", sorted(gateways, key=lambda gateway: gateway["gateway_id"]))
    gateways = read_json_array(bundle_dir / "gateways.json")
    config["dataset"]["time_window"]["epoch_utc"] = manifest["time"]["epoch"]
    config["dataset"]["time_window"].pop("epoch_policy", None)
    source_sha = manifest["provenance"]["sources"][0]["sha256"]
    canonical_requests_path = bundle_dir.parent / "canonical_requests.jsonl"
    canonical_sessions_path = bundle_dir.parent / "canonical_sessions.jsonl"
    write_jsonl(canonical_requests_path, requests)
    write_jsonl(canonical_sessions_path, sessions)
    description = build_description(config, regions, gateways, source_sha, bundle_dir.name)
    description_hash = _description_hash(description)
    timebins = build_timebins(requests, float(manifest["time"]["time_horizon_s"]), float(manifest["time"]["time_bin_s"]))
    write_json(bundle_dir / "workload_summary.json", build_summary(requests, sessions, regions, gateways, timebins, description_hash, config))
    descriptors = {
        "regions": descriptor(bundle_dir / "regions.json", *BUNDLE_FILES["regions"], len(regions)),
        "gateways": descriptor(bundle_dir / "gateways.json", *BUNDLE_FILES["gateways"], len(gateways)),
        "requests": descriptor(bundle_dir / "requests.jsonl", *BUNDLE_FILES["requests"], len(requests)),
        "sessions": descriptor(bundle_dir / "sessions.jsonl", *BUNDLE_FILES["sessions"], len(sessions)),
        "timebin_demand": descriptor(bundle_dir / "timebin_demand.csv", *BUNDLE_FILES["timebin_demand"], len(timebins)),
        "summary": descriptor(bundle_dir / "workload_summary.json", *BUNDLE_FILES["summary"], 1),
    }
    manifest.update({
        "module_version": config["bundle_output"]["module_version"],
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION, "prng": "sha256-stable-seed-v1"},
        "source": {"mode": "explicit_requests", "inputs": [
            {"path": "canonical_requests.jsonl", "sha256": sha256_bytes(canonical_requests_path.read_bytes()), "record_count": len(requests)},
            {"path": "canonical_sessions.jsonl", "sha256": sha256_bytes(canonical_sessions_path.read_bytes()), "record_count": len(sessions)},
        ]},
        "files": descriptors,
        "description_sha256": description_hash,
        "bundle_sha256": bundle_hash(manifest["workload_id"], descriptors),
        "counts": {"request_count": len(requests), "session_count": len(sessions), "region_count": len(regions), "gateway_count": len(gateways), "timebin_count": len(timebins)},
        "provenance": description["provenance"],
    })
    write_json(bundle_dir / "workload_manifest.json", manifest)
    write_json(bundle_dir.parent / "generated_description.json", description)
    validate(bundle_dir, bundle_dir.parent / "generated_description.json")
    print(f"refreshed metadata for {bundle_dir}")


def read_jsonl_output(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.read_bytes() else []


def validate(bundle_dir: Path, description_path: Path | None) -> None:
    actual = {path.name for path in bundle_dir.iterdir() if path.is_file()}
    expected = {"workload_manifest.json", *(name for name, _ in BUNDLE_FILES.values())}
    if actual != expected:
        raise AdapterError(f"bundle must contain exactly seven files; actual={sorted(actual)}")
    manifest = read_json(bundle_dir / "workload_manifest.json")
    if manifest.get("schema_version") != BUNDLE_SCHEMA or manifest.get("source_mode") != "explicit_requests":
        raise AdapterError("manifest schema_version/source_mode is invalid")
    descriptors = manifest.get("files")
    if not isinstance(descriptors, dict) or set(descriptors) != set(BUNDLE_FILES):
        raise AdapterError("manifest must describe exactly six data files")
    for key, (name, media_type) in BUNDLE_FILES.items():
        item = descriptors[key]
        if item.get("path") != name or item.get("media_type") != media_type:
            raise AdapterError(f"manifest descriptor mismatch for {key}")
        if item.get("sha256") != sha256_bytes((bundle_dir / name).read_bytes()):
            raise AdapterError(f"hash mismatch for {name}")
    if manifest.get("bundle_sha256") != bundle_hash(manifest["workload_id"], descriptors):
        raise AdapterError("bundle_sha256 mismatch")
    regions = read_json_array(bundle_dir / "regions.json")
    gateways = read_json_array(bundle_dir / "gateways.json")
    if any(set(row) != set(REGION_KEYS) for row in regions):
        raise AdapterError("regions.json does not have the exact canonical Region fields")
    if any(set(row) != set(GATEWAY_KEYS) for row in gateways):
        raise AdapterError("gateways.json does not have the exact canonical Gateway fields")
    requests = read_jsonl_output(bundle_dir / "requests.jsonl")
    sessions = read_jsonl_output(bundle_dir / "sessions.jsonl")
    if any(set(row) != set(REQUEST_KEYS) for row in requests):
        raise AdapterError("requests.jsonl does not have the exact canonical request fields")
    if any(set(row) != set(SESSION_KEYS) for row in sessions):
        raise AdapterError("sessions.jsonl does not have the exact canonical session fields")
    if requests != sorted(requests, key=lambda row: (row["arrival_time_s"], row["request_id"])):
        raise AdapterError("requests are not sorted by (arrival_time_s, request_id)")
    region_ids, gateway_ids = {row["region_id"] for row in regions}, {row["gateway_id"] for row in gateways}
    if len({row["request_id"] for row in requests}) != len(requests):
        raise AdapterError("request_id values are not unique")
    horizon = float(manifest["time"]["time_horizon_s"])
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in requests:
        if row["region_id"] != row["user_region_id"] or row["region_id"] not in region_ids or row["gateway_id"] not in gateway_ids:
            raise AdapterError("request region/gateway reference error")
        if not (0 <= row["arrival_time_s"] < horizon) or row["prompt_tokens"] != row["context_tokens_before_turn"] + row["new_user_tokens"]:
            raise AdapterError("request time or token invariant error")
        if row["input_bytes"] <= 0 or row["output_bytes"] <= 0 or row["session_id"] is None:
            raise AdapterError("request byte/session invariant error")
        by_session[row["session_id"]].append(row)
    if {row["session_id"] for row in sessions} != set(by_session):
        raise AdapterError("sessions.jsonl does not match request sessions")
    for session in sessions:
        rows = sorted(by_session[session["session_id"]], key=lambda row: row["turn_index"])
        if [row["turn_index"] for row in rows] != list(range(len(rows))) or session["session_turn_count"] != len(rows):
            raise AdapterError("session turn indices/count are invalid")
        if any(row["slo_class"] != session["session_slo_class"] for row in rows):
            raise AdapterError("a canonical session must have one SLO class")
    if manifest["counts"]["request_count"] != len(requests) or manifest["counts"]["session_count"] != len(sessions):
        raise AdapterError("manifest count mismatch")
    summary = read_json(bundle_dir / "workload_summary.json")
    if summary.get("provenance", {}).get("description_sha256") != manifest.get("description_sha256"):
        raise AdapterError("summary and manifest description hashes do not match")
    if any(summary.get(key) != manifest["counts"].get(key) for key in ("request_count", "session_count", "region_count", "gateway_count", "timebin_count")):
        raise AdapterError("summary and manifest counts do not match")
    if description_path is not None:
        description = read_json(description_path)
        if description.get("schema_version") != DESCRIPTION_SCHEMA or manifest["description_sha256"] != _description_hash(description):
            raise AdapterError("description schema/hash mismatch")
        required_description = {"schema_version", "workload_id", "source_mode", "seed", "time", "geography", "access", "arrival", "tokens", "slo", "sessions", "bursts", "source", "validation", "output", "provenance"}
        if not required_description.issubset(description):
            raise AdapterError("description is missing required canonical top-level fields")
        if abs(sum(item["weight"] for item in description["slo"]["classes"]) - 1.0) > 1e-9:
            raise AdapterError("description SLO class weights must sum to one")
        if description["sessions"].get("session_ratio") != 1.0:
            raise AdapterError("explicit WildChat sessions require session_ratio=1.0")
        if description["geography"].get("regions") != regions or description["access"].get("gateways") != gateways:
            raise AdapterError("description geography/access does not match bundle catalogs")
    print(f"validated bundle: {len(requests)} requests, {len(sessions)} sessions")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="map one official WildChat Parquet shard to a bundle")
    build_parser.add_argument("--config", type=Path, required=True)
    build_parser.add_argument("--input", type=Path, required=True, help="verified local official .parquet shard")
    build_parser.add_argument("--out", type=Path, required=True)
    build_parser.add_argument("--overwrite", action="store_true")
    refresh_parser = subparsers.add_parser("refresh-metadata", help="repair sidecar metadata for an existing bundle")
    refresh_parser.add_argument("--config", type=Path, required=True)
    refresh_parser.add_argument("--bundle", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate", help="validate a generated bundle")
    validate_parser.add_argument("--bundle", type=Path, required=True)
    validate_parser.add_argument("--description", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args.config, args.input, args.out, args.overwrite)
        elif args.command == "refresh-metadata":
            refresh_metadata(args.config, args.bundle)
        else:
            validate(args.bundle, args.description)
    except AdapterError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
