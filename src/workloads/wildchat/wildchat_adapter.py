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
import random
import re
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


from orbitserve.workload import (
    BUNDLE_FILES, BUNDLE_SCHEMA, CANONICALIZATION, DESCRIPTION_SCHEMA,
    GATEWAY_KEYS, REGION_KEYS, REQUEST_KEYS, SESSION_KEYS, SUMMARY_SCHEMA,
    TIMEBIN_FIELDS, bundle_hash, canonical_json_bytes as canonical_bytes, sha256_bytes,
)


class AdapterError(ValueError):
    """A deterministic, user-correctable adapter input error."""


LOAD_SCALE_FRACTIONS = {"1x": 0.1, "2x": 0.2, "5x": 0.5, "10x": 1.0}
MONTH_COMPRESSED_DAY_MODE = "month_compressed_day"
MACRO_REGION_GROUPING = "macro_region"
COUNTRY_LOCAL_GROUPING = "country_local"
MACRO_REGION_LOCAL_GROUPING = "macro_region_local"
SKELETON_GRAFT_TIME_MODE = "skeleton_graft"
REST_PROFILE_KEY = "__rest__"
GENERATOR_NAME = "wildchat_workload_adapter"
GENERATOR_VERSION = "1.5.0"


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
        schema_names = set(parquet.read_schema(path).names)
        wanted = [name for name in ("conversation", "model", "timestamp") if name in schema_names]
        rows = parquet.read_table(path, columns=wanted or None).to_pylist()
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
            history_ids = _token_id_list(self.tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=False)) if history else []
            prompt_ids = _token_id_list(self.tokenizer.apply_chat_template(history + [user], tokenize=True, add_generation_prompt=True))
            context = len(history_ids)
            new_user = max(1, len(prompt_ids) - context)
        new_user = min(new_user, context_window)
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


def extract_pairs(record: dict[str, Any], source_key: str, config: dict[str, Any], *, light: bool = False, skip_region: bool = False) -> list[dict[str, Any]]:
    """Extract adjacent user→assistant pairs from one WildChat conversation.

    With ``light=True`` the returned pairs omit message contents (no deepcopy);
    only arrival/region/index metadata is kept. Filtering is identical.
    With ``skip_region=True`` (skeleton graft content extraction) pairs keep
    ``region=None`` and region resolution is skipped entirely; the grafted
    slot supplies the region later. Content/timestamp/window filters still
    apply.
    """
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
        region = None if skip_region else find_region(item, mapping, state_centroids, country_centroids)
        try:
            arrival = (parse_utc(assistant.get("timestamp")) - epoch).total_seconds()
        except AdapterError:
            if policy["missing_timestamp_policy"] == "drop":
                history.extend([item, assistant])
                index += 2
                continue
            raise
        if not content_ok or (region is None and not skip_region) or not (0 <= arrival < horizon):
            if policy["invalid_pair_policy"] != "drop" and not content_ok:
                raise AdapterError(f"{source_key}: invalid user→assistant pair")
            if region is None and not skip_region and policy["unknown_region_policy"] != "drop":
                raise AdapterError(f"{source_key}: no configured region for user location")
            history.extend([item, assistant])
            index += 2
            continue
        pairs.append({
            "source_key": source_key,
            "pair_index": len(pairs),
            "arrival_time_s": arrival,
            "region": region,
            "history": None if light else deepcopy(history),
            "user": None if light else item,
            "assistant": None if light else assistant,
        })
        history.extend([item, assistant])
        index += 2
    return pairs


def _token_id_list(encoded: Any) -> list[int]:
    """Unwrap BatchEncoding (transformers>=5) or a plain id list."""
    return list(encoded["input_ids"]) if hasattr(encoded, "keys") else list(encoded)


def _batch_token_counts(
    token_counter: TokenCounter,
    flat_pairs: list[dict[str, Any]],
    context_window: int,
) -> list[tuple[int, int, int, int]]:
    """Return (context, new_user, prompt, output) per pair.

    The huggingface_auto path renders chat templates to text and batch-encodes,
    which is equivalent to apply_chat_template(tokenize=True) but much faster.
    """
    if token_counter.backend == "simple_unicode_v1":
        results = []
        for pair in flat_pairs:
            context, new_user, prompt = token_counter.prompt_counts(pair["history"], pair["user"], context_window)
            results.append((context, new_user, prompt, token_counter.output_tokens(pair["assistant"])))
        return results
    tokenizer = token_counter.tokenizer
    assert tokenizer is not None
    eos_extra = int(token_counter.spec.get("assistant_token_policy") == "content_plus_eos")
    results: list[tuple[int, int, int, int]] = []
    chunk_size = 4096
    for start in range(0, len(flat_pairs), chunk_size):
        part = flat_pairs[start : start + chunk_size]
        prompt_texts = [
            tokenizer.apply_chat_template(pair["history"] + [pair["user"]], tokenize=False, add_generation_prompt=True)
            for pair in part
        ]
        history_texts = [
            tokenizer.apply_chat_template(pair["history"], tokenize=False, add_generation_prompt=False)
            if pair["history"]
            else ""
            for pair in part
        ]
        contents = [str(pair["assistant"].get("content", "")) for pair in part]
        prompt_ids = tokenizer(prompt_texts, add_special_tokens=False)["input_ids"]
        history_ids = tokenizer(history_texts, add_special_tokens=False)["input_ids"]
        content_ids = tokenizer(contents, add_special_tokens=False)["input_ids"]
        for pair, prompt, history, content in zip(part, prompt_ids, history_ids, content_ids, strict=True):
            context = len(history) if pair["history"] else 0
            new_user = min(max(1, len(prompt) - context), context_window)
            context = min(context, max(0, context_window - new_user))
            output = max(1, len(content) + eos_extra)
            results.append((context, new_user, context + new_user, output))
    return results


def extract_session_pairs(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    input_sha256: str,
    *,
    line_numbers: list[int] | None = None,
    light: bool = False,
) -> list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]]:
    """Extract valid (session_id, source_key, raw, pairs) conversations, kept intact.

    With ``light=True`` the pairs omit message contents (used when token counts
    are supplied externally and only scheduling metadata is needed).
    """
    extracted: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] = []
    for position, raw in enumerate(raw_rows):
        line_number = position + 1 if line_numbers is None else line_numbers[position]
        source_key = stable_hex(config["dataset"]["revision"], input_sha256, str(line_number), length=32)
        pairs = extract_pairs(raw, source_key, config, light=light)
        if not pairs:
            continue
        session_id = f"wc_sess_{stable_hex(config['workload_id'], source_key)}"
        extracted.append((session_id, source_key, raw, pairs))
    if not extracted:
        raise AdapterError("no valid user→assistant pairs remained after filtering")
    return extracted


def assemble_requests(
    config: dict[str, Any],
    extracted: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]],
    token_counts: list[tuple[int, int, int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gateways_config = config["gateways"]
    raw_gateways = gateways_config["items"]
    rules = config["slo_policy"]["rules"]
    tokenization = config["tokenization"]
    if config["slo_policy"].get("session_assignment") != "max_output_tokens":
        raise AdapterError("slo_policy.session_assignment must be max_output_tokens for canonical workload session consistency")
    window = int(tokenization["context_window_tokens"])

    all_requests: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    used_regions: dict[str, dict[str, Any]] = {}
    flat_index = 0
    for session_id, source_key, raw, pairs in extracted:
        session_counts = token_counts[flat_index : flat_index + len(pairs)]
        flat_index += len(pairs)
        slo = select_slo(max(count[3] for count in session_counts), rules)
        previous_arrival: float | None = None
        session_rows: list[dict[str, Any]] = []
        for pair, (context, new_user, prompt, output_tokens) in zip(pairs, session_counts, strict=True):
            arrival = float(pair["arrival_time_s"])
            if previous_arrival is not None and arrival < previous_arrival:
                raise AdapterError(f"{session_id}: assistant completion proxy moved backwards")
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


def build_requests(config: dict[str, Any], raw_rows: list[dict[str, Any]], input_sha256: str, *, line_numbers: list[int] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    extracted = extract_session_pairs(config, raw_rows, input_sha256, line_numbers=line_numbers)
    flat_pairs = [pair for _, _, _, pairs in extracted for pair in pairs]
    token_counts = _batch_token_counts(
        TokenCounter(config["tokenization"]), flat_pairs, int(config["tokenization"]["context_window_tokens"])
    )
    return assemble_requests(config, extracted, token_counts)


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


def _expected_source_shas(dataset: dict[str, Any]) -> dict[str, str]:
    files = dataset.get("parquet_files")
    if files:
        return {
            Path(str(item["path"])).name: str(item["expected_source_sha256"])
            for item in files
        }
    if dataset.get("parquet_file") and dataset.get("expected_source_sha256"):
        return {Path(str(dataset["parquet_file"])).name: str(dataset["expected_source_sha256"])}
    return {}


def resolve_load_scale(
    config: dict[str, Any],
    load_scale: str | None,
    scale_flags: dict[str, bool],
    *,
    month_mode: bool,
) -> str:
    flagged = [name for name, enabled in scale_flags.items() if enabled]
    if len(flagged) > 1:
        raise AdapterError(f"load-scale shortcuts are mutually exclusive; got {sorted(flagged)}")
    if flagged and load_scale is not None:
        raise AdapterError(f"--load-scale {load_scale} conflicts with --{flagged[0]}")
    scale = flagged[0] if flagged else load_scale
    if scale is None:
        scale = str(config.get("load", {}).get("default_scale", "10x"))
    if scale not in LOAD_SCALE_FRACTIONS:
        raise AdapterError(f"unsupported load scale {scale!r}; use one of {sorted(LOAD_SCALE_FRACTIONS)}")
    if not month_mode and (flagged or load_scale is not None):
        raise AdapterError(
            "load-scale options require dataset.time_window.time_mode='month_compressed_day'"
        )
    return scale


MACRO_REGION_GROUPS = ("US", "CN", "EU", "OTHER")

_EU_COUNTRY_KEYS = frozenset({
    "albania", "andorra", "austria", "belarus", "belgium", "bosnia_and_herzegovina",
    "bulgaria", "croatia", "cyprus", "czechia", "denmark", "estonia", "finland",
    "france", "germany", "greece", "hungary", "iceland", "ireland", "italy",
    "latvia", "liechtenstein", "lithuania", "luxembourg", "malta", "moldova",
    "monaco", "montenegro", "netherlands", "north_macedonia", "norway", "poland",
    "portugal", "romania", "san_marino", "serbia", "slovakia", "slovenia", "spain",
    "sweden", "switzerland", "ukraine", "united_kingdom", "vatican_city",
})


def macro_region_group(country: Any) -> str:
    """Map a WildChat country string to a macro region group (US/CN/EU/OTHER)."""
    if not isinstance(country, str):
        return "OTHER"
    key = _location_key(country)
    if key == "united_states":
        return "US"
    if key == "china":
        return "CN"
    if key in _EU_COUNTRY_KEYS:
        return "EU"
    return "OTHER"


def session_location(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """(country, state) of a conversation, read from its first user turn.

    WildChat Parquet rows carry no top-level country; the country/state fields
    live on each conversation turn (assistant turns have them null).
    """
    conversation = record.get("conversation")
    if not isinstance(conversation, list):
        return None, None
    for item in conversation:
        if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("country"), str):
            state = item.get("state")
            return item["country"], state if isinstance(state, str) else None
    return None, None


def session_country(record: dict[str, Any]) -> str | None:
    """Country of a conversation, read from its first user turn."""
    return session_location(record)[0]


# ---------------- legal (IANA) timezones for country-local envelopes ----------------
# US states: full names normalized with _location_key (WildChat uses full state
# names). Arizona has no DST (America/Phoenix). Unknown states fall back to
# America/New_York. Mirrors scratch/legal_tz_diurnal.py's US_STATE_TZ.
_US_STATE_TZ: dict[str, str] = {}
for _s in ("connecticut delaware florida georgia indiana kentucky maine maryland massachusetts "
           "michigan new_hampshire new_jersey new_york north_carolina ohio pennsylvania "
           "rhode_island south_carolina vermont virginia west_virginia district_of_columbia").split():
    _US_STATE_TZ[_s] = "America/New_York"
for _s in ("alabama arkansas illinois iowa kansas louisiana minnesota mississippi missouri "
           "nebraska north_dakota oklahoma south_dakota tennessee texas wisconsin").split():
    _US_STATE_TZ[_s] = "America/Chicago"
for _s in "colorado montana new_mexico utah wyoming idaho".split():
    _US_STATE_TZ[_s] = "America/Denver"
_US_STATE_TZ["arizona"] = "America/Phoenix"  # no DST
for _s in "california nevada oregon washington".split():
    _US_STATE_TZ[_s] = "America/Los_Angeles"
_US_STATE_TZ["alaska"] = "America/Anchorage"
_US_STATE_TZ["hawaii"] = "Pacific/Honolulu"
del _s

# Russian states observed in WildChat; unknown/missing states default to Moscow time.
_RU_STATE_TZ: dict[str, str] = {
    "moscow": "Europe/Moscow",
    "moscow_oblast": "Europe/Moscow",
    "yaroslavl_oblast": "Europe/Moscow",
    "st_petersburg": "Europe/Moscow",
    "nizhny_novgorod_oblast": "Europe/Moscow",
    "tatarstan_republic": "Europe/Moscow",
    "bashkortostan_republic": "Europe/Moscow",
    "perm_krai": "Asia/Yekaterinburg",
    "kemerovo_oblast": "Asia/Krasnoyarsk",
    "novosibirsk_oblast": "Asia/Krasnoyarsk",
    "krasnoyarsk_krai": "Asia/Krasnoyarsk",
}

# Country-level zones: the top-10 content countries plus all of Europe (Europe
# dominates the REST envelope, so its local times stay legal-tz precise too).
_COUNTRY_TZ: dict[str, str] = {
    "china": "Asia/Shanghai",
    "japan": "Asia/Tokyo",
    "south_korea": "Asia/Seoul",
    "india": "Asia/Kolkata",
    "brazil": "America/Sao_Paulo",
    "united_kingdom": "Europe/London",
    "ireland": "Europe/Dublin",
    "portugal": "Europe/Lisbon",
    "iceland": "Atlantic/Reykjavik",
    "finland": "Europe/Helsinki",
    "greece": "Europe/Athens",
    "romania": "Europe/Bucharest",
    "bulgaria": "Europe/Sofia",
    "latvia": "Europe/Riga",
    "lithuania": "Europe/Vilnius",
    "estonia": "Europe/Tallinn",
    "cyprus": "Asia/Nicosia",
    "ukraine": "Europe/Kyiv",
    "moldova": "Europe/Chisinau",
    "belarus": "Europe/Minsk",
}
for _c in ("germany austria belgium denmark spain france italy luxembourg netherlands norway sweden "
           "switzerland poland czechia slovakia slovenia hungary croatia malta vatican_city san_marino "
           "monaco liechtenstein andorra albania bosnia_and_herzegovina montenegro north_macedonia serbia").split():
    _COUNTRY_TZ[_c] = "Europe/Berlin"
del _c


def legal_tz_name(country: Any, state: Any) -> str | None:
    """IANA timezone name for a WildChat (country, state) pair, if mapped."""
    if not isinstance(country, str):
        return None
    key = _location_key(country)
    state_key = _location_key(state) if isinstance(state, str) else ""
    if key == "united_states":
        return _US_STATE_TZ.get(state_key, "America/New_York")
    if key == "russia":
        return _RU_STATE_TZ.get(state_key, "Europe/Moscow")
    return _COUNTRY_TZ.get(key)


_TZ_OFFSET_CACHE: dict[tuple[str, str], int] = {}


def local_time_offset_s(country: Any, state: Any, reference_date: str, *, fallback_lon_deg: Any = None) -> int:
    """UTC offset (seconds) of the session's legal timezone on reference_date.

    The reference date (YYYY-MM-DD) pins the DST regime of the represented
    month (evaluated at local noon via zoneinfo). Countries without a legal-tz
    mapping fall back to the region's longitude solar offset. Missing IANA tz
    data is a hard AdapterError (install the tzdata package), never a silent
    fallback.
    """
    tz_name = legal_tz_name(country, state)
    if tz_name is None:
        if fallback_lon_deg is None:
            raise AdapterError(f"no legal timezone or longitude fallback for country {country!r}")
        return max(-12, min(14, round(float(fallback_lon_deg) / 15.0))) * 3600
    cache_key = (tz_name, reference_date)
    offset = _TZ_OFFSET_CACHE.get(cache_key)
    if offset is None:
        try:
            zone = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError as exc:
            raise AdapterError(
                f"IANA timezone {tz_name!r} is unavailable; install the tzdata package (pip install tzdata)"
            ) from exc
        try:
            year, month, day = (int(part) for part in reference_date.split("-"))
            delta = zone.utcoffset(datetime(year, month, day, 12, 0))
        except ValueError as exc:
            raise AdapterError(f"invalid diurnal_profile reference date {reference_date!r}") from exc
        if delta is None:
            raise AdapterError(f"cannot resolve the UTC offset of {tz_name} on {reference_date}")
        offset = int(delta.total_seconds())
        _TZ_OFFSET_CACHE[cache_key] = offset
    return offset


def collect_month_pool(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    input_sha256: str,
    *,
    line_offset: int = 0,
    pools: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Scan source rows and bucket sessions by time-of-day without retaining content.

    Pass an existing pools dict plus line_offset to accumulate across multiple
    input shards while keeping global (1-based) line numbers stable.
    """
    time_window = config["dataset"]["time_window"]
    bin_s = float(time_window["time_bin_s"])
    if pools is None:
        pools = defaultdict(list)
    for position, raw in enumerate(raw_rows):
        line_number = position + 1 + line_offset
        source_key = stable_hex(config["dataset"]["revision"], input_sha256, str(line_number), length=32)
        pairs = extract_pairs(raw, source_key, config, light=True)
        if not pairs:
            continue
        first = float(pairs[0]["arrival_time_s"])
        span = float(pairs[-1]["arrival_time_s"]) - first
        bin_index = int((first % 86400.0) // bin_s)
        session_id = f"wc_sess_{stable_hex(config['workload_id'], source_key)}"
        pools[bin_index].append(
            {
                "line_number": line_number,
                "session_id": session_id,
                "first_arrival": first,
                "span": span,
            }
        )
    return pools


def sample_month_pool(
    config: dict[str, Any],
    pools: dict[int, list[dict[str, Any]]],
    scale: str,
) -> tuple[dict[str, str], dict[str, float], list[dict[str, Any]], list[str]]:
    """Deterministically sample each bin's pool and assign session starts."""
    time_window = config["dataset"]["time_window"]
    bin_s = float(time_window["time_bin_s"])
    horizon = float(time_window.get("output_time_horizon_s", 86400.0))
    fraction = LOAD_SCALE_FRACTIONS[scale]
    if not pools:
        raise AdapterError("no valid user→assistant pairs remained after filtering")

    line_to_session: dict[str, float] = {}
    session_starts: dict[int, float] = {}
    report: list[dict[str, Any]] = []
    warnings: list[str] = []
    for bin_index in sorted(pools):
        pool = sorted(pools[bin_index], key=lambda entry: entry["session_id"])
        rng = random.Random(stable_seed(config["workload_id"], scale, str(bin_index)))
        target = len(pool) if fraction >= 1.0 else int(round(len(pool) * fraction))
        if target > len(pool):
            warnings.append(
                f"bin {bin_index}: target {target} sessions exceeds pool {len(pool)}; taking all"
            )
            target = len(pool)
        chosen = pool if target == len(pool) else sorted(rng.sample(pool, target), key=lambda entry: entry["session_id"])
        dropped_long = 0
        for entry in chosen:
            offset = rng.uniform(0.0, bin_s)
            max_start = horizon - entry["span"] - 1e-6
            if max_start < 0:
                dropped_long += 1
                continue
            start = round(min(bin_index * bin_s + offset, max_start), 6)
            line_to_session[str(entry["line_number"])] = entry["session_id"]
            session_starts[entry["session_id"]] = start
        if dropped_long:
            warnings.append(
                f"bin {bin_index}: dropped {dropped_long} session(s) spanning more than the synthetic day"
            )
        report.append(
            {"bin": bin_index, "pool": len(pool), "target": target, "actual": len(chosen) - dropped_long}
        )
    return line_to_session, session_starts, report, warnings


def plan_month_compression(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    input_sha256: str,
    scale: str,
) -> tuple[dict[str, str], dict[str, float], list[dict[str, Any]], list[str]]:
    """Assign each source conversation to a time-of-day bin and sample per bin.

    Returns (line_number -> session_id, session_id -> compressed first arrival,
    per-bin report rows, warnings). Conversations are kept intact: the whole
    session is shifted so its first turn lands inside the sampled bin slot.
    """
    pools = collect_month_pool(config, raw_rows, input_sha256)
    return sample_month_pool(config, pools, scale)


def _diurnal_profile_scan_config(config: dict[str, Any]) -> dict[str, Any]:
    """Config view whose dataset time window matches the profile shards' month.

    The profile shards come from a different month than the content shards, so
    their pair extraction must filter against the profile window declared in
    the optional ``diurnal_profile`` config block (defaulting to the dataset
    window when the block omits it).
    """
    profile_block = config.get("diurnal_profile")
    if not isinstance(profile_block, dict):
        raise AdapterError("--profile-input requires a diurnal_profile block in the config")
    grouping = profile_block.get("grouping", MACRO_REGION_GROUPING)
    if grouping not in (MACRO_REGION_GROUPING, COUNTRY_LOCAL_GROUPING, MACRO_REGION_LOCAL_GROUPING):
        raise AdapterError(f"unsupported diurnal_profile.grouping: {grouping!r}")
    scan_config = dict(config)
    dataset = dict(config["dataset"])
    time_window = dict(dataset["time_window"])
    epoch = str(profile_block.get("epoch_utc", time_window["epoch_utc"]))
    time_window["epoch_utc"] = epoch
    window_end = profile_block.get("window_end_utc")
    if window_end:
        span = (parse_utc(str(window_end)) - parse_utc(epoch)).total_seconds()
        if span <= 0:
            raise AdapterError("diurnal_profile.window_end_utc must be after epoch_utc")
        time_window["time_horizon_s"] = span
    dataset["time_window"] = time_window
    scan_config["dataset"] = dataset
    return scan_config


def collect_profile_counts(
    scan_config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    counts: dict[str, Counter] | None = None,
) -> dict[str, Counter]:
    """Count profile sessions per (macro group, time-of-day bin); no content kept.

    Mirrors collect_month_pool's light scan, but only tallies sessions: the bin
    definition is identical (first-turn arrival_time_s % 86400 // time_bin_s).
    Pass an existing counts dict to accumulate across profile shards.
    """
    bin_s = float(scan_config["dataset"]["time_window"]["time_bin_s"])
    if counts is None:
        counts = {group: Counter() for group in MACRO_REGION_GROUPS}
    for raw in raw_rows:
        pairs = extract_pairs(raw, "diurnal_profile", scan_config, light=True)
        if not pairs:
            continue
        first = float(pairs[0]["arrival_time_s"])
        bin_index = int((first % 86400.0) // bin_s)
        counts[macro_region_group(session_country(raw))][bin_index] += 1
    return counts


def collect_grouped_candidates(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    input_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    """Scan content shards into per-group candidate sessions (no content kept).

    Every country participates (no allowlist filtering). The state/lon_deg
    fields are only consumed by local-time groupings.
    """
    candidates: dict[str, list[dict[str, Any]]] = {group: [] for group in MACRO_REGION_GROUPS}
    for position, raw in enumerate(raw_rows):
        line_number = position + 1
        source_key = stable_hex(config["dataset"]["revision"], input_sha256, str(line_number), length=32)
        pairs = extract_pairs(raw, source_key, config, light=True)
        if not pairs:
            continue
        country, state = session_location(raw)
        first = float(pairs[0]["arrival_time_s"])
        span = float(pairs[-1]["arrival_time_s"]) - first
        session_id = f"wc_sess_{stable_hex(config['workload_id'], source_key)}"
        candidates[macro_region_group(country)].append(
            {
                "line_number": line_number,
                "session_id": session_id,
                "first_arrival": first,
                "span": span,
                "country": country,
                "state": state,
                "lon_deg": pairs[0]["region"].get("lon_deg"),
            }
        )
    return candidates


def _bin_weights(counts: Counter) -> dict[int, float]:
    total = sum(counts.values())
    return {bin_index: count / total for bin_index, count in counts.items() if count > 0}


def _largest_remainder_targets(weights: dict[int, float], total: int) -> dict[int, int]:
    """Apportion ``total`` sessions across bins by weight (largest remainder).

    Ties on fractional parts are broken by ascending bin index so the result
    is deterministic.
    """
    quotas = {bin_index: total * weight for bin_index, weight in weights.items()}
    targets = {bin_index: int(math.floor(quota)) for bin_index, quota in quotas.items()}
    remainder = total - sum(targets.values())
    by_fraction = sorted(quotas, key=lambda bin_index: (-(quotas[bin_index] - targets[bin_index]), bin_index))
    for bin_index in by_fraction[:remainder]:
        targets[bin_index] += 1
    return targets


def deal_grouped_sessions(
    config: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    profile_counts: dict[str, Counter],
    scale: str,
) -> tuple[dict[str, str], dict[str, float], list[dict[str, Any]], list[str]]:
    """Deal each group's sampled sessions into bins following the profile.

    Per group: sample round(N*fraction) candidates (stable_seed(workload_id,
    scale, group)), apportion them across bins with the largest-remainder
    method using the group's profile weights (empty groups fall back to the
    global profile distribution), then shuffle with stable_seed(workload_id,
    scale, group, "deal") and deal into bins in ascending bin order.
    """
    time_window = config["dataset"]["time_window"]
    bin_s = float(time_window["time_bin_s"])
    horizon = float(time_window.get("output_time_horizon_s", 86400.0))
    fraction = LOAD_SCALE_FRACTIONS[scale]
    if not any(candidates[group] for group in MACRO_REGION_GROUPS):
        raise AdapterError("no valid user→assistant pairs remained after filtering")
    global_counts: Counter = Counter()
    for group in MACRO_REGION_GROUPS:
        global_counts.update(profile_counts.get(group, {}))
    if sum(global_counts.values()) == 0:
        raise AdapterError("diurnal profile shards contain no usable sessions")

    line_to_session: dict[str, str] = {}
    session_starts: dict[str, float] = {}
    report: list[dict[str, Any]] = []
    warnings: list[str] = []
    for group in MACRO_REGION_GROUPS:
        pool = sorted(candidates[group], key=lambda entry: entry["session_id"])
        if not pool:
            continue
        target = len(pool) if fraction >= 1.0 else int(round(len(pool) * fraction))
        chosen = pool
        if target < len(pool):
            rng_pick = random.Random(stable_seed(config["workload_id"], scale, group))
            chosen = sorted(rng_pick.sample(pool, target), key=lambda entry: entry["session_id"])
        group_counts = profile_counts.get(group, Counter())
        if sum(group_counts.values()) == 0:
            warnings.append(
                f"group {group}: no diurnal profile sessions; falling back to the global profile distribution"
            )
            group_counts = global_counts
        targets = _largest_remainder_targets(_bin_weights(group_counts), len(chosen))
        rng_deal = random.Random(stable_seed(config["workload_id"], scale, group, "deal"))
        deck = list(chosen)
        rng_deal.shuffle(deck)
        position = 0
        for bin_index in sorted(targets):
            bin_target = targets[bin_index]
            if bin_target == 0:
                continue
            dealt = deck[position : position + bin_target]
            position += bin_target
            dropped_long = 0
            for entry in dealt:
                offset = rng_deal.uniform(0.0, bin_s)
                max_start = horizon - entry["span"] - 1e-6
                if max_start < 0:
                    dropped_long += 1
                    continue
                start = round(min(bin_index * bin_s + offset, max_start), 6)
                line_to_session[str(entry["line_number"])] = entry["session_id"]
                session_starts[entry["session_id"]] = start
            if dropped_long:
                warnings.append(
                    f"group {group} bin {bin_index}: dropped {dropped_long} session(s) spanning more than the synthetic day"
                )
            report.append(
                {"group": group, "bin": bin_index, "pool": len(pool), "target": bin_target, "actual": len(dealt) - dropped_long}
            )
    return line_to_session, session_starts, report, warnings


def plan_diurnal_profile_warp(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    input_sha256: str,
    profile_counts: dict[str, Counter],
    scale: str,
) -> tuple[dict[str, str], dict[str, float], list[dict[str, Any]], list[str]]:
    """Assign session starts from a foreign-month diurnal profile.

    Returns the same tuple as plan_month_compression
    (line_number -> session_id, session_id -> warped first arrival, report
    rows with an extra ``group`` field, warnings) so the downstream shift and
    validation path is reused unchanged. Arrival times follow the profile
    shards' per-group day envelope while conversation content comes from the
    input shards.
    """
    candidates = collect_grouped_candidates(config, raw_rows, input_sha256)
    return deal_grouped_sessions(config, candidates, profile_counts, scale)


def _local_envelope_settings(config: dict[str, Any], *, grouping: str) -> dict[str, Any]:
    """Validated diurnal_profile block for local-time envelope groupings."""
    profile_block = config.get("diurnal_profile")
    if not isinstance(profile_block, dict):
        raise AdapterError("--profile-input requires a diurnal_profile block in the config")
    required = {"profile_reference_date", "content_reference_date"}
    if grouping == COUNTRY_LOCAL_GROUPING:
        required.add("content_countries")
    missing = required - set(profile_block)
    if missing:
        raise AdapterError(f"diurnal_profile ({grouping}) is missing fields: {sorted(missing)}")
    if grouping == COUNTRY_LOCAL_GROUPING:
        countries = profile_block["content_countries"]
        if not isinstance(countries, list) or not countries or any(not isinstance(item, str) for item in countries):
            raise AdapterError("diurnal_profile.content_countries must be a non-empty array of strings")
    for key in ("profile_reference_date", "content_reference_date"):
        try:
            date.fromisoformat(str(profile_block[key]))
        except ValueError as exc:
            raise AdapterError(f"diurnal_profile.{key} must be an ISO date (YYYY-MM-DD)") from exc
    return profile_block


def _country_local_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Validated diurnal_profile block for grouping='country_local'."""
    return _local_envelope_settings(config, grouping=COUNTRY_LOCAL_GROUPING)


def _country_allowlist_keys(profile_block: dict[str, Any]) -> dict[str, str]:
    """Normalized country key -> configured display name, in config order."""
    keys: dict[str, str] = {}
    for country in profile_block["content_countries"]:
        keys.setdefault(_location_key(str(country)), str(country))
    return keys


def _scan_local_profile_row(
    scan_config: dict[str, Any],
    raw: dict[str, Any],
    reference_date: str,
    bin_s: float,
) -> tuple[str | None, int] | None:
    """(country, local-time bin) of one profile session, or None if filtered.

    Local bin = ((first-turn arrival + the session's legal UTC offset on
    reference_date) mod 86400) // bin_s.
    """
    pairs = extract_pairs(raw, "diurnal_profile", scan_config, light=True)
    if not pairs:
        return None
    country, state = session_location(raw)
    first = float(pairs[0]["arrival_time_s"])
    offset = local_time_offset_s(
        country, state, reference_date, fallback_lon_deg=pairs[0]["region"].get("lon_deg")
    )
    return country, int(((first + offset) % 86400.0) // bin_s)


def collect_local_profile_counts(
    scan_config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    allowlist_keys: dict[str, str],
    reference_date: str,
    counts: dict[str, Counter] | None = None,
) -> dict[str, Counter]:
    """Count profile sessions per (country, local-time bin); no content kept.

    Allowlisted countries get their own counter; every other session folds
    into the REST counter.
    """
    bin_s = float(scan_config["dataset"]["time_window"]["time_bin_s"])
    if counts is None:
        counts = {country: Counter() for country in allowlist_keys.values()}
        counts[REST_PROFILE_KEY] = Counter()
    for raw in raw_rows:
        scanned = _scan_local_profile_row(scan_config, raw, reference_date, bin_s)
        if scanned is None:
            continue
        country, local_bin = scanned
        key = allowlist_keys.get(_location_key(country) if isinstance(country, str) else "", REST_PROFILE_KEY)
        counts[key][local_bin] += 1
    return counts


def collect_macro_local_profile_counts(
    scan_config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    reference_date: str,
    counts: dict[str, Counter] | None = None,
) -> dict[str, Counter]:
    """Count profile sessions per (macro group, local-time bin); no content kept.

    Same local-time binning as collect_local_profile_counts, aggregated by
    macro_region_group (US/CN/EU/OTHER); every country participates.
    """
    bin_s = float(scan_config["dataset"]["time_window"]["time_bin_s"])
    if counts is None:
        counts = {group: Counter() for group in MACRO_REGION_GROUPS}
    for raw in raw_rows:
        scanned = _scan_local_profile_row(scan_config, raw, reference_date, bin_s)
        if scanned is None:
            continue
        country, local_bin = scanned
        counts[macro_region_group(country)][local_bin] += 1
    return counts


def collect_country_candidates(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    input_sha256: str,
    allowlist_keys: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Scan content shards into per-country candidates (no content kept).

    Sessions whose country is not on the allowlist are skipped outright and
    never enter the candidate pools.
    """
    candidates: dict[str, list[dict[str, Any]]] = {country: [] for country in allowlist_keys.values()}
    for position, raw in enumerate(raw_rows):
        country, state = session_location(raw)
        key = allowlist_keys.get(_location_key(country) if isinstance(country, str) else "")
        if key is None:
            continue
        line_number = position + 1
        source_key = stable_hex(config["dataset"]["revision"], input_sha256, str(line_number), length=32)
        pairs = extract_pairs(raw, source_key, config, light=True)
        if not pairs:
            continue
        first = float(pairs[0]["arrival_time_s"])
        span = float(pairs[-1]["arrival_time_s"]) - first
        session_id = f"wc_sess_{stable_hex(config['workload_id'], source_key)}"
        candidates[key].append(
            {
                "line_number": line_number,
                "session_id": session_id,
                "first_arrival": first,
                "span": span,
                "state": state,
                "lon_deg": pairs[0]["region"].get("lon_deg"),
            }
        )
    return candidates


def deal_country_sessions(
    config: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    profile_counts: dict[str, Counter],
    scale: str,
) -> tuple[dict[str, str], dict[str, float], list[dict[str, Any]], list[str], dict[str, Any]]:
    """Deal each country's sampled sessions into local-time bins.

    Same flow as deal_grouped_sessions, but bins are *local* times: a dealt
    local start bin*bin_s + rng.uniform(0, bin_s) is converted back to UTC via
    the session region's legal offset on the content reference date. Countries
    whose profile holds fewer than min_country_profile_sessions sessions use
    the REST local-time envelope instead of their own.
    """
    profile_block = _country_local_settings(config)
    allowlist = [str(country) for country in profile_block["content_countries"]]
    min_sessions = int(profile_block.get("min_country_profile_sessions", 1000))
    content_reference = str(profile_block["content_reference_date"])
    time_window = config["dataset"]["time_window"]
    bin_s = float(time_window["time_bin_s"])
    horizon = float(time_window.get("output_time_horizon_s", 86400.0))
    fraction = LOAD_SCALE_FRACTIONS[scale]
    if not any(candidates.get(country) for country in allowlist):
        raise AdapterError("no valid user→assistant pairs remained after filtering")
    rest_counts = profile_counts.get(REST_PROFILE_KEY, Counter())
    if sum(rest_counts.values()) == 0:
        raise AdapterError("diurnal profile shards contain no usable REST sessions")

    line_to_session: dict[str, str] = {}
    session_starts: dict[str, float] = {}
    report: list[dict[str, Any]] = []
    warnings: list[str] = []
    fallback_countries: list[str] = []
    envelope_sessions: dict[str, int] = {}
    for country in allowlist:
        pool = sorted(candidates.get(country, []), key=lambda entry: entry["session_id"])
        if not pool:
            continue
        target = len(pool) if fraction >= 1.0 else int(round(len(pool) * fraction))
        chosen = pool
        if target < len(pool):
            rng_pick = random.Random(stable_seed(config["workload_id"], scale, country))
            chosen = sorted(rng_pick.sample(pool, target), key=lambda entry: entry["session_id"])
        country_counts = profile_counts.get(country, Counter())
        country_total = sum(country_counts.values())
        if country_total < min_sessions:
            fallback_countries.append(country)
            warnings.append(
                f"country {country}: only {country_total} profile sessions (< {min_sessions}); "
                f"falling back to the REST local-time envelope"
            )
            country_counts = rest_counts
        envelope_sessions[country] = sum(country_counts.values())
        targets = _largest_remainder_targets(_bin_weights(country_counts), len(chosen))
        rng_deal = random.Random(stable_seed(config["workload_id"], scale, country, "deal"))
        deck = list(chosen)
        rng_deal.shuffle(deck)
        position = 0
        for bin_index in sorted(targets):
            bin_target = targets[bin_index]
            if bin_target == 0:
                continue
            dealt = deck[position : position + bin_target]
            position += bin_target
            dropped_long = 0
            for entry in dealt:
                offset = rng_deal.uniform(0.0, bin_s)
                max_start = horizon - entry["span"] - 1e-6
                if max_start < 0:
                    dropped_long += 1
                    continue
                local_start = bin_index * bin_s + offset
                utc_offset = local_time_offset_s(
                    country, entry.get("state"), content_reference, fallback_lon_deg=entry.get("lon_deg")
                )
                utc_start = (local_start - utc_offset) % 86400.0
                start = round(min(utc_start, max_start), 6)
                line_to_session[str(entry["line_number"])] = entry["session_id"]
                session_starts[entry["session_id"]] = start
            if dropped_long:
                warnings.append(
                    f"country {country} bin {bin_index}: dropped {dropped_long} session(s) spanning more than the synthetic day"
                )
            report.append(
                {"country": country, "bin": bin_index, "pool": len(pool), "target": bin_target, "actual": len(dealt) - dropped_long}
            )
    info = {"fallback_countries": fallback_countries, "envelope_sessions": envelope_sessions}
    return line_to_session, session_starts, report, warnings, info


def plan_country_local_warp(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    input_sha256: str,
    profile_counts: dict[str, Counter],
    scale: str,
) -> tuple[dict[str, str], dict[str, float], list[dict[str, Any]], list[str], dict[str, Any]]:
    """Assign session starts from per-country local-time profile envelopes.

    Returns plan_month_compression's tuple (report rows carry a ``country``
    field) plus an info dict with the REST-fallback country list and per-
    country envelope sizes, so the shift/validation path is reused unchanged.
    """
    allowlist_keys = _country_allowlist_keys(_country_local_settings(config))
    candidates = collect_country_candidates(config, raw_rows, input_sha256, allowlist_keys)
    return deal_country_sessions(config, candidates, profile_counts, scale)


def deal_macro_local_sessions(
    config: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    profile_counts: dict[str, Counter],
    scale: str,
) -> tuple[dict[str, str], dict[str, float], list[dict[str, Any]], list[str], dict[str, Any]]:
    """Deal each macro group's sampled sessions into local-time bins.

    Same flow as deal_country_sessions but grouped by macro_region_group
    (US/CN/EU/OTHER) with no country filtering: a dealt local start
    bin*bin_s + rng.uniform(0, bin_s) converts back to UTC via the session
    region's legal offset on the content reference date. Groups whose profile
    is empty fall back to the global local-time distribution (all profile
    sessions merged).
    """
    profile_block = _local_envelope_settings(config, grouping=MACRO_REGION_LOCAL_GROUPING)
    content_reference = str(profile_block["content_reference_date"])
    time_window = config["dataset"]["time_window"]
    bin_s = float(time_window["time_bin_s"])
    horizon = float(time_window.get("output_time_horizon_s", 86400.0))
    fraction = LOAD_SCALE_FRACTIONS[scale]
    if not any(candidates[group] for group in MACRO_REGION_GROUPS):
        raise AdapterError("no valid user→assistant pairs remained after filtering")
    global_counts: Counter = Counter()
    for group in MACRO_REGION_GROUPS:
        global_counts.update(profile_counts.get(group, {}))
    if sum(global_counts.values()) == 0:
        raise AdapterError("diurnal profile shards contain no usable sessions")

    line_to_session: dict[str, str] = {}
    session_starts: dict[str, float] = {}
    report: list[dict[str, Any]] = []
    warnings: list[str] = []
    fallback_groups: list[str] = []
    envelope_sessions: dict[str, int] = {}
    for group in MACRO_REGION_GROUPS:
        pool = sorted(candidates[group], key=lambda entry: entry["session_id"])
        if not pool:
            continue
        target = len(pool) if fraction >= 1.0 else int(round(len(pool) * fraction))
        chosen = pool
        if target < len(pool):
            rng_pick = random.Random(stable_seed(config["workload_id"], scale, group))
            chosen = sorted(rng_pick.sample(pool, target), key=lambda entry: entry["session_id"])
        group_counts = profile_counts.get(group, Counter())
        if sum(group_counts.values()) == 0:
            fallback_groups.append(group)
            warnings.append(
                f"group {group}: no diurnal profile sessions; falling back to the global local-time distribution"
            )
            group_counts = global_counts
        envelope_sessions[group] = sum(group_counts.values())
        targets = _largest_remainder_targets(_bin_weights(group_counts), len(chosen))
        rng_deal = random.Random(stable_seed(config["workload_id"], scale, group, "deal"))
        deck = list(chosen)
        rng_deal.shuffle(deck)
        position = 0
        for bin_index in sorted(targets):
            bin_target = targets[bin_index]
            if bin_target == 0:
                continue
            dealt = deck[position : position + bin_target]
            position += bin_target
            dropped_long = 0
            for entry in dealt:
                offset = rng_deal.uniform(0.0, bin_s)
                max_start = horizon - entry["span"] - 1e-6
                if max_start < 0:
                    dropped_long += 1
                    continue
                local_start = bin_index * bin_s + offset
                utc_offset = local_time_offset_s(
                    entry.get("country"), entry.get("state"), content_reference,
                    fallback_lon_deg=entry.get("lon_deg"),
                )
                utc_start = (local_start - utc_offset) % 86400.0
                start = round(min(utc_start, max_start), 6)
                line_to_session[str(entry["line_number"])] = entry["session_id"]
                session_starts[entry["session_id"]] = start
            if dropped_long:
                warnings.append(
                    f"group {group} bin {bin_index}: dropped {dropped_long} session(s) spanning more than the synthetic day"
                )
            report.append(
                {"group": group, "bin": bin_index, "pool": len(pool), "target": bin_target, "actual": len(dealt) - dropped_long}
            )
    info = {"fallback_groups": fallback_groups, "envelope_sessions": envelope_sessions}
    return line_to_session, session_starts, report, warnings, info


def plan_macro_local_warp(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    input_sha256: str,
    profile_counts: dict[str, Counter],
    scale: str,
) -> tuple[dict[str, str], dict[str, float], list[dict[str, Any]], list[str], dict[str, Any]]:
    """Assign session starts from per-macro-group local-time profile envelopes.

    Returns plan_month_compression's tuple (report rows carry a ``group``
    field) plus an info dict with the fallback group list and per-group
    envelope sizes, so the shift/validation path is reused unchanged.
    """
    _local_envelope_settings(config, grouping=MACRO_REGION_LOCAL_GROUPING)
    candidates = collect_grouped_candidates(config, raw_rows, input_sha256)
    return deal_macro_local_sessions(config, candidates, profile_counts, scale)


# ---------------- skeleton graft ("2023 skeleton + 2025 content") ----------------
# region_name 的国家部分（"State, Country" 或 "Country"）与 WildChat 内容里的
# country 字符串之间的已知差异（键为 _location_key 归一化后的槽位国家名）。
_SLOT_COUNTRY_ALIASES: dict[str, str] = {
    "netherlands": "The Netherlands",
    "turkey": "Turkiye",
    "united_states_of_america": "United States",
    "russian_federation": "Russia",
    "republic_of_korea": "South Korea",
}


def slot_country(region_name: str) -> str:
    """Country of a skeleton slot, parsed from its region_name."""
    return region_name.split(", ")[-1].strip()


def load_skeleton_slots(skeleton_dir: Path, skeleton_block: dict[str, Any]) -> list[dict[str, Any]]:
    """Load (region, start, country) slots from an existing bundle, kept intact."""
    regions_name = str(skeleton_block.get("regions_file", "regions.json"))
    sessions_name = str(skeleton_block.get("sessions_file", "sessions.jsonl"))
    try:
        regions = {row["region_id"]: row for row in read_json_array(skeleton_dir / regions_name)}
    except AdapterError as exc:
        raise AdapterError(f"cannot read skeleton bundle regions: {exc}") from exc
    try:
        session_rows = read_jsonl(skeleton_dir / sessions_name)
    except AdapterError as exc:
        raise AdapterError(f"cannot read skeleton bundle sessions: {exc}") from exc
    slots: list[dict[str, Any]] = []
    for row in session_rows:
        region = regions.get(row["user_region_id"])
        if region is None:
            raise AdapterError(f"skeleton session {row.get('session_id')} references unknown region {row.get('user_region_id')}")
        slots.append(
            {
                "slot_session_id": str(row["session_id"]),
                "region_id": region["region_id"],
                "start": float(row["session_start_time_s"]),
                "region": region,
                "country": slot_country(str(region["region_name"])),
            }
        )
    if not slots:
        raise AdapterError(f"skeleton bundle {skeleton_dir} contains no sessions")
    slots.sort(key=lambda slot: (slot["start"], slot["slot_session_id"]))
    return slots


def collect_graft_content_pool(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    input_sha256: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Extract full 2025 content conversations into per-country pools.

    Region resolution is skipped (slots supply regions); conversations whose
    country is missing form the global fallback pool. Everything else (content,
    timestamp, content-month window filters) matches extract_pairs.
    """
    pool: dict[str, list[dict[str, Any]]] = {}
    global_pool: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_rows):
        line_number = position + 1
        source_key = stable_hex(config["dataset"]["revision"], input_sha256, str(line_number), length=32)
        pairs = extract_pairs(raw, source_key, config, skip_region=True)
        if not pairs:
            continue
        first = float(pairs[0]["arrival_time_s"])
        entry = {
            "line_number": line_number,
            "source_key": source_key,
            "raw": raw,
            "pairs": pairs,
            "span": float(pairs[-1]["arrival_time_s"]) - first,
        }
        country, _state = session_location(raw)
        if country is None:
            global_pool.append(entry)
        else:
            pool.setdefault(country, []).append(entry)
    if not pool and not global_pool:
        raise AdapterError("no valid user→assistant pairs remained after filtering")
    return pool, global_pool


def _content_pool_key(slot_country_name: str, pool: dict[str, list[dict[str, Any]]]) -> str | None:
    """Map a skeleton slot country to a content pool key (exact/alias/normalized)."""
    if slot_country_name in pool:
        return slot_country_name
    alias = _SLOT_COUNTRY_ALIASES.get(_location_key(slot_country_name))
    if alias is not None and alias in pool:
        return alias
    key = _location_key(slot_country_name)
    for name in pool:
        if _location_key(name) == key:
            return name
    return None


def plan_skeleton_graft(
    config: dict[str, Any],
    slots: list[dict[str, Any]],
    pool: dict[str, list[dict[str, Any]]],
    global_pool: list[dict[str, Any]],
    scale: str,
) -> tuple[list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]], dict[str, Any], list[str]]:
    """Graft 2025 content conversations onto 2023 skeleton slots.

    Slot regions and start times are preserved verbatim. Each slot draws one
    content conversation from its country's pool with replacement
    (stable_seed(workload_id, slot session_id)); empty country pools fall back
    to the global (null-country) pool. A grafted session is dropped (with a
    warning) when slot start + content span would cross the synthetic day.
    Returns (extracted tuples for assemble_requests, stats, warnings).
    """
    fraction = LOAD_SCALE_FRACTIONS[scale]
    time_window = config["dataset"]["time_window"]
    horizon = float(time_window.get("output_time_horizon_s", 86400.0))
    ordered = sorted(slots, key=lambda slot: slot["slot_session_id"])
    target = len(ordered) if fraction >= 1.0 else int(round(len(ordered) * fraction))
    chosen = ordered
    if target < len(ordered):
        rng_slots = random.Random(stable_seed(config["workload_id"], scale, "slots"))
        chosen = sorted(rng_slots.sample(ordered, target), key=lambda slot: (slot["start"], slot["slot_session_id"]))

    extracted: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] = []
    warnings: list[str] = []
    stats: dict[str, dict[str, Any]] = {}
    pool_key_cache: dict[str, str | None] = {}
    dropped_long = 0
    for slot in chosen:
        slot_country_name = slot["country"]
        if slot_country_name not in pool_key_cache:
            pool_key_cache[slot_country_name] = _content_pool_key(slot_country_name, pool)
        pool_key = pool_key_cache[slot_country_name]
        country_pool = pool.get(pool_key) if pool_key is not None else None
        from_fallback = not country_pool
        use_pool = global_pool if from_fallback else country_pool
        if not use_pool:
            raise AdapterError(
                f"skeleton graft: no content pool for slot country {slot_country_name!r} and the global pool is empty"
            )
        rng = random.Random(stable_seed(config["workload_id"], slot["slot_session_id"]))
        content = use_pool[rng.randrange(len(use_pool))]
        max_start = horizon - content["span"] - 1e-6
        if slot["start"] > max_start:
            dropped_long += 1
            continue
        graft_key = stable_hex(config["dataset"]["revision"], slot["slot_session_id"], content["source_key"], length=32)
        new_session_id = f"wc_sess_{stable_hex(config['workload_id'], graft_key)}"
        first = float(content["pairs"][0]["arrival_time_s"])
        new_pairs: list[dict[str, Any]] = []
        for pair in content["pairs"]:
            new_pairs.append(
                {
                    "source_key": graft_key,
                    "pair_index": pair["pair_index"],
                    "arrival_time_s": round(slot["start"] + (float(pair["arrival_time_s"]) - first), 6),
                    "region": slot["region"],
                    "history": pair["history"],
                    "user": pair["user"],
                    "assistant": pair["assistant"],
                }
            )
        extracted.append((new_session_id, graft_key, content["raw"], new_pairs))
        country_stats = stats.setdefault(
            slot_country_name,
            {"slots": 0, "pool_sessions": 0, "fallback_slots": 0, "distinct_content": set()},
        )
        country_stats["slots"] += 1
        country_stats["fallback_slots"] += int(from_fallback)
        country_stats["pool_sessions"] = len(global_pool) if from_fallback else len(country_pool)
        country_stats["distinct_content"].add(content["source_key"])
    if dropped_long:
        warnings.append(
            f"skeleton graft: dropped {dropped_long} session(s) whose slot start plus content span exceeds the synthetic day"
        )
    country_report = {
        country: {
            "slots": value["slots"],
            "pool_sessions": value["pool_sessions"],
            "fallback_slots": value["fallback_slots"],
            "distinct_content_sessions": len(value["distinct_content"]),
            "reuse_multiplier": round(value["slots"] / max(1, len(value["distinct_content"])), 4),
        }
        for country, value in sorted(stats.items())
    }
    info = {
        "slot_count_total": len(slots),
        "slot_count_used": len(chosen),
        "grafted_sessions": len(extracted),
        "dropped_sessions": dropped_long,
        "countries": country_report,
    }
    return extracted, info, warnings


def apply_month_compression_shifts(
    requests: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    session_starts: dict[str, float],
) -> None:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in requests:
        by_session[row["session_id"]].append(row)
    for session in sessions:
        rows = by_session[session["session_id"]]
        first = min(rows, key=lambda row: row["turn_index"])
        shift = session_starts[session["session_id"]] - float(first["arrival_time_s"])
        for row in rows:
            row["arrival_time_s"] = round(float(row["arrival_time_s"]) + shift, 6)
        session["session_start_time_s"] = round(float(session["session_start_time_s"]) + shift, 6)
    requests.sort(key=lambda row: (row["arrival_time_s"], row["request_id"]))
    sessions.sort(key=lambda row: (row["session_start_time_s"], row["session_id"]))


def prepare_config(config_path: Path) -> dict[str, Any]:
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
    return config


def _hash_parquet_inputs(
    input_paths: list[Path],
    expected_shas: dict[str, str],
    *,
    label: str,
) -> tuple[str, list[tuple[str, str]]]:
    input_paths = list(input_paths)
    if not input_paths or any(path.suffix.lower() != ".parquet" for path in input_paths):
        raise AdapterError("the experiment adapter accepts official WildChat .parquet shard(s)")
    per_file_shas: list[tuple[str, str]] = []
    for input_path in input_paths:
        input_sha256 = sha256_bytes(input_path.read_bytes())
        expected_sha = expected_shas.get(input_path.name)
        if expected_shas and expected_sha is None:
            raise AdapterError(f"no expected_source_sha256 configured for {label} {input_path.name}")
        if expected_sha and input_sha256 != expected_sha:
            raise AdapterError(f"source SHA-256 mismatch for {input_path}; expected {expected_sha}, got {input_sha256}")
        per_file_shas.append((input_path.name, input_sha256))
    combined = sha256_bytes(
        ("\n".join(f"{name}:{digest}" for name, digest in per_file_shas) + "\n").encode("utf-8")
    )
    return combined, per_file_shas


def verify_and_hash_inputs(config: dict[str, Any], input_paths: list[Path]) -> tuple[str, list[tuple[str, str]]]:
    return _hash_parquet_inputs(input_paths, _expected_source_shas(config["dataset"]), label="input")


def verify_and_hash_profile_inputs(config: dict[str, Any], profile_paths: list[Path]) -> tuple[str, list[tuple[str, str]]]:
    """Verify diurnal-profile shards against diurnal_profile.expected_source_sha256."""
    profile_block = config.get("diurnal_profile")
    if not isinstance(profile_block, dict):
        raise AdapterError("--profile-input requires a diurnal_profile block in the config")
    expected_shas = {
        Path(str(name)).name: str(digest)
        for name, digest in profile_block.get("expected_source_sha256", {}).items()
    }
    return _hash_parquet_inputs(profile_paths, expected_shas, label="profile input")


def compute_bundle_data(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    input_sha256: str,
    *,
    month_mode: bool,
    scale: str,
    line_numbers: list[int] | None,
    session_starts: dict[str, float],
    compression_report: list[dict[str, Any]],
    compression_warnings: list[str],
    out_dir_name: str,
    token_counts: list[tuple[int, int, int, int]] | None = None,
    extracted: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] | None = None,
    light_extract: bool = False,
    diurnal_profile_info: dict[str, Any] | None = None,
    graft_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble all in-memory bundle artifacts (nothing is written to disk)."""
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
    if token_counts is None:
        requests, sessions, regions, gateways = build_requests(config, raw_rows, input_sha256, line_numbers=line_numbers)
    else:
        if extracted is None:
            extracted = extract_session_pairs(
                config, raw_rows, input_sha256, line_numbers=line_numbers, light=light_extract
            )
        requests, sessions, regions, gateways = assemble_requests(config, extracted, token_counts)
    if graft_info is not None:
        # Grafted arrivals are already final (slot start + content offsets);
        # only the representative-day epoch/horizon metadata is overridden.
        source_epoch = time_window["epoch_utc"]
        source_end = str(time_window.get("window_end_utc", ""))
        time_window["epoch_utc"] = str(time_window.get("output_epoch_utc", time_window["epoch_utc"]))
        time_window["time_horizon_s"] = float(time_window.get("output_time_horizon_s", 86400.0))
        time_window["compressed_from"] = f"{source_epoch}/{source_end}"
    elif month_mode:
        apply_month_compression_shifts(requests, sessions, session_starts)
        source_epoch = time_window["epoch_utc"]
        source_end = str(time_window.get("window_end_utc", ""))
        time_window["epoch_utc"] = str(time_window.get("output_epoch_utc", time_window["epoch_utc"]))
        time_window["time_horizon_s"] = float(time_window.get("output_time_horizon_s", 86400.0))
        time_window["compressed_from"] = f"{source_epoch}/{source_end}"
    time = config["dataset"]["time_window"]
    description = build_description(config, regions, gateways, input_sha256, out_dir_name)
    if month_mode and graft_info is None:
        description["provenance"]["notes"].append(
            f"month-compressed synthetic day: all WildChat conversations in {time_window['compressed_from']} "
            f"were pooled into {int(float(time_window['time_bin_s']))}s time-of-day bins; load scale {scale} "
            f"(fraction {LOAD_SCALE_FRACTIONS[scale]}) was sampled per bin with stable-seeded RNGs."
        )
    if graft_info is not None:
        description["provenance"]["skeleton_graft"] = graft_info
        description["provenance"]["notes"].append(
            f"skeleton graft: {graft_info['grafted_sessions']} of {graft_info['slot_count_used']} skeleton slots "
            f"from {graft_info['skeleton_bundle']} (regions and start times preserved verbatim) were grafted with "
            f"full 2025 content conversations drawn per slot country with replacement; content shards "
            f"{time_window['compressed_from']}; load scale {scale} (fraction {LOAD_SCALE_FRACTIONS[scale]}); "
            f"{graft_info['dropped_sessions']} grafted sessions were dropped for crossing the synthetic day."
        )
    if diurnal_profile_info is not None:
        description["provenance"]["diurnal_profile"] = diurnal_profile_info
        if diurnal_profile_info["grouping"] == COUNTRY_LOCAL_GROUPING:
            description["provenance"]["notes"].append(
                f"diurnal profile warp: session start times were dealt per country against the "
                f"{diurnal_profile_info.get('profile_window', 'profile-month')} local-time envelopes "
                f"(legal IANA timezones at reference dates {diurnal_profile_info.get('profile_reference_date')}/"
                f"{diurnal_profile_info.get('content_reference_date')}; content countries: "
                f"{', '.join(diurnal_profile_info.get('content_countries', []))}); conversation content comes "
                f"from the input shards in {time_window['compressed_from'] if month_mode else time_window['epoch_utc']}."
            )
        elif diurnal_profile_info["grouping"] == MACRO_REGION_LOCAL_GROUPING:
            description["provenance"]["notes"].append(
                f"diurnal profile warp: session start times were dealt per macro region (US/CN/EU/OTHER) "
                f"against the {diurnal_profile_info.get('profile_window', 'profile-month')} local-time envelopes "
                f"(legal IANA timezones at reference dates {diurnal_profile_info.get('profile_reference_date')}/"
                f"{diurnal_profile_info.get('content_reference_date')}; no country filtering); conversation "
                f"content comes from the input shards in {time_window['compressed_from'] if month_mode else time_window['epoch_utc']}."
            )
        else:
            description["provenance"]["notes"].append(
                f"diurnal profile warp: session time-of-day bins were dealt per macro region (US/CN/EU/OTHER) "
                f"to match the {diurnal_profile_info.get('profile_window', 'profile-month')} profile shards "
                f"(combined sha256 {diurnal_profile_info['profile_sha256']}); conversation content comes from "
                f"the input shards in {time_window['compressed_from'] if month_mode else time_window['epoch_utc']}."
            )
    description_hash = _description_hash(description)
    timebins = build_timebins(requests, float(time["time_horizon_s"]), float(time["time_bin_s"]))
    summary = build_summary(requests, sessions, regions, gateways, timebins, description_hash, config)
    if graft_info is not None:
        summary["provenance"]["skeleton_graft"] = {
            "time_mode": SKELETON_GRAFT_TIME_MODE,
            "skeleton_bundle": graft_info["skeleton_bundle"],
            "skeleton_bundle_sha256": graft_info.get("skeleton_bundle_sha256"),
            "content_window": time_window["compressed_from"],
            "representative_epoch_utc": time_window["epoch_utc"],
            "load_scale": scale,
            "load_fraction": LOAD_SCALE_FRACTIONS[scale],
            "slot_count_total": graft_info["slot_count_total"],
            "slot_count_used": graft_info["slot_count_used"],
            "grafted_sessions": graft_info["grafted_sessions"],
            "dropped_sessions": graft_info["dropped_sessions"],
            "countries": graft_info["countries"],
        }
    elif month_mode:
        summary["provenance"]["month_compression"] = {
            "time_mode": MONTH_COMPRESSED_DAY_MODE,
            "source_window": time_window["compressed_from"],
            "representative_epoch_utc": time_window["epoch_utc"],
            "load_scale": scale,
            "load_fraction": LOAD_SCALE_FRACTIONS[scale],
            "bin_count": len(compression_report),
            "bins": compression_report,
        }
        if diurnal_profile_info is not None:
            summary_profile = {
                "grouping": diurnal_profile_info["grouping"],
                "profile_window": diurnal_profile_info.get("profile_window"),
                "profile_sha256": diurnal_profile_info["profile_sha256"],
            }
            for key in ("content_countries", "rest_fallback_countries", "rest_fallback_groups", "profile_reference_date", "content_reference_date"):
                if key in diurnal_profile_info:
                    summary_profile[key] = diurnal_profile_info[key]
            summary["provenance"]["month_compression"]["diurnal_profile"] = summary_profile
    return {
        "requests": requests,
        "sessions": sessions,
        "regions": regions,
        "gateways": gateways,
        "description": description,
        "description_hash": description_hash,
        "timebins": timebins,
        "summary": summary,
        "compression_warnings": list(compression_warnings),
        "month_mode": month_mode,
        "scale": scale,
        "graft": graft_info is not None,
    }


def write_bundle(config: dict[str, Any], data: dict[str, Any], out_dir: Path, overwrite: bool, *, run_validation: bool = True) -> None:
    """Write the seven canonical bundle files plus sidecars, then validate."""
    requests = data["requests"]
    sessions = data["sessions"]
    regions = data["regions"]
    gateways = data["gateways"]
    description = data["description"]
    description_hash = data["description_hash"]
    timebins = data["timebins"]
    summary = data["summary"]
    month_mode = data["month_mode"]
    scale = data["scale"]
    is_graft = bool(data.get("graft"))
    time = config["dataset"]["time_window"]

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    canonical_requests_path = out_dir.parent / "canonical_requests.jsonl"
    canonical_sessions_path = out_dir.parent / "canonical_sessions.jsonl"
    write_jsonl(canonical_requests_path, requests)
    write_jsonl(canonical_sessions_path, sessions)

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
        "tokenizer": {
            "backend": config["tokenization"]["backend"],
            "target_model": config["tokenization"]["target_model"],
            "tokenizer_id": config["tokenization"].get("tokenizer_id"),
            "revision": config["tokenization"].get("tokenizer_revision"),
            "local_path": config["tokenization"].get("local_path"),
            "context_window_tokens": config["tokenization"].get("context_window_tokens"),
        },
        "warnings": list(data["compression_warnings"]),
        "provenance": description["provenance"],
    }
    if is_graft:
        manifest["time"]["time_mode"] = SKELETON_GRAFT_TIME_MODE
        manifest["time"]["compressed_from"] = time["compressed_from"]
        manifest["load_scale"] = scale
    elif month_mode:
        manifest["time"]["time_mode"] = MONTH_COMPRESSED_DAY_MODE
        manifest["time"]["compressed_from"] = time["compressed_from"]
        manifest["load_scale"] = scale
    write_json(out_dir / "workload_manifest.json", manifest)
    write_json(out_dir.parent / "generated_description.json", description)
    if run_validation:
        validate(out_dir, out_dir.parent / "generated_description.json")
    print(f"built {len(requests)} requests, {len(sessions)} sessions, {len(regions)} regions in {out_dir}")


def build(
    config_path: Path,
    input_paths: list[Path],
    out_dir: Path,
    overwrite: bool,
    load_scale: str | None = None,
    scale_flags: dict[str, bool] | None = None,
    profile_paths: list[Path] | None = None,
    skeleton_bundle: Path | None = None,
) -> None:
    config = prepare_config(config_path)
    time_window = config["dataset"]["time_window"]
    month_mode = time_window.get("time_mode") == MONTH_COMPRESSED_DAY_MODE
    if skeleton_bundle is not None and profile_paths:
        raise AdapterError("--skeleton-bundle conflicts with --profile-input; choose one graft/warp mode")
    if skeleton_bundle is not None and not month_mode:
        raise AdapterError(
            "--skeleton-bundle requires dataset.time_window.time_mode='month_compressed_day'"
        )
    if profile_paths and not month_mode:
        raise AdapterError(
            "--profile-input requires dataset.time_window.time_mode='month_compressed_day'"
        )
    input_sha256, input_file_shas = verify_and_hash_inputs(config, input_paths)
    raw_rows: list[dict[str, Any]] = []
    for input_path in input_paths:
        raw_rows.extend(read_source_rows(input_path))
    max_records = int(config["dataset"].get("max_source_records", len(raw_rows)))
    if max_records <= 0:
        raise AdapterError("dataset.max_source_records must be positive")
    raw_rows = raw_rows[:max_records]
    scale = resolve_load_scale(config, load_scale, scale_flags or {}, month_mode=month_mode)
    compression_report: list[dict[str, Any]] = []
    compression_warnings: list[str] = []
    line_numbers: list[int] | None = None
    session_starts: dict[str, float] = {}
    diurnal_profile_info: dict[str, Any] | None = None
    graft_info: dict[str, Any] | None = None
    graft_extracted: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] | None = None
    graft_token_counts: list[tuple[int, int, int, int]] | None = None
    if month_mode:
        config["workload_id"] = f"{config['workload_id']}_{scale}"
        if skeleton_bundle is not None:
            skeleton_block = config.get("skeleton_bundle", {}) if isinstance(config.get("skeleton_bundle"), dict) else {}
            slots = load_skeleton_slots(skeleton_bundle, skeleton_block)
            pool, global_pool = collect_graft_content_pool(config, raw_rows, input_sha256)
            graft_extracted, graft_stats, graft_warnings = plan_skeleton_graft(config, slots, pool, global_pool, scale)
            compression_warnings = list(graft_warnings)
            flat_pairs = [pair for _, _, _, pairs in graft_extracted for pair in pairs]
            graft_token_counts = _batch_token_counts(
                TokenCounter(config["tokenization"]), flat_pairs, int(config["tokenization"]["context_window_tokens"])
            )
            graft_info = {
                "time_mode": SKELETON_GRAFT_TIME_MODE,
                "skeleton_bundle": str(skeleton_bundle),
                "skeleton_bundle_sha256": _skeleton_bundle_sha256(skeleton_bundle),
                "content_files": [{"path": name, "sha256": digest} for name, digest in input_file_shas],
                "content_sha256": input_sha256,
                **graft_stats,
            }
        elif profile_paths:
            profile_sha256, profile_file_shas = verify_and_hash_profile_inputs(config, profile_paths)
            scan_config = _diurnal_profile_scan_config(config)
            profile_block = config["diurnal_profile"]
            grouping = str(profile_block.get("grouping", MACRO_REGION_GROUPING))
            profile_counts: dict[str, Counter] | None = None
            diurnal_profile_info = {
                "grouping": grouping,
                "profile_sha256": profile_sha256,
                "profile_files": [
                    {"path": name, "sha256": digest} for name, digest in profile_file_shas
                ],
                "profile_window": f"{profile_block.get('epoch_utc', '')}/{profile_block.get('window_end_utc', '')}",
            }
            if grouping == COUNTRY_LOCAL_GROUPING:
                allowlist_keys = _country_allowlist_keys(_country_local_settings(config))
                for profile_path in profile_paths:
                    # One shard at a time so profile conversation content is
                    # freed right after its (light) count scan.
                    profile_counts = collect_local_profile_counts(
                        scan_config,
                        read_source_rows(profile_path),
                        allowlist_keys,
                        str(profile_block["profile_reference_date"]),
                        profile_counts,
                    )
                assert profile_counts is not None
                line_to_session, session_starts, compression_report, compression_warnings, warp_info = plan_country_local_warp(
                    config, raw_rows, input_sha256, profile_counts, scale
                )
                diurnal_profile_info.update({
                    "content_countries": [str(country) for country in profile_block["content_countries"]],
                    "min_country_profile_sessions": int(profile_block.get("min_country_profile_sessions", 1000)),
                    "profile_reference_date": str(profile_block["profile_reference_date"]),
                    "content_reference_date": str(profile_block["content_reference_date"]),
                    "rest_fallback_countries": warp_info["fallback_countries"],
                    "envelope_sessions": warp_info["envelope_sessions"],
                })
            elif grouping == MACRO_REGION_LOCAL_GROUPING:
                _local_envelope_settings(config, grouping=MACRO_REGION_LOCAL_GROUPING)
                for profile_path in profile_paths:
                    # One shard at a time so profile conversation content is
                    # freed right after its (light) count scan.
                    profile_counts = collect_macro_local_profile_counts(
                        scan_config,
                        read_source_rows(profile_path),
                        str(profile_block["profile_reference_date"]),
                        profile_counts,
                    )
                assert profile_counts is not None
                line_to_session, session_starts, compression_report, compression_warnings, warp_info = plan_macro_local_warp(
                    config, raw_rows, input_sha256, profile_counts, scale
                )
                diurnal_profile_info.update({
                    "profile_reference_date": str(profile_block["profile_reference_date"]),
                    "content_reference_date": str(profile_block["content_reference_date"]),
                    "rest_fallback_groups": warp_info["fallback_groups"],
                    "envelope_sessions": warp_info["envelope_sessions"],
                })
            else:
                for profile_path in profile_paths:
                    # One shard at a time so profile conversation content is
                    # freed right after its (light) count scan.
                    profile_counts = collect_profile_counts(scan_config, read_source_rows(profile_path), profile_counts)
                assert profile_counts is not None
                line_to_session, session_starts, compression_report, compression_warnings = plan_diurnal_profile_warp(
                    config, raw_rows, input_sha256, profile_counts, scale
                )
        else:
            line_to_session, session_starts, compression_report, compression_warnings = plan_month_compression(
                config, raw_rows, input_sha256, scale
            )
        if graft_info is None:
            line_numbers = sorted(int(line) for line in line_to_session)
            raw_rows = [raw_rows[line - 1] for line in line_numbers]
    data = compute_bundle_data(
        config,
        raw_rows,
        input_sha256,
        month_mode=month_mode,
        scale=scale,
        line_numbers=line_numbers,
        session_starts=session_starts,
        compression_report=compression_report,
        compression_warnings=compression_warnings,
        out_dir_name=out_dir.name,
        diurnal_profile_info=diurnal_profile_info,
        token_counts=graft_token_counts,
        extracted=graft_extracted,
        graft_info=graft_info,
    )
    write_bundle(config, data, out_dir, overwrite)


def _skeleton_bundle_sha256(skeleton_dir: Path) -> str | None:
    """bundle_sha256 of the skeleton bundle's manifest, if readable."""
    try:
        manifest = read_json(skeleton_dir / "workload_manifest.json")
    except AdapterError:
        return None
    digest = manifest.get("bundle_sha256")
    return str(digest) if isinstance(digest, str) else None


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
    old_profile_info = manifest.get("provenance", {}).get("diurnal_profile")
    if old_profile_info is not None:
        description["provenance"]["diurnal_profile"] = old_profile_info
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
    build_parser = subparsers.add_parser("build", help="map official WildChat Parquet shard(s) to a bundle")
    build_parser.add_argument("--config", type=Path, required=True)
    build_parser.add_argument("--input", type=Path, required=True, nargs="+", help="verified local official .parquet shard(s)")
    build_parser.add_argument("--out", type=Path, required=True)
    build_parser.add_argument("--overwrite", action="store_true")
    build_parser.add_argument("--profile-input", type=Path, default=None, nargs="+",
                              help="diurnal-profile .parquet shard(s); only valid with "
                                   "dataset.time_window.time_mode='month_compressed_day' and "
                                   "a diurnal_profile block in the config")
    build_parser.add_argument("--skeleton-bundle", type=Path, default=None,
                              help="existing bundle directory whose session slots (region + start time) "
                                   "are grafted with content from --input shard(s); mutually exclusive "
                                   "with --profile-input")
    build_parser.add_argument("--load-scale", choices=sorted(LOAD_SCALE_FRACTIONS), default=None,
                              help="load intensity for month_compressed_day mode (default: config load.default_scale or 10x)")
    for scale_name in sorted(LOAD_SCALE_FRACTIONS):
        build_parser.add_argument(f"--{scale_name}", dest=f"load_scale_{scale_name}", action="store_true",
                                  help=f"shortcut for --load-scale {scale_name}")
    refresh_parser = subparsers.add_parser("refresh-metadata", help="repair sidecar metadata for an existing bundle")
    refresh_parser.add_argument("--config", type=Path, required=True)
    refresh_parser.add_argument("--bundle", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate", help="validate a generated bundle")
    validate_parser.add_argument("--bundle", type=Path, required=True)
    validate_parser.add_argument("--description", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "build":
            flags = {name: getattr(args, f"load_scale_{name}") for name in LOAD_SCALE_FRACTIONS}
            build(args.config, args.input, args.out, args.overwrite, load_scale=args.load_scale, scale_flags=flags, profile_paths=args.profile_input, skeleton_bundle=args.skeleton_bundle)
        elif args.command == "refresh-metadata":
            refresh_metadata(args.config, args.bundle)
        else:
            validate(args.bundle, args.description)
    except AdapterError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
