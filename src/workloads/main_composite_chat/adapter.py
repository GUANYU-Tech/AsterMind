#!/usr/bin/env python3
"""Build a second-resolution, multi-turn OrbitServe bundle from ShareChat CSVs
with Profound Figure 2 per-region diurnal profiles.

ShareChat stores one message per CSV row.  This adapter groups rows by
``platform + url`` and emits one OrbitServe session per original shared
conversation.  A session is assigned a country (from Microsoft-derived top-40
weights) and a second-level local-time start (drawn from Profound Figure 2
regional digitized diurnal curves); its later turns retain the observed
timestamp gaps.

Four profile patterns are supported via ``--profile-pattern``:
  work_weekday | work_weekend | nonwork_weekday | nonwork_weekend

Only the ChatGPT and Grok releases contain per-message timestamps.  The
default ``--require-turn-timestamps`` therefore rejects Gemini, Claude and
Perplexity rows instead of inventing a real-looking inter-turn gap.
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
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from orbitserve.workload import (
    BUNDLE_FILES, BUNDLE_SCHEMA, CANONICALIZATION, bundle_hash,
    canonical_json_bytes, load_workload_bundle,
)


_HERE = Path(__file__).resolve().parent
DEFAULT_COUNTRIES = _HERE / "countries.csv"
PROFILE_PATTERNS = {
    "work_weekday":      (_HERE / "profiles" / "work_weekday.csv",      "weekday"),
    "work_weekend":      (_HERE / "profiles" / "work_weekend.csv",      "weekend"),
    "nonwork_weekday":   (_HERE / "profiles" / "nonwork_weekday.csv",   "weekday"),
    "nonwork_weekend":   (_HERE / "profiles" / "nonwork_weekend.csv",   "weekend"),
}
DEFAULT_PATTERN = "work_weekday"
DEFAULT_GATEWAYS = _HERE / "gateways.json"
DEFAULT_LLAMA_TOKENIZER = _HERE / "tokenizer_llama_31"
DEFAULT_LLAMA_ID = "meta-llama/Llama-3.1-70B-Instruct"

GENERATOR = {"name": "profound_sharechat_adapter", "version": "1.0.0", "prng": "python-random-seeded-v1"}
HORIZON_S = 86_400.0

# A representative civil timezone for every top-40 country.  Multi-zone
# countries intentionally use their population/economic centre; this is a
# scheduling assumption, not a ShareChat field.
TIMEZONES = {
    "China": "Asia/Shanghai", "India": "Asia/Kolkata", "United States": "America/New_York",
    "Brazil": "America/Sao_Paulo", "Indonesia": "Asia/Jakarta", "France": "Europe/Paris",
    "United Kingdom": "Europe/London", "Russia": "Europe/Moscow", "Vietnam": "Asia/Ho_Chi_Minh",
    "Mexico": "America/Mexico_City", "Pakistan": "Asia/Karachi", "Japan": "Asia/Tokyo",
    "Germany": "Europe/Berlin", "Philippines": "Asia/Manila", "Spain": "Europe/Madrid",
    "Nigeria": "Africa/Lagos", "South Korea": "Asia/Seoul", "Italy": "Europe/Rome",
    "Egypt": "Africa/Cairo", "Turkiye": "Europe/Istanbul", "South Africa": "Africa/Johannesburg",
    "Canada": "America/Toronto", "Colombia": "America/Bogota", "Bangladesh": "Asia/Dhaka",
    "Iran": "Asia/Tehran", "Saudi Arabia": "Asia/Riyadh", "Poland": "Europe/Warsaw",
    "Australia": "Australia/Sydney", "Argentina": "America/Argentina/Buenos_Aires",
    "United Arab Emirates": "Asia/Dubai", "Thailand": "Asia/Bangkok", "Ethiopia": "Africa/Addis_Ababa",
    "Malaysia": "Asia/Kuala_Lumpur", "Taiwan": "Asia/Taipei", "Congo (DRC)": "Africa/Kinshasa",
    "Netherlands": "Europe/Amsterdam", "Algeria": "Africa/Algiers", "Peru": "America/Lima",
    "Myanmar": "Asia/Yangon", "Iraq": "Asia/Baghdad",
}

# Representative country locations make an independently usable geography
# bundle.  More precise regional locations can be substituted later.
COORDS = {
    "China": (35.86, 104.20), "India": (20.59, 78.96), "United States": (39.83, -98.58),
    "Brazil": (-14.24, -51.93), "Indonesia": (-0.79, 113.92), "France": (46.23, 2.21),
    "United Kingdom": (55.38, -3.44), "Russia": (55.75, 37.62), "Vietnam": (14.06, 108.28),
    "Mexico": (23.63, -102.55), "Pakistan": (30.38, 69.35), "Japan": (36.20, 138.25),
    "Germany": (51.17, 10.45), "Philippines": (12.88, 121.77), "Spain": (40.46, -3.75),
    "Nigeria": (9.08, 8.68), "South Korea": (35.91, 127.77), "Italy": (41.87, 12.57),
    "Egypt": (26.82, 30.80), "Turkiye": (38.96, 35.24), "South Africa": (-30.56, 22.94),
    "Canada": (56.13, -106.35), "Colombia": (4.57, -74.30), "Bangladesh": (23.68, 90.36),
    "Iran": (32.43, 53.69), "Saudi Arabia": (23.89, 45.08), "Poland": (51.92, 19.15),
    "Australia": (-25.27, 133.78), "Argentina": (-38.42, -63.62), "United Arab Emirates": (23.42, 53.85),
    "Thailand": (15.87, 100.99), "Ethiopia": (9.15, 40.49), "Malaysia": (4.21, 101.98),
    "Taiwan": (23.70, 120.96), "Congo (DRC)": (-4.04, 21.76), "Netherlands": (52.13, 5.29),
    "Algeria": (28.03, 1.66), "Peru": (-9.19, -75.02), "Myanmar": (21.91, 95.96), "Iraq": (33.22, 43.68),
}


class AdapterError(ValueError):
    pass


def stable_id(prefix: str, *parts: object) -> str:
    value = "\0".join(map(str, parts)).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(value).hexdigest()[:24]}"


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def timebin_csv_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    """Local serialization for this adapter's seven-file bundle output."""
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=(
        "time_bin_start_s", "time_bin_end_s", "region_id", "request_count",
        "prompt_tokens_total", "new_user_tokens_total", "output_tokens_total",
        "interactive_request_count", "burst_request_count",
    ), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def file_descriptor(path: Path, media_type: str, record_count: int) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": path.name, "media_type": media_type, "sha256": hashlib.sha256(payload).hexdigest(), "record_count": int(record_count)}


def parse_message_time(value: object) -> float | None:
    """Support ShareChat's ``ts:unix`` and ISO-8601 timestamp variants."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    if text.startswith("ts:"):
        try:
            return float(text[3:])
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


def allow_large_csv_fields() -> None:
    """Lift the stdlib CSV 128 KiB field cap for long ShareChat messages."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role(value: object) -> str | None:
    text = str(value or "").lower()
    if text in {"user", "human"}:
        return "user"
    if text in {"assistant", "llm", "model", "chatbot"}:
        return "assistant"
    return None


def read_sharechat(path: Path, *, max_sessions: int, require_timestamps: bool) -> list[dict[str, Any]]:
    """Stream valid multi-turn conversations from ShareChat's grouped CSV.

    Official ShareChat CSVs are ordered by conversation URL.  Streaming means
    a 2.3 GB ChatGPT CSV does not need to fit in memory; construction stops as
    soon as ``max_sessions`` valid conversations have been collected.
    """
    allow_large_csv_fields()
    conversations: list[dict[str, Any]] = []

    def finalize(key: tuple[str, str] | None, messages: list[dict[str, str]]) -> None:
        if key is None or len(messages) < 4:
            return
        platform, url = key
        messages.sort(key=lambda item: int(item["index"]) if item["index"].isdigit() else 0)
        pairs = []
        for index in range(len(messages) - 1):
            user, assistant = messages[index], messages[index + 1]
            if user["role"] != "user" or assistant["role"] != "assistant":
                continue
            user_time, assistant_time = parse_message_time(user["time"]), parse_message_time(assistant["time"])
            arrival = user_time if user_time is not None else assistant_time
            pairs.append({"user": user, "assistant": assistant, "arrival": arrival})
        if len(pairs) < 2 or (require_timestamps and any(pair["arrival"] is None for pair in pairs)):
            return
        arrivals = [float(pair["arrival"]) for pair in pairs if pair["arrival"] is not None]
        if require_timestamps and any(b <= a for a, b in zip(arrivals, arrivals[1:])):
            return
        conversations.append({"platform": platform, "url": url, "pairs": pairs, "model": pairs[0]["assistant"]["model"]})

    current_key: tuple[str, str] | None = None
    current_messages: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"platform", "url", "message_index", "role", "plain_text"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise AdapterError(f"{path} lacks ShareChat core columns: {sorted(required)}")
        for row in reader:
            key = (str(row.get("platform") or "").lower(), str(row.get("url") or ""))
            if current_key is not None and key != current_key:
                finalize(current_key, current_messages)
                if len(conversations) >= max_sessions:
                    break
                current_messages = []
            current_key = key
            message_role = role(row.get("role"))
            content = str(row.get("plain_text") or "").strip()
            if not key[0] or not key[1] or message_role is None or not content:
                continue
            current_messages.append({
                "role": message_role,
                "content": content,
                "index": str(row.get("message_index") or "0"),
                "time": str(row.get("message_create_time") or ""),
                "model": str(row.get("model") or ""),
            })
        else:
            finalize(current_key, current_messages)
    if not conversations:
        requirement = "with real per-message timestamps" if require_timestamps else "with at least two user turns"
        raise AdapterError(f"no ShareChat conversations {requirement} in {path}")
    return conversations


def simple_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))


class TokenCounter:
    """Target-model tokenization, with an explicit smoke-test fallback."""

    def __init__(self, backend: str, local_dir: Path, tokenizer_id: str) -> None:
        self.backend = backend
        self.tokenizer_id = tokenizer_id
        self.tokenizer: Any | None = None
        if backend == "huggingface_auto":
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise AdapterError("huggingface_auto requires transformers") from exc
            if not local_dir.is_dir():
                raise AdapterError(
                    f"Llama tokenizer is absent at {local_dir}; download it first or use --tokenizer-backend simple_unicode_v1 for smoke tests"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, trust_remote_code=False)
        elif backend != "simple_unicode_v1":
            raise AdapterError(f"unsupported tokenizer backend {backend!r}")

    def counts(
        self, history: list[dict[str, str]], user: dict[str, str], assistant: dict[str, str], context_window: int
    ) -> tuple[int, int, int]:
        if self.backend == "simple_unicode_v1":
            new_user = min(simple_tokens(user["content"]) + 2, context_window)
            context = min(
                sum(simple_tokens(message["content"]) + 2 for message in history),
                max(0, context_window - new_user),
            )
            return context, new_user, simple_tokens(assistant["content"]) + 1
        assert self.tokenizer is not None
        history_ids = self.tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=False) if history else []
        prompt_ids = self.tokenizer.apply_chat_template(history + [user], tokenize=True, add_generation_prompt=True)
        context = len(history_ids)
        new_user = min(max(1, len(prompt_ids) - context), context_window)
        context = min(context, max(0, context_window - new_user))
        output = len(self.tokenizer.encode(assistant["content"], add_special_tokens=False)) + 1  # EOS
        return context, new_user, output


def load_countries(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            country = row["country_or_region"]
            if country not in TIMEZONES or country not in COORDS:
                raise AdapterError(f"missing timezone or coordinates for {country}")
            rows.append({"country": country, "weight": float(row["relative_request_index_us_100"])})
    return rows


def load_regional_hour_weights(path: Path, day_type: str = "workday") -> dict[str, list[float]]:
    """Read a Profound Figure 2 digitized regional profile CSV.

    Returns ``{country: [24 local-hour weights]}`` where each country gets its
    region's distinct diurnal shape (4 regions: Asia, Europe, North America,
    Latin America), manually digitized from Profound 2026 Figure 2.
    """
    raw: dict[str, dict[int, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["day_type"] != day_type:
                continue
            country = row["country"]
            hour = int(row["local_hour"])
            weight = float(row["hour_weight"])
            raw.setdefault(country, {})[hour] = weight

    profiles: dict[str, list[float]] = {}
    for country, hour_map in raw.items():
        weights = [hour_map.get(h, 0.0) for h in range(24)]
        total = sum(weights)
        if total <= 0:
            continue
        profiles[country] = [w / total for w in weights]
    if not profiles:
        raise AdapterError(f"no {day_type} profiles found in {path}")
    return profiles


def timezone_offset(timezone_name: str, epoch: datetime) -> float:
    return ZoneInfo(timezone_name).utcoffset(epoch).total_seconds() / 3600.0


def sample_start(rng: random.Random, country: str, weights: list[float], epoch: datetime) -> float:
    local_hour = rng.choices(range(24), weights=weights, k=1)[0]
    local_second = local_hour * 3600 + rng.random() * 3600
    offset = timezone_offset(TIMEZONES[country], epoch)
    return (local_second - offset * 3600) % HORIZON_S


def gateways() -> list[dict[str, Any]]:
    return json.loads(DEFAULT_GATEWAYS.read_text(encoding="utf-8"))


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = p2 - p1, math.radians(b_lon - a_lon)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(x))


def nearest_gateway(country: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    lat, lon = COORDS[country]
    return min(items, key=lambda item: haversine_km(lat, lon, float(item["lat_deg"]), float(item["lon_deg"])))


def slo(output: int) -> tuple[str, float, float, str]:
    if output <= 256:
        return "interactive_short", 3.0, 30.0, "interactive"
    if output <= 1024:
        return "interactive_long", 3.0, 120.0, "interactive"
    return "normal_long", 10.0, 300.0, "normal"


def build(args: argparse.Namespace) -> Path:
    epoch = datetime.fromisoformat(args.epoch.replace("Z", "+00:00")).astimezone(timezone.utc)
    source = Path(args.input)
    source_hash = sha256_file(source) if getattr(args, "compute_source_sha256", False) else None
    conversations = read_sharechat(source, max_sessions=args.max_sessions, require_timestamps=args.require_turn_timestamps)
    countries = load_countries(Path(args.countries))
    country_names, country_weights = [row["country"] for row in countries], [row["weight"] for row in countries]
    profile_path, profile_day_type = PROFILE_PATTERNS[args.profile_pattern]
    regional_profiles = load_regional_hour_weights(profile_path, profile_day_type)
    n_regional = len(regional_profiles)
    fallback_profile = [0.0] * 24
    for weights in regional_profiles.values():
        for h in range(24):
            fallback_profile[h] += weights[h]
    fallback_profile = [v / n_regional for v in fallback_profile]
    print(f"Loaded {n_regional} country-specific regional profiles from {profile_path}")
    counter = TokenCounter(args.tokenizer_backend, Path(args.tokenizer_local_dir), args.tokenizer_id)
    rng = random.Random(args.seed)
    gateway_items = gateways()

    requests: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    used_countries: Counter[str] = Counter()
    dropped = 0
    n_source = len(conversations)
    n_target = args.max_sessions
    oversample = n_target > n_source
    if oversample:
        print(f"Oversampling: {n_source} source conversations → {n_target} sessions "
              f"(each source reused ~{n_target / n_source:.1f}× with fresh country/time)")

    for ordinal in range(n_target):
        conversation = conversations[ordinal % n_source]
        country = rng.choices(country_names, weights=country_weights, k=1)[0]
        country_profile = regional_profiles.get(country, fallback_profile)
        start = sample_start(rng, country, country_profile, epoch)
        pairs = conversation["pairs"]
        source_first = float(pairs[0]["arrival"])
        relative = [float(pair["arrival"]) - source_first for pair in pairs]
        if relative[-1] >= HORIZON_S or start + relative[-1] >= HORIZON_S:
            dropped += 1
            continue
        session_id = stable_id("sc_sess", conversation["platform"], conversation["url"], ordinal, args.seed)
        gateway = nearest_gateway(country, gateway_items)
        lat, lon = COORDS[country]
        history: list[dict[str, str]] = []
        local_rows: list[dict[str, Any]] = []
        for turn, (pair, gap) in enumerate(zip(pairs, relative)):
            context, new_user, output = counter.counts(
                history, pair["user"], pair["assistant"], args.context_window_tokens
            )
            slo_class, ttft, e2e, priority = slo(output)
            request_id = stable_id("sc_req", session_id, turn)
            arrival = round(start + gap, 6)
            local_rows.append({
                "request_id": request_id, "arrival_time_s": arrival, "user_region_id": f"country_{country.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                "region_id": f"country_{country.lower().replace(' ', '_').replace('(', '').replace(')', '')}", "lat_deg": lat, "lon_deg": lon,
                "access_mode": "gateway_sgl", "ingress_id": gateway["gateway_id"], "ingress_lat_deg": float(gateway["lat_deg"]), "ingress_lon_deg": float(gateway["lon_deg"]), "gateway_id": gateway["gateway_id"],
                "task_type": "conversation", "model_family": "sharechat_target_model", "prompt_tokens": context + new_user, "new_user_tokens": new_user,
                "context_tokens_before_turn": context, "output_tokens": output, "input_bytes": 4 * (context + new_user), "output_bytes": 4 * output,
                "context_window_tokens": args.context_window_tokens, "slo_class": slo_class, "ttft_deadline_s": ttft, "e2e_deadline_s": e2e, "priority": priority,
                "session_id": session_id, "turn_index": turn, "session_turn_count": len(pairs), "inter_turn_gap_s": None if turn == 0 else round(gap - relative[turn - 1], 6),
                "session_ttl_s": args.session_ttl_s, "burst_id": None, "burst_multiplier": 1.0, "seed": int(hashlib.sha256(request_id.encode()).hexdigest()[:16], 16) & ((1 << 63) - 1),
            })
            history.extend([pair["user"], pair["assistant"]])
        sessions.append({"session_id": session_id, "user_region_id": local_rows[0]["user_region_id"], "session_start_time_s": local_rows[0]["arrival_time_s"], "session_turn_count": len(local_rows), "session_slo_class": local_rows[0]["slo_class"], "context_window_tokens": args.context_window_tokens, "session_ttl_s": args.session_ttl_s, "seed": int(hashlib.sha256(session_id.encode()).hexdigest()[:16], 16) & ((1 << 63) - 1)})
        requests.extend(local_rows)
        used_countries[country] += 1
    if not requests:
        raise AdapterError("all sessions spill beyond the one-day horizon")
    requests.sort(key=lambda row: (row["arrival_time_s"], row["request_id"]))
    sessions.sort(key=lambda row: (row["session_start_time_s"], row["session_id"]))

    region_rows = []
    for country in sorted(used_countries):
        region_id = f"country_{country.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        lat, lon = COORDS[country]
        region_rows.append({"region_id": region_id, "region_name": country, "lat_deg": lat, "lon_deg": lon, "timezone_utc_offset_h": timezone_offset(TIMEZONES[country], epoch), "demand_weight": used_countries[country] / len(sessions), "weight_source": "country_relative_request_index_top40", "market_class": None})
    served = defaultdict(set)
    for request in requests:
        served[request["gateway_id"]].add(request["region_id"])
    gateway_rows = [{"gateway_id": item["gateway_id"], "lat_deg": float(item["lat_deg"]), "lon_deg": float(item["lon_deg"]), "served_region_ids": sorted(served[item["gateway_id"]]), "gateway_bandwidth_bps": int(item["gateway_bandwidth_bps"]), "availability_profile_id": item.get("availability_profile_id"), "cost_usd_per_year": item.get("cost_usd_per_year")} for item in gateway_items]

    bins: dict[tuple[int, str], dict[str, Any]] = {}
    for request in requests:
        start = math.floor(request["arrival_time_s"] / args.time_bin_s) * args.time_bin_s
        key = (start, request["region_id"])
        row = bins.setdefault(key, {"time_bin_start_s": start, "time_bin_end_s": start + args.time_bin_s, "region_id": request["region_id"], "request_count": 0, "prompt_tokens_total": 0, "new_user_tokens_total": 0, "output_tokens_total": 0, "interactive_request_count": 0, "burst_request_count": 0})
        row["request_count"] += 1; row["prompt_tokens_total"] += request["prompt_tokens"]; row["new_user_tokens_total"] += request["new_user_tokens"]; row["output_tokens_total"] += request["output_tokens"]
        row["interactive_request_count"] += int(request["priority"] == "interactive")
    timebins = [bins[key] for key in sorted(bins)]
    summary = {"workload_id": args.workload_id, "request_count": len(requests), "session_count": len(sessions), "source_session_count": len(conversations), "dropped_horizon_sessions": dropped, "multi_turn_session_count": sum(row["session_turn_count"] > 1 for row in sessions), "time_resolution": "seconds", "tokenization": args.tokenizer_backend, "country_assignment": "synthetic draw from country_relative_request_index_top40", "diurnal_profile": f"Profound Figure 2 digitized per-region {args.profile_pattern}; seconds uniformly sampled within selected local hour"}

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise AdapterError(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    write_json(out / "regions.json", region_rows); write_json(out / "gateways.json", gateway_rows); write_jsonl(out / "requests.jsonl", requests); write_jsonl(out / "sessions.jsonl", sessions)
    (out / "timebin_demand.csv").write_bytes(timebin_csv_bytes(timebins)); write_json(out / "workload_summary.json", summary)
    descriptors = {
        key: file_descriptor(out / name, media_type, count)
        for key, ((name, media_type), count) in {
            "regions": (BUNDLE_FILES["regions"], len(region_rows)),
            "gateways": (BUNDLE_FILES["gateways"], len(gateway_rows)),
            "requests": (BUNDLE_FILES["requests"], len(requests)),
            "sessions": (BUNDLE_FILES["sessions"], len(sessions)),
            "timebin_demand": (BUNDLE_FILES["timebin_demand"], len(timebins)),
            "summary": (BUNDLE_FILES["summary"], 1),
        }.items()
    }
    warnings = [
        "Country assignment and local-time profile are explicit synthetic assumptions; ShareChat has no country field.",
        "Only source rows with real per-message timestamps were accepted.",
    ]
    if source_hash is None:
        warnings.append("Source CSV SHA-256 was skipped; pass --compute-source-sha256 for a reproducibility audit.")
    manifest = {"schema_version": BUNDLE_SCHEMA, "workload_id": args.workload_id, "source_mode": "explicit_requests", "seed": args.seed, "module_version": "1.0.0", "canonicalization": CANONICALIZATION, "generator": GENERATOR, "time": {"epoch_kind": "absolute_utc", "epoch": epoch.isoformat().replace("+00:00", "Z"), "time_horizon_s": HORIZON_S, "time_bin_s": args.time_bin_s, "interval_semantics": "half_open", "time_mode": "country_local_second_resolution"}, "source": {"mode": "explicit_requests", "inputs": [{"path": str(source), "sha256": source_hash, "record_count": len(conversations)}]}, "files": descriptors, "counts": {"request_count": len(requests), "session_count": len(sessions), "region_count": len(region_rows), "gateway_count": len(gateway_rows), "timebin_count": len(timebins)}, "tokenizer": {"backend": args.tokenizer_backend, "target_model": args.tokenizer_id, "local_path": str(args.tokenizer_local_dir), "context_window_tokens": args.context_window_tokens, "output_policy": "assistant_content_plus_eos"}, "warnings": warnings, "provenance": {"sources": [{"source_id": "tucnguyen/ShareChat", "url": "https://huggingface.co/datasets/tucnguyen/ShareChat", "sha256": source_hash}], "timing": "user message timestamp when available; otherwise assistant-completion proxy", "session_key": "platform + url", "relative_gaps": "observed message timestamp differences"}, "bundle_sha256": bundle_hash(args.workload_id, descriptors)}
    write_json(out / "workload_manifest.json", manifest)
    # Cross-module validation uses OrbitServe's public loader only.
    load_workload_bundle(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ShareChat ChatGPT or Grok CSV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workload-id", default="profound_sharechat")
    parser.add_argument("--countries", default=str(DEFAULT_COUNTRIES))
    parser.add_argument("--profile-pattern", choices=tuple(PROFILE_PATTERNS), default=DEFAULT_PATTERN,
                        help="work_weekday | work_weekend | nonwork_weekday | nonwork_weekend")
    parser.add_argument("--epoch", default="2026-08-03T00:00:00Z")
    parser.add_argument("--max-sessions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--time-bin-s", type=int, default=1)
    parser.add_argument("--context-window-tokens", type=int, default=131072)
    parser.add_argument("--session-ttl-s", type=float, default=600.0)
    parser.add_argument("--tokenizer-backend", choices=("huggingface_auto", "simple_unicode_v1"), default="huggingface_auto")
    parser.add_argument("--tokenizer-id", default=DEFAULT_LLAMA_ID)
    parser.add_argument("--tokenizer-local-dir", default=str(DEFAULT_LLAMA_TOKENIZER))
    parser.add_argument("--compute-source-sha256", action="store_true", help="stream the full source CSV once for an audit hash")
    parser.add_argument("--require-turn-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.time_bin_s <= 0 or 86400 % args.time_bin_s:
        parser.error("--time-bin-s must be a positive divisor of 86400")
    try:
        result = build(args)
    except AdapterError as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
