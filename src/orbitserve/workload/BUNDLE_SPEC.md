# OrbitServe Canonical Bundle Specification

A workload bundle is a directory containing exactly **6 files**.

## Module Responsibilities

### Adapter

Position: `src/workloads/<name>/adapter.py` (e.g. `main_composite_chat/adapter.py`)

Converts raw datasets into a canonical 6-file bundle on disk.

- Reads: raw CSV/Parquet, diurnal profile CSVs, country weight CSVs, gateway config JSON
- Writes: 6-file bundle (regions.json, gateways.json, session_requests.jsonl, timebin_demand.csv, workload_summary.json, workload_manifest.json)
- Combines multiple independent data sources (conversation content + temporal distribution + geographic distribution). Responsible for tokenization, SLO assignment, time sampling.

### contract.py

Position: `src/orbitserve/workload/contract.py`

Shared schema definitions and validation logic. Stdlib-only; used by both adapters and the engine.

- Defines `SESSION_REQUEST_KEYS` (24), `REGION_KEYS` (5), `GATEWAY_KEYS` (4), `TIMEBIN_FIELDS` (7), `BUNDLE_FILES` (6 files)
- Defines data types: `RegionSpec`, `GatewaySpec`, `WorkloadGenerationResult`, etc.
- Provides canonical serialisation (`canonical_json_bytes`, `canonical_timebin_csv_bytes`), hashing (`bundle_hash`, `sha256_bytes`)
- Provides validation functions (`validate_manifest_and_files`, `validate_exact_keys`, `validate_session_consistency`, `validate_reference_resolution`)
- Contains legacy aliases (`REQUEST_KEYS`, `SESSION_KEYS`, `REQUEST_FILES`, `coerce_request_row`, `coerce_session_row`, `descriptor`) to keep `generation.py` and `upsample.py` compatible without changes

### loader.py

Position: `src/orbitserve/workload/loader.py`

Loads a bundle from disk and returns an in-memory `WorkloadGenerationResult`.

- Reads: 6-file bundle directory
- Validates: manifest schema -> bundle_sha256 -> exact keys -> session consistency -> reference resolution
- Returns: `WorkloadGenerationResult` (containing `session_requests`, `regions`, `gateways`, `timebin_demand`, `summary`)

### upsample.py

Position: `src/orbitserve/workload/upsample.py`

Runtime lazy upsampling — replicates each session `factor` times from a loaded bundle, streaming one (session, [requests]) pair at a time without materialising the full expansion.

Not modified in this round. It accesses data through `WorkloadGenerationResult.requests` and `.sessions` compatibility properties (`.requests` -> `session_requests`, `.sessions` -> empty tuple), preserving the original behaviour: factor-x replication, bin-local jitter, and turn-structure preservation.

---

## File Index

| File | Format | Purpose |
|------|--------|---------|
| `workload_manifest.json` | JSON | Bundle identity + integrity |
| `regions.json` | JSON array | Geographic regions |
| `gateways.json` | JSON array | Ground gateways |
| `session_requests.jsonl` | JSONL (one per line) | All requests, session-aware |
| `timebin_demand.csv` | CSV | 60-second aggregate demand |
| `workload_summary.json` | JSON | Statistical summary |

---

## `workload_manifest.json`

| Field | Type | Description |
|------|------|-------------|
| `schema_version` | string | `orbitserve.workload` |
| `canonicalization` | string | `orbitserve-json-c14n` |
| `workload_id` | string | Unique bundle identifier |
| `seed` | int | Base random seed |
| `time_horizon_s` | float | Simulation horizon (86400.0) |
| `bundle_sha256` | string | Composite hash of all data files |

---

## `regions.json` — 5 fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `region_id` | string | Unique ID, e.g. `country_china` |
| 2 | `region_name` | string | Human-readable name, e.g. `China` |
| 3 | `lat_deg` | float | Representative latitude |
| 4 | `lon_deg` | float | Representative longitude |
| 5 | `demand_weight` | float | Fraction of total demand (sum = 1.0) |

---

## `gateways.json` — 4 fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `gateway_id` | string | Unique ID, e.g. `gw_beijing` |
| 2 | `lat_deg` | float | Latitude |
| 3 | `lon_deg` | float | Longitude |
| 4 | `served_region_ids` | [string] | Regions served by this gateway |

---

## `session_requests.jsonl` — 24 fields

### Identity

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `request_id` | string | Unique request ID |
| 24 | `seed` | int | Deterministic seed |

### Time

| # | Field | Type | Description |
|---|-------|------|-------------|
| 2 | `arrival_time_s` | float | Arrival time in UTC seconds (0-86400) |

### Geography & Access

| # | Field | Type | Description |
|---|-------|------|-------------|
| 3 | `user_region_id` | string | Region where the user is located |
| 4 | `access_mode` | string | `direct_user_sgl` or `gateway_sgl` |
| 5 | `ingress_id` | string | Entry point (region_id or gateway_id) |
| 6 | `gateway_id` | string\|null | Gateway ID (null for direct access) |

### Task

| # | Field | Type | Description |
|---|-------|------|-------------|
| 7 | `model_family` | string | Model series, e.g. `chatgpt`, `grok` |

### Tokens

| # | Field | Type | Description |
|---|-------|------|-------------|
| 8 | `prompt_tokens` | int | Total prompt = context + new_user |
| 9 | `new_user_tokens` | int | New tokens in this turn |
| 10 | `context_tokens_before_turn` | int | Context tokens carried from prior turns |
| 11 | `output_tokens` | int | Output tokens generated |
| 12 | `input_bytes` | int | Input bytes (= new_user x 4 + 512) |
| 13 | `output_bytes` | int | Output bytes (= output x 4 + 256) |
| 14 | `context_window_tokens` | int | Context window limit |

### SLO

| # | Field | Type | Description |
|---|-------|------|-------------|
| 15 | `slo_class` | string | `interactive_short` / `interactive_long` / `normal_long` |
| 16 | `ttft_deadline_s` | float\|null | Time-to-first-token deadline |
| 17 | `e2e_deadline_s` | float | End-to-end latency deadline |

### Session

| # | Field | Type | Description |
|---|-------|------|-------------|
| 18 | `session_id` | string\|null | Session ID (null = single-turn) |
| 19 | `turn_index` | int | 0-based turn index within session |
| 20 | `session_turn_count` | int\|null | Total turns in this session |
| 21 | `inter_turn_gap_s` | float\|null | Observed gap from previous turn |
| 22 | `session_start_time_s` | float\|null | First turn arrival time |
| 23 | `session_ttl_s` | float\|null | KV cache time-to-live (seconds) |

---

## `timebin_demand.csv` — 7 fields

Aggregated at **60-second** granularity: GROUP BY floor(arrival_time_s / 60) x region_id.

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `time_bin_start_s` | float | Bin start time |
| 2 | `time_bin_end_s` | float | Bin end time |
| 3 | `region_id` | string | Region |
| 4 | `request_count` | int | Total requests in bin |
| 5 | `prompt_tokens_total` | int | Sum of prompt_tokens |
| 6 | `new_user_tokens_total` | int | Sum of new_user_tokens |
| 7 | `output_tokens_total` | int | Sum of output_tokens |

---

## `workload_summary.json` — 14 fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `workload_id` | string | Same as manifest |
| 2 | `request_count` | int | Total requests |
| 3 | `session_count` | int | Total sessions |
| 4 | `multi_turn_session_count` | int | Sessions with >= 2 turns |
| 5 | `session_turn_count_p50` | float | Turn count per session (P50) |
| 6 | `session_turn_count_p95` | float | Turn count per session (P95) |
| 7 | `request_prompt_tokens_p50` | float | Prompt tokens per request (P50) |
| 8 | `request_prompt_tokens_p95` | float | Prompt tokens per request (P95) |
| 9 | `request_output_tokens_p50` | float | Output tokens per request (P50) |
| 10 | `request_output_tokens_p95` | float | Output tokens per request (P95) |
| 11 | `session_tokens_p50` | float | Total tokens per session (P50) |
| 12 | `session_tokens_p95` | float | Total tokens per session (P95) |
| 13 | `session_duration_s_p50` | float | Session duration in seconds (P50) |
| 14 | `session_duration_s_p95` | float | Session duration in seconds (P95) |
