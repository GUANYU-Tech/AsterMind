# OrbitServe Workload Module and Canonical Bundle Interface

The workload module converts heterogeneous request datasets into one stable
interface consumed by OrbitServe simulation and runtime engines.

```text
raw dataset + geographic distribution + diurnal profile
                         |
                         v
                 dataset adapter
                         |
                         v
              canonical six-file bundle
                         |
                         v
             loader -> upsample -> runtime
```

Dataset-specific semantics belong in `src/workloads/<name>/adapter.py`.
Consumers depend only on `src/orbitserve/workload` and the canonical bundle.

## Module responsibilities

### Dataset adapters

An adapter reads a source dataset, maps it onto a geographic and temporal
pattern, performs token/SLO conversion, and writes a canonical bundle.

Reference patterns:

| Pattern | Adapter | Source semantics |
|---|---|---|
| ChatLLM | `src/workloads/main_composite_chat/adapter.py` | ShareChat user/assistant conversations |
| Agent | `src/workloads/semianalasys/adapter.py` | Agent and subagent model-call traces with hash-based KV reuse |

Both patterns may use the same four regional curves (`work_weekday`,
`work_weekend`, `nonwork_weekday`, `nonwork_weekend`). The curve determines
the nominal daily distribution; the source dataset determines request/session
content and dependency structure.

### `contract.py`

Defines schemas, exact row keys, data classes, canonical serialization,
bundle hashing, coercion, and cross-file validation. It is stdlib-only so both
adapters and runtime code can import it.

### `loader.py`

`load_workload_bundle(path)` reads and validates a bundle and returns a
`WorkloadGenerationResult` containing:

- `manifest`
- `regions`
- `gateways`
- `session_requests`
- `timebin_demand`
- `summary`

`materialize_workload_bundle(ref, base_dir)` resolves a `WorkloadBundleRef`
and calls the same loader.

### `upsample.py`

`stream_upsampled_sessions(workload, factor, seed=...)` lazily replicates each
session. It re-derives session, request, and predecessor IDs, applies one
deterministic bin-local shift to the session's nominal timeline, and preserves
turn order and `inter_turn_gap_s`.

## Bundle directory interface

A canonical bundle contains exactly six files:

| File | Format | Purpose |
|---|---|---|
| `workload_manifest.json` | JSON object | Bundle identity, schema, horizon, provenance, and integrity hash |
| `regions.json` | JSON array | User-region coordinates and demand weights |
| `gateways.json` | JSON array | Gateway coordinates and served regions |
| `session_requests.jsonl` | JSONL | Request, session, token, and dependency records |
| `timebin_demand.csv` | CSV | Nominal demand aggregates |
| `workload_summary.json` | JSON object | Adapter-produced workload statistics |

All JSON is UTF-8. Canonical JSON uses sorted keys, compact separators, one
trailing newline, finite numeric values, and at most six decimal places.

## Schemas

| `schema_version` | Use |
|---|---|
| `orbitserve.workload` | Absolute-arrival workloads using the 24 base request fields |
| `orbitserve.workload.causal.v1` | Completion-relative workloads using the 24 base fields plus four dependency/provenance fields |

New multi-turn ChatLLM and Agent adapters should emit
`orbitserve.workload.causal.v1`.

## `workload_manifest.json`

Common fields are:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | One of the supported schemas above |
| `canonicalization` | string | `orbitserve-json-c14n` |
| `workload_id` | string | Stable bundle identity |
| `seed` | integer | Base deterministic seed |
| `time_horizon_s` | number | Nominal workload horizon in seconds |
| `bundle_sha256` | string | Composite hash of the five data files and `workload_id` |

Adapters may add provenance such as source dataset, profile pattern, arrival
semantics, and token-reuse semantics.

## `regions.json`

Each row contains exactly five fields:

| Field | Type | Meaning |
|---|---|---|
| `region_id` | string | Unique request-facing region ID |
| `region_name` | string | Human-readable name |
| `lat_deg` | number | Representative latitude |
| `lon_deg` | number | Representative longitude |
| `demand_weight` | number | Region share of bundle demand |

## `gateways.json`

Each row contains exactly four fields:

| Field | Type | Meaning |
|---|---|---|
| `gateway_id` | string | Unique gateway ID |
| `lat_deg` | number | Gateway latitude |
| `lon_deg` | number | Gateway longitude |
| `served_region_ids` | array of strings | Regions reachable through the gateway |

## `session_requests.jsonl`

### Base fields

| Group | Field | Type | Meaning |
|---|---|---|---|
| Identity | `request_id` | string | Globally unique request ID |
| Time | `arrival_time_s` | number | Absolute release time for legacy rows; nominal curve time for causal rows |
| Geography | `user_region_id` | string | Region referenced by `regions.json` |
| Access | `access_mode` | string | `direct_user_sgl` or `gateway_sgl` |
| Access | `ingress_id` | string | Entry region or gateway ID |
| Access | `gateway_id` | string or null | Gateway ID for gateway access |
| Model | `model_family` | string | Requested model family |
| Tokens | `prompt_tokens` | integer | Complete model input represented by the request |
| Tokens | `new_user_tokens` | integer | Tokens that require prefill for this request |
| Tokens | `context_tokens_before_turn` | integer | Prompt tokens whose KV is reusable for this request |
| Tokens | `output_tokens` | integer | Tokens decoded by the request |
| Transport | `input_bytes` | integer | Request payload bytes used by the transport model |
| Transport | `output_bytes` | integer | Response payload bytes used by the transport model |
| Model | `context_window_tokens` | integer | Model context-window limit |
| SLO | `slo_class` | string | Workload-defined service class |
| SLO | `ttft_deadline_s` | number or null | TTFT deadline relative to actual arrival |
| SLO | `e2e_deadline_s` | number | E2E deadline relative to actual arrival |
| Session | `session_id` | string or null | Context-chain ID; null for an independent single turn |
| Session | `turn_index` | integer | Zero-based index within `session_id` |
| Session | `session_turn_count` | integer or null | Number of rows in the session |
| Session | `inter_turn_gap_s` | number or null | Completion-to-next-submission think time |
| Session | `session_start_time_s` | number or null | Nominal first-turn time |
| Session | `session_ttl_s` | number or null | Runtime KV/session TTL |
| Identity | `seed` | integer | Deterministic per-request seed |

### Causal fields

`orbitserve.workload.causal.v1` appends exactly four fields:

| Field | Type | Meaning |
|---|---|---|
| `arrival_time_mode` | string | `absolute` or `after_dependency_completion` |
| `depends_on_request_id` | string or null | Request that must complete before this request can be released |
| `source_trace_time_s` | number | Original request-start offset retained as provenance |
| `source_api_time_s` | number | Original observed request duration retained as provenance |

## Token accounting contract

Every request satisfies:

```text
prompt_tokens
  = context_tokens_before_turn
  + new_user_tokens
```

`context_tokens_before_turn` is the declared reusable KV portion. Runtime may
reduce the actual hit when KV is absent, expired, evicted, or placed elsewhere.
`new_user_tokens` is the prefill work even when it includes non-reusable
historical or tool tokens rather than literal user text.

For a hash-block Agent trace, the adapter computes reusable tokens from the
known hash overlap. For a ChatLLM chain, the adapter declares the preceding
conversation context reusable; runtime placement and KV policy determine the
actual hit.

## Arrival and dependency contract

### Absolute row

```text
arrival_time_mode = "absolute"
depends_on_request_id = null
inter_turn_gap_s = null
actual_arrival_time_s = arrival_time_s
```

### Completion-relative row

```text
arrival_time_mode = "after_dependency_completion"

actual_arrival_time_s
  = completion_time_s(depends_on_request_id)
  + inter_turn_gap_s
```

`completion_time_s` is full simulated E2E completion, not TTFT.
`inter_turn_gap_s` is non-negative client, tool, or agent think time measured
from predecessor completion to current submission. Independent roots may run
concurrently.

For causal rows, `arrival_time_s` remains the nominal time sampled from the
daily curve. It is used for bundle inspection, visualization, and
`timebin_demand.csv`; runtime performance metrics use dynamically resolved
actual arrival times.

Reference adapter derivations:

- SemiAnalysis Agent: use source `think_time`; otherwise derive
  `max(0, current.t - (previous.t + previous.api_time))`.
- ShareChat: derive `current_user_timestamp - previous_assistant_timestamp`.

## Session and reference invariants

- Request IDs are unique.
- `turn_index` values are contiguous and zero-based within a session.
- `session_turn_count` equals the number of rows in that session.
- A causal predecessor exists in the same bundle and is not the request itself.
- Dependency graphs are acyclic.
- `user_region_id` resolves through `regions.json`.
- `gateway_id` resolves through `gateways.json` when non-null.
- For `gateway_sgl`, `ingress_id == gateway_id`.
- For `direct_user_sgl`, `ingress_id == user_region_id`.

## `timebin_demand.csv`

The canonical columns are:

```text
time_bin_start_s, time_bin_end_s, region_id, request_count,
prompt_tokens_total, new_user_tokens_total, output_tokens_total
```

Adapters aggregate by nominal `arrival_time_s` and region. The current
reference adapters use 60-second bins. For causal bundles this file describes
the input demand curve, not performance-dependent runtime release times.

## `workload_summary.json`

The summary is adapter-defined and intended for inspection. It should include
at least `workload_id`, `request_count`, and `session_count`. Percentiles,
source counts, reuse ratios, and pattern metadata may be added as appropriate.
Runtime correctness must not depend on optional summary fields.

## Loader validation order

`load_workload_bundle` performs:

1. schema and canonicalization validation;
2. exact six-file and composite-hash validation;
3. exact row-key validation for the selected schema;
4. type coercion and rejection of runtime-result fields;
5. token and session invariant validation;
6. causal and geographic reference validation.

Example:

```python
from orbitserve.workload import load_workload_bundle

workload = load_workload_bundle("path/to/experiment_bundle_1x")
print(len(workload.session_requests))
```
