# OrbitServe Workloads — WildChat Adapter

Converts real [WildChat](https://huggingface.co/datasets/allenai/WildChat-4.8M) conversations into **canonical 7-file workload bundles** consumed by the OrbitServe simulation engine.

## Module Layout

| Path | Role |
|---|---|
| `src/workloads/wildchat/` | Dataset adapter: WildChat parquet → bundle (tokenization, geo-mapping, month compression, grafting) |
| `src/orbitserve/workload/contract.py` | Shared contract: bundle format, field sets, data types, validation, canonicalization/hashing (stdlib-only) |
| `src/orbitserve/workload/generation.py` | Build-side: synthesize/compile bundles from statistical descriptions (offline; not used by the WildChat path) |
| `src/orbitserve/workload/loader.py` | Runtime-side: `load_workload_bundle` (read + validate) and `materialize_workload_bundle` (resolve reference → load) |
| `src/orbitserve/workload/upsample.py` | Runtime-side: lazy deterministic session upsampling stream for load scaling |

The public API (`orbitserve/workload/__init__.py`) re-exports 26 names from `contract` / `generation` / `loader`; it is byte-compatible with previous releases.

## Inputs

| Input | Notes |
|---|---|
| WildChat parquet shards (`train-*.parquet`) | One row per conversation (turns, UTC timestamps, country/state, hashed IP, user-agent). Each shard is SHA-256-verified against the pinned hashes in the experiment config before use. |
| DeepSeek-V3 tokenizer | Pinned local directory (`tokenizer_deepseek_v3/`). Every request is tokenized per turn with the chat template; context window 131,072 (128K). |
| Geo catalogs | `country_centroids.json` / `state_centroids.json` map raw country/state names to representative coordinates; explicit `region_mapping` entries in the config take precedence. |

## Data Setup (not committed)

The raw dataset and tokenizer are **not** in the repo — fetch them before building:

**1. WildChat parquet shards** → place under `src/workloads/wildchat/data/`

Download from Hugging Face ([allenai/WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M/tree/main)), e.g.:

```bash
# shards for the 2023-05 skeleton config
curl -L -o src/workloads/wildchat/data/train-00001-of-00086.parquet \
  https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/train-00001-of-00086.parquet
curl -L -o src/workloads/wildchat/data/train-00002-of-00086.parquet \
  https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/train-00002-of-00086.parquet
curl -L -o src/workloads/wildchat/data/train-00003-of-00086.parquet \
  https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/train-00003-of-00086.parquet

# shards for the 2025-03 content config
curl -L -o src/workloads/wildchat/data/train-00066-of-00086.parquet \
  https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/train-00066-of-00086.parquet
curl -L -o src/workloads/wildchat/data/train-00067-of-00086.parquet \
  https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/train-00067-of-00086.parquet
curl -L -o src/workloads/wildchat/data/train-00068-of-00086.parquet \
  https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/train-00068-of-00086.parquet
```

Each experiment config pins the exact shard filenames and their expected SHA-256 under `dataset.expected_source_sha256` — check there for the required filenames; the build fails on any hash mismatch.

**2. DeepSeek-V3 tokenizer** → place under `src/workloads/wildchat/tokenizer_deepseek_v3/`

```bash
curl -L -o src/workloads/wildchat/tokenizer_deepseek_v3/tokenizer.json \
  https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json
curl -L -o src/workloads/wildchat/tokenizer_deepseek_v3/tokenizer_config.json \
  https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer_config.json
```

Only the tokenizer files are needed (a few MB), no model weights. The adapter loads them offline via `tokenization.local_path`; if the tokenizer backend reports a missing file, download it from the same repo path pattern.

## Output: Canonical Bundle

A bundle directory contains exactly seven files, validated by `load_workload_bundle` (exact field sets, per-file SHA-256, cross-file consistency):

| File | Contents |
|---|---|
| `workload_manifest.json` | Schema, workload ID, per-file SHA-256 and record counts, `bundle_sha256`, source, seed, time range, provenance |
| `regions.json` | One row per region: ID, name, coordinates, timezone offset, demand weight and its source |
| `gateways.json` | One row per ground gateway: ID, coordinates, served regions, bandwidth, availability profile |
| `requests.jsonl` | One row per request: arrival time, region/ingress references, task type, model family, `prompt_tokens = context_tokens_before_turn + new_user_tokens`, `output_tokens`, I/O bytes, context window, TTFT/E2E deadlines, SLO class, priority, deterministic seed; multi-turn rows carry `session_id` and turn info |
| `sessions.jsonl` | One row per session: region, start time, turn count, context window, session SLO, TTL (KV-cache residency), seed |
| `timebin_demand.csv` | Pre-aggregated request and token counts per time bin × region |
| `workload_summary.json` | Totals and token/session statistics |

## Build Modes

The adapter supports four mutually exclusive modes (each with its own config):

1. **Single-day** — one natural day from a parquet shard (initial mode; sensitive to single-day anomalies).
2. **Month-compressed day** (`month_compressed_day`) — aggregates a full month into 5-minute time-of-day bins and compresses them into one representative day. Load intensity is defined on the month total (`10x` = full month; `1x/2x/5x` are deterministic subsamples).
3. **Profile warp** (transitional) — `--profile-input`: re-deals content sessions of month B onto the time-of-day envelope of month A (macro-region UTC → macro-region local → per-country local iterations). Superseded by skeleton graft.
4. **Skeleton graft** (current) — `--skeleton-bundle`: keeps every session slot (region + start time) of an existing skeleton bundle (e.g. 2023-05) and replaces the content with whole real conversations from another month (e.g. 2025-03), same-country preferred with global fallback. The session-level diurnal curve and country proportions therefore match the skeleton exactly, while contexts are real long-context (prompt p99 ≈ 43K, max 131,072).

### Build the graft bundle

```bash
# one-time: build the 2023-05 skeleton (month-compressed day)
python src/workloads/wildchat/wildchat_adapter.py build \
  --config src/workloads/wildchat/wildchat_experiment_config_2023-05.json \
  --input src/workloads/wildchat/data/train-0000{1,2,3}-of-00086.parquet \
  --out src/workloads/wildchat/bundles/2023-05/experiment_bundle_10x --10x --overwrite

# graft 2025-03 content onto the skeleton
python src/workloads/wildchat/wildchat_adapter.py build \
  --config src/workloads/wildchat/wildchat_experiment_config_graft_2023skel_2025content.json \
  --skeleton-bundle src/workloads/wildchat/bundles/2023-05/experiment_bundle_10x \
  --input src/workloads/wildchat/data/train-0006{6,7,8}-of-00086.parquet \
  --out src/workloads/wildchat/bundles/2025-content-on-2023-skeleton/experiment_bundle_10x \
  --10x --overwrite

# validate
PYTHONPATH=src python -c "from orbitserve.workload import load_workload_bundle; \
  load_workload_bundle('src/workloads/wildchat/bundles/2025-content-on-2023-skeleton/experiment_bundle_10x')"
```

The 10x graft bundle contains 135,355 requests / 82,382 sessions / 268 regions. Request count is lower than the skeleton's because turn counts come from the content (2023: ≈2.55 turns/session; 2025: ≈1.85).

## Load Scaling

The 10x bundle (≈135K requests/day) is the base shape. Constellation-scale loads are produced at runtime by `orbitserve.workload.upsample.stream_upsampled_sessions(workload, factor, seed=...)`: a lazy deterministic stream replicating each session `factor` times (new IDs, bin-local jitter, turn structure preserved) with no file or memory blow-up. Required factor: `K = N_cell × r_cell × 86400 / 135355`.

## Testing

```bash
PYTHONPATH=src python -m pytest tests/unit/workload tests/unit/test_module_boundaries.py -q
```

- `tests/unit/workload/test_upsample.py` — upsampling determinism, jitter bounds, laziness (9 tests)
- `tests/unit/workload/test_wildchat_skeleton_graft.py` — slot preservation, country matching, fallback, determinism, mutex, end-to-end mini build (7 tests)

## Repository Notes

- Raw parquet shards (`data/`), the pinned tokenizer, and large bundles are **not** committed; only code, configs, tests and the designated sample bundle are.
- `.gitattributes` at the repo root forces LF line endings for `src/workloads/**` data files; bundle SHA-256 is computed over LF bytes, so any EOL conversion breaks validation.
