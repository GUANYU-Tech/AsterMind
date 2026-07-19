# WildChat to OrbitServe Experiment Workload

This is a real-data adapter for experiments, not a minimal example. It reads one pinned public WildChat-4.8M Parquet shard and produces the canonical seven-file bundle required by `orbitserve.workload`.

## Inputs, Outputs, and Boundaries

The input is an official raw Parquet file and remains in its original format. WildChat does not need to be, and should not be, rewritten as JSON. The checked-in example configuration, `wildchat_experiment_config_2024-11-11.json`, selects `data/train-00050-of-00086.parquet` from dataset revision `c827c6df8fcf008219ffaffa4d1dd77491099367` and filters one UTC day, 2024-11-11. The shard is approximately 136 MB and contains 37,208 raw Parquet records. These records are not final OrbitServe sessions or requests. The adapter extracts eligible adjacent user-to-assistant pairs from each record's `conversation` field. The configured SHA-256 verifies the raw source file, and the adapter stops on a mismatch.

The 2024-11-11 output directory contains exactly these seven consumer-visible files:

```text
bundles/
  2024-11-11/
    experiment_bundle/
      workload_manifest.json
      regions.json
      gateways.json
      requests.jsonl
      sessions.jsonl
      timebin_demand.csv
      workload_summary.json
```

The sibling files `bundles/2024-11-11/canonical_requests.jsonl`, `bundles/2024-11-11/canonical_sessions.jsonl`, and `bundles/2024-11-11/generated_description.json` are audit sidecars for `source_mode=explicit_requests`. They are neither an eighth bundle file nor raw WildChat data.

`bundles/2024-11-11/` is a validated example committed with the adapter so that the seven-file format and integration path can be checked without downloading Parquet data. Other `bundles/YYYY-MM-DD/` directories and the root `experiment_bundle/` are local adapter outputs and are not tracked. Use the `build` command below to reconstruct the 2024-11-11 example from the pinned source shard or to generate a different date directory.

The raw Parquet file may be placed in the Git-ignored `src/workloads/wildchat/data/` directory. Its SHA-256 must match the expected value in the configuration. The file is not tracked, but it can be retained for reconstruction and audit. Before building, the adapter validates the byte-level SHA-256, Parquet structure, and required fields.

This adapter shares the `orbitserve.workload` contract with the synthetic, explicit-request, and trace-replay generators. Before handing a bundle to the main simulation, the core loader validates the exact seven-file set, fields, per-file SHA-256 values, and bundle hash.

## File Roles

The following table describes the source files and validated example. Raw data, downloaded tokenizer caches, and newly generated bundles are not tracked. `bundles/2024-11-11/` is the only committed generated example. JSON files under `tokenizer/` are tokenizer vocabulary and template files, not duplicate workload configuration.

| File or directory | Type | Purpose | Consumer |
|---|---|---|---|
| `wildchat_adapter.py` | Python | Performs the complete Parquet-to-canonical-requests-to-bundle conversion and validation | Integrator or experiment script |
| `wildchat_experiment_config_2024-11-11.json` | Input JSON | Runnable 2024-11-11 configuration covering dataset revision/SHA, time window, regions, 26 gateways, tokenizer, model, SLO, filters, and output policy | Adapter |
| `state_centroids.json` | Input JSON | Representative coordinates for 162 commonly used states, provinces, and federal subjects in China, the United States, Russia, Australia, Canada, and Brazil | State/province geography fallback |
| `country_centroids.json` | Input JSON | External country-name-to-representative-coordinate table, used only when WildChat has no state mapping | Geography mapping |
| `tokenizer/` | Local tokenizer files | Qwen `tokenizer.json`, `tokenizer_config.json`, `config.json`, and related files for offline tokenization; no 7B model weights | Adapter and Transformers |
| `bundles/2024-11-11/canonical_requests.jsonl` | Generated sidecar | Explicit request input with adapter-completed fields; the manifest records its source hash and record count | Adapter audit and `explicit_requests` consumer |
| `bundles/2024-11-11/canonical_sessions.jsonl` | Generated sidecar | Session input corresponding to canonical requests, used for session consistency and audit | Adapter audit and `explicit_requests` consumer |
| `bundles/2024-11-11/generated_description.json` | Generated sidecar JSON | Complete workload description covering geography, gateways, tokens, SLO, and provenance; not part of the seven-file bundle, but consumed by `validate --description` | Validator and audit |
| `bundles/2024-11-11/experiment_bundle/workload_manifest.json` | Canonical output JSON | Bundle schema, SHA values for the six data files, record counts, source mode, and bundle hash | OrbitServe workload reader |
| `bundles/2024-11-11/experiment_bundle/regions.json` | Canonical output JSON | Active demand regions, representative coordinates, time zones, and demand weights | OrbitServe workload reader |
| `bundles/2024-11-11/experiment_bundle/gateways.json` | Canonical output JSON | 26 gateways, coordinates, bandwidth, and served regions | OrbitServe workload reader |
| `bundles/2024-11-11/experiment_bundle/requests.jsonl` | Canonical output JSONL | One final request per adjacent user-to-assistant pair, including arrival, token, byte, gateway, model, SLO, and session fields | OrbitServe workload reader |
| `bundles/2024-11-11/experiment_bundle/sessions.jsonl` | Canonical output JSONL | Session start time, turn count, session SLO, context window, and TTL | OrbitServe workload reader |
| `bundles/2024-11-11/experiment_bundle/timebin_demand.csv` | Canonical output CSV | Request count and token demand aggregated by 300-second time bin and region | OrbitServe demand model and scheduler |
| `bundles/2024-11-11/experiment_bundle/workload_summary.json` | Canonical output JSON | Request, session, region, gateway, and time-bin counts; token/byte statistics; and provenance hash | OrbitServe and experiment reports |

OrbitServe reads only the seven files under `bundles/2024-11-11/experiment_bundle/`. The external configuration, country centroids, tokenizer, and three sidecars support adapter generation, reproduction, and audit only.

## One-Time Setup

Install the two runtime dependencies:

```powershell
python -m pip install pyarrow transformers
```

Download the pinned tokenizer files. They require only a few megabytes and do not include model weights. After this step, tokenization can run offline.

```powershell
hf download Qwen/Qwen2.5-7B-Instruct `
  --revision a09a35458c702b33eeacc393d103063234e8bc28 `
  --local-dir src/workloads/wildchat/tokenizer `
  --include tokenizer.json tokenizer_config.json config.json
```

Download the official pinned shard locally. Retain the raw Parquet file for reproducible experiment builds.

```powershell
New-Item -ItemType Directory -Force src/workloads/wildchat/data | Out-Null
Invoke-WebRequest `
  'https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/c827c6df8fcf008219ffaffa4d1dd77491099367/data/train-00050-of-00086.parquet?download=true' `
  -OutFile src/workloads/wildchat/data/train-00050-of-00086.parquet
Get-FileHash src/workloads/wildchat/data/train-00050-of-00086.parquet -Algorithm SHA256
```

Expected SHA-256:

```text
203f57614c093871cbe42dd9da0528cb33354230a511d3ec509c80c6fce0b116
```

## Build and Validation

First run the geography regression check to verify state/province precedence and nearest-gateway selection:

```powershell
python src/workloads/wildchat/check_geography.py
```

The output must include at least these key cases:

```text
PASS China|Guangdong -> china_guangdong -> gw_hong_kong
PASS China|Hebei -> cn_hebei -> gw_beijing
PASS France|<none> -> country_france (country fallback)
```

Then run the complete path on the real Parquet file:

```powershell
python src/workloads/wildchat/wildchat_adapter.py build `
  --config src/workloads/wildchat/wildchat_experiment_config_2024-11-11.json `
  --input src/workloads/wildchat/data/train-00050-of-00086.parquet `
  --out src/workloads/wildchat/bundles/2024-11-11/experiment_bundle --overwrite

python src/workloads/wildchat/wildchat_adapter.py validate `
  --bundle src/workloads/wildchat/bundles/2024-11-11/experiment_bundle `
  --description src/workloads/wildchat/bundles/2024-11-11/generated_description.json
```

Each adjacent `user -> assistant` pair becomes one request. `arrival_time_s` is the subsequent assistant timestamp minus the experiment epoch. It is explicitly a response-completion-time proxy, not the actual user send time. Prompt and output tokens are computed with the pinned `Qwen/Qwen2.5-7B-Instruct` tokenizer revision and its default chat template. Byte counts equal the configured four bytes per token multiplied by the token count.

## Changing the Dataset Shard or Date

`wildchat_adapter.py` is generic conversion code. Changing the shard or date does not require Python changes. Copy `wildchat_experiment_config_2024-11-11.json`, create a configuration for the new experiment, and update at least these fields:

```text
workload_id                         # Unique experiment name
dataset.parquet_file                # Local official Parquet path
dataset.expected_source_sha256      # SHA-256 of that Parquet file
dataset.max_source_records          # Actual shard record count; remove to read all records
dataset.time_window.epoch_utc       # 00:00:00Z of the target UTC day
```

Pass the new configuration, Parquet path, and `bundles/YYYY-MM-DD/experiment_bundle/` output path through `--config`, `--input`, and `--out`. Change `region_mapping`, gateways, tokenizer, and SLO policy only when the experiment assumptions change; these are not date-dependent fields. A Parquet shard may cover multiple days, so confirm that it includes the target date instead of assuming one shard per day.

One day here means records in the downloaded shard that fall within the selected UTC day. It is not the complete set of global WildChat-4.8M requests for that day. The official release unit is a Parquet shard, not a date-partitioned file, and the current adapter reads one shard at a time. To collect all records for one day across the full dataset, locate and download every shard covering that day and extend the adapter for multi-shard input. That is outside the scope of this example.

## Geography, Gateways, and SLO

- WildChat `country` and `state` fields follow this mapping precedence: explicit `region_mapping`, then `state_centroids.json`, then `country_centroids.json`. Country centroids provide representative country-level points, not precise user coordinates.
- `state_centroids.json` currently covers China, the United States, Russia, Australia, Canada, and Brazil. Other countries continue to use country-centroid fallback.
- The 26 global gateways are external experiment assumptions, not WildChat fields. Each request selects the nearest gateway by Haversine distance. Gateways represent terrestrial entry points into the satellite network, not user locations.
- SLO classes are also external to WildChat. The default policy selects four classes from the maximum output-token count in a session: 3-second TTFT with 30-, 60-, or 120-second E2E deadlines, or a long-task class with 10-second TTFT and 300-second E2E.

To scale load without changing the within-day arrival shape, retain the original 1x workload and optionally combine a pinned set of shards before constructing 10x, 50x, or 100x replication experiments. Every scaled configuration must pin its shard list, file hashes, seed, and output directory. Random amplification must not be presented as observed traffic.

See the [WildChat-4.8M dataset card](https://huggingface.co/datasets/allenai/WildChat-4.8M) for official data documentation, field definitions, and licensing.
