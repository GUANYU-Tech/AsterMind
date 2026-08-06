# Main Composite Chat Workload

A synthetic global ChatLLM workload that combines three independent data sources
into a single OrbitServe canonical 7-file bundle:

| Input | Source | Location |
|------|------|------|
| Multi-turn conversations | HuggingFace: tucnguyen/ShareChat | `data/` (download required, gitignored) |
| Country request weights | Microsoft + CNNIC Bridge | `countries.csv` |
| Regional diurnal profiles | Profound 2026 Fig.2 manual digitization | `profiles/{pattern}.csv` + `figure2_extracted_{pattern}.json` |
| Gateway configuration | Reuses WildChat global gateways | `gateways.json` |
| Tokenizer | Llama-3.1-70B | `tokenizer_llama_31/` (download required, gitignored) |

## Four Patterns

```
--profile-pattern work_weekday    work conversations × weekday
--profile-pattern work_weekend    work conversations × weekend
--profile-pattern nonwork_weekday non-work conversations × weekday
--profile-pattern nonwork_weekend non-work conversations × weekend
```

Each pattern uses a different 24-hour local-time arrival curve per region
(Asia, Europe, North America, Latin America), manually digitized from Profound
Figure 2.  Countries in the same region share the same curve shape.

## Setup

```bash
# 1. Log in to HuggingFace (the dataset is gated)
huggingface-cli login

# 2. Download ShareChat data
hf download tucnguyen/ShareChat chatgpt_results_final_language_filtered.csv \
  --repo-type dataset --local-dir data/

# 3. Download Llama tokenizer
hf download meta-llama/Llama-3.1-70B-Instruct \
  tokenizer.json tokenizer_config.json config.json \
  --local-dir tokenizer_llama_31/
```

## Build

```bash
PYTHONPATH=src python adapter.py \
  --input data/chatgpt_results_final_language_filtered.csv \
  --out bundles/chatgpt_work_weekday \
  --profile-pattern work_weekday \
  --max-sessions 20000 \
  --overwrite
```

Repeat for `work_weekend`, `nonwork_weekday`, and `nonwork_weekend`.

For smoke testing without the Llama tokenizer:

```bash
PYTHONPATH=src python adapter.py \
  --input data/chatgpt_results_final_language_filtered.csv \
  --out bundles/smoke_test \
  --profile-pattern work_weekday \
  --max-sessions 10 \
  --tokenizer-backend simple_unicode_v1 \
  --overwrite
```

## Validate

```bash
PYTHONPATH=src python -c "
from orbitserve.workload import load_workload_bundle
b = load_workload_bundle('bundles/chatgpt_work_weekday')
print(f'{b.summary[\"request_count\"]} requests, {b.summary[\"session_count\"]} sessions')
"
```
