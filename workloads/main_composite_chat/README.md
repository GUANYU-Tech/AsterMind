# Main Composite Chat Workload

A synthetic global ChatLLM workload that combines three independent data sources
into a single OrbitServe canonical six-file bundle:

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
  --repo-type dataset --local-dir workloads/main_composite_chat/data/

# 3. Download Llama tokenizer
hf download meta-llama/Llama-3.1-70B-Instruct \
  tokenizer.json tokenizer_config.json config.json \
  --local-dir workloads/main_composite_chat/tokenizer_llama_31/
```

## Build

```bash
python -m workloads.main_composite_chat.adapter \
  --input workloads/main_composite_chat/data/chatgpt_results_final_language_filtered.csv \
  --out workloads/main_composite_chat/bundles/chatgpt_work_weekday \
  --profile-pattern work_weekday \
  --max-sessions 20000 \
  --overwrite
```

Repeat for `work_weekend`, `nonwork_weekday`, and `nonwork_weekend`.
The production build attempts 20,000 timestamp-valid conversations; sessions
that cannot fit wholly inside the selected one-day window are omitted, leaving
about 18.5K sessions and 107K requests per pattern.

## Timing and token semantics

The adapter emits `astermind.workload.causal.v1`. The first turn of each
conversation uses its sampled absolute arrival time. Every later turn is
released only after its predecessor finishes, plus:

```text
inter_turn_gap_s = next user message time - previous assistant completion time
```

ShareChat must therefore provide both user and assistant
`message_create_time` values. Inputs without those timestamps are rejected;
the adapter no longer substitutes adjacent request-arrival differences.

Token accounting uses one field set:

```text
prompt_tokens = context_tokens_before_turn + new_user_tokens
```

`context_tokens_before_turn` is reusable cached context and
`new_user_tokens` is the current prefill work. The adapter assumes the full
preceding ShareChat history is reusable when the chain stays on its KV owner.

For smoke testing without the Llama tokenizer:

```bash
python -m workloads.main_composite_chat.adapter \
  --input workloads/main_composite_chat/data/chatgpt_results_final_language_filtered.csv \
  --out workloads/main_composite_chat/bundles/smoke_test \
  --profile-pattern work_weekday \
  --max-sessions 10 \
  --tokenizer-backend simple_unicode_v1 \
  --overwrite
```

## Validate

```bash
python -c "
from astermind.workload import load_workload_bundle
b = load_workload_bundle('workloads/main_composite_chat/bundles/chatgpt_work_weekday')
print(f'{b.summary[\"request_count\"]} requests, {b.summary[\"session_count\"]} sessions')
"
```
