# SkyWalker-Inspired Global LLM Workload Sample

This directory contains a small `explicit_requests` workload input for OrbitServe:

- `workload_description.json`
- `requests.jsonl`
- `sessions.jsonl`

The bundle is synthetic. It is inspired by the workload structure discussed in SkyWalker/SkyLB multi-region LLM serving work, and by public chatbot workload patterns such as WildChat and LMSYS/Chatbot Arena style multi-turn conversations. It is not a real SkyWalker, WildChat, ChatBot Arena, or production trace, and no original trace rows are copied.

## Intended Use

Use this as a small, reviewable user-supplied workload input for parser, scheduler, and experiment smoke tests. These historical request/session rows are separate from the current six-file workload bundle and SimulationConfig.requests. They intentionally avoid runtime result fields such as `assigned_satellite`, `selected_path`, `session_action`, `served`, `TTFT`, `E2E`, `energy_violation`, or `cost_per_request`.

The files are retained as source examples; no current compiler or Native entry automatically converts this format.

## Shape

| Item | Value |
|---|---:|
| Requests | 60 |
| Sessions | 20 |
| Horizon | 86,400 s |
| Source mode | `explicit_requests` |
| Access mode | `gateway_sgl` |
| Model family | `llama_70b_like` |

## Region Mix

| Region | Requests | Share |
|---|---:|---:|
| `us_east` | 24 | 40% |
| `europe_west` | 18 | 30% |
| `asia_east` | 18 | 30% |

The timing is manually arranged to show a diurnal shift: Asia is active around UTC 01:00 and 11:00-13:50, Europe around UTC 08:10-09:04 and 16:30-20:20, and the US around UTC 00:10-02:10 and 14:30-23:05.

## Sessions and KV Locality

Every request has `session_id` and `turn_index`. The sessions include single-turn, short two-turn chats, and longer 3-5 turn conversations. `context_tokens_before_turn` and `prompt_tokens = context_tokens_before_turn + new_user_tokens` are precomputed so downstream schedulers can test sticky routing, KV retention, migration, or recompute policies without adding workload-side runtime decisions.

Session turn-count distribution:

| Turns | Sessions |
|---:|---:|
| 1 | 2 |
| 2 | 5 |
| 3 | 5 |
| 4 | 7 |
| 5 | 1 |

## Token Tail

| Token field | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| `prompt_tokens` | 3236 | 16612 | 22720 | 22720 |
| `output_tokens` | 256 | 2048 | 3072 | 3072 |

The long prompt tail is carried by `long_context_qa` and later session turns with accumulated context. The output tail is carried by `math_reasoning` and `code` requests.

## Burst Design

| Burst | Region | Requests | Purpose |
|---|---|---:|---|
| `asia_tot_fanout_burst` | `asia_east` | 7 | Tree-of-Thought-like fan-out: clustered math, code, summarization, and long-context requests. |
| `us_early_evening_reasoning_burst` | `us_east` | 2 | Tight local-evening math reasoning pair with long decode pressure. |
| `us_workday_code_burst` | `us_east` | 3 | Tight code/math cluster during the US workday. |
| `eu_evening_launch_burst` | `europe_west` | 6 | Europe evening mixed launch-like event with summarization, chat, code, and math. |
| `us_late_workday_reasoning_burst` | `us_east` | 4 | Late US workday reasoning cluster with mixed follow-up chat and summarization. |

## Provenance

This bundle is synthetic and paper-structure-inspired only. It should be cited or described as `synthetic SkyWalker-inspired workload sample`, not as a SkyWalker original trace.

Reference links:

- SkyWalker/SkyLB arXiv preprint: <https://arxiv.org/abs/2505.24095>
- WildChat arXiv preprint: <https://arxiv.org/abs/2405.01470>
- LMSYS-Chat-1M / Chatbot Arena dataset paper: <https://arxiv.org/abs/2309.11998>
