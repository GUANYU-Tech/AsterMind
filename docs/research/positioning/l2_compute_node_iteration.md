# L2 compute-node iteration contract

This iteration treats every object exposed by Orbit and scheduled by Runtime as
one logical `compute_node`. A compute node encapsulates
`serving.formation.member_count` physical satellites. Orbit does not expand the
formation, Runtime does not schedule its members, and formation-internal KV
movement is outside the model.

## Serving input and validation

```json
{
  "serving": {
    "model": {"name": "llama3.3-70b"},
    "formation": {
      "member_count": 4,
      "topology": "ring",
      "link_bandwidth_gbps": 100.0,
      "link_propagation_delay_s": 0.00001
    },
    "parallelism": {"tp": 4, "pp": 1, "ep": 1},
    "prefill_compute_efficiency": 0.5,
    "decode_memory_efficiency": 0.7,
    "max_concurrency": 8
  }
}
```

`topology` accepts `ring`, `switch`, or `fully_connected`.
`link_propagation_delay_s` is the one-way propagation delay of one physical
hop. Link bandwidth is supplied directly and is not constrained by the
Architecture terminal profile. The number of nodes implied by TP/PP/EP must
equal `formation.member_count`; the former `parallelism.satellite_count` input
does not exist.

Serving has no batch-size input and no prefill/decode placement mode. Every
estimate represents homogeneous static batches with colocated prefill and decode. The only
supported model names are `llama3.3-70b`, `deepseek-v3`, and `kimi-k3`; their
model coefficients are frozen in `serving/model_profiles.py` rather than
repeated in manifests.

## Serving calculation and output

For prompt length `P`, Serving models full-prompt prefill FLOPs as

`F_prefill(P) = A_model * P + B_model * P^2`.

For concurrency `C`, Prefill scales total FLOPs and activation communication by
`C`. Decode reads active weights once per step and scales KV traffic and
activation communication by `C`. Both stages add topology-aware alpha-beta
communication implied by TP/PP/EP. Each output rate is a per-request rate.

Serving emits no workload-aggregated scalar rate. It emits:

- `prefill_rate_table`: rows containing `prompt_length`, `concurrency`, and
  `prefill_tokens_per_s`;
- `decode_rate_table`: rows containing `context_length`, `concurrency`, and
  `decode_tokens_per_s`;
- `kv_cache_bytes_per_token`, selected by canonical model name;
- `kv_cache_capacity_tokens` and the model maximum context length.

Runtime selects the exact admission-concurrency slice, then linearly interpolates cumulative prefill time `T_prefill(P,C)` rather
than the rate itself. A reusable prefix of length `K` therefore costs
`T_prefill(P) - T_prefill(K)`. Decode integrates the table's context-dependent
per-token time over generated output tokens while keeping `C` fixed. Each
logical node admits at most `max_concurrency` requests; a request holds one
slot from Prefill start through Decode completion. Requests that find all slots
occupied wait FIFO. Existing requests are not re-rated when concurrency changes.

Runtime's inference-efficiency output contains only TTFT, TPOT, and E2E. Each
is summarized with average, p50, p95, and p99 across completed requests; TPOT
excludes single-output-token requests. Completion counts are retained as
request-summary metadata, while Runtime performs no SLO evaluation.

## Runtime KV cache decision

`runtime_strategy.kv_cache_copy_enabled` is a boolean. Reuse is legal only for
the same non-null `session_id`, and reusable tokens are capped by
`context_tokens_before_turn`; current-turn `new_user_tokens` are never treated
as a reusable prefix.

For each reachable source holding the same session, Runtime compares

`T_copy(source, target) + T_prefill(P,C_compare) - T_prefill(S_source,C_compare)`

with local execution

`T_prefill(P,C_compare) - T_prefill(L_target,C_compare)`.

`C_compare` is the same target-node concurrency snapshot for every source. It
copies only when the first value is smaller and chooses the source with the
smallest copy-plus-compute time. This comparison intentionally excludes target
queue waiting. Copy retains the source entry, introduces no migration ratio,
and adds no new Runtime result fields.

## Cost aggregation and iteration boundary

Experiments aggregate the closed Architecture cost as

`total_cost = compute_node_count * formation_member_count * single_satellite_cost`.

The single-satellite cost already includes configured laser terminals, so no
second formation-link or terminal surcharge is added. Launch-cost removal,
hardware degradation, formation expansion in Orbit, formation-internal Runtime
policies, and periodic topology refresh are outside this iteration.
