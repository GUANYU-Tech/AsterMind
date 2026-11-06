# Runtime inference metric contract

Implementation reviewed: 2026-11-03. This contract describes the current physical
network Runtime, including failures and causal multi-turn requests.

Runtime reports exactly three inference-efficiency metrics under
`inference_metrics`: TTFT, TPOT, and E2E. Each metric contains `average`,
`p50`, `p95`, and `p99`, expressed in seconds.

For completed request `i`, let `arrival_i` be its arrival time, `first_i` the
time its first generated token reaches the user, `last_i` the time its complete
response reaches the user, and `N_i` its number of output tokens:

`TTFT_i = first_i - arrival_i`

`E2E_i = last_i - arrival_i`

`TPOT_i = (last_i - first_i) / (N_i - 1)`

TPOT is null for requests with one output token because there is no interval
between generated tokens. Such requests remain in the TTFT and E2E populations
but do not enter the TPOT population. Uncompleted requests enter none of the
three latency populations.

The percentile implementation sorts the valid request-level samples and uses
linear interpolation at position `q * (n - 1)`. `average` is the arithmetic
mean. When a population is empty, all four values are null.

Runtime no longer evaluates SLOs or emits SLO placeholders. Request completion
counts and uncompleted reasons remain under `request_summary` so latency
distributions cannot be interpreted without their population outcome. Energy,
KV cache, region, state, and decision audit outputs remain separate from
inference efficiency.

Workload deadline and SLO-class fields may remain as source-trace metadata, but
Runtime does not consume or copy them into its result.

## Event time and delivery semantics

Runtime executes requests, hop transmissions, arrivals, computation and delivery
in chronological event order. Physical positions and environment-derived link
attributes use the latest saved sample. This sample-hold rule does not round
request arrival, computation completion or continuous fault/repair timestamps.
Configured capacity-schedule changes also retain their exact event times.

The response model is `first_token_then_remaining_output`: Runtime delivers the
first token separately and the remaining response after its computed decode
completion. It does not emit one network delivery event per generated token.
Consequently, TPOT is an average derived from the delivered first/full-response
times, including intervening network delays, rather than a measured GPU decode
step time. The selected queue, fluid, SimGrid or ns-3 backend determines actual
hop transmission completion; propagation is added separately.

For `arrival_time_mode: "after_dependency_completion"`, the effective arrival
time is the predecessor's actual response completion plus `inter_turn_gap_s`.
Latency uses this effective arrival, rather than the input's placeholder arrival
time. A failed predecessor produces an uncompleted dependent request instead of
an artificial successful turn.

## Output locations and failure populations

| Output | Meaning |
| --- | --- |
| `inference_metrics.ttft_s`, `.tpot_s`, `.e2e_s` | Completed-request distributions defined above |
| `request_records[].timing` | Per-request latency and stage-level audit timings |
| `request_records[].outcome` | `completed` and `uncompleted_reason` |
| `request_summary` | Total/completed/uncompleted counts and reason breakdown |
| `state_summary.event_log` | Chronological execution evidence |
| `provenance` | Engine, transport and response-model assumptions |

Incomplete requests have null TTFT, TPOT and E2E even when partial stage timings
or an observed first-token event are available for diagnosis. Successful
recovery includes actual waiting, rerouting and retransmission in delivery time;
it does not restart the latency clock. Interrupted hops discard their incomplete
payload while completed upstream hops remain delivered. Loss of the only
buffer-owning node prevents that message from recovering.

Computation failure currently has no checkpoint/automatic continuation. A
failed active cell member produces `compute_member_failed_no_checkpoint`; loss
of required cell connectivity can produce
`compute_cell_disconnected_during_service`. These failures must be reported in
the completion population alongside latency percentiles. A lower p95 among a
smaller successful population alone does not demonstrate improved service.

## Implementation and regression evidence

- [Aggregation and percentile calculation](../../../src/orbitserve/runtime/result.py).
- [Request record and TPOT calculation](../../../src/orbitserve/runtime/metrics/records.py).
- [Chronological execution and response delivery](../../../src/orbitserve/runtime/engine/events.py).
- [Result contract tests](../../../tests/unit/runtime/test_runtime_result_contract.py),
  [event-order tests](../../../tests/unit/runtime/test_runtime_event_order.py) and
  [network recovery tests](../../../tests/integration/fixed_design/test_network_recovery.py).
