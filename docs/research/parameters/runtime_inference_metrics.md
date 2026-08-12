# Runtime inference metric contract

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
