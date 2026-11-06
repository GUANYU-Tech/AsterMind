# Serving model profile constants

Implementation reviewed: 2026-11-03. The frozen coefficients below remain the
current repository defaults; the physical-network iteration changes how they
are combined with physical cells and Runtime execution, not their values.

OrbitServe freezes the following model-level analytical constants in
`src/orbitserve/serving/model_profiles.py`. They are modeling inputs, not
measured end-to-end throughput. Two-dimensional per-request rates are produced only after combining them
with the selected single-satellite hardware profile, L2 formation, parallelism,
efficiency factors, communication model, prompt/context length, and static-batch concurrency.

| Canonical model | Total parameters | Active parameters/token | Layers | Hidden size | Prefill A (FLOPs/token) | Prefill B (FLOPs/token²) | KV bytes/token | Max context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama3.3-70b` | 70B | 70B | 80 | 8192 | 136,902,082,560 | 2,621,440 | 327,680 | 131,072 |
| `deepseek-v3` | 671B | 37B | 61 | 7168 | 71,182,974,976 | 4,997,120 | 70,272 | 128,000 |
| `kimi-k3` | 2.8T | 104B | 93 | 7168 | 208,000,000,000 | 1,474,560 | 27,648 | 1,048,576 |

The full-prompt function is `F_prefill(P) = A * P + B * P²`. Weight and
activation storage default to BF16 (2 bytes). DeepSeek V3 KV stores the MLA
compressed latent and RoPE key. Kimi K3's quadratic term covers its 24 full MLA
layers; its fixed KDA recurrent state and vision tower are excluded from the
per-token, text-only approximation.

These coefficients must be treated as explicit bounded assumptions in paper
experiments. Hardware efficiency and formation link parameters should be
swept independently rather than modifying the frozen model dictionary.

## Physical cell and hardware contract

`serving.formation.member_count` counts physical compute satellites per cell.
Orbit expands and propagates those members; Runtime schedules the cell through
its logical compute target. Every physical cell in a fixed-design/DSE evaluation
must match this member count. The parallel node count is
`pp * lcm(tp, ep)`, and must equal `member_count`; TP and EP are not blindly
multiplied together. Counts must be positive integers, excluding booleans and
fractional values.

Architecture supplies the resolved per-satellite hardware. The evaluator checks
physical terminal counts and configured link-rate ceilings against the relevant
compute or communication hardware profile. Serving's static formation bandwidth
and one-way propagation delay describe its analytical collective model; they do
not replace the Runtime graph's per-edge capacities, visibility or acquisition.
Analytical `ring`, `switch` and `fully_connected` are Serving choices; the
physical cell graph currently supports ring or fully connected neighbors.

Weights must fit per-member fast memory under the uniform partition assumption.
KV capacity uses the cell's configured storage capacity divided by the model's
KV bytes/token. Model context limits and the configured table grids still bound
the requests that can be evaluated. Capacity estimates are modeling inputs,
separate from the Runtime cache's actual session entries and eviction history.

## Rate tables and Runtime use

Serving produces per-request `prefill_rate_table` rows indexed by
`prompt_length` and `concurrency`, and `decode_rate_table` rows indexed by
`context_length` and `concurrency`. It does not reduce these tables to one
workload-averaged tokens/s scalar. `max_concurrency` defaults to 1.

At admission, Runtime selects the exact concurrency slice and keeps it fixed
through that request's prefill/decode service. Existing requests are not re-rated
when another request enters or leaves. A slot remains occupied until decode
completion; requests wait FIFO when all slots are occupied.

Prefill interpolates cumulative time `T_prefill(P,C)`, not rate. For a reusable
prefix of `K` tokens, suffix work is
`T_prefill(P,C) - T_prefill(K,C)`. Decode integrates interpolated per-token
time over growing context at the fixed admission concurrency. Out-of-range
lengths and missing concurrency slices are rejected instead of extrapolated.

Internal collective time is embedded in the analytical Serving tables. Runtime
checks actual cell connectivity and member availability without simulating each
TP/PP/EP operator as another network transfer. A connected cell with a changed
link bandwidth is not a complete dynamic collective-performance model.

## Optional AsterMind measurement tables

`serving.external_profile` accepts `orbitserve.astermind_measurements.v1`, an
explicit table import, rather than invoking or building AsterMind. The document
must match the resolved `model_name`, `hardware`, `formation` and `parallelism`,
with scope `phase_latency_including_cell_communication`.

Samples cover both phases at every configured concurrency. Each contains phase,
length, concurrency and a successful, versioned AsterMind result with finite
positive `metrics.total_latency_s`. Decode samples represent exactly one
generated token (`decode_tokens: 1`). Prefill rate is length/latency; decode
rate is 1/latency. Samples must cover the lengths used by the experiment.

Measured phase latency already includes internal cell communication, so the
analytical communication term is not added again. The measurement document's
SHA-256 is recorded in provenance. A single end-to-end latency number cannot
replace these phase/concurrency tables, and a static measured table does not
represent arbitrary future topology changes or checkpoint recovery.

## Implementation and regression evidence

- [Frozen coefficients](../../../src/orbitserve/serving/model_profiles.py),
  [Serving validation and estimates](../../../src/orbitserve/serving/capacity.py) and
  [measurement import](../../../src/orbitserve/serving/external.py).
- [Physical/hardware consistency checks](../../../src/experiments/design_evaluation.py)
  and [Runtime table lookup](../../../src/orbitserve/runtime/setup/serving_time.py).
- [Serving-time tests](../../../tests/unit/runtime/test_serving_time_tables.py),
  [measurement tests](../../../tests/unit/serving/test_external_measurements.py) and
  [input validation tests](../../../tests/unit/serving/test_input_validation.py).
