# Physical L2 compute-cell contract

Implementation reviewed: 2026-11-03. This supersedes develop's earlier
logical-node-only iteration. Physical compute-cell members now appear separately
in Orbit and the network graph, while service placement and concurrency remain
attached to one logical compute target per cell.

## Physical and logical identities

| Object | Current responsibility |
| --- | --- |
| Communication satellite | Physical relay with its own orbit, terminals, links and energy state |
| Compute satellite | Physical cell member with its own orbit, terminals, faults and energy state |
| Compute cell | Members sharing `cell_id`, with exactly one `compute_target` for logical service placement |
| Cell ingress/egress | Members allowed to establish external connections; independent of the compute target |
| Ground node | Gateway or user endpoint with configured RF limits |

Communication shells and compute formations have independent Walker/TLE inputs.
SGP4 materializes and propagates physical members instead of one orbit point per
logical cell. Communication shells use fixed grid/honeycomb neighbors and remain
independent of one another. Compute cells use fixed ring/fully connected
neighbors. A configured neighbor is not guaranteed UP: physical visibility,
terminal allocation, acquisition delays and link budgets still apply.

`cell_egress_policies` supports `single`, `multi` and `free` external-member
selection. Logical placement does not independently schedule a model on every
member or automatically change TP/PP/EP after a member fails. Compute work/energy
is shared across physical members. Participant energy checks include non-target
members and relay satellites.

## Serving input and validation

This is the Serving fragment of an experiment. It additionally requires physical
network, hardware and Workload inputs; this JSON alone is not an executable scene.

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

Serving `topology` accepts `ring`, `switch` or `fully_connected` as analytical
collective models. Its bandwidth and one-way propagation delay describe that
static model. Runtime's physical topology and directional budgets are configured
separately in `physical_network`; the physical evaluator validates terminal counts
and configured rate ceilings against Architecture hardware.

Parallel node count is `pp * lcm(tp, ep)` and must equal
`formation.member_count`. Every physical compute cell must contain that many
members. The former `parallelism.satellite_count` input does not exist. Counts
must be positive integers, excluding booleans and fractional values.

Serving has no batch-size input or separate prefill/decode placement mode. It
models homogeneous static batches with colocated prefill/decode. Canonical model
names are `llama3.3-70b`, `deepseek-v3` and `kimi-k3`; frozen model constants and
supported aliases are in `serving/model_profiles.py`.

## Serving calculation and Runtime use

For prompt length `P`, full-prompt work is `F_prefill(P) = A * P + B * P^2`.
At concurrency `C`, prefill scales total FLOPs and activation communication by
`C`. Decode reads active weights once per step and scales KV traffic and
activation communication by `C`. Analytical topology-aware TP/PP/EP communication
is embedded in the resulting per-request rates.

Serving emits `prefill_rate_table` indexed by prompt length/concurrency and
`decode_rate_table` indexed by context length/concurrency, plus KV bytes/token,
KV capacity and model context limits. Runtime selects the exact admission
concurrency slice and interpolates cumulative prefill time. A reusable prefix
`K` costs `T_prefill(P,C) - T_prefill(K,C)`. Decode integrates context-dependent
per-token time with `C` fixed. A request holds a slot from prefill start through
decode completion; excess requests wait FIFO. Existing requests are not re-rated
when concurrency changes.

Runtime gates computation on actual member availability and cell connectivity.
It does not execute a per-operator collective schedule. Optional
`serving.external_profile` imports matched AsterMind phase/concurrency tables
with internal communication already included, avoiding analytical double
counting. See [Serving model profiles](../parameters/serving_model_profiles.md)
for constants, measurement validation and the static-table boundary.

Runtime reports delivered-response TTFT, TPOT and E2E for completed requests;
there is no Runtime SLO evaluation. See the
[metric contract](../parameters/runtime_inference_metrics.md) for failure
populations, streaming assumptions and output locations.

## Physical datasets and chronological execution

Physical preparation publishes an immutable dataset containing the materialized
input, SGP4 trace, sampled positions, directional attributes and initial causal
allocation timeline. `physical_dataset.json` is the completion marker. Runtime
verifies identity, physics version and artifact hashes before reuse. Core data
is sufficient for execution; expanded graph/playback exports are optional. See
[physical data layout](../../../src/experiments/PHYSICAL_DATA.md).

Runtime holds physical attributes at the latest sample but executes requests,
transmissions, arrivals, acquisition and computation at their event times.
Workload, routing and Runtime fault settings can reuse the same physical dataset.
Fault realizations do not alter the baseline or reach online route forecasts
before observation. Explicit stale, corrupt or incompatible dataset references
fail instead of silently regenerating physics.

Queue, fluid, native SimGrid and native ns-3 provide alternative hop transmission
and competition models. They retain whole-message store-and-forward between hops
and separate propagation from serialization. The ns-3 adapter uses packet
serialization/admission rather than an end-to-end TCP stack.

## Fault scope and recovery boundary

Random node faults cover communication and compute satellites, excluding ground
nodes. Random link faults cover every prepared candidate type: communication
ISLs, cell ISLs, cross-layer links, inter-cell links and ground access links when
present. Node/link processes have independent seeds and parameters. There are no
separate random parameter groups per satellite role or link kind. Scheduled or
manual faults can also target an existing ground node.

Default `fault_timing: "continuous"` accepts fractional seconds independently of
physical sampling. Random processes with `time_basis: "seconds"` use exponential
or Weibull healthy periods and fixed, uniform, exponential or lognormal outage
durations; fixed-ratio interval scenarios are also supported. Explicit boundary
timing and legacy step-based inputs retain their own semantics. Sampled repairs
beyond the horizon remain censored instead of shortened or redrawn.

| Failure | Implemented effect |
| --- | --- |
| Satellite node down | Removes incident links from the usable graph and releases/reallocates terminals |
| Active compute-cell member down | Terminates affected computation with `compute_member_failed_no_checkpoint` |
| Cell-internal link down | Affects connectivity; a disconnected active cell fails with `compute_cell_disconnected_during_service` |
| Transfer interrupted | Cancels the incomplete hop; completed upstream hops and a healthy sender's buffer remain |
| Only buffer-owning node down | Fails that message because its retained forwarding buffer is lost |

With `failure_policy: "retry_next_sample"`, Runtime attempts recovery at the
observed change and waits for further observed changes if no route exists;
the name does not force continuous-mode recovery to wait for a physical sample.
`max_transfer_retries` bounds attempts. Default `failure_policy: "fail"` fails
interrupted transfers. Recovered links must satisfy geometry/terminal constraints
and acquisition delay; unaffected contacts retain their acquisition clocks.

The optional in-process controller/local console supports manual node/link
failure and restoration, persistent administrative disconnection and physically
validated new satellite links. Commands run on the engine thread. Clearing a
manual cause does not clear a simultaneous scheduled or random cause.

There is no computation checkpoint, automatic model migration/continuation,
dynamic cell-parallelism shrink or per-operator collective scheduling. Restoring
a node/link does not resume a request already marked failed.

## Runtime KV cache decision

`runtime_strategy.kv_cache_copy_enabled` controls optional copy. Reuse requires
the same non-null `session_id` and is capped by `context_tokens_before_turn`;
current-turn `new_user_tokens` never become a reusable prefix.

For each reachable source with that session, Runtime compares
`T_copy(source,target) + T_prefill(P,C_compare) - T_prefill(S_source,C_compare)`
with local `T_prefill(P,C_compare) - T_prefill(L_target,C_compare)`.
`C_compare` is the same target-concurrency snapshot for every source. Copy is
selected only when faster, using the lowest copy-plus-compute time; this
comparison excludes target queue waiting. Copy retains the source entry.
Actual copies, reuse, evictions and transfer paths remain in the audit outputs.

## Cost aggregation and iteration boundary

`compute_cost = compute_cell_count * formation_member_count * compute_satellite_cost`

`total_cost = compute_cost + communication_satellite_count * communication_satellite_cost`

Compute and communication may use distinct Architecture profiles. A satellite's
cost already includes its configured terminals, so no second formation-link or
terminal surcharge is added. Fixed designs and DSE share the materialized-design
evaluator. Geometric screening is an access upper bound, not a substitute for
actual Runtime latency or completion results.

Dynamic physical networking does not establish long-term formation stability,
measured hardware degradation or a flight-qualified recovery mechanism. Model
constants, link assumptions and imported measurements remain explicit experiment
inputs.

## Implementation and regression evidence

- [Physical materialization](../../../src/experiments/network_inputs.py),
  [candidate links](../../../src/orbitserve/runtime/engine/topology.py) and
  [validation/costing](../../../src/experiments/design_evaluation.py).
- [Fault generation](../../../src/orbitserve/runtime/setup/faults.py),
  [event execution](../../../src/orbitserve/runtime/engine/events.py) and
  [manual interventions](../../../src/orbitserve/runtime/state/interventions.py).
- [Cell service/disconnect tests](../../../tests/unit/runtime/test_runtime_event_order.py),
  [continuous fault tests](../../../tests/unit/runtime/test_continuous_fault_models.py),
  [dataset tests](../../../tests/integration/fixed_design/test_physical_dataset.py)
  and [cell network exchange tests](../../../tests/unit/runtime/test_cell_network_exchange.py).
