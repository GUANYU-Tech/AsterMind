# Communication execution

The sole runtime is the source RuntimeCommunication state machine, planner and
native progressive-filling allocator. See `CELL_INTEGRATION.md` for the exact
host boundary and `../README.md` for configuration.

For each operator, Comm resolves physical scope and source policy. Supported
single-SU or complete cross-SU collectives can be predicted as one whole star-local
operation; `IntraSatelliteRoofline` uses the original native predictor context.
Unsupported whole-operation shapes use the source collective phase/P2P expansion
or raise the original explicit unsupported error. Variable partition and traffic
matrix byte semantics and phase barriers remain source-defined.

Cross-star P2P uses cached topology paths, then source path/protocol budgets
and one shared allocator. Current physical links, RF pools and terminal gates are
projected from the existing network. Every selected edge must be simultaneously
available. A route that requires intermediate storage waits/fails instead of
silently simulating store-and-forward. Routing does not forecast whether this
payload can finish before a future contact closes; Comm handles actual loss,
waiting, retry and deadlines on the authoritative timeline.

`routing.policy` in SimulationConfig selects `min_hops` (default) or
`min_propagation` (minimum summed snapshot distance). Hop count uses the existing
directed native device graph, including internal resource gates; physical-only
route probes use physical links. No implicit hop-count granularity change is made.
Positive link capacity is an availability condition, never a routing weight.
Message size and current traffic are absent from the path selector and cache key.
Routing and Comm share one physical snapshot and adjacency table per valid
interval. Endpoint restrictions are checked during search rather than by copying
the whole graph for each new endpoint pair. Offline distance reads use numeric
columns without expanding full link-attribute records.

The engine owns refresh timing; there is no routing interval setting. Paths
expire at orbital geometry samples. Between samples, observed path loss or a
zero-capacity direction invalidates affected entries immediately; positive
capacity changes do not. New connectivity releases cached unreachable results.
Valid in-flight attempts remain on their original path across table refreshes.
The table is bounded and computed on demand, not an all-pairs precomputation.

The ordinary route API returns path/resource descriptions, without a completion
estimate. `cell_selection` defaults to `least_loaded`: assigned input/computing
request count first, summed route distance second. `nearest` reverses these two
criteria. Both inspect only candidate Cell load, never active link traffic.
Only explicit `estimated_arrival` opts into `estimation_context()` once per Cell
decision and `estimate_transfer()` per reachable candidate, using estimated
upload time as the tie-break after assigned-request count. A pure allocator call
adds the candidate to observed active demands without mutating running flows.
This freezes current competition; it predicts neither future arrivals/contact
changes nor actual completion. The isolated estimate API is also available to
explicit callers by omitting the context. Neither path invokes offline routing.
`cell_estimate_calls`, `cell_estimate_wall_s`, `cell_estimate_context_calls` and
`cell_estimate_context_wall_s` account for this work separately from `route_wall_s`.
Network reports also include path-table hits, searches, invalidations and builds.
`snapshot_wall_s` measures shared physical snapshot construction;
`search_wall_s` measures path searches. Comm's `network_frame_wall_s` includes
snapshot acquisition and resource-frame construction. These timings overlap and
must not be added together. The existing `route_wall_s` covers execution route
queries after resource-frame refresh, not total topology processing.
The legacy `NetworkState.transfer()` remains an offline forecast API, never the
production communication route selector.

Custom policies are trusted Python callbacks registered before creating the
runtime, in each process that performs routing:

```python
from astermind.engine.policies.routing import register_routing_policy, min_hops

def my_policy(graph, source, target):
    # Return node IDs from immutable graph.adjacency, or None if unreachable.
    return min_hops(graph, source, target)

register_routing_policy("my-policy", my_policy)
# SimulationConfig input: {"routing": {"policy": "my-policy"}, ...}
```

Unknown names fail before simulation. Returned paths must be simple, contiguous,
currently available and obey endpoint/ground/device forwarding restrictions.
JSON contains policy names only; it never imports or executes user-supplied code.
Custom policy registration must also be installed in a frontend worker's startup
when using that worker; the policy textbox alone does not install Python code.

Source star-local predictions do not contend globally or reprice in flight.
External capacity changes settle old rates before reallocating remaining bytes;
path loss retries an entire attempt and records discarded bytes. Explicit
source deadlines, cancellation and upstream failure isolation remain in force.
Native compute COMM estimates are not charged in addition to the actual Comm
completion callback. No old backend or legacy ProtoBridge wrapper remains active.

Reports contain policy/model selection, assumptions, partitions/rounds, raw
communication events and allocator statistics. `trace_service=true` additionally
records each allocation's resource capacities, flow demands and rates, enabling
conservation audit; this can be large for WideEP. Exact ticks are authoritative.
The frontend exposes routing policy selection and topology-only route comparison.
