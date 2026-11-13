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

Cross-star P2P uses the existing route policy, then source path/protocol budgets
and one shared allocator. Current physical links, RF pools and terminal gates are
projected from the existing network. Every selected edge must be simultaneously
available. A route that requires intermediate storage waits/fails instead of
silently simulating store-and-forward. The retained isolated route forecast is
still conservative near a contact/horizon boundary.

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
The frontend is unchanged and new modes are raw-report only.
# Global routing configuration

`communication.routing_policy` selects `min_hops`, `min_propagation` (default), or
`earliest_arrival` for the shared Comm runtime. All Cells and both
model operators and system transfers use this setting; no per-Cell or per-task
override is exposed. This chooses physical paths, not request placement.
The existing priority-queue search and persistent cache are reused, without a
candidate-path count limit. ISL/GSL budgets remain separate, while their shared
link, RF pool and terminal resources are allocated in one Comm runtime.

`min_hops` counts only links crossing satellite/ground owners: each ISL or GSL
counts as one hop, while internal device links and capacity gates count as zero.
Equal-hop paths retain the existing predicted-arrival tie-breaker.
`min_propagation` minimizes summed
propagation delay among routes admitted by the existing temporal feasibility
checks. The hop metric is adapted from Dynamic-Link's original graph-edge count.
Both policies still use its heap search
and per-edge feasibility forecasts, so changing the default alone does not
guarantee faster simulation. No new allocator-based route trial is introduced.

The global policy also applies to intra-satellite paths. In the DS05 fixture,
minimum propagation selects a terminal detour rather than the faster SO link;
this fails the whole H-S topology check and correctly falls back to P2P.
The H-S model-boundary test gives that optional terminal detour a large delay
to test the supported SO path independently of the routing default.

## Earliest Arrival: whole-path selection score

For currently usable paths, the score in seconds is:

`sum(propagation_seconds) + payload_bytes * 8 / bottleneck_bps`

Each directed link/device gate uses `capacity_bps / (current_contenders + 1)`.
RF/shared pools use the same rule with their own contender count. The smallest
rate along the path is the bottleneck. Taking a minimum does not subtract a
shared pool twice. Device terminal gates therefore limit the path as well as
external links. Existing paths that traverse a pool multiple times still have
their exact resource demand enforced by Comm; this simple route score does not
solve that coupled allocation problem.

This is a snapshot forecast, not a trial insertion into FlowSim. It cannot
predict future arrivals or departures, max-min redistribution, or protocol
overhead. Actual task completion remains owned exclusively by Comm.

The existing sequential-hop feasibility check is retained for contacts,
capacity schedules, waiting and reservations. If waiting is necessary, its
explicit waiting time is added to the score. A future-only link uses its first
service-time capacity when its query-time capacity is zero. This can be
conservative near contact boundaries and does not guarantee the globally best
fluid completion time. Comm still requires simultaneous path availability.

The queue retains propagation/wait, bottleneck and feasibility-arrival labels;
a lower prefix score alone cannot prune a wider-bottleneck alternative.
For `earliest_arrival`, `transfer(commit=False)` returns the whole-path forecast
as its first result. Returned leg records retain the sequential feasibility
times. Legacy `commit=True` calendar reservations retain their sequential
arrival semantics; production Comm and Scheduler queries use `commit=False`.

## Path cache

- The cache is in-memory and belongs to the running network state. It does not
  read paths from earlier benchmark files. Forks copy the cache independently.
- Keys include source, destination, policy, waiting permission, link-kind
  restrictions and avoided nodes. Capacity is bounded at 16,384 entries.
- A hit reuses link identities, rechecks transfer feasibility and recalculates
  timing/score using current loads. It never reuses an old completion timestamp.
- Traffic changes or a newly available shortcut alone do not trigger rerouting.
  Cached-path physical changes invalidate the affected entry; failed replay
  triggers a fresh search. `recompute=True` performs a fresh query without
  replacing the persistent choice.
- In-flight FlowSim rate redistribution is independent of route caching.
  Congestion does not automatically move an active flow to a new path.
