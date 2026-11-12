# Communication scope in native scenes

GPU compute kernels always use the existing AsterMind/M-L compute model. The
communication method replaces only COMM solving; it does not replace the Cell
with a throughput table or change its rank placement, operator dependencies,
Scheduler, KV ownership, or compute-resource reservations.

| `communication_mode` | GPUs on one satellite | Satellites in one Cell | Outside the Cell |
| --- | --- | --- | --- |
| `flowsim` | FlowSim | FlowSim | FlowSim |
| `protobridge_satellite` | ProtoBridge | FlowSim | FlowSim |
| `protobridge_cell` | ProtoBridge | ProtoBridge | FlowSim |

New `orbitserve.native_cells.v1` scene documents and the editor default to
`protobridge_cell`, reducing communication events most aggressively. Historical
numeric-schema-v1 files retain their previous FlowSim default when this field is
absent. Explicit selections are never silently changed. External KV migration,
ground traffic and constellation forwarding always use FlowSim in the editor.

## Execution and evidence

`astermind/engine/device_communication.py` calls AsterMind's actual
`src.kernel.predictors.protobridge.whitebox_surrogate_v2.TheoryDerivedSurrogate`.
Homogeneous ring all-reduce/all-gather and PP send/receive use collective-level
predictions when every participant is in scope. Mixed-satellite groups in
satellite mode and ragged collectives retain the existing phase expansion;
eligible point-to-point pieces use ProtoBridge individually.

Predictions use current directed, available device routes restricted to the
selected satellite/Cell, including GPU/switch gates, orbital link bandwidth and
propagation. They do not reuse a satellite-local NVLink profile for orbital
communication. Ring bottlenecks conservatively account for shared route links.
The default startup cost is an explicit zero-microsecond assumption, configurable
through `protobridge_startup_us`; this is not a new hardware calibration.

Each surrogate operation reserves the participating satellite fabrics
exclusively, including all members of its Cell in Cell mode. Existing conflicting
FlowSim transfers drain first and subsequent conflicting transfers wait.
Unrelated Cells and external flows continue. This is a coarse contention
approximation: select full FlowSim when per-link sharing or overlap is the
measurement of interest. It is not claimed to give the same modeled latency as
full FlowSim. Startup, credits, FEC and retry assumptions are recorded with the
predictor inputs; protocol credit/FEC corrections are disabled in this adapter.

Completion is scheduled on the same global Runtime timeline. Dependencies cannot
start before the completion barrier. A changed link budget or fault invalidates
an unfinished prediction and restarts it with current parameters; route recovery
obeys the configured retry policy. Partial-progress restart is conservative and
does not model packet-level checkpoint recovery.

Reports retain each prediction's inputs, source hash, queue/start/completion
times, restart attempts, resource policy and method. Replay distinguishes
ProtoBridge communication barriers from FlowSim phases. Native COMM reference
duration is never charged a second time. Wall time and simulated time are separate
measurements; fewer events do not guarantee the same speedup for every workload.

## Scene input validation

`astermind.builder.scene` projects a verified physical dataset into explicit GPUs,
satellite fabrics and physical terminal links. The physical allocator still owns
terminal concurrency; adding GPUs cannot create extra active orbital links.
The local editor validates native schema, finite values, placement, request
endpoints, physical-dataset correspondence, migration references and native
model/deployment support before saving or launching. Manual JSON remains usable;
invalid drafts are retained for correction rather than silently clamped.

The editor initially exposes Qwen3-8B BF16 / A800 with TP1/PP1, TP2/PP1 and
TP2/PP2, plus Qwen3-4B with PP1. Qwen3-4B's tied embeddings require a separate
cross-stage storage contract for PP2 and are rejected by the native adapter.
Advanced native configurations must pass the existing
native adapter's model and hardware checks. Explicit KV block count is a declared
simulation capacity, not hardware-derived HBM sizing; a bound memory-policy file
is required for the latter. Legacy rate-table scenes require explicit conversion
and remain unchanged on disk.

## Validation

Run with the repository's native dependencies installed:

```bash
python -m unittest tests.unit.runtime.test_scoped_protobridge
python -m unittest tests.external.native_cells.test_communication_modes
```

The native suite runs the real pinned Scheduler and kernels in all three modes,
checks dependency/communication audits, verifies progressively fewer FlowSim
events in a TP2/PP2 scenario, and verifies cross-Cell KV migration remains on
FlowSim. The generated comparison report records measured startup and event-loop
wall times; it is not an accuracy benchmark.

## Unified execution clock

The hosted FlowSim kernel, global event heap, GPU dependency release and
ProtoBridge completion use integer attoseconds (1 tick = 10^-18 seconds).
Python integers and the native unsigned 128-bit clock cross the binding as
decimal strings. Float seconds and legacy integer-nanosecond fields are display
projections; neither is converted back into an execution timestamp. GPU model
durations retain the native nanosecond resolution and are added to the exact
current tick. They do not round the absolute completion time to a nanosecond.

Positive model durations and serialization completion round upward to a tick.
Four bytes at 200 Gbit/s take exactly 160,000,000 ticks (0.16 ns). Resolution
does not imply corresponding physical model accuracy. A serialization completion
can be delayed by less than one tick; its residual service is clipped to the
remaining bytes. Propagation belongs to the host and follows the same clock.
Events use (tick, priority, sequence), and scheduling even one tick in the past
is rejected. A failure at a boundary precedes subsequent starts at that boundary;
already provided service remains accounted for.

FlowSim exports allocated rates directly and records sums for every directed
link and shared capacity pool. Service integrates the integer tick difference,
never a subtraction of large floating absolute times. The audits independently
check shared allocation sums, serviced bytes against rate × duration, per-hop
byte conservation, monotonic event time, and exact communication dependency
release. Failed invariants reject the run. Logical payload quantization remains
1e-6 bytes and is recorded in transport provenance.

The local native dependency must include `astermind/engine/adapters/flowsim_clock.patch` and
be rebuilt. An older extension fails closed instead of falling back to float
time. `third_party/` and compiled binaries remain local under company rules.

New replay records include decimal tick strings; the browser uses BigInt when
grouping or stepping events. Animation and geometry use a seconds projection.
Historical float-clock runs keep their original evidence and are labelled as
old-clock records. Their ULP precision review is diagnostic, not proof that
their execution has been corrected; verification requires a new run.
