# Cell integration on the dynamic Runtime

The primary runnable integration is documented in
[astermind/README.md](../README.md). Both the standard
`astermind.experiments.cli` (`orbitserve.native_cells.v1`) and `python -m astermind`
use the real first-party Cell adapters, external Mid-Layer execution and official
vLLM Scheduler. This replaces the colleague's standalone network pump with the
existing dynamic `_EventEngine`.

## Device execution path

`Simulation` builds an explicit `PhysicalTopology`, loads a hash-verified physical
dataset (or prepares it with the existing network experiment), and creates
`DeviceRuntimeHost`. Its `DeviceNetworkState` projects GPU ports, switch/terminal
capacity gates and external device links onto one authoritative `NetworkState`.
Ranks on one satellite remain distinct GPUs. GPU inputs cannot forward traffic;
ground resources cannot relay between unrelated satellites. External physical
links cannot be duplicated to multiply capacity. Device port limits and physical
terminal allocation must agree.

Local device resources and external hops share the selected real FlowSim or ns-3
transport. External hops retain live directional link budgets and satellite RF
pools. Physical acquisition, faults and capacity changes remain causal. In ns-3,
resource gates consume shared service without generating fictitious PPP packets.

`GlobalTimeline(host=engine)` reads `engine.now` and schedules all native callbacks
on the existing event heap. Physical updates and hop completions precede Cell
callbacks at equal times. It cannot independently run a second clock.
`bind_batch_executor()` retains native kernel timing, DAG dependencies and
collective lowering; native COMM predictions are reference values, with zero
additional communication charge. Token-ID distribution completes before the
batch starts. `NativeStepExecution` supplies the actual scheduled token count.

The native DAG uses integer nanoseconds. `DeviceCommunicationRuntime` notifies it
at the ceiling of actual physical completion time, so dependencies are never
released early. `network.flows` and the host log preserve continuous physical
completion times; executor transfer records use the delivered integer timestamp.
The original `RuntimeCellNetwork` member adapter retains fractional ns callbacks.

## Storage and accuracy boundaries

`SessionHistoryStore` owns declared linear histories. Native vLLM owns admission,
preemption, KV blocks and prefix reuse. `NetworkKVMigration` pauses at native
batch boundaries, reserves target storage, reads and transmits every rank's KV,
waits for all deliveries and target writes, and only then commits. Move releases
source blocks after commit; failure rolls back target reservations and source
protection. Placement changes only after successful completion. Different TP/PP
layouts require redistribution and are rejected.

Explicit `num_kv_blocks` is a debug pool, not an inferred HBM capacity. A bound
`memory_policy_file` instead uses native per-rank weight residency, runtime
reserves and the limiting rank's available HBM. Hardware/model hashes and runtime
parameters must match. See the setup guide for the stale A800 policy found in
the external snapshot; its measured values are not silently rebound.

DP/SP and TP/PP redistribution, PD/AF, recovery of failed active compute, and
native kernel energy are outside this entry point. Link faults use Runtime's
configured fail/retry policy. Energy is reported as `not_evaluated`. Tested
operator/communication semantics do not establish hardware prediction accuracy.

## Shared adapters

The old `CellRuntime`/`CellDeployment`/`AsterMindRuntime` wrappers and their
`run_simulation` request entry point are removed. SimulationConfig and Simulation
own Native request execution. RuntimeCellNetwork, CellTransferExecutor,
NetworkKVMigration, bind_batch_executor and native_cell_factory remain available.

`RuntimeNetworkFrames`, `FrameRouter` and `CommunicationExecutor.for_runtime()`
remain additive compatibility APIs and reuse current causal physical state.
`Ns3PointToPointDriver` is a standalone sequential utility; hosted simulations
use the shared transport. Required backend capabilities are checked at startup.

## Merge and validation

Keep this branch's dynamic Runtime, ns-3 protocol, traces/counters, directional RF
pools and existing public exports when resolving overlapping files. The actual
first-party native adapters now live in `astermind/engine/adapters/`,
with configuration and topology in `astermind/builder/`; do not reintroduce the colleague's static physical network pump or
ISLSim envelope. Third-party sources/calibration, binaries, local packaging aids
and validation outputs remain ignored. Ignoring an already tracked upstream
snapshot does not remove it from a merge: exclude it explicitly during review.

```bash
python -m unittest tests.unit.runtime.test_device_network \
  tests.unit.runtime.test_cell_contracts tests.unit.runtime.test_cell_migration \
  tests.integration.fixed_design.test_cell_shared_runtime
python -m unittest discover -s tests/external/native_cells -v
python -m unittest discover -s tests -t .
```

The explicit native suite uses real vLLM/Mid-Layer, with no Cell/storage mocks:
cross-satellite TP2/PP2 through an outage, same-satellite TP2 plus two-rank KV move
on FlowSim/ns-3, migration rollback, heterogeneous Cells, HBM accounting and
MoE TP2/EP2. Missing dependencies fail this explicit suite. The small protocol
unit tests remain useful for edge cases; they are not scheduler validation.
