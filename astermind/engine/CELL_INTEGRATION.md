# Compute / communication boundary

Source: Communication_backend 9c08c9cad81df5a4fee5e606114753df8dab6c2d.
The native algorithms/configuration snapshot is copied unchanged; its 0.30 worker
launcher only preserves inherited PYTHONPATH to import host tick objects outside
the source checkout. First-party source
wrappers and Comm are moved into this branch's Engine package, with imports and
explicit boundary adapters recorded in compute/SOURCE.json.

1. Builder validates SimulationConfig, physical devices, rank/pool placements,
   source deployment restrictions, protocol contracts and optional SU metadata.
2. Simulation retains automatic Cell assignment. Upload arrival, predecessor
   compute completion and committed compatible KV gate admission exactly once.
3. Original ServingScheduler/PrefillDecodeCell produce batches, native KernelPlans
   and local compute predictions. RuntimeBatchExecutor preserves operator dependencies,
   host launch gaps and source failure callbacks on the integer host timeline.
4. Compute communication is submitted to the source RuntimeCommunication planner.
   Native COMM predictions are not charged again. Supported whole star-local
   collectives use the source surrogate/hierarchical/Roofline path. Other supported
   plans lower to phased P2P with the original barriers and byte counts.
5. TargetNetworkFrames projects current device/physical resources without importing
   a route solver. TargetRoutes delegates ordered path selection to existing
   NetworkState.transfer(commit=False); active Comm loads feed original forecasting.
   All selected edges must currently exist. Device terminal gates collapse to one
   payload pool demand; physical shared RF coefficients retain source semantics.
6. Source progressive filling shares external resources; exact relative intervals
   settle bytes before updates. Capacity changes retain progress. Broken paths
   retry the whole transfer, discard partial bytes and respect timeout/retry bounds.
   No store-and-forward, intra global contention, or intra mid-flight repricing is added.
7. Transfer, compute, PD and migration callbacks stay on the existing attosecond
   heap. Inclusive host horizon drains terminal callbacks before cleanup; explicit
   communication deadlines retain upstream priority-zero behavior.
8. Raw reports preserve real Scheduler pools/replicas, source runtime inputs,
   kernel predictions/dependencies, communication trace/stats, KV and failure state.
   Audit checks exact causal ticks, native compute duration, capacity conservation
   and commit barriers. Historical replay support does not fabricate new-mode data.

The physical orbital/link Builder, route/cache algorithms, request assignment,
workload inputs and viz source are not transplanted. Old transports and old
compute wrappers are removed. Source contracts that cannot represent a requested
layout or missing hardware data fail explicitly.
