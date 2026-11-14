# Native scene timeline engines

Native scene configurations expose two validated execution combinations:

| `timeline_mode` | `backend` | Time and event authority |
| --- | --- | --- |
| `orbitserve` (default) | `flowsim` | OrbitServe global event heap, with hosted FlowSim byte service |
| `simgrid` | `simgrid` | SimGrid engine and multiplexed native timed waits, with SimGrid communication activities |

Existing documents without `timeline_mode` retain their OrbitServe behavior.
Legacy `ns3` documents retain the OrbitServe timeline. Mixed combinations such as
`timeline_mode=simgrid, backend=flowsim` are rejected instead of silently changed.

The scene editor's engine selector changes both fields when generating a GPU
deployment. Changing it invalidates preflight and requires deployment generation
again. Editing JSON is supported; inconsistent modes fail validation. Results,
live state and replay record the actual backend and timeline authority.

## Communication scope remains independent

All three scope choices remain available on both engines: explicit simulation
of every communication, ProtoBridge for satellite-local communication, and
ProtoBridge for Cell-local communication. GPU/rank placement, kernel dependency
graphs, Scheduler state, KV transactions and fabric reservations remain intact.
Cell-external transmission always uses the selected communication engine.

For compatibility, the historical `communication_mode=flowsim` value denotes
the all-communication scope, including when `backend=simgrid`. It does not send
SimGrid-mode traffic through FlowSim. The UI names the actual selected backend.
The fastest Cell-local ProtoBridge scope remains the editor default.

## SimGrid event ownership

`NativeGlobalEvents` creates a native SimGrid sleeping actor for each scheduled
domain deadline. Communication completion watchers and timer actors wake the
controller. SimGrid decides the next physical clock boundary. Same-time domain
events are dispatched by time, priority and insertion sequence, including newly
scheduled same-time events. Python retains event payloads/callbacks in a registry;
there is no Python future-time heap driving this mode.

GPU completion, host launch, ProtoBridge barriers, request arrival, propagation,
topology sampling, faults, retries, KV callbacks and interactive control deadlines
all enter that scheduler. AsterMind still predicts GPU service and owns GPU/rank
resource policy; those predictions are not replaced by SimGrid's generic CPU
FLOP model. Real dependency imports/Cell construction are startup wall time, not
invented simulated model-loading time.

The worker multiplexes domain deadlines onto the controller's native timed
wait. `ActivitySet.wait_any_for` wakes on the first communication completion or
the next deadline; `sleep_until` handles idle intervals. Future events and flows
do not each allocate a Python/OS actor thread. Priority and sequence resolve
same-time callbacks after SimGrid reaches their deadline.

Native timed waits may wake a floating ULP after a requested deadline.
Callbacks execute at the actual `Engine.clock`, never the older requested label.
New link registration rebuilds the sealed SimGrid platform at the current native
time, restoring remaining bytes and all pending domain deadlines. Cancelled or
superseded events remove their pending deadlines. Shutdown cancels activities
and reaps the worker.

The public Runtime SimGrid backend uses the same native global scheduler. Legacy
float-nanosecond Cell callbacks can round their representation of the current
instant slightly backward; only that identical public "now" value is aligned,
within its one-ULP conversion bound. Explicit integer-tick deadlines and genuine
past submissions are never relaxed. Successor-callback exceptions propagate even
when the predecessor transfer has already completed.

## Clock and service evidence

FlowSim retains `orbitserve.clock.v1` integer attoseconds. SimGrid declares
`orbitserve.simgrid_clock.v1`, a float64-seconds native clock with a configured
1e-15 second timing tolerance. Integer timestamp fields in its output are record
projections, not a claim that SimGrid executes integer attoseconds.

SimGrid service records preserve the original native interval endpoints,
native byte-progress rates and an explicit bound from floating byte arithmetic
and the configured timing tolerance. Capacity and duration audits apply that
contract and reject violations outside it. Actual native doubles must survive
round trips unchanged; projected integer labels cannot move the native clock.
FlowSim's exact service checks remain unchanged.

KV migration records stamp request/start, each partition's terminal time, the
network barrier and target-write completion. Both audits compare those stamps
directly; integer nanosecond display fields cannot falsely move a partition past
its own barrier. Older files without migration stamps are compared only within
their shared upward-rounded nanosecond representation.

## Modeling boundaries

This mode uses CM02, zero native link latency, no reverse ACK crosstraffic and
equal sharing weights. OrbitServe's domain callbacks account for propagation
and hop eligibility on the SimGrid clock. It remains flow-level whole-message
hop forwarding. Finite forwarding buffers, MTU packetization, TCP packet loss,
and hardware-accuracy calibration are not introduced by changing the scheduler.

## Validation matrix

The implementation is checked at these boundaries:

1. Configuration defaults, strict mode/backend combinations and JSON round trips.
2. Native timer order, same-time priorities, dynamic callbacks, cancellation,
   fractional deadlines, communication preemption and horizon termination.
3. Shared directed links, independent directions, satellite aggregate pools,
   new contenders, bandwidth changes, zero-capacity pauses and cancellation.
4. Native platform rebuilding with pending timers and partially served flows.
5. Device/GPU callbacks and all three communication scopes under SimGrid.
6. Native GPU TP2/PP2 execution on both timelines and all communication scopes.
7. Fault recovery, request delivery, KV migration and prefix ownership.
8. Byte conservation, resource capacity, dependency barriers, kernel duration,
   explicit float-clock evidence and injected-invalid-record rejection.
9. Live control, scene save/reload, result selection and correct engine labels.
10. Existing unit, integration, smoke and slow regressions, plus lint and browser
    verification. Run results and the function-by-function review are recorded
    separately; this matrix does not assert that an unexecuted check passed.
