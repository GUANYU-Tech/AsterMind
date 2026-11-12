# AsterMind V1.0 cleanup validation

Base: `feature/runtime-dynamic-links` at `cf96341e52ee2bdbb249ce54ffd3aef97aa9250e`.
Working branch: `version/AsterMind-V1.0`. Validation host: SeeSpace1.

## Scope

- The experiment CLI invokes the existing Native wrapper directly. SimulationConfig
  is the run configuration; numeric-v1 and native_cells.v1 communication defaults
  are unchanged. Removed orbit/link_parameters fields fail validation even when empty.
- Deleted DSE, old Analytical performance generation/interpolation, request admission,
  energy/SOC/DVFS execution, and their exclusive configuration/metrics/test modules.
- Deleted the old request API and its CellRuntime/CellDeployment/AsterMindRuntime
  adapters after explicit scope confirmation. Shared clock, transport, topology,
  fault/retry, Native Cell execution and KV migration remain in their original places.
- Independent architecture sizing/cost, workload and ground-baseline tools remain.
  The historical JSON output utility remains independent of the Native CLI; its
  DSE-specific output branches were removed. No dependency versions were changed.

## Environment

Python 3.12.16; pinned CPU Scheduler lock; official vLLM 0.31.0 Python source identity
verified by the existing Native audit (2771 files, manifest SHA-256
`1ddbe0ebcca48d6ed773a0ab92d55bbaaeffb883ddad2e97ac08d6fec5d446d5`).
Mid-Layer commit `6fe573479c37a931744a6442e7a4b3503eec02e6`;
ns3 3.44.post0; SimGrid 4.1 built with the existing pybind11 2.13.6 constraint.
SimGrid required explicit Python 3.12 CMake include/library paths and a local
LD_LIBRARY_PATH. Build settings, external source and binaries were not committed.
This CPU environment is not a full GPU/HTTP vLLM dependency validation.

## Results

| Check | Result |
| --- | --- |
| Pre-change real ns-3 Native baseline | 2 requests complete, 2717 kernels, audit passes |
| Batch 1 entry/config smoke | 8 tests; Native summary CLI matches baseline requests |
| Batch 2 DSE removal smoke | 20 tests; Native full CLI matches baseline requests |
| Batch 3 shared-event trim smoke | 15 tests; Native ns-3 requests still match baseline |
| Retained non-FlowSim unit/integration/smoke suite | 551 tests pass, no skips in selected set |
| Real Native integrations | 9 scenarios pass, described below |
| Review follow-up | 10 tests pass, including real ns-3/SimGrid transfers with physical regeneration forbidden |
| Installed sdist/wheel | Build succeeds; 108 packaged modules import; resource hashes and portable CLI pass |
| Static checks | `ruff check --select F src simulator integration network_sim tests` and `git diff --check` pass |

The 551-test run initially used a stdin launcher: 544 passed and 7 multiprocessing
spawn cases could not reopen `<stdin>`. All 7 were rerun with `python -m unittest`
and passed. This was a launcher issue, not an implementation correction.
The full broad Ruff style command still reports repository findings; this cleanup
did not bulk-format unrelated code. The Python F-rule check is clean.

The 9 actual Native scenarios cover: SimGrid direct/satellite/cell communication
scopes; SimGrid KV migration and prefix reuse; ns-3 TP2 KV move and prefix reuse;
SimGrid cross-satellite TP2/PP2 recovery, allocator rollback, heterogeneous Cells/HBM,
and MoE TP2/EP2. The final four reuse the existing Native assertions with explicit
SimGrid/direct-communication settings. These execute real Mid-Layer/vLLM, not mocked
compute or storage. The ns-3 migration test now explicitly selects direct communication,
so it does not accidentally inherit the string-schema ProtoBridge default.

At the user's request, 20 FlowSim-dependent tests were excluded. Neither skips nor
an incompatible upstream FlowSim extension were counted as validation. The external
ISLSim suite was not run. FlowSim production implementation remains unchanged.

## Coverage retained during test migration

`tests/network_case.py` drives actual byte transfers through the shared event engine;
it does not synthesize LLM computation, KV or energy. Directional RF, capacity changes,
byte conservation, route recovery and acquisition timing assertions remain.
Cell migration placement/cleanup tests now exercise Simulation callbacks directly.
Physical dataset tests retain identity/hash rejection, non-sample-time equivalence,
fork isolation, causal random faults and execution without physical recomputation.

Local test environment and logs are under `/data1/guolidong/.orbitserve-tools/`:
`test-env.sh`, `full-retained.log`, `spawn-rerun.log`, `native-final.log`,
`native-ns3-final.log`, `native-simgrid-extra.log`, `review-followup-tests.log`,
`distribution-check.log`. Initial unsuccessful launches remain in the logs; the
successful reruns above provide the final results.

Static review found no blocking production regression. The obsolete README entry
and the weakened dataset execution assertion identified during review were fixed.

## Follow-up removal — 2026-11-13

Base: `5042a5600548dddad4c64768abefe695ca9390f3`, on the same
`version/AsterMind-V1.0` branch and SeeSpace1 environment described above.
The preceding sections record the first cleanup; this follow-up supersedes their
statement that independent architecture/cost and ground-baseline tools remain.

### Scope

- Removed the old architecture/static-cost package, constellation-cost utility,
  ground inference baseline and its dedicated runtime_dispatch entry.
- Removed capacity_study/capacity_native/capacity_metrics: this separate parameter
  sweep used external AsterMindConfig rather than the current SimulationConfig.
- Removed exclusive tests, examples and package declarations for these branches.
- Deleted root docs/ and output/, their obsolete current-document references,
  orphan bytecode for removed modules and resulting empty directories.
- Preserved workload tools and data, ground network resources, native output and
  replay, all core simulation source and the existing directory/package layout.
  There are no changes to simulator/, integration/, network_sim/, the shared Runtime,
  orbit implementation or physical-data preparation. No dependency versions changed.

### Results

| Check | Result |
| --- | --- |
| Batch 1 cost/ground removal smoke | 12 tests pass |
| Batch 2 capacity-sweep removal smoke | 26 tests pass |
| Retained non-FlowSim regression suite | 453 tests pass in 184.623 s; no import errors or skips |
| Real Native integrations | 5 tests pass: ns3 TP2 KV move/prefix reuse; SimGrid direct, satellite and Cell communication; SimGrid KV migration/prefix reuse |
| Real ns3 CLI after each of the 3 batches | Each matches the pre-change baseline: 2 completed requests, 2717 kernels, 6 transfers and passing audit |
| Replay export from actual ns3 and SimGrid results | Both catalogs/reports load; calls and transfers are present; HTML/JS/CSS match packaged source assets |
| sdist -> wheel -> installation outside source | Build/install pass; 88 modules import; resource hashes, portable CLI and web resources pass |
| Static checks and scope review | Ruff F rules, git diff --check and independent read-only review pass; no blocking findings |

The 98 removed test methods exclusively covered retired cost/ground/capacity-study
features, reducing the retained set from 551 to 453. The same 20 FlowSim-dependent
tests remain explicitly excluded at the user's request. The external ISLSim suite
was not run. No skipped or mocked native execution is counted as backend validation.

The three ns3 comparisons check exact request timing/status, decisions, compute
completions, outputs, transfers, batches, input distributions, migrations, event
logs, simulated horizon, final positions, provenance and audit counters.
Host wall-clock measurements and current host resource availability are not expected
to match. Current request, computation and communication semantics are unchanged.

Logs and generated artifacts stay outside the repository under
`/data1/guolidong/.orbitserve-tools/`: `v2-batch1-tests.log`,
`v2-batch2-tests.log`, `v2-full-retained.log`, `v2-native.log`,
`v2-distribution.log`, `v2-replay-and-comparison.log`, `v2-native-reports/`
and `v2-replay/`. The spawn-safe launcher is `v2-retained-suite.py`.

## Builder / Engine layout migration (2026-11-13)

Branch: `version/AsterMind-V1.0`. Frozen baseline:
`58f5cf6818bef90d10a406e3c4766c99a410cfad`.

### Scope

- Move the Python package to root `astermind/`; configuration, orbit, topology,
  scene validation and physical preparation live in `builder/`.
- Merge the current Runtime and Native execution under `engine/`, retaining
  `setup/`, `state/`, and the native dependency adapters under `adapters/`.
- Keep `load_physics` and `validate_scene_config` intact. Their existing Engine
  dependencies and call timing remain unchanged. The Native CLI still loads
  `SimulationConfig` directly; validation is not inserted into the ns3 path.
- Split static model types, execution event contracts and shared scalar contracts
  without changing their class bodies. Remove only the unused canonical topology
  adapter; retain `device_numbers` and native graph compilation.
- Put source dataset adapters in root `workloads/`, Native examples in
  `examples/native/`, and replay export plus static assets in `viz/`.
- Preserve serialized schemas, provenance identities, environment variables and
  computation/communication semantics. Update imports, worker module names,
  dependency/resource anchors, CLI instructions and distribution resources.
- A real browser reproduced an existing scene-switch bug: a selected call index
  from the larger scene was used against the smaller scene before resetting it.
  Reset that selection before rebuilding the options. No UI redesign is included.

### Differential contract

Two original nine-scenario runs established baseline repeatability. A further
baseline run retained temporary physical datasets and memory-policy inputs so
path equivalence could be checked by content, not assumed from filenames.

`tests/scripts/compare_migration.py` compares complete reports and all preserved
inputs. Exceptions are limited to:

1. Explicit host wall timings and memory-availability probes, including the four
   known kernel preparation/prediction wall-time fields. Other measurement,
   cache, network, Cell and audit fields remain exact.
2. Validated host Cell initialization progress. `FIRST_COMPLETED` can observe
   either worker first or several workers together. Capacity, total workers,
   initial/final completion and valid active membership are checked; intermediate
   host poll granularity is not treated as a simulated event.
3. Paths whose saved physical dataset or memory-policy contents have been
   verified equivalent. Physical file hashes are checked before comparison;
   only explicit physical-preparation wall timings and their corresponding
   `core_result.json` manifest hash are canonicalized. Orbit manifests,
   propagation values, numeric shards and physical-state databases remain exact.
4. Independent transfer observations at the same clock tick. No sorting crosses
   time or a global control record, and each individual transfer's order stays
   intact. Request/compute/dependency/decision sequences are not sorted.

The comparison tests deliberately change timing, byte counters, migration state,
provenance, input identities and observation order to verify that real differences
are rejected. They also reject invalid/coalesced startup records that violate
capacity or completion constraints.

### Verification

- Batch 1: 16 configuration, physical contract and scene catalog smoke tests pass.
- Batch 2: 23 module import, orbit identity, frozen corpus and SimGrid timeline
  smoke tests pass after correcting serialized identities and test imports.
- Retained regression: all 459 tests pass in 188.906 s, with no skips (453
  original tests plus 6 comparison checks); final run is recorded in `retained-459.log`. The same 20 FlowSim-dependent tests are explicitly
  excluded at the user's request. External ISLSim tests are outside this run.
- Nine real Native scenarios: ns3 TP2 KV move/prefix reuse; SimGrid direct,
  satellite and Cell communication; KV move/prefix reuse; cross-satellite TP2/PP2
  recovery; failed migration rollback; heterogeneous HBM; MoE TP2/EP2.
  These use the pinned Midlayer, official vLLM Scheduler, ns3 and SimGrid.
  An import guard rejects execution of `_flowsim`; the legacy communication mode
  value `flowsim` denotes direct communication and does not select that backend.
- All 12 reports from those nine scenarios and 8 saved input sets compare without
  simulation differences. All 13 JSON files exported from the normalized reports
  (12 replays and catalog) are byte-identical between old and new exporters.
- Three unchanged real reports additionally produce byte-identical replay JSON.
  Chromium verifies case/deep-link/reload, play/pause/speed, slider/time entry,
  event/key-moment jumps, filters/search, call dependencies, topology device/link
  inspection, batch/KV details, canvas hover/click/wheel and audit/source views.
  The original cross-case selection failure is reproduced on baseline and passes
  after the fix. No page errors or failed HTTP responses occur.
- Six fixed-state screenshots, at 1440x1000 and 768x1000, are byte-identical.
- sdist -> wheel -> installation outside source imports 126 package modules,
  verifies orbital/native/frontend/workload resources, executes real ns3 Native
  simulation and exports its result. The installed report and calculated audit
  match the original CLI baseline under the same comparison contract; the
  installed replay passes the browser interaction checks.
- Static review compares 629 top-level production function/class ASTs: after
  import relocation, differences are limited to five reviewed bootstrap/build/
  diagnostic-path adjustments. Split model bodies and retained topology methods
  match the original ASTs. Ruff F rules and `git diff --check` pass.

Logs, complete reports, preserved inputs, built distributions and browser evidence
are outside the repository at `/data1/guolidong/.orbitserve-v3/`. Nothing is pushed
as part of this iteration.

## Communication prerequisite source port

Target baseline: `ddbdf1540cef3c3db3022b2bcfed839370a1d3e2` on
`version/AsterMind-V1.0`. Source: `Communication_backend` commit
`9c08c9cad81df5a4fee5e606114753df8dab6c2d`. This is a selective source port,
not a merge of the complete communication or computation stacks.

### Source and adaptation boundaries

| Source | Destination | Scope |
| --- | --- | --- |
| `simulator/config.py`, `simulator/scheduler.py` | `astermind/builder/config.py` | `before_request`, explicit receiver/context checks, predecessor lookup and cycle rejection |
| `simulator/engine.py` | `astermind/engine/simulation.py` | Input/parent/KV admission barrier, terminal callbacks, cancellation and failure propagation, request state trace and metrics |
| `simulator/migration.py` terminal callbacks | `astermind/engine/cell_migration.py` | Notify waiting requests after the existing transaction releases its resources |
| Engine/Cell report observations | `simulation.py`, `timeline.py` | Export existing Cell load history and the currently executing event identity |

The upstream prerequisite control flow is retained. Necessary adaptations are
limited to the existing interfaces: `SimulationConfig` replaces the external
scheduler's config access; transfers use the current executor/cancellation API;
times use the authoritative integer tick clock; migrations retain the current
physical rank partitions, batch-boundary pauses, read/write delays and allocator
transaction. `current_event` is an observation only: event keys, times and
priorities are unchanged. Native interrupts still propagate after cleanup.

The current history store names an omitted context by request ID. Predecessor
lookup preserves that convention and rejects duplicate effective contexts.
Parent readiness means computation completion; failed user-output delivery
after computation does not invalidate a committed parent context. A receiver
bound by `before_request` must explicitly name the migration target Cell.
Ordinary requests keep the original automatic Cell selection unchanged.
Unbound legacy migrations remain supported.

Reports add `request_state_events`, input/parent/KV readiness times (nanoseconds
and exact ticks), TTFT/TPOT/E2E metrics and existing per-Cell `load_history`.
Undefined metrics remain null. Existing fields remain present. Communication
allocator/service events and statistics, compute-mode provenance and DP/PD
deployment-specific reports remain tied to their deferred model upgrades;
this port does not emit invented or empty substitutes for them.

No compute adapter, external Midlayer, communication implementation, dynamic
network policy, automatic Cell selector, frontend asset or frontend logic is
changed. DP/WideEP/PD, the global Midlayer/roofline switch and communication
model changes remain deferred. Before the later Midlayer replacement, preserve
the currently used ProtoBridge communication source with a pinned source hash.

### Tests and comparison

- Batch 1: 13 configuration/physical-contract tests pass.
- Lifecycle and cleanup smoke: 24 tests pass, including input/KV ordering,
  parent-only waiting, duplicate notifications, capacity denial, failed parent,
  export/reservation/commit/source-release failure, close, sibling partitions
  and `KeyboardInterrupt` cleanup. Independent review reruns all 24 successfully.
- Retained regression: 476 tests pass in 183.375 s, no skips. The same 20
  FlowSim-dependent tests are excluded by the user's instruction.
- Final focused run: 29 tests pass, adding an incompatible-storage rejection
  test, one hosted/standalone current-event cleanup test and three strict-
  comparison checks to the 24 lifecycle tests. This gives 481 distinct passing
  dependency-light/retained tests overall.
- The nine existing real Native scenarios pass before and after the port,
  using the unchanged pinned Midlayer/vLLM, ns3 and SimGrid dependencies.
  All 12 reports and 8 saved input sets compare with zero legacy differences.
- Four new real Native scenarios pass: bound TP2 move on ns3, bound TP2 copy
  on SimGrid, same-Cell parent waiting, and failed bound migration on SimGrid.
  They check admission against exact input/parent/commit ticks, prefix reuse,
  once-only admission, request failure and allocator rollback. The same-Cell
  case was rerun after correcting a test assertion: an unused Cell may have
  empty load history.
- `tests/scripts/compare_prerequisites.py` reuses the strict comparison rules
  above and removes only named additive observations and null `before_request`.
  It rejects bound-receiver scenes as legacy baselines and detects changed
  compute/transfer/admission times or unknown added fields.
- Existing replay export accepts raw new success/failure reports. The ns3
  legacy replay is identical after the same approved report normalization.
  Browser checks cover playback, case switching, operator search/Cell filters,
  dependency details, migration jumps, reload, failure rollback and retained
  Cell ProtoBridge mode. No browser console errors or warnings were observed.
  Frontend source files are unchanged.
- Ruff F checks and `git diff --check` pass. FlowSim is not executed.

Evidence is outside the repository at
`/data1/guolidong/.orbitserve-kv-iteration/`: `baseline/`, `candidate/`,
`prerequisites/`, `comparison.log`, `retained.log`, `final-focused.log`,
`parent-rerun.log`, `viz-comparison.json` and exported replay directories.
Run the baseline comparison from the repository root with:

```bash
python -m tests.scripts.compare_prerequisites BEFORE_DIR AFTER_DIR
```
