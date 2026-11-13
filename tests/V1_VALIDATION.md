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

## Compute / Comm replacement (2026-11-13)

This section supersedes earlier transport/compute descriptions. Target baseline:
`1c15450cb3f89ad6eec7bfbf23f2a49a9910f939`; imported source:
`Communication_backend` `9c08c9cad81df5a4fee5e606114753df8dab6c2d`.
The current branch remains `version/AsterMind-V1.0`.

Scope: replace old compute and all old communication execution modes with source
Midlayer/Roofline (global choice), source-supported DP/WideEP/PD and RuntimeCommunication.
Retain target Builder/Engine, request assignment/prerequisites, route/cache policy,
physical/orbit/dynamic links and integer attosecond clock. `viz/` has no changes;
new modes output complete raw schema-2 reports only.

Source preservation: 3,624 mapped files, 3,605 byte-identical and 19 boundary-adapted
files, verified against the frozen source. The native algorithms, profiles and
calibrations are untouched. The sole native snapshot edit retains parent PYTHONPATH
in the 0.30 subprocess launcher, fixing host Time unpickling after installation
outside a source checkout. Original vendor license/provenance files remain present.
Source and target hashes plus adaptations are in `engine/compute/SOURCE.json`.

Validation completed:

- 418 module/unit/integration/smoke tests pass, no skips (`retained-418.log`).
  This includes exact-tick corruption/missing-evidence checks, configured request
  prerequisites, resource capacity/protocol accounting, temporal-path failure,
  fault recovery, cancellation and all-shard migration commit barriers.
- Generated/imported workload requests also pass an explicit real native test:
  all four request identities survive generation/CSV import and complete.
- Source-vs-port compute: 12 equal operator-plan hashes and 9,576 exactly equal
  local predictions, across prefill/decode, TP/PP/EP and both compute modes.
- Source-vs-port Comm: 24 identical path/resource/protocol oracle cases pass.
  Completion projection differs by at most 1 ns; operation/phase/flow/byte semantics
  match. This projection tolerance does not permit changing routing or input data.
- Retained route/physical regression: 144 ordered queries under all three route
  policies, including caches, capacity schedules and faults, produce byte-identical
  JSON before/after. Zero-capacity Time+infinity is guarded without changing route
  costs, candidate search, cache or policy selection; a new regression checks that
  such a candidate cannot invalidate a reachable alternative.
- Native DS01–06 and DS08: 24 distinct cases pass, including both intra predictors,
  cross-star TP/PP/EP, independent Cell competition, hierarchical SU collectives,
  real KV copy/move/prefix reuse, PD one-token bypass/full handoff and failed KV
  preventing Decode. DS02 is lower-level integration because the unchanged physical
  Builder does not allow two Cells to own the same satellite.
- Full-platform E2E: both compute modes × normal/repeat/fault/timeout = 8 cases.
  Each normal case has 8 moving satellites, 2 grounds, 4 TP2 Cells and 6 requests,
  10,590 compute calls, 814 communications and 20 request/KV transfers. Repeats
  have exactly equal simulated records. Fault cases verify injection/restoration;
  timeout cases fail explicitly and release communication resources.
- WideEP DS07: all 6 cases pass. PP1 uses 32 GPUs and executes 103,680 kernels,
  1,872 communications and 239,926 dependencies in each compute mode. PP2 uses
  64 GPUs and executes 103,296 kernels, 2,608 communications and 240,076 dependencies.
  Both idle-replica cases fail explicitly before prediction output, matching the
  original source's unbound token-ownership error. Source probes reproduce this
  error in both modes; their pending-step cleanup diagnostics are not counted as
  successful end-to-end execution. In total, 30 matrix cases plus 8 E2E cases pass.
- Distribution: sdist→wheel→outside-source install, 133 imported modules and pinned
  resources verified, both native helpers built, actual 0.31 and 0.30 CLI runs and
  audits pass. Build dependencies are isolated from Scheduler environments.
- Historical frontend: unchanged assets export correctly; browser smoke loads two
  recorded calls, switches details, opens device details and reloads with no JS or
  HTTP errors. Desktop/mobile screenshots retained. This is historical-schema
  compatibility, not a claim that new DP/PD reports render in the old frontend.
- Independent different-model compute and Comm reviews completed. Required issues
  were fixed and reviewed again. Undefined-name/shadowing Ruff checks and Git
  whitespace checks on host/adapted files pass. The staged full-tree check reports
  inherited whitespace in 17 byte-identical upstream files; each staged blob was
  verified against its pinned source hash and retained without formatting changes.
  Untouched source unused-import warnings are also retained.

Environment: Python 3.12; official vLLM 0.31 package identity is verified by the
shipped adapter. vLLM 0.30 source is pinned to
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` and uses the configured CPU worker Python.
PD uses a separate official vLLM 0.11 installation with Transformers 4.57.6,
Tokenizers 0.22.2 and Hugging Face Hub 0.36.2, sharing the tested CPU Torch/runtime
libraries. The original complete 0.11 CPU lock installation could not fetch from
its PyTorch endpoint because of a certificate hostname mismatch; TLS verification
was not disabled. The verified environment is a CPU Scheduler simulation setup,
not a fully resolved GPU/HTTP serving stack.

Preserved upstream boundaries: WideEP idle-replica token ownership may be rejected;
PD/layout/model restrictions remain explicit; star-local whole-operation predictions
have neither global contention nor mid-flight repricing. Cross-star execution is
whole-path, with no store-and-forward. Trace-heavy WideEP fixtures disable detailed
allocation dumps while retaining operator/dependency records and aggregate Comm
statistics; dedicated conservation cases keep full trace evidence. Hardware-model
accuracy is separate from these integration and differential correctness checks.

Evidence on SeeSpace1 is under
`/data1/guolidong/.astermind-compute-comm-20261113/`, including `results/`, frozen
archives, `distribution-complete/` and `browser-evidence/`. Commands and scenario
coverage are maintained in `tests/README.md`. No generated native binary or raw
result artifact is committed; this iteration is committed locally without a push.

## Unused-code and audit-boundary cleanup (2026-11-13)

Baseline: `0f8975a6260dd9c6f93873823d172f5123482df1`, on
`version/AsterMind-V1.0`. This section supersedes the earlier layout-migration
decision to retain `validate_scene_config`; that statement records historical scope.

Scope is limited to deleting unconsumed `engine/contracts.py`, `engine/timing.py`
and `builder/validation.py`; removing unused Cell/Comm fields, initialization and
imports, including the always-None local capacity branch; and moving
`migration_barrier_valid` unchanged into the mainline `engine/audit.py`.
`clock_audit.py` re-exports that helper for existing callers. Its historical kernel
and clock checks are unchanged. No validation rules were transferred into
`config.py` or `request_batch.py`.

Physical Builder organization, independent network/scheduling APIs, prediction
cache, route/link policy, model algorithms, native allocator and `viz/` are
unchanged. Imported source remains `9c08c9cad81df5a4fee5e606114753df8dab6c2d`:
3,624 file mappings, 3,604 byte-identical and 20 adapted. Only five target hashes
and the cleanup explanation changed; all source hashes and identity are retained.

Validation:

- Each batch passed its relevant smoke tests: 37 communication/Cell/boundary
  tests, 20 audit/clock/replay tests and 22 request/boundary tests, plus CLI
  scenario validation.
- All 423 retained tests pass with no skips: the original 418 remain unchanged,
  with five new audit compatibility tests. They cover exact ticks, legacy ceil-ns
  projection, early commit, late/negative delivery, absent/failed partitions and
  malformed evidence through both import paths.
- A fresh process forbids importing `clock_audit` while auditing a completed KV
  migration. Both real normal-E2E reports pass under the new mainline audit;
  the same guard fails on the baseline's former lazy import, proving sensitivity.
- Before/after comparison: 22 static communication estimates, 24 dynamic Comm
  oracle cases, 12 compute plans and 9,576 local predictions are exactly equal.
- All eight native reports pass audit and have exactly equal simulated records:
  DS01 in both compute modes and both intra modes (four), DS03 TP2/PP2 in both
  compute modes (two), and normal full-platform E2E in both modes (two). Each E2E
  includes 8 moving satellites, 2 grounds, 4 TP2 Cells, 6 requests and KV migration.
  Only explicit host observations are excluded: phase/startup/event-loop/Comm
  wall timings, worker PIDs, available host memory and startup memory slots.
  Source/output path prefixes are rebased; simulated clocks, events, plans,
  allocations, bytes, failures and semantic measurements are compared exactly.
- Three real historical KV reports have byte-identical audit output; the
  supported historical replay fixture, catalog and frontend assets also export
  byte-identically. Older raw reports containing removed backend/timeline config
  fields are rejected identically by baseline and candidate. This pre-existing
  limitation is recorded, not repaired by changing the frontend or configuration.
- sdist -> wheel -> installation outside source passes: 130 modules imported,
  pinned resources checked, native helpers built, actual vLLM 0.31 and 0.30 CLI
  simulations and audits pass. The three deleted modules are absent from both
  distributions and the installation.
- Different-model review approves the change with no Critical/Required findings;
  Brooks review finds no new decay risk in this scope. Source/scope checks verify
  the unchanged migration helper and historical checks, untouched protected
  modules, and the five updated imported target hashes. Git diff checks pass.

Evidence is under `/data1/guolidong/.astermind-cleanup-20261113/`. The baseline
reports are reused from `.astermind-boundary-audit-j1di17c7/results/baseline/`;
all 3,957 tracked files of that frozen source were rechecked against the baseline
Git blobs. Candidate tests and reports are fresh executions using the same Python
3.12 / pinned vLLM 0.31 and 0.30 setup. No dependency versions were changed.
